from decision_log import log_decision
import json
import threading
from datetime import datetime
from functools import partial
from ha_client import get_state, call_service
from ollama_client import ask_ollama, OLLAMA_COLOR_MODEL

# --- Zone Configuration ---
LIGHT_ENTITIES = {
    "living_room": "light.wiz_rgbww_tunable_a480ec",
    "lab": "light.wiz_rgbww_tunable_225a0a",
}
DEFAULT_ZONE = "living_room"

# --- State (persists while Flask is running) ---
last_turn_on_time = {}  # Now per-zone: {"living_room": datetime, "lab": datetime}
pending_off_timer = {}  # Now per-zone: {"living_room": Timer, "lab": Timer}
COOLDOWN_SECONDS = 120
OFF_DELAY_SECONDS = 120

def get_brightness(hour):
    """Deterministic brightness lookup — no LLM needed."""
    if 6 <= hour < 9:
        return 180
    elif 9 <= hour < 17:
        return 255
    elif 17 <= hour < 21:
        return 150
    else:
        return 80

def get_color_temp(hour, weather, zone):
    """Ask Jarvis only for color temperature reasoning."""
    prompt = f"""You control a {zone} light's color temperature.
Current hour (24h): {hour}
Current weather: {weather}

Color temperature guide (Kelvin, range 2000-6500):
- Clear/sunny daytime: 4000-5000 Kelvin (cool white)
- Overcast/rainy: 2700-3200 Kelvin (warm white)
- Evening: 2500-3000 Kelvin (warm)
- Night/clear-night: 2200-2700 Kelvin (very warm)

Time of day takes priority over weather when they conflict.

Respond with ONLY a JSON object, no other text:
{{"color_temp_kelvin": 2000-6500, "reason": "one sentence explanation"}}"""

    response = ask_ollama(prompt, model=OLLAMA_COLOR_MODEL)
    try:
        result = json.loads(response)
        return (
            result.get("color_temp_kelvin", 3000),
            result.get("reason", "")
        )
    except json.JSONDecodeError:
        return 3000, "LLM response failed, using default"

def delayed_turn_off(zone, light_entity):
    """Runs after OFF_DELAY_SECONDS if not cancelled."""
    global pending_off_timer
    pending_off_timer[zone] = None

    light_state = get_state(light_entity)
    weather = get_state("weather.forecast_home")

    if light_state == "on":
        call_service("light", "turn_off", light_entity)
        log_decision(
            event="occupancy_cleared",
            reasoning=f"Occupancy cleared for {OFF_DELAY_SECONDS}s — turning off",
            action="turn_off",
            details={
                "zone": zone,
                "light_entity": light_entity,
                "brightness": None,
                "color_temp_kelvin": None,
                "weather": weather
            }
        )
        print(f"[{zone}] Delayed turn_off executed after {OFF_DELAY_SECONDS}s")
    else:
        print(f"[{zone}] Delayed turn_off fired but light already off")

def run_jarvis(event_data=None):
    global last_turn_on_time, pending_off_timer

    event = event_data.get("event", "unknown") if event_data else "unknown"
    zone = event_data.get("zone", DEFAULT_ZONE) if event_data else DEFAULT_ZONE
    light_entity = LIGHT_ENTITIES.get(zone, LIGHT_ENTITIES[DEFAULT_ZONE])

    if event == "person_detected":
        # Cancel any pending turn-off for this zone
        if pending_off_timer.get(zone):
            pending_off_timer[zone].cancel()
            pending_off_timer[zone] = None
            print(f"[{zone}] Pending turn-off cancelled — person detected")

        # Cooldown check for this zone
        zone_last_turn_on = last_turn_on_time.get(zone)
        if zone_last_turn_on:
            elapsed = (datetime.now() - zone_last_turn_on).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                print(f"[{zone}] Cooldown active, skipping ({elapsed:.0f}s elapsed)")
                log_decision(
                    event=event,
                    reasoning="Cooldown active, skipping",
                    action="skip",
                    details={
                        "zone": zone,
                        "cooldown_seconds": COOLDOWN_SECONDS
                    }
                )
                return

        light_state = get_state(light_entity)
        weather = get_state("weather.forecast_home")
        hour = datetime.now().hour
        light_is_on = light_state == "on"

        if not light_is_on:
            action = "turn_on"
            brightness = get_brightness(hour)
            color_temp, reason = get_color_temp(hour, weather, zone)

            call_service("light", "turn_on",
                         light_entity,
                         brightness=brightness,
                         color_temp_kelvin=color_temp)
            last_turn_on_time[zone] = datetime.now()
        else:
            action = "do_nothing"
            brightness = None
            color_temp = None
            reason = "Person detected, light already on"

        log_decision(
            event=event,
            reasoning=reason,
            action=action,
            details={
                "zone": zone,
                "light_entity": light_entity,
                "brightness": brightness,
                "color_temp_kelvin": color_temp,
                "weather": weather
            }
        )
        print(f"[{zone}] Action: {action} | Brightness: {brightness} | Color temp: {color_temp}")

    elif event == "occupancy_cleared":
        # Cancel any existing timer for this zone
        if pending_off_timer.get(zone):
            pending_off_timer[zone].cancel()

        # Schedule turn-off for this zone
        pending_off_timer[zone] = threading.Timer(
            OFF_DELAY_SECONDS,
            partial(delayed_turn_off, zone, light_entity)
        )
        pending_off_timer[zone].start()

        print(f"[{zone}] Occupancy cleared — turn-off scheduled in {OFF_DELAY_SECONDS}s")
        log_decision(
            event=event,
            reasoning=f"Occupancy cleared — turn-off scheduled in {OFF_DELAY_SECONDS}s",
            action="scheduled_off",
            details={
                "zone": zone,
                "light_entity": light_entity,
                "delay_seconds": OFF_DELAY_SECONDS
            }
        )

    else:
        print(f"[{zone}] Unhandled event: {event}")
