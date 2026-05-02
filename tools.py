"""
tools.py — Jarvis Tool Registry

Defines the tools the LLM can call during conversation.
Each tool is a plain Python function + a schema the LLM reads.
The registry connects tool names to functions so the agentic loop
can dispatch calls.
"""

import os
import json
from datetime import datetime
from ha_client import get_state, call_service
from decision_log import get_recent_decisions

# ---------------------------------------------------------------------------
# Entity mapping — same one from jarvis_bot.py
# The LLM speaks friendly names, tools translate to entity IDs
# ---------------------------------------------------------------------------
KNOWN_ENTITIES = {
    "living room light": "light.wiz_rgbww_tunable_a480ec",
    "lab light": "light.wiz_rgbww_tunable_225a0a",
}

HA_URL = os.environ.get("HA_URL", "http://localhost:8123")
HA_TOKEN = os.environ["HA_TOKEN"]
HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

DECISIONS_FILE = os.environ.get(
    "DECISIONS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "decisions.jsonl"),
)


# ---------------------------------------------------------------------------
# Tool functions — these are what actually execute when the LLM calls a tool
# ---------------------------------------------------------------------------

def tool_get_light_state(args):
    """Get the current state of a light."""
    entity_name = args.get("entity", "").lower()
    entity_id = KNOWN_ENTITIES.get(entity_name)

    if not entity_id:
        return f"Unknown light '{entity_name}'. Available: {', '.join(KNOWN_ENTITIES.keys())}"

    try:
        import requests
        resp = requests.get(
            f"{HA_URL}/api/states/{entity_id}",
            headers=HA_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        state = data.get("state", "unknown")
        attrs = data.get("attributes", {})
        brightness_raw = attrs.get("brightness")
        color_temp = attrs.get("color_temp_kelvin")

        result = f"{entity_name}: {state}"
        if brightness_raw is not None:
            result += f", brightness {round((brightness_raw / 255) * 100)}%"
        if color_temp is not None:
            result += f", color temp {color_temp}K"
        return result
    except Exception as e:
        return f"Error checking {entity_name}: {e}"


def tool_set_light(args):
    """Turn a light on or off, optionally with brightness."""
    entity_name = args.get("entity", "").lower()
    action = args.get("action", "").lower()
    brightness = args.get("brightness")

    entity_id = KNOWN_ENTITIES.get(entity_name)
    if not entity_id:
        return f"Unknown light '{entity_name}'. Available: {', '.join(KNOWN_ENTITIES.keys())}"

    if action not in ("turn_on", "turn_off"):
        return f"Unknown action '{action}'. Use 'turn_on' or 'turn_off'."

    try:
        status = call_service(
            domain="light",
            service=action,
            entity_id=entity_id,
            brightness=int(brightness) if brightness is not None else None,
        )
        bri_str = f" at brightness {brightness}" if brightness else ""
        return f"{action.replace('_', ' ')} {entity_name}{bri_str} (status: {status})"
    except Exception as e:
        return f"Failed to {action} {entity_name}: {e}"


def tool_get_weather(args):
    """Get current weather from HA."""
    try:
        import requests
        resp = requests.get(
            f"{HA_URL}/api/states/weather.forecast_home",
            headers=HA_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        state = data.get("state", "unknown")
        attrs = data.get("attributes", {})
        temp = attrs.get("temperature")
        humidity = attrs.get("humidity")
        cloud_coverage = attrs.get("cloud_coverage")

        result = f"Weather: {state}"
        if temp is not None:
            result += f", {temp}°F"
        if humidity is not None:
            result += f", humidity {humidity}%"
        if cloud_coverage is not None:
            result += f", cloud cover {cloud_coverage}%"
        return result
    except Exception as e:
        return f"Error getting weather: {e}"


def tool_get_time(args):
    """Get current time with context."""
    now = datetime.now()
    hour = now.hour
    time_str = now.strftime("%I:%M %p")

    # Same brightness guide from the original bot
    if 6 <= hour < 9:
        period = "morning"
        suggested_brightness = 180
    elif 9 <= hour < 17:
        period = "daytime"
        suggested_brightness = 255
    elif 17 <= hour < 21:
        period = "evening"
        suggested_brightness = 150
    else:
        period = "night"
        suggested_brightness = 80

    return f"Time: {time_str}, period: {period}, suggested brightness: {suggested_brightness}/255"

def tool_get_decision_log(args):
    """Read recent Jarvis decisions."""
    count = args.get("count", 5)
    decisions = get_recent_decisions(count)

    if not decisions:
        return "No decisions logged yet."

    result = ""
    for d in decisions:
        result += f"[{d.get('timestamp')}] {d.get('event')} → {d.get('action')} (reason: {d.get('reasoning', 'none')})\n"
    return result.strip()


# ---------------------------------------------------------------------------
# Tool definitions — the "menu" the LLM reads
# Each tool has a name, description, and parameter schema
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "get_light_state",
        "description": "Get the current state of a light (on/off, brightness, color temp). Use this before changing a light if you need to know its current state.",
        "parameters": {
            "entity": "The light to check. One of: 'living room light', 'lab light'"
        }
    },
    {
        "name": "set_light",
        "description": "Turn a light on or off, optionally setting brightness (0-255). Use this to execute light commands.",
        "parameters": {
            "entity": "The light to control. One of: 'living room light', 'lab light'",
            "action": "'turn_on' or 'turn_off'",
            "brightness": "(optional) Brightness level 0-255. Only used with turn_on."
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather conditions including temperature, humidity, and cloud coverage. Useful for deciding color temperature.",
        "parameters": {}
    },
    {
        "name": "get_time",
        "description": "Get the current time, time of day period (morning/daytime/evening/night), and suggested brightness level.",
        "parameters": {}
    },
    {
        "name": "get_decision_log",
        "description": "Read recent Jarvis automated decisions. Use this when the user asks why Jarvis did something.",
        "parameters": {
            "count": "(optional) Number of recent decisions to return. Default 5."
        }
    },
]


# ---------------------------------------------------------------------------
# Registry — connects tool names to functions
# When the LLM says "call get_light_state", this is how Python finds the function
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "get_light_state": tool_get_light_state,
    "set_light": tool_set_light,
    "get_weather": tool_get_weather,
    "get_time": tool_get_time,
    "get_decision_log": tool_get_decision_log,
}


def execute_tool(name, args):
    """
    Look up a tool by name and run it.
    Returns the tool's result as a string, or an error message.
    """
    func = TOOL_REGISTRY.get(name)
    if not func:
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY.keys())}"

    try:
        return func(args)
    except Exception as e:
        return f"Tool '{name}' failed: {e}"
