"""
eval_models.py — Phase 9 Model Evaluation Harness

Sends standardized test prompts to two Ollama models and compares:
  - JSON compliance rate (valid parse + correct keys)
  - Response quality (value ranges, reasoning presence)
  - Inference latency

Usage:
    python3 eval_models.py --model-a llama3.1:8b-instruct-q5_K_M --model-b qwen3:8b \
                           --ollama-url http://localhost:11434 \
                           --runs 3

Output: eval_report.json + printed summary
"""

import argparse
import json
import time
import requests
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Color temp prompt (from jarvis.py get_color_temp)
def build_color_temp_prompt(hour, weather):
    return f"""You control a living room light's color temperature.
Current hour (24h): {hour}
Current weather: {weather}

Color temperature guide (Kelvin, range 2000-6500):
- Clear/sunny daytime: 4000-5000 Kelvin (cool white)
- Overcast/rainy: 2700-3200 Kelvin (warm white)
- Evening: 2500-3000 Kelvin (warm)
- Night/clear-night: 2200-2700 Kelvin (very warm)

Respond with ONLY a JSON object, no other text:
{{"color_temp_kelvin": 2000-6500, "reason": "one sentence explanation"}}"""


# Bot prompt (from jarvis_bot.py ask_jarvis)
def build_bot_prompt(user_message, hour, light_states, decision_log=""):
    time_str = f"{hour}:00"
    return f"""You are Jarvis, an AI smart home assistant running fully locally on a private homelab.
You have direct control over the smart home via Home Assistant.

CURRENT STATE:
- Time: {time_str} (hour {hour})
- Lights:
{light_states}

{decision_log}

AVAILABLE ACTIONS:
You can control these devices:
- "living room light" — turn_on (with brightness 0-255) or turn_off
- "lab light" — turn_on (with brightness 0-255) or turn_off

Brightness guide for context:
- Morning (6-9): 180
- Daytime (9-17): 255
- Evening (17-21): 150
- Night (21-6): 80

RESPONSE FORMAT:
Always respond with a SINGLE JSON object. Nothing else — no markdown, no backticks, no multiple objects, just ONE JSON object.

If the user is asking you to DO something (control a light, change brightness, etc.):
{{"response": "your conversational response", "actions": [{{"action": "turn_on" or "turn_off", "entity": "living room light" or "lab light", "brightness": 0-255}}]}}

If the user is just chatting or asking a question (why did you do X, what's going on, etc.):
{{"response": "your conversational response"}}

CRITICAL: Put "response" and "actions" in the SAME JSON object. Never split them into separate objects.

CONVERSATION RULES:
- Be conversational and natural, not robotic
- When asked WHY you did something, look at your decision log and explain your actual reasoning in your own words — don't just list log entries
- When asked to control something, do it and confirm naturally
- If you're not sure what the user wants, ask for clarification
- Keep responses concise but thoughtful

User: {user_message}
Jarvis:"""


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

COLOR_TEMP_TESTS = [
    {"hour": 7,  "weather": "sunny",       "label": "morning_sunny"},
    {"hour": 10, "weather": "cloudy",       "label": "midday_cloudy"},
    {"hour": 14, "weather": "rainy",        "label": "afternoon_rainy"},
    {"hour": 19, "weather": "clear",        "label": "evening_clear"},
    {"hour": 23, "weather": "clear-night",  "label": "night_clear"},
    {"hour": 3,  "weather": "partly_cloudy","label": "late_night_overcast"},
    {"hour": 12, "weather": "fog",          "label": "noon_fog"},
    {"hour": 20, "weather": "snowy",        "label": "evening_snow"},
]

LIGHT_STATES_ON = "- living room light: on (brightness: 70%)\n- lab light: off"
LIGHT_STATES_OFF = "- living room light: off\n- lab light: off"

BOT_TESTS = [
    # Command tests — should produce actions
    {
        "message": "turn on the living room light",
        "hour": 14, "light_states": LIGHT_STATES_OFF,
        "label": "cmd_turn_on", "expect_action": True,
    },
    {
        "message": "kill the living room light",
        "hour": 23, "light_states": LIGHT_STATES_ON,
        "label": "cmd_turn_off_slang", "expect_action": True,
    },
    {
        "message": "set the lab light to 50%",
        "hour": 20, "light_states": LIGHT_STATES_OFF,
        "label": "cmd_brightness_pct", "expect_action": True,
    },
    {
        "message": "turn on both lights",
        "hour": 18, "light_states": LIGHT_STATES_OFF,
        "label": "cmd_multi_entity", "expect_action": True,
    },
    {
        "message": "dim the living room light a bit",
        "hour": 21, "light_states": LIGHT_STATES_ON,
        "label": "cmd_vague_dim", "expect_action": True,
    },
    # Chat tests — should NOT produce actions
    {
        "message": "what's up?",
        "hour": 10, "light_states": LIGHT_STATES_ON,
        "label": "chat_greeting", "expect_action": False,
    },
    {
        "message": "why did you turn on the light?",
        "hour": 20, "light_states": LIGHT_STATES_ON,
        "label": "chat_why_question", "expect_action": False,
        "decision_log": (
            "My recent decision log (newest last):\n"
            "  [2026-04-08 19:45:00] event=person_detected action=turn_on "
            'reason="Evening with clear skies, warm 2800K"\n'
        ),
    },
    {
        "message": "how's the weather looking?",
        "hour": 15, "light_states": LIGHT_STATES_ON,
        "label": "chat_off_topic", "expect_action": False,
    },
]


# ---------------------------------------------------------------------------
# Inference + validation
# ---------------------------------------------------------------------------

def query_ollama(ollama_url, model, prompt, think=False):
    """Send a prompt to Ollama, return (raw_response, latency_ms)."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "think": think},
            timeout=120,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        raw = resp.json().get("response", "").strip()
        return raw, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return f"ERROR: {e}", latency_ms


def clean_json_response(raw):
    """Strip markdown fences and whitespace, attempt JSON parse."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    return cleaned


def validate_color_temp(raw):
    """Validate a color temp response. Returns (parsed_dict, issues_list)."""
    issues = []
    cleaned = clean_json_response(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, ["json_parse_failed"]

    if not isinstance(parsed, dict):
        return None, ["not_a_dict"]

    if "color_temp_kelvin" not in parsed:
        issues.append("missing_color_temp_kelvin")
    else:
        val = parsed["color_temp_kelvin"]
        if not isinstance(val, (int, float)):
            issues.append("color_temp_not_numeric")
        elif val < 2000 or val > 6500:
            issues.append(f"color_temp_out_of_range({val})")

    if "reason" not in parsed:
        issues.append("missing_reason")
    elif not isinstance(parsed.get("reason"), str) or len(parsed["reason"]) < 5:
        issues.append("reason_too_short")

    extra_keys = set(parsed.keys()) - {"color_temp_kelvin", "reason"}
    if extra_keys:
        issues.append(f"extra_keys({extra_keys})")

    return parsed, issues


def validate_bot_response(raw, expect_action):
    """Validate a bot response. Returns (parsed_dict, issues_list)."""
    issues = []
    cleaned = clean_json_response(raw)

    # Try direct parse first
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try the split-object recovery from jarvis_bot.py
        try:
            merged = {}
            for chunk in cleaned.replace("}\n{", "}|||{").replace("} {", "}|||{").split("|||"):
                part = json.loads(chunk.strip())
                merged.update(part)
            if merged:
                parsed = merged
                issues.append("required_split_recovery")
        except (json.JSONDecodeError, Exception):
            pass

    if parsed is None:
        return None, ["json_parse_failed"]

    if not isinstance(parsed, dict):
        return None, ["not_a_dict"]

    if "response" not in parsed:
        issues.append("missing_response")
    elif not isinstance(parsed.get("response"), str) or len(parsed["response"]) < 2:
        issues.append("response_empty_or_invalid")

    actions = parsed.get("actions", [])
    has_actions = isinstance(actions, list) and len(actions) > 0

    if expect_action and not has_actions:
        issues.append("expected_action_but_none")
    elif not expect_action and has_actions:
        issues.append("unexpected_action_produced")

    # Validate action structure if present
    for i, act in enumerate(actions):
        if not isinstance(act, dict):
            issues.append(f"action_{i}_not_dict")
            continue
        if act.get("action") not in ("turn_on", "turn_off"):
            issues.append(f"action_{i}_invalid_type({act.get('action')})")
        if act.get("entity") not in ("living room light", "lab light"):
            issues.append(f"action_{i}_unknown_entity({act.get('entity')})")

    return parsed, issues


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_eval(ollama_url, model_a, model_b, runs):
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ollama_url": ollama_url,
            "model_a": model_a,
            "model_b": model_b,
            "runs_per_test": runs,
        },
        "color_temp": [],
        "bot": [],
    }

    total_tests = (len(COLOR_TEMP_TESTS) + len(BOT_TESTS)) * 2 * runs
    current = 0

    # --- Color temp tests ---
    for test in COLOR_TEMP_TESTS:
        prompt = build_color_temp_prompt(test["hour"], test["weather"])
        for model_label, model_name in [("a", model_a), ("b", model_b)]:
            for run_i in range(runs):
                current += 1
                tag = f"[{current}/{total_tests}]"
                print(f"  {tag} color_temp | {test['label']} | {model_name} | run {run_i+1}")

                raw, latency = query_ollama(ollama_url, model_name, prompt)
                parsed, issues = validate_color_temp(raw)

                results["color_temp"].append({
                    "test": test["label"],
                    "model": model_label,
                    "model_name": model_name,
                    "run": run_i + 1,
                    "latency_ms": round(latency, 1),
                    "json_valid": parsed is not None,
                    "issues": issues,
                    "parsed": parsed,
                    "raw": raw[:500],
                })

    # --- Bot tests ---
    for test in BOT_TESTS:
        prompt = build_bot_prompt(
            test["message"], test["hour"], test["light_states"],
            test.get("decision_log", ""),
        )
        for model_label, model_name in [("a", model_a), ("b", model_b)]:
            for run_i in range(runs):
                current += 1
                tag = f"[{current}/{total_tests}]"
                print(f"  {tag} bot | {test['label']} | {model_name} | run {run_i+1}")

                raw, latency = query_ollama(ollama_url, model_name, prompt)
                parsed, issues = validate_bot_response(raw, test["expect_action"])

                results["bot"].append({
                    "test": test["label"],
                    "model": model_label,
                    "model_name": model_name,
                    "run": run_i + 1,
                    "latency_ms": round(latency, 1),
                    "json_valid": parsed is not None,
                    "issues": issues,
                    "parsed": parsed,
                    "raw": raw[:500],
                })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results):
    model_a = results["config"]["model_a"]
    model_b = results["config"]["model_b"]
    runs = results["config"]["runs_per_test"]

    print("\n" + "=" * 70)
    print("PHASE 9 — MODEL EVALUATION REPORT")
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b}")
    print(f"  Runs per test: {runs}")
    print("=" * 70)

    for category in ["color_temp", "bot"]:
        print(f"\n{'─' * 70}")
        print(f"  {category.upper()} TESTS")
        print(f"{'─' * 70}")

        for model_label, model_name in [("a", model_a), ("b", model_b)]:
            entries = [e for e in results[category] if e["model"] == model_label]
            total = len(entries)
            json_ok = sum(1 for e in entries if e["json_valid"])
            no_issues = sum(1 for e in entries if len(e["issues"]) == 0)
            latencies = [e["latency_ms"] for e in entries]
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            min_lat = min(latencies) if latencies else 0
            max_lat = max(latencies) if latencies else 0

            print(f"\n  [{model_name}]")
            print(f"    JSON parse rate:  {json_ok}/{total} ({json_ok/total*100:.0f}%)")
            print(f"    Perfect (0 issues): {no_issues}/{total} ({no_issues/total*100:.0f}%)")
            print(f"    Latency avg/min/max: {avg_lat:.0f} / {min_lat:.0f} / {max_lat:.0f} ms")

            # Issue breakdown
            all_issues = []
            for e in entries:
                all_issues.extend(e["issues"])
            if all_issues:
                from collections import Counter
                issue_counts = Counter(all_issues)
                print(f"    Issues:")
                for issue, count in issue_counts.most_common():
                    print(f"      {issue}: {count}")

        # Per-test breakdown
        test_labels = list(dict.fromkeys(e["test"] for e in results[category]))
        print(f"\n  Per-test breakdown:")
        print(f"  {'Test':<25} {'Model A':>12} {'Model B':>12}")
        print(f"  {'─'*25} {'─'*12} {'─'*12}")
        for label in test_labels:
            a_entries = [e for e in results[category] if e["test"] == label and e["model"] == "a"]
            b_entries = [e for e in results[category] if e["test"] == label and e["model"] == "b"]
            a_ok = sum(1 for e in a_entries if len(e["issues"]) == 0)
            b_ok = sum(1 for e in b_entries if len(e["issues"]) == 0)
            print(f"  {label:<25} {a_ok}/{len(a_entries):>9} {b_ok}/{len(b_entries):>9}")

    # Overall winner suggestion
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for model_label, model_name in [("a", model_a), ("b", model_b)]:
        all_entries = [e for e in results["color_temp"] + results["bot"] if e["model"] == model_label]
        total = len(all_entries)
        perfect = sum(1 for e in all_entries if len(e["issues"]) == 0)
        json_ok = sum(1 for e in all_entries if e["json_valid"])
        avg_lat = sum(e["latency_ms"] for e in all_entries) / total if total else 0
        print(f"\n  {model_name}:")
        print(f"    Overall perfect: {perfect}/{total} ({perfect/total*100:.0f}%)")
        print(f"    JSON parse rate: {json_ok}/{total} ({json_ok/total*100:.0f}%)")
        print(f"    Avg latency:     {avg_lat:.0f} ms")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 9 — Jarvis Model Evaluation")
    parser.add_argument("--model-a", required=True, help="First model (e.g. llama3.1:8b-instruct-q5_K_M)")
    parser.add_argument("--model-b", required=True, help="Second model to compare")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test (default 3)")
    parser.add_argument("--output", default="eval_report.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"\nPhase 9 — Model Evaluation")
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Ollama:  {args.ollama_url}")
    print(f"  Runs:    {args.runs}")
    print()

    # Quick connectivity check
    try:
        resp = requests.get(f"{args.ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        print(f"  Available models: {', '.join(available)}")
        for model in [args.model_a, args.model_b]:
            if model not in available:
                print(f"  WARNING: {model} not found in Ollama. Pull it first.")
                sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Can't reach Ollama at {args.ollama_url}: {e}")
        sys.exit(1)

    print(f"\nRunning {(len(COLOR_TEMP_TESTS) + len(BOT_TESTS)) * 2 * args.runs} total inferences...\n")

    results = run_eval(args.ollama_url, args.model_a, args.model_b, args.runs)

    # Save full results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output}")

    print_report(results)


if __name__ == "__main__":
    main()
