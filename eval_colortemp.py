"""
eval_color_temp.py — Color Temp Model Comparison

Focused eval for the color temperature task only.
Tests multiple models to find the smallest one that still
produces reliable JSON with correct values.

Usage:
    python3 eval_color_temp.py \
        --models llama3.2:3b qwen3.5:2b \
        --runs 10

Expected ranges are generous — they reject obviously wrong answers
(5000K at midnight) but don't split hairs between reasonable values.
"""

import argparse
import json
import time
import requests
import sys
from datetime import datetime
from collections import Counter


# ---------------------------------------------------------------------------
# Color temp prompt — identical to jarvis.py get_color_temp()
# ---------------------------------------------------------------------------

def build_prompt(hour, weather):
    return f"""You control a living room light's color temperature.
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


# ---------------------------------------------------------------------------
# Test scenarios
# Ranges are generous — reject clearly wrong, accept reasonable variation
# ---------------------------------------------------------------------------

TESTS = [
    # Daytime scenarios — should be warm-to-cool white depending on weather
    {"hour": 7,  "weather": "sunny",         "label": "morning_sunny",         "expected_range": (3500, 5500)},
    {"hour": 10, "weather": "cloudy",         "label": "midday_cloudy",         "expected_range": (2700, 4000)},
    {"hour": 14, "weather": "rainy",          "label": "afternoon_rainy",       "expected_range": (2700, 3500)},
    {"hour": 12, "weather": "fog",            "label": "noon_fog",              "expected_range": (2700, 4000)},
    {"hour": 12, "weather": "sunny",          "label": "noon_sunny",            "expected_range": (4000, 5500)},
    {"hour": 6,  "weather": "sunny",          "label": "sunrise_boundary",      "expected_range": (3500, 5500)},

    # Evening scenarios — should be warm
    {"hour": 19, "weather": "clear",          "label": "evening_clear",         "expected_range": (2200, 3500)},
    {"hour": 20, "weather": "snowy",          "label": "evening_snow",          "expected_range": (2200, 3200)},
    {"hour": 17, "weather": "clear",          "label": "evening_boundary",      "expected_range": (2500, 3500)},

    # Night scenarios — should be very warm
    {"hour": 23, "weather": "clear-night",    "label": "night_clear",           "expected_range": (2000, 2800)},
    {"hour": 3,  "weather": "partly_cloudy",  "label": "late_night_overcast",   "expected_range": (2000, 2800)},
    {"hour": 0,  "weather": "clear-night",    "label": "midnight",              "expected_range": (2000, 2800)},
]


# ---------------------------------------------------------------------------
# Inference + validation
# ---------------------------------------------------------------------------

def query_ollama(ollama_url, model, prompt):
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "think": False},
            timeout=120,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        raw = resp.json().get("response", "").strip()
        return raw, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return f"ERROR: {e}", latency_ms


def validate(raw, expected_range):
    """Validate color temp response. Returns (parsed, issues)."""
    issues = []

    # Clean markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Parse JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, ["json_parse_failed"]

    if not isinstance(parsed, dict):
        return None, ["not_a_dict"]

    # Check color_temp_kelvin
    if "color_temp_kelvin" not in parsed:
        issues.append("missing_key")
    else:
        val = parsed["color_temp_kelvin"]
        if not isinstance(val, (int, float)):
            issues.append("not_numeric")
        elif val < 2000 or val > 6500:
            issues.append(f"out_of_range({val})")
        elif expected_range:
            low, high = expected_range
            if val < low or val > high:
                issues.append(f"wrong_value({val}_expected_{low}-{high})")

    # Check reason
    if "reason" not in parsed:
        issues.append("missing_reason")
    elif not isinstance(parsed.get("reason"), str) or len(parsed["reason"]) < 5:
        issues.append("bad_reason")

    # Extra keys
    extra = set(parsed.keys()) - {"color_temp_kelvin", "reason"}
    if extra:
        issues.append(f"extra_keys({extra})")

    return parsed, issues


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_eval(ollama_url, models, runs):
    results = {}
    total = len(TESTS) * len(models) * runs
    current = 0

    for model in models:
        results[model] = []
        for test in TESTS:
            prompt = build_prompt(test["hour"], test["weather"])
            for run_i in range(runs):
                current += 1
                print(f"  [{current}/{total}] {test['label']} | {model} | run {run_i+1}")

                raw, latency = query_ollama(ollama_url, model, prompt)
                parsed, issues = validate(raw, test.get("expected_range"))

                results[model].append({
                    "test": test["label"],
                    "run": run_i + 1,
                    "latency_ms": round(latency, 1),
                    "json_valid": parsed is not None,
                    "issues": issues,
                    "value": parsed.get("color_temp_kelvin") if parsed else None,
                    "raw": raw[:300],
                })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results, runs):
    models = list(results.keys())

    print("\n" + "=" * 70)
    print("COLOR TEMP MODEL COMPARISON")
    print(f"  Models: {', '.join(models)}")
    print(f"  Tests: {len(TESTS)} scenarios × {runs} runs = {len(TESTS) * runs} per model")
    print("=" * 70)

    # Per-model summary
    summaries = {}
    for model in models:
        entries = results[model]
        total = len(entries)
        json_ok = sum(1 for e in entries if e["json_valid"])
        perfect = sum(1 for e in entries if len(e["issues"]) == 0)
        latencies = [e["latency_ms"] for e in entries]
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        summaries[model] = {
            "total": total, "json_ok": json_ok, "perfect": perfect,
            "avg_lat": avg_lat, "min_lat": min_lat, "max_lat": max_lat,
        }

        print(f"\n{'─' * 70}")
        print(f"  {model}")
        print(f"{'─' * 70}")
        print(f"    JSON parse rate:    {json_ok}/{total} ({json_ok/total*100:.0f}%)")
        print(f"    Perfect (0 issues): {perfect}/{total} ({perfect/total*100:.0f}%)")
        print(f"    Value-correct rate: {sum(1 for e in entries if e['json_valid'] and not any('wrong_value' in i for i in e['issues']))}/{total}")
        print(f"    Latency avg/min/max: {avg_lat:.0f} / {min_lat:.0f} / {max_lat:.0f} ms")

        # Issue breakdown
        all_issues = []
        for e in entries:
            all_issues.extend(e["issues"])
        if all_issues:
            print(f"    Issues:")
            for issue, count in Counter(all_issues).most_common():
                print(f"      {issue}: {count}")

    # Per-test breakdown across models
    test_labels = list(dict.fromkeys(e["test"] for e in results[models[0]]))

    # Build header
    header = f"  {'Test':<25}"
    for m in models:
        short = m.split(":")[-1] if ":" in m else m[-12:]
        header += f" {short:>10}"
    print(f"\n{'─' * 70}")
    print("  Per-test breakdown (perfect/runs)")
    print(f"{'─' * 70}")
    print(header)
    print(f"  {'─'*25}" + " ─────────" * len(models))

    for label in test_labels:
        row = f"  {label:<25}"
        for model in models:
            entries = [e for e in results[model] if e["test"] == label]
            ok = sum(1 for e in entries if len(e["issues"]) == 0)
            row += f" {ok}/{len(entries):>7}"
        print(row)

    # Recommendation
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATION")
    print(f"{'=' * 70}")

    for model in models:
        s = summaries[model]
        pct = s["perfect"] / s["total"] * 100
        status = "✅ VIABLE" if pct >= 95 else "⚠️  MARGINAL" if pct >= 85 else "❌ NOT READY"
        print(f"  {model}: {pct:.0f}% perfect, {s['avg_lat']:.0f}ms avg — {status}")

    print(f"\n  Viability threshold: ≥95% perfect rate")
    print(f"  Current production model should be the smallest VIABLE option")
    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Color Temp Model Comparison")
    parser.add_argument("--models", nargs="+", required=True, help="Models to compare")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--runs", type=int, default=10, help="Runs per test (default 10)")
    parser.add_argument("--output", default="eval_color_temp.json")
    args = parser.parse_args()

    print(f"\nColor Temp Model Comparison")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Runs:   {args.runs}")
    print()

    # Check connectivity + model availability
    try:
        resp = requests.get(f"{args.ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        print(f"  Available: {', '.join(available)}")
        for model in args.models:
            if model not in available:
                print(f"  WARNING: {model} not found — pull it first")
                sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Can't reach Ollama: {e}")
        sys.exit(1)

    total = len(TESTS) * len(args.models) * args.runs
    print(f"\n  Running {total} total inferences...\n")

    results = run_eval(args.ollama_url, args.models, args.runs)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output}")

    print_report(results, args.runs)


if __name__ == "__main__":
    main()
