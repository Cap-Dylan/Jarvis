# Jarvis

Fully local agentic smart home system. Two input paths — automated motion-triggered events and a Matrix chat interface — share one decision log, one Home Assistant backend, and a model-routing inference layer. No cloud LLMs, no SaaS dependencies.

---

## System Architecture

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         Input Layer                                  │
    │                                                                      │
    │   Tapo C121 ──► Frigate NVR ──► HA Automation ──► POST /webhook     │
    │                                                        │             │
    │   Element (Matrix) ──────────────────────► jarvis_bot.py             │
    └──────────────────────────────────────┬──────────────┬────────────────┘
                                           │              │
    ┌──────────────────────────────────────┴──────────────┴────────────────┐
    │                      Inference Layer (Ollama)                         │
    │                                                                      │
    │   webhook.py ──► llama3.2:3b ──► JSON color-temp decision            │
    │                  (deterministic, ~800ms, 100% JSON compliance)        │
    │                                                                      │
    │   jarvis_bot.py ──► Qwen3.5:9b ──► tool-use loop (≤5 iterations)    │
    │                     (conversational, ~88% tool-call accuracy)         │
    └──────────────────────────────────────┬──────────────┬────────────────┘
                                           │              │
    ┌──────────────────────────────────────┴──────────────┴────────────────┐
    │                       Execution Layer                                │
    │                                                                      │
    │   ha_client.py ──► Home Assistant REST API ──► Wiz RGBWW bulbs      │
    │                                                (2 zones: living      │
    │                                                 room + lab)          │
    │                                                                      │
    │   decisions.jsonl ◄── both paths log every decision with full        │
    │                       context: inputs, prompt, model output, action   │
    └─────────────────────────────────────────────────────────────────────┘
```

When a user asks Jarvis "why did you turn on the light?" in chat, the bot reads the same `decisions.jsonl` that the automated path writes to. The two input paths converge at the log and at the hardware they control.

---

## Model Routing

Both paths could run on the same model. They don't, because the workloads have different shapes — and the decision is backed by evaluation data, not intuition.

| Path | Model | Why | Eval |
|------|-------|-----|------|
| Color temperature | `llama3.2:3b` | Tight, deterministic — output is `{"color_temp_kelvin": int, "reason": str}`. Small model is fast and reliable for constrained JSON. | `eval_colortemp.py`: 100% JSON compliance, sub-second latency across 8 scenarios |
| Chat + tool calls | `Qwen3.5:9b` | Conversational coherence + clean `tool_call` JSON across multi-turn exchanges. Native tool-calling support. | `eval_models.py`: 88% accuracy across 8 bot conversation scenarios |

`format=json` is enforced in the Ollama call for the color-temp path, constraining the decoder so the model cannot emit tokens that break JSON structure. Combined with schema validation, parse rate is ~100%.

Eval harnesses and results are committed to this repository.

---

## Tool Definitions

Defined in [`tools.py`](tools.py), exposed through a `TOOL_DEFINITIONS` schema the LLM reads as part of the system prompt:

| Tool | Purpose |
|------|---------|
| `get_light_state` | Read on/off, brightness%, and color temp for a named light |
| `set_light` | Turn a named light on/off with optional brightness; logged to `decisions.jsonl` |
| `get_weather` | Pull weather state from HA — temp, humidity, cloud cover |
| `get_time` | Current time + period (morning/daytime/evening/night) + suggested brightness |
| `get_decision_log` | Read recent decisions for "why did you do X?" queries |

The agentic loop in `jarvis_bot.py` caps tool iterations at `MAX_TOOL_LOOPS = 5`.

---

## Reliability Engineering

- **Per-zone cooldown and turn-off timers** — `jarvis.py` keeps separate `last_turn_on_time` and `pending_off_timer` dicts keyed by zone. Motion in the lab can't cancel the living-room timer.
- **JSON parse fallback** — when the chat model splits its response into multiple JSON objects, a recovery path joins them rather than crashing. Tested explicitly in `eval_models.py`.
- **Markdown-fence stripping** — small models occasionally wrap JSON in code fences despite instructions. Both the bot and eval harness strip those before parsing.
- **Connection/timeout handling** — every Ollama and HA call has a timeout and returns a useful message to the user on failure.
- **No-LLM shortcuts** — `!status` queries HA directly. No inference cost, no failure modes from model behavior.

---

## Deployment

### Docker (production)

```bash
cp .env.example .env   # Fill in HA_URL, HA_TOKEN, OLLAMA_URL, MATRIX_*
docker compose up -d --build
```

Both `jarvis-webhook` and `jarvis-bot` come up with `restart: unless-stopped`. The decision log lives in a bind-mounted `./data/` directory.

### Local (development)

```bash
git clone https://github.com/Cap-Dylan/Jarvis.git && cd Jarvis
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python webhook.py        # automated path on :5050
python jarvis_bot.py     # chat path
```

### HA Automations

Two Home Assistant automations POST to the webhook on motion events:

```yaml
service: rest_command.jarvis_event
data:
  event: person_detected    # or occupancy_cleared
```

For the lab zone, POST to `/webhook/lab` instead of `/webhook`.

---

## Stack

| Layer | Technology |
|-------|------------|
| Inference (chat) | Qwen3.5:9b via Ollama |
| Inference (color temp) | llama3.2:3b via Ollama |
| Orchestration | Python 3.11 (Flask + requests) |
| Chat transport | Continuwuity (Matrix homeserver) + Element |
| Computer vision | Tapo C121 → Frigate NVR → Home Assistant |
| Actuation | Home Assistant REST API → Wiz RGBWW bulbs |
| Containers | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana → Matrix alerts |
| Mesh VPN | Tailscale |

---

## Roadmap

- **Control Center integration** — [Jarvis Control Center](https://github.com/Cap-Dylan/control-center) provides a real-time operator console for decision inspection, HA overrides, and Prometheus-based alerting. Five of six phases complete.
- **Planner/Executor/Validator architecture** — the current single-loop tool-use design works well for short tasks; multi-step plans would benefit from explicit planning.
- **Voice surface** — local STT/TTS on the M4 Pro fronting the same agent.

---

## License

MIT — see [LICENSE](LICENSE).
