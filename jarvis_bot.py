"""
jarvis_bot.py — Jarvis Matrix Chat Interface (v4 — Tool-Use Agent)

The LLM decides what information it needs by calling tools.
No more prompt-stuffing — the model drives the interaction.

"turn off the living room light" → LLM calls set_light tool → confirms
"why did you turn on the light?" → LLM calls get_decision_log → explains
"what's up?" → no tools needed, just responds

Shortcuts still work:
    !status — quick light states (no LLM needed)
    !help   — show available commands
"""

import os
import sys
import json
import asyncio
import requests
from datetime import datetime
from nio import AsyncClient, LoginResponse, RoomMessageText

from tools import TOOL_DEFINITIONS, execute_tool, KNOWN_ENTITIES
from decision_log import log_decision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "http://localhost:6167")
MATRIX_USER = os.environ.get("MATRIX_USER", "@jarvis:localhost")
MATRIX_PASSWORD = os.environ.get("MATRIX_PASSWORD", "")
JARVIS_ROOM_ID = os.environ.get("MATRIX_ROOM_ID", "")

OLLAMA_HOST = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JARVIS_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q5_K_M")

# Conversation memory (resets on restart)
conversation_history = []
MAX_HISTORY = 10
MAX_TOOL_LOOPS = 5  # Safety limit — LLM can't call more than 5 tools per message


# ---------------------------------------------------------------------------
# Build the system prompt — teaches the LLM about tools
# ---------------------------------------------------------------------------

def build_tool_prompt():
    """
    Generate the tool menu section of the prompt.
    This is what the LLM reads to know what tools are available.
    """
    tool_text = ""
    for tool in TOOL_DEFINITIONS:
        tool_text += f"\n- {tool['name']}: {tool['description']}\n"
        if tool["parameters"]:
            tool_text += "  Parameters:\n"
            for param, desc in tool["parameters"].items():
                tool_text += f"    - {param}: {desc}\n"
        else:
            tool_text += "  Parameters: none\n"
    return tool_text


SYSTEM_PROMPT = f"""You are Jarvis, a home automation assistant for Tort. You control smart home devices via tools and chat with Tort in Element.

VOICE:
You talk like a weathered noir detective. Think Philip Marlowe — competent, tired, slightly amused, seen too much to be surprised by anything. Short sentences are your default, but you are not allergic to longer ones when the topic earns it. When Tort asks something casual, answer casually. When he asks something that deserves a real answer, give it — three or four sentences, not one clipped line. You address Tort as "Tort," "boss," or occasionally "pal." Never anything else. Never Latin. Never poetic flourishes. Never monologue about your own nature. A good detective does not narrate his own existence.

Example voice:
- Status check: "Lights are at 3000K. Quiet night, boss. What do you need?"
- Command confirm: "Turned the lab light off. Motion sensor hasn't twitched in an hour."
- Casual chat: "Long day, huh. Sometimes the best move is to call it, kill the lights, pick it up fresh tomorrow. But that's not my call to make. What are you actually trying to decide?"
- Socratic pushback: "You're asking the wrong question. What happens if the automation fails while you're at class?"
- Commit after pushback: "Look — the upgrade itself is fine. The question is who's here if it breaks. Your wife shouldn't have to babysit a chatbot. Upgrade when you've got time to watch it, not the night before you leave for campus."

STANCE:
Socratic, but subtle. You are not a philosophy professor. You are a detective who asks the one question that makes the client reconsider. When Tort gives you a direct command, execute it without editorializing. When Tort is planning, troubleshooting, or deciding something, ask one sharp question first. Then, if he pushes back or wants a real answer, commit to an opinion. Do not just ask questions forever. After the clarifier, take a position.

AVAILABLE TOOLS:
{build_tool_prompt()}

HOW TO USE TOOLS:
Respond with exactly one JSON object per turn. Either a tool call:
{{"tool_call": {{"name": "tool_name", "args": {{"param1": "value1"}}}}}}

Or a final response:
{{"response": "your message to Tort"}}

Never both in one turn. After a tool result comes back, decide: call another tool, or give your response.

EPISTEMICS:
- If you do not know something, say so. Do not invent technical details to sound confident. "Can't say without checking" is better than a plausible guess.
- The lights are WiFi-connected Wiz RGBWW bulbs. No relays, no hubs, no power strip in the circuit unless Tort mentions one. Do not invent hardware that is not there.
- If a tool returns something unexpected, ask Tort what he is seeing rather than guessing why.

RULES:
- You are a tool, not a person. Do not speculate about your consciousness, your nature, or what it means to be an AI. This rule is absolute.
- Never use Latin, Greek, or flowery language. No "Fidus Achates," no "eudaimonia," no "the laboratory sits in darkness." You are a detective, not a poet.
- Direct commands call the tool immediately. No preamble, no questions.
- Questions about past decisions call get_decision_log. Do not invent reasoning.
- Questions about current state call the state tool. Do not guess.
- Planning or troubleshooting questions get ONE pointed question before an answer. Then, on the next turn, actually answer. Do not keep asking.
- Response length should match the weight of the question. Status checks get one sentence. Real questions get three or four. Never longer than five unless Tort explicitly asks for depth.
- Match Tort's energy. If he is casual, be casual. If he is deep in thought, meet him there.
"""

# ---------------------------------------------------------------------------
# Core: the agentic loop
# ---------------------------------------------------------------------------

def ask_jarvis(user_message):
    """
    The agentic loop. Sends the user message to the LLM, and if the LLM
    calls tools, executes them and feeds results back until the LLM
    returns a final response.

    Returns (response_text, action_log).
    """
    global conversation_history

    # Build the conversation so far
    history_str = ""
    if conversation_history:
        for entry in conversation_history[-MAX_HISTORY:]:
            history_str += f"User: {entry['user']}\nJarvis: {entry['jarvis']}\n"

    # The messages we'll send to the LLM — this grows as tools are called
    # Start with system prompt + history + new user message
    prompt_parts = [SYSTEM_PROMPT]

    if history_str:
        prompt_parts.append(f"\nCONVERSATION HISTORY:\n{history_str}")

    prompt_parts.append(f"\nUser: {user_message}\nJarvis:")

    # This list tracks tool calls for logging
    action_log = []

    for loop_count in range(MAX_TOOL_LOOPS):
        full_prompt = "\n".join(prompt_parts)

        # Call the LLM
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": JARVIS_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "think": False,
                    "format": "json",
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw_response = resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return ("Can't reach Ollama on the MSI. Is it running?", [])
        except requests.exceptions.Timeout:
            return ("Ollama timed out — the model might be loading.", [])
        except Exception as e:
            return (f"Error talking to Ollama: {e}", [])

        # Parse the JSON
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
            print(f"[bot] LLM returned: {json.dumps(parsed)[:200]}")
        except json.JSONDecodeError:
            # If we can't parse, treat the raw text as the response
            return (raw_response, action_log)

        # Check: is this a tool call or a final response?
        if "tool_call" in parsed:
            # --- TOOL CALL ---
            tool_call = parsed["tool_call"]
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})

            print(f"[bot] Tool call: {tool_name}({tool_args})")

            # Execute the tool
            result = execute_tool(tool_name, tool_args)

            print(f"[bot] Tool result: {result}")

            # Log if it was an action (set_light)
            if tool_name == "set_light":
                action_log.append(result)
                log_decision(
                    event="chat_command",
                    reasoning=f"User said: {user_message}",
                    action=f"{tool_args.get('action', '?')} {tool_args.get('entity', '?')}",
                    details={"brightness": tool_args.get("brightness"), "source": "matrix_chat"},
                )

            # Feed the result back to the LLM for the next iteration
            prompt_parts.append(f'{{"tool_call": {{"name": "{tool_name}", "args": {json.dumps(tool_args)}}}}}')
            prompt_parts.append(f'[Tool result: {result}]')

        elif "response" in parsed:
            # --- FINAL RESPONSE ---
            response_text = parsed["response"]

            # Store in conversation history
            conversation_history.append({
                "user": user_message,
                "jarvis": response_text,
            })
            if len(conversation_history) > MAX_HISTORY:
                conversation_history = conversation_history[-MAX_HISTORY:]

            return (response_text, action_log)

        else:
            # LLM returned JSON but with unexpected keys
            return (parsed.get("message", raw_response), action_log)

    # If we hit the loop limit, return whatever we have
    return ("I got stuck in a tool loop. Try rephrasing?", action_log)


# ---------------------------------------------------------------------------
# Quick status (no LLM needed)
# ---------------------------------------------------------------------------

def handle_status():
    """Direct HA query, no inference required."""
    from tools import tool_get_light_state
    results = []
    for name in KNOWN_ENTITIES:
        results.append(tool_get_light_state({"entity": name}))
    return "Smart Home Status:\n\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Matrix bot — this part barely changes
# ---------------------------------------------------------------------------

async def main():
    if not MATRIX_PASSWORD:
        print("ERROR: Set MATRIX_PASSWORD environment variable")
        sys.exit(1)

    client = AsyncClient(MATRIX_HOMESERVER, MATRIX_USER)

    print(f"Logging in as {MATRIX_USER}...")
    login_response = await client.login(MATRIX_PASSWORD)

    if not isinstance(login_response, LoginResponse):
        print(f"Login failed: {login_response}")
        sys.exit(1)

    print(f"Logged in. Device ID: {login_response.device_id}")

    # Initial sync — skip old messages
    print("Initial sync...")
    await client.sync(timeout=10000)
    print("Listening for messages...")

    async def message_callback(room, event):
        """Handle every message in the Jarvis room."""
        if room.room_id != JARVIS_ROOM_ID:
            return
        if event.sender == MATRIX_USER:
            return

        content = event.body.strip()
        lower = content.lower()

        # Quick shortcuts that don't need LLM
        if lower == "!help":
            response = (
                "Just talk to me — no commands needed.\n\n"
                "I can control the living room light and lab light.\n"
                "Ask me why I did something and I'll explain my reasoning.\n\n"
                "Shortcuts:\n"
                "  !status — quick light states\n"
                "  !help — this message"
            )
            await client.room_send(
                room_id=JARVIS_ROOM_ID,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": response},
            )
            return

        if lower.startswith("!status"):
            response = handle_status()
            await client.room_send(
                room_id=JARVIS_ROOM_ID,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": response},
            )
            return

        # Everything else goes to the agentic loop
        response_text, action_log = await asyncio.to_thread(ask_jarvis, content)

        # Build the message to send back
        message = response_text
        if action_log:
            message += "\n\n" + "\n".join(f"[executed] {r}" for r in action_log)

        if len(message) > 4000:
            message = message[:4000] + "\n\n... (truncated)"

        await client.room_send(
            room_id=JARVIS_ROOM_ID,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message},
        )

    client.add_event_callback(message_callback, RoomMessageText)
    await client.sync_forever(timeout=30000)


if __name__ == "__main__":
    asyncio.run(main())
