from flask import Flask, request, jsonify
from jarvis import run_jarvis
import os
import requests
import uuid
from urllib.parse import quote
from datetime import datetime

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Matrix alerting config
# ---------------------------------------------------------------------------
MATRIX_HOMESERVER = os.environ.get("MATRIX_HOMESERVER", "http://localhost:6167")
MATRIX_ACCESS_TOKEN = os.environ.get("MATRIX_ACCESS_TOKEN", "")
MATRIX_ROOM_ID = os.environ.get("MATRIX_ROOM_ID", "")


def post_to_matrix(message):
    """Send a plain text message to the Jarvis Matrix room."""
    encoded_room = quote(MATRIX_ROOM_ID, safe="")
    txn_id = str(uuid.uuid4())
    url = f"{MATRIX_HOMESERVER}/_matrix/client/v3/rooms/{encoded_room}/send/m.room.message/{txn_id}"
    headers = {
        "Authorization": f"Bearer {MATRIX_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "msgtype": "m.text",
        "body": message,
    }
    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        print(f"[alerts] Posted to Matrix: {message[:80]}...")
    except Exception as e:
        print(f"[alerts] Failed to post to Matrix: {e}")


# ---------------------------------------------------------------------------
# Grafana alert webhook
# ---------------------------------------------------------------------------
@app.route("/alerts", methods=["POST"])
def alerts():
    data = request.json
    print(f"[alerts] Received: {data}")

    alert_list = data.get("alerts", [])

    for alert in alert_list:
        alert_status = alert.get("status", "unknown")
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})

        name = labels.get("alertname", "Unknown Alert")
        summary = annotations.get("summary", "")
        description = annotations.get("description", "")
        instance = labels.get("instance", "")

        emoji = "🔴" if alert_status == "firing" else "✅"
        timestamp = datetime.now().strftime("%I:%M %p")

        message = f"{emoji} [{alert_status.upper()}] {name}\n"
        if instance:
            message += f"Node: {instance}\n"
        if summary:
            message += f"Summary: {summary}\n"
        if description:
            message += f"Detail: {description}\n"
        message += f"Time: {timestamp}"

        post_to_matrix(message)

    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Jarvis zone webhooks
# ---------------------------------------------------------------------------
@app.route("/webhook/lab", methods=["POST"])
def webhook_lab():
    data = request.json
    print(f"Event received: {data}")
    data["zone"] = "lab"
    run_jarvis(data)
    return jsonify({"status": "ok"})


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print(f"Event received: {data}")
    data["zone"] = "living_room"
    run_jarvis(data)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
