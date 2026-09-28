import base64
import json

import laya

from request import parse_request

# Load lazily on first invoke (not in the init phase), then cache.
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = laya.load("/opt/model", device="cpu")
    return _agent


def _response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event, context):
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    try:
        request = parse_request(body)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        result = get_agent().predict(request["state"], request["questions"])
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": str(exc)})

    return _response(200, result)
