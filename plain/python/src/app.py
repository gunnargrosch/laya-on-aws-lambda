import base64
import json
import logging
import os

import laya

from request import parse_request

logger = logging.getLogger(__name__)

# Load lazily on first invoke (not in the init phase), then cache.
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = laya.load("/opt/model", device="cpu")
    return _agent


# Provisioned concurrency environments aren't held to the 10s init cap (they get up to 130s
# or the function timeout), so load the model during init there. Otherwise the first request
# on every provisioned environment would still pay the model load.
# The first predict() also pays one-time runtime setup, so run a throwaway prediction too.
if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "provisioned-concurrency":
    get_agent().predict("warm-up", {"warm_up": {"type": "noul", "instructions": "Is this a warm-up?"}})


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
    except Exception:  # noqa: BLE001
        # Log the details for CloudWatch; don't echo internals on a public URL.
        logger.exception("predict failed")
        return _response(500, {"error": "internal_error"})

    return _response(200, result)
