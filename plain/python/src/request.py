import json


def parse_request(raw):
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_json") from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("state") is None
        or parsed.get("questions") is None
    ):
        raise ValueError("state and questions are required")
    return {"state": parsed["state"], "questions": parsed["questions"]}
