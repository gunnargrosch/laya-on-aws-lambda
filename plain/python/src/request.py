import json


def _is_question(value):
    return isinstance(value, dict) and isinstance(value.get("type"), str)


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
    questions = parsed["questions"]
    if not isinstance(questions, dict):
        raise ValueError("questions must be an object keyed by question id")
    invalid_question_id = next((qid for qid, qdef in questions.items() if not _is_question(qdef)), None)
    if invalid_question_id is not None:
        raise ValueError(f'question "{invalid_question_id}" must be an object with a string type')
    return {"state": parsed["state"], "questions": questions}
