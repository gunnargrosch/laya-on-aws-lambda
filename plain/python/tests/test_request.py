import pytest

from request import parse_request


def test_parse_request_accepts_state_and_questions():
    req = parse_request('{"state": {"body": "hi"}, "questions": {"q": {}}}')
    assert req["state"] == {"body": "hi"}
    assert req["questions"] == {"q": {}}


def test_parse_request_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_request("{not json")


def test_parse_request_requires_state_and_questions():
    with pytest.raises(ValueError):
        parse_request('{"state": {}}')
