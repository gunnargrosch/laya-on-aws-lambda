import pytest

from request import parse_request


def test_parse_request_accepts_state_and_questions():
    req = parse_request('{"state": {"body": "hi"}, "questions": {"q": {"type": "noul", "instructions": "?"}}}')
    assert req["state"] == {"body": "hi"}
    assert req["questions"] == {"q": {"type": "noul", "instructions": "?"}}


def test_parse_request_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_request("{not json")


def test_parse_request_requires_state_and_questions():
    with pytest.raises(ValueError):
        parse_request('{"state": {}}')


def test_parse_request_rejects_questions_that_are_not_an_object():
    with pytest.raises(ValueError):
        parse_request('{"state": {}, "questions": ["q"]}')


def test_parse_request_rejects_a_question_without_a_string_type():
    with pytest.raises(ValueError, match='question "q" must be an object with a string type'):
        parse_request('{"state": {}, "questions": {"q": {"instructions": "?"}}}')
