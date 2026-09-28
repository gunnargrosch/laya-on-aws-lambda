import { test } from "node:test";
import assert from "node:assert/strict";
import { BadRequest, parseRequest } from "./request.ts";

test("parseRequest accepts state and questions", () => {
  const req = parseRequest(
    '{"state": {"body": "hi"}, "questions": {"q": {"type": "noul", "instructions": "?"}}}',
  );
  assert.deepEqual(req.state, { body: "hi" });
  assert.deepEqual(req.questions, { q: { type: "noul", instructions: "?" } });
});

test("parseRequest rejects invalid JSON", () => {
  assert.throws(() => parseRequest("{not json"), BadRequest);
});

test("parseRequest requires state and questions", () => {
  assert.throws(() => parseRequest('{"state": {}}'), BadRequest);
});
