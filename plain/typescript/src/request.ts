import type { QuestionDef } from "laya-ts";

export interface SystemOneRequest {
  state: unknown;
  questions: Record<string, QuestionDef>;
}

export class BadRequest extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isQuestionDef(value: unknown): value is QuestionDef {
  return isRecord(value) && typeof value.type === "string";
}

export function parseRequest(raw: string): SystemOneRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new BadRequest("invalid_json");
  }
  if (!isRecord(parsed) || parsed.state == null || parsed.questions == null) {
    throw new BadRequest("state and questions are required");
  }
  const { state, questions } = parsed;
  if (!isRecord(questions)) {
    throw new BadRequest("questions must be an object keyed by question id");
  }
  const invalidQuestionId = Object.keys(questions).find((questionId) => !isQuestionDef(questions[questionId]));
  if (invalidQuestionId !== undefined) {
    throw new BadRequest(`question "${invalidQuestionId}" must be an object with a string type`);
  }
  return { state, questions: questions as Record<string, QuestionDef> };
}
