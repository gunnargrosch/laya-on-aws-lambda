export interface SystemOneRequest {
  state: unknown;
  questions: Record<string, unknown>;
}

export class BadRequest extends Error {}

export function parseRequest(raw: string): SystemOneRequest {
  let parsed: any;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new BadRequest("invalid_json");
  }
  if (
    !parsed ||
    typeof parsed !== "object" ||
    parsed.state == null ||
    parsed.questions == null
  ) {
    throw new BadRequest("state and questions are required");
  }
  return { state: parsed.state, questions: parsed.questions };
}
