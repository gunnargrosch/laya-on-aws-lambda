import { Agent } from "laya-ts";
import { BadRequest, parseRequest } from "./request.ts";

// Load the model lazily (not in the init phase). Lambda caps the init phase at 10s; a slow
// model read there gets interrupted, so we load on first invoke instead and cache it.
let agentPromise: Promise<Agent> | undefined;
function getAgent(): Promise<Agent> {
  if (!agentPromise) {
    agentPromise = Agent.load("/opt/model", {
      numThreads: Number(process.env.LAYA_THREADS) || 2,
    });
  }
  return agentPromise;
}

function json(statusCode: number, payload: unknown) {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}

export const handler = async (event: any) => {
  try {
    const raw = event.isBase64Encoded
      ? Buffer.from(event.body ?? "", "base64").toString("utf8")
      : event.body ?? "";
    const { state, questions } = parseRequest(raw);
    const agent = await getAgent();
    const result = await agent.predict(state, questions);
    return json(200, result);
  } catch (err) {
    if (err instanceof BadRequest) {
      return json(400, { error: err.message });
    }
    return json(500, {
      error: err instanceof Error ? err.message : "internal_error",
    });
  }
};
