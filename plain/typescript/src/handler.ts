import { Agent } from "laya-ts";
import { BadRequest, parseRequest } from "./request.ts";

// Load the model lazily (not in the init phase). Lambda caps the init phase at 10s; a slow
// model read there gets interrupted, so we load on first invoke instead and cache it.
let agentPromise: Promise<Agent> | undefined;
function getAgent(): Promise<Agent> {
  if (!agentPromise) {
    agentPromise = Agent.load("/opt/model", {
      numThreads: Number(process.env.LAYA_THREADS) || 2,
    }).catch((err: unknown) => {
      // Don't cache a failed load, so the next invoke retries it.
      agentPromise = undefined;
      throw err;
    });
  }
  return agentPromise;
}

// Provisioned concurrency environments aren't held to the 10s init cap (they get up to 130s
// or the function timeout), so load the model during init there. Otherwise the first request
// on every provisioned environment would still pay the model load.
// The first predict() also pays one-time ONNX Runtime setup (several seconds), so run a
// throwaway prediction here too.
if (process.env.AWS_LAMBDA_INITIALIZATION_TYPE === "provisioned-concurrency") {
  const agent = await getAgent();
  await agent.predict("warm-up", { warmUp: { type: "noul", instructions: "Is this a warm-up?" } });
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
    // Log the details for CloudWatch; don't echo internals on a public URL.
    console.error("predict failed", err);
    return json(500, { error: "internal_error" });
  }
};
