#!/usr/bin/env bash
# Build and deploy this sample, then send one warm-up request.
#
# Deploys x86_64 by default on every machine: fp32 inference measured 2-4x slower on Lambda arm64
# (Graviton2 ~3.8 s, Graviton3 ~2.1 s, x86_64 ~1.0 s at 4096 MB), so it costs more per request
# even at the lower arm64 price. On an arm64 machine the image builds under emulation, which is
# slower but works. Set LAMBDA_ARCHITECTURE=arm64 to deploy to Graviton anyway. Extra args are
# forwarded to `sam deploy`; --profile and --region are also used for the warm-up.
set -euo pipefail

stack="${STACK_NAME:-laya-decision-api-ts}"

arch="${LAMBDA_ARCHITECTURE:-x86_64}"
case "$arch" in
  arm64|x86_64) ;;
  *) echo "LAMBDA_ARCHITECTURE must be x86_64 or arm64, got: $arch" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
echo "Host $(uname -m) -> Lambda $arch"

# Always pass both parameters explicitly: SAM keeps a parameter's previous value when it isn't
# passed, so a plain ./deploy.sh would otherwise leave provisioned concurrency on, or deploy
# an image built for one architecture onto a function set to the other.
overrides="LambdaArchitecture=$arch ProvisionedConcurrency=${PROVISIONED_CONCURRENCY:-0}"

sam build --parameter-overrides $overrides
sam deploy \
  --stack-name "$stack" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --resolve-image-repos \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --parameter-overrides $overrides \
  "$@"

# The first request after a deploy reads the whole model from an image Lambda hasn't cached
# yet, which can take a few minutes. Later cold starts take seconds. Send one request now so
# real traffic doesn't pay for it.
if [ "${SKIP_WARMUP:-0}" = "1" ]; then
  exit 0
fi

aws_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile|--region) aws_args+=("$1" "${2:-}"); shift 2 || shift ;;
    --profile=*|--region=*) aws_args+=("$1"); shift ;;
    *) shift ;;
  esac
done

url=$(aws cloudformation describe-stacks --stack-name "$stack" ${aws_args[@]+"${aws_args[@]}"} \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" --output text)

echo "Warming up (the first request after a deploy can take a few minutes)..."
warmup_body='{"state": "warm-up", "questions": {"warm_up": {"type": "noul", "instructions": "Is this a warm-up?"}}}'
status=000
SECONDS=0
for attempt in 1 2 3 4 5 6; do
  status=$(curl -s -o /dev/null -w '%{http_code}' -m 900 "$url" \
    -H 'content-type: application/json' -d "$warmup_body") || status=000
  # A new Function URL can return 403 for a few seconds while its permissions apply.
  if [ "$status" = "403" ] && [ "$attempt" -lt 6 ]; then
    sleep 10
    continue
  fi
  break
done

if [ "$status" = "200" ]; then
  echo "Warm after ${SECONDS}s."
else
  echo "Warm-up returned HTTP $status after ${SECONDS}s. The function is deployed, but the next request may be slow." >&2
fi
echo "Function URL: $url"
