#!/usr/bin/env bash
# Build and deploy this sample for your machine's architecture.
#
# Lambda container images are single-architecture, and SAM builds for the host, so this
# detects the host and passes the matching Lambda architecture. A Mac (arm64) and an x86
# machine both work with no extra steps. Extra args are forwarded to `sam deploy`.
set -euo pipefail

case "$(uname -m)" in
  arm64|aarch64) arch=arm64 ;;
  x86_64|amd64)  arch=x86_64 ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
echo "Host $(uname -m) -> Lambda $arch"

# Always pass ProvisionedConcurrency explicitly, so a plain ./deploy.sh turns it back off
# (otherwise SAM keeps the previous value and the change is a no-op).
overrides="LambdaArchitecture=$arch ProvisionedConcurrency=${PROVISIONED_CONCURRENCY:-0}"

sam build
exec sam deploy \
  --stack-name "${STACK_NAME:-laya-decision-api-py}" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --resolve-image-repos \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --parameter-overrides $overrides \
  "$@"
