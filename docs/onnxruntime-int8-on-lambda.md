# ONNX Runtime INT8 quantized models compute incorrect (uniform) results on AWS Lambda x86_64

Status: resolved (cause found 2026-09-28, see Resolution)
Reported by: Gunnar Grosch
Date: 2026-09-28

## Resolution (2026-09-28)

**Not a Lambda or ONNX Runtime bug.** Lambda x86_64 environments (eu-north-1, 6 fresh environments) report `Intel(R) Xeon(R) Processor @ 2.50GHz` with **AVX2 only**: no AVX-512, no AVX512-VNNI, no AVX-VNNI, no AMX. `quantize_dynamic` defaults to U8S8 (uint8 activations, int8 weights), which the [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) say can saturate on x86 CPUs without VNNI. The development CPU has AVX-VNNI, so it doesn't saturate. The AVX-512 hypothesis below was wrong.

Max abs diff vs fp32 on Lambda, same models as the repro:

| Variant | matmul_add (128x256) | matmul_big (768x768) |
| --- | --- | --- |
| int8, per-channel (as reported) | 0.239 | 0.708 |
| int8, per-tensor | 0.236 | 0.707 |
| int8, `reduce_range=True` | 0.009 | 0.030 |
| uint8 weights (`QuantType.QUInt8`) | 0.006 | 0.018 |
| development CPU, int8 | 0.005 | 0.017 |

Both documented fixes work on the single-MatMul models. On the full Laya model they still drift too much to ship: `reduce_range=True` moves one `noul` answer from 0.973 to 0.468, and uint8 weights move probabilities by 0.07-0.14. The sample keeps fp32.

## Summary (original report)

An ONNX model quantized to INT8 with `onnxruntime.quantization.quantize_dynamic` predicts correctly when run on a workstation (and inside the same container image under Docker on that workstation), but returns **uniform / zero logits** when the identical container image runs on AWS Lambda x86_64. The same model run unquantized (fp32), or converted to fp16, predicts correctly on Lambda. The failure is therefore specific to the **INT8 execution path**, and it only appears on Lambda's CPU, not on the development CPU.

The most likely cause is an ONNX Runtime MLAS INT8 kernel issue on the AVX-512 (VNNI/AMX) code path, which the development CPU does not exercise (it has AVX-VNNI but no AVX-512).

## Environment

| | Development (correct) | AWS Lambda (incorrect) |
| --- | --- | --- |
| CPU | Intel Core Ultra 7 258V (Lunar Lake) | Intel Xeon @ 2.50GHz (Lambda x86_64) |
| SIMD | AVX2, AVX-VNNI (no AVX-512) | AVX2 only, no VNNI (confirmed from `/proc/cpuinfo`) |
| Runtime | Docker container, Lambda RIE (`docker run ... -p 9000:8080`) | AWS Lambda, x86_64, eu-north-1 |
| Node.js | v24.21.0 (same image) | v24.21.0 |
| onnxruntime-node | 1.30.0 | 1.30.0 |
| Image | identical (same digest) | identical (same digest) |

The image is byte-identical between the two environments: locally we run `docker run` on the exact image that was pushed to ECR and deployed, so the container, model files, and code are not variables.

## Symptom

For a multi-question classification request, the INT8 model returns an identical probability for every option (softmax of equal logits):

- `choice` question with 3 options: `{"billing": 0.3333, "technical": 0.3333, "other": 0.3334}`, `confidence: 0`
- `score` question with 3 levels: `{"0": 0.3336, "1": 0.3336, "2": 0.3328}`
- `noul` (boolean): `noul: 0.5`

The fp32 and fp16 versions of the same model return the correct, confident distribution for the same input on Lambda (e.g. `{"billing": 0.9863, "technical": 0.0064, "other": 0.0073}`).

## What works vs what does not (all on Lambda, same image, same input)

| Model | Precision | Local (Lunar Lake) | Lambda x86_64 |
| --- | --- | --- | --- |
| encoder.onnx + head.onnx | fp32 | correct | correct |
| encoder.onnx + head.onnx | fp16 (ORT converter) | correct | correct |
| encoder.onnx + head.onnx | **INT8 (dynamic, per-channel)** | **correct** | **incorrect (uniform)** |

So the failure is specific to INT8 and specific to the Lambda CPU.

## Model and quantization

- Source model: `convaiinnovations/laya` (ModernBERT-large encoder + a decision head), exported to a split ONNX graph with `laya-ts/scripts/export_onnx.py` (`encoder.onnx`, `head.onnx`, opset 18, fp32).
- Quantization (Python, in the export venv):

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

for name in ("encoder", "head"):
    model = onnx.load(f"model/{name}.onnx")   # loads external data
    del model.graph.value_info[:]
    quantize_dynamic(
        model_input=model,
        model_output=f"model/{name}.int8.onnx",
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
```

- Resulting size: encoder 554 MB (from ~1.5 GB), head 46 MB (from ~101 MB).

## Reproduction

1. Build the quantized models with the snippet above.
2. Run them locally with `onnxruntime-node` or via the Lambda RIE container: **correct output**.
3. Deploy the same image to Lambda (x86_64) with a handler that loads `encoder.onnx` / `head.onnx` from `/opt/model` and runs `InferenceSession` + the laya-ts decision head: **uniform output**.

The handler is a plain `onnxruntime-node` `InferenceSession.create()` + `session.run()`. No Lambda-specific code path is involved beyond the container runtime.

## Evidence

The following is emitted by `onnxruntime-node` in both environments; only the numeric result differs.

Local (Docker RIE, same image), correct:

```json
{"answers":{"department":{"choice":"billing","probabilities":{"billing":0.8765,"technical":0.0644,"other":0.0591},"answer_confidence":0.8765}, ...}}
```

Lambda (same image, same request), incorrect:

```json
{"answers":{"department":{"choice":"other","probabilities":{"billing":0.3333,"technical":0.3333,"other":0.3334},"confidence":0,"answer_confidence":0.3334}, ...}}
```

CloudWatch confirms it is not a timeout or memory issue: the invocation completes in ~600 ms (warm) with `Max Memory Used` well under the limit, and returns HTTP 200 with the uniform body.

## Minimal reproduction

We reduced this to a single quantized `MatMul` with no Laya code involved. `gen.py` builds four small fp32 models, quantizes each with the same `quantize_dynamic` call, and the repro deploys a tiny Lambda (`onnxruntime-int8-repro`) that runs fp32 and INT8 on a fixed input and reports the max absolute difference. Same model files, same image, same input in both environments:

| Model | max abs diff (local, AVX-VNNI) | max abs diff (Lambda x86_64) |
| --- | --- | --- |
| `matmul_add` (MatMul + Add) | 0.004964 | **0.239** |
| `matmul_softmax` (MatMul + Softmax) | 0.000042 | 0.001647 |
| `matmul_add_softmax` | 0.000039 | 0.001724 |
| `matmul_big` (768x768) | 0.016641 | **0.708** |

A single INT8 `MatMul + Add` is off by ~48x between the two environments (0.004964 locally vs 0.239405 on Lambda), and `matmul_big` by ~43x. The standalone, deployable repro is published at **https://github.com/gunnargrosch/onnxruntime-int8-on-lambda**. This is the artifact to hand to the ONNX Runtime / Lambda teams.

## Related known issues in ONNX Runtime

This is not an isolated oddity: ONNX Runtime has an active family of ISA-dependent quantized-GEMM correctness bugs.

- **#29853** (open, Jul 2026) "MLAS AVX2 M=1 CompInt8 SQNBit GEMM produces wrong results for asymmetric 4-bit weights": the AVX2 M=1 kernel is wrong on hosts **without** AVX-512; the AVX-512 kernel is correct. Fix in PR #29886. Same shape of bug as ours: single-row (M=1) quantized GEMM computes wrong numbers on one ISA path and is correct on another, with the same model/code.
- **#29849** (open, Jul 2026) "CPU MatMulNBits accuracy_level=4 (int8 activation quant) selects wrong argmax token on massive-activation LLMs".
- **#29862** (open, Jul 2026) "MLAS: Validating x86 kernels that no CI machine can reach": ONNX Runtime itself notes that some x86 kernel paths are not exercised in CI on the CPUs that reach them.
- **#26992** "ORT_ENABLE_ALL graph optimization causes significant slowdowns on Intel CPUs (vs AMD)": Intel-specific kernel-selection differences.
- **#7524** (closed stale, 2021) "Bug in MatMulInteger": asymmetric zero-point `MatMulInteger` produced wrong results.

None of these is an exact match for what we see (ours is 8-bit dynamic `QInt8` MatMul, M=1, wrong on the Lambda CPU while correct on an AVX-VNNI host; #29853 is 4-bit and wrong on AVX2). But together they establish that MLAS quantized MatMul / GEMV kernels have had ISA-specific wrong-results bugs, and that the failure can be exactly this pattern: one ISA path right, another wrong, same model and code. That makes an ONNX Runtime kernel bug the leading explanation, with the Lambda CPU simply being the hardware that reaches the affected path.

Links: [#29853](https://github.com/microsoft/onnxruntime/issues/29853) / [PR #29886](https://github.com/microsoft/onnxruntime/pull/29886), [#29849](https://github.com/microsoft/onnxruntime/issues/29849), [#29862](https://github.com/microsoft/onnxruntime/issues/29862), [#26992](https://github.com/microsoft/onnxruntime/issues/26992), [#7524](https://github.com/microsoft/onnxruntime/issues/7524).

## Analysis / hypothesis (superseded by Resolution)

- The failure only occurs on the AVX-512-capable Lambda CPU; it does not occur on the AVX-VNNI development CPU, and it does not occur for fp32 or fp16 on either. This points at the INT8 (QInt8) MatMul kernels in ONNX Runtime MLAS, specifically the AVX-512 VNNI / AMX implementation.
- Equal, uniform logits are consistent with the quantized MatMul producing zeros (or a constant) rather than the correct integer accumulation, i.e. the quantized weights/scales are not being applied correctly on that kernel path.
- `per_channel=True` is the quantization mode in use. It would be worth testing `per_channel=False` on Lambda to see whether the failure is specific to per-channel scales on the AVX-512 path.

## Impact

- INT8 is the only precision that meaningfully reduces the model (roughly 4x vs fp32). Its failure on Lambda removes the best option for cutting cold-start time and image size for CPU inference. fp16 halved the size but provided no clear cold-start benefit and introduced small numerical drift (~0.01 on probabilities).
- This affects any workload deploying ONNX Runtime INT8 models to Lambda, not just this project. Serving quantized classification/embedding models on CPU Lambda is a common pattern.

## What is ruled out

- Not the image or the code: the identical image digest runs correctly locally under the Lambda RIE and incorrectly on Lambda.
- Not the ONNX Runtime version: onnxruntime-node 1.30.0 in both.
- Not Node version: v24.21.0 in both.
- Not memory or timeout: the call completes successfully and returns uniform numbers.
- Not the model export: the fp32 and fp16 exports of the same graph are correct on the same Lambda CPU.

## Related but separate issue (not Lambda-specific)

While investigating, we also found that the Laya Python ONNX export (`laya/scripts/export_onnx.py`, a different single-graph export) fails for multi-question batches with:

```
Non-zero status code returned while running Add node. Name:'node_Add_3057'
... Attempting to broadcast an axis by a dimension other than 1. 51 by 153
```

This reproduces **locally** on onnxruntime 1.22 and 1.30, so it is an upstream Laya-export / ONNX issue, not a Lambda issue. Mentioned only so it is not mistaken for the same bug.

## Suggested next steps / what would help

- A minimal standalone repro would strengthen this: a small ONNX model with one `MatMulInteger`/`QLinearMatMul` op, quantized with `quantize_dynamic`, that returns the correct result on an AVX-512 Xeon and incorrect/uniform on Lambda (or vice versa). Happy to build one with guidance.
- Confirm the exact Lambda host CPU microarchitecture for the affected invocations and whether AVX-512 VNNI / AMX is in use.
- Test `per_channel=False` and `op_types_to_quantize` variants to narrow the kernel.
- Determine whether the bug reproduces on ONNX Runtime CPU with `ORT_DISABLE_ALL` / reduced optimization, or with `intra_op_num_threads=1`.

## Artifacts

- Sample repo and full context: `github.com/gunnargrosch/laya-on-aws-lambda` (TypeScript and Python variants; the TypeScript path is where this was found).
- Minimal standalone repro: **https://github.com/gunnargrosch/onnxruntime-int8-on-lambda** (`sam build && sam deploy`), plus `gen.py` to regenerate the models.
- Model: `convaiinnovations/laya` (Apache-2.0); quantized splits are build artifacts of `scripts/export.sh`.
