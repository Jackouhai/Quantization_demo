# AWQ Quantization

This document describes the AWQ quantization process for Qwen 3-4B with two calibration sample configurations: `SAMPLES215` and `SAMPLES512`.

## Directories & Models
- `Qwen3-4B-W4A16-awq/`: AWQ model with standard configuration
- `Qwen3-4B-W4A16-awq-SAMPLES512/`: AWQ model calibrated with 512 samples

## Serving with vLLM
Example serving the standard model:
```bash
vllm serve /home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq \
  --served-model-name Qwen3-4B-AWQ \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8001
```
The `--served-model-name` alias is used for benchmarking. If benchmark fails to find tokenizer, add `--tokenizer` pointing to the same model directory.

## Quick Benchmark
Using `bench.txt`:
```bash
vllm bench serve \
  --backend vllm \
  --model Qwen3-4B-AWQ \
  --tokenizer /home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 100 \
  --result-dir "./log/llm-compressor" \
  --save-result
```
Replace tokenizer/model directory accordingly to benchmark `SAMPLES512`.

## Notes
- AWQ optimizes based on activation-aware quantization, suitable for long context workloads
- Fewer calibration samples (`SAMPLES512`) provides faster and lighter processing, but consider quality trade-offs
- AWQ is typically faster at prefill; quality depends on calibration standards
