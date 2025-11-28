# GPTQ Quantization

This document describes the GPTQ quantization process for Qwen 3-4B with two calibration sample configurations: `SAMPLES215` and `SAMPLES512`.

## Directories & Models
- `Qwen3-4B-W4A16-GPTQ/`: GPTQ model with standard configuration
- `Qwen3-4B-W4A16-GPTQSAMPLES512/`: GPTQ model calibrated with 512 samples

## Serving with vLLM
Example serving the standard model:
```bash
vllm serve /home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ \
  --served-model-name Qwen3-4B-GPTQ \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```
Note: The `--served-model-name` alias is used for client/benchmark. If benchmark fails to find tokenizer, add `--tokenizer` pointing to the same model directory.

## Quick Benchmark
Using `bench.txt`:
```bash
vllm bench serve \
  --backend vllm \
  --model Qwen3-4B-GPTQ \
  --tokenizer /home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 100 \
  --result-dir "./log/llm-compressor" \
  --save-result
```
Replace tokenizer/model directory accordingly to benchmark `SAMPLES512`.

## Notes
- GPTQ W4A16 configuration is suitable for limited GPU VRAM, balancing accuracy
- Calibration data significantly affects quality; `SAMPLES512` reduces time but may slightly reduce quality
