# Accuracy Benchmark Guide

This document guides you through benchmarking ACCURACY for quantized models (GPTQ/AWQ) using the `benchmark/main.py` script. No need to serve via vLLM endpoint; the script calls `lm-eval` with internal VLLM backend to evaluate test suites like MMLU, Hellaswag, ARC.

## How It Works
- `main.py` uses `lm_eval.simple_evaluate` with `VLLM` model adapter to load models from local paths (`pretrained=path`)
- Runs tasks: `mmlu`, `hellaswag`, `arc_easy`, `arc_challenge` with `num_fewshot=0` and `limit=50` (configurable)
- Results include multiple metrics; script automatically selects `acc_norm` (if available) or `acc` and converts to percentage
- Saves 3 types of files:
  - Full raw: `benchmark/benchmark_logs/raw/{MODEL}_FULL_{TIMESTAMP}.json`
  - Summary JSON: `benchmark/benchmark_logs/summary/{MODEL}_SUMMARY_{TIMESTAMP}.json`
  - CSV report: `benchmark/benchmark_logs/csv/{MODEL}_REPORT_{TIMESTAMP}.csv`

## Preparation
- Identify local model path (quantized GPTQ/AWQ). Example for GPTQ:
  - `quantization/gptq/Qwen3-4B-W4A16-GPTQ/`
  - or `SAMPLES512` version: `quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512/`
- Verify Python environment has all dependencies installed (see `pyproject.toml` or `requirements.txt`)

## Running Accuracy Benchmark
1) Open `benchmark/main.py` and adjust the `path` variable to point to the corresponding model directory, for example:
```python
path = "/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ"
model_name = "Qwen3-4B-W4A16-GPTQ"
```
2) Run the script:
```bash
uv run main.py
```

## Reading Results
- Console output: table of tasks and percentage scores
- Files saved in `benchmark/benchmark_logs/` in 3 groups: `raw/`, `summary/`, `csv/`
- For MMLU, script displays additional "MMLU Average" if available

## Additional Customization
- Adjust `tasks`, `limit`, `batch_size`, `max_model_len`, `gpu_memory_utilization` in `main.py` as needed
- Change `path` to AWQ model or `SAMPLES512` version to compare accuracy between methods and calibration sets

## Notes
- If using Hugging Face cache paths (e.g., `~/.cache/huggingface/...`), ensure the snapshot has complete tokenizer and weights
- When switching models, only change `path` and `model_name` in `main.py`; no need to modify the run command
