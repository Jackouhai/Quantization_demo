# Quantization Demo: GPTQ & AWQ Benchmarks for Qwen 3-4B

This project benchmarks and implements quantization for the Qwen 3-4B model using two methods: **GPTQ** and **AWQ**. Each method is tested with different calibration sample configurations to compare accuracy and performance.

## Project Overview

This repository contains:
- **Quantized Models**: GPTQ and AWQ W4A16 quantized versions of Qwen 3-4B
- **Performance Benchmarks**: Throughput, latency, and memory usage analysis using vLLM
- **Accuracy Benchmarks**: Quality evaluation using MMLU, Hellaswag, ARC Challenge/Easy
- **Comprehensive Analysis**: Detailed comparison reports with recommendations

## Key Findings

**Performance (vLLM Inference)**:
- Both GPTQ and AWQ deliver **~50-53% throughput improvement** over baseline
- Performance is virtually identical between GPTQ and AWQ (~1% difference)
- Memory capacity nearly doubles (+89% KV cache)
- See: `quantization/log/llm-compressor/ANALYSIS.md`

**Accuracy (lm-eval)**:
- **GPTQ-512** achieves 65.61% average (closest to baseline 65.85%)
- **GPTQ outperforms AWQ** by ~2-3% on average
- 512 calibration samples > 215 samples for both methods
- See: `benchmark/benchmark_logs/ANALYSIS.md`

## Directory Structure

```
Quantization_Demo/
├── quantization/              # Quantized models and serving configs
│   ├── gptq/                 # GPTQ quantization
│   │   ├── Qwen3-4B-W4A16-GPTQ/              # Standard config
│   │   ├── Qwen3-4B-W4A16-GPTQSAMPLES512/    # 512 calibration samples
│   │   ├── main.py           # Quantization script
│   │   └── README.md         # GPTQ setup guide
│   ├── awq/                  # AWQ quantization
│   │   ├── Qwen3-4B-W4A16-awq/               # Standard config
│   │   ├── Qwen3-4B-W4A16-awq-SAMPLES512/    # 512 calibration samples
│   │   ├── main.py           # Quantization script
│   │   └── README.md         # AWQ setup guide
│   └── log/
│       └── llm-compressor/   # Performance benchmark results
│           └── ANALYSIS.md   # Performance analysis report
│
├── benchmark/                # Accuracy evaluation
│   ├── main.py              # lm-eval benchmark script
│   ├── README.md            # Benchmark guide
│   └── benchmark_logs/
│       ├── ANALYSIS.md      # Accuracy analysis report
│       ├── raw/             # Full benchmark results (JSON)
│       ├── summary/         # Summary results (JSON)
│       └── csv/             # Tabular reports (CSV)
│
├── ShareGPT_V3_unfiltered_cleaned_split.json  # Benchmark dataset
├── pyproject.toml           # Project dependencies
└── README.md               # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
uv sync
# or
pip install -r requirements.txt
```

### 2. Serve a Quantized Model

**GPTQ (W4A16, 512 samples)**:
```bash
vllm serve quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512 \
  --served-model-name Qwen3-4B-GPTQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```

**AWQ (W4A16, 512 samples)**:
```bash
vllm serve quantization/awq/Qwen3-4B-W4A16-awq-SAMPLES512 \
  --served-model-name Qwen3-4B-AWQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```

### 3. Run Performance Benchmark

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen3-4B-GPTQ \
  --tokenizer quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 100 \
  --result-dir "./quantization/log/llm-compressor" \
  --save-result
```

### 4. Run Accuracy Benchmark

Edit `benchmark/main.py` to set model path:
```python
path = "/path/to/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512"
model_name = "Qwen3-4B-W4A16-GPTQSAMPLES512"
```

Run evaluation:
```bash
cd benchmark
uv run main.py
```

## Benchmark Results Summary

### Performance (vLLM 0.11.2)

| Model | Throughput | TTFT | TPOT | KV Cache |
|-------|-----------|------|------|----------|
| **Baseline (BF16)** | 1,824 tok/s | 2,342 ms | 40.90 ms | ~35,648 tokens |
| **GPTQ W4A16** | 2,796 tok/s (+53%) | 2,586 ms | 34.29 ms (-16%) | ~67,216 tokens (+89%) |
| **AWQ W4A16** | 2,763 tok/s (+51%) | 2,622 ms | 34.64 ms (-15%) | ~67,000 tokens (+88%) |

### Accuracy (lm-eval)

| Model | ARC Challenge | ARC Easy | Hellaswag | MMLU | Average |
|-------|---------------|----------|-----------|------|---------|
| **Baseline** | 52.0% | 80.0% | 60.0% | 71.40% | **65.85%** |
| **GPTQ-512** | 58.0% | 70.0% | 64.0% | 70.42% | **65.61%** |
| **GPTQ-215** | 54.0% | 74.0% | 62.0% | 69.93% | 64.98% |
| **AWQ-512** | 52.0% | 74.0% | 56.0% | 69.51% | 62.88% |
| **AWQ-215** | 54.0% | 70.0% | 56.0% | 69.16% | 62.29% |

## Recommendations

**For Production Deployment**:
- **GPTQ-512**: Best accuracy-performance balance (65.61% accuracy, +53% throughput)
- Choice between GPTQ/AWQ based on ecosystem compatibility rather than performance

**For Resource-Constrained Scenarios**:
- **GPTQ-215**: Good accuracy (64.98%) with fewer calibration samples

**For Research/Baseline**:
- Keep baseline model if resources allow

## Documentation

- **Performance Analysis**: `quantization/log/llm-compressor/ANALYSIS.md`
- **Accuracy Analysis**: `benchmark/benchmark_logs/ANALYSIS.md`
- **GPTQ Guide**: `quantization/gptq/README.md`
- **AWQ Guide**: `quantization/awq/README.md`
- **Benchmark Guide**: `benchmark/README.md`

## Technical Details

- **Base Model**: Qwen/Qwen3-4B
- **Quantization**: W4A16 (4-bit weights, 16-bit activations)
- **Calibration Samples**: 215 (standard) and 512 (limited)
- **Calibration Dataset**: ShareGPT_V3
- **Inference Engine**: vLLM 0.11.2 with FlashAttention
- **Evaluation Framework**: lm-evaluation-harness

## Configuration Notes

- **GPU Memory Utilization**: 0.8 for all benchmarks (changed from 0.9)
- **Chunked Prefill**: Enabled with `max_num_batched_tokens=2048`
- **Max Model Length**: 8192 tokens
- **Evaluation Tasks**: MMLU, Hellaswag, ARC Easy, ARC Challenge
- **Evaluation Config**: num_fewshot=0, limit=50

---

**Last Updated**: November 27, 2025  
**Environment**: vLLM 0.11.2, lm-eval-harness, Python 3.11
