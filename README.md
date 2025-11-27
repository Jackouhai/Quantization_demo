# Quantization Demo: GPTQ & AWQ Benchmarks for Qwen 3-4B

This project benchmarks and implements quantization for the Qwen 3-4B model using two methods: **GPTQ** and **AWQ**. Each method is tested with different calibration sample configurations to compare accuracy and performance.

## Project Overview

This repository contains:
- **Quantized Models**: GPTQ and AWQ W4A16 quantized versions of Qwen 3-4B
- **Performance Benchmarks**: Throughput, latency, and memory usage analysis using vLLM
- **Accuracy Benchmarks**: Quality evaluation using MMLU, Hellaswag, ARC Challenge/Easy
- **Comprehensive Analysis**: Detailed comparison reports with recommendations

## Pre-quantized Models

The quantized models are available on Hugging Face Hub:

| Model | Hugging Face Link | Size | Calibration Samples |
|-------|------------------|------|---------------------|
| **GPTQ W4A16 (512 samples)** | [🤗 Jackouhai/Qwen3-4B-W4A16-GPTQ-512](https://huggingface.co/Jackouhai/Qwen3-4B-W4A16-GPTQ-512) | ~2.3GB | 512 |
| **GPTQ W4A16 (215 samples)** | [🤗 Jackouhai/Qwen3-4B-W4A16-GPTQ](https://huggingface.co/Jackouhai/Qwen3-4B-W4A16-AWQ-215) | ~2.3GB | 215 |
| **AWQ W4A16 (512 samples)** | [🤗 Jackouhai/Qwen3-4B-W4A16-AWQ-512](https://huggingface.co/Jackouhai/Qwen3-4B-W4A16-AWQ-512) | ~2.3GB | 512 |
| **AWQ W4A16 (215 samples)** | [🤗 Jackouhai/Qwen3-4B-W4A16-AWQ](https://huggingface.co/Jackouhai/Qwen3-4B-W4A16-AWQ-215) | ~2.3GB | 215 |

**Quick Use**:
```bash
# Download and serve directly from Hugging Face
vllm serve username/Qwen3-4B-W4A16-GPTQ-512 \
  --served-model-name Qwen3-4B-GPTQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```


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

### Option A: Use Pre-quantized Models from Hugging Face

```bash
# 1. Install dependencies
uv sync  # or: pip install -r requirements.txt

# 2. Serve directly from Hugging Face (recommended)
vllm serve username/Qwen3-4B-W4A16-GPTQ-512 \
  --served-model-name Qwen3-4B-GPTQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```

### Option B: Quantize Models Yourself

See the [Quantization Guide](#how-to-quantize) section below for detailed instructions.

### Option C: Use Locally Quantized Models

If you already have quantized models in the `quantization/` directory:

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

## How to Quantize

If you want to create quantized models yourself instead of using pre-quantized versions:

### GPTQ Quantization

```bash
# 1. Install llm-compressor
pip install llmcompressor

# 2. Prepare calibration dataset (ShareGPT_V3)
# Download from: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

# 3. Run GPTQ quantization
cd quantization/gptq
python main.py
```

The `main.py` script will:
- Load Qwen/Qwen3-4B from Hugging Face
- Apply GPTQ W4A16 quantization with 512 calibration samples
- Save to `Qwen3-4B-W4A16-GPTQSAMPLES512/`

**Configuration options** in `main.py`:
```python
# Adjust these parameters as needed
num_samples = 512  # or 215 for faster calibration
scheme = "W4A16"   # 4-bit weights, 16-bit activations
```

### AWQ Quantization

```bash
# 1. Install llm-compressor
pip install llmcompressor

# 2. Run AWQ quantization
cd quantization/awq
python main.py
```

The `main.py` script will:
- Load Qwen/Qwen3-4B from Hugging Face
- Apply AWQ W4A16 quantization with 512 calibration samples
- Save to `Qwen3-4B-W4A16-awq-SAMPLES512/`

**Expected Output**:
- Model size: ~2.3GB (down from ~8GB for BF16)
- Quantization time: ~10-30 minutes depending on GPU
- Output directory contains: `model.safetensors`, `config.json`, tokenizer files

### Upload to Hugging Face (Optional)

```bash
# Install huggingface-hub
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login

# Upload your quantized model
cd quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512
huggingface-cli upload username/Qwen3-4B-W4A16-GPTQ-512 . --repo-type model
```

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
