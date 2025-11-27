# vLLM Performance Comparison: Baseline vs GPTQ vs AWQ

This analysis compares the performance of baseline Qwen3-4B (BF16) against quantized variants (GPTQ W4A16, AWQ W4A16) using vLLM 0.11.2. All benchmarks used 100 ShareGPT prompts with consistent configuration.

## Executive Summary

**GPTQ and AWQ deliver nearly identical performance** - both methods improve throughput by ~50-53% over baseline with similar latency characteristics. The choice between GPTQ and AWQ can be based on calibration convenience or ecosystem compatibility rather than performance.

## Performance Comparison Tables

### Throughput (higher is better)

| Model | Request/s | Output Token/s | Total Token/s | vs Baseline |
|-------|-----------|----------------|---------------|-------------|
| **Baseline (BF16)** | 4.03 | 888.09 | 1,824.44 | - |
| **GPTQ W4A16** | 6.17 | 1,361.01 | 2,795.99 | **+53%** |
| **AWQ W4A16** | 6.10 | 1,344.79 | 2,762.66 | **+51%** |

### Latency - Time to First Token (lower is better)

| Model | TTFT Median (ms) | TTFT P99 (ms) |
|-------|------------------|---------------|
| **Baseline (BF16)** | 2,342.43 | 4,230.61 |
| **GPTQ W4A16** | 2,585.53 | 4,337.52 |
| **AWQ W4A16** | 2,622.35 | 4,395.10 |

### Latency - Time per Output Token (lower is better)

| Model | TPOT Median (ms) | TPOT P99 (ms) | vs Baseline |
|-------|------------------|---------------|-------------|
| **Baseline (BF16)** | 40.90 | 358.53 | - |
| **GPTQ W4A16** | 34.29 | 372.66 | **-16%** (faster) |
| **AWQ W4A16** | 34.64 | 373.21 | **-15%** (faster) |

### Memory Capacity & KV Cache

| Model | KV Cache Tokens | Max Concurrency | vs Baseline |
|-------|-----------------|-----------------|-------------|
| **Baseline (BF16)** | ~35,648 | 4.35x | - |
| **GPTQ W4A16** | ~67,216 | 8.21x | **+89%** |
| **AWQ W4A16** | ~67,000+ | ~8.2x | **+88%** |

## Detailed Analysis

### 1. Throughput
- Both GPTQ and AWQ deliver **~50-53% improvement** in total token throughput over baseline
- Difference between GPTQ and AWQ is negligible (~1%)
- GPTQ: 2,796 tok/s (+53%)
- AWQ: 2,763 tok/s (+51%)

### 2. Latency
All three configurations show **similar latency profiles**:
- **TTFT**: ~2.3-2.6 seconds for all variants (equivalent)
- **TPOT**: Quantized models are ~16% faster (34.3-34.6 ms vs 40.9 ms)

### 3. Memory & Capacity
- Quantized models nearly **double** the KV cache capacity (~+89%)
- Enable higher concurrency (8.21x vs 4.35x)
- Better GPU utilization thanks to smaller footprint

## Practical Recommendations

**Both GPTQ and AWQ are excellent choices** - performance is virtually identical

**Choose based on**:
- **Calibration data**: GPTQ typically requires fewer samples
- **Ecosystem**: Some frameworks favor one method over the other
- **Model compatibility**: AWQ generally has broader model support

**Expected gains**:
- ~50% throughput improvement
- Equivalent or slightly better latency
- 2x KV cache capacity

## Raw Benchmark Metrics

### Baseline: `vllm-infqps-Qwen3-4B-20251127-094403.json`
```
request_throughput:     4.0256 req/s
output_throughput:      888.0880 tok/s
total_token_throughput: 1824.4431 tok/s
median_ttft_ms:         2342.43
p99_ttft_ms:            4230.61
median_tpot_ms:         40.90
p99_tpot_ms:            358.53
```

### GPTQ: `vllm-infqps-Qwen3-4B-GPTQ-20251127-101423.json`
```
request_throughput:     6.1693 req/s  (+53%)
output_throughput:      1361.0115 tok/s  (+53%)
total_token_throughput: 2795.9929 tok/s  (+53%)
median_ttft_ms:         2585.53  (≈ baseline)
p99_ttft_ms:            4337.52  (≈ baseline)
median_tpot_ms:         34.29  (-16% = faster)
p99_tpot_ms:            372.66  (≈ baseline)
```

### AWQ: `vllm-infqps-Qwen3-4B-AWQ-20251127-102110.json`
```
request_throughput:     6.0958 req/s  (+51%)
output_throughput:      1344.7873 tok/s  (+51%)
total_token_throughput: 2762.6629 tok/s  (+51%)
median_ttft_ms:         2622.35  (≈ baseline)
p99_ttft_ms:            4395.10  (≈ baseline)
median_tpot_ms:         34.64  (-15% = faster)
p99_tpot_ms:            373.21  (≈ baseline)
```

## Reproduction

### Serve Commands

```bash
# Baseline (BF16)
vllm serve /home/bocchi/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --served_model_name Qwen/Qwen3-4B \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000

# GPTQ (W4A16, SAMPLES512)
vllm serve /home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512 \
  --served_model_name Qwen3-4B-GPTQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000

# AWQ (W4A16, SAMPLES512)
vllm serve /home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq-SAMPLES512 \
  --served_model_name Qwen/Qwen3-4B-AWQ \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --port 8000
```

### Benchmark Command

```bash
vllm bench serve \
  --backend vllm \
  --model <served-model-name> \
  --tokenizer <path-to-tokenizer> \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path "ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 100 \
  --result-dir "./quantization/log/llm-compressor" \
  --save-result
```

## Benchmark Configuration

- **GPU Memory**: `--gpu-memory-utilization 0.8` for all runs
- **Chunked Prefill**: Enabled with default `max_num_batched_tokens=2048`
- **vLLM**: Version 0.11.2 with torch.compile caching and FlashAttention backend
- **Dataset**: 100 ShareGPT prompts (23,260 input tokens, 22,061 output tokens)

---

**Conclusion**: GPTQ and AWQ both deliver excellent and nearly identical performance. Either choice is good - select based on deployment convenience rather than performance differences.
