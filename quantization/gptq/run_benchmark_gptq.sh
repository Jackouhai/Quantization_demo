#!/bin/bash
set -e

COMMON_ARGS="--tasks gsm8k --num_fewshot 5 --batch_size auto"
GPU_UTIL=0.8
MAX_LEN=4096

echo "======================================================="
echo "BẮT ĐẦU QUÁ TRÌNH BENCHMARK SO SÁNH (A/B TESTING)"
echo "======================================================="

QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k/results_benchmark_qwen3-4b-GPTQ-256x2048.json

echo ">>> Xong phần 1. Kết quả lưu tại ./benchmark/gsm8k"
echo "-------------------------------------------------------"
   
MODEL=/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512

echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k/results_benchmark_qwen3-4b-GPTQ-512x2048.json

echo "======================================================="
echo "HOÀN TẤT! HÃY KIỂM TRA THƯ MỤC ./benchmark ĐỂ SO SÁNH"
echo "=======================================================" 