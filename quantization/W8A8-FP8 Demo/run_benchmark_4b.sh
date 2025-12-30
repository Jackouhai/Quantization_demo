#!/bin/bash
set -e

COMMON_ARGS="--tasks gsm8k --num_fewshot 5 --batch_size auto"
GPU_UTIL=0.8
MAX_LEN=4096

echo "======================================================="
echo "BẮT ĐẦU QUÁ TRÌNH BENCHMARK SO SÁNH (A/B TESTING)"
echo "======================================================="

QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/W8A8-FP8 Demo/Qwen3-4B-W8A8-FP8-DYNAMIC"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k-QuantizeModel

echo ">>> Xong phần 1. Kết quả lưu tại ./results/gsm8k-Quantized"
echo "-------------------------------------------------------"
   
MODEL=Qwen/Qwen3-4B

echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k-BaseModel

echo "======================================================="
echo "HOÀN TẤT! HÃY KIỂM TRA THƯ MỤC ./benchmark ĐỂ SO SÁNH"
echo "======================================================="