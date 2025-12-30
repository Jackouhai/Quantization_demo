#!/bin/bash
set -e

COMMON_ARGS="--tasks gsm8k --num_fewshot 5 --batch_size 32"
GPU_UTIL=0.9
MAX_LEN=2048
cpu_offload_gb=3
max_num_batched_tokens=4096
enforce_eager=True

echo "======================================================="
echo "BẮT ĐẦU QUÁ TRÌNH BENCHMARK SO SÁNH (A/B TESTING)"
echo "======================================================="

MODEL=Qwen/Qwen3-8B


echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN,cpu_offload_gb=$cpu_offload_gb,max_num_batched_tokens=$max_num_batched_tokens,enforce_eager=$enforce_eager" \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k-BaseModel

echo ">>> Xong phần 1. Kết quả lưu tại ./benchmark/gsm8k-BaseModel"
echo "-------------------------------------------------------"

QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/W8A8-FP8 Demo/Qwen3-8B-W8A8-FP8-DYNAMIC"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --output_path ./benchmark/gsm8k-QuantizeModel


   
echo "======================================================="
echo "HOÀN TẤT! HÃY KIỂM TRA THƯ MỤC ./benchmark ĐỂ SO SÁNH"
echo "======================================================="