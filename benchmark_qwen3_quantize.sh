#!/bin/bash
set -e

COMMON_ARGS="--tasks gsm8k,mmlu,ai2_arc,hellaswag,piqa --num_fewshot 3 --batch_size auto"
GPU_UTIL=0.7
MAX_LEN=4092

export HF_ALLOW_CODE_EVAL="1"
echo "======================================================="
echo "BẮT ĐẦU QUÁ TRÌNH BENCHMARK SO SÁNH (A/B TESTING)"
echo "======================================================="

# --- 1. BASE MODEL ---
MODEL=Qwen/Qwen3-4B

echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN \
  $COMMON_ARGS \
  --confirm_run_unsafe_code \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/BaseModel.json \
  --limit 100

echo ">>> Xong phần 1. Kết quả lưu tại /home/bocchi/Work/Quantization_Demo/benchmark/full/BaseModel.json"
echo "-------------------------------------------------------"


# --- 2. W8A8 DYNAMIC ---
QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/W8A8-FP8 Demo/Qwen3-4B-W8A8-FP8-DYNAMIC"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --confirm_run_unsafe_code \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_W8A8.json \
  --limit 100

echo ">>> Xong phần 2. Kết quả lưu tại /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_W8A8.json"
echo "-------------------------------------------------------"


# --- 3. AWQ 256 ---
QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_AWQ256.json \
  --confirm_run_unsafe_code \
  --limit 100

echo ">>> Xong phần 3. Kết quả lưu tại /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_AWQ256.json"
echo "-------------------------------------------------------"


# --- 4. AWQ 512 ---
MODEL=/home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq-SAMPLES512

echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN \
  $COMMON_ARGS \
  --confirm_run_unsafe_code \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_AWQ512.json \
  --limit 100

echo ">>> Xong phần 4. Kết quả lưu tại /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_AWQ512.json"
echo "-------------------------------------------------------"


# --- 5. GPTQ 256 ---
QUANT_MODEL="/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ"

echo ">>> Đang chạy Benchmark Model Nén: $QUANT_MODEL"

lm_eval \
  --model vllm \
  --model_args "pretrained=$QUANT_MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN" \
  $COMMON_ARGS \
  --confirm_run_unsafe_code \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_GPTQ_256x2048.json \
  --limit 100

echo ">>> Xong phần 5. Kết quả lưu tại /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_GPTQ_256x2048.json"
echo "-------------------------------------------------------"


# --- 6. GPTQ 512 ---
MODEL=/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512

echo ">>> Đang chạy Benchmark Model Nén: $MODEL"

lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_LEN \
  $COMMON_ARGS \
  --confirm_run_unsafe_code \
  --output_path /home/bocchi/Work/Quantization_Demo/benchmark/full/QuantizeModel_GPTQ_512x2048.json \
  --limit 100

echo "======================================================="
echo "HOÀN TẤT! HÃY KIỂM TRA THƯ MỤC /home/bocchi/Work/Quantization_Demo/benchmark/full ĐỂ SO SÁNH"
echo "======================================================="