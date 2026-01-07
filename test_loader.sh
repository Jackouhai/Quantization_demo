#!/bin/bash

# --- CẤU HÌNH ---
GPU_UTIL=0.7
MAX_LEN=4096
WAIT_TIME=20
OUTPUT_FILE="vram_results.txt"

# --- DANH SÁCH MODEL ---
MODELS=(
    "Qwen/Qwen3-4B"
    "/home/bocchi/Work/Quantization_Demo/quantization/W8A8-FP8 Demo/Qwen3-4B-W8A8-FP8-DYNAMIC"
    "/home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq"
    "/home/bocchi/Work/Quantization_Demo/quantization/awq/Qwen3-4B-W4A16-awq-SAMPLES512"
    "/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQ"
    "/home/bocchi/Work/Quantization_Demo/quantization/gptq/Qwen3-4B-W4A16-GPTQSAMPLES512"
)

# --- XÓA FILE KẾT QUẢ CŨ ---
echo "Model Name : VRAM Usage (MiB)" > $OUTPUT_FILE
echo "-----------------------------" >> $OUTPUT_FILE

echo "======================================================="
echo "BẮT ĐẦU ĐO VRAM HÀNG LOẠT"
echo "Kết quả sẽ lưu vào: $OUTPUT_FILE"
echo "======================================================="

# === INITIAL CLEANUP: Kill tất cả vllm processes cũ ===
echo ">>> Cleanup tiến trình vllm cũ (nếu có)..."
sudo pkill -9 -f "vllm serve" 2>/dev/null || true
sudo pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sudo pkill -9 -f "ray::VLLM" 2>/dev/null || true
sudo pkill -9 -f "vllm" 2>/dev/null || true
sleep 3

echo ">>> Kiểm tra GPU ban đầu..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0
echo ""

for MODEL in "${MODELS[@]}"; do
    # --- CẮT TÊN NGẮN TẠI ĐÂY ---
    SHORT_NAME=$(basename "$MODEL")

    echo ""
    echo "------------------------------------------------"
    echo ">>> [1/3] Đang khởi động: $SHORT_NAME"  # 
    
    vllm serve "$MODEL" \
        --dtype auto \
        --gpu-memory-utilization $GPU_UTIL \
        --max-model-len $MAX_LEN \
        --port 8000 > /dev/null 2>&1 &
    
    PID=$!

    echo ">>> [2/3] Đang chờ $WAIT_TIME giây..."
    sleep $WAIT_TIME

    if ps -p $PID > /dev/null; then
        VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
        
        echo ">>> SUCCESS. VRAM: $VRAM_USED MiB"
        
        # Ghi tên ngắn vào file báo cáo cho đẹp
        echo "$SHORT_NAME: $VRAM_USED MiB" >> $OUTPUT_FILE
    else
        echo ">>> CRASH/OOM!"
        echo "$SHORT_NAME: CRASH/OOM" >> $OUTPUT_FILE
    fi

    # ===== LUÔN KILL PROCESSES (dù SUCCESS hay CRASH) =====
    echo ">>> [3/3] Đang cleanup và chờ GPU giải phóng..."
    
    # Kill process group của PID chính
    sudo pkill -9 -P $PID 2>/dev/null || true
    sudo kill -9 $PID 2>/dev/null || true
    
    # Force kill TẤT CẢ vllm processes (by pattern matching) với sudo
    sudo pkill -9 -f "vllm serve" 2>/dev/null || true
    sudo pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sudo pkill -9 -f "ray::VLLM" 2>/dev/null || true
    sudo pkill -9 -f "vllm" 2>/dev/null || true
    
    # Kill python processes đang chiếm GPU nếu cần
    sudo pkill -9 -f "python.*vllm" 2>/dev/null || true
    
    sleep 3
    
    # Đợi GPU free với timeout
    MAX_WAIT=60  # tăng lên 60s
    elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        CURRENT_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
        if [ "$CURRENT_VRAM" -lt 500 ]; then
            echo ">>> GPU đã free (${CURRENT_VRAM} MiB). Tiếp tục..."
            break
        fi
        echo ">>> Đợi GPU free... (${CURRENT_VRAM} MiB, ${elapsed}s/${MAX_WAIT}s)"
        
        # Thêm aggressive cleanup mỗi 9s
        if [ $((elapsed % 9)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            echo ">>> Re-cleanup vllm processes..."
            sudo pkill -9 -f "vllm" 2>/dev/null || true
            sudo pkill -9 -f "ray::" 2>/dev/null || true
            sudo pkill -9 -f "python.*vllm" 2>/dev/null || true
        fi
        
        sleep 3
        elapsed=$((elapsed + 3))
    done
    
    # Nếu timeout mà vẫn không free, cảnh báo
    FINAL_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
    if [ "$FINAL_VRAM" -ge 500 ]; then
        echo ">>> CẢNH BÁO: GPU chưa free hoàn toàn (${FINAL_VRAM} MiB). Vẫn tiếp tục..."
    fi
    
    sleep 2
done

echo ""
echo "======================================================="
echo "HOÀN TẤT. KIỂM TRA FILE $OUTPUT_FILE"
cat $OUTPUT_FILE