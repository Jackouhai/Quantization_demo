from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
import json
import os 
import random
import numpy as np
import torch
import pandas as pd  
from datetime import datetime

# --- CẤU HÌNH ---
path = "/home/bocchi/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
model_name = os.path.basename(path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SEED = 42  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---KHỞI TẠO MODEL---
print(f"Đang tải model: {model_name}...")
model = VLLM(
    pretrained=path,
    dtype="auto",             
    trust_remote_code=True,
    add_bos_token=True,
    max_model_len=8192,
    gpu_memory_utilization=0.90, 
)

# ---CHẠY ĐÁNH GIÁ---
print("Đang chạy benchmark... (Vui lòng chờ)")
results = simple_evaluate(
    model=model,
    tasks=["mmlu", "hellaswag", "arc_easy", "arc_challenge"],
    num_fewshot=0,     
    batch_size="auto",
    limit=50,           
)

# ---XỬ LÝ KẾT QUẢ---
summary_data = {}

for task, metrics in results["results"].items():
    # Logic lọc metric chuẩn
    if "acc_norm,none" in metrics:
        score = metrics["acc_norm,none"]
        metric_type = "acc_norm"
    elif "acc,none" in metrics:
        score = metrics["acc,none"]
        metric_type = "acc"
    else:
        score = 0.0
        metric_type = "unknown"
    
    summary_data[task] = {
        "Score (%)": round(score * 100, 2),
        "Metric": metric_type
    }

# --- LƯU FILE ---

# A. Full Raw Data
os.makedirs("./benchmark_logs/raw", exist_ok=True) 
full_log_file = f"./benchmark_logs/raw/{model_name}_FULL_{timestamp}.json"
with open(full_log_file, "w") as f:
    json.dump(results, f, indent=2)

# B. Summary JSON
os.makedirs("./benchmark_logs/summary", exist_ok=True)
# SỬA LỖI: Đảm bảo tên thư mục khớp với lệnh makedirs
summary_file = f"./benchmark_logs/summary/{model_name}_SUMMARY_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump(summary_data, f, indent=2)

# C. CSV Report
os.makedirs("./benchmark_logs/csv", exist_ok=True)
df = pd.DataFrame.from_dict(summary_data, orient='index')
# SỬA LỖI: Thêm đường dẫn thư mục vào tên file
csv_file = f"./benchmark_logs/csv/{model_name}_REPORT_{timestamp}.csv"
df.to_csv(csv_file)



print("\n" + "="*40)
print(f"KẾT QUẢ ĐÁNH GIÁ: {model_name}")
print("="*40)
print(f"{'TASK':<30} | {'SCORE (%)':<10} | {'METRIC'}")
print("-" * 55)

for task, data in summary_data.items():
    if "mmlu_" in task and task != "mmlu": 
        continue 
        
    print(f"{task:<30} | {data['Score (%)']:<10} | {data['Metric']}")

print("-" * 55)
if "mmlu" in summary_data:
    print(f"MMLU Average: {summary_data['mmlu']['Score (%)']}%")
print("="*40)
print(f"Đã lưu Log gốc tại: {full_log_file}")
print(f"Đã lưu CSV tại:     {csv_file}")