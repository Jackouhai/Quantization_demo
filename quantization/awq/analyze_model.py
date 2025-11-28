from transformers import AutoModelForCausalLM
import sys

model_name = "Qwen/Qwen3-4B"
output_filename = "model_structure.txt"

print(f"Đang tải model: {model_name}...")
try:
    model= AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map = "auto", 
        trust_remote_code = "True"
        )

except OSError:
    print("Load model không thành công")
    sys.exit(1)


print(f"Đã tải xong! Đang ghi cấu trúc vào file '{output_filename}'...")
with open(output_filename, "w", encoding = "utf-8") as f:
    f.write(f"=== CẤU TRÚC MODEL: {model_name} ===\n\n")
    for name,module in model.named_modules():
        if hasattr(module, "weight"):
            line = f"{name}  ->  {module}"
            print(line)
            f.write(line + "\n")