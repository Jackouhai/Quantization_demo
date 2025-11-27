from huggingface_hub import constants
import os

print("--- KIỂM TRA VỊ TRÍ MODEL ---")
print(f"Thư mục Cache mặc định: {constants.HF_HOME}")


custom_path = os.getenv("HF_HOME")
if custom_path:
    print(f"Đang dùng đường dẫn tùy chỉnh (HF_HOME): {custom_path}")
else:
    print("Đang dùng đường dẫn mặc định.")