from huggingface_hub import HfApi

api = HfApi()


username = "Jackouhai" 
model_name = "Qwen3-4B-W8A8-FP8-DYNAMIC"
repo_id = f"{username}/{model_name}"
folder_path = "/home/bocchi/Work/Quantization_Demo/quantization/W8A8-FP8 Demo/Qwen3-4B-W8A8-FP8-DYNAMIC"


print(f"Creating repo: {repo_id}...")
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)


print(f"Uploading folder: {folder_path}...")
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)

print("Done!")