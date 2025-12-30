import os

# Prefer a more robust attention backend for experimental FP8
# os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
# os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(
    model="./Qwen3-4B-W8A8-FP8-DYNAMIC",
    # tokenizer="./Qwen3-4B-W8A8-FP8-DYNAMIC",
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    max_model_len=2048,
    # enforce_eager=True,
    # enable_chunked_prefill=False,
    # quantization="compressed-tensors",
    # dtype="auto",
    # disable_log_stats=True,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048,
)

outputs = llm.generate(["Hà Nội là gì"], sampling_params)
print(outputs[0].outputs[0].text)