import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from accelerate import dispatch_model
import torch



# MODEL_ID = "Qwen/Qwen3-4B"

MODEL_ID = "Qwen/Qwen3-14B"
OFFLOAD_DIR = "./offload"  
# NUM_CALIBRATION_SAMPLES = 512
# MAX_SEQUENCE_LENGTH = 512
SAVE_DIR = MODEL_ID.split("/")[-1] + "-W8A8-FP8-DYNAMIC"

# Ensure internal redispatch uses the offload dir
os.makedirs(OFFLOAD_DIR, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    offload_state_dict=True,
    trust_remote_code=True,
    offload_folder=OFFLOAD_DIR,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code = True
)

model = dispatch_model(
    model,
    device_map=model.hf_device_map,
    offload_dir=OFFLOAD_DIR,
)

recipe = [
    QuantizationModifier(
        targets=["Linear"],
        ignore = ["lm_head"],
        scheme="FP8_DYNAMIC"
    )
]

#Apply algorithms
oneshot(
    model=model,
    recipe=recipe,
)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)