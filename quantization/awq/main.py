from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation



# ============ DATASET CONFIGURATION ============
# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# ============ MODEL CONFIGURATION ============
#MODEL INFO + LOAD MODEL
MODEL_ID = "Qwen/Qwen3-4B"
SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-awq" + "-SAMPLES" + str(NUM_CALIBRATION_SAMPLES)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", 
    device_map="auto", 
    trust_remote_code="True",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
recipe = [
    AWQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ============ TEST GENERATION ============
print("\n" + "="*50)
print("TESTING QUANTIZED MODEL GENERATION")
print("="*50)

dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model. generate(input_ids, max_new_tokens=100)
print("\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("="*50 + "\n")

# ============ SAVE MODEL ============
print(f"Saving model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Model saved successfully to '{SAVE_DIR}'")
print("\nTo load in vLLM, use:")
print(f'  from vllm import LLM')
print(f'  model = LLM("{SAVE_DIR}", trust_remote_code=True)')