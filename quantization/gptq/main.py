from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers. quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# ============ DATASET CONFIGURATION ============
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

# NUM_CALIBRATION_SAMPLES = 256
# MAX_SEQUENCE_LENGTH = 2048 

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

print(f"Loading dataset: {DATASET_ID}")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)
print(f"Loaded {len(ds)} calibration samples\n")


# ============ MODEL INFO + LOAD MODEL ============
MODEL_ID = "Qwen/Qwen3-4B"
SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-GPTQ" + "SAMPLES" + str(NUM_CALIBRATION_SAMPLES)

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Model loaded successfully\n")

# Preprocess function
def preprocess(example):
    return {
        "text": tokenizer. apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

# Tokenize function
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds. column_names)

# ============ GPTQ QUANTIZATION RECIPE ============
print("Configuring GPTQ quantization...")
recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W4A16",          
        actorder= "static"
    ),
]
print("Recipe configured\n")

# ============ APPLY QUANTIZATION ============
print("="*70)
print("Starting GPTQ quantization...")
print("="*70)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\nQuantization completed!\n")

# ============ TEST GENERATION ============
print("="*70)
print("Testing quantized model generation...")
print("="*70)

dispatch_for_generation(model)

test_prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Explain machine learning in one sentence:",
]

for prompt in test_prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=50)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generated}")
    print("-"*70)

# ============ SAVE MODEL ============
print(f"\nSaving model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n{'='*70}")
print(f"GPTQ Model saved successfully to '{SAVE_DIR}'")
print(f"{'='*70}")

# ============ MODEL INFO ============
import os
model_size = sum(
    os.path.getsize(os.path.join(SAVE_DIR, f))
    for f in os.listdir(SAVE_DIR)
    if f.endswith('. safetensors')
) / (1024**3)

print(f"\nModel Statistics:")
print(f"  - Quantization method: GPTQ")
print(f"  - Scheme: W4A16")
print(f"  - Group size: 128")
print(f"  - Model size: {model_size:.2f} GB")
print(f"  - Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"  - Max sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"\nTo load in vLLM:")
print(f"  vllm serve {SAVE_DIR} --trust-remote-code --port 8001")