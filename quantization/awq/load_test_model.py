import os
os.environ["VLLM_USE_V1"] = "1"  

from vllm import LLM, SamplingParams  

print("Loading quantized model...")
llm = LLM(
    "Qwen3-4B-W4A16-awq",
    trust_remote_code=True,
)

#Dùng SamplingParams object
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,  
)

# Test prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum computing in simple terms:",
]

print("\n" + "="*60)
print("TESTING INFERENCE")
print("="*60)

#Pass SamplingParams object
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-"*60)

print("\n Model loaded and working correctly!")