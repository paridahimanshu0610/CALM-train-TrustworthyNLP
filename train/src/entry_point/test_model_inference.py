# import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM

# model_name = "ChanceFocus/finma-7b-full"

# # Select device
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Using device:", device)

# tokenizer = LlamaTokenizer.from_pretrained(model_name)

# model = LlamaForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if device == "mps" else torch.float32,
#     device_map=None  # We will move it manually
# )

# # Move model to device
# model = model.to(device)

# # Test inference
# prompt = "Explain what credit scoring means in simple terms."
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=80,
#         do_sample=False
#     )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# # Choose device
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Using device:", device)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model
# # NOTE: remove 4-bit / bitsandbytes settings for Mac
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float16 if device == "mps" else torch.float32,
#     device_map=None,  # we'll move model manually
#     low_cpu_mem_usage=False,  # avoids auto quantization
# )

# # Move model to device
# model = model.to(device)

# # Test prompt
# prompt = "Explain briefly: Why is the sky blue?"

# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# # Generate output
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,  # deterministic / greedy decoding
#     )

# # Decode and print
# print(tokenizer.decode(output[0], skip_special_tokens=True))

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "hirundo-io/DeepSeek-R1-Distill-Llama-8B-Debiased"

# # Choose device
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Using device:", device)

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model (FP16 for MPS, FP32 for CPU)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float16 if device == "mps" else torch.float32,
#     device_map=None,              # move manually
#     low_cpu_mem_usage=False       # avoid auto quantization
# )

# # Move model to device
# model = model.to(device)

# # Test prompt
# prompt = "Only answer with either 'yes' or 'no' to the question. Do not provide any additional information. Question: Is elephant a mammal?"

# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# # Generate output (greedy for speed)
# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=3,
#         do_sample=False
#     )

# # Decode and print
# print(tokenizer.decode(output[0], skip_special_tokens=True))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TheFinAI/Fin-o1-8B"

max_new_tokens = 3
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=3
)

# Choose device: MPS for Apple Silicon, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
# FP16 for MPS to reduce memory usage; FP32 for CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    device_map=None,  # manual move
    low_cpu_mem_usage=False  # avoid auto-quantization
)

# Move model to device
model.to(device)
model.eval()  # set model to evaluation mode

# Input prompt
input_text = "What is the result of 3-5? Just tell the answer."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        **generation_config
    )

# Decode and print
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))