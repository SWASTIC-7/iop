!pip install pdfplumber
!pip install -q bitsandbytes accelerate transformers

import torch, gc

gc.collect()
torch.cuda.empty_cache()
import os

for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        print(os.path.join(root, file))
import pdfplumber

file_path = "/kaggle/input/datasets/naman223/ex3700/EX3700.pdf"

full_text = ""

with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

print("Total characters:", len(full_text))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-3B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config,
    low_cpu_mem_usage=True
)


print("Model loaded successfully")

def chunk_text_by_tokens(text, tokenizer, max_tokens=1200):
    tokens = tokenizer(text)["input_ids"]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        
    return chunks

chunks = chunk_text_by_tokens(full_text, tokenizer, max_tokens=1200)

print("Total chunks:", len(chunks))
import json

all_results = []

for idx, chunk in enumerate(chunks):
    
    print(f"\nProcessing chunk {idx+1}/{len(chunks)}")
    
    prompt = f"""
Extract device information.

Respond with ONLY a valid JSON object.
If information is not present in this document chunk, return null or empty list.
Do not infer or assume missing information.

Do NOT repeat the instructions.
Do NOT include explanations.
Do NOT include text outside JSON.

JSON format:
{{
  "vendor": string or null,
  "model": string or null,
  "hardware_components": list,
  "software_components": list,
  "communication_protocols": list,
  "functional_capabilities": list
}}

Document:
{chunk}
"""


    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    all_results.append(result)
    
    # Clean memory after each chunk
    del inputs
    del outputs
    torch.cuda.empty_cache()
    gc.collect()

for i, r in enumerate(all_results):
    print(f"\nChunk {i+1} Output:\n", r)

import json
import re

parsed_results = []

def extract_valid_json(text):
    candidates = re.findall(r"\{[^{}]*\}", text, re.DOTALL)

    for c in reversed(candidates):  # check from last to first
        try:
            obj = json.loads(c)
            return obj
        except:
            continue

    return None

for r in all_results:
    obj = extract_valid_json(r)
    if obj:
        parsed_results.append(obj)

print("Valid parsed chunks:", len(parsed_results))



# Step 2: Initialize final merged structure
final_output = {
    "vendor": None,
    "model": None,
    "hardware_components": set(),
    "software_components": set(),
    "communication_protocols": set(),
    "functional_capabilities": set()
}


# Step 3: Merge intelligently (FIX 3 included)
for item in parsed_results:
    
    # FIX 3 — Only set vendor if not already set
    if item.get("vendor") and not final_output["vendor"]:
        final_output["vendor"] = item["vendor"]
        
    # FIX 3 — Only set model if not already set
    if item.get("model") and not final_output["model"]:
        final_output["model"] = item["model"]
    
    
    # Merge list fields safely
    for key in [
        "hardware_components",
        "software_components",
        "communication_protocols",
        "functional_capabilities"
    ]:
        
        value = item.get(key)
        
        if isinstance(value, list):
            final_output[key].update(value)


# Step 4: Convert sets → lists
for key in final_output:
    if isinstance(final_output[key], set):
        final_output[key] = list(final_output[key])


# Step 5: Replace None with empty string (optional)
if final_output["vendor"] is None:
    final_output["vendor"] = ""

if final_output["model"] is None:
    final_output["model"] = ""


# Final Output
print("\nFINAL MERGED OUTPUT:\n")
print(json.dumps(final_output, indent=2))