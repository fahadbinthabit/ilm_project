# ilm_project/02_scripts/step_06_run_model.py

from unsloth import FastLanguageModel
import torch

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="insurance_llm",  # your saved folder
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Optional: move to GPU
if torch.cuda.is_available():
    model.to("cuda")

# Simple chat loop
print("🦙 Insurance LLM is ready. Type 'exit' to quit.\n")
while True:
    prompt = input("User: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"ILM: {reply}\n")
