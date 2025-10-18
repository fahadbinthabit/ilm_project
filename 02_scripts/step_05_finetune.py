# ilm_project/02_scripts/step_05_finetune.py

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

# Set the working directory to the project root
# This ensures the script can find your dataset file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
print(f"Working directory set to: {os.getcwd()}")


# 1. Load the base model and tokenizer
max_seq_length = 2048
dtype = None # Auto-detect
load_in_4bit = True # Use 4-bit quantization to save VRAM

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Model loaded successfully.")

# 2. Configure the model for fine-tuning with LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# 3. Load and format your training data
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["context"]
    outputs      = examples["response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

print("Loading and formatting dataset...")
dataset = load_dataset("json", data_files="training_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
print("Dataset ready.")

# 4. Configure and run the fine-tuning process
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # IMPORTANT: Start with a small number to test your setup
        max_steps = 20,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting the fine-tuning process...")
trainer.train()
print("Fine-tuning complete!")

# 5. Save your new, specialized model locally
print("Saving fine-tuned model...")
model.save_pretrained("insurance_llm")
tokenizer.save_pretrained("insurance_llm")
print("Model saved to the 'insurance_llm' folder.")