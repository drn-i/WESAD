import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

def train_model(model_name, dataset_file, output_dir):
    """Loads and fine-tunes a model on a given dataset."""
    
    print(f"--- Starting Training ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")

    # Check for BF16 support
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not USE_BF16:
        print("Warning: BF16 not supported. Training in FP16/FP32 may be slower or less stable.")

    # --- 1. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare the Dataset ---
    try:
        dataset = load_dataset('json', data_files={'train': dataset_file})['train']
    except Exception as e:
        print(f"Error loading dataset {dataset_file}: {e}")
        return

    def tokenize_function(examples):
        text = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        tokenized_inputs = tokenizer(
            text, padding="max_length", max_length=512, truncation=True
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # --- 3. Set Up Training ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Start small for full-tuning
        bf16=USE_BF16,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # --- 4. Start Training ---
    print("Starting trainer.train()...")
    trainer.train()

    # --- 5. Save the Final Model ---
    print("Training complete. Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama model.")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model to fine-tune.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl training file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    
    args = parser.parse_args()
    train_model(args.model_name, args.dataset_file, args.output_dir)
