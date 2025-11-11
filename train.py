import os
import torch
import argparse
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model_config(model_name):
    """Returns appropriate training config based on model size."""
    if '1B' in model_name or '1.1B' in model_name:
        return {
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'max_length': 512
        }
    elif '3B' in model_name or '3.1B' in model_name:
        return {
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'max_length': 512
        }
    elif '8B' in model_name or '8.1B' in model_name:
        return {
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'max_length': 512
        }
    else:
        # Default for unknown models
        return {
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'max_length': 512
        }

def train_model(model_name, dataset_file, output_dir, num_epochs=3, learning_rate=2e-5, seed=42):
    """Loads and fine-tunes a model on a given dataset."""
    
    # Set random seed for reproducibility
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    print(f"--- Starting Training ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")

    # Check for BF16 support
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not USE_BF16:
        print("Warning: BF16 not supported. Training in FP16/FP32 may be slower or less stable.")

    # Get model-specific config
    model_config = get_model_config(model_name)
    print(f"Model config: {model_config}")

    # --- 1. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Model Parameters Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*60}\n")

    # --- 2. Load and Prepare the Dataset ---
    try:
        dataset = load_dataset('json', data_files={'train': dataset_file})['train']
    except Exception as e:
        print(f"Error loading dataset {dataset_file}: {e}")
        return

    def tokenize_function(examples):
        text = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        
        tokenized_inputs = tokenizer(
            text, 
            padding="max_length", 
            max_length=model_config['max_length'], 
            truncation=True
            # I Removed return_tensors="pt"
        )
        
        # Create labels by copying input_ids
        labels = []
        for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
            label = input_ids.copy()
            # Set padding tokens to -100 (ignored in loss calculation)
            attention_mask = tokenized_inputs["attention_mask"][i]
            for j, mask_val in enumerate(attention_mask):
                if mask_val == 0:  # This is a padding position
                    label[j] = -100
            labels.append(label)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # --- 3. Set Up Training ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=model_config['per_device_train_batch_size'],
        gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
        bf16=USE_BF16,
        seed=seed,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        save_total_limit=2,
        load_best_model_at_end=False,
        data_seed=seed,
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
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl training file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    train_model(args.model_name, args.dataset_file, args.output_dir, args.num_epochs, args.learning_rate, args.seed)