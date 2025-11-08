import os
import torch
import argparse
import json
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

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

def get_model_config(model_name, fine_tune_method="full"):
    """Returns appropriate training config based on model size and fine-tuning method."""
    base_config = {
        '1B': {
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'max_length': 512
        },
        '3B': {
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 4,
            'max_length': 512
        },
        '8B': {
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'max_length': 512
        }
    }
    
    # Determine model size
    if '1B' in model_name or '1.1B' in model_name:
        config = base_config['1B'].copy()
    elif '3B' in model_name or '3.1B' in model_name:
        config = base_config['3B'].copy()
    elif '8B' in model_name or '8.1B' in model_name:
        config = base_config['8B'].copy()
    else:
        config = base_config['3B'].copy()  # Default
    
    # Adjust for QLoRA (can use smaller batch sizes)
    if fine_tune_method == "qlora":
        config['per_device_train_batch_size'] = max(1, config['per_device_train_batch_size'])
    
    config['fine_tune_method'] = fine_tune_method
    return config

def train_model(model_name, dataset_file, output_dir, num_epochs=3, learning_rate=2e-5, fine_tune_method="full", seed=42):
    """
    Loads and fine-tunes a model on a given dataset.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_file: Path to training .jsonl file
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        fine_tune_method: "full", "lora", or "qlora"
        seed: Random seed for reproducibility
    """
    
    # Set random seed FIRST, before any other operations
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Clear GPU cache at start
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    print(f"--- Starting Training ---")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")
    print(f"Fine-tuning Method: {fine_tune_method.upper()}")

    # Check for BF16 support
    USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not USE_BF16:
        print("Warning: BF16 not supported. Training in FP16/FP32 may be slower or less stable.")
    else:
        print("Using bfloat16 (bf16) for faster training and better stability")

    # Get model-specific config
    model_config = get_model_config(model_name, fine_tune_method)
    print(f"Model config: {model_config}")

    # --- 1. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup quantization config for QLoRA
    quantization_config = None
    if fine_tune_method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization (QLoRA)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Setup LoRA for LoRA/QLoRA methods ---
    if fine_tune_method in ["lora", "qlora"]:
        if fine_tune_method == "qlora":
            # Prepare model for k-bit training (required for QLoRA)
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # CRITICAL FIX: Ensure model is in training mode
        model.train()
        
        # Verify and enable gradients for LoRA parameters
        trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            # Ensure all LoRA parameters require gradients
            if 'lora' in name.lower():
                if not param.requires_grad:
                    print(f"Warning: Enabling gradients for {name}")
                    param.requires_grad = True
                    trainable_params += param.numel()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Model Parameters Summary ({fine_tune_method.upper()}):")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*60}\n")
        
        model.print_trainable_parameters()
        print(f"Using LoRA for parameter-efficient fine-tuning (method: {fine_tune_method})")
    else:
        # Full fine-tuning - calculate and print parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Model Parameters Summary (FULL FINE-TUNING):")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        print(f"{'='*60}\n")
        print("Using full fine-tuning (all parameters will be updated)")
        model.train()  # Ensure model is in training mode

    # --- 3. Load and Prepare the Dataset ---
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
            truncation=True, 
            return_tensors="pt"
        )
        
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    # --- 4. Set Up Training ---
    # IMPORTANT: Disable gradient checkpointing for LoRA/QLoRA to avoid the gradient error
    use_gradient_checkpointing = fine_tune_method == "full" and "8B" in model_name
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=model_config['per_device_train_batch_size'],
        gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
        bf16=USE_BF16,
        gradient_checkpointing=use_gradient_checkpointing,  # Only for full fine-tuning of 8B
        seed=seed,  # Add seed
        data_seed=seed,  # Add data seed
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        save_total_limit=2,
        load_best_model_at_end=False,
        # Use 8-bit optimizer for LoRA/QLoRA to save memory
        optim="paged_adamw_8bit" if fine_tune_method in ["lora", "qlora"] else "adamw_torch",
        dataloader_pin_memory=False,  # Reduce memory usage
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # --- 5. Start Training ---
    print("Starting trainer.train()...")
    # Ensure model is in training mode before training
    model.train()
    
    # Final check: verify at least some parameters require gradients
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise RuntimeError("No trainable parameters found! Check LoRA configuration.")
    print(f"Found {trainable_count} parameter groups with gradients enabled")
    
    trainer.train()

    # --- 6. Save the Final Model ---
    print("Training complete. Saving final model...")
    
    if fine_tune_method in ["lora", "qlora"]:
        # For LoRA/QLoRA, save the adapter (much smaller)
        model.save_pretrained(output_dir)
        
        # Ensure base_model_name_or_path is saved in adapter config
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            if 'base_model_name_or_path' not in adapter_config or not adapter_config['base_model_name_or_path']:
                adapter_config['base_model_name_or_path'] = model_name
                with open(adapter_config_path, 'w') as f:
                    json.dump(adapter_config, f, indent=2)
        
        print(f"LoRA adapter saved to: {output_dir}")
        print("Note: To use this model, load the base model and then load this adapter")
    else:
        # For full fine-tuning, save the entire model
        model.save_pretrained(output_dir)
        print(f"Full fine-tuned model saved to: {output_dir}")
    
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama model.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model to fine-tune.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl training file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--fine_tune_method", 
        type=str, 
        default="full", 
        choices=["full", "lora", "qlora"],
        help="Fine-tuning method: 'full' (full fine-tuning), 'lora' (LoRA), or 'qlora' (QLoRA with 4-bit quantization)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    train_model(
        args.model_name, 
        args.dataset_file, 
        args.output_dir, 
        args.num_epochs, 
        args.learning_rate,
        args.fine_tune_method,
        args.seed
    )