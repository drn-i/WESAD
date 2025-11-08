import torch
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

def run_evaluation(model_path, dataset_file):
    """Loads a fine-tuned model and evaluates it against a .jsonl test file."""
    
    print("--- Starting Evaluation ---")
    print(f"Model: {model_path}")
    print(f"Test File: {dataset_file}")

    # Check for BF16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        # Check if this is a LoRA adapter (has adapter_config.json)
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            print("Detected LoRA/QLoRA adapter. Loading base model first...")
            
            # Read adapter config to get base model name
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
            
            if not base_model_name:
                # Fallback: try to infer from model path or use default
                if "8B" in model_path or "8b" in model_path:
                    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
                elif "3B" in model_path or "3b" in model_path:
                    base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
                elif "1B" in model_path or "1b" in model_path:
                    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
                else:
                    raise ValueError("Could not determine base model name. Please specify in adapter_config.json")
            
            print(f"Base model: {base_model_name}")
            
            # Check if QLoRA (4-bit) was used by checking quantization config in adapter
            quantization_config = None
            if adapter_config.get("quantization_config"):
                # QLoRA was used - load base model with 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("Using 4-bit quantization (QLoRA)")
            
            # Load base model
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quantization_config,
            )
            
            # Load LoRA adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
            print("LoRA adapter loaded successfully")
            
        else:
            # Regular full fine-tuned model
            print("Detected full fine-tuned model")
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, device_map="auto", local_files_only=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, device_map="auto"
                )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        print(f"Make sure the model path exists: {model_path}")
        import traceback
        traceback.print_exc()
        return

    try:
        test_dataset = load_dataset('json', data_files=dataset_file)['train']
    except Exception as e:
        print(f"An error occurred loading the dataset: {e}")
        return

    correct_predictions = 0
    total_samples = len(test_dataset)
    
    print(f"Running evaluation on {total_samples} samples...")
    
    for item in tqdm(test_dataset):
        prompt = item['prompt']
        true_label = item['completion'].replace('</s>', '').strip().lower()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Use greedy decoding for consistency
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred_label = response.strip().lower()

        if pred_label == true_label:
            correct_predictions += 1
        elif true_label in pred_label:  # Handle simple cases like 'stress.'
            correct_predictions += 1
            
    # --- Calculate Final Accuracy ---
    accuracy = (correct_predictions / total_samples) * 100
    
    print("\n" + "="*80)
    print("--- EVALUATION COMPLETE ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print("="*80)
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama model on a test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl test file.")
    
    args = parser.parse_args()
    run_evaluation(args.model_path, args.dataset_file)