import torch
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm # For a nice progress bar

def run_evaluation(model_path, dataset_file):
    """Loads a fine-tuned model and evaluates it against a .jsonl test file."""
    
    print("--- Starting Evaluation ---")
    print(f"Model: {model_path}")
    print(f"Test File: {dataset_file}")

    # Check for BF16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
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
            **inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred_label = response.strip().lower()

        if pred_label == true_label:
            correct_predictions += 1
        elif true_label in pred_label: # Handle simple cases like 'stress.'
             correct_predictions += 1
            
    # --- Calculate Final Accuracy ---
    accuracy = (correct_predictions / total_samples) * 100
    
    print("\n" + "="*80)
    print("--- EVALUATION COMPLETE ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama model on a test set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl test file.")
    
    args = parser.parse_args()
    run_evaluation(args.model_path, args.dataset_file)
