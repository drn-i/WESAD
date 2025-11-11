import torch
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

def run_evaluation(model_path, dataset_file, debug=False, max_debug_samples=10):
    """Loads a fine-tuned model and evaluates it against a .jsonl test file."""
    
    print("--- Starting Evaluation ---")
    print(f"Model: {model_path}")
    print(f"Test File: {dataset_file}")
    print(f"Debug Mode: {debug}")

    # Check for BF16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        # Try loading as local path first, then as HuggingFace model
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
    
    # Debug statistics
    prompt_lengths = []
    token_counts = []
    generated_responses = []
    true_labels_list = []
    pred_labels_list = []
    mismatch_examples = []
    
    print(f"Running evaluation on {total_samples} samples...")
    
    for idx, item in enumerate(tqdm(test_dataset)):
        prompt = item['prompt']
        true_label = item['completion'].replace('</s>', '').strip().lower()
        
        # Tokenize and analyze prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs.input_ids.shape[-1]
        
        # Collect statistics
        prompt_lengths.append(len(prompt))
        token_counts.append(prompt_tokens)
        true_labels_list.append(true_label)
        
        # Generate response
        try:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=15,  # Increased from 5
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            pred_label = response.strip().lower()
            
            generated_responses.append(response)
            pred_labels_list.append(pred_label)
            
        except Exception as e:
            print(f"\nERROR generating response for sample {idx}: {e}")
            response = ""
            pred_label = ""
            pred_labels_list.append("")
            generated_responses.append("")
        
        # Check if prediction matches
        is_correct = False
        if pred_label == true_label:
            correct_predictions += 1
            is_correct = True
        elif true_label in pred_label:
            correct_predictions += 1
            is_correct = True
        
        # Debug output
        if debug and (idx < max_debug_samples or not is_correct):
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{total_samples}")
            print(f"Prompt length: {len(prompt)} chars, {prompt_tokens} tokens")
            print(f"True label: '{true_label}'")
            print(f"Generated response (raw): '{response}'")
            print(f"Predicted label (cleaned): '{pred_label}'")
            print(f"Match: {is_correct}")
            if not is_correct:
                mismatch_examples.append({
                    'idx': idx,
                    'true': true_label,
                    'pred': pred_label,
                    'raw_response': response
                })
            print(f"{'='*80}")
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_samples) * 100
    
    print("\n" + "="*80)
    print("--- EVALUATION COMPLETE ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print("="*80)
    
    # Debug statistics
    if debug or accuracy == 0.0:
        print("\n" + "="*80)
        print("--- DEBUG STATISTICS ---")
        print(f"Prompt Statistics:")
        print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} chars")
        print(f"  Average token count: {sum(token_counts)/len(token_counts):.1f} tokens")
        print(f"  Max token count: {max(token_counts)} tokens")
        print(f"  Max context (512): {'⚠️ EXCEEDED' if max(token_counts) > 512 else '✓ OK'}")
        
        print(f"\nLabel Distribution:")
        print(f"  True labels: {Counter(true_labels_list)}")
        print(f"  Predicted labels: {Counter(pred_labels_list)}")
        
        print(f"\nTop Generated Responses:")
        unique_responses = Counter(generated_responses)
        for resp, count in unique_responses.most_common(10):
            print(f"  '{resp[:60]}...' ({count} times)")
        
        print("="*80)
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama model on a test set.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_debug_samples", type=int, default=10)
    
    args = parser.parse_args()
    run_evaluation(args.model_path, args.dataset_file, debug=args.debug, max_debug_samples=args.max_debug_samples)