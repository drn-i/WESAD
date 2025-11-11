import torch
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
                max_new_tokens=15,  # Increased from 5 to capture more output
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Use greedy decoding for consistency
                temperature=None,  # Disable temperature for deterministic output
                top_p=None
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            pred_label = response.strip().lower()
            
            # Collect generated responses
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
        elif true_label in pred_label:  # Handle simple cases like 'stress.'
            correct_predictions += 1
            is_correct = True
        
        # Debug output for first N samples or mismatches
        if debug and (idx < max_debug_samples or not is_correct):
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{total_samples}")
            print(f"Prompt length: {len(prompt)} chars, {prompt_tokens} tokens")
            print(f"True label: '{true_label}'")
            print(f"Generated response (raw): '{response}'")
            print(f"Predicted label (cleaned): '{pred_label}'")
            print(f"Match: {is_correct}")
            if not is_correct:
                print(f"❌ MISMATCH DETECTED")
                mismatch_examples.append({
                    'idx': idx,
                    'true': true_label,
                    'pred': pred_label,
                    'raw_response': response,
                    'prompt_tokens': prompt_tokens
                })
            print(f"{'='*80}")
    
    # --- Calculate Final Accuracy ---
    accuracy = (correct_predictions / total_samples) * 100
    
    # --- Print Statistics ---
    print("\n" + "="*80)
    print("--- EVALUATION COMPLETE ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print("="*80)
    
    # --- Print Debug Statistics ---
    if debug or accuracy == 0.0:
        print("\n" + "="*80)
        print("--- DEBUG STATISTICS ---")
        print(f"Prompt Statistics:")
        print(f"  Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} chars")
        print(f"  Min prompt length: {min(prompt_lengths)} chars")
        print(f"  Max prompt length: {max(prompt_lengths)} chars")
        print(f"  Average token count: {sum(token_counts)/len(token_counts):.1f} tokens")
        print(f"  Min token count: {min(token_counts)} tokens")
        print(f"  Max token count: {max(token_counts)} tokens")
        print(f"  Max context (512): {'⚠️ EXCEEDED' if max(token_counts) > 512 else '✓ OK'}")
        
        print(f"\nLabel Distribution:")
        print(f"  True labels: {Counter(true_labels_list)}")
        print(f"  Predicted labels: {Counter(pred_labels_list)}")
        
        print(f"\nGenerated Response Analysis:")
        unique_responses = Counter(generated_responses)
        print(f"  Unique responses: {len(unique_responses)}")
        print(f"  Top 10 most common responses:")
        for resp, count in unique_responses.most_common(10):
            print(f"    '{resp[:50]}...' (appears {count} times)")
        
        if mismatch_examples:
            print(f"\n⚠️  Found {len(mismatch_examples)} mismatch examples:")
            for ex in mismatch_examples[:5]:  # Show first 5
                print(f"  Sample {ex['idx']}: True='{ex['true']}' vs Pred='{ex['pred']}'")
                print(f"    Raw response: '{ex['raw_response'][:100]}...'")
                print(f"    Prompt tokens: {ex['prompt_tokens']}")
        
        # Check for common issues
        print(f"\n--- DIAGNOSTIC CHECKS ---")
        if all(not resp.strip() for resp in generated_responses):
            print("⚠️  WARNING: All generated responses are empty!")
        elif all(resp.strip() == pred_labels_list[0] for resp in generated_responses):
            print(f"⚠️  WARNING: Model is generating the same response for all samples: '{pred_labels_list[0]}'")
        
        if max(token_counts) > 512:
            print("⚠️  WARNING: Some prompts exceed 512 token limit!")
            print(f"   This may cause truncation during training/inference.")
        
        if not any('baseline' in pred or 'stress' in pred for pred in pred_labels_list):
            print("⚠️  WARNING: Model is not generating expected labels (baseline/stress)")
            print(f"   Most common prediction: '{Counter(pred_labels_list).most_common(1)[0][0] if pred_labels_list else 'N/A'}'")
        
        print("="*80)
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Llama model on a test set with debugging.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the .jsonl test file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output.")
    parser.add_argument("--max_debug_samples", type=int, default=10, help="Number of samples to show in debug mode.")
    
    args = parser.parse_args()
    run_evaluation(args.model_path, args.dataset_file, debug=args.debug, max_debug_samples=args.max_debug_samples)
