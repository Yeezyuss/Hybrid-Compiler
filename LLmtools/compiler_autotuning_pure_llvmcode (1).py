#!/usr/bin/env python
import os
import pandas as pd
import datasets
import argparse
import json
import random
import ast
from tqdm import tqdm
import numpy as np
from verl.utils.hdfs_io import copy, makedirs # Assuming this is available in your environment
from typing import List, Dict, Optional, Any
# Ensure these imports are correct based on your project structure
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs

def read_llvm_ir_file(file_path: str) -> Optional[str]:
    """
    Read LLVM IR code from a file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def get_autophase_features(ll_code: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    获取LLVM IR代码的autophase特征
    """
    if ll_code is None:
        # print("Warning: ll_code is None in get_autophase_features.")
        return None
    try:
        features = get_autophase_obs(ll_code)
        return features
    except Exception as e:
        # print(f"Error getting autophase features: {e}") # Less verbose
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='/PATH_PLACEHOLDER/NIPS_Material/examples/data_preprocess/Experiment_3.csv',
                        help='Path to the compiler autotuning data CSV file')
    parser.add_argument('--llvm_ir_dir', required=True, # Made required
                        help='Directory containing LLVM IR files')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning_sft/pure_llvmcode/', # Changed dir name
                        help='Local directory to save the processed data')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to save the processed data')
    parser.add_argument('--train_ratio', type=float, default=0.99,
                        help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.005,
                        help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.005,
                        help='Ratio of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if not os.path.isdir(args.llvm_ir_dir):
        print(f"Error: LLVM IR directory not found: {args.llvm_ir_dir}")
        return

    print(f"Loading compiler autotuning data from {args.data_file}...")
    if os.path.isabs(args.data_file):
        csv_path = args.data_file
    else:
        script_dir_path = os.path.dirname(__file__)
        if os.path.exists(os.path.join(script_dir_path, args.data_file)):
             csv_path = os.path.join(script_dir_path, args.data_file)
        elif os.path.exists(args.data_file):
            csv_path = args.data_file
        else:
            raise FileNotFoundError(f"Could not find data_file: {args.data_file}")
    
    df = pd.read_csv(csv_path)
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed).copy()
    print(f"Loaded {len(df)} samples")
    
    data_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        filename = str(row['Filename']).strip() # Ensure filename is string and stripped
        
        ll_file_path = os.path.join(args.llvm_ir_dir, filename)
        ll_code = read_llvm_ir_file(ll_file_path)

        if ll_code is None:
            print(f"Skipping {filename}: Could not read LLVM IR code.")
            continue
            
        initial_autophase_features = get_autophase_features(ll_code)
        if initial_autophase_features is None:
            print(f"Skipping {filename}: Could not get initial autophase features.")
            continue
        
        initial_inst_count = initial_autophase_features.get('TotalInsts', 'N/A')
        
        try:
            pass_sequence_str = row['PassSequence']
            # Handle potential 'nan' or other non-string values safely
            if pd.isna(pass_sequence_str) or not isinstance(pass_sequence_str, str):
                 print(f"Skipping {filename}: PassSequence is not a valid string representation of a list ('{pass_sequence_str}').")
                 continue
            pass_sequence = ast.literal_eval(pass_sequence_str)
            if not isinstance(pass_sequence, list):
                print(f"Skipping {filename}: Parsed PassSequence is not a list.")
                continue
        except (ValueError, SyntaxError) as e:
            print(f"Skipping {filename}: Error parsing PassSequence '{row['PassSequence']}': {e}")
            continue
            
        over_oz_str = str(row.get('OverOz', '0.0')) # Default to '0.0' if missing
        try:
            over_oz = float(over_oz_str)
        except ValueError:
            print(f"Warning for {filename}: Could not parse OverOz '{over_oz_str}' as float. Defaulting to 0.0.")
            over_oz = 0.0

        # Truncate ll_code for prompt if too long (e.g., > 30000 chars), add indicator
        # This is a practical consideration for very large IR files in prompts.
        # Adjust limit as needed.
        MAX_LL_CODE_LEN_FOR_PROMPT = 30000 
        display_ll_code = ll_code
        # if len(ll_code) > MAX_LL_CODE_LEN_FOR_PROMPT:
        #     display_ll_code = ll_code[:MAX_LL_CODE_LEN_FOR_PROMPT] + "\n\n... (LLVM IR truncated due to length) ..."
            
        question = f"""Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count.
The LLVM IR code is:
```llvm
{display_ll_code}
Initial instruction count for this code: {initial_inst_count}
"""
            # Generate thinking process (uncomment if you want the full simulation)
        # Ensure generate_thinking_process is robust to None initial_autophase_features
        # For now, we've ensured it's not None before this point.
        # full_process_str = generate_thinking_process(filename, ll_code, initial_autophase_features, pass_sequence)
        
        # If you want a simpler answer for SFT without the full thinking process:
        simple_answer_pass_sequence_json = json.dumps(pass_sequence) # Ensure it's JSON array
        full_process_str = f"<|im_start|>assistant\n<answer>\n{simple_answer_pass_sequence_json}\n</answer>\n<|im_end|>"

        record = {
            'question': question,
            'answer': full_process_str, # Use the generated full process
            'filename': filename,
            # 'initial_autophase_embedding': initial_autophase_features, # Keep for metadata if needed
            'target_pass_sequence': pass_sequence, # Store the ground truth sequence
            'target_over_oz': over_oz
        }
        data_records.append(record)

    if not data_records:
        print("No data records were successfully processed. Exiting.")
        return

    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data_records))

    if len(dataset) == 0:
        print("Dataset is empty after processing. Cannot split. Exiting.")
        return
        
    # Ensure there are enough samples for splitting
    min_samples_for_split = 2 # Need at least 1 for train, 1 for test to split further
    if args.val_ratio + args.test_ratio > 0 and len(dataset) < min_samples_for_split:
        print(f"Warning: Dataset has only {len(dataset)} samples. Not enough to create train/val/test splits as configured. Using all for training.")
        train_dataset = dataset
        validation_dataset = datasets.Dataset.from_dict({}) # Empty dataset
        test_dataset = datasets.Dataset.from_dict({}) # Empty dataset
    elif args.val_ratio + args.test_ratio == 0: # Only training
        train_dataset = dataset
        validation_dataset = datasets.Dataset.from_dict({})
        test_dataset = datasets.Dataset.from_dict({})
    else:
        test_val_size = args.val_ratio + args.test_ratio
        if test_val_size >= 1.0: # Handle case where test+val is 100% or more
            print("Warning: val_ratio + test_ratio >= 1.0. Adjusting to leave at least one sample for training if possible.")
            if len(dataset) > 1:
                test_val_size = (len(dataset) -1) / len(dataset)
            else: # Cannot split, put all in train
                train_dataset = dataset
                validation_dataset = datasets.Dataset.from_dict({})
                test_dataset = datasets.Dataset.from_dict({})
                test_val_size = 0 # Skip further splitting

        if test_val_size > 0 :
            splits = dataset.train_test_split(
                test_size=test_val_size,
                seed=args.seed
            )
            train_dataset = splits['train']
            
            if args.val_ratio > 0 and args.test_ratio > 0 and len(splits['test']) > 1 :
                val_test_splits = splits['test'].train_test_split(
                    test_size=args.test_ratio / (args.val_ratio + args.test_ratio), # Proportion of combined val/test for test
                    seed=args.seed
                )
                validation_dataset = val_test_splits['train']
                test_dataset = val_test_splits['test']
            elif args.val_ratio > 0 and len(splits['test']) > 0: # Only validation needed from remainder
                validation_dataset = splits['test']
                test_dataset = datasets.Dataset.from_dict({})
            elif args.test_ratio > 0 and len(splits['test']) > 0: # Only test needed from remainder
                test_dataset = splits['test']
                validation_dataset = datasets.Dataset.from_dict({})
            else: # Remainder is too small or one ratio is 0
                validation_dataset = datasets.Dataset.from_dict({}) if args.val_ratio == 0 else splits['test']
                test_dataset = datasets.Dataset.from_dict({}) if args.test_ratio == 0 else splits['test']
                if args.val_ratio > 0 and args.test_ratio > 0: # Both wanted but couldn't split, put all in val
                    validation_dataset = splits['test']
                    test_dataset = datasets.Dataset.from_dict({})


        else: # No validation or test split needed based on ratios
            train_dataset = dataset
            validation_dataset = datasets.Dataset.from_dict({})
            test_dataset = datasets.Dataset.from_dict({})


    print(f"Dataset split: {len(train_dataset)} train, {len(validation_dataset)} validation, {len(test_dataset)} test")

    def process_for_sft(example):
        return {
            "extra_info": {
                "question": example["question"],
                "answer": example["answer"]
            },
            "data_source": "compiler_autotuning",
            "ability": "compiler_autotuning"
        }

    train_dataset = train_dataset.map(process_for_sft, load_from_cache_file=False)
    if len(validation_dataset) > 0:
        validation_dataset = validation_dataset.map(process_for_sft, load_from_cache_file=False)
    if len(test_dataset) > 0:
        test_dataset = test_dataset.map(process_for_sft, load_from_cache_file=False)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if len(validation_dataset) > 0:
        validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    if len(test_dataset) > 0:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved processed datasets to {local_dir}")

    if args.hdfs_dir is not None:
        print(f"Copying datasets to HDFS: {args.hdfs_dir}")
        try:
            makedirs(args.hdfs_dir, exist_ok=True) # Ensure target HDFS dir exists
            copy(src=local_dir, dst=args.hdfs_dir, overwrite=True) # Overwrite if files exist
            print("Copy completed")
        except Exception as e:
            print(f"Error copying to HDFS: {e}")



main()