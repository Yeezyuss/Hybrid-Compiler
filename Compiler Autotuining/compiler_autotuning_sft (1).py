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
from verl.utils.hdfs_io import copy, makedirs
from typing import List, Dict, Optional, Any
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_overOz
from agent_r1.tool.tools.comiler_autotuning.raw_tool.find_best_pass_sequence import find_best_pass_sequence


def read_llvm_ir_file(file_path):
    """
    Read LLVM IR code from a file
    
    Args:
        file_path: Path to the LLVM IR file
        
    Returns:
        LLVM IR code as string
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def get_autophase_features(ll_code):
    """
    获取LLVM IR代码的autophase特征
    
    Args:
        ll_code: LLVM IR代码
        
    Returns:
        autophase特征字典，如果发生错误则返回None
    """
    try:
        # 获取autophase特征
        features = get_autophase_obs(ll_code)
        return features
    except Exception as e:
        print(f"Error getting autophase features: {e}")
        return None

# --- Main SFT Data Generation Function ---

def generate_thinking_process(filename: str, initial_autophase: Dict, pass_sequence: List[str]) -> str:
    """
    Generates a SFT sample using the instrcount_tool and find_best_pass_sequence function.
    Logic:
    1. Call instrcount_tool with given pass_sequence to check overoz
    2. If overoz > 0, provide pass_sequence as answer
    3. If overoz <= 0, call find_best_pass_sequence function up to 5 times until improvement_percentage > 0
    4. If all attempts fail, fallback to ["-Oz"]

    Args:
        filename: LLVM IR filename relative to llvmir_datasets dir.
        initial_autophase: Initial features of the original code.
        pass_sequence: The initial pass sequence to try.

    Returns:
        A string containing the full SFT sample.
    """
    result_parts = []

    # --- Input Validation ---
    if not filename or initial_autophase is None or not isinstance(pass_sequence, list):
        return "<error>Invalid input: Filename, initial features, and pass sequence required.</error>"
    pass_sequence = [str(p) for p in pass_sequence if p]
    if not pass_sequence:
        return "<error>Empty pass sequence provided.</error>"

    # --- Setup paths ---
    llvm_tools_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      '../../agent_r1/tool/tools/comiler_autotuning/raw_tool/'))
    llvm_ir_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './llvmir_datasets/')
    ll_file_path = os.path.join(llvm_ir_dir, filename)
    
    # --- STEP 1: Initial thinking with the proposed pass sequence ---
    initial_think = f"""<think>
[Initial Pass Sequence Analysis]
- Based on the provided autophase features, I'll try the following pass sequence:
{pass_sequence}

- First, I'll use the instrcount tool to check if this sequence provides improvement over -Oz optimization.
- If improvement_over_oz > 0, I'll use this sequence as my answer.
- If improvement_over_oz <= 0, I'll call find_best_pass_sequence tool to find a better sequence.
</think>"""

    # --- STEP 2: Call instrcount_tool to check overoz ---
    instrcount_tool_call_args = {
        "filename": filename,
        "optimization_flags": pass_sequence
    }
    instrcount_tool_call_content = json.dumps({"name": "instrcount", "arguments": instrcount_tool_call_args}, separators=(',', ':'))
    instrcount_tool_call_str = f"<tool_call>\n{instrcount_tool_call_content}\n</tool_call>"
    
    initial_assistant_turn = f"<|im_start|>assistant\n{initial_think}\n{instrcount_tool_call_str}\n<|im_end|>"
    result_parts.append(initial_assistant_turn)

    # --- Actually calculate overoz value using the raw function ---
    tool_response_dict = {"status": "error", "error": "Error calculating improvement over -Oz"}
    over_oz_value = None
    
    try:
        # Read the LLVM IR code
        with open(ll_file_path, 'r') as file:
            ll_code = file.read()
            
        # Directly calculate overoz using the raw function
        over_oz_value = get_overOz(ll_code, pass_sequence, llvm_tools_path)
        
        # Create tool response
        tool_response_dict = {
            "status": "success",
            "improvement_over_oz": over_oz_value
        }
        
    except Exception as e:
        error_msg = f"Exception during improvement calculation: {e}"
        tool_response_dict = {"status": "error", "error": error_msg}
        print(f"Warning: {error_msg} for {filename}")

    # --- First user turn with instrcount_tool response ---
    initial_user_turn = f"<|im_start|>user\n<tool_response>\n{json.dumps(tool_response_dict, indent=2)}\n</tool_response>\n<|im_end|>"
    result_parts.append(initial_user_turn)

    # --- STEP 3: Check overoz and decide next action ---
    current_pass_sequence = pass_sequence
    
    if tool_response_dict.get("status") == "success" and tool_response_dict.get("improvement_over_oz", 0) > 0:
        # Improvement over -Oz is positive, use the initial pass sequence
        final_thinking = f"""<think>
[Result Analysis]
- The instrcount_tool reports an improvement_over_oz value of {tool_response_dict.get("improvement_over_oz")}.
- Since this value is positive, the pass sequence provides better optimization than -Oz.
- I'll use this pass sequence as my final answer.
</think>"""
        
        final_assistant_turn = f"<|im_start|>assistant\n{final_thinking}\n<answer>\n{current_pass_sequence}\n</answer>\n<|im_end|>"
        result_parts.append(final_assistant_turn)
        return "\n".join(result_parts)
    
    # --- STEP 4: If overoz <= 0, call find_best_pass_sequence up to 5 times ---
    max_attempts = 1
    current_attempt = 0
    best_improvement = 0
    best_sequence = None
    
    while current_attempt < max_attempts:
        current_attempt += 1
        
        # Generate thinking for this attempt
        find_best_thinking = f"""<think>
[Finding Better Pass Sequence]
- The previous sequence did not provide improvement over -Oz.
- Calling find_best_pass_sequence tool to find a better sequence.
</think>"""
        
        # Call find_best_pass_sequence_tool
        find_best_args = {"filename": filename}
        find_best_tool_call_content = json.dumps({"name": "find_best_pass_sequence", "arguments": find_best_args}, separators=(',', ':'))
        find_best_tool_call_str = f"<tool_call>\n{find_best_tool_call_content}\n</tool_call>"
        
        find_best_assistant_turn = f"<|im_start|>assistant\n{find_best_thinking}\n{find_best_tool_call_str}\n<|im_end|>"
        result_parts.append(find_best_assistant_turn)
        
        # Actually call find_best_pass_sequence function
        find_best_response_dict = {"status": "error", "error": "Error finding best pass sequence"}
        improvement_percentage = None
        recommended_sequence = None
        
        try:
            # Directly call the raw function
            best_result = find_best_pass_sequence(ll_file_path, llvm_tools_path=llvm_tools_path)

            if best_result and 'best_pass_sequence' in best_result and 'improvement_percentage' in best_result:
                recommended_sequence = best_result['best_pass_sequence']
                improvement_percentage = best_result['improvement_percentage']
                
                find_best_response_dict = {
                    "status": "success",
                    "best_pass_sequence": recommended_sequence,
                    "improvement_percentage": improvement_percentage
                }
                
                # Keep track of best sequence found so far
                if improvement_percentage is not None and improvement_percentage > best_improvement:
                    best_improvement = improvement_percentage
                    best_sequence = recommended_sequence
            else:
                find_best_response_dict = {
                    "status": "error", 
                    "error": "Invalid result from find_best_pass_sequence function"
                }
                
        except Exception as e:
            error_msg = f"Exception during find_best_pass_sequence call (attempt {current_attempt}): {e}"
            find_best_response_dict = {"status": "error", "error": error_msg}
            print(f"Warning: {error_msg} for {filename}")
        
        # User turn with find_best_pass_sequence_tool response
        find_best_user_turn = f"<|im_start|>user\n<tool_response>\n{json.dumps(find_best_response_dict, indent=2)}\n</tool_response>\n<|im_end|>"
        result_parts.append(find_best_user_turn)
        
        # Check if we found a good sequence
        if find_best_response_dict.get("status") == "success" and find_best_response_dict.get("improvement_percentage", 0) > 0:
            current_pass_sequence = find_best_response_dict.get("best_pass_sequence", current_pass_sequence)
            break
            
        # If this was the last attempt and we failed, but we had a previous best sequence
        if current_attempt == max_attempts and best_improvement > 0 and best_sequence:
            current_pass_sequence = best_sequence
            break
            
        # If this was the last attempt and we still don't have a good sequence, fallback to -Oz
        if current_attempt == max_attempts:
            current_pass_sequence = ["-Oz"]  # Fallback to -Oz
            break
    
    # --- STEP 5: Final answer with the best sequence found or fallback ---
    if current_pass_sequence == ["-Oz"]:
        final_thinking = f"""<think>
[Final Decision - Fallback to -Oz]
- After attempting with find_best_pass_sequence tool, no sequence with positive improvement was found.
- Falling back to the standard -Oz optimization as the final answer.
</think>"""
    else:
        final_thinking = f"""<think>
[Final Decision - Found Improved Sequence]
- Found a pass sequence with positive improvement: {best_improvement if best_improvement > 0 else 'positive'}.
- Using this sequence as the final answer.
</think>"""
    
    final_assistant_turn = f"<|im_start|>assistant\n{final_thinking}\n<answer>\n{current_pass_sequence}\n</answer>\n<|im_end|>"
    result_parts.append(final_assistant_turn)
    
    # print("\n".join(result_parts))
    return "\n".join(result_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='compiler_autotuning_data.csv',
                        help='Path to the compiler autotuning data CSV file')
    parser.add_argument('--llvm_ir_dir', default=None, 
                        help='Directory containing LLVM IR files (optional)')
    parser.add_argument('--local_dir', default='~/data/compiler_autotuning_sft',
                        help='Local directory to save the processed data')
    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS directory to save the processed data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of sequences to retrieve in RAG approach')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 加载数据集
    print(f"Loading compiler autotuning data from {args.data_file}...")
    
    # 确定CSV文件的完整路径
    if os.path.isabs(args.data_file):
        csv_path = args.data_file
    else:
        # 如果是相对路径，检查它是否在当前目录
        if os.path.exists(args.data_file):
            csv_path = args.data_file
        # 检查它是否在与此脚本相同的目录
        elif os.path.exists(os.path.join(os.path.dirname(__file__), args.data_file)):
            csv_path = os.path.join(os.path.dirname(__file__), args.data_file)
        else:
            raise FileNotFoundError(f"Could not find {args.data_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 限制样本数量（如果需要）
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
    
    print(f"Loaded {len(df)} samples")
    
    # 处理数据帧
    data_records = []
    print(f"Processing data with simple approach...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        # 提取文件名
        filename = row['Filename']
        
        # 解析Autophase_embedding和PassSequence
        try:
            ll_code = None
            if args.llvm_ir_dir is not None:
                ll_file_path = os.path.join(args.llvm_ir_dir, filename)
                ll_code = read_llvm_ir_file(ll_file_path)

            # 获取Autophase特征和pass序列
            autophase_embedding = get_autophase_features(ll_code)
            initial_inst_count = autophase_embedding.get('TotalInsts', 'N/A')
            formatted_features = json.dumps(autophase_embedding, indent=2)
            pass_sequence = ast.literal_eval(row['PassSequence'])
            over_oz = float(row['OverOz'])
            
            # 为SFT构造问题和答案
            question = f"""Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count.
The LLVM IR code is represented by autophase features, the initial autophase features are:
```json
{formatted_features}
```
Initial instruction count: {initial_inst_count}

Your task is to:

Evaluate the provided Initial Candidate Pass Sequence using the instrcount tool to determine its instruction count improvement compared to the default -Oz optimization.
If the initial sequence provides a positive improvement (improvement_over_oz > 0), recommend it as the final answer.
If the initial sequence does not provide a positive improvement (improvement_over_oz <= 0), use the find_best_pass_sequence tool to search for a better sequence.
If the search finds a sequence with positive improvement (improvement_percentage > 0), recommend that sequence.
If the search tool fails to find a sequence with positive improvement, recommend the default ['-Oz'] sequence as the safest option.
Present your reasoning step-by-step using <think> tags and tool interactions using <tool_call> and <tool_response> structure, concluding with the final recommended sequence in an <answer> tag.
"""
            
            # 生成思考过程和工具调用
            full_process = generate_thinking_process(filename, autophase_embedding, pass_sequence)

            # 创建记录
            record = {
                'question': question,
                'answer': full_process,
                'filename': filename,
                'autophase_embedding': autophase_embedding,
                'pass_sequence': pass_sequence,
                'over_oz': over_oz
            }
            
            data_records.append(record)
        except Exception as e:
            print(f"\nError processing row for {filename}: {e}")
            continue
    
    print(f"Successfully processed {len(data_records)} records out of {len(df)} samples")
    
    # 创建数据集
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data_records))
    
    # 拆分数据集
    splits = dataset.train_test_split(
        test_size=args.val_ratio + args.test_ratio,
        seed=args.seed
    )
    train_dataset = splits['train']
    
    val_test_splits = splits['test'].train_test_split(
        test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
        seed=args.seed
    )
    validation_dataset = val_test_splits['train']
    test_dataset = val_test_splits['test']
    
    print(f"Dataset split: {len(train_dataset)} train, {len(validation_dataset)} validation, {len(test_dataset)} test")
    
    # 构造SFT格式的数据
    def process_for_sft(example):
        return {
            "extra_info": {
                "question": example["question"],
                "answer": example["answer"]
            },
            "data_source": "compiler_autotuning",
            "ability": "compiler_autotuning"
        }
    
    train_dataset = train_dataset.map(process_for_sft)
    validation_dataset = validation_dataset.map(process_for_sft)
    test_dataset = test_dataset.map(process_for_sft)
    
    # 保存数据集
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Saved processed datasets to {local_dir}")
    
    # 如果提供了HDFS目录，将数据集复制到那里
    if args.hdfs_dir is not None:
        print(f"Copying datasets to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy completed")

if __name__ == '__main__':
    main() 