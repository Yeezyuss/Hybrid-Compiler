#!/usr/bin/env python
# Copyright 2024 XXX and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import GenerateOptimizedLLCode
from agent_r1.tool.tools.comiler_autotuning.raw_tool.gen_pass_from_number import Actions_LLVM_10_0_0

# LLVM优化pass的功能描述，帮助生成思考过程
PASS_DESCRIPTIONS = {
    "--add-discriminators": "Add discriminators for better debug info.",
    "--adce": "Aggressively eliminate dead code.",
    "--aggressive-instcombine": "Aggressive instruction combining.",
    "--alignment-from-assumptions": "Optimize memory alignment based on assumptions.",
    "--always-inline": "Inline all always_inline functions.",
    "--argpromotion": "Promote arguments from byref to byval.",
    "--attributor": "Propagate attributes across modules.",
    "--barrier": "Place barriers before code generation.",
    "--bdce": "Bit-level dead code elimination.",
    "--break-crit-edges": "Break critical edges to simplify CFG.",
    "--simplifycfg": "Simplify the control flow graph.",
    "--callsite-splitting": "Split indirect call sites based on constants.",
    "--called-value-propagation": "Propagate called values at indirect call sites.",
    "--canonicalize-aliases": "Canonicalize aliases for better analysis.",
    "--consthoist": "Hoist constants to higher scopes.",
    "--constmerge": "Merge duplicate constants.",
    "--constprop": "Simple constant propagation.",
    "--coro-cleanup": "Remove coroutine scheduling remnants.",
    "--coro-early": "Early coroutine transformation.",
    "--coro-elide": "Remove unnecessary coroutine constructs.",
    "--coro-split": "Split coroutines into multiple functions.",
    "--correlated-propagation": "Propagate correlated value info.",
    "--cross-dso-cfi": "Cross-DSO control flow integrity.",
    "--deadargelim": "Remove unused function arguments.",
    "--dce": "Dead code elimination.",
    "--die": "Dead instruction elimination.",
    "--dse": "Dead store elimination.",
    "--reg2mem": "Convert registers to stack memory references.",
    "--div-rem-pairs": "Optimize division and remainder pairs.",
    "--early-cse-memssa": "Early CSE based on memory SSA.",
    "--early-cse": "Early common subexpression elimination.",
    "--elim-avail-extern": "Convert available external globals to definitions.",
    "--ee-instrument": "Instrument exception handling for stack space.",
    "--flattencfg": "Flatten the control flow graph.",
    "--float2int": "Optimize floating-point to integer computations.",
    "--forceattrs": "Force setting function attributes.",
    "--inline": "Inline function code at call sites.",
    "--insert-gcov-profiling": "Insert GCOV-compatible instrumentation.",
    "--gvn-hoist": "Hoist redundant expressions.",
    "--gvn": "Global value numbering.",
    "--globaldce": "Global dead code elimination.",
    "--globalopt": "Global variable optimization.",
    "--globalsplit": "Split global variables into fragments.",
    "--guard-widening": "Widen guard conditions.",
    "--hotcoldsplit": "Split hot and cold paths.",
    "--ipconstprop": "Interprocedural constant propagation.",
    "--ipsccp": "Interprocedural sparse conditional constant propagation.",
    "--indvars": "Canonicalize loop induction variables.",
    "--irce": "Inductive range check elimination.",
    "--infer-address-spaces": "Infer address spaces.",
    "--inferattrs": "Infer attributes for unknown functions.",
    "--inject-tli-mappings": "Inject target library info mappings.",
    "--instsimplify": "Remove redundant instructions.",
    "--instcombine": "Combine instructions into simpler forms.",
    "--instnamer": "Assign names to unnamed instructions.",
    "--jump-threading": "Thread conditional jumps.",
    "--lcssa": "Convert loops to loop-closed SSA form.",
    "--licm": "Move loop-invariant code out of loops.",
    "--libcalls-shrinkwrap": "Optimize library call wrappers.",
    "--load-store-vectorizer": "Vectorize adjacent loads and stores.",
    "--loop-data-prefetch": "Prefetch data in loops.",
    "--loop-deletion": "Delete useless loops.",
    "--loop-distribute": "Distribute loops for parallelism.",
    "--loop-fusion": "Fuse loops to reduce overhead.",
    "--loop-guard-widening": "Widen loop guard conditions.",
    "--loop-idiom": "Recognize and replace common idioms in loops.",
    "--loop-instsimplify": "Simplify instructions in loops.",
    "--loop-interchange": "Interchange nested loops.",
    "--loop-load-elim": "Eliminate redundant loads in loops.",
    "--loop-predication": "Convert branches in loops to selects.",
    "--loop-reroll": "Reroll unrolled loops.",
    "--loop-rotate": "Rotate loops for better execution.",
    "--loop-simplifycfg": "Simplify loop control flow graph.",
    "--loop-simplify": "Canonicalize loop form.",
    "--loop-sink": "Sink instructions in loops.",
    "--loop-reduce": "Loop strength reduction.",
    "--loop-unroll-and-jam": "Unroll and jam nested loops.",
    "--loop-unroll": "Unroll loops.",
    "--loop-unswitch": "Extract conditions from loops.",
    "--loop-vectorize": "Vectorize loops.",
    "--loop-versioning-licm": "Create loop versions for LICM.",
    "--loop-versioning": "Create multiple loop versions.",
    "--loweratomic": "Lower atomic instructions.",
    "--lower-constant-intrinsics": "Lower constant intrinsics.",
    "--lower-expect": "Lower llvm.expect intrinsics.",
    "--lower-guard-intrinsic": "Lower guard intrinsics.",
    "--lowerinvoke": "Lower invoke and unwind instructions.",
    "--lower-matrix-intrinsics": "Lower matrix operation intrinsics.",
    "--lowerswitch": "Lower switch instructions.",
    "--lower-widenable-condition": "Lower widenable conditions.",
    "--memcpyopt": "Optimize memory copy operations.",
    "--mergefunc": "Merge duplicate functions.",
    "--mergeicmps": "Merge consecutive compare instructions.",
    "--mldst-motion": "Move memory load/store operations.",
    "--sancov": "Instrument sanitizer coverage.",
    "--name-anon-globals": "Name anonymous global variables.",
    "--nary-reassociate": "Reassociate n-ary expressions.",
    "--newgvn": "New global value numbering.",
    "--pgo-memop-opt": "Profile-guided memory operation optimization.",
    "--partial-inliner": "Partially inline hot paths.",
    "--partially-inline-libcalls": "Partially inline library calls.",
    "--post-inline-ee-instrument": "Instrument exception handling after inlining.",
    "--functionattrs": "Infer function attributes.",
    "--mem2reg": "Convert memory references to registers.",
    "--prune-eh": "Remove unreachable exception handling code.",
    "--reassociate": "Reassociate expressions.",
    "--redundant-dbg-inst-elim": "Remove redundant debug instructions.",
    "--rpo-functionattrs": "Infer function attributes in reverse postorder.",
    "--rewrite-statepoints-for-gc": "Rewrite statepoints for garbage collection.",
    "--sccp": "Sparse conditional constant propagation.",
    "--slp-vectorizer": "Superword-level parallelism vectorization.",
    "--sroa": "Scalar replacement of aggregates.",
    "--scalarizer": "Convert vector operations to scalar operations.",
    "--separate-const-offset-from-gep": "Separate constant offsets from GEP instructions.",
    "--simple-loop-unswitch": "Simplified loop unswitching.",
    "--sink": "Sink instructions to their use points.",
    "--speculative-execution": "Speculatively execute instructions.",
    "--slsr": "Straight-line strength reduction.",
    "--strip-dead-prototypes": "Remove unused function prototypes.",
    "--strip-debug-declare": "Remove debug declarations.",
    "--strip-nondebug": "Remove all non-debug information.",
    "--strip": "Remove all symbol information.",
    "--tailcallelim": "Eliminate tail calls.",
    "--mergereturn": "Merge multiple return points into one.",
    "-Oz": "Optimize aggressively for size."
}

DEFAULT_PASS_DESC = "General optimization pass."

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

def analyze_feature_changes(prev_features_dict, new_features_dict):
    """
    分析特征变化，为下一轮优化提供依据
    
    Args:
        prev_features_dict: 上一轮的特征（字典格式）
        new_features_dict: 新的特征（字典格式）
        
    Returns:
        特征变化分析
    """
    # 检查输入格式
    if not isinstance(prev_features_dict, dict) or not isinstance(new_features_dict, dict):
        return "无法比较特征变化：输入格式不正确"
    
    analysis = []
    try:
        # 找出所有变化的特征
        changed_features = []
        
        # 检查两个字典中共有的键
        common_keys = set(prev_features_dict.keys()) & set(new_features_dict.keys())
        for key in common_keys:
            if prev_features_dict[key] != new_features_dict[key]:
                change = new_features_dict[key] - prev_features_dict[key]
                direction = "increase" if change > 0 else "decrease"
                changed_features.append((key, abs(change), direction, change))
        
        # 按变化幅度排序，取前5个变化最大的特征
        changed_features.sort(key=lambda x: x[1], reverse=True)
        top_changes = changed_features[:5]
        
        # 生成分析文本：只输出变化最大的前5个特征
        for feature, change_abs, direction, change in top_changes:
            analysis.append(f"{feature}: {prev_features_dict[feature]} -> {new_features_dict[feature]} ({direction} {change_abs})")
        
        # 添加TotalInsts的变化情况
        if "TotalInsts" in common_keys:
            total_insts_change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
            if total_insts_change > 0:
                analysis.append(f"Total InstCount increased by {total_insts_change}")
            elif total_insts_change < 0:
                analysis.append(f"Total InstCount decreased by {abs(total_insts_change)}")
            else:
                analysis.append("Total InstCount unchanged")

        # 特殊情况：无显著变化
        if not analysis:
            # 检查总指令数
            if "TotalInsts" in common_keys:
                change = new_features_dict["TotalInsts"] - prev_features_dict["TotalInsts"]
                if change != 0:
                    direction = "increase" if change > 0 else "decrease"
                    analysis.append(f"Total InstCount {direction} by {abs(change)}")
                else:
                    analysis.append("Feature changes are not obvious")
            else:
                analysis.append("Feature changes are not obvious")
                
    except Exception as e:
        analysis.append(f"Feature analysis error: {e}")
    
    return ", ".join(analysis) if analysis else "Feature changes are not obvious"


# --- Helper to safely get instruction count ---
def _safe_get_inst_count(features: Optional[Dict]) -> Optional[int]:
     """Safely extracts TotalInsts, returning None if unavailable or not integer."""
     if not features: return None
     count = features.get("TotalInsts")
     if isinstance(count, int): return count
     # Add more flexible checks if TotalInsts might be string etc.
     try: return int(count)
     except (ValueError, TypeError): return None

# --- Main SFT Data Generation Function ---

def generate_thinking_process(filename: str, initial_autophase: Dict, pass_sequence: List[str]) -> str:
    """
    Generates a multi-round SFT sample using only the 'analyze_autophase' tool,
    including an initial -Oz check and final comparison.

    Args:
        filename: LLVM IR filename relative to llvmir_datasets dir.
        initial_autophase: Initial features of the *original* code.
        pass_sequence: The sequence guiding the 5 optimization rounds simulation.

    Returns:
        A string containing the full SFT sample.
    """
    result_parts = []

    # --- Initial Setup & Input Validation ---
    if not filename or initial_autophase is None or not isinstance(pass_sequence, list):
         return "<error>Invalid input: Filename, initial features, and pass sequence required.</error>"
    pass_sequence = [str(p) for p in pass_sequence if p]
    if not pass_sequence: return "<error>Empty pass sequence provided.</error>"

    # --- Simulation Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llvmir_base_path = os.path.normpath(os.path.join(script_dir, './llvmir_datasets/'))
    ll_file_path = os.path.join(llvmir_base_path, filename)
    llvm_tools_path = os.path.normpath(os.path.join(script_dir, '../../agent_r1/tool/tools/comiler_autotuning/raw_tool/')) # Adjust if needed

    try:
        original_ll_code = read_llvm_ir_file(ll_file_path)
        if original_ll_code is None: raise FileNotFoundError(f"Cannot read: {ll_file_path}")
    except Exception as e: return f"<error>Failed reading IR '{ll_file_path}': {e}</error>"

    # --- <<< STEP 1: Initial -Oz Check using analyze_autophase >>> ---
    initial_oz_think = """<think>
[Initial Baseline Check]
- Goal: Establish baseline performance and instruction count using standard '-Oz'.
- Plan: Call 'analyze_autophase' with just '-Oz' to see its effect compared to the original code.
- The resulting instruction count will be the benchmark.
</think>"""
    initial_oz_tool_call_args = {"filename": filename, "optimization_passes": ["-Oz"]}
    initial_oz_tool_call_content = json.dumps({"name": "analyze_autophase", "arguments": initial_oz_tool_call_args}, separators=(',', ':'))
    initial_oz_tool_call_str = f"<tool_call>\n{initial_oz_tool_call_content}\n</tool_call>"
    initial_assistant_turn_oz = f"<|im_start|>assistant\n{initial_oz_think}\n{initial_oz_tool_call_str}\n<|im_end|>"
    result_parts.append(initial_assistant_turn_oz)

    # Simulate user's response for the -Oz check
    initial_oz_tool_response_dict = {"status": "error", "feature_analysis": "Simulation error for -Oz check"}
    oz_inst_count_from_tool: Optional[int] = None # Store count from this tool response
    try:
        # Simulate tool applying '-Oz' and getting features
        oz_optimized_code = GenerateOptimizedLLCode(original_ll_code, ['-Oz'], llvm_tools_path)
        if oz_optimized_code is None: raise ValueError("GenerateOptimizedLLCode failed for -Oz")
        oz_features = get_autophase_features(oz_optimized_code)
        if oz_features is None: raise ValueError("get_autophase_features failed for -Oz")

        # Analyze change from initial state
        oz_analysis_text = analyze_feature_changes(initial_autophase, oz_features)
        initial_oz_tool_response_dict["status"] = "success"
        initial_oz_tool_response_dict["feature_analysis"] = oz_analysis_text
        oz_inst_count_from_tool = _safe_get_inst_count(oz_features) # Extract count
        if oz_inst_count_from_tool is not None:
            initial_oz_tool_response_dict["current_total_insts"] = oz_inst_count_from_tool
        else:
            # If count extraction fails even if feature generation worked
            initial_oz_tool_response_dict["feature_analysis"] += " (Warning: Could not extract TotalInsts count)"
            initial_oz_tool_response_dict["status"] = "error" # Mark as error if count missing

    except Exception as e:
         error_msg = f"Exception during -Oz analyze_autophase simulation: {e}"
         initial_oz_tool_response_dict["feature_analysis"] = error_msg
         print(f"Warning: {error_msg} for {filename}")

    initial_user_turn_oz = f"<|im_start|>user\n<tool_response>\n{json.dumps(initial_oz_tool_response_dict, indent=2)}\n</tool_response>\n<|im_end|>"
    result_parts.append(initial_user_turn_oz)


    # --- <<< STEP 2: Multi-Round Optimization (Using analyze_autophase) >>> ---

    # --- Pad and Distribute the provided 'good' sequence for simulation guidance ---
    full_pass_sequence_for_sim = list(pass_sequence)
    total_rounds = 5
    # (Padding/Distribution logic remains the same as previous version)
    if len(full_pass_sequence_for_sim) < total_rounds:
        padding_pass = full_pass_sequence_for_sim[-1] if full_pass_sequence_for_sim else "--instcombine"
        while len(full_pass_sequence_for_sim) < total_rounds: full_pass_sequence_for_sim.append(padding_pass)
    n_passes = len(full_pass_sequence_for_sim)
    passes_per_round = max(1, (n_passes + total_rounds - 1) // total_rounds)
    assigned_passes_per_round: List[List[str]] = []
    current_start_index = 0
    for i in range(total_rounds):
        current_end_index = min(current_start_index + passes_per_round, n_passes)
        round_passes = full_pass_sequence_for_sim[current_start_index:current_end_index]
        if not round_passes and current_start_index < n_passes: round_passes = [full_pass_sequence_for_sim[current_start_index]]; current_end_index += 1
        elif not round_passes: round_passes = [full_pass_sequence_for_sim[-1]]
        assigned_passes_per_round.append(round_passes)
        current_start_index = current_end_index
        if current_start_index >= n_passes and i < total_rounds - 1:
             for j in range(i + 1, total_rounds): assigned_passes_per_round.append([full_pass_sequence_for_sim[-1]])
             break

    # --- State Variables for the multi-round part ---
    # Analysis text starts from the *result* of the -Oz check
    previous_analysis_text = initial_oz_tool_response_dict["feature_analysis"]
    previous_round_passes_added: List[str] = ['-Oz'] # conceptually, the first "previous" step was Oz
    # Features start from the *original* code before the first optimization round (after Oz check)
    current_features = initial_autophase
    current_inst_count_sequence = _safe_get_inst_count(initial_autophase)
    cumulative_passes_agent: List[str] = [] # Track passes agent adds *during the 5 rounds*
    final_round_tool_response: Optional[Dict] = None # Store the *entire* tool response dict from the last round

    # --- Loop Through Optimization Rounds 1 to 5 ---
    for round_idx in range(total_rounds):
        round_num_display = round_idx + 1
        current_round_new_passes_agent = assigned_passes_per_round[round_idx]
        cumulative_passes_agent.extend(current_round_new_passes_agent)
        all_passes_agent_so_far = list(cumulative_passes_agent)

        # --- 1. Generate Thinking ---
        thinking_lines = ["<think>"]
        thinking_lines.append(f"[Optimization Round {round_num_display}/{total_rounds}]")
        current_inst_str = str(current_inst_count_sequence) if current_inst_count_sequence is not None else "N/A"
        oz_count_str = str(oz_inst_count_from_tool) if oz_inst_count_from_tool is not None else "N/A"

        if round_idx == 0:
            thinking_lines.append(f"- State: After checking -Oz baseline (Result: {oz_count_str} instructions).")
            thinking_lines.append(f"- Current Code InstCount (Original): {current_inst_str}.") # Still original before this round
            thinking_lines.append("- Goal: Start building a custom sequence to try and beat the -Oz baseline.")
            thinking_lines.append(f"- Plan (Round 1): Apply initial custom pass(es): {current_round_new_passes_agent}")
            for p in current_round_new_passes_agent: thinking_lines.append(f"  - {p}: {PASS_DESCRIPTIONS.get(p, DEFAULT_PASS_DESC)}")
            thinking_lines.append("- Cumulative: Tool call will use these passes.")
        else:
            thinking_lines.append(f"- Recap: Round {round_idx} added passes: {previous_round_passes_added}.")
            thinking_lines.append(f"- Result (Analysis after Round {round_idx}): {previous_analysis_text}")
            thinking_lines.append(f"- Current InstCount (Sequence): {current_inst_str}.")
            thinking_lines.append(f"- Baseline InstCount (-Oz): {oz_count_str}.")
            thinking_lines.append("- Goal: Continue building the optimization sequence.")
            reasoning = "Proceeding to the next optimization step." # Basic reasoning simulation
            if "decreased" in previous_analysis_text: reasoning = "Improvement seen. Adding next passes."
            elif "increased" in previous_analysis_text: reasoning = "Regression seen. Adding next planned passes."
            elif "unchanged" in previous_analysis_text: reasoning = "No change. Trying next passes."
            elif "Error" in previous_analysis_text or "error" in previous_analysis_text: reasoning = "Previous step had issues. Attempting next passes."
            thinking_lines.append(f"- Plan (Round {round_num_display}): {reasoning} Add passes: {current_round_new_passes_agent}")
            for p in current_round_new_passes_agent: thinking_lines.append(f"  - {p}: {PASS_DESCRIPTIONS.get(p, DEFAULT_PASS_DESC)}")
            thinking_lines.append(f"- Cumulative: Tool call uses all {len(all_passes_agent_so_far)} passes accumulated in this sequence.")

        thinking_lines.append("\nTool call analyzes the effect of applying the *cumulative* sequence generated so far (compared to previous round's state).")
        thinking_lines.append("</think>")
        thinking_content = "\n".join(thinking_lines)

        # --- 2. Generate Tool Call (Using analyze_autophase) ---
        tool_call_args = {"filename": filename, "optimization_passes": all_passes_agent_so_far}
        tool_call_content = json.dumps({"name": "analyze_autophase", "arguments": tool_call_args}, separators=(',', ':'))
        tool_call_str = f"<tool_call>\n{tool_call_content}\n</tool_call>"

        # --- 3. Simulate Tool Execution & Prepare Response ---
        tool_response_content_dict = {"status": "error", "feature_analysis": "Simulation error before execution"}
        feature_analysis_this_round = "Error: Simulation step failed"
        optimized_features_this_round_sim = None
        try:
            optimized_code_this_round_sim = GenerateOptimizedLLCode(original_ll_code, all_passes_agent_so_far, llvm_tools_path)
            if optimized_code_this_round_sim is None: raise ValueError(f"GenerateOptimizedLLCode failed for sequence: {all_passes_agent_so_far}")

            optimized_features_this_round_sim = get_autophase_features(optimized_code_this_round_sim)
            if optimized_features_this_round_sim is None: raise ValueError("get_autophase_features failed")

            # Compare features from *this* round simulation vs features from *previous* round state
            feature_analysis_this_round = analyze_feature_changes(current_features, optimized_features_this_round_sim)
            tool_response_content_dict["status"] = "success"
            tool_response_content_dict["feature_analysis"] = feature_analysis_this_round

            new_inst_count_sim = _safe_get_inst_count(optimized_features_this_round_sim)
            if new_inst_count_sim is not None:
                 tool_response_content_dict["current_total_insts"] = new_inst_count_sim
                 current_inst_count_sequence = new_inst_count_sim # Update sequence count state
            else:
                 feature_analysis_this_round += " (Warning: Could not extract TotalInsts count from tool result)"
                 tool_response_content_dict["feature_analysis"] = feature_analysis_this_round
                 # Mark status as error if essential count is missing
                 tool_response_content_dict["status"] = "error"

        except Exception as e:
            error_message = f"Error during simulation round {round_num_display}: {str(e)}"
            print(f"Error details for {filename}, round {round_num_display}: {e}")
            feature_analysis_this_round = error_message
            tool_response_content_dict["feature_analysis"] = error_message
            tool_response_content_dict["status"] = "error"
            optimized_features_this_round_sim = None

        # Store the *entire response dictionary* for the last round
        if round_idx == total_rounds - 1:
            final_round_tool_response = tool_response_content_dict

        # --- Update State for NEXT Iteration ---
        previous_analysis_text = feature_analysis_this_round
        previous_round_passes_added = current_round_new_passes_agent
        if optimized_features_this_round_sim and tool_response_content_dict["status"] == "success":
            # Only update features if the round was successful AND features were obtained
            current_features = optimized_features_this_round_sim

        # --- Assemble Turn Strings ---
        assistant_turn = f"<|im_start|>assistant\n{thinking_content}\n{tool_call_str}\n<|im_end|>"
        user_turn = f"<|im_start|>user\n<tool_response>\n{json.dumps(tool_response_content_dict, indent=2)}\n</tool_response>\n<|im_end|>"
        result_parts.append(assistant_turn)
        result_parts.append(user_turn)


    # --- <<< STEP 3: Final Assistant Turn - Comparison and Answer >>> ---
    final_answer_sequence = cumulative_passes_agent # Default: use the sequence built over 5 rounds
    final_decision_reasoning = ""

    # Get the final sequence instruction count from the LAST tool response
    final_sequence_inst_count_from_tool: Optional[int] = None
    if final_round_tool_response and final_round_tool_response.get("status") == "success":
        final_sequence_inst_count_from_tool = final_round_tool_response.get("current_total_insts")
        # If key exists but value is not int (e.g., None), _safe_get_inst_count handles it too
        if final_sequence_inst_count_from_tool is None:
             # Attempt to re-parse from features if key was missing but analysis seemed ok
             # This part is less likely needed if response dict is built correctly
             pass # For simplicity, rely on current_total_insts key

    final_think_lines = ["<think>", "[Final Decision]"]
    final_think_lines.append(f"- Completed {total_rounds} optimization rounds.")
    seq_count_str = str(final_sequence_inst_count_from_tool) if final_sequence_inst_count_from_tool is not None else "N/A (Error or missing)"
    oz_count_str = str(oz_inst_count_from_tool) if oz_inst_count_from_tool is not None else "N/A (Error or missing)"
    final_think_lines.append(f"- Final InstCount (Result of {total_rounds}-Round Sequence): {seq_count_str}.")
    final_think_lines.append(f"- Baseline InstCount (Result of Initial -Oz): {oz_count_str}.")

    comparison_possible = (oz_inst_count_from_tool is not None and final_sequence_inst_count_from_tool is not None)

    if comparison_possible:
        if oz_inst_count_from_tool < final_sequence_inst_count_from_tool:
            final_answer_sequence = ['-Oz']
            final_decision_reasoning = f"- Comparison: Initial '-Oz' ({oz_count_str}) resulted in fewer instructions than the final multi-round sequence ({seq_count_str})."
            final_think_lines.append(final_decision_reasoning)
            final_think_lines.append("- Conclusion: Selecting '-Oz' as the final answer.")
        else:
            final_decision_reasoning = f"- Comparison: The multi-round sequence ({seq_count_str}) resulted in fewer or equal instructions compared to initial '-Oz' ({oz_count_str})."
            final_think_lines.append(final_decision_reasoning)
            final_think_lines.append("- Conclusion: Selecting the multi-round sequence as the final answer.")
            # final_answer_sequence already defaults to cumulative_passes_agent
            if not final_answer_sequence: # Ensure it's not empty if chosen
                 final_answer_sequence = ["--instcombine"] # Fallback
    # Handle cases where comparison isn't possible
    elif oz_inst_count_from_tool is None:
         final_decision_reasoning = "- Comparison Failed: Could not determine baseline count from the initial -Oz tool call."
         final_think_lines.append(final_decision_reasoning)
         final_think_lines.append("- Conclusion: Using the result of the multi-round sequence attempt.")
         if not final_answer_sequence: final_answer_sequence = ["--instcombine"] # Fallback
    elif final_sequence_inst_count_from_tool is None:
         final_decision_reasoning = "- Comparison Failed: Could not determine final count from the multi-round sequence's last step."
         final_think_lines.append(final_decision_reasoning)
         if oz_inst_count_from_tool is not None:
             final_think_lines.append("- Conclusion: Defaulting to '-Oz' as its baseline count is known.")
             final_answer_sequence = ['-Oz']
         else:
             final_think_lines.append("- Conclusion: Both initial -Oz and final sequence steps failed to provide counts. Cannot determine the best option.")
             final_answer_sequence = ["-Oz"] # Default to Oz maybe? Or keep sequence? Let's default to Oz here.
    else:
        final_decision_reasoning = "- Comparison Status Uncertain."
        final_think_lines.append(final_decision_reasoning)
        final_think_lines.append("- Conclusion: Defaulting to the multi-round sequence attempt.")
        if not final_answer_sequence: final_answer_sequence = ["--instcombine"] # Fallback

    final_think_lines.append("</think>")
    final_thinking_content = "\n".join(final_think_lines)

    # Assemble final assistant response
    final_assistant_answer_parts = [
        "<|im_start|>assistant",
        final_thinking_content,
        f"<answer>\n{final_answer_sequence}\n</answer>", # Use the determined sequence
        "<|im_end|>"
    ]
    result_parts.append("\n".join(final_assistant_answer_parts))

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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
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
"""
            
            # 生成思考过程和工具调用
            # full_process = generate_thinking_process(filename, autophase_embedding, pass_sequence)
            full_process = [
                "<|im_start|>assistant",
                f"<answer>\n{pass_sequence}\n</answer>", # Use the determined sequence
                "<|im_end|>"
            ]
            
            full_process = "\n".join(full_process)

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
            print(f"Error processing row for {filename}: {e}")
            continue
    
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