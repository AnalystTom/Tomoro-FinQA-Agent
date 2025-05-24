import json
import httpx
import asyncio
import os
import re
import argparse
from typing import List, Dict, Any, Optional, Union
import math
from collections import defaultdict
# import statistics # Not used in this version, can be removed if not needed elsewhere

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROCESS_QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/qa/process-query"
DEFAULT_EVALUATION_FILE_PATH = "evaluation/datasets/qa_eval_dataset.json"

# --- Predefined Paper Benchmarks ---
PAPER_BENCHMARKS = {
    "FinQANet_Roberta_Large": {"EA": 0.6890, "PA": 0.6824, "description": "FinQANet (RoBERTa-large)"},
    "FinQANet_Gold_Roberta_Large": {"EA": 0.7732, "PA": 0.7646, "description": "FinQANet-Gold (RoBERTa-large)"},
    "Human_Expert": {"EA": 0.8944, "PA": 0.8634, "description": "Human Expert"},
    "General_Crowd": {"EA": 0.4690, "PA": 0.4552, "description": "General Crowd"},
}

def get_context_for_evaluation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts context fields from an evaluation item."""
    # item_id = item.get("id", "unknown_id") # Not used here, but good for debugging
    return {
        "pre_text": item.get("pre_text"),
        "post_text": item.get("post_text"),
        "table_ori": item.get("table_ori")
    }

def extract_numerical_value(answer: Optional[Union[str, float, int]]) -> (Optional[float], bool):
    """
    Extracts a numerical value from a string and indicates if it was a percentage.
    Handles percentages, commas, LaTeX-like math blocks for simple numbers, and common textual phrases.
    Returns a tuple: (numerical_value, is_percentage_flag).
    The numerical_value for percentages is its magnitude (e.g., 50.0 for "50%").
    """
    if answer is None: return None, False
    text = str(answer).strip()
    
    is_percentage = "%" in text or "\\%" in text

    # Basic LaTeX cleanup for numbers like \(-32.82\%\) -> -32.82%
    text = re.sub(r'\\\(', '', text)
    text = re.sub(r'\\\)', '', text)

    words_to_remove = [
        "approximately", "about", "around", "is approximately", "was approximately",
        "is about", "was about", "is around", "was around", "is", "was",
        "the result is", "result:", "total value is", "total is", "value is", "amount is",
        "the change is", "change is", "decrease of", "increase of", "an increase of", "a decrease of"
    ]
    
    cleaned_text = text.lower()
    for word in words_to_remove:
        # Use regex to remove whole words/phrases to avoid partial replacements
        cleaned_text = re.sub(r'\b' + re.escape(word) + r'\b', '', cleaned_text, flags=re.IGNORECASE)
    
    cleaned_text = cleaned_text.replace('$', '').replace('€', '').replace('£', '') # Currency symbols
    cleaned_text = cleaned_text.replace(',', '') # Thousands separators
    cleaned_text = cleaned_text.replace('(', '-').replace(')', '') # Parentheses for negatives
    cleaned_text = cleaned_text.replace('\\', '') # Remove stray backslashes
    cleaned_text = cleaned_text.replace('%', '') # Remove percentage sign for number parsing
    cleaned_text = cleaned_text.strip().strip('.').strip()

    potential_numbers = list(re.finditer(r"([-+]?\d*\.?\d+)", cleaned_text))

    if not potential_numbers:
        # print(f"DEBUG extract_numerical_value: No number pattern in '{text}' (cleaned: '{cleaned_text}')")
        return None, is_percentage # Return original percentage flag even if no number found

    num_str = potential_numbers[-1].group(1)

    try:
        val = float(num_str)
        # print(f"DEBUG extract_numerical_value: Extracted '{num_str}', original contained '%': {is_percentage}. Final float value: {val} from original: '{answer}'")
        return val, is_percentage
    except ValueError:
        # print(f"DEBUG extract_numerical_value: ValueError converting '{num_str}'. Original: '{answer}'")
        return None, is_percentage

def get_string_precision(value_str: str) -> int:
    """Helper to get the number of decimal places from a string representation of a number."""
    if '.' in value_str:
        return len(value_str.split('.')[-1])
    return 0

def compare_answers(
    item_id: str, # For debug logging
    agent_answer_str: Optional[str], 
    ground_truth_str: Optional[str], 
    agent_raw_tool_result_str: Optional[str] = None, # Direct output from calculator
    default_rel_tol=0.015, # 1.5% relative tolerance for general numbers
    default_abs_tol=0.0005, # Absolute tolerance for general numbers (esp. near zero)
    perc_rel_tol=0.01, # 1% relative tolerance for percentages when compared as decimals
    perc_abs_tol=0.0005 # 0.05 percentage points absolute tolerance when compared as decimals
    ) -> bool:
    """
    Compares agent's answer with the ground truth, implementing robust checks.
    """
    # print(f"DEBUG [{item_id}] compare_answers INPUT - Agent: '{agent_answer_str}', GT: '{ground_truth_str}', ToolResult: '{agent_raw_tool_result_str}'")

    if agent_answer_str is None and ground_truth_str is None: return True
    if agent_answer_str is None or ground_truth_str is None: return False

    # Extract numerical values and percentage flags
    # Value will be percentage magnitude (e.g., 50.0 for "50%")
    agent_num_mag, agent_is_perc = extract_numerical_value(agent_answer_str)
    gt_num_mag, gt_is_perc = extract_numerical_value(ground_truth_str)
    
    agent_raw_calc_mag = None
    if agent_raw_tool_result_str:
        # Assume tool result is already a direct numerical string
        try:
            agent_raw_calc_mag = float(agent_raw_tool_result_str)
        except (ValueError, TypeError):
            pass # Could not parse tool result

    # If agent's final answer couldn't be parsed, but tool result is available, use tool result
    if agent_num_mag is None and agent_raw_calc_mag is not None:
        agent_num_mag = agent_raw_calc_mag
        # If the question context implies a percentage, the raw tool result should also be treated as such
        # This assumes agent_is_perc was correctly determined from agent_answer_str or context
        # If agent_answer_str was pure text and agent_is_perc is False, but gt_is_perc is True,
        # it implies the agent might have missed the percentage aspect.
        # For now, if agent_answer_str implies percentage, raw_calc is also percentage magnitude.

    # print(f"DEBUG [{item_id}] Extracted - AgentNum: {agent_num_mag} (isPerc: {agent_is_perc}), GTNum: {gt_num_mag} (isPerc: {gt_is_perc}), AgentRawCalc: {agent_raw_calc_mag}")

    # Scenario 1: Both are numbers and both are identified as percentages
    if agent_num_mag is not None and gt_num_mag is not None and agent_is_perc and gt_is_perc:
        # Compare as decimals (val/100) using math.isclose
        agent_decimal = agent_num_mag / 100.0
        gt_decimal = gt_num_mag / 100.0
        if math.isclose(agent_decimal, gt_decimal, rel_tol=perc_rel_tol, abs_tol=perc_abs_tol):
            # print(f"DEBUG [{item_id}]: PASS - Percentage (decimal form) isclose. AgentDec: {agent_decimal}, GTDec: {gt_decimal}")
            return True

        # --- Approach 1: Specific Handling for Integer-like GT Percentage vs. More Precise Agent Percentage ---
        # This uses percentage magnitudes (e.g., -32.82 vs -32.0)
        # Use the most precise agent calculation available (raw tool result if possible)
        precise_agent_perc_mag = agent_raw_calc_mag if agent_raw_calc_mag is not None else agent_num_mag

        # Check if original GT string for the number part was integer-like
        # e.g., "-32%" -> numeric part "-32" has precision 0
        # e.g., "14.1%" -> numeric part "14.1" has precision 1
        original_gt_numeric_str_match = re.search(r"([-+]?\d*\.?\d+)", str(ground_truth_str).replace(',', ''))
        gt_str_precision = 0
        if original_gt_numeric_str_match:
            gt_str_precision = get_string_precision(original_gt_numeric_str_match.group(1))

        if gt_str_precision == 0: # GT is an integer percentage like "-32%"
            # Rule A: Agent's precise value rounded to 0 decimal places is within 1 of GT's integer value.
            # Example: GT is -32. Agent calculates -32.8197 (rounds to -33). abs(-33 - (-32)) == 1.
            rounded_agent_to_gt_precision = round(precise_agent_perc_mag, 0)
            if abs(rounded_agent_to_gt_precision - gt_num_mag) <= 1.0:
                # And ensure the original unrounded difference wasn't excessively large.
                if abs(precise_agent_perc_mag - gt_num_mag) < 1.5: # Max 1.5 pp original difference
                    print(f"DEBUG [{item_id}]: PASS (Rule A) - Agent precise ({precise_agent_perc_mag}) rounded to GT precision ({rounded_agent_to_gt_precision}) is within 1 of integer GT ({gt_num_mag}). Original diff: {abs(precise_agent_perc_mag - gt_num_mag)}")
                    return True
            
            # Rule B: Absolute difference of percentage magnitudes is within a specific tolerance for this case.
            # Example: GT is -32. Agent -32.8197. Diff is 0.8197.
            INTEGER_GT_SPECIFIC_PERC_EPSILON = 0.85 # Allow up to 0.85 percentage point difference
            if abs(precise_agent_perc_mag - gt_num_mag) < INTEGER_GT_SPECIFIC_PERC_EPSILON:
                print(f"DEBUG [{item_id}]: PASS (Rule B) - Integer GT Specific Epsilon. Agent Precise: {precise_agent_perc_mag}, GT: {gt_num_mag}, Diff: {abs(precise_agent_perc_mag - gt_num_mag)}")
                return True
        
        # If it's a percentage comparison and still hasn't passed, it fails numerically for percentages
        # print(f"DEBUG [{item_id}]: FAIL - Percentage numerical checks. AgentMag: {agent_num_mag}, GTMag: {gt_num_mag}")
        # return False # Let it fall through to generic number or string comparison if needed

    # Scenario 2: Both are numbers, but not necessarily both percentages (or percentage check failed)
    if agent_num_mag is not None and gt_num_mag is not None:
        if math.isclose(agent_num_mag, gt_num_mag, rel_tol=default_rel_tol, abs_tol=default_abs_tol):
            # print(f"DEBUG [{item_id}]: PASS - Generic numerical isclose. Agent: {agent_num_mag}, GT: {gt_num_mag}")
            return True
        # else:
            # print(f"DEBUG [{item_id}]: FAIL - Generic numerical isclose. Agent: {agent_num_mag}, GT: {gt_num_mag}, Diff: {abs(agent_num_mag-gt_num_mag)}")
            # return False # Let it fall to string comparison

    # Scenario 3: Fallback to normalized string comparison
    # (Useful for "Yes"/"No" or if numerical parsing/comparison failed but strings might match)
    norm_agent = str(agent_answer_str).strip().lower()
    norm_gt = str(ground_truth_str).strip().lower()
    
    # Basic normalization for string comparison (can be expanded)
    common_noise = ['$', ',', '%', '\\(', '\\)', 'approximately', 'about', '.', 'the final answer is']
    for noise in common_noise:
        norm_agent = norm_agent.replace(noise, '').strip()
        norm_gt = norm_gt.replace(noise, '').strip()

    # print(f"DEBUG [{item_id}]: Fallback String Compare - NormAgent: '{norm_agent}', NormGT: '{norm_gt}'")
    if norm_agent == norm_gt:
        return True
        
    print(f"DEBUG [{item_id}]: FAIL - All checks. AgentNum: {agent_num_mag}, GTNum: {gt_num_mag}, NormAgent: '{norm_agent}', NormGT: '{norm_gt}'")
    return False


def _extract_dialogue_turns_from_eval_item(eval_item_top_level: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Kept as is from your script
    evaluation_details = eval_item_top_level.get("evaluation_details", {})
    dialogue_turns = evaluation_details.get("dialogue_turns", [])
    if dialogue_turns: return dialogue_turns
    if "annotation" in eval_item_top_level:
        annotation_data = eval_item_top_level.get("annotation", {})
        dialogue_break = (annotation_data.get("dialogue_break") or 
                          annotation_data.get("dialogue_break_ori") or [])
        programs = (annotation_data.get("turn_program") or 
                    annotation_data.get("turn_program_ori") or [])
        exe_answers = annotation_data.get("exe_ans_list") or []
        min_len = min(len(dialogue_break), len(programs), len(exe_answers))
        for i in range(min_len):
            dialogue_turns.append({
                "turn_question": dialogue_break[i],
                "expected_turn_program_or_value": programs[i],
                "expected_turn_numeric_answer": exe_answers[i],
            })
    return dialogue_turns

def parse_program_string(program_str: str) -> List[Dict[str, Any]]:
    # Kept as is from your script
    steps = []
    if not program_str: return steps
    operations = program_str.strip().split('),') 
    for i, op_str in enumerate(operations):
        op_str = op_str.strip()
        if not op_str: continue
        if i < len(operations) - 1:
            op_str += ')'
        match = re.match(r"(\w+)\((.*)\)?", op_str)
        if match:
            op_name = match.group(1)
            args_str = match.group(2)
            args_str_cleaned = args_str.replace('#', 'REF_') 
            args_list = [arg.strip() for arg in args_str_cleaned.split(',')]
            parsed_args = []
            for arg in args_list:
                try: 
                    parsed_args.append(float(arg))
                except ValueError: 
                    parsed_args.append(arg)
            steps.append({"op": op_name, "args": parsed_args})
    return steps

def compare_programs(agent_tool_calls: List[Dict[str, Any]], expected_program_str: Optional[str]) -> bool:
    # Kept as is from your script, but ensure it aligns with how you want to evaluate programs
    if not expected_program_str: return False 
    expected_steps = parse_program_string(expected_program_str)
    agent_calc_steps = []
    for call in agent_tool_calls:
        if call.get("tool_name") == "calculator":
            math_expr = call.get("tool_args", {}).get("math_expression", "")
            agent_numbers = set(re.findall(r"[-+]?\d*\.?\d+", math_expr.replace(',',''))) # Added comma replace here
            agent_calc_steps.append({"op": "calculator", "numbers_used": agent_numbers, "expr": math_expr})
    if not expected_steps and not agent_calc_steps: return True
    if not expected_steps or not agent_calc_steps: return False
    if len(agent_calc_steps) > 0 and len(expected_steps) > 0:
        last_agent_calc_numbers = agent_calc_steps[-1]["numbers_used"]
        last_expected_step_args_numeric = set()
        if expected_steps[-1].get("args"):
            last_expected_step_args_numeric = {
                str(arg) for arg in expected_steps[-1]["args"] 
                if isinstance(arg, (int, float)) or 
                   (isinstance(arg, str) and arg.replace('.','',1).replace('-','',1).isdigit())
            }
        if last_agent_calc_numbers and last_expected_step_args_numeric:
            intersection_len = len(last_agent_calc_numbers.intersection(last_expected_step_args_numeric))
            union_len = len(last_agent_calc_numbers.union(last_expected_step_args_numeric))
            if union_len > 0 and (intersection_len / union_len) > 0.5:
                return True
    expects_calculation_ops = ["subtract", "divide", "add", "multiply", "negate"]
    expects_calculation = any(op.get("op") in expects_calculation_ops for op in expected_steps)
    if expects_calculation and agent_calc_steps: return True
    if not expects_calculation and not agent_calc_steps: return True
    return False


async def run_single_shot_evaluation_for_item(
    client: httpx.AsyncClient,
    eval_item: Dict[str, Any]
) -> Dict[str, Any]:
    item_id = eval_item.get("id", "unknown_id")
    print(f"\n--- Processing Single-Shot for ID: {item_id} ---")

    qa_data_level = eval_item.get("qa", eval_item)
    main_question = eval_item.get("question", qa_data_level.get("question"))
    final_ground_truth_answer = eval_item.get("ground_truth_answer", qa_data_level.get("answer"))

    if not main_question:
        return {
            "id": item_id, "main_question": None, "final_ground_truth": None,
            "final_agent_answer": None, "is_execution_correct": False, "is_program_correct": False,
            "agent_tool_calls": [], "error": "Missing main question",
            "expected_program": eval_item.get("evaluation_details", {}).get("expected_final_program")
        }

    evaluation_details = eval_item.get("evaluation_details", {})
    expected_program = evaluation_details.get("expected_final_program")
    context_data = get_context_for_evaluation_item(eval_item)
    
    payload = {
        "question": main_question,
        "pre_text": context_data.get("pre_text"),
        "post_text": context_data.get("post_text"),
        "table_ori": context_data.get("table_ori"),
        "messages_history": None,
        "item_id": item_id,
        "request_id": f"eval_single_{item_id.replace('/', '_')}" # Ensure request_id is filename-friendly
    }
    
    final_agent_answer: Optional[str] = None
    agent_tool_calls: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    agent_last_calculator_result_str: Optional[str] = None # To store the raw calculator output

    print(f"Question: {main_question}")
    try:
        response = await client.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=120.0)
        response.raise_for_status()
        response_data = response.json()
        
        final_agent_answer = response_data.get("answer")
        agent_tool_calls = response_data.get("tool_calls_log", [])

        print(f"  Agent Answer: {str(final_agent_answer)[:100]}...")
        if agent_tool_calls:
            print(f"  Agent Tool Calls ({len(agent_tool_calls)}):")
            for tc_log in agent_tool_calls:
                tool_name = tc_log.get('tool_name')
                tool_result_str = str(tc_log.get('tool_result'))
                print(f"    - Tool: {tool_name}, Args: {tc_log.get('tool_args')}, Result: {tool_result_str[:50]}..., Error: {tc_log.get('error')}")
                if tool_name == "calculator" and not tc_log.get('error'):
                    agent_last_calculator_result_str = tool_result_str # Get the last successful calculator result
    except httpx.RequestError as e:
        error_message = f"HTTP Request Error for item {item_id}: {e}"
        print(error_message)
    except httpx.HTTPStatusError as e:
        error_message = f"HTTP Status Error for item {item_id}: {e.response.status_code} - {e.response.text}"
        print(error_message)
    except Exception as e:
        error_message = f"Generic Error processing item {item_id}: {e}"
        print(error_message)
    
    is_execution_correct = compare_answers(
        item_id, # Pass item_id for debugging
        final_agent_answer, 
        final_ground_truth_answer,
        agent_last_calculator_result_str # Pass the raw tool result
    )
    is_program_correct = compare_programs(agent_tool_calls, expected_program) 
    
    print(f"Final Ground Truth: {final_ground_truth_answer}")
    print(f"Final Agent Answer: {final_agent_answer}")
    print(f"Is Execution Correct: {is_execution_correct}")
    print(f"Is Program Correct (Heuristic): {is_program_correct}")

    return_dict = {
        "id": item_id, "main_question": main_question, 
        "final_ground_truth": final_ground_truth_answer, 
        "final_agent_answer": final_agent_answer, 
        "is_execution_correct": is_execution_correct,
        "is_program_correct": is_program_correct, 
        "agent_tool_calls": agent_tool_calls, 
        "agent_last_calculator_result_str_for_eval": agent_last_calculator_result_str, # Log what was passed
        "expected_program": expected_program,
        "dialogue_turns_reference": _extract_dialogue_turns_from_eval_item(eval_item), 
        "error": error_message 
    }
    for key in ["retrieval_precision", "retrieval_recall", "retrieval_f1"]:
        return_dict.pop(key, None)
    return return_dict

def print_overall_metrics(
    full_results: List[Dict[str, Any]], 
    ea_paper_to_compare: Optional[float],
    pa_paper_to_compare: Optional[float],
    benchmark_source_name: str
):
    # Kept as is from your script
    total_items_processed_successfully = 0
    C_exec = 0 
    C_prog = 0 
    for res in full_results:
        if res.get("error") is None: 
            total_items_processed_successfully +=1
            if res.get("is_execution_correct"): C_exec +=1
            if res.get("is_program_correct"): C_prog +=1
    print(f"\n--- Overall Evaluation Metrics ({total_items_processed_successfully} items processed without critical error) ---")
    EA_app = 0.0
    PA_app = 0.0
    if total_items_processed_successfully > 0:
        EA_app = (C_exec / total_items_processed_successfully)
        PA_app = (C_prog / total_items_processed_successfully)
        print(f"Execution Accuracy (EA_app): {EA_app*100:.2f}% ({C_exec}/{total_items_processed_successfully})")
        print(f"Program Accuracy (PA_app - Heuristic): {PA_app*100:.2f}% ({C_prog}/{total_items_processed_successfully})")
    else:
        print("No items were processed without critical errors to calculate summary metrics.")
    print(f"\n--- Paper vs. App Gaps (Comparing against: {benchmark_source_name}) ---")
    if ea_paper_to_compare is not None:
        if total_items_processed_successfully > 0:
            delta_EA = EA_app - ea_paper_to_compare
            relative_imp_EA = (delta_EA / ea_paper_to_compare) * 100 if ea_paper_to_compare != 0 else float('inf')
            print(f"EA_paper ({benchmark_source_name}): {ea_paper_to_compare*100:.2f}%")
            print(f"Delta EA (EA_app - EA_paper): {delta_EA*100:.2f} percentage points")
            print(f"Relative Improvement EA: {relative_imp_EA:.2f}%")
        else:
            print(f"EA_paper ({benchmark_source_name}): {ea_paper_to_compare*100:.2f}% (No app results for EA, cannot calculate gap)")
    else:
        print("EA_paper not provided or selected, cannot calculate EA gap.")
    if pa_paper_to_compare is not None:
        if total_items_processed_successfully > 0: 
            delta_PA = PA_app - pa_paper_to_compare
            relative_imp_PA = (delta_PA / pa_paper_to_compare) * 100 if pa_paper_to_compare != 0 else float('inf')
            print(f"PA_paper ({benchmark_source_name}): {pa_paper_to_compare*100:.2f}%")
            print(f"Delta PA (PA_app - PA_paper): {delta_PA*100:.2f} percentage points")
            print(f"Relative Improvement PA: {relative_imp_PA:.2f}%")
        else:
            print(f"PA_paper ({benchmark_source_name}): {pa_paper_to_compare*100:.2f}% (No app results for PA, cannot calculate PA gap)")
    else:
        print("PA_paper not provided or selected, cannot calculate PA gap.")

async def main(cli_args: argparse.Namespace): 
    # Kept as is from your script
    current_eval_file_path = cli_args.eval_file 
    try:
        with open(current_eval_file_path, 'r', encoding='utf-8') as f:
            evaluation_dataset = json.load(f) 
        if not isinstance(evaluation_dataset, list):
            print(f"Error: Evaluation data in {current_eval_file_path} is not a JSON list.")
            return
    except FileNotFoundError:
        print(f"Error: Evaluation file not found at {current_eval_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {current_eval_file_path}")
        return
    print(f"Loaded {len(evaluation_dataset)} items from {current_eval_file_path}")
    num_to_evaluate = cli_args.num_items
    if num_to_evaluate is None or num_to_evaluate <= 0: 
        print(f"Processing all {len(evaluation_dataset)} items...")
        num_to_evaluate = len(evaluation_dataset)
    else:
        num_to_evaluate = min(num_to_evaluate, len(evaluation_dataset))
        print(f"Processing the first {num_to_evaluate} items...")
    effective_ea_paper = cli_args.ea_paper
    effective_pa_paper = cli_args.pa_paper
    benchmark_display_name = "Custom/Manual" 
    if cli_args.benchmark:
        if cli_args.benchmark in PAPER_BENCHMARKS:
            benchmark_data = PAPER_BENCHMARKS[cli_args.benchmark]
            benchmark_display_name = benchmark_data["description"]
            if cli_args.ea_paper is None:
                effective_ea_paper = benchmark_data["EA"]
            if cli_args.pa_paper is None:
                effective_pa_paper = benchmark_data["PA"]
            if cli_args.ea_paper is not None and cli_args.ea_paper != benchmark_data["EA"]:
                benchmark_display_name += " (EA overridden)"
            if cli_args.pa_paper is not None and cli_args.pa_paper != benchmark_data["PA"]:
                benchmark_display_name += " (PA overridden)"
        else:
            print(f"Warning: Benchmark '{cli_args.benchmark}' not found. Available: {list(PAPER_BENCHMARKS.keys())}")
            benchmark_display_name = f"Unknown Benchmark ({cli_args.benchmark}) / Custom"
    elif cli_args.ea_paper is not None or cli_args.pa_paper is not None:
        benchmark_display_name = "Manually Specified Paper Values"
    full_results: List[Dict[str, Any]] = [] 
    async with httpx.AsyncClient() as client:
        for i, item_from_dataset in enumerate(evaluation_dataset):
            if i >= num_to_evaluate:
                print(f"\nReached limit of {num_to_evaluate} items. Stopping evaluation.")
                break 
            item_result = await run_single_shot_evaluation_for_item(client, item_from_dataset) 
            full_results.append(item_result)
    if full_results:
        output_filename = f"single_shot_evaluation_summary_first_{len(full_results)}.json" 
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2)
        print(f"\nSingle-shot evaluation summary for {len(full_results)} items saved to {output_filename}")
    else:
        print("\nNo items were evaluated, so no summary file was generated.")
    print_overall_metrics(full_results, effective_ea_paper, effective_pa_paper, benchmark_display_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-shot QA evaluation against the API and compare with paper benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n", "--num-items", type=int, default=None, 
        help="Number of items from the evaluation dataset to process. Processes all if not specified or <= 0."
    )
    parser.add_argument(
        "--eval-file", type=str, default=DEFAULT_EVALUATION_FILE_PATH, 
        help="Path to the evaluation dataset JSON file."
    )
    parser.add_argument(
        "--benchmark", type=str, default=None, choices=list(PAPER_BENCHMARKS.keys()),
        help=(
            "Name of a predefined benchmark to compare against. "
            "Values from this benchmark (EA, PA) will be used unless explicitly overridden by "
            "--ea-paper or --pa-paper arguments. "
            f"Available benchmarks: {', '.join(PAPER_BENCHMARKS.keys())}"
        )
    )
    parser.add_argument(
        "--ea-paper", type=float, default=None, 
        help="Paper's reported Execution Accuracy (e.g., 0.689 for 68.90%%). Overrides benchmark's EA if --benchmark is also used."
    )
    parser.add_argument(
        "--pa-paper", type=float, default=None, 
        help="Paper's reported Program Accuracy (e.g., 0.6824 for 68.24%%). Overrides benchmark's PA if --benchmark is also used."
    )
    
    cli_args = parser.parse_args()
    
    print("Reminder: Ensure your FastAPI application is running and accessible at API_BASE_URL.")
    print(f"Using evaluation file: {cli_args.eval_file}")
    print("Running in SINGLE-SHOT evaluation mode.")
    if cli_args.benchmark:
        print(f"Selected benchmark for comparison: {PAPER_BENCHMARKS[cli_args.benchmark]['description']}")
        if cli_args.ea_paper is not None:
            print(f"  Using command-line override for EA_paper: {cli_args.ea_paper*100:.2f}%")
        if cli_args.pa_paper is not None:
            print(f"  Using command-line override for PA_paper: {cli_args.pa_paper*100:.2f}%")
    elif cli_args.ea_paper is not None or cli_args.pa_paper is not None:
        print("Using manually specified EA_paper and/or PA_paper values for comparison.")
    print("Retrieval metrics (Precision, Recall, F1) and related gap calculations are not part of this script's output.")
    
    asyncio.run(main(cli_args))