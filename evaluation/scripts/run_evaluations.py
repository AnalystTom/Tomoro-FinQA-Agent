import json
import httpx # Using httpx for asynchronous HTTP requests
import asyncio
import os
import re
import argparse # Import argparse for CLI arguments
from typing import List, Dict, Any, Optional, Union
import math # For isclose

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") # Adjust if your API runs elsewhere
PROCESS_QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/qa/process-query"
EVALUATION_FILE_PATH = "/home/russki/projects/Tomoro-FinQA-Agent/evaluation/datasets/qa_eval_dataset.json" # IMPORTANT: Update this path

# --- Helper function to get context for an evaluation item ---
def get_context_for_evaluation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves pre_text, post_text, and table_ori directly from the
    evaluation item (which is assumed to be one entry from your train.json).
    """
    item_id = item.get("id", "unknown_id") 
    
    pre_text_data = item.get("pre_text")
    post_text_data = item.get("post_text")
    table_ori_data = item.get("table_ori")

    if pre_text_data is None:
        print(f"Info: 'pre_text' is not present or is null for item_id: {item_id}")
    if post_text_data is None:
        print(f"Info: 'post_text' is not present or is null for item_id: {item_id}")
    if table_ori_data is None:
        print(f"Info: 'table_ori' is not present or is null for item_id: {item_id}. Table-dependent questions might not be answerable by table query tools.")
    
    return {
        "pre_text": pre_text_data,
        "post_text": post_text_data,
        "table_ori": table_ori_data 
    }

def extract_numerical_value(answer: Optional[Union[str, float, int]]) -> Optional[float]:
    """
    Extracts a numerical value from a string.
    Handles percentages, commas, and common currency symbols.
    Returns a float or None.
    """
    if answer is None:
        return None
    
    text = str(answer).strip()
    original_text_for_percent_check = text # Keep original to check for '%' presence reliably

    # General cleanup of common words that might surround numbers
    # This list can be expanded. Be careful not to remove parts of potential numbers.
    words_to_remove = [
        "approximately", "about", "around", "is", "was", "the result is", "result:",
        "total", "value", "amount", "change", "decrease of", "increase of",
        "dollars", "usd", "million", "billion", "thousand", r"\$", "£", "€" 
    ] # Added r"\$" for literal $
    
    cleaned_text = text.lower()
    for word in words_to_remove:
        cleaned_text = cleaned_text.replace(word, "")
    
    # Remove commas from numbers, handle parentheses for negatives
    cleaned_text = cleaned_text.replace(',', '').replace('(', '-').replace(')', '')
    # Remove any remaining non-numeric characters except ., -, and %
    # This regex keeps signed numbers, decimals, and percentage signs
    # It finds the last potential number in the string.
    
    # Try to find a number pattern. This regex is quite greedy.
    # It looks for a number that can be preceded by a sign, contain commas (which we removed),
    # and a decimal point, optionally followed by a percentage sign.
    # We iterate to find the "best" or last number if multiple are present.
    
    potential_numbers = list(re.finditer(r"([-+]?\d*\.?\d+)", cleaned_text))
    
    if not potential_numbers:
        print(f"DEBUG extract_numerical_value: No number found in '{text}' (cleaned: '{cleaned_text}')")
        return None
        
    # Take the number from the last match as it's often the final answer part
    num_str = potential_numbers[-1].group(1)

    try:
        val = float(num_str)
        # Check if the original string (before extensive cleaning for number extraction) contained '%'
        if "%" in original_text_for_percent_check:
            val = val / 100.0
        print(f"DEBUG extract_numerical_value: Extracted '{num_str}', original contained '%': {'%' in original_text_for_percent_check}. Final float value: {val} from original: '{answer}'")
        return val
    except ValueError:
        print(f"DEBUG extract_numerical_value: ValueError converting '{num_str}' to float. Original text: '{answer}', Cleaned for num extraction: '{cleaned_text}'")
        return None

def compare_answers(agent_answer_str: Optional[str], ground_truth_str: Optional[str], rel_tol=1e-2, abs_tol=1e-4) -> bool:
    """
    Compares agent answer with ground truth.
    Prioritizes numerical comparison with tolerance if both can be converted to numbers.
    rel_tol: relative tolerance (e.g., 0.01 for 1%)
    abs_tol: absolute tolerance (e.g., 0.0001)
    """
    print(f"DEBUG compare_answers - Original GT: '{ground_truth_str}', Original Agent: '{agent_answer_str}'")

    if agent_answer_str is None and ground_truth_str is None:
        print("DEBUG compare_answers: Both None -> True")
        return True
    if agent_answer_str is None or ground_truth_str is None:
        print(f"DEBUG compare_answers: One is None. GT: '{ground_truth_str}', Agent: '{agent_answer_str}' -> False")
        return False

    agent_num = extract_numerical_value(agent_answer_str)
    gt_num = extract_numerical_value(ground_truth_str)

    print(f"DEBUG compare_answers - Extracted Numerical: GT_Num='{gt_num}', Agent_Num='{agent_num}'")

    if agent_num is not None and gt_num is not None:
        # Both are numbers, compare them with tolerance
        is_close = math.isclose(agent_num, gt_num, rel_tol=rel_tol, abs_tol=abs_tol)
        print(f"DEBUG compare_answers - Numerical Comparison: math.isclose({agent_num}, {gt_num}, rel_tol={rel_tol}, abs_tol={abs_tol}) -> {is_close}")
        return is_close
    else:
        # At least one is not a number, fallback to normalized string comparison
        # For string comparison, we still want a basic normalization
        norm_agent_ans_for_str_comp = str(agent_answer_str).strip().lower().replace('%','').replace('$','').replace(',','')
        norm_gt_ans_for_str_comp = str(ground_truth_str).strip().lower().replace('%','').replace('$','').replace(',','')
        
        # Remove common phrases that might differ but value is same
        for phrase in ["approximately", "about", "around", "is", "was", "the result is", "result:", "decrease of", "increase of"]:
            norm_agent_ans_for_str_comp = norm_agent_ans_for_str_comp.replace(phrase, "").strip()

        print(f"DEBUG compare_answers - String Fallback Comparison: GT_str='{norm_gt_ans_for_str_comp}', Agent_str='{norm_agent_ans_for_str_comp}'")
        return norm_agent_ans_for_str_comp == norm_gt_ans_for_str_comp


async def run_single_evaluation(
    client: httpx.AsyncClient, 
    eval_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs a single evaluation item against the /process-query endpoint.
    """
    question_data = eval_item.get("qa", eval_item) 

    question = question_data.get("question", eval_item.get("question")) 
    ground_truth_answer = question_data.get("answer", eval_item.get("ground_truth_answer")) 
    item_id = eval_item.get("id", "unknown_id")

    evaluation_details = eval_item.get("evaluation_details", {})
    expected_program = evaluation_details.get("expected_final_program")
    expected_steps = evaluation_details.get("expected_calculation_steps")


    if not question:
        print(f"Skipping item {item_id} due to missing question.")
        return {
            "id": item_id,
            "question": question,
            "ground_truth": ground_truth_answer,
            "agent_answer": None,
            "error": "Missing question in evaluation data.",
            "is_correct": False,
            "expected_program": expected_program,
            "expected_steps": expected_steps,
            "agent_tool_calls": []
        }

    context_data = get_context_for_evaluation_item(eval_item)
    pre_text = context_data.get("pre_text")
    post_text = context_data.get("post_text")
    table_ori = context_data.get("table_ori")

    payload = {
        "question": question,
        "pre_text": pre_text,
        "post_text": post_text,
        "table_ori": table_ori,
    }

    print(f"\nProcessing ID: {item_id}")
    print(f"Question: {question}")
    if table_ori:
        print(f"Table Data (type): {type(table_ori)}, Snippet: {str(table_ori)[:150]}...")
    else:
        print("No table_ori data provided for this item by get_context_for_evaluation_item.")

    agent_answer = None
    explanation = None
    tool_calls_log = []
    error_message = None
    is_correct = False

    try:
        response = await client.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=120.0) 
        response.raise_for_status() 
        
        response_data = response.json()
        agent_answer = response_data.get("answer")
        explanation = response_data.get("explanation")
        tool_calls_log = response_data.get("tool_calls_log", [])
        
        print(f"Ground Truth: {ground_truth_answer}")
        print(f"Agent Answer: {agent_answer}")
        
        # Using a relative tolerance of 1% (0.01) and a small absolute tolerance
        is_correct = compare_answers(agent_answer, ground_truth_answer, rel_tol=0.01, abs_tol=0.0005) 
        print(f"Correct: {is_correct}")

        if explanation:
            print(f"Explanation: {explanation}")
        if tool_calls_log:
            print(f"Agent Tool Calls ({len(tool_calls_log)}):")
            for tc_log in tool_calls_log:
                print(f"  - Tool: {tc_log.get('tool_name')}, Args: {tc_log.get('tool_args')}, Result: {str(tc_log.get('tool_result'))[:50]}..., Error: {tc_log.get('error')}")
        
        if expected_program:
            print(f"Expected Program: {expected_program}")
        if expected_steps:
            print(f"Expected Calc Steps: {json.dumps(expected_steps, indent=2)}")


    except httpx.HTTPStatusError as e:
        error_message = f"HTTP error: {e.response.status_code} - {e.response.text}"
        print(error_message)
    except Exception as e:
        error_message = str(e)
        print(f"Error processing item {item_id}: {e}")
    
    return {
        "id": item_id,
        "question": question,
        "ground_truth": ground_truth_answer,
        "agent_answer": agent_answer,
        "explanation": explanation,
        "agent_tool_calls": tool_calls_log,
        "expected_program": expected_program,
        "expected_steps": expected_steps,
        "is_correct": is_correct,
        "error": error_message
    }

async def main(cli_args: argparse.Namespace): 
    try:
        with open(EVALUATION_FILE_PATH, 'r', encoding='utf-8') as f:
            evaluation_dataset = json.load(f)
        if not isinstance(evaluation_dataset, list):
            print(f"Error: Evaluation data in {EVALUATION_FILE_PATH} is not a JSON list.")
            return
    except FileNotFoundError:
        print(f"Error: Evaluation file not found at {EVALUATION_FILE_PATH}")
        print("Please update EVALUATION_FILE_PATH in the script.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {EVALUATION_FILE_PATH}")
        return

    print(f"Loaded {len(evaluation_dataset)} items from {EVALUATION_FILE_PATH}")
    
    num_to_evaluate = cli_args.num_items
    if num_to_evaluate is None or num_to_evaluate <= 0: 
        print(f"Processing all {len(evaluation_dataset)} items...")
        num_to_evaluate = len(evaluation_dataset)
    else:
        print(f"Processing the first {num_to_evaluate} items...")


    results_summary = []
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(evaluation_dataset):
            if i >= num_to_evaluate:
                print(f"\nReached limit of {num_to_evaluate} items. Stopping evaluation.")
                break 
            result = await run_single_evaluation(client, item)
            results_summary.append(result)
            

    output_results_path = f"evaluation_summary_first_{len(results_summary)}.json" 
    with open(output_results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nEvaluation summary for the first {len(results_summary)} items saved to {output_results_path}")

    correct_answers_count = 0
    processed_count = 0
    for res in results_summary:
        if res["error"] is None: 
            processed_count +=1
            if res["is_correct"]:
                correct_answers_count +=1
    
    if processed_count > 0:
        success_rate = (correct_answers_count / processed_count) * 100
        print(f"\n--- Overall Summary ({processed_count} items processed without error) ---")
        print(f"Correct Answers (Recall/Accuracy): {success_rate:.2f}% ({correct_answers_count}/{processed_count})")
    else:
        print("\nNo questions were processed without errors to calculate summary metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation against the QA API.")
    parser.add_argument(
        "-n", "--num-items",
        type=int,
        default=None, 
        help="Number of items from the evaluation dataset to process. Processes all if not specified or <= 0."
    )
    cli_args = parser.parse_args()

    print("Reminder: Ensure your FastAPI application is running and accessible at API_BASE_URL.")
    print(f"Reminder: Update EVALUATION_FILE_PATH if it's not '{EVALUATION_FILE_PATH}'.")
    print("Reminder: QAService should be configured to parse 'table_ori' for table-dependent questions.")
    
    asyncio.run(main(cli_args)) 
