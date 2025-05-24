import json
import httpx
import asyncio
import os
import re
import argparse
from typing import List, Dict, Any, Optional, Union
import math
from collections import defaultdict
import statistics

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROCESS_QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/qa/process-query"
EVALUATION_FILE_PATH = "evaluation/datasets/qa_eval_dataset.json"

def get_context_for_evaluation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    item_id = item.get("id", "unknown_id")
    return {
        "pre_text": item.get("pre_text"),
        "post_text": item.get("post_text"),
        "table_ori": item.get("table_ori")
    }

def extract_numerical_value(answer: Optional[Union[str, float, int]]) -> Optional[float]:
    if answer is None:
        return None
    
    text = str(answer).strip()
    original_text_for_percent_check = text

    phrases_to_strip = [
        "approximately", "about", "around", "is approximately", "was approximately",
        "is about", "was about", "is around", "was around", "is", "was",
        "the result is", "result:", "total value is", "total is", "value is", "amount is",
        "the change is", "change is", "decrease of", "increase of", "an increase of", "a decrease of"
    ]
    temp_text = text.lower()
    for phrase in phrases_to_strip:
        if temp_text.startswith(phrase.lower()):
            temp_text = temp_text[len(phrase):].strip()
        if temp_text.endswith(phrase.lower()):
            temp_text = temp_text[:-len(phrase)].strip()
    
    text_for_num_extraction = temp_text if temp_text else text
    text_for_num_extraction = text_for_num_extraction.replace('$', '').replace('£', '').replace('€', '').replace(',', '')
    text_for_num_extraction = text_for_num_extraction.replace('(', '-').replace(')', '')
    text_for_num_extraction = text_for_num_extraction.strip().strip('.').strip()

    matches = list(re.finditer(r"([-+]?\d*\.?\d+)", text_for_num_extraction))
    if not matches:
        print(f"DEBUG extract_numerical_value: No number found in '{text_for_num_extraction}' (original: '{text}')")
        return None
    num_str = matches[-1].group(1)

    try:
        val = float(num_str)
        if "%" in original_text_for_percent_check:
            val /= 100.0
        print(f"DEBUG extract_numerical_value: extracted {val} from '{answer}'")
        return val
    except ValueError:
        print(f"DEBUG extract_numerical_value: could not convert '{num_str}'")
        return None

def compare_answers(agent_answer_str: Optional[str], ground_truth_str: Optional[str], rel_tol=0.015, abs_tol=0.00051) -> bool:
    print(f"DEBUG compare_answers - GT: '{ground_truth_str}', Agent: '{agent_answer_str}'")

    if agent_answer_str is None and ground_truth_str is None:
        return True
    if agent_answer_str is None or ground_truth_str is None:
        return False

    agent_num = extract_numerical_value(agent_answer_str)
    gt_num = extract_numerical_value(ground_truth_str)
    print(f"DEBUG compare_answers - numbers: agent={agent_num}, gt={gt_num}")

    if agent_num is not None and gt_num is not None:
        return math.isclose(agent_num, gt_num, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        norm_agent = str(agent_answer_str).strip().lower().replace('%','').replace('$','').replace(',','')
        norm_gt    = str(ground_truth_str).strip().lower().replace('%','').replace('$','').replace(',','')
        for phrase in ["approximately","about","around","is","was","result:","decrease of","increase of"]:
            norm_agent = norm_agent.replace(phrase, "").strip()
            norm_gt    = norm_gt.replace(phrase, "").strip()
        return norm_agent.rstrip('.') == norm_gt.rstrip('.')


def _extract_dialogue_turns_from_eval_item(eval_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    evaluation_details = eval_item.get("evaluation_details", {})
    dialogue_turns = evaluation_details.get("dialogue_turns", [])
    if dialogue_turns:
        return dialogue_turns
    if "annotation" in eval_item:
        annotation = eval_item["annotation"]
        dialogue_break = annotation.get("dialogue_break") or []
        programs = annotation.get("turn_program") or []
        exe_answers = annotation.get("exe_ans_list") or []
        min_len = min(len(dialogue_break), len(programs), len(exe_answers))
        for i in range(min_len):
            dialogue_turns.append({
                "turn_question": dialogue_break[i],
                "expected_turn_program_or_value": programs[i],
                "expected_turn_numeric_answer": exe_answers[i],
            })
    return dialogue_turns


def parse_program_string(program_str: str) -> List[Dict[str, Any]]:
    steps = []
    if not program_str:
        return steps
    for op in program_str.split('),'):
        op = op.strip()
        if not op:
            continue
        match = re.match(r"(\w+)\((.*)\)?", op)
        if match:
            name = match.group(1)
            args = [a.strip() for a in match.group(2).replace('#','REF_').split(',')]
            parsed = []
            for a in args:
                try:
                    parsed.append(float(a))
                except:
                    parsed.append(a)
            steps.append({"op": name, "args": parsed})
    return steps

def compare_programs(calls: List[Dict[str, Any]], expected_program_str: Optional[str]) -> bool:
    if not expected_program_str:
        return False
    expected = parse_program_string(expected_program_str)
    calc_steps = []
    for call in calls:
        if call.get("tool_name") == "calculator":
            expr = call.get("tool_args", {}).get("math_expression", "")
            nums = set(re.findall(r"[-+]?\d*\.?\d+", expr))
            calc_steps.append({"op":"calculator","numbers":nums})
    if not expected and not calc_steps:
        return True
    if not expected or not calc_steps:
        return False
    last_nums = calc_steps[-1]["numbers"]
    last_expected = {str(arg) for arg in expected[-1].get("args",[]) if isinstance(arg,(int,float)) or (isinstance(arg,str) and arg.replace('.','',1).replace('-','',1).isdigit())}
    if last_nums and last_expected:
        if last_nums.issubset(last_expected) or last_expected.issubset(last_nums) or (len(last_nums & last_expected)/len(last_nums | last_expected) > 0.5):
            return True
    expects_calc = any(op.get("op") in ["subtract","divide","add","multiply","negate"] for op in expected)
    return expects_calc and bool(calc_steps) or (not expects_calc and not calc_steps)

async def run_single_item_conversation(
    client: httpx.AsyncClient,
    eval_item: Dict[str, Any]
) -> Dict[str, Any]:
    item_id = eval_item.get("id", "unknown_id")
    qa_data = eval_item.get("qa", eval_item)
    question = eval_item.get("question", qa_data.get("question"))
    ground_truth = eval_item.get("ground_truth_answer", qa_data.get("answer"))

    if not question:
        return {"id": item_id, "main_question": None, "final_ground_truth": None,
                "final_agent_answer": None, "is_execution_correct": False, "is_program_correct": False,
                "turn_accuracies": [], "conversation_log": [], "error": "Missing main question"}

    expected_prog = eval_item.get("evaluation_details",{}).get("expected_final_program")
    dialogue_turns = _extract_dialogue_turns_from_eval_item(eval_item)
    context = get_context_for_evaluation_item(eval_item)

    payload = {"question": question, "pre_text": context.get("pre_text"),
               "post_text": context.get("post_text"), "table_ori": context.get("table_ori"),
               "messages_history": None, "item_id": item_id, "request_id": f"eval_{item_id}_turn_0"}

    conversation_log = []
    all_calls = []
    turn_accuracies = []
    final_answer = None
    error_msg = None

    try:
        resp = await client.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("updated_messages_history")
        ans = data.get("answer")
        final_answer = ans
        calls = data.get("tool_calls_log", [])
        all_calls.extend(calls)
        conversation_log.append({"turn": 0, "question": question, "answer": ans, "calls": calls})
    except Exception as e:
        err = f"Error on Turn 0 for {item_id}: {e}"
        conversation_log.append({"turn":0,"question":question,"answer":None,"calls":[],"error":err})
        return {"id": item_id, "main_question": question, "final_ground_truth": ground_truth,
                "final_agent_answer": None, "is_execution_correct": False, "is_program_correct": False,
                "turn_accuracies": [], "conversation_log": conversation_log, "error": err}

    if dialogue_turns:
        history = history
        for i, turn in enumerate(dialogue_turns, start=1):
            tq = turn.get("turn_question")
            expected = str(turn.get("expected_turn_numeric_answer", turn.get("expected_turn_program_or_value")))
            if not tq:
                continue
            payload = {"question": tq, "messages_history": history,
                       "item_id": item_id, "request_id": f"eval_{item_id}_turn_{i}"}
            try:
                resp = await client.post(PROCESS_QUERY_ENDPOINT, json=payload, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                history = data.get("updated_messages_history")
                ans = data.get("answer")
                final_answer = ans
                calls = data.get("tool_calls_log", [])
                all_calls.extend(calls)
                correct = compare_answers(ans, expected)
                turn_accuracies.append(correct)
                conversation_log.append({"turn":i,"question":tq,"answer":ans,
                                         "expected":expected,"is_correct":correct,"calls":calls})
            except Exception as e:
                err = f"Error on Turn {i} for {item_id}: {e}"
                turn_accuracies.append(False)
                conversation_log.append({"turn":i,"question":tq,"answer":None,"calls":[],"error":err,
                                         "expected":expected})
                break

    exec_correct = compare_answers(final_answer, ground_truth)
    prog_correct = compare_programs(all_calls, expected_prog)

    return {"id": item_id, "main_question": question, "final_ground_truth": ground_truth,
            "final_agent_answer": final_answer, "is_execution_correct": exec_correct,
            "is_program_correct": prog_correct, "turn_accuracies": turn_accuracies,
            "conversation_log": conversation_log, "error": error_msg}

async def main(cli_args: argparse.Namespace):
    try:
        with open(cli_args.eval_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
            print(f"Error: Data in {cli_args.eval_file} is not a JSON list.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at {cli_args.eval_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {cli_args.eval_file}")
        return

    total = len(dataset)
    num = cli_args.num_items or total
    print(f"Processing {min(num,total)} items...")

    results = []
    async with httpx.AsyncClient() as client:
        for idx, item in enumerate(dataset):
            if idx >= num:
                break
            res = await run_single_item_conversation(client, item)
            results.append(res)

    out_file = f"multi_turn_evaluation_summary_first_{len(results)}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary for {len(results)} items to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn evaluation against the QA API.")
    parser.add_argument("-n", "--num-items", type=int, default=None,
                        help="Number of items to process. All if not specified.")
    parser.add_argument("--eval-file", type=str, default=EVALUATION_FILE_PATH,
                        help=f"Path to evaluation JSON file. Default: {EVALUATION_FILE_PATH}")
    cli_args = parser.parse_args()
    EVALUATION_FILE_PATH = cli_args.eval_file
    print("Ensure QAService is running at API_BASE_URL.")
    print(f"Using evaluation file: {EVALUATION_FILE_PATH}")
    asyncio.run(main(cli_args))
