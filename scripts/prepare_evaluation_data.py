"""Utility to create QA evaluation dataset from train.json."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


QA_PAIR = Dict[str, Any]


def _extract_dialogue_turns(annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build dialogue turn details from annotation dict."""
    dialogue_break = (
        annotation.get("dialogue_break")
        or annotation.get("dialogue_break_ori")
        or []
    )
    programs = (
        annotation.get("turn_program")
        or annotation.get("turn_program_ori")
        or []
    )
    exe_answers = annotation.get("exe_ans_list") or []

    turns: List[Dict[str, Any]] = []
    # Ensure all lists have the same length for zipping
    min_len = min(len(dialogue_break), len(programs), len(exe_answers))
    for i in range(min_len):
        turns.append(
            {
                "turn_question": dialogue_break[i],
                "expected_turn_program_or_value": programs[i],
                "expected_turn_numeric_answer": exe_answers[i],
            }
        )
    if not turns and (dialogue_break or programs or exe_answers) : 
        # This warning might be too verbose if many items don't have full dialogue turns.
        # Consider adjusting logging level or condition if needed.
        # print(f"Warning: Mismatch or empty dialogue turn lists for an item. Dialogue_break: {len(dialogue_break)}, programs: {len(programs)}, exe_answers: {len(exe_answers)}")
        pass
    return turns


def extract_qa_pairs(data: List[dict]) -> List[QA_PAIR]:
    """Extract question-answer pairs with evaluation details,
    including pre_text, post_text, table_ori, and ann_table_rows."""
    qa_pairs: List[QA_PAIR] = []
    for item in data:
        qa = item.get("qa", {}) # Main QA details are usually in this sub-dictionary
        annotation = item.get("annotation", {}) # Annotations for dialogue turns

        # Get main question and answer
        # If 'question' or 'answer' can also be at top level of 'item', adjust accordingly
        question = qa.get("question")
        answer = qa.get("answer")
        
        if question is None or answer is None:
            print(f"Skipping item with id '{item.get('id')}' due to missing question or answer in 'qa' block.")
            continue

        # --- MODIFICATION START: Extract ann_table_rows ---
        ann_table_rows = qa.get("ann_table_rows") # Typically a list of integers
        if ann_table_rows is None:
            print(f"Warning: 'ann_table_rows' missing for item id '{item.get('id')}' (source: {item.get('filename')}) from 'qa' block.")
        # --- MODIFICATION END ---

        evaluation_details = {
            "expected_final_program": qa.get("program"),
            "expected_calculation_steps": qa.get("steps") or [], # List of dicts
            "expected_final_numeric_answer": qa.get("exe_ans"), # Float or int
            "dialogue_turns": _extract_dialogue_turns(annotation), # List of dicts
            "ann_table_rows": ann_table_rows # <<< ADDED ann_table_rows HERE
        }

        pre_text = item.get("pre_text")
        post_text = item.get("post_text")
        table_ori = item.get("table_ori")
        
        if pre_text is None:
            print(f"Warning: 'pre_text' missing for item id '{item.get('id')}' (source: {item.get('filename')})")
        if post_text is None:
            print(f"Warning: 'post_text' missing for item id '{item.get('id')}' (source: {item.get('filename')})")
        if table_ori is None:
            print(f"Warning: 'table_ori' missing for item id '{item.get('id')}' (source: {item.get('filename')})")

        qa_pairs.append(
            {
                "id": item.get("id"),
                "source_filename": item.get("filename"), 
                "question": question,
                "ground_truth_answer": answer,
                "pre_text": pre_text,
                "post_text": post_text,
                "table_ori": table_ori,
                "evaluation_details": evaluation_details, # This now includes ann_table_rows
            }
        )
    return qa_pairs


def run(input_path: Path, output_path: Path) -> List[QA_PAIR]:
    """Generate QA eval dataset from input file and save to output path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    print(f"Loading data from: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(qa_pairs)} QA pairs to: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print("Save complete.")
    return qa_pairs


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare QA evaluation dataset")
    parser.add_argument(
        "--input",
        default="data/raw_documents/train.json", # Default input path
        help="Path to the source JSON data (e.g., train.json)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/datasets/qa_eval_dataset.json", # Default output path
        help="Output path for the generated QA evaluation dataset",
    )
    args = parser.parse_args(argv)

    print(f"Starting dataset generation: {args.input} -> {args.output}")
    run(Path(args.input), Path(args.output))
    print("Dataset generation finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
