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
    for question, program, ans in zip(dialogue_break, programs, exe_answers):
        turns.append(
            {
                "turn_question": question,
                "expected_turn_program_or_value": program,
                "expected_turn_numeric_answer": ans,
            }
        )
    return turns


def extract_qa_pairs(data: List[dict]) -> List[QA_PAIR]:
    """Extract question-answer pairs with evaluation details,
    including pre_text, post_text, and table_ori.""" # Updated docstring
    qa_pairs: List[QA_PAIR] = []
    for item in data:
        qa = item.get("qa", {})
        annotation = item.get("annotation", {})

        question = qa.get("question")
        answer = qa.get("answer")
        if question is None or answer is None:
            # Also ensure that essential context items are present if needed,
            # or decide how to handle their absence. For now, just skip if no Q/A.
            print(f"Skipping item with id '{item.get('id')}' due to missing question or answer.")
            continue

        evaluation_details = {
            "expected_final_program": qa.get("program"),
            "expected_calculation_steps": qa.get("steps") or [],
            "expected_final_numeric_answer": qa.get("exe_ans"),
            "dialogue_turns": _extract_dialogue_turns(annotation),
        }

        # --- MODIFICATION START ---
        # Include pre_text, post_text, and table_ori in the output
        pre_text = item.get("pre_text")
        post_text = item.get("post_text")
        table_ori = item.get("table_ori")
        
        # Optional: Add a check or logging if these are missing,
        # as they are crucial for context.
        if pre_text is None:
            print(f"Warning: 'pre_text' missing for item id '{item.get('id')}' (source: {item.get('filename')})")
        if post_text is None:
            print(f"Warning: 'post_text' missing for item id '{item.get('id')}' (source: {item.get('filename')})")
        if table_ori is None:
            print(f"Warning: 'table_ori' missing for item id '{item.get('id')}' (source: {item.get('filename')})")
        # --- MODIFICATION END ---

        qa_pairs.append(
            {
                "id": item.get("id"),
                "source_filename": item.get("filename"), # from your example data
                "question": question,
                "ground_truth_answer": answer,
                # --- MODIFICATION START ---
                "pre_text": pre_text,
                "post_text": post_text,
                "table_ori": table_ori,
                # --- MODIFICATION END ---
                "evaluation_details": evaluation_details,
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
