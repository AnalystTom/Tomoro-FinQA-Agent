import json
from pathlib import Path

import pytest

from scripts.prepare_evaluation_data import main


def test_train_json_exists():
    assert Path("data/raw_documents/train.json").exists()


def test_prepare_evaluation_data(tmp_path, monkeypatch):
    input_path = Path("data/raw_documents/train.json")
    output_path = tmp_path / "qa_eval_dataset.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_evaluation_data.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    main()

    assert output_path.exists()
    with output_path.open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
    sample = data[0]
    assert {
        "id",
        "source_filename",
        "question",
        "ground_truth_answer",
        "evaluation_details",
    }.issubset(sample.keys())

    details = sample["evaluation_details"]
    assert {
        "expected_final_program",
        "expected_calculation_steps",
        "expected_final_numeric_answer",
        "dialogue_turns",
    }.issubset(details.keys())