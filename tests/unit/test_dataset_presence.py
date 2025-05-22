from pathlib import Path


def test_train_json_exists():
    assert Path('data/train.json').exists()
