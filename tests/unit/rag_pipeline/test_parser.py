import pytest
import pandas as pd
import json
import os
from unittest.mock import mock_open
from app.rag_pipeline.parser import TableParser, DataParser, parse_train_json

@pytest.fixture
def table_parser():
    return TableParser()

@pytest.fixture
def data_parser():
    return DataParser()

@pytest.fixture
def dummy_json_file(tmp_path):
    file_path = tmp_path / "test_data.json"
    def _create_dummy_json_file(content, is_valid_json=True):
        with open(file_path, 'w', encoding='utf-8') as f:
            if is_valid_json:
                json.dump(content, f)
            else:
                f.write(content)
        return file_path
    return _create_dummy_json_file

class TestTableParser:

    def test_table_ori_to_dataframe_list_of_lists_valid(self, table_parser):
        table_ori = [
            ["Header1", "Header2"],
            ["Data1A", "Data1B"],
            ["Data2A", "Data2B"]
        ]
        df = table_parser.table_ori_to_dataframe(table_ori)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Header1", "Header2"]
        assert df.iloc[0, 0] == "Data1A"
        assert df.iloc[1, 1] == "Data2B"

    def test_table_ori_to_dataframe_list_of_lists_only_header(self, table_parser):
        table_ori = [["Header1", "Header2"]]
        df = table_parser.table_ori_to_dataframe(table_ori)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (0, 2)
        assert list(df.columns) == ["Header1", "Header2"]

    def test_table_ori_to_dataframe_list_of_dicts_valid(self, table_parser):
        table_ori = [
            {"Header1": "Data1A", "Header2": "Data1B"},
            {"Header1": "Data2A", "Header2": "Data2B"}
        ]
        df = table_parser.table_ori_to_dataframe(table_ori)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Header1", "Header2"]
        assert df.iloc[0, 0] == "Data1A"
        assert df.iloc[1, 1] == "Data2B"

    def test_table_ori_to_dataframe_empty_input(self, table_parser):
        assert table_parser.table_ori_to_dataframe([]) is None
        assert table_parser.table_ori_to_dataframe(None) is None # type: ignore

    def test_table_ori_to_dataframe_empty_header_list(self, table_parser):
        table_ori = [[], ["data"]]
        assert table_parser.table_ori_to_dataframe(table_ori) is None

    def test_table_ori_to_dataframe_mixed_types_in_outer_list(self, table_parser):
        table_ori = [["Header"], {"key": "value"}] # type: ignore
        assert table_parser.table_ori_to_dataframe(table_ori) is None

    def test_table_ori_to_dataframe_inconsistent_row_lengths(self, table_parser):
        table_ori = [
            ["H1", "H2"],
            ["D1A", "D1B", "D1C"], # Extra column
            ["D2A"] # Missing column
        ]
        df = table_parser.table_ori_to_dataframe(table_ori)
        assert isinstance(df, pd.DataFrame)
        # Pandas handles this by filling with NaN for missing, and creating new columns for extra
        assert df.shape == (2, 3) # Expect 3 columns due to "D1C"
        assert pd.isna(df.iloc[1, 1]) # D2A, D2B, D2C -> D2A, NaN, NaN

    def test_table_ori_to_dataframe_invalid_input_type(self, table_parser):
        assert table_parser.table_ori_to_dataframe("not a list") is None # type: ignore
        assert table_parser.table_ori_to_dataframe(123) is None # type: ignore

    def test_dataframe_to_markdown_valid_input(self, table_parser):
        df = pd.DataFrame({
            'ColA': [1, 2, 3],
            'ColB': ['X', 'Y', 'Z'],
            'ColC': [1.1, None, 3.3]
        })
        expected_markdown = (
            "| ColA | ColB | ColC |\n"
            "|------|------|------|\n"
            "| 1    | X    | 1.1  |\n"
            "| 2    | Y    |      |\n" # None becomes empty string
            "| 3    | Z    | 3.3  |"
        )
        assert table_parser.dataframe_to_markdown(df).strip() == expected_markdown.strip()

    def test_dataframe_to_markdown_empty_dataframe(self, table_parser):
        df = pd.DataFrame()
        expected_markdown = "| |\n|---|" # Empty table markdown
        assert table_parser.dataframe_to_markdown(df).strip() == expected_markdown.strip()

    def test_dataframe_to_markdown_none_dataframe(self, table_parser):
        result = table_parser.dataframe_to_markdown(None) # type: ignore
        assert "Error: No DataFrame provided to convert to Markdown." in result

    def test_dataframe_to_markdown_invalid_input_type(self, table_parser):
        result = table_parser.dataframe_to_markdown("not a dataframe") # type: ignore
        assert "Error: Invalid input type for DataFrame to Markdown conversion." in result

class TestDataParser:

    def test_parse_json_file_valid_list_of_dicts(self, data_parser, dummy_json_file):
        data = [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]
        file_path = dummy_json_file(data)
        parsed_data = data_parser.parse_json_file(file_path)
        assert parsed_data == data

    def test_parse_json_file_non_existent_file(self, data_parser):
        parsed_data = data_parser.parse_json_file("non_existent.json")
        assert parsed_data is None

    def test_parse_json_file_malformed_json(self, data_parser, dummy_json_file):
        file_path = dummy_json_file("not json {", is_valid_json=False)
        parsed_data = data_parser.parse_json_file(file_path)
        assert parsed_data is None

    def test_parse_json_file_not_list_at_top_level(self, data_parser, dummy_json_file):
        data = {"key": "value"}
        file_path = dummy_json_file(data)
        parsed_data = data_parser.parse_json_file(file_path)
        assert parsed_data is None

class TestParseTrainJson:

    def test_parse_train_json_calls_data_parser(self, mocker):
        mock_parser_instance = mocker.MagicMock(spec=DataParser)
        mock_parser_instance.parse_json_file.return_value = [{"mock": "data"}]
        mocker.patch('app.rag_pipeline.parser.DataParser', return_value=mock_parser_instance)
        
        file_path = "dummy_train.json"
        result = parse_train_json(file_path)
        
        mock_parser_instance.parse_json_file.assert_called_once_with(file_path)
        assert result == [{"mock": "data"}]
