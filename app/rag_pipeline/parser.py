# app/rag_pipeline/parser.py
import logging
import json
from typing import List, Dict, Any, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

class TableParser:
    """
    Parses table data, converting between different formats.
    """

    def table_ori_to_dataframe(self, table_ori: Union[List[List[Any]], List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
        """
        Converts the original table format (e.g., list of lists from train.json or QAInput)
        into a pandas DataFrame.

        The first list in table_ori (if list of lists) is assumed to be the header.
        If table_ori is a list of dictionaries, keys of the first dict are headers.

        Args:
            table_ori: The original table data.
                       Expected formats:
                       1. List of lists: `[["Header1", "Header2"], ["Data1A", "Data1B"], ["Data2A", "Data2B"]]`
                       2. List of dicts: `[{"Header1": "Data1A", "Header2": "Data1B"}, {"Header1": "Data2A", "Header2": "Data2B"}]`

        Returns:
            A pandas DataFrame, or None if parsing fails or input is empty/invalid.
        """
        if not table_ori:
            logger.warning("table_ori_to_dataframe received empty or None input.")
            return None

        try:
            if isinstance(table_ori, list) and len(table_ori) > 0:
                if all(isinstance(row, list) for row in table_ori):
                    # Assuming list of lists where the first list is the header
                    if not table_ori[0]: # Empty header list
                        logger.warning("table_ori_to_dataframe: Header row is empty. Cannot create DataFrame.")
                        return None
                    if len(table_ori) == 1: # Only header, no data
                        df = pd.DataFrame(columns=table_ori[0])
                    else:
                        num_header_cols = len(table_ori[0])
                        # Pandas will handle mismatched column lengths by creating NaNs or raising errors
                        # depending on the exact nature of the mismatch.
                        # We log if we anticipate issues but let pandas try.
                        # if not all(len(row) == num_header_cols for row in table_ori[1:]):
                        #     logger.warning("Data rows have inconsistent number of columns compared to header. Pandas will attempt to reconcile.")
                        df = pd.DataFrame(table_ori[1:], columns=table_ori[0])
                    logger.debug(f"Parsed table_ori (list of lists) into DataFrame with shape {df.shape}")
                    return df
                elif all(isinstance(row, dict) for row in table_ori):
                    # Assuming list of dictionaries
                    df = pd.DataFrame(table_ori)
                    logger.debug(f"Parsed table_ori (list of dicts) into DataFrame with shape {df.shape}")
                    return df
                else:
                    logger.error("table_ori has mixed types in the outer list, expected all lists or all dicts.")
                    return None
            else:
                logger.warning(f"table_ori_to_dataframe received invalid format or empty list: {type(table_ori)}")
                return None
        except Exception as e:
            logger.error(f"Error converting table_ori to DataFrame: {e}", exc_info=True)
            try:
                logger.debug(f"Problematic table_ori data (first 5 rows/items): {json.dumps(table_ori[:5], indent=2, default=str)}")
            except: #NOSONAR
                logger.debug(f"Problematic table_ori data (could not serialize to JSON): {str(table_ori)[:500]}")
            return None

    def dataframe_to_markdown(self, dataframe: pd.DataFrame) -> str:
        """
        Converts a pandas DataFrame into a Markdown string representation.
        Applies a very robust string conversion to each cell.

        Args:
            dataframe: The pandas DataFrame to convert.

        Returns:
            A string containing the Markdown representation of the table.
        """
        if dataframe is None: 
            logger.warning("dataframe_to_markdown received None DataFrame.")
            return "Error: No DataFrame provided to convert to Markdown."
        if not isinstance(dataframe, pd.DataFrame):
            logger.error(f"Input must be a pandas DataFrame, got {type(dataframe)}")
            return "Error: Invalid input type for DataFrame to Markdown conversion."
            
        try:
            # Create a copy to avoid modifying the original DataFrame
            df_copy = dataframe.copy()

            # Most aggressive string conversion: apply str() to every non-null cell.
            # Convert None/NaN to empty strings, as to_markdown handles them well.
            # This is generally safer than astype(str) for entire columns if there are tricky objects.
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if pd.notnull(x) and x is not None else '')
            
            markdown_table = df_copy.to_markdown(index=False) 
            logger.debug(f"Converted DataFrame of shape {dataframe.shape} to Markdown using aggressive stringification.")
            return markdown_table
        except Exception as e:
            logger.error(f"Error converting DataFrame to Markdown (even after aggressive stringification): {e}", exc_info=True)
            logger.error("DataFrame details that caused the error:")
            logger.error(f"Shape: {dataframe.shape}")
            logger.error(f"Columns: {list(dataframe.columns)}")
            logger.error(f"Data types (dtypes):\n{dataframe.dtypes.to_string()}") 
            try:
                logger.error(f"Head (first 3 rows):\n{dataframe.head(3).to_string()}")
            except Exception as head_e:
                logger.error(f"Could not get DataFrame head: {head_e}")
            
            for col in dataframe.columns:
                try:
                    sample_values = dataframe[col].dropna().unique()[:5] 
                    has_complex_type = any(not isinstance(val, (str, int, float, bool)) for val in sample_values)
                    if has_complex_type:
                        logger.error(f"Sample of potentially problematic column '{col}' (type {dataframe[col].dtype}): {sample_values}")
                except Exception as col_e:
                    logger.error(f"Could not get sample for column '{col}': {col_e}")
            return "Error: Could not convert DataFrame to Markdown."


# The original plan also mentioned DataParser for train.json ingestion.
# Including a placeholder for it here for completeness, though it's not
# directly used by the online QAService for table_ori parsing.
class DataParser:
    """
    Parses data from various sources, e.g., the train.json file.
    (Primarily for offline ingestion as per original plan)
    """
    def parse_json_file(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parses a JSON file into a list of dictionaries.

        Args:
            file_path: Path to the JSON file.

        Returns:
            A list of dictionaries, or None if parsing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list): 
                logger.info(f"Successfully parsed JSON file: {file_path}, found {len(data)} items.")
                return data
            else:
                logger.warning(f"JSON file {file_path} does not contain a list at the top level.")
                return None 
        except FileNotFoundError:
            logger.error(f"JSON file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {file_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON file {file_path}: {e}", exc_info=True)
            return None

# Function to parse train.json as per original plan (for offline use)
def parse_train_json(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Helper function to specifically parse the train.json dataset.
    """
    parser = DataParser()
    return parser.parse_json_file(file_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    tp = TableParser()

    table_lol = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 24, "Paris"],
        ["Charlie", 29, "London"]
    ]
    df_from_lol = tp.table_ori_to_dataframe(table_lol)
    if df_from_lol is not None:
        print("DataFrame from List of Lists:\n", df_from_lol)
        print("\nMarkdown from List of Lists DataFrame:\n", tp.dataframe_to_markdown(df_from_lol))
    else:
        print("Failed to parse list of lists.")

    table_complex_data = [
        ["ID", "Data", "MaybeNone"],
        [1, {"detail": "info", "values": [1,2,3]}, None], # A dict in a cell
        [2, ["a", "b", "c"], "ValidString"] # A list in a cell
    ]
    df_complex = tp.table_ori_to_dataframe(table_complex_data)
    if df_complex is not None:
        print("\nDataFrame with complex data:\n", df_complex)
        print("\nMarkdown from complex DataFrame (should be stringified):\n", tp.dataframe_to_markdown(df_complex))

    table_malformed_rows = [
        ["Header1", "Header2"],
        ["Data1A", "Data1B", "ExtraCol"], 
        ["Data2A"] 
    ]
    df_malformed = tp.table_ori_to_dataframe(table_malformed_rows)
    if df_malformed is not None:
        print("\nDataFrame from malformed rows (pandas might handle this with NaN or error):\n", df_malformed)
        print("\nMarkdown from malformed DataFrame:\n", tp.dataframe_to_markdown(df_malformed))
    else:
        print("\nFailed to parse malformed rows into DataFrame.")
    
    table_with_nones = [
        ["ColA", "ColB"],
        [1, None],
        [2, None]
    ]
    df_with_nones = tp.table_ori_to_dataframe(table_with_nones)
    if df_with_nones is not None:
        print("\nDataFrame with None values:\n", df_with_nones)
        print("\nMarkdown from DataFrame with None values:\n", tp.dataframe_to_markdown(df_with_nones))

