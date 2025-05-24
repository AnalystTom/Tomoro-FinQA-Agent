# app/tools/table_query_tool.py
import logging
import pandas as pd
from typing import Any, Dict

logger = logging.getLogger(__name__)

TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_table_by_cell_coordinates",
        "description": "Retrieves the value of a specific cell from a given table (DataFrame) using 0-based row and column indices. Use this to get specific data points from the table provided in the initial context.",
        "parameters": {
            "type": "object",
            "properties": {
                # 'dataframe' is not part of the LLM's callable schema, 
                # it's passed by the FunctionCaller from context.
                "row_index": {
                    "type": "integer",
                    "description": "The 0-based index of the row to query."
                },
                "col_index": {
                    "type": "integer",
                    "description": "The 0-based index of the column to query."
                }
            },
            "required": ["row_index", "col_index"]
        }
    }
}

def query_table_by_cell_coordinates(dataframe: pd.DataFrame, row_index: int, col_index: int) -> str:
    """
    Retrieves the value from a DataFrame at the specified 0-based row and column index.

    Args:
        dataframe: The pandas DataFrame to query.
        row_index: The 0-based integer index for the row.
        col_index: The 0-based integer index for the column.

    Returns:
        A string representation of the cell value, or an error message if access fails.
    """
    if not isinstance(dataframe, pd.DataFrame):
        logger.error(f"query_table_by_cell_coordinates received non-DataFrame input. Type: {type(dataframe)}")
        return "Error: Invalid table data provided for querying."
    if dataframe.empty:
        logger.warning("query_table_by_cell_coordinates received an empty DataFrame.")
        return "Error: The provided table is empty."

    try:
        # Ensure indices are integers
        row_idx = int(row_index)
        col_idx = int(col_index)

        logger.info(f"Querying table by coordinates: row={row_idx}, col={col_idx}")
        
        # Check bounds
        if not (0 <= row_idx < dataframe.shape[0]):
            error_msg = f"Error: Row index {row_idx} is out of bounds. Table has {dataframe.shape[0]} rows (0 to {dataframe.shape[0]-1})."
            logger.warning(error_msg)
            return error_msg
        
        if not (0 <= col_idx < dataframe.shape[1]):
            error_msg = f"Error: Column index {col_idx} is out of bounds. Table has {dataframe.shape[1]} columns (0 to {dataframe.shape[1]-1})."
            logger.warning(error_msg)
            return error_msg
            
        cell_value = dataframe.iloc[row_idx, col_idx]
        logger.info(f"Retrieved value '{cell_value}' from table at ({row_idx}, {col_idx})")
        return str(cell_value) # Return value as a string for the LLM

    except ValueError: # If row_index or col_index cannot be converted to int
        error_msg = f"Error: row_index ('{row_index}') and col_index ('{col_index}') must be integers."
        logger.warning(error_msg)
        return error_msg
    except IndexError: # Should be caught by bounds check, but as a safeguard
        error_msg = f"Error: Index out of bounds when accessing table at row {row_index}, column {col_index}."
        logger.error(error_msg, exc_info=True) # Log full traceback for unexpected IndexErrors
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while querying table by coordinates: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

# Optional: query_table_by_name (and its schema) if you plan to implement it.
# For now, we focus on the coordinates-based tool as per the current FunctionCaller setup.

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create a sample DataFrame for testing
    data = {'col1': [10, 20, 30], 'col2': ['A', 'B', 'C'], 'col3': [1.1, 2.2, 3.3]}
    sample_df = pd.DataFrame(data)
    print("Sample DataFrame:\n", sample_df)

    # Test cases
    tests = [
        {"name": "Valid query", "row": 0, "col": 1, "expected_partial": "A"},
        {"name": "Valid query numeric", "row": 1, "col": 0, "expected_partial": "20"},
        {"name": "Valid query float", "row": 2, "col": 2, "expected_partial": "3.3"},
        {"name": "Row out of bounds (too high)", "row": 5, "col": 0, "expected_partial": "Error: Row index 5 is out of bounds"},
        {"name": "Column out of bounds (too high)", "row": 0, "col": 5, "expected_partial": "Error: Column index 5 is out of bounds"},
        {"name": "Row out of bounds (negative)", "row": -1, "col": 0, "expected_partial": "Error: Row index -1 is out of bounds"},
        {"name": "Column out of bounds (negative)", "row": 0, "col": -1, "expected_partial": "Error: Column index -1 is out of bounds"},
        {"name": "Non-integer row index", "row": "a", "col": 0, "expected_partial": "Error: row_index ('a') and col_index ('0') must be integers."},
    ]

    for test in tests:
        print(f"\nRunning test: {test['name']}")
        result = query_table_by_cell_coordinates(sample_df, test['row'], test['col']) # type: ignore
        print(f"  Args: row={test['row']}, col={test['col']} -> Result: '{result}'")
        assert test['expected_partial'] in result

    print("\nTesting with empty DataFrame:")
    empty_df = pd.DataFrame()
    result_empty = query_table_by_cell_coordinates(empty_df, 0, 0)
    print(f"  Result for empty DF: '{result_empty}'")
    assert "Error: The provided table is empty" in result_empty
    
    print("\nTesting with None DataFrame:")
    result_none_df = query_table_by_cell_coordinates(None, 0, 0) # type: ignore
    print(f"  Result for None DF: '{result_none_df}'")
    assert "Error: Invalid table data provided" in result_none_df

    print("\nAll direct tests passed.")

