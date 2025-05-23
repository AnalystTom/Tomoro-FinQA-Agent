# standalone_parser_test_script.py
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union

# Ensure this import path matches the location of your TableParser class
# If your project structure is different, you might need to adjust sys.path
# or run this script from a location where 'app' is a recognizable package.
try:
    from app.rag_pipeline.parser import TableParser
except ImportError:
    print("Failed to import TableParser. Make sure the script is run from a location")
    print("where 'app.rag_pipeline.parser' is accessible, or adjust Python's sys.path.")
    print("For example, if this script is in your project root, and 'app' is a subdir, it should work.")
    # You might need to add project root to sys.path if running from a different location:
    # import sys
    # import os
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust '..' as needed
    # from app.rag_pipeline.parser import TableParser
    exit()


# --- CONFIGURE LOGGING TO SEE DETAILED OUTPUT FROM TableParser ---
# Set logging level to DEBUG to see all logs from TableParser
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_specific_table(table_name: str, table_data: Union[List[List[Any]], List[Dict[str, Any]]]):
    """
    Tests the TableParser with a specific table_ori data.
    """
    logger.info(f"\n--- Testing Table: {table_name} ---")
    
    tp = TableParser()

    # 1. Convert table_ori to DataFrame
    logger.info("Attempting to convert table_ori to DataFrame...")
    df: Optional[pd.DataFrame] = tp.table_ori_to_dataframe(table_data)

    if df is None:
        logger.error("DataFrame creation failed. Check previous logs from TableParser for details.")
        return
    
    logger.info("DataFrame created successfully.")
    print("\nDataFrame Info:")
    df.info(verbose=True, show_counts=True) # Provides dtypes, non-null counts
    
    print("\nDataFrame Head (first 5 rows):")
    try:
        print(df.head().to_string())
    except Exception as e:
        print(f"Could not display DataFrame head: {e}")
        # If head fails, try to print a smaller portion or just column names
        print(f"Columns: {df.columns.tolist()}")


    # 2. Convert DataFrame to Markdown
    logger.info("\nAttempting to convert DataFrame to Markdown...")
    markdown_output: str = tp.dataframe_to_markdown(df)

    if markdown_output.startswith("Error:"):
        logger.error(f"Markdown conversion failed. Result: {markdown_output}")
    else:
        logger.info("Markdown conversion successful.")
        print("\nGenerated Markdown (first 500 characters):")
        print(markdown_output[:500])
        if len(markdown_output) > 500:
            print("... (markdown truncated)")
        # You can also save the full markdown to a file if it's very long:
        # with open(f"{table_name}_output.md", "w", encoding="utf-8") as f_md:
        #     f_md.write(markdown_output)
        # logger.info(f"Full markdown saved to {table_name}_output.md")

    logger.info(f"--- Finished Testing Table: {table_name} ---\n")


if __name__ == "__main__":
    # --- PASTE YOUR PROBLEMATIC table_ori DATA HERE ---
    # Example: Replace this with the actual table_ori from your failing evaluation item
    
    problematic_table_1_id = "Single_RSG/2008/page_114.pdf-2" # From your logs
    problematic_table_1_data = [ # This is an EXAMPLE, replace with actual data
        ["", "Year Ended December 31, 2008 (Unaudited)", "Year Ended December 31, 2007 (Unaudited)"],
        ["Revenue", "$9,362.2", "$9,244.9"],
        ["Income from continuing operations available to common stockholders", "285.7", "423.2"],
        ["Basic earnings per share", ".76", "1.10"],
        ["Diluted earnings per share", ".75", "1.09"]
    ]
    
    problematic_table_2_id = "Single_AAPL/2002/page_23.pdf-1" # From your logs
    problematic_table_2_data = [ # This is an EXAMPLE, replace with actual data
        ["", "2002", "2001", "2000"],
        ["Net sales", "$5,742", "$5,363", "$7,983"],
        ["Cost of sales", "4,139", "4,128", "5,817"],
        ["Gross margin", "$1,603", "$1,235", "$2,166"]
    ]

    # Add more problematic tables as needed
    # problematic_table_3_id = "some_other_id"
    # problematic_table_3_data = [ ... ]

    # --- RUN TESTS ---
    if problematic_table_1_data: # Check if data is actually pasted
        test_specific_table(problematic_table_1_id, problematic_table_1_data)
    else:
        logger.warning(f"No data provided for {problematic_table_1_id}. Skipping test.")

    if problematic_table_2_data:
        test_specific_table(problematic_table_2_id, problematic_table_2_data)
    else:
        logger.warning(f"No data provided for {problematic_table_2_id}. Skipping test.")
        
    # test_specific_table(problematic_table_3_id, problematic_table_3_data)

    logger.info("Standalone TableParser testing script finished.")
    logger.info("Check the console output above for detailed logs from TableParser,")
    logger.info("especially if 'Error: Could not convert DataFrame to Markdown' occurred.")
    logger.info("The logs should include DataFrame shape, dtypes, and head content.")

