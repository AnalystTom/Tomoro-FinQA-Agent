# app/agent_components/function_caller.py
import logging
import json
from typing import Callable, Dict, List, Any, Optional

# Tool implementations and schemas
from app.tools.calculation_tool import safe_calculate, CALCULATOR_TOOL_SCHEMA
# The knowledge base tool was removed from the simplified plan, so its import is not needed here.
# from app.tools.knowledge_base_tool import query_financial_knowledge_base_impl, QUERY_FINANCIAL_KB_TOOL_SCHEMA
# from app.services.retrieval_service import RetrievalService # No longer needed if KB tool is removed

# Import for table query tools (from Chunk C of the original plan)
# These are still relevant for the simplified QAService.
try:
    from app.tools.table_query_tool import (
        query_table_by_cell_coordinates,
        TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA,
        # query_table_by_name, # Optional tool
        # TABLE_QUERY_BY_NAME_TOOL_SCHEMA # Optional schema
    )
    TABLE_TOOLS_AVAILABLE = True
    logger_fc = logging.getLogger(__name__) # Define logger here for consistency
    logger_fc.info("Successfully imported table query tools.")
except ImportError:
    logger_fc = logging.getLogger(__name__) # Define logger here if import fails
    logger_fc.warning(
        "Table query tools from 'app.tools.table_query_tool' not found. "
        "FunctionCaller will operate without them. "
        "Ensure table_query_tool.py is implemented as per original plan's Chunk C if table queries are needed."
    )
    TABLE_TOOLS_AVAILABLE = False
    # Define dummy placeholders if these tools are critical for other parts,
    # or ensure _register_tools handles their absence gracefully.
    def query_table_by_cell_coordinates(*args, **kwargs): # type: ignore
        return "Error: Table query tool (query_table_by_cell_coordinates) is not implemented or import failed."
    TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA = { # type: ignore
        "type": "function",
        "function": {
            "name": "query_table_by_cell_coordinates",
            "description": "Placeholder: Queries a table by cell coordinates. Not fully implemented due to import failure.",
            "parameters": {"type": "object", "properties": {
                "dataframe": {"type": "object", "description": "The pandas DataFrame to query."},
                "row_index": {"type": "integer", "description": "0-based row index."},
                "col_index": {"type": "integer", "description": "0-based column index."}
            }, "required": ["dataframe", "row_index", "col_index"]}
        }
    }


logger = logging.getLogger(__name__) # Main logger for the class

class ToolNotFoundException(Exception):
    """Custom exception for when a tool is not found."""
    pass

class FunctionCaller:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        
        # RetrievalService is no longer needed here if KB tool is removed
        # self.retrieval_service = None 

        self._register_tools()
        logger.info(f"FunctionCaller initialized with tools: {list(self.tools.keys())}")

    def _register_tools(self):
        """
        Registers available tools along with their schemas.
        """
        # Calculator Tool
        if CALCULATOR_TOOL_SCHEMA and callable(safe_calculate):
            self.tools[CALCULATOR_TOOL_SCHEMA["function"]["name"]] = safe_calculate
            self.tool_schemas.append(CALCULATOR_TOOL_SCHEMA)
            logger.debug(f"Registered tool: {CALCULATOR_TOOL_SCHEMA['function']['name']}")
        else:
            logger.error("Calculator tool or schema not available for registration.")

        # Knowledge Base Tool - REMOVED as per user request for simplification

        # Table Query Tools
        if TABLE_TOOLS_AVAILABLE:
            if TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA and callable(query_table_by_cell_coordinates):
                self.tools[TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"]] = query_table_by_cell_coordinates
                self.tool_schemas.append(TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA)
                logger.debug(f"Registered tool: {TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA['function']['name']}")
            else:
                logger.error("Table query by coordinates tool or schema not available for registration despite TABLE_TOOLS_AVAILABLE=True.")
        else:
            logger.warning("Table query tools were not registered as they are unavailable (import failed).")


    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all registered tool schemas.
        """
        return self.tool_schemas

    async def execute_tool_call(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        context_data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Executes a specified tool with the given arguments and context.

        Args:
            tool_name: The name of the tool to execute.
            tool_args: A dictionary of arguments for the tool.
            context_data: Optional dictionary containing context (e.g., {"dataframe": pd.DataFrame}).

        Returns:
            The result of the tool execution.
        """
        logger.info(f"Attempting to execute tool: {tool_name} with args: {tool_args}")
        if context_data: # Log only if context_data is not None or empty
            # Avoid logging large dataframes directly. Log keys or types.
            context_keys = list(context_data.keys()) if isinstance(context_data, dict) else "Non-dict context"
            logger.debug(f"Context data provided with keys: {context_keys}")

        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
            raise ToolNotFoundException(f"Tool '{tool_name}' not found.")

        tool_function = self.tools[tool_name]

        try:
            if tool_name == CALCULATOR_TOOL_SCHEMA["function"]["name"]:
                return tool_function(math_expression=tool_args["math_expression"])
            
            # KB Tool execution logic removed
            
            elif TABLE_TOOLS_AVAILABLE and tool_name == TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"]:
                if not context_data or "dataframe" not in context_data:
                    logger.error(f"DataFrame missing in context_data for tool {tool_name}")
                    return "Error: DataFrame context required for this table query tool is missing."
                
                # Ensure all required args for query_table_by_cell_coordinates are present
                required_args = ["row_index", "col_index"]
                for arg_name in required_args:
                    if arg_name not in tool_args:
                        raise KeyError(f"Missing required argument '{arg_name}' for tool {tool_name}")

                return tool_function(
                    dataframe=context_data["dataframe"],
                    row_index=tool_args["row_index"],
                    col_index=tool_args["col_index"]
                )
            
            else:
                # This case should ideally not be hit if tool_name is in self.tools
                # and all registered tools have explicit handling.
                logger.error(f"Tool '{tool_name}' is registered but has no specific execution logic in FunctionCaller.")
                return f"Error: No execution logic for tool '{tool_name}'."

        except KeyError as e:
            logger.error(f"Missing argument for tool {tool_name}: {e}", exc_info=True)
            return f"Error: Missing argument '{str(e)}' for tool {tool_name}."
        except Exception as e:
            logger.error(f"Error during execution of tool {tool_name}: {e}", exc_info=True)
            return f"Error executing tool {tool_name}: {str(e)}"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger_fc_main = logging.getLogger(__name__) # Separate logger for main block
    
    # This __main__ block is for basic, manual testing of FunctionCaller.
    # It assumes that the tool implementation files (calculation_tool.py, table_query_tool.py)
    # are present and importable from their respective locations.

    # Mock table_query_tool for this standalone test if it's not implemented yet
    # to prevent ImportError during the test run.
    if not TABLE_TOOLS_AVAILABLE:
        logger_fc_main.info("Using placeholder for table query tool in __main__ test as import failed.")
        # The global placeholders defined at the top will be used.

    caller = FunctionCaller() # Initialize FunctionCaller
    logger_fc_main.info("Registered tool schemas for test:")
    for schema in caller.get_all_tool_schemas():
        print(json.dumps(schema, indent=2))

    async def test_calls():
        # Test calculator
        calc_result = await caller.execute_tool_call(CALCULATOR_TOOL_SCHEMA["function"]["name"], {"math_expression": "10 / 2"})
        logger_fc_main.info(f"Calculator result: {calc_result}")
        
        calc_result_error = await caller.execute_tool_call(CALCULATOR_TOOL_SCHEMA["function"]["name"], {"math_expression": "10 / 0"})
        logger_fc_main.info(f"Calculator error result: {calc_result_error}")

        # Test table query (will use placeholder if table_query_tool.py is missing)
        if TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"] in caller.tools:
            import pandas as pd # Import pandas here as it's only for this test case
            sample_df = pd.DataFrame({'ColA': [100, 200], 'ColB': [300, 400]})
            table_context = {"dataframe": sample_df}
            table_args = {"row_index": 0, "col_index": 1}
            
            table_result = await caller.execute_tool_call(
                TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"],
                table_args,
                context_data=table_context
            )
            logger_fc_main.info(f"Table Query result: {table_result}") # Expects 300 if real, or error if placeholder
        else:
            logger_fc_main.warning(f"Skipping table query tool test as '{TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA['function']['name']}' is not in registered tools.")


        # Test non-existent tool
        try:
            await caller.execute_tool_call("imaginary_tool", {})
        except ToolNotFoundException as e:
            logger_fc_main.info(f"Correctly caught error for non-existent tool: {e}")
        except Exception as e:
            logger_fc_main.error(f"Unexpected error testing non-existent tool: {e}")

    import asyncio
    asyncio.run(test_calls())