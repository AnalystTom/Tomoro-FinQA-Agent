# app/agent_components/function_caller.py
import logging
import json
from typing import Callable, Dict, List, Any, Optional

# Tool implementations and schemas
from app.tools.calculation_tool import safe_calculate, CALCULATOR_TOOL_SCHEMA
# RetrievalService and knowledge base tool are not used in the simplified setup

# Table query tools are being removed
# try:
#     from app.tools.table_query_tool import (
#         query_table_by_cell_coordinates,
#         TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA,
#     )
#     TABLE_TOOLS_AVAILABLE = True
#     logger_fc = logging.getLogger(__name__) 
#     logger_fc.info("Successfully imported table query tools.") # This line would change
# except ImportError:
#     logger_fc = logging.getLogger(__name__) 
#     logger_fc.warning("Table query tools from 'app.tools.table_query_tool' not found.")
#     TABLE_TOOLS_AVAILABLE = False


logger = logging.getLogger(__name__) 

class ToolNotFoundException(Exception):
    """Custom exception for when a tool is not found."""
    pass

class FunctionCaller:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        
        # No services needed if only calculator is present
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

        # Table Query Tools are removed from registration
        # if TABLE_TOOLS_AVAILABLE:
        #    ...
        # else:
        #    logger.info("Table query tools were not registered as they are unavailable or removed by design.")
        logger.info("Table query tools are intentionally not registered in this configuration.")


    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all registered tool schemas.
        """
        return self.tool_schemas

    async def execute_tool_call(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any], 
        context_data: Optional[Dict[str, Any]] = None # context_data for dataframe is no longer used
    ) -> Any:
        """
        Executes a specified tool with the given arguments.
        """
        logger.info(f"Attempting to execute tool: {tool_name} with args: {tool_args}")

        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
            raise ToolNotFoundException(f"Tool '{tool_name}' not found.")

        tool_function = self.tools[tool_name]

        try:
            if tool_name == CALCULATOR_TOOL_SCHEMA["function"]["name"]:
                return tool_function(math_expression=tool_args["math_expression"])
            # No other tools are expected in this simplified setup
            else:
                logger.error(f"Tool '{tool_name}' is registered but has no specific execution logic.")
                return f"Error: No execution logic for tool '{tool_name}'."

        except KeyError as e:
            logger.error(f"Missing argument for tool {tool_name}: {e}", exc_info=True)
            return f"Error: Missing argument '{str(e)}' for tool {tool_name}."
        except Exception as e:
            logger.error(f"Error during execution of tool {tool_name}: {e}", exc_info=True)
            return f"Error executing tool {tool_name}: {str(e)}"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger_fc_main = logging.getLogger(__name__) 
    
    caller = FunctionCaller() 
    logger_fc_main.info("Registered tool schemas for test:")
    for schema in caller.get_all_tool_schemas():
        print(json.dumps(schema, indent=2)) # Should only show calculator

    async def test_calls():
        calc_result = await caller.execute_tool_call(CALCULATOR_TOOL_SCHEMA["function"]["name"], {"math_expression": "10 / 2"})
        logger_fc_main.info(f"Calculator result: {calc_result}")
        
        try:
            await caller.execute_tool_call("query_table_by_cell_coordinates", {})
        except ToolNotFoundException as e:
            logger_fc_main.info(f"Correctly caught error for non-existent table tool: {e}")

    import asyncio
    asyncio.run(test_calls())
