import pytest
from unittest.mock import MagicMock, AsyncMock
from app.agent_components.function_caller import FunctionCaller, ToolNotFoundException
from app.tools.calculation_tool import CALCULATOR_TOOL_SCHEMA, safe_calculate
from app.tools.table_query_tool import TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA, query_table_by_cell_coordinates
import pandas as pd

@pytest.fixture
def function_caller_with_mocks(mocker):
    mock_safe_calculate = mocker.patch('app.agent_components.function_caller.safe_calculate')
    mocker.patch('app.agent_components.function_caller.CALCULATOR_TOOL_SCHEMA', new=CALCULATOR_TOOL_SCHEMA)
    
    # Patch _register_tools to prevent it from running during initial FunctionCaller() call
    # We'll manually set up tools for specific tests or let it run its default if needed.
    mocker.patch('app.agent_components.function_caller.FunctionCaller._register_tools')
    
    caller = FunctionCaller()
    # Manually set up the tools that _register_tools would normally set up
    caller.tools[CALCULATOR_TOOL_SCHEMA["function"]["name"]] = mock_safe_calculate
    caller.tool_schemas.append(CALCULATOR_TOOL_SCHEMA)
    
    return caller, mock_safe_calculate

class TestFunctionCaller:

    def test_initialization_and_tool_registration(self, function_caller_with_mocks):
        caller, mock_safe_calculate = function_caller_with_mocks
        # Check that calculator tool is registered
        assert CALCULATOR_TOOL_SCHEMA["function"]["name"] in caller.tools
        assert CALCULATOR_TOOL_SCHEMA in caller.tool_schemas
        assert caller.tools[CALCULATOR_TOOL_SCHEMA["function"]["name"]] is mock_safe_calculate
        
        # Check that table query tool is NOT registered by default
        assert TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"] not in caller.tools
        assert TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA not in caller.tool_schemas

    def test_get_all_tool_schemas(self, function_caller_with_mocks):
        caller, _ = function_caller_with_mocks
        schemas = caller.get_all_tool_schemas()
        assert isinstance(schemas, list)
        assert CALCULATOR_TOOL_SCHEMA in schemas
        assert len(schemas) == 1 # Only calculator should be registered by default

    @pytest.mark.asyncio
    async def test_execute_tool_call_calculator_success(self, function_caller_with_mocks):
        caller, mock_safe_calculate = function_caller_with_mocks
        mock_safe_calculate.return_value = "4"
        tool_name = CALCULATOR_TOOL_SCHEMA["function"]["name"]
        tool_args = {"math_expression": "2+2"}
        
        result = await caller.execute_tool_call(tool_name, tool_args)
        
        mock_safe_calculate.assert_called_once_with(math_expression="2+2")
        assert result == "4"

    @pytest.mark.asyncio
    async def test_execute_tool_call_tool_not_found(self, function_caller_with_mocks):
        caller, _ = function_caller_with_mocks
        tool_name = "non_existent_tool"
        tool_args = {}
        
        with pytest.raises(ToolNotFoundException, match=f"Tool '{tool_name}' not found."):
            await caller.execute_tool_call(tool_name, tool_args)

    @pytest.mark.asyncio
    async def test_execute_tool_call_missing_argument(self, function_caller_with_mocks):
        caller, mock_safe_calculate = function_caller_with_mocks
        tool_name = CALCULATOR_TOOL_SCHEMA["function"]["name"]
        tool_args = {} # Missing 'math_expression'
        
        result = await caller.execute_tool_call(tool_name, tool_args)
        
        assert "Error: Missing argument 'math_expression' for tool calculator." in result
        mock_safe_calculate.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_tool_call_tool_execution_exception(self, function_caller_with_mocks):
        caller, mock_safe_calculate = function_caller_with_mocks
        tool_name = CALCULATOR_TOOL_SCHEMA["function"]["name"]
        tool_args = {"math_expression": "5/0"}
        mock_safe_calculate.side_effect = ValueError("Division by zero")
        
        result = await caller.execute_tool_call(tool_name, tool_args)
        
        assert "Error executing tool calculator: Division by zero" in result
        mock_safe_calculate.assert_called_once_with(math_expression="5/0")

    @pytest.mark.asyncio
    async def test_execute_tool_call_table_query_with_context(self, mocker):
        mock_query_table_by_cell_coordinates = mocker.patch('app.agent_components.function_caller.query_table_by_cell_coordinates')
        mocker.patch('app.agent_components.function_caller.TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA', new=TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA)
        mocker.patch('app.agent_components.function_caller.CALCULATOR_TOOL_SCHEMA', new=CALCULATOR_TOOL_SCHEMA)
        mocker.patch('app.agent_components.function_caller.safe_calculate')

        # Manually set up FunctionCaller to include the table tool for this test
        mocker.patch('app.agent_components.function_caller.FunctionCaller._register_tools')
        caller = FunctionCaller()
        caller.tools[CALCULATOR_TOOL_SCHEMA["function"]["name"]] = mocker.patch('app.agent_components.function_caller.safe_calculate')
        caller.tool_schemas.append(CALCULATOR_TOOL_SCHEMA)
        caller.tools[TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"]] = mock_query_table_by_cell_coordinates
        caller.tool_schemas.append(TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA)
        
        tool_name = TABLE_QUERY_BY_COORDINATES_TOOL_SCHEMA["function"]["name"]
        tool_args = {"row_index": 0, "col_index": 0}
        mock_df = pd.DataFrame([["test_value"]])
        context_data = {"initial_dataframe": mock_df}
        
        mock_query_table_by_cell_coordinates.return_value = "test_value"

        result = await caller.execute_tool_call(tool_name, tool_args, context_data=context_data)
        
        mock_query_table_by_cell_coordinates.assert_called_once_with(
            dataframe=mock_df,
            row_index=0,
            col_index=0
        )
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_execute_tool_call_unhandled_registered_tool(self, function_caller_with_mocks, mocker):
        caller, _ = function_caller_with_mocks
        # Simulate a tool that is registered but doesn't have specific execution logic in execute_tool_call
        mock_unhandled_tool_schema = {
            "type": "function",
            "function": {"name": "unhandled_tool", "parameters": {}}
        }
        mock_unhandled_tool_func = MagicMock(return_value="Unhandled tool result")

        # Manually add the unhandled tool to the FunctionCaller instance
        caller.tools["unhandled_tool"] = mock_unhandled_tool_func
        caller.tool_schemas.append(mock_unhandled_tool_schema)

        tool_name = "unhandled_tool"
        tool_args = {}
        
        result = await caller.execute_tool_call(tool_name, tool_args)
        
        assert "Error: No execution logic for tool 'unhandled_tool'." in result
        mock_unhandled_tool_func.assert_not_called() # The function itself should not be called
