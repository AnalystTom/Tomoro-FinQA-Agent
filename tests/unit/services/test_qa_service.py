import pytest
from unittest.mock import AsyncMock, MagicMock
import json
from app.services.qa_service import QAService
from app.api.v1.schemas.qa_schemas import QAInput, QAResponse, ToolCallLog
from app.clients.llm_client import LLMClient
from app.agent_components.prompt_manager import PromptManager, TABLE_QUERY_COORDS_TOOL_NAME
from app.agent_components.function_caller import FunctionCaller, ToolNotFoundException
from app.rag_pipeline.parser import TableParser
from app.config.settings import settings

@pytest.fixture
def mock_qa_service_dependencies(mocker):
    mock_llm_client = AsyncMock(spec=LLMClient)
    mock_prompt_manager = MagicMock(spec=PromptManager)
    mock_function_caller = MagicMock(spec=FunctionCaller)
    mock_table_parser = MagicMock(spec=TableParser)
    mock_settings = mocker.patch('app.services.qa_service.settings')

    mocker.patch('app.services.qa_service.LLMClient', return_value=mock_llm_client)
    mocker.patch('app.services.qa_service.PromptManager', return_value=mock_prompt_manager)
    mocker.patch('app.services.qa_service.FunctionCaller', return_value=mock_function_caller)
    mocker.patch('app.services.qa_service.TableParser', return_value=mock_table_parser)

    mock_settings.MAX_TOOL_ITERATIONS = 3

    # Default mock for get_all_tool_schemas
    mock_function_caller.get_all_tool_schemas.return_value = []

    return {
        "llm_client": mock_llm_client,
        "prompt_manager": mock_prompt_manager,
        "function_caller": mock_function_caller,
        "table_parser": mock_table_parser,
        "settings": mock_settings
    }

@pytest.fixture
def qa_service(mock_qa_service_dependencies):
    return QAService()

class TestQAService:

    def test_initialization_success(self, qa_service, mock_qa_service_dependencies):
        assert qa_service.llm_client is mock_qa_service_dependencies["llm_client"]
        assert qa_service.prompt_manager is mock_qa_service_dependencies["prompt_manager"]
        assert qa_service.function_caller is mock_qa_service_dependencies["function_caller"]
        assert qa_service.table_parser is mock_qa_service_dependencies["table_parser"]
        assert qa_service.max_tool_iterations == 3

    def test_initialization_failure(self, mocker):
        mocker.patch('app.services.qa_service.LLMClient', side_effect=Exception("LLM init failed"))
        mocker.patch('app.services.qa_service.PromptManager')
        mocker.patch('app.services.qa_service.FunctionCaller')
        mocker.patch('app.services.qa_service.TableParser')
        mocker.patch('app.services.qa_service.settings')
        with pytest.raises(RuntimeError, match="QAService failed to initialize"):
            QAService()

    @pytest.mark.asyncio
    async def test_process_conversation_turn_new_conversation_direct_answer(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="What is the capital of France?",
            pre_text=["Some context."],
            post_text=[],
            table_ori=[],
            item_id="test_id_1"
        )
        
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "Paris."
        }

        response = await qa_service.process_conversation_turn(qa_input)

        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_called_once_with(
            question=qa_input.question,
            pre_text=qa_input.pre_text,
            post_text=qa_input.post_text,
            initial_table_markdown=None
        )
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.assert_called_once()
        assert response.answer == "Paris."
        assert "Answer for current turn generated after 1 interaction(s)" in response.explanation
        assert len(response.tool_calls_log) == 0
        assert len(response.updated_messages_history) == 3 # System, User, Assistant

    @pytest.mark.asyncio
    async def test_process_conversation_turn_new_conversation_with_table_direct_answer(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="What is Net Sales for 2001?",
            pre_text=[],
            post_text=[],
            table_ori=[["", "2001"], ["Net Sales", "100"]],
            item_id="test_id_2"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.return_value = MagicMock(empty=False)
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.return_value = "| | 2001 |\n|---|---|\n| Net Sales | 100 |"
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt with table"},
            {"role": "user", "content": "What is Net Sales for 2001?"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "Net Sales for 2001 is 100."
        }

        response = await qa_service.process_conversation_turn(qa_input)

        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.assert_called_once_with(qa_input.table_ori)
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.assert_called_once()
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_called_once()
        assert "Net Sales for 2001 is 100." in response.answer
        assert "Answer for current turn generated after 1 interaction(s)" in response.explanation

    @pytest.mark.asyncio
    async def test_process_conversation_turn_new_conversation_table_parsing_error(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="What is Net Sales for 2001?",
            table_ori=[["invalid", "table"]],
            item_id="test_id_3"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.side_effect = Exception("Parsing error")
        
        response = await qa_service.process_conversation_turn(qa_input)
        
        assert "I'm sorry, I encountered an error preparing your request." in response.answer
        assert "Error during initial prompt construction" in response.explanation
        assert len(response.updated_messages_history) == 0 # No history if initial prompt fails

    @pytest.mark.asyncio
    async def test_process_conversation_turn_existing_conversation(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="What about 2002?",
            messages_history=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."}
            ],
            item_id="test_id_4"
        )
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "The capital of Germany is Berlin."
        }

        response = await qa_service.process_conversation_turn(qa_input)

        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_not_called() # Should not be called for existing conv
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.assert_called_once()
        assert response.answer == "The capital of Germany is Berlin."
        assert len(response.updated_messages_history) == 4 # Original 3 + new user + new assistant

    @pytest.mark.asyncio
    async def test_process_conversation_turn_tool_call_success(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Calculate 2+2", item_id="test_id_5")
        
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Calculate 2+2"}
        ]
        
        # First LLM call: requests tool
        mock_tool_call = {
            "id": "call_calc_1",
            "function": {"name": "calculator", "arguments": '{"math_expression": "2+2"}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [mock_tool_call]},
            {"role": "assistant", "content": "The result is 4."}
        ]
        
        mock_qa_service_dependencies["function_caller"].execute_tool_call.return_value = "4"
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}}
        ]

        response = await qa_service.process_conversation_turn(qa_input)

        assert mock_qa_service_dependencies["llm_client"].generate_response_with_tools.call_count == 2
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_called_once_with(
            tool_name="calculator",
            tool_args={"math_expression": "2+2"},
            context_data={} # Assuming context_data is empty for now, will need to refine if it's used
        )
        assert response.answer == "The result is 4."
        assert len(response.tool_calls_log) == 1
        assert response.tool_calls_log[0].tool_name == "calculator"
        assert response.tool_calls_log[0].tool_result == "4"
        assert not response.tool_calls_log[0].error
        assert len(response.updated_messages_history) == 5 # System, User, Assistant (tool call), Tool, Assistant (answer)

    @pytest.mark.asyncio
    async def test_process_conversation_turn_tool_call_malformed_json_args(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Call tool with bad args", item_id="test_id_6")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Call tool with bad args"}
        ]
        mock_tool_call = {
            "id": "call_bad_args",
            "function": {"name": "calculator", "arguments": '{"math_expression": "2+2'}, # Malformed JSON
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [mock_tool_call]},
            {"role": "assistant", "content": "I encountered an error with the tool arguments."}
        ]
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}}
        ]

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_not_called()
        assert "I encountered an error with the tool arguments." in response.answer
        assert len(response.tool_calls_log) == 1
        assert response.tool_calls_log[0].error
        assert "Invalid JSON arguments" in response.tool_calls_log[0].tool_result

    @pytest.mark.asyncio
    async def test_process_conversation_turn_tool_not_found(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Call unknown tool", item_id="test_id_7")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Call unknown tool"}
        ]
        mock_tool_call = {
            "id": "call_unknown",
            "function": {"name": "unknown_tool", "arguments": '{}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [mock_tool_call]},
            {"role": "assistant", "content": "I'm sorry, I don't know how to use that tool."}
        ]
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [] # No tools registered
        mock_qa_service_dependencies["function_caller"].execute_tool_call.side_effect = ToolNotFoundException("unknown_tool")

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_called_once()
        assert "I'm sorry, I don't know how to use that tool." in response.answer
        assert len(response.tool_calls_log) == 1
        assert response.tool_calls_log[0].error
        assert "not recognized by the system" in response.tool_calls_log[0].tool_result

    @pytest.mark.asyncio
    async def test_process_conversation_turn_tool_execution_exception(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Call tool that fails", item_id="test_id_8")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Call tool that fails"}
        ]
        mock_tool_call = {
            "id": "call_fail",
            "function": {"name": "calculator", "arguments": '{"math_expression": "5/0"}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [mock_tool_call]},
            {"role": "assistant", "content": "The tool encountered an error."}
        ]
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}}
        ]
        mock_qa_service_dependencies["function_caller"].execute_tool_call.side_effect = Exception("Division by zero error")

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_called_once()
        assert "The tool encountered an error." in response.answer
        assert len(response.tool_calls_log) == 1
        assert response.tool_calls_log[0].error
        assert "unexpected error occurred while executing tool" in response.tool_calls_log[0].tool_result

    @pytest.mark.asyncio
    async def test_process_conversation_turn_max_iterations_reached(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Looping question", item_id="test_id_9")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Looping question"}
        ]
        # Simulate LLM always requesting a tool, hitting max_tool_iterations (set to 3 in setUp)
        mock_tool_call = {
            "id": "call_loop",
            "function": {"name": "dummy_tool", "arguments": '{}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant", "tool_calls": [mock_tool_call], "content": "Still thinking..."
        }
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "dummy_tool", "parameters": {}}}
        ]
        mock_qa_service_dependencies["function_caller"].execute_tool_call.return_value = "Dummy result"

        response = await qa_service.process_conversation_turn(qa_input)
        
        assert mock_qa_service_dependencies["llm_client"].generate_response_with_tools.call_count == qa_service.max_tool_iterations
        assert "I have reached the maximum number of processing steps for this turn." in response.answer
        assert "Processing for current turn stopped after 3 iterations." in response.explanation
        assert len(response.tool_calls_log) == qa_service.max_tool_iterations
        assert all(not log.error for log in response.tool_calls_log) # Assuming dummy tool doesn't error

    @pytest.mark.asyncio
    async def test_process_conversation_turn_llm_client_returns_none(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="No LLM response", item_id="test_id_10")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "No LLM response"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = None # Simulate no response

        response = await qa_service.process_conversation_turn(qa_input)
        
        assert "I could not get a response from the AI model at this time." in response.answer
        assert "No response from LLM." in response.explanation
        assert len(response.tool_calls_log) == 0

    @pytest.mark.asyncio
    async def test_process_conversation_turn_llm_client_raises_exception(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="LLM error", item_id="test_id_11")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "LLM error"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = Exception("LLM API error")

        response = await qa_service.process_conversation_turn(qa_input)
        
        assert "I encountered an error communicating with the AI model." in response.answer
        assert "LLM API call error: LLM API error" in response.explanation
        assert len(response.tool_calls_log) == 0

    @pytest.mark.asyncio
    async def test_process_conversation_turn_table_ori_empty_dataframe(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="Query empty table",
            table_ori=[["header1", "header2"]], # Valid input that results in empty DF
            item_id="test_id_12"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.return_value = MagicMock(empty=True)
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.return_value = "[TABLE NOTE: The provided table data resulted in an empty table after parsing.]"
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt with empty table note"},
            {"role": "user", "content": "Query empty table"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "The table is empty."
        }

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.assert_called_once()
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.assert_called_once()
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_called_once()
        assert "The table is empty." in response.answer
        assert "Answer for current turn generated after 1 interaction(s)" in response.explanation

    @pytest.mark.asyncio
    async def test_process_conversation_turn_table_ori_none_from_parser(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="Query unparseable table",
            table_ori=[["invalid", "data"]],
            item_id="test_id_13"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.return_value = None # Simulate parser returning None
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt with table parsing error"},
            {"role": "user", "content": "Query unparseable table"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "I could not understand the table data."
        }

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.assert_called_once()
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_called_once()
        assert "I could not understand the table data." in response.answer
        assert "Answer for current turn generated after 1 interaction(s)" in response.explanation

    @pytest.mark.asyncio
    async def test_process_conversation_turn_table_markdown_conversion_error(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="Query table with markdown error",
            table_ori=[["header"]],
            item_id="test_id_14"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.return_value = MagicMock(empty=False)
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.return_value = "Error: Markdown conversion failed"
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt with table markdown error"},
            {"role": "user", "content": "Query table with markdown error"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": "There was an issue with the table format."
        }

        response = await qa_service.process_conversation_turn(qa_input)
        
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.assert_called_once()
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.assert_called_once()
        assert "There was an issue with the table format." in response.answer
        assert "Answer for current turn generated after 1 interaction(s)" in response.explanation

    @pytest.mark.asyncio
    async def test_process_conversation_turn_llm_no_content_no_tool_calls(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Empty LLM response", item_id="test_id_15")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Empty LLM response"}
        ]
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.return_value = {
            "role": "assistant",
            "content": None, # No content
            "tool_calls": None # No tool calls
        }

        response = await qa_service.process_conversation_turn(qa_input)
        
        assert "I was unable to generate a specific answer for this turn." in response.answer
        assert "The AI model did not provide a conclusive answer or request further actions for this turn." in response.explanation
        assert len(response.tool_calls_log) == 0

    @pytest.mark.asyncio
    async def test_process_conversation_turn_malformed_tool_call_from_llm(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(question="Malformed tool call", item_id="test_id_16")
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Malformed tool call"}
        ]
        # Simulate LLM returning a tool call with missing 'name' or 'id'
        malformed_tool_call_no_name = {
            "id": "call_malformed_1",
            "function": {"arguments": '{}'},
            "type": "function"
        }
        malformed_tool_call_no_id = {
            "function": {"name": "calculator", "arguments": '{}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [malformed_tool_call_no_name]},
            {"role": "assistant", "tool_calls": [malformed_tool_call_no_id]},
            {"role": "assistant", "content": "Handled malformed tool calls."}
        ]
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}}
        ]

        response = await qa_service.process_conversation_turn(qa_input)
        
        assert mock_qa_service_dependencies["llm_client"].generate_response_with_tools.call_count == 3 # Two malformed, then final answer
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_not_called() # Should not attempt to execute malformed calls
        assert len(response.tool_calls_log) == 2 # Two malformed calls logged
        assert response.tool_calls_log[0].error
        assert "malformed request" in response.tool_calls_log[0].tool_result
        assert response.tool_calls_log[1].error
        assert "malformed request" in response.tool_calls_log[1].tool_result
        assert "Handled malformed tool calls." in response.answer

    @pytest.mark.asyncio
    async def test_process_conversation_turn_table_query_coords_tool_context(self, qa_service, mock_qa_service_dependencies):
        qa_input = QAInput(
            question="What is the value at (0,0)?",
            table_ori=[["A", "B"], ["C", "D"]],
            item_id="test_id_17"
        )
        mock_qa_service_dependencies["table_parser"].table_ori_to_dataframe.return_value = MagicMock(empty=False)
        mock_qa_service_dependencies["table_parser"].dataframe_to_markdown.return_value = "| A | B |\n|---|---|\n| C | D |"
        mock_qa_service_dependencies["prompt_manager"].construct_initial_agent_prompt.return_value = [
            {"role": "system", "content": "System prompt with table"},
            {"role": "user", "content": "What is the value at (0,0)?"}
        ]
        
        mock_tool_call = {
            "id": "call_table_coords",
            "function": {"name": TABLE_QUERY_COORDS_TOOL_NAME, "arguments": '{"row": 0, "col": 0}'},
            "type": "function"
        }
        mock_qa_service_dependencies["llm_client"].generate_response_with_tools.side_effect = [
            {"role": "assistant", "tool_calls": [mock_tool_call]},
            {"role": "assistant", "content": "The value is A."}
        ]
        mock_qa_service_dependencies["function_caller"].get_all_tool_schemas.return_value = [
            {"type": "function", "function": {"name": TABLE_QUERY_COORDS_TOOL_NAME, "parameters": {}}}
        ]
        mock_qa_service_dependencies["function_caller"].execute_tool_call.return_value = "A"

        response = await qa_service.process_conversation_turn(qa_input)
        
        # Verify that context_data was passed to execute_tool_call for TABLE_QUERY_COORDS_TOOL_NAME
        mock_qa_service_dependencies["function_caller"].execute_tool_call.assert_called_once()
        call_args, call_kwargs = mock_qa_service_dependencies["function_caller"].execute_tool_call.call_args
        assert "context_data" in call_kwargs
        assert call_kwargs["context_data"].get("initial_dataframe") is not None
        assert not call_kwargs["context_data"]["initial_dataframe"].empty
        assert "The value is A." in response.answer
