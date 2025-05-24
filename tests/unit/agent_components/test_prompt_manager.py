import pytest
from unittest.mock import mock_open
import os
from app.agent_components.prompt_manager import PromptManager, CALCULATOR_TOOL_NAME, DEFAULT_SYSTEM_PROMPT_FILENAME

@pytest.fixture
def mock_prompt_manager_env(mocker):
    mock_abspath = mocker.patch('os.path.abspath')
    mock_abspath.side_effect = lambda x: f"/mock/project/root/{x.split('/')[-1]}" if ".." not in x else "/mock/project/root"

    mock_join = mocker.patch('os.path.join')
    mock_join.side_effect = os.path.join # Use real join for simplicity, but mock base path

    mock_prompts_dir = "/mock/project/root/prompts"
    mock_system_prompt_path = os.path.join(mock_prompts_dir, DEFAULT_SYSTEM_PROMPT_FILENAME)
    dummy_system_prompt_content = "You are a test financial AI assistant.\nAvailable Tools: calculator."

    mock_file_open = mocker.patch('builtins.open', mock_open(read_data=dummy_system_prompt_content))
    
    mock_exists = mocker.patch('os.path.exists')
    mock_exists.return_value = True # Assume file exists by default

    return {
        "mock_abspath": mock_abspath,
        "mock_join": mock_join,
        "mock_prompts_dir": mock_prompts_dir,
        "mock_system_prompt_path": mock_system_prompt_path,
        "dummy_system_prompt_content": dummy_system_prompt_content,
        "mock_file_open": mock_file_open,
        "mock_exists": mock_exists
    }

class TestPromptManager:

    def test_initialization_success(self, mock_prompt_manager_env):
        manager = PromptManager()
        assert manager.tool_names == [CALCULATOR_TOOL_NAME]
        assert manager.prompts_dir == mock_prompt_manager_env["mock_prompts_dir"]
        assert manager.system_prompt_filename == DEFAULT_SYSTEM_PROMPT_FILENAME
        assert manager.system_prompt_template == mock_prompt_manager_env["dummy_system_prompt_content"]
        mock_prompt_manager_env["mock_file_open"].assert_called_once_with(mock_prompt_manager_env["mock_system_prompt_path"], 'r', encoding='utf-8')

    def test_initialization_with_custom_params(self, mock_prompt_manager_env, mocker):
        custom_tools = ["tool_a", "tool_b"]
        custom_path = "/custom/prompts/path"
        custom_filename = "my_custom_prompt.md"
        
        mock_prompt_manager_env["mock_exists"].return_value = True
        mocker.patch('builtins.open', mock_open(read_data="Custom prompt content"))
        mock_prompt_manager_env["mock_join"].side_effect = lambda *args: os.path.join(*args)

        manager = PromptManager(
            tool_names=custom_tools,
            prompts_base_path=custom_path,
            system_prompt_filename=custom_filename
        )
        assert manager.tool_names == custom_tools
        assert manager.prompts_dir == custom_path
        assert manager.system_prompt_filename == custom_filename
        assert manager.system_prompt_template == "Custom prompt content"
        mocker.patch('builtins.open').assert_called_once_with(os.path.join(custom_path, custom_filename), 'r', encoding='utf-8')

    def test_load_system_prompt_file_not_found(self, mock_prompt_manager_env):
        mock_prompt_manager_env["mock_exists"].return_value = False
        manager = PromptManager()
        assert "Critical error: System prompt file was not found." in manager.system_prompt_template
        mock_prompt_manager_env["mock_file_open"].assert_called_once()

    def test_load_system_prompt_other_exception(self, mock_prompt_manager_env):
        mock_prompt_manager_env["mock_file_open"].side_effect = IOError("Permission denied")
        manager = PromptManager()
        assert "Error: Could not load detailed system instructions." in manager.system_prompt_template
        mock_prompt_manager_env["mock_file_open"].assert_called_once()

    def test_construct_initial_agent_prompt_full_context(self, mock_prompt_manager_env):
        manager = PromptManager()
        question = "What is the revenue?"
        pre_text = ["Intro text."]
        post_text = ["Outro text."]
        initial_table_markdown = "| Header |\n|---|\n| Data |"

        messages = manager.construct_initial_agent_prompt(
            question, pre_text, post_text, initial_table_markdown
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == mock_prompt_manager_env["dummy_system_prompt_content"]
        
        user_content = messages[1]["content"]
        assert messages[1]["role"] == "user"
        assert f"Question: {question}" in user_content
        assert "--- Initial Context Begins ---" in user_content
        assert "[PRE-TEXT BEGIN]\nIntro text.\n[PRE-TEXT END]" in user_content
        assert "[TABLE BEGIN]\n| Header |\n|---|\n| Data |\n[TABLE END]" in user_content
        assert "[POST-TEXT BEGIN]\nOutro text.\n[POST-TEXT END]" in user_content
        assert "--- Initial Context Ends ---" in user_content
        assert "No specific text or table context was provided" not in user_content
        assert "Note: No specific table was provided" not in user_content

    def test_construct_initial_agent_prompt_only_question(self, mock_prompt_manager_env):
        manager = PromptManager()
        question = "Just a question."
        messages = manager.construct_initial_agent_prompt(question, None, None, None)

        user_content = messages[1]["content"]
        assert f"Question: {question}" in user_content
        assert "No specific text or table context was provided." in user_content
        assert "[PRE-TEXT BEGIN]" not in user_content
        assert "[TABLE BEGIN]" not in user_content
        assert "[POST-TEXT BEGIN]" not in user_content

    def test_construct_initial_agent_prompt_text_no_table(self, mock_prompt_manager_env):
        manager = PromptManager()
        question = "Question with text."
        pre_text = ["Some text."]
        messages = manager.construct_initial_agent_prompt(question, pre_text, None, None)

        user_content = messages[1]["content"]
        assert f"Question: {question}" in user_content
        assert "[PRE-TEXT BEGIN]\nSome text.\n[PRE-TEXT END]" in user_content
        assert "Note: No specific table was provided in the initial context." in user_content
        assert "[TABLE BEGIN]" not in user_content
        assert "No specific text or table context was provided." not in user_content

    def test_construct_initial_agent_prompt_empty_table(self, mock_prompt_manager_env):
        manager = PromptManager()
        question = "Question with empty table."
        initial_table_markdown = "[TABLE NOTE: The provided table data resulted in an empty table after parsing.]"
        messages = manager.construct_initial_agent_prompt(question, None, None, initial_table_markdown)

        user_content = messages[1]["content"]
        assert f"Question: {question}" in user_content
        assert "[TABLE BEGIN]\n[TABLE NOTE: The provided table data resulted in an empty table after parsing.]\n[TABLE END]" in user_content
        assert "No specific text or table context was provided." not in user_content
        assert "Note: No specific table was provided" not in user_content
