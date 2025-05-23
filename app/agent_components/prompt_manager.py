# app/agent_components/prompt_manager.py
import logging
import os # For path operations
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Tool names that will be available to the LLM
CALCULATOR_TOOL_NAME = "calculator"
TABLE_QUERY_COORDS_TOOL_NAME = "query_table_by_cell_coordinates"

DEFAULT_SYSTEM_PROMPT_FILENAME = "financial_assistant_system_prompt.md"
DEFAULT_PROMPTS_DIR = "prompts" # Relative to the project root or a known base path

class PromptManager:
    """
    Manages the construction of prompts for the LLM agent.
    Loads the system prompt from an external Markdown file.
    """

    def __init__(self, 
                 tool_names: Optional[List[str]] = None, 
                 prompts_base_path: Optional[str] = None,
                 system_prompt_filename: Optional[str] = None):
        """
        Initializes the PromptManager.

        Args:
            tool_names: An optional list of tool names. If not provided,
                        default tool names (calculator, table query) will be used.
            prompts_base_path: The base directory where prompt files are stored.
                               Defaults to a 'prompts' directory relative to where
                               this script might assume the project root is.
                               For robustness, this should ideally be an absolute path
                               or a path derived from app settings.
            system_prompt_filename: The filename of the system prompt Markdown file.
        """
        if tool_names is None:
            self.tool_names = [
                CALCULATOR_TOOL_NAME,
                TABLE_QUERY_COORDS_TOOL_NAME
            ]
        else:
            self.tool_names = tool_names
        
        # Determine the path to the prompts directory
        # A more robust way in a real app would be to use settings or a fixed anchor point.
        # For this example, we'll try to construct it relative to this file's directory,
        # assuming a project structure like:
        # project_root/
        #   app/
        #     agent_components/
        #       prompt_manager.py
        #   prompts/
        #     financial_assistant_system_prompt.md
        if prompts_base_path is None:
            # Assuming this file is in app/agent_components/
            # Go up two levels to project_root, then into 'prompts'
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_approx = os.path.abspath(os.path.join(current_dir, "..", ".."))
            self.prompts_dir = os.path.join(project_root_approx, DEFAULT_PROMPTS_DIR)
        else:
            self.prompts_dir = prompts_base_path

        self.system_prompt_filename = system_prompt_filename or DEFAULT_SYSTEM_PROMPT_FILENAME
        self.system_prompt_template = self._load_system_prompt()

        logger.info(f"PromptManager initialized. Tool names: {self.tool_names}. System prompt loaded from: {os.path.join(self.prompts_dir, self.system_prompt_filename)}")

    def _load_system_prompt(self) -> str:
        """Loads the system prompt content from the specified Markdown file."""
        prompt_file_path = os.path.join(self.prompts_dir, self.system_prompt_filename)
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Replace placeholders if any (e.g., if tool names were dynamic in the MD file)
            # For now, the prompt uses hardcoded tool names, but this is where you'd format:
            # content = content.replace("{{CALCULATOR_TOOL_NAME}}", CALCULATOR_TOOL_NAME)
            # content = content.replace("{{TABLE_QUERY_COORDS_TOOL_NAME}}", TABLE_QUERY_COORDS_TOOL_NAME)
            return content
        except FileNotFoundError:
            logger.error(f"System prompt file not found at: {prompt_file_path}")
            # Fallback to a basic hardcoded prompt or raise an error
            return (
                "You are a helpful AI assistant. Critical error: System prompt file was not found. "
                "Please use your general knowledge. Available tools: calculator, query_table_by_cell_coordinates."
            )
        except Exception as e:
            logger.error(f"Error loading system prompt from {prompt_file_path}: {e}", exc_info=True)
            return (
                "You are a helpful AI assistant. Error: Could not load detailed system instructions. "
                "Please use your general knowledge. Available tools: calculator, query_table_by_cell_coordinates."
            )


    def construct_initial_agent_prompt(
        self, 
        question: str, 
        pre_text: Optional[List[str]], 
        post_text: Optional[List[str]], 
        initial_table_markdown: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Constructs the initial system and user messages for the LLM agent.

        Args:
            question: The user's primary question.
            pre_text: A list of text segments appearing before the table in the original context.
            post_text: A list of text segments appearing after the table in the original context.
            initial_table_markdown: Markdown representation of the initially provided table.

        Returns:
            A list containing two dictionaries: the system message and the user message.
        """
        
        system_message = {"role": "system", "content": self.system_prompt_template}

        # --- User Message Construction ---
        user_message_parts = []
        user_message_parts.append(f"Question: {question}\n")

        user_message_parts.append("--- Initial Context Begins ---")
        
        context_provided = False
        if pre_text:
            user_message_parts.append("\n[PRE-TEXT BEGIN]")
            user_message_parts.append("\n".join(pre_text))
            user_message_parts.append("[PRE-TEXT END]\n")
            context_provided = True
        
        if initial_table_markdown:
            user_message_parts.append("\n[TABLE BEGIN]")
            user_message_parts.append(initial_table_markdown)
            user_message_parts.append("[TABLE END]\n")
            context_provided = True
        
        if post_text:
            user_message_parts.append("\n[POST-TEXT BEGIN]")
            user_message_parts.append("\n".join(post_text))
            user_message_parts.append("[POST-TEXT END]\n")
            context_provided = True
            
        if not context_provided: 
             user_message_parts.append("\nNo specific text or table context was provided. Please answer based on general knowledge if possible, or indicate if the question cannot be answered without context or tools beyond the calculator.\n")
        elif not initial_table_markdown and (pre_text or post_text): 
             user_message_parts.append("\nNote: No specific table was provided in the initial context. Analyze the text context.\n")

        user_message_parts.append("--- Initial Context Ends ---")
        
        user_message_content = "\n".join(user_message_parts)
        user_message = {"role": "user", "content": user_message_content}

        logger.debug(f"Using system prompt (first 200 chars): {self.system_prompt_template[:200]}...")
        logger.debug(f"Constructed user message (first 200 chars): {user_message_content[:200]}...")
        return [system_message, user_message]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # To run this __main__ block directly, ensure the 'prompts' directory
    # and the 'financial_assistant_system_prompt.md' file exist
    # in the correct relative location (e.g., ../../prompts/ from this file if it's in app/agent_components)
    # Or, provide an absolute path to `prompts_base_path` when creating PromptManager.

    # Create a dummy prompts dir and file for direct execution test
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    mock_project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    mock_prompts_dir = os.path.join(mock_project_root, "prompts")
    mock_prompt_file = os.path.join(mock_prompts_dir, DEFAULT_SYSTEM_PROMPT_FILENAME)

    if not os.path.exists(mock_prompts_dir):
        os.makedirs(mock_prompts_dir)

    # Create a dummy system prompt file for the test
    dummy_system_prompt_content = (
        "You are a test financial AI assistant from a file.\n"
        "Available Tools:\n"
        f"- `{CALCULATOR_TOOL_NAME}`: Test calculator.\n"
        f"- `{TABLE_QUERY_COORDS_TOOL_NAME}`: Test table query.\n"
        "Instructions: Test instructions."
    )
    with open(mock_prompt_file, 'w', encoding='utf-8') as f:
        f.write(dummy_system_prompt_content)
    
    logger.info(f"Dummy prompt file created at: {mock_prompt_file}")

    # Initialize PromptManager, it will try to find the prompts dir relative to itself
    # or you can specify `prompts_base_path`.
    prompt_manager = PromptManager() 
                                    
    # Example 1: Full context
    print("\n--- Example 1: Full Context ---")
    messages_full = prompt_manager.construct_initial_agent_prompt(
        question="What was the total revenue in 2023?",
        pre_text=["Company X reported its earnings.", "The following table shows key financial data:"],
        initial_table_markdown="| Year | Revenue (M) | Profit (M) |\n|------|-------------|------------|\n| 2023 | 1500        | 300        |\n| 2022 | 1200        | 250        |",
        post_text=["Further details can be found in the annual report.", "All figures are in millions USD."]
    )
    print("System Message:")
    print(messages_full[0]["content"]) # This will now be from the loaded file
    print("\nUser Message:")
    print(messages_full[1]["content"])

    # Clean up dummy prompt file and dir if empty
    if os.path.exists(mock_prompt_file):
        os.remove(mock_prompt_file)
    if os.path.exists(mock_prompts_dir) and not os.listdir(mock_prompts_dir):
        os.rmdir(mock_prompts_dir)
    logger.info("Cleaned up dummy prompt file and directory.")
