# tests/unit/clients/test_llm_client.py
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import openai # For exception types
from openai import AsyncOpenAI # Import AsyncOpenAI for mocking
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from app.clients.llm_client import LLMClient
from app.config.settings import settings # Used by LLMClient for defaults

# A helper to create a mock ChatCompletion response object
def create_mock_chat_completion(message_content: str = None, tool_calls: list = None):
    mock_choice = MagicMock()
    mock_choice.message = ChatCompletionMessage(
        role="assistant",
        content=message_content,
        tool_calls=tool_calls
    )
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_choice]
    return mock_chat_completion

class TestLLMClient(unittest.IsolatedAsyncioTestCase):

    @patch('app.clients.llm_client.settings') # Patch where LLMClient imports settings
    @patch('app.clients.llm_client.AsyncOpenAI') # Patch AsyncOpenAI where LLMClient uses it
    def setUp(self, MockAsyncOpenAI, mock_llm_client_settings):
        # Configure the settings mock that LLMClient will see
        self.mock_settings = mock_llm_client_settings
        self.mock_settings.OPENAI_API_KEY = "test_api_key"
        self.mock_settings.LLM_MODEL_NAME = "test-gpt-model"

        # Mock the AsyncOpenAI client instance and its methods
        self.mock_openai_client_instance = MagicMock(spec=AsyncOpenAI) # Use spec for better mocking
        self.mock_chat_completions_create = AsyncMock() # chat.completions.create is async
        self.mock_openai_client_instance.chat.completions.create = self.mock_chat_completions_create
        MockAsyncOpenAI.return_value = self.mock_openai_client_instance # AsyncOpenAI() constructor returns our mock

        self.llm_client = LLMClient() # Initialize with patched settings and AsyncOpenAI client

    async def test_init_success(self):
        self.assertIsNotNone(self.llm_client.client)
        self.assertEqual(self.llm_client.model_name, "test-gpt-model")

    @patch('app.clients.llm_client.settings')
    @patch('app.clients.llm_client.AsyncOpenAI') # Patch AsyncOpenAI
    def test_init_no_api_key(self, MockAsyncOpenAI_no_key, mock_settings_no_key):
        mock_settings_no_key.OPENAI_API_KEY = None
        mock_settings_no_key.LLM_MODEL_NAME = "test-gpt-model"
        
        client_no_key = LLMClient()
        self.assertIsNone(client_no_key.client) # Expect client not to be initialized
        MockAsyncOpenAI_no_key.assert_not_called() # AsyncOpenAI() should not be called

    async def test_generate_response_no_tools_direct_answer(self):
        messages = [{"role": "user", "content": "Hello"}]
        expected_content = "Hi there!"
        self.mock_chat_completions_create.return_value = create_mock_chat_completion(
            message_content=expected_content
        )

        response = await self.llm_client.generate_response_with_tools(messages)

        self.mock_chat_completions_create.assert_called_once_with(
            model="test-gpt-model",
            messages=messages,
            temperature=self.llm_client.default_temperature # Check default temp
            # tools and tool_choice should not be present
        )
        self.assertNotIn("tools", self.mock_chat_completions_create.call_args.kwargs)
        self.assertNotIn("tool_choice", self.mock_chat_completions_create.call_args.kwargs)

        self.assertIsNotNone(response)
        self.assertEqual(response["role"], "assistant")
        self.assertEqual(response["content"], expected_content)
        self.assertIsNone(response.get("tool_calls"))

    async def test_generate_response_with_tools_tool_call_requested(self):
        messages = [{"role": "user", "content": "What's the weather in London?"}]
        tool_schemas = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        
        mock_tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            function=Function(name="get_weather", arguments='{"location": "London"}'),
            type="function"
        )
        self.mock_chat_completions_create.return_value = create_mock_chat_completion(
            tool_calls=[mock_tool_call]
        )

        response = await self.llm_client.generate_response_with_tools(messages, tools=tool_schemas)

        self.mock_chat_completions_create.assert_called_once_with(
            model="test-gpt-model",
            messages=messages,
            temperature=self.llm_client.default_temperature,
            tools=tool_schemas,
            tool_choice="auto" # Default tool_choice when tools are present
        )
        self.assertIsNotNone(response)
        self.assertEqual(response["role"], "assistant")
        self.assertIsNone(response.get("content")) # Expect no direct content if tool call is made
        self.assertIsNotNone(response.get("tool_calls"))
        self.assertEqual(len(response["tool_calls"]), 1)
        self.assertEqual(response["tool_calls"][0]["id"], "call_123")
        self.assertEqual(response["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(response["tool_calls"][0]["function"]["arguments"], '{"location": "London"}')

    async def test_generate_response_custom_temperature_and_tool_choice(self):
        messages = [{"role": "user", "content": "Call a specific tool."}]
        tool_schemas = [{"type": "function", "function": {"name": "specific_tool", "parameters": {}}}]
        custom_temp = 0.8
        custom_tool_choice = {"type": "function", "function": {"name": "specific_tool"}}

        # Corrected argument name for the helper function
        self.mock_chat_completions_create.return_value = create_mock_chat_completion(message_content="OK")

        await self.llm_client.generate_response_with_tools(
            messages, 
            tools=tool_schemas, 
            temperature=custom_temp,
            tool_choice=custom_tool_choice
        )

        self.mock_chat_completions_create.assert_called_once_with(
            model="test-gpt-model",
            messages=messages,
            temperature=custom_temp,
            tools=tool_schemas,
            tool_choice=custom_tool_choice
        )

    @patch('app.clients.llm_client.logger') # To check logging
    async def test_generate_response_api_retryable_error_then_success(self, mock_logger):
        messages = [{"role": "user", "content": "Retry test"}]
        # Simulate a RateLimitError then success
        self.mock_chat_completions_create.side_effect = [
            openai.RateLimitError("Simulated rate limit", response=MagicMock(), body=None),
            create_mock_chat_completion(message_content="Success after retry")
        ]

        response = await self.llm_client.generate_response_with_tools(messages)
        
        self.assertEqual(self.mock_chat_completions_create.call_count, 2) # Called twice
        self.assertIsNotNone(response)
        self.assertEqual(response["content"], "Success after retry")
    
    @patch('app.clients.llm_client.logger')
    async def test_generate_response_api_retryable_error_all_attempts_fail(self, mock_logger):
        messages = [{"role": "user", "content": "Retry fail test"}]
        self.mock_chat_completions_create.side_effect = openai.APITimeoutError("Simulated timeout")

        with self.assertRaises(openai.APITimeoutError): # Expect tenacity to reraise
            await self.llm_client.generate_response_with_tools(messages)
        
        self.assertEqual(self.mock_chat_completions_create.call_count, 3) # Default 3 attempts

    @patch('app.clients.llm_client.logger')
    async def test_generate_response_bad_request_error(self, mock_logger):
        messages = [{"role": "user", "content": "Bad request test"}]
        mock_bad_request_exception = openai.BadRequestError(
            message="Invalid request parameter.", 
            response=MagicMock(status_code=400), 
            body={"error": {"message": "Invalid parameter 'x'."}} 
        )
        self.mock_chat_completions_create.side_effect = mock_bad_request_exception

        response = await self.llm_client.generate_response_with_tools(messages)
        
        self.mock_chat_completions_create.assert_called_once()
        self.assertIsNotNone(response)
        self.assertIn("Error: The request to the AI model was invalid", response["content"])
        self.assertIn("Invalid request parameter", response["content"]) 

    async def test_generate_response_authentication_error(self):
        messages = [{"role": "user", "content": "Auth error test"}]
        # Set the side_effect on the mock that is actually called
        self.mock_chat_completions_create.side_effect = openai.AuthenticationError(
            message="Invalid API key", response=MagicMock(), body=None
        )

        with self.assertRaises(openai.AuthenticationError): 
            await self.llm_client.generate_response_with_tools(messages)
        
        self.mock_chat_completions_create.assert_called_once()

    async def test_client_not_initialized_generate_response(self):
        with patch('app.clients.llm_client.settings') as mock_settings_no_key_local:
            mock_settings_no_key_local.OPENAI_API_KEY = None
            mock_settings_no_key_local.LLM_MODEL_NAME = "test-gpt-model"
            # Patch AsyncOpenAI for this specific instantiation context
            with patch('app.clients.llm_client.AsyncOpenAI') as MockAsyncOpenAI_local:
                client_no_init = LLMClient()
                MockAsyncOpenAI_local.assert_not_called() # Ensure AsyncOpenAI wasn't called if key is None
        
        self.assertIsNone(client_no_init.client)
        response = await client_no_init.generate_response_with_tools([{"role": "user", "content": "Hello"}])
        self.assertIsNone(response)


if __name__ == '__main__':
    unittest.main()
