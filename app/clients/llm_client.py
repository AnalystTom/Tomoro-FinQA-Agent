# app/clients/llm_client.py
import logging
from typing import List, Dict, Any, Optional
import openai # Using the official OpenAI library
from openai import AsyncOpenAI # Import AsyncOpenAI for asynchronous operations
from openai.types.chat import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import settings # For API keys, model names, etc.

# Import real tool schemas for example usage
from app.tools.calculation_tool import CALCULATOR_TOOL_SCHEMA
from app.tools.knowledge_base_tool import QUERY_FINANCIAL_KB_TOOL_SCHEMA

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with an LLM (e.g., OpenAI's GPT models)
    with support for tool usage using asynchronous operations.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initializes the LLMClient.

        Args:
            api_key: OpenAI API key. Defaults to settings.OPENAI_API_KEY.
            model_name: The LLM model name. Defaults to settings.LLM_MODEL_NAME.
        """
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model_name = model_name if model_name is not None else settings.LLM_MODEL_NAME
        
        if not self.api_key:
            logger.error("OpenAI API key is not configured. LLMClient will not function.")
            self.client: Optional[AsyncOpenAI] = None # Ensure type hint matches
        else:
            try:
                # Use AsyncOpenAI for asynchronous calls
                self.client: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=self.api_key)
                logger.info(f"AsyncOpenAI client initialized for model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
                self.client = None
        
        self.default_temperature = 0.2 # Default temperature for factual tasks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APIStatusError, # Includes 5xx errors
            openai.APITimeoutError
        )),
        reraise=True # Reraise the exception if all retries fail
    )
    async def generate_response_with_tools(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[str] = "auto" # "auto" is default, "none" or {"type": "function", "function": {"name": "my_function"}}
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a response from the LLM, potentially using tools.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "Hello"}]).
            tools: An optional list of tool schemas to provide to the LLM.
            temperature: Optional temperature for generation. Defaults to self.default_temperature.
            tool_choice: Optional tool choice parameter for OpenAI API.

        Returns:
            A dictionary representing the assistant's message, including 'role', 'content',
            and 'tool_calls' if any. Returns None if client is not initialized or API call fails definitively.
        """
        if not self.client:
            logger.error("AsyncOpenAI client is not initialized. Cannot generate response.")
            return None
        
        if temperature is None:
            temperature = self.default_temperature

        request_params: Dict[str, Any] = { # Ensure request_params is typed for clarity
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice # Only set tool_choice if tools are provided
            logger.debug(f"Calling LLM with tools. Tool choice: {tool_choice}. Number of tools: {len(tools)}")
        else:
            logger.debug("Calling LLM without tools.")


        try:
            logger.info(f"Sending request to OpenAI API. Model: {self.model_name}. Messages count: {len(messages)}.")
            
            # The create method on AsyncOpenAI().chat.completions is awaitable
            chat_completion = await self.client.chat.completions.create(**request_params)
            
            if chat_completion.choices and len(chat_completion.choices) > 0:
                assistant_message: ChatCompletionMessage = chat_completion.choices[0].message
                response_dict = assistant_message.model_dump(exclude_none=True)
                
                logger.info("Received response from OpenAI API.")
                if response_dict.get("tool_calls"):
                    logger.info(f"LLM responded with tool calls: {len(response_dict['tool_calls'])}")
                else:
                    logger.info("LLM responded with a direct message.")
                return response_dict
            else:
                logger.warning("OpenAI API response did not contain any choices.")
                return None
        except openai.BadRequestError as e: 
            logger.error(f"OpenAI API BadRequestError: {e.status_code} - {e.message}", exc_info=True)
            return {"role": "assistant", "content": f"Error: The request to the AI model was invalid. Details: {e.message}"}
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API AuthenticationError: {e.status_code} - {e.message}. Check your API key.", exc_info=True)
            raise 
        except openai.APIError as e: 
            logger.error(f"OpenAI APIError: {e.status_code} - {e.message}", exc_info=True)
            return None 
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
            return None


if __name__ == '__main__':
    # Example usage for LLMClient (requires OPENAI_API_KEY to be set in env or settings)
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_default_key_if_not_set":
            print("Skipping LLMClient direct test: OPENAI_API_KEY not set in app.core.config.settings or environment.")
            return

        llm_client = LLMClient()
        if not llm_client.client:
             print("LLMClient could not initialize its AsyncOpenAI client. Aborting test.")
             return

        sample_messages_no_tools = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("\n--- Testing LLM call without tools ---")
        response_no_tools = await llm_client.generate_response_with_tools(sample_messages_no_tools)
        if response_no_tools:
            print(f"LLM Response (no tools): {response_no_tools.get('content')}")
        else:
            print("Failed to get response without tools.")

        real_tools_schemas = [
            CALCULATOR_TOOL_SCHEMA,
            QUERY_FINANCIAL_KB_TOOL_SCHEMA
        ]
        
        sample_messages_with_real_tools = [
            {"role": "system", "content": "You are a financial assistant with access to a calculator and a knowledge base. Use them if needed."},
            {"role": "user", "content": "What were the main revenue drivers for Apple Inc. last year? Also, what is 15% of $750?"} 
        ]

        print("\n--- Testing LLM call WITH REAL tools (expecting potential tool call) ---")
        response_with_real_tools = await llm_client.generate_response_with_tools(
            sample_messages_with_real_tools, 
            tools=real_tools_schemas
        )
        if response_with_real_tools:
            print(f"LLM Full Response (with real tools): {response_with_real_tools}")
            if response_with_real_tools.get("tool_calls"):
                tool_calls = response_with_real_tools['tool_calls']
                print(f"Tool calls requested: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"  - Tool: {tc.get('function', {}).get('name')}, Args: {tc.get('function', {}).get('arguments')}")
            else:
                print(f"Direct answer (with real tools available): {response_with_real_tools.get('content')}")
        else:
            print("Failed to get response with real tools.")

    import asyncio
    asyncio.run(main())
