# app/services/qa_service.py
import logging
import json # For parsing tool arguments if they are JSON strings
from typing import List, Dict, Any, Optional

from app.api.v1.schemas.qa_schemas import QAInput, QAResponse, ToolCallLog # Pydantic models
from app.clients.llm_client import LLMClient
from app.agent_components.prompt_manager import PromptManager, TABLE_QUERY_COORDS_TOOL_NAME
from app.agent_components.function_caller import FunctionCaller, ToolNotFoundException 
from app.rag_pipeline.parser import TableParser 
from app.config.settings import settings # For MAX_TOOL_ITERATIONS

logger = logging.getLogger(__name__)

class QAService:
    """
    Service to process Q&A requests, orchestrating LLM calls and tool usage.
    """
    def __init__(self):
        try:
            self.llm_client = LLMClient()
            self.prompt_manager = PromptManager() 
            self.function_caller = FunctionCaller()
            self.table_parser = TableParser() 
            self.max_tool_iterations = settings.MAX_TOOL_ITERATIONS
            logger.info("QAService initialized with LLMClient, PromptManager, FunctionCaller, and TableParser.")
        except Exception as e:
            logger.error(f"Error during QAService initialization: {e}", exc_info=True)
            raise RuntimeError(f"QAService failed to initialize: {e}")


    async def process_single_entry_query(self, qa_input: QAInput) -> QAResponse:
        """
        Processes a single Q&A input, interacts with the LLM, and uses tools if necessary.
        """
        tool_calls_log: List[ToolCallLog] = []
        
        initial_df = None 
        initial_table_markdown: Optional[str] = None 
        try:
            if qa_input.table_ori: 
                logger.info(f"Processing table_ori provided in QAInput. Type: {type(qa_input.table_ori)}")
                initial_df = self.table_parser.table_ori_to_dataframe(qa_input.table_ori)
                if initial_df is not None and not initial_df.empty: 
                    initial_table_markdown = self.table_parser.dataframe_to_markdown(initial_df)
                    logger.info("Successfully parsed table_ori to DataFrame and Markdown.")
                else:
                    logger.info("Provided table_ori was empty or could not be parsed into a non-empty DataFrame.")
            else:
                logger.info("No table_ori provided in QAInput.")
        except Exception as e:
            logger.error(f"Error parsing initial table: {e}", exc_info=True)

        try:
            messages: List[Dict[str, Any]] = self.prompt_manager.construct_initial_agent_prompt(
                question=qa_input.question,
                pre_text=qa_input.pre_text,
                post_text=qa_input.post_text,
                initial_table_markdown=initial_table_markdown 
            )
        except Exception as e:
            logger.error(f"Error constructing initial prompt: {e}", exc_info=True)
            return QAResponse(
                answer="I'm sorry, I encountered an error preparing your request.",
                explanation=f"Error during prompt construction: {str(e)}",
                tool_calls_log=[]
            )

        all_tool_schemas = self.function_caller.get_all_tool_schemas()
        active_tool_schemas = all_tool_schemas

        for iteration in range(self.max_tool_iterations):
            logger.info(f"Agent Iteration: {iteration + 1}/{self.max_tool_iterations}")
            
            try:
                assistant_response_dict = await self.llm_client.generate_response_with_tools(
                    messages=messages, 
                    tools=active_tool_schemas 
                )
            except Exception as e: 
                logger.error(f"LLMClient generate_response_with_tools failed: {e}", exc_info=True)
                return QAResponse(
                    answer="I'm sorry, I encountered an error communicating with the AI model.",
                    explanation=f"LLM API call error: {str(e)}",
                    tool_calls_log=tool_calls_log
                )

            if not assistant_response_dict:
                logger.warning("LLMClient returned no response. Ending interaction.")
                return QAResponse(
                    answer="I'm sorry, I could not get a response from the AI model at this time.",
                    explanation="No response from LLM.",
                    tool_calls_log=tool_calls_log
                )

            if "role" not in assistant_response_dict:
                 assistant_response_dict["role"] = "assistant" 
            messages.append(assistant_response_dict)

            if assistant_response_dict.get("tool_calls"):
                tool_calls = assistant_response_dict["tool_calls"]
                logger.info(f"LLM requested {len(tool_calls)} tool call(s).")
                
                for tool_call_request in tool_calls:
                    tool_name = tool_call_request.get("function", {}).get("name")
                    tool_id = tool_call_request.get("id")
                    tool_args_str = tool_call_request.get("function", {}).get("arguments", "{}")

                    if not tool_name or not tool_id:
                        logger.error(f"Malformed tool call request from LLM: {tool_call_request}")
                        messages.append({
                            "tool_call_id": tool_id or "unknown_id",
                            "role": "tool",
                            "name": tool_name or "unknown_tool",
                            "content": "Error: Malformed tool call request received from LLM."
                        })
                        continue 

                    logger.info(f"Executing tool: {tool_name} with ID: {tool_id}, Args string: {tool_args_str}")
                    
                    parsed_tool_args: Dict[str, Any] = {} # Ensure it's a dict
                    try:
                        parsed_tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON arguments for tool {tool_name}: {tool_args_str}")
                        tool_result_content = f"Error: Invalid JSON arguments provided for tool {tool_name}."
                        # Log with the original string args if parsing failed
                        tool_calls_log.append(ToolCallLog(tool_name=tool_name, tool_args=tool_args_str, tool_result=tool_result_content, error=True))
                        messages.append({"tool_call_id": tool_id, "role": "tool", "name": tool_name, "content": tool_result_content})
                        continue 

                    context_data_for_tool = None
                    if tool_name == TABLE_QUERY_COORDS_TOOL_NAME: 
                        if initial_df is not None:
                            context_data_for_tool = {"dataframe": initial_df}
                            logger.debug(f"Providing DataFrame context for tool: {tool_name}")
                        else:
                            logger.warning(f"Tool {tool_name} called, but initial DataFrame is not available (was not provided or failed to parse).")
                            tool_result_content = f"Error: Cannot execute {tool_name} because no table was provided or parsed successfully from the input."
                            tool_calls_log.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                            messages.append({"tool_call_id": tool_id, "role": "tool", "name": tool_name, "content": tool_result_content})
                            continue 

                    try:
                        tool_result_content = await self.function_caller.execute_tool_call(
                            tool_name=tool_name,
                            tool_args=parsed_tool_args, # Use the parsed dictionary
                            context_data=context_data_for_tool
                        )
                        logger.info(f"Tool {tool_name} execution result (first 100 chars): {str(tool_result_content)[:100]}")
                        # ** FIX IS HERE: Pass parsed_tool_args (dict) instead of json.dumps(parsed_tool_args) or tool_args_str **
                        tool_calls_log.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=str(tool_result_content), error=False))
                    except ToolNotFoundException as e:
                        logger.error(f"ToolNotFoundException: {e}")
                        tool_result_content = f"Error: The tool '{tool_name}' is not recognized by the system."
                        tool_calls_log.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                    except Exception as e: 
                        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                        tool_result_content = f"Error: An unexpected error occurred while executing tool {tool_name}: {str(e)}"
                        tool_calls_log.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                    
                    messages.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(tool_result_content) 
                    })
                continue 
            
            else: 
                final_answer = assistant_response_dict.get("content")
                if final_answer:
                    logger.info(f"LLM provided final answer in iteration {iteration + 1}.")
                    explanation = f"Answer generated after {iteration + 1} interaction(s) with the AI model."
                    if tool_calls_log:
                        explanation += f" {len(tool_calls_log)} tool(s) were used."
                    
                    return QAResponse(
                        answer=str(final_answer),
                        explanation=explanation,
                        tool_calls_log=tool_calls_log
                    )
                else: 
                    logger.warning("LLM response had no content and no tool calls. Ending.")
                    return QAResponse(
                        answer="I'm sorry, I was unable to generate a specific answer based on the information.",
                        explanation="The AI model did not provide a conclusive answer or request further actions.",
                        tool_calls_log=tool_calls_log
                    )

        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached. Ending interaction.")
        last_llm_content = messages[-1].get("content") if messages and messages[-1].get("role") == "assistant" else None
        
        answer_on_max_iter = "I have reached the maximum number of processing steps. "
        if last_llm_content:
            answer_on_max_iter += f"My last thought was: {last_llm_content}"
        else:
            answer_on_max_iter += "I could not finalize an answer within the allowed steps."

        return QAResponse(
            answer=answer_on_max_iter,
            explanation=f"Processing stopped after {self.max_tool_iterations} iterations.",
            tool_calls_log=tool_calls_log
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) 
    logger.info("Starting QAService direct execution test (Table Parsing Re-enabled, ToolCallLog fix)...")

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_default_key_if_not_set":
        logger.error("OPENAI_API_KEY is not set. QAService test cannot run LLM calls.")
    else:
        try:
            qa_service = QAService()
            logger.info("QAService initialized for direct test.")

            sample_input_data_with_table = {
                "question": "What is the value in the first row, second column of the table? Also, what is 2+2?",
                "table_ori": [ 
                    ["HeaderA", "HeaderB", "HeaderC"],
                    ["Data_R1C1", "Data_R1C2", "Data_R1C3"],
                    ["Data_R2C1", "Data_R2C2", "Data_R2C3"]
                ],
                "pre_text": ["Context: The following table is provided."],
                "post_text": ["End of context."]
            }
            qa_input_obj = QAInput(**sample_input_data_with_table)

            async def run_test():
                logger.info(f"Processing query: {qa_input_obj.question}")
                response = await qa_service.process_single_entry_query(qa_input_obj)
                print("\n--- QAService Response ---")
                print(f"Answer: {response.answer}")
                print(f"Explanation: {response.explanation}")
                print("Tool Calls Log:")
                if response.tool_calls_log:
                    for log_entry in response.tool_calls_log:
                        print(f"  - Tool: {log_entry.tool_name}, Args: {log_entry.tool_args}, Result: {str(log_entry.tool_result)[:100]}..., Error: {log_entry.error}")
                else:
                    print("  No tools were called.")
            
            import asyncio
            asyncio.run(run_test())

        except RuntimeError as e:
            logger.error(f"Could not run QAService direct test due to initialization error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during QAService direct test: {e}", exc_info=True)
