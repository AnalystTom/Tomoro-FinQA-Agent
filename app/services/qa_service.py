# app/services/qa_service.py
import logging
import json 
from typing import List, Dict, Any, Optional

# Updated import for QAInput and QAResponse from the Canvas
from app.api.v1.schemas.qa_schemas import QAInput, QAResponse, ToolCallLog 
from app.clients.llm_client import LLMClient
from app.agent_components.prompt_manager import PromptManager, TABLE_QUERY_COORDS_TOOL_NAME
from app.agent_components.function_caller import FunctionCaller, ToolNotFoundException 
from app.rag_pipeline.parser import TableParser 
from app.config.settings import settings 

logger = logging.getLogger(__name__)

class QAService:
    """
    Service to process Q&A requests, orchestrating LLM calls, tool usage,
    and managing conversation history for multi-turn interactions.
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

    async def process_conversation_turn(self, qa_input: QAInput) -> QAResponse: # Renamed for clarity
        """
        Processes a single turn in a conversation, using existing history if provided,
        interacts with the LLM, and uses tools if necessary.

        Args:
            qa_input: The QAInput object containing the current question,
                      initial context (for first turn), and optional message history.

        Returns:
            A QAResponse object with the answer for the current turn, explanation,
            tool call logs for this turn, and the updated full conversation history.
        """
        tool_calls_log_for_this_turn: List[ToolCallLog] = []
        messages: List[Dict[str, Any]] = []
        
        # 1. Initialize or Continue Conversation History
        if qa_input.messages_history and len(qa_input.messages_history) > 0:
            logger.info(f"Continuing conversation with {len(qa_input.messages_history)} existing messages.")
            messages = list(qa_input.messages_history) # Start with a mutable copy
            # Append the current user question to the history
            messages.append({"role": "user", "content": qa_input.question})
        else:
            # This is the first turn of a new conversation
            logger.info("Starting a new conversation.")
            initial_df = None 
            initial_table_markdown: Optional[str] = None 
            table_parsing_error_message: Optional[str] = None

            try:
                if qa_input.table_ori: 
                    logger.debug(f"Processing table_ori for new conversation. Type: {type(qa_input.table_ori)}")
                    initial_df = self.table_parser.table_ori_to_dataframe(qa_input.table_ori)
                    if initial_df is not None and not initial_df.empty: 
                        markdown_candidate = self.table_parser.dataframe_to_markdown(initial_df)
                        if markdown_candidate.startswith("Error:"):
                            logger.warning(f"TableParser failed to convert DataFrame to Markdown: {markdown_candidate}")
                            table_parsing_error_message = markdown_candidate
                            initial_table_markdown = f"[TABLE PARSING ERROR: {markdown_candidate}]" 
                        else:
                            initial_table_markdown = markdown_candidate
                            logger.debug(f"Generated Markdown Table for LLM:\n{initial_table_markdown}")
                    elif initial_df is not None and initial_df.empty:
                        logger.info("Provided table_ori parsed into an empty DataFrame.")
                        initial_table_markdown = "[TABLE NOTE: The provided table data resulted in an empty table after parsing.]"
                    else: 
                        logger.info("Provided table_ori could not be parsed into a DataFrame (returned None).")
                        table_parsing_error_message = "Table data could not be understood or was invalid."
                        initial_table_markdown = f"[TABLE PARSING ERROR: {table_parsing_error_message}]"
                else:
                    logger.info("No table_ori provided in QAInput for new conversation.")
            except Exception as e:
                logger.error(f"Error processing initial table data for new conversation: {e}", exc_info=True)
                table_parsing_error_message = f"Unexpected error during table processing: {str(e)}"
                initial_table_markdown = f"[TABLE PARSING ERROR: {table_parsing_error_message}]"

            try:
                # Construct initial system and user messages
                messages = self.prompt_manager.construct_initial_agent_prompt(
                    question=qa_input.question, # This is the main/first question
                    pre_text=qa_input.pre_text,
                    post_text=qa_input.post_text,
                    initial_table_markdown=initial_table_markdown 
                )
            except Exception as e:
                logger.error(f"Error constructing initial prompt: {e}", exc_info=True)
                # For a new conversation, if prompt fails, it's critical.
                # Return an error QAResponse with an empty history.
                return QAResponse(
                    answer="I'm sorry, I encountered an error preparing your request.",
                    explanation=f"Error during initial prompt construction: {str(e)}",
                    tool_calls_log=[],
                    updated_messages_history=[], # No history to return
                    item_id=qa_input.item_id,
                    request_id=qa_input.request_id
                )
        # `initial_df` is only relevant for the current turn if it's the first turn and a table tool is called.
        # For subsequent turns, the table context is already in the `messages` history (as markdown).
        # If a table tool is called in a later turn, it might need to re-parse from history or this logic needs adjustment.
        # For now, table tool context is only from the initial `qa_input.table_ori`.

        all_tool_schemas = self.function_caller.get_all_tool_schemas()
        active_tool_schemas = all_tool_schemas

        # 2. Iterative LLM Interaction Loop (for the current turn)
        for iteration in range(self.max_tool_iterations):
            logger.info(f"Agent Iteration for current turn: {iteration + 1}/{self.max_tool_iterations}")
            
            try:
                assistant_response_dict = await self.llm_client.generate_response_with_tools(
                    messages=messages, # Pass the potentially updated history
                    tools=active_tool_schemas 
                )
            except Exception as e: 
                logger.error(f"LLMClient generate_response_with_tools failed: {e}", exc_info=True)
                return QAResponse(
                    answer="I'm sorry, I encountered an error communicating with the AI model.",
                    explanation=f"LLM API call error: {str(e)}",
                    tool_calls_log=tool_calls_log_for_this_turn,
                    updated_messages_history=messages, # Return history up to the failure point
                    item_id=qa_input.item_id,
                    request_id=qa_input.request_id
                )

            if not assistant_response_dict:
                logger.warning("LLMClient returned no response. Ending interaction for this turn.")
                return QAResponse(
                    answer="I'm sorry, I could not get a response from the AI model at this time.",
                    explanation="No response from LLM.",
                    tool_calls_log=tool_calls_log_for_this_turn,
                    updated_messages_history=messages,
                    item_id=qa_input.item_id,
                    request_id=qa_input.request_id
                )

            if "role" not in assistant_response_dict:
                 assistant_response_dict["role"] = "assistant" 
            messages.append(assistant_response_dict) # Append LLM's response to history

            if assistant_response_dict.get("tool_calls"):
                tool_calls = assistant_response_dict["tool_calls"]
                logger.info(f"LLM requested {len(tool_calls)} tool call(s) in this turn.")
                
                for tool_call_request in tool_calls:
                    tool_name = tool_call_request.get("function", {}).get("name")
                    tool_id = tool_call_request.get("id")
                    tool_args_str = tool_call_request.get("function", {}).get("arguments", "{}")

                    if not tool_name or not tool_id:
                        logger.error(f"Malformed tool call request from LLM: {tool_call_request}")
                        tool_error_msg = "Error: Malformed tool call request received from LLM."
                        messages.append({
                            "tool_call_id": tool_id or "unknown_id", "role": "tool",
                            "name": tool_name or "unknown_tool", "content": tool_error_msg
                        })
                        tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name or "unknown_tool", tool_args={"error": "malformed request"}, tool_result=tool_error_msg, error=True))
                        continue 

                    logger.info(f"Executing tool: {tool_name} with ID: {tool_id}, Args string: {tool_args_str}")
                    
                    parsed_tool_args: Dict[str, Any] = {} 
                    try:
                        parsed_tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON arguments for tool {tool_name}: {tool_args_str}")
                        tool_result_content = f"Error: Invalid JSON arguments provided for tool {tool_name}."
                        tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name, tool_args={"raw_args": tool_args_str}, tool_result=tool_result_content, error=True))
                        messages.append({"tool_call_id": tool_id, "role": "tool", "name": tool_name, "content": tool_result_content})
                        continue 

                    context_data_for_tool = None
                    # DataFrame context is primarily relevant if this is the first turn AND a table tool is called.
                    # For subsequent turns, the table (as markdown) is already in `messages`.
                    # The `query_table_by_cell_coordinates` tool, as implemented, needs a DataFrame.
                    # This implies that if it's called on a subsequent turn, it might not work unless
                    # we re-parse the table from the message history or the tool is adapted.
                    # For now, we only provide `initial_df` if available from the very first input.
                    if tool_name == TABLE_QUERY_COORDS_TOOL_NAME: 
                        # Only provide initial_df if it was parsed in *this current request cycle*
                        # (i.e., if messages_history was None, indicating first turn of a context block)
                        if qa_input.messages_history is None or len(qa_input.messages_history) == 0:
                            if initial_df is not None: 
                                context_data_for_tool = {"dataframe": initial_df}
                                logger.debug(f"Providing initial DataFrame context for tool: {tool_name}")
                            elif table_parsing_error_message:
                                 tool_result_content = f"Error: Cannot execute {tool_name}. The table data could not be processed initially. Details: {table_parsing_error_message}"
                                 logger.warning(f"Tool {tool_name} called, but initial table parsing failed.")
                            else: # No table_ori was provided initially
                                logger.warning(f"Tool {tool_name} called, but no initial table was provided.")
                                tool_result_content = f"Error: Cannot execute {tool_name} because no table was provided in the initial input."
                        else: # Not the first turn, initial_df from this cycle is not relevant
                            logger.warning(f"Tool {tool_name} called on a subsequent turn. The current QAService design provides DataFrame context only on the first turn. Tool may fail or need adaptation.")
                            tool_result_content = f"Error: Table query tool called on a subsequent turn; direct DataFrame access from initial input is not available in this state. The table (as markdown) should be in conversation history."
                        
                        if context_data_for_tool is None: # If df not set for any reason above
                             tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                             messages.append({"tool_call_id": tool_id, "role": "tool", "name": tool_name, "content": tool_result_content})
                             continue


                    try:
                        tool_result_content = await self.function_caller.execute_tool_call(
                            tool_name=tool_name,
                            tool_args=parsed_tool_args, 
                            context_data=context_data_for_tool
                        )
                        logger.info(f"Tool {tool_name} execution result (first 100 chars): {str(tool_result_content)[:100]}")
                        tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=str(tool_result_content), error=False))
                    except ToolNotFoundException as e:
                        logger.error(f"ToolNotFoundException: {e}")
                        tool_result_content = f"Error: The tool '{tool_name}' is not recognized by the system."
                        tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                    except Exception as e: 
                        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                        tool_result_content = f"Error: An unexpected error occurred while executing tool {tool_name}: {str(e)}"
                        tool_calls_log_for_this_turn.append(ToolCallLog(tool_name=tool_name, tool_args=parsed_tool_args, tool_result=tool_result_content, error=True))
                    
                    messages.append({
                        "tool_call_id": tool_id, "role": "tool",
                        "name": tool_name, "content": str(tool_result_content) 
                    })
                continue # Go to next LLM call with tool results in messages
            
            else: # No tool calls, LLM provided a direct answer for this turn
                final_answer_for_turn = assistant_response_dict.get("content")
                if final_answer_for_turn:
                    logger.info(f"LLM provided final answer for this turn in iteration {iteration + 1}.")
                    explanation = f"Answer for current turn generated after {iteration + 1} interaction(s) with the AI model."
                    if tool_calls_log_for_this_turn: # Log tools used in this specific turn
                        explanation += f" {len(tool_calls_log_for_this_turn)} tool(s) were used in this step."
                    
                    return QAResponse(
                        answer=str(final_answer_for_turn),
                        explanation=explanation,
                        tool_calls_log=tool_calls_log_for_this_turn,
                        updated_messages_history=messages, # Return the full updated history
                        item_id=qa_input.item_id,
                        request_id=qa_input.request_id
                    )
                else: 
                    logger.warning("LLM response had no content and no tool calls. Ending turn.")
                    return QAResponse(
                        answer="I'm sorry, I was unable to generate a specific answer for this turn.",
                        explanation="The AI model did not provide a conclusive answer or request further actions for this turn.",
                        tool_calls_log=tool_calls_log_for_this_turn,
                        updated_messages_history=messages,
                        item_id=qa_input.item_id,
                        request_id=qa_input.request_id
                    )

        # Max Iterations Reached for the current turn
        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached for current turn. Ending interaction.")
        last_llm_content = messages[-1].get("content") if messages and messages[-1].get("role") == "assistant" else None
        
        answer_on_max_iter = "I have reached the maximum number of processing steps for this turn. "
        if last_llm_content:
            answer_on_max_iter += f"My last thought was: {last_llm_content}"
        else:
            answer_on_max_iter += "I could not finalize an answer within the allowed steps for this turn."

        return QAResponse(
            answer=answer_on_max_iter,
            explanation=f"Processing for current turn stopped after {self.max_tool_iterations} iterations.",
            tool_calls_log=tool_calls_log_for_this_turn,
            updated_messages_history=messages, # Return history up to this point
            item_id=qa_input.item_id,
            request_id=qa_input.request_id
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) 
    logger.info("Starting QAService direct execution test (Multi-turn Logic)...")

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_default_key_if_not_set":
        logger.error("OPENAI_API_KEY is not set. QAService test cannot run LLM calls.")
    else:
        try:
            qa_service = QAService()
            logger.info("QAService initialized for direct test.")

            # --- Test Scenario: Multi-turn ---
            # Turn 1: Initial question with table
            turn1_input_data = {
                "question": "What is the Net Sales for 2001 from the table?",
                "table_ori": [ 
                    ["", "2002", "2001", "2000"],
                    ["Net sales", "$5,742", "$5,363", "$7,983"],
                    ["Cost of sales", "4,139", "4,128", "5,817"]
                ],
                "pre_text": ["Financial data below."],
                "item_id": "multi_turn_test_1"
            }
            turn1_qa_input = QAInput(**turn1_input_data)

            async def run_conversation_test():
                logger.info(f"\n--- TURN 1 ---")
                logger.info(f"Processing query: {turn1_qa_input.question}")
                response1 = await qa_service.process_conversation_turn(turn1_qa_input)
                print("\n--- QAService Response (Turn 1) ---")
                print(f"Answer: {response1.answer}")
                print(f"Explanation: {response1.explanation}")
                print(f"Tool Calls Log (Turn 1): {response1.tool_calls_log}")
                # print(f"Updated History (Turn 1): {json.dumps(response1.updated_messages_history, indent=2)}")

                # Turn 2: Follow-up question, using history from Turn 1
                if response1.updated_messages_history:
                    turn2_question = "And what about for 2000?"
                    turn2_input_data = {
                        "question": turn2_question,
                        "messages_history": response1.updated_messages_history, # Pass history
                        # pre_text, post_text, table_ori are not needed again if history is passed
                        "item_id": "multi_turn_test_1" 
                    }
                    turn2_qa_input = QAInput(**turn2_input_data)
                    
                    logger.info(f"\n--- TURN 2 ---")
                    logger.info(f"Processing query: {turn2_qa_input.question}")
                    response2 = await qa_service.process_conversation_turn(turn2_qa_input)
                    print("\n--- QAService Response (Turn 2) ---")
                    print(f"Answer: {response2.answer}")
                    print(f"Explanation: {response2.explanation}")
                    print(f"Tool Calls Log (Turn 2): {response2.tool_calls_log}")
                    # print(f"Updated History (Turn 2): {json.dumps(response2.updated_messages_history, indent=2)}")

            import asyncio
            asyncio.run(run_conversation_test())

        except RuntimeError as e:
            logger.error(f"Could not run QAService direct test due to initialization error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during QAService direct test: {e}", exc_info=True)

