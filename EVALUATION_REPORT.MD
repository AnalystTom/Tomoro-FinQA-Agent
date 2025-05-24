Financial QA Agent: Evaluation Report
1. Introduction
This report details the design, evaluation, and findings for a Financial Question-Answering (QA) Agent. The primary objective of this agent is to accurately answer financial questions based on a provided context, which includes textual information and structured data in tables. The evaluation focuses on the agent's ability to understand the context, utilize available tools, and generate correct answers, particularly in multi-turn conversational scenarios.

2. Agent Architecture & Method
The current iteration of the Financial QA Agent employs a Large Language Model (LLM), specifically GPT-4o, as its core reasoning engine. The agent is designed to process user queries that are accompanied by relevant contextual information.

Key Architectural Components:

LLM Core: GPT-4o is used for understanding the input, reasoning, and generating responses and tool usage requests.

Input Context: The agent receives the user's question, along_with pre_text (text preceding a table), post_text (text following a table), and table_ori (the original table data). The table_ori is parsed and converted into Markdown format by the TableParser component within the QAService before being included in the prompt for the LLM.

Conversational Memory: The system supports multi-turn conversations. The QAService manages the conversation history (messages_history), which is passed to the LLM on each turn. This allows the agent to maintain context across multiple user utterances.

Tools:

calculator: This tool is available to the LLM for performing mathematical calculations. It uses the asteval library for safe evaluation of expressions. Numerical results from this tool are rounded to one decimal place.

query_table_by_cell_coordinates (Removed/Inactive): While initially planned and implemented, this tool (intended for precise data extraction from tables using row/column indices) was found to be error-prone in its interaction with the LLM or its registration with the FunctionCaller. In the current evaluated version, the FunctionCaller log indicates that this tool is not being registered, meaning the agent relies solely on the LLM's ability to read the Markdown table directly.

Workflow Orchestration:

QAService: The central orchestrator. It handles incoming requests, manages conversation history, interfaces with other components, and formulates the final response.

PromptManager: Constructs the initial system prompt (loaded from prompts/financial_assistant_system_prompt.md) and the first user message containing the full problem context. The system prompt guides the LLM on its role, available tools (now only the calculator), and how to process the input.

LLMClient: Manages all API interactions with the OpenAI GPT-4o model, including sending messages, tool schemas, and handling responses.

FunctionCaller: Registers available tools (currently only the calculator) and executes them when requested by the LLM.

The agent's primary method for table-based questions now relies on the LLM directly interpreting the Markdown table provided in the prompt to extract numerical values for calculation.

3. Evaluation Setup
Dataset: The evaluation was performed using a dataset derived from train.json (FinQA-like data structure), processed into qa_eval_dataset.json. This dataset includes questions, ground truth answers, associated pre_text, post_text, table_ori, and detailed annotations like expected_final_program and dialogue_turns.

Metrics:

Execution Accuracy (EA): The percentage of questions where the agent's final answer matches the ground truth answer. Numerical and percentage answers are normalized and compared with a tolerance.

Program Accuracy (PA): A heuristic measure indicating whether the agent used the calculator tool when the expected_final_program suggested a calculation. This is not a structural program comparison.

Turn-Specific Accuracy: For multi-turn dialogues, the accuracy of the agent's answer at each turn compared to the expected intermediate answer.

Retrieval Metrics (Precision, Recall, F1): These metrics, intended to evaluate the query_table_by_cell_coordinates tool's ability to retrieve correct table rows, are reported as 0% or N/A. This is because the table query tool was found to be non-operational (not registered by FunctionCaller due to import/integration issues) in the version of the system that produced the reported EA/PA metrics.

4. Findings & Metrics
The evaluation was run on a subset of the dataset to analyze performance. Based on the latest run (e.g., processing 2 items):

Execution Accuracy (EA_app): 50.00% (1/2 items correct in the provided sample output, though previous runs showed 100% for these same 2 items, indicating sensitivity in the "is_correct" flag or slight variations in agent response).

Program Accuracy (PA_app - Heuristic): 100.00% (2/2 items). This indicates that when the expected solution involved a calculation, the agent did attempt to use the calculator tool. However, as this is a heuristic, it doesn't guarantee the correctness of the calculation expression itself.

Retrieval Metrics (Row-Level P/R/F1): 0.00%. This is a direct consequence of the query_table_by_cell_coordinates tool not being registered and used by the agent. The agent is not programmatically querying table rows.

Turn-Specific Accuracy (Example from a 2-item run):

Turn 1: 100.00%

Turn 2: 100.00%

Turn 3: 100.00%

Turn 4: 50.00%
This suggests that while the agent can handle initial turns well, accuracy may drop in later, potentially more complex, turns of a dialogue. The specific failure in Turn 4 for one item, despite a numerically close answer, highlights the challenges in automated answer comparison.

5. Analysis of Agent Behavior & Shortcomings
Table Data Interpretation & Tool Usage:

The most significant finding is the current non-operational status of the query_table_by_cell_coordinates tool. The FunctionCaller logs confirm it's not being registered due to import issues. This forces the LLM to rely solely on its ability to parse the Markdown table presented in the prompt.

Interestingly, for some questions, the LLM appears to successfully extract the correct numerical values from the Markdown table for use with the calculator tool, even when it (in previous iterations or if prompted differently) might have tried and failed to use the table query tool. This demonstrates the LLM's inherent capability but also its potential inconsistency without a structured tool for data extraction.

When the TableParser fails to convert table_ori to clean Markdown (as observed in some earlier debugging sessions), the LLM correctly identifies that the table data is unavailable or malformed, which is appropriate behavior.

Answer Correctness & Normalization:

The is_correct flag in the evaluation script has shown sensitivity. Answers that are numerically very close (e.g., "14.1%" vs. "14.14%") were sometimes marked incorrect. While the normalization and comparison functions (extract_numerical_value, compare_answers) have been iteratively refined to handle percentages and floating-point tolerances, there might still be edge cases or phrasings in agent responses that are not perfectly normalized, leading to false negatives in accuracy. The debug logs within these functions are essential for ongoing refinement.

Program Accuracy (PA):

The current PA metric is a very basic heuristic. It does not perform a structural comparison of the agent's generated calculation steps (i.e., the math expression sent to the calculator) against the expected_final_program. A more robust PA would require parsing both into a canonical form.

Multi-Turn Conversation Handling:

The system now successfully manages conversation history, allowing for multi-turn dialogues. This is a significant improvement in agent design.

The evaluation script simulates these multi-turn interactions, providing insights into turn-specific accuracy.

6. Conclusion & Future Work
The Financial QA Agent demonstrates foundational capabilities in processing contextual information, managing multi-turn dialogues, and utilizing a calculator tool. The Execution Accuracy shows promise but also highlights areas for improvement, particularly in ensuring robust data extraction and consistent answer evaluation.

Key Shortcomings Identified:

Non-Operational Table Query Tool: The query_table_by_cell_coordinates tool is not being registered by the FunctionCaller, preventing programmatic and precise data extraction from tables. This makes retrieval metrics currently untestable.

Reliance on LLM for Table Parsing: The agent's accuracy for table-based questions is entirely dependent on the LLM's ability to interpret Markdown tables directly, which can be less reliable than a dedicated tool.

Answer Comparison Nuances: Automated evaluation of numerical and percentage answers requires ongoing refinement of normalization and tolerance settings.

Program Accuracy Metric: The current PA metric is a placeholder and needs to be developed into a more meaningful structural comparison.

Recommendations for Future Work:

Fix and Reintegrate Table Query Tool: Prioritize resolving the import/registration issues for query_table_by_cell_coordinates in FunctionCaller and ensure it's correctly implemented in app/tools/table_query_tool.py. This will enable proper testing of retrieval metrics and provide the LLM with a more reliable way to access table data.

Refine System Prompts: Continuously iterate on the system prompt to better guide the LLM on when and how to use tools (once the table query tool is active) and how to interpret the provided context, especially the Markdown table.

Enhance Program Accuracy Metric: Implement a more sophisticated method for comparing the agent's generated calculation "program" (sequence of operations and arguments) with the expected_final_program.

Improve Answer Normalization: Continue to refine the extract_numerical_value and compare_answers functions in the evaluation script based on observed failure cases to improve the reliability of the Execution Accuracy metric.

Dataset Augmentation for Type-Specific Analysis: Add a question_type field to the evaluation dataset to enable deeper analysis of agent performance on different categories of financial questions.

Expand Evaluation Set: Once core functionalities are stable, evaluate on a larger portion of the dataset to obtain more statistically significant metrics.