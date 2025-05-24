1. Overview

This project implements a Financial QA Agent capable of:

Interpreting financial context (text and tables).

Performing mathematical reasoning via a built-in calculator tool.

Managing multi-turn dialogues.

Evaluating performance against datasets inspired by FinQA research.

The backend is built with FastAPI, while Python scripts handle dataset generation and evaluation.

2. Project Structure

Tomoro-FinQA-Agent/
├── app/
│   ├── api/v1/
│   │   ├── routers/qa.py           # FastAPI QA endpoints
│   │   └── schemas/qa_schemas.py   # Pydantic models
│   ├── agent_components/
│   │   ├── function_caller.py      # Tool registration & execution
│   │   └── prompt_manager.py       # LLM prompt construction
│   ├── clients/llm_client.py       # AsyncOpenAI wrapper
│   ├── core/config.py              # App configuration
│   ├── rag_pipeline/               # (Legacy / offline)
│   │   └── parser.py               # TableParser for Markdown conversion
│   ├── services/qa_service.py      # Core QA logic
│   └── main.py                     # FastAPI entrypoint
├── evaluation/
│   ├── datasets/qa_eval_dataset.json
│   └── scripts/
│       ├── generate_eval_script_updated.py
│       ├── run_evaluation_script.py
│       ├── run_evaluation_single_shot.py
│       └── run_evaluation_turns.py
├── prompts/financial_assistant_system_prompt.md
├── scripts/test_parser.py          # TableParser tests
├── tests/unit/
│   ├── agent_components/test_prompt_manager.py
│   ├── clients/test_llm_client.py
│   └── services/test_qa_service.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md                       # You are here

3. Setup & Installation

Clone the repository

git clone https://github.com/AnalystTom/Tomoro-FinQA-Agent.git
cd Tomoro-FinQA-Agent

Install uv (a fast Python package installer)

Via pip:

pip install uv

Standalone installer:

curl -LsSf https://astral.sh/uv/install.sh | sh

Create & activate a virtual environment

uv venv .venv            # Create venv
source .venv/bin/activate  # Linux/macOS
# On Windows:
# .venv\Scripts\activate

Install dependencies

uv pip install -r requirements.txt

Configure environment variables

cp .env.example .env
# Then edit .env and set:
# OPENAI_API_KEY=your_api_key

4. Running the Application

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API endpoint: POST /api/v1/qa/process-query

5. Evaluation Scripts

5.1 Single-Shot Evaluation

python evaluation/scripts/run_evaluation_single_shot.py -n <num_items>

5.2 Multi-Turn Evaluation

python evaluation/scripts/run_evaluation_turns.py -n <num_items>

6. Running Tests

pytest

7. Key Functionalities & Modules

app/main.py & app/api/: FastAPI server and QA endpoints.

app/services/qa_service.py: Orchestrates conversation, tool use, and LLM calls.

agent_components/prompt_manager.py: Loads system prompt and builds message history.

agent_components/function_caller.py: Registers and executes tools (e.g., calculator).

clients/llm_client.py: Handles model interactions and function calls.

rag_pipeline/parser.py: Converts raw table data to Markdown.

evaluation/scripts/: Generates datasets and runs evaluations.

8. Current Status & Known Issues

Missing query_table_by_cell_coordinates tool: Not yet registered; impacts retrieval metrics.

Program Accuracy heuristic: Needs robustness improvements for structural comparison.

Answer comparison logic: May require tuning for varied numeric/percentage formats.

