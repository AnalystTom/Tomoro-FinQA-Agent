# Tomoro FinQA Agent

This repository contains a prototype financial question answering agent built for the ConvFinQA dataset. The system combines retrieval augmented generation and tool calling to solve multi‑step numeric questions over corporate reports.

## Repository structure

- **`app/`** – FastAPI application and agent logic
  - `main.py` bootstraps the API.
  - `services/qa_service.py` orchestrates the agent interaction loop.
  - `clients/` holds the OpenAI and embedding wrappers.
  - `agent_components/` implements prompt management and function calling.
  - `rag_pipeline/` provides utilities to parse documents, embed text and manage the FAISS index.
  - `tools/` defines the calculator and knowledge base retrieval tools.
- **`scripts/`** – command line utilities for data preparation and evaluation.
- **`evaluation/`** – helper scripts and datasets used during automated accuracy checks.
- **`tests/`** – unit and integration tests executed via `pytest`.

## Setup

1. Install Python 3.12 and create a virtual environment.
2. `pip install -r requirements.txt` installs runtime dependencies.
3. Optional dev tools (`black`, `isort`, `flake8`, `pytest-cov`) are listed in `requirements-dev.txt`.
4. Provide an OpenAI API key via `.env` or environment variables if you intend to run the full agent.

The default dataset file is `data/train.json`. The preprocessing script `scripts/prepare_evaluation_data.py` extracts a compact evaluation set used by tests and the evaluation harness.

## Running the pipeline

### 1. Build the vector store

```
python scripts/build_vector_store.py
```

This parses `data/train.json`, embeds the text using `EmbeddingClient` and stores the FAISS index under `data/vector_store/`.

### 2. Launch the API

```
uvicorn app.main:app --reload
```

The main endpoint is `POST /api/v1/qa/process-query`. It accepts a question, surrounding text snippets and the raw table from ConvFinQA. `QAService` routes the prompt through the LLM and executes tools such as the calculator when needed.

### 3. Evaluate

```
python evaluation/scripts/run_evaluations.py -n 5
```

This script runs a small benchmark against the API and reports exact‑match accuracy. Example results for the first three questions can be found in `evaluation_summary_first_3.json`.

## Key components

- **LLMClient** (`app/clients/llm_client.py`) wraps the OpenAI chat completions API with automatic retries and optional function calling.
- **FunctionCaller** (`app/agent_components/function_caller.py`) registers available tools (calculator and table queries) and executes them on behalf of the LLM.
- **PromptManager** (`app/agent_components/prompt_manager.py`) loads the system prompt from `prompts/financial_assistant_system_prompt.md` and formats the initial user message including `[PRE‑TEXT]`, `[TABLE]` and `[POST‑TEXT]` sections.
- **RetrievalService** (`app/services/retrieval_service.py`) handles embedding generation and FAISS search for the knowledge base tool.

## Testing

Run all tests with:

```
pytest -q
```

Unit tests cover the calculator, knowledge base tool, embedding client and retrieval service. End‑to‑end tests validate vector store construction. The dataset presence test ensures `data/train.json` is available.

## Future work

- Implement the table querying tool used in prompts – currently the placeholder returns an error.
- Expand evaluation coverage to the entire training set and report `EmAcc`.
- Add caching and batching around OpenAI calls to reduce latency.

