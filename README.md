# Tomoro-FinQA-Agent

## 🧠 Overview

**Tomoro-FinQA-Agent** is a Financial Question Answering (QA) agent that can:

- Interpret financial documents, including text and tables.
- Perform mathematical reasoning using a built-in calculator tool.
- Manage multi-turn conversations intelligently.
- Evaluate performance on datasets inspired by FinQA research.

The backend is powered by **FastAPI**, with supporting **Python scripts** for evaluation and dataset generation.

---

## 📁 Project Structure

```
├── .env.example
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── agent_components
│   │   ├── function_caller.py
│   │   └── prompt_manager.py
│   ├── api
│   │   ├── __init__.py
│   │   └── v1
│   │       ├── __init__.py
│   │       ├── routers
│   │       │   ├── __init__.py
│   │       │   └── qa.py
│   │       └── schemas
│   │           ├── __init__.py
│   │           ├── document_schemas.py
│   │           └── qa_schemas.py
│   ├── clients
│   │   ├── __init__.py
│   │   ├── embedding_client.py
│   │   ├── llm_client.py
│   │   └── vector_db_client.py
│   ├── config
│   │   └── settings.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── domain_models.py
│   ├── rag_pipeline
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   ├── parser.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── qa_service.py
│   │   └── retrieval_service.py
│   └── tools
│       ├── __init__.py
│       ├── calculation_tool.py
│       ├── knowledge_base_tool.py
│       ├── query_financial_knowledge_base_tool.py
│       └── table_query_tool.py
├── data
│   ├── raw_documents
│   │   └── train.json
│   ├── train.json
│   └── vector_store
│       ├── faiss_index.idx
│       └── faiss_metadata.json
├── docker-compose.yml
├── evaluation
│   ├── compare_answers_testing.py
│   ├── datasets
│   │   ├── prep
│   │   │   └── qa_eval_dataset.json
│   │   └── qa_eval_dataset.json
│   ├── results
│   │   ├── multi_turn_evaluation_summary_first_30.json
│   │   ├── single_shot_evaluation_summary_first_10.json
│   │   ├── single_shot_evaluation_summary_first_100_62.json
│   │   ├── single_shot_evaluation_summary_first_100_faulty_decimal.json
│   │   └── single_shot_evaluation_summary_first_5.json
│   └── scripts
│       ├── run_evaluation_single_shot.py
│       ├── run_evaluation_single_shot_cost_latency.py
│       └── run_evaluation_turns.py
├── prompts
│   ├── financial_assistant_system_prompt.md
│   ├── financial_assistant_system_prompt_v1.md
│   └── financial_assistant_system_prompt_v2.md
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── build_vector_store.py
│   ├── ingest_data.py
│   ├── prepare_evaluation_data.py
│   ├── run_evaluations.py
│   └── test_parser.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── e2e
│   │   ├── __init__.py
│   │   ├── test_build_vector_store.py
│   │   └── test_full_qa_flow.py
│   ├── integration
│   │   ├── __init__.py
│   │   ├── api
│   │   │   └── __init__.py
│   │   ├── test_embedding_client.py
│   │   ├── test_example_integration.py
│   │   └── test_llm_client.py
│   └── unit
│       ├── __init__.py
│       ├── agent_components
│       │   ├── test_function_caller.py
│       │   └── test_prompt_manager.py
│       ├── api
│       │   └── test_qa_api.py
│       ├── clients
│       │   ├── __init__.py
│       │   └── test_llm_client.py
│       ├── evaluation_pipeline
│       │   └── evaluation_extraction.py
│       ├── rag_pipeline
│       │   ├── __init__.py
│       │   ├── test_chunker.py
│       │   ├── test_indexer.py
│       │   └── test_parser.py
│       ├── services
│       │   ├── __init__.py
│       │   ├── test_qa_service.py
│       │   └── test_retrieval_service.py
│       ├── test_dataset_presence.py
│       ├── test_example_unit.py
│       └── tools
│           ├── test_calculation_tool.py
│           └── test_knowledge_base_tool.py
└── uv.lock
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository**

```bash
git clone https://github.com/AnalystTom/Tomoro-FinQA-Agent.git
cd Tomoro-FinQA-Agent
```

2. **Install `uv` (Fast Python Installer)**

Via pip:

```bash
pip install uv
```

Or via script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Create and Activate Virtual Environment**

```bash
uv venv .venv
source .venv/bin/activate      # On Linux/macOS
# .venv\Scripts\activate       # On Windows
```

4. **Install Dependencies**

```bash
uv pip install -r requirements.txt
```

5. **Set Environment Variables**

```bash
cp .env.example .env
```

Then edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key
```

---

## 🚀 Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Endpoint:

```
POST /api/v1/qa/process-query
```

---

## 📊 Evaluation Scripts

### 🟢 Single-Shot Evaluation

```bash
python evaluation/scripts/run_evaluation_single_shot.py -n <num_items>
```

### 🔁 Multi-Turn Evaluation

```bash
python evaluation/scripts/run_evaluation_turns.py -n <num_items>
```

---

## 🧪 Running Tests

Run all unit tests with:

```bash
pytest
```

---

## 🔍 Key Modules & Responsibilities

- **`app/main.py`, `api/`** – FastAPI server setup and QA endpoints.
- **`qa_service.py`** – Orchestrates dialogue, reasoning, and LLM tool usage.
- **`prompt_manager.py`** – Loads system prompt and builds message history.
- **`function_caller.py`** – Registers and executes tools (e.g., calculator).
- **`llm_client.py`** – Manages async OpenAI API calls and function calling.
- **`parser.py`** – Parses table data and converts to Markdown (legacy RAG).
- **`evaluation/scripts/`** – Contains tools for dataset generation and evaluation workflows.

---

## ⚠️ Known Issues

- **Missing Tool**: `query_table_by_cell_coordinates` is not implemented yet, affecting retrieval accuracy.
- **Evaluation Accuracy**: Heuristic for program output comparison needs improvements for structural robustness.
- **Answer Comparison**: Logic requires tuning for numeric and percentage-based formats.
