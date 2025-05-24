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
Tomoro-FinQA-Agent/
├── app/
│   ├── api/v1/
│   │   ├── routers/qa.py             # FastAPI QA endpoints
│   │   └── schemas/qa_schemas.py     # Pydantic models
│   ├── agent_components/
│   │   ├── function_caller.py        # Tool registration & execution
│   │   └── prompt_manager.py         # LLM prompt construction
│   ├── clients/llm_client.py         # AsyncOpenAI wrapper
│   ├── core/config.py                # App configuration
│   ├── rag_pipeline/
│   │   └── parser.py                 # Table-to-Markdown converter (legacy)
│   ├── services/qa_service.py        # Core QA logic
│   └── main.py                       # FastAPI app entry point
├── evaluation/
│   ├── datasets/qa_eval_dataset.json
│   └── scripts/
│       ├── generate_eval_script_updated.py
│       ├── run_evaluation_script.py
│       ├── run_evaluation_single_shot.py
│       └── run_evaluation_turns.py
├── prompts/financial_assistant_system_prompt.md
├── scripts/test_parser.py            # TableParser tests
├── tests/unit/
│   ├── agent_components/test_prompt_manager.py
│   ├── clients/test_llm_client.py
│   └── services/test_qa_service.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md                         # You are here
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
