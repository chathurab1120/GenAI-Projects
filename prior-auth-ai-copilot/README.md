# PA-Genie: Prior Authorization AI Copilot

A hands-on GenAI project exploring how large language models, retrieval-augmented 
generation, and workflow orchestration can be applied to healthcare 
prior authorization review.

Built entirely with synthetic data to practice end-to-end AI application 
development — from document ingestion and vector search through to 
structured LLM outputs and a working UI.

---

## Concept

Prior authorization is a healthcare process where insurers review whether 
a requested treatment or service meets medical necessity criteria before 
approving coverage. The review involves comparing clinical notes against 
policy guidelines — a document-heavy, repetitive task that is well-suited 
to a RAG-based AI workflow.

This project builds an AI copilot that:

- Accepts a prior authorization request and clinical note
- Retrieves relevant policy criteria from a vector knowledge base
- Summarizes the clinical case using an LLM
- Extracts structured evidence from the clinical note
- Compares evidence against policy criteria
- Generates a structured decision: **APPROVE / DENY / NEED MORE INFO**
- Produces a clinical reviewer note with policy citations
- Logs every review to an audit database

---

## What I learned building this

- How to build a full RAG pipeline from document ingestion to vector retrieval
- How to use LangGraph to orchestrate multi-step LLM workflows as explicit nodes
- How to structure LLM outputs using Pydantic for reliable parsing
- How to wrap OpenAI API calls with retry logic, token logging, and fallback handling
- How to separate prompts from code and keep them versioned as templates
- How to build a FastAPI backend and Streamlit frontend that work together
- How to write pytest tests for AI pipelines using mocks — no real API calls in tests
- How to think about responsible AI design: citations, disclaimers, human review

---

## Architecture
User Request
│
▼
FastAPI /review endpoint
│
▼
LangGraph Workflow (8 nodes)
│
├── intake_node                 validates input fields
├── retrieve_policy_node        ChromaDB vector search
├── summarize_case_node         GPT-4o-mini case summary
├── extract_evidence_node       structured evidence extraction
├── compare_criteria_node       evidence vs policy criteria
├── recommend_decision_node     APPROVE / DENY / NEED_MORE_INFO
├── generate_reviewer_note_node clinical reviewer note
└── audit_log_node              SQLite audit record
│
▼
Structured JSON Response
│
▼
Streamlit UI

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Workflow orchestration | LangGraph |
| LLM framework | LangChain |
| Vector store | ChromaDB (local persistent) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Data models | Pydantic v2 |
| Database | SQLite via SQLAlchemy |
| Document parsing | pypdf |
| Retry logic | Tenacity |
| Testing | pytest |
| Config management | pydantic-settings + python-dotenv |

---

## Project structure
prior-auth-ai-copilot/
├── app/
│   ├── api/                   FastAPI routes and Pydantic schemas
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   └── auth_review.py
│   │   └── schemas/
│   │       ├── request_models.py
│   │       └── response_models.py
│   ├── core/                  Config, logging, constants
│   ├── ingestion/             Document loading and chunking
│   ├── retrieval/             Embeddings and ChromaDB vector search
│   ├── workflows/             LangGraph nodes, graph, state
│   ├── llm/                   LLM factory, prompts, output parsers
│   └── db/                    SQLite models and session
├── frontend/
│   └── streamlit_app.py       Streamlit UI
├── data/
│   ├── policy_docs/           Synthetic policy documents (Markdown)
│   ├── sample_requests/       Synthetic PA request JSON files
│   └── eval/                  Gold evaluation cases
├── tests/                     pytest test suite — 22 tests, all mocked
├── scripts/                   Verification and ingestion scripts
├── .env.example               Environment variable template
├── requirements.txt           Python dependencies
└── CONTEXT.md                 Project context for AI-assisted development

---

## Local setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/prior-auth-ai-copilot.git
cd prior-auth-ai-copilot
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:
OPENAI_API_KEY="your-key-here"

### 5. Run the backend
```bash
uvicorn app.api.main:app --reload
```

API available at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

### 6. Run the frontend

Open a second terminal:
```bash
streamlit run frontend\streamlit_app.py
```

UI available at `http://localhost:8501`

---

## Running tests
```bash
pytest tests/ -v
```

22 tests covering API endpoints, output schema validation,
document chunking, and workflow logic. All LLM and API calls
are mocked — no real OpenAI calls during testing.

---

## Demo scenarios

Three synthetic prior auth cases are included in the UI sidebar:

| Case | Service requested | Expected decision |
|---|---|---|
| PA-DEMO-001 | Lumbar spine MRI | APPROVE |
| PA-DEMO-002 | Sleep study (incomplete note) | NEED MORE INFO |
| PA-DEMO-003 | Biologic therapy for RA | APPROVE |

---

## Sample API request
```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "PA-2024-001",
    "patient_age": 52,
    "diagnosis": "Low back pain with radiculopathy",
    "requested_service": "MRI Lumbar Spine without contrast",
    "provider_specialty": "Orthopedic Surgery",
    "clinical_note_text": "Patient has 8 weeks of low back pain radiating to the left leg. Completed 8 PT sessions and naproxen 500mg BID. Positive straight leg raise at 45 degrees."
  }'
```

## Sample API response
```json
{
  "case_id": "PA-2024-001",
  "decision": "APPROVE",
  "confidence": 0.95,
  "case_summary": "52-year-old male with 8 weeks of low back pain...",
  "rationale": "All criteria met including symptom duration, conservative therapy, and neurological findings.",
  "missing_information": [],
  "criteria_results": [
    {
      "criterion": "Duration of symptoms",
      "status": "MET",
      "evidence": "8 weeks documented",
      "chunk_id": "mri_lumbar_spine_policy.md::chunk_0001"
    }
  ],
  "reviewer_note": "Prior authorization approved. All medical necessity criteria satisfied...",
  "citations": ["mri_lumbar_spine_policy.md::chunk_0001"],
  "disclaimer": "AI-generated draft — human review required",
  "prompt_tokens_total": 1800,
  "completion_tokens_total": 600
}
```

---

## Key design decisions

**LangGraph for workflow orchestration**  
Each review step is a separate node with a single responsibility.
This makes the workflow explicit, easy to debug, and straightforward
to extend — adding a new step means adding a new node without
touching existing logic.

**ChromaDB for local vector storage**  
Keeps the project fully self-contained with no external services
required. The vector store path is configurable via `.env` so it
can be swapped to a hosted vector database for production use.

**Pydantic v2 for structured LLM outputs**  
Every LLM response is validated against a strict Pydantic schema
before being used downstream. Malformed outputs fall back to a
safe default rather than crashing the pipeline.

**Prompts as code-separate templates**  
All prompts live in `app/llm/prompts.py` as named constants and
builder functions. No prompt strings are scattered across the
codebase — making them easy to version, test, and iterate on.

**gpt-4o-mini for cost-effective development**  
Each full 8-node review costs approximately $0.001–$0.003 in tokens.
The model is configurable via `.env` — swap to gpt-4o or Azure
OpenAI without changing any application code.

---

## Responsible AI notes

- All outputs include a mandatory disclaimer: *"AI-generated draft — human review required"*
- No real patient data is used anywhere in this project
- Policy citations are included in every response for transparency
- This project is a learning exercise — not intended for real clinical use

---

## Future directions

- Azure OpenAI provider support switchable via environment variable
- Evaluation pipeline running against the gold standard cases in `data/eval/`
- Human-in-the-loop approval step before the audit log node
- PostgreSQL for production-scale audit logging
- React frontend with PDF clinical note upload
- LangSmith integration for LLM call tracing and monitoring

---

*PA-Genie v0.1.0 — Built with synthetic data — Not for real clinical use*
