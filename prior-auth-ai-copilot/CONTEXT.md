# Project Context — Prior Auth AI Copilot

## What this project does
A GenAI healthcare copilot for prior authorization review.
Accepts a prior auth request and clinical notes, retrieves matching
policy criteria, extracts evidence, and generates a structured
recommendation: APPROVE / DENY / NEED_MORE_INFO with citations.

## Stack
- Python 3.14
- FastAPI backend
- Streamlit frontend
- LangChain + LangGraph workflow orchestration
- ChromaDB local vector store
- OpenAI gpt-4o-mini (LLM)
- OpenAI text-embedding-3-small (embeddings)
- SQLite audit logging
- Pydantic v2 data models

## Folder purposes
- app/api/ — FastAPI routes and Pydantic schemas
- app/core/ — config, logging, constants
- app/ingestion/ — document loading and chunking
- app/retrieval/ — embeddings and vector search
- app/workflows/ — LangGraph nodes, graph, state
- app/llm/ — LLM factory, prompts, output parsers
- app/services/ — review, citation, audit services
- app/db/ — SQLite models and session
- app/utils/ — shared helpers
- frontend/ — Streamlit UI
- data/ — synthetic sample data only
- tests/ — pytest tests

## Key rules
- All LLM calls go through app/llm/llm_factory.py
- All prompts loaded from app/llm/prompts.py
- All outputs validated with Pydantic before returning
- temperature=0.0 for all decision calls
- Never log PHI — synthetic data only
- Never call real OpenAI API in tests — always mock
