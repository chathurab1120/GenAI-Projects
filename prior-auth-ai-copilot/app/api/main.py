from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, auth_review
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.db.init_db import init_db

setup_logging()
init_db()

settings = get_settings()

app = FastAPI(
    title="Prior Auth AI Copilot",
    description=(
        "GenAI-powered prior authorization review copilot. "
        "Uses RAG and LangGraph workflow to generate structured "
        "approve/deny/need-more-info recommendations. "
        "All outputs are AI-generated drafts — human review required."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(auth_review.router, tags=["Prior Auth Review"])


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
