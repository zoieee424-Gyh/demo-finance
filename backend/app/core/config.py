"""
Application configuration constants.

All runtime-tuneable settings read environment variables on init.
"""
from __future__ import annotations

import os
from enum import Enum


# ── Intent types (matching pre-existing consultation.py literals) ─

class IntentType(str, Enum):
    """Five core financial service intents."""
    ADVISORY = "advisory"
    REPORT_ANALYSIS = "financial_report"
    RISK_CONTROL = "risk_control"
    COMPLIANCE = "compliance"
    EDUCATION = "education"


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies. MVP uses SEMANTIC only."""
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


# ── Settings (expected by app/main.py) ──────────────────────────

class Settings:
    project_name: str = "智慧金融服务平台"
    version: str = "0.1.0"
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ]

    # LLM configuration (MVP: disabled by default)
    llm_model: str = "deepseek-v4-flash"
    llm_api_key: str | None = None
    llm_enabled: bool = False
    llm_mock_mode: bool = True

    # ── RAG / Chroma configuration ───────────────────────────────

    # Enable Chroma-based retrieval in the service layer.
    # When False (default), KnowledgeRetriever always uses mock fallback.
    # When True, the service attempts to initialise ChromaStore; if Chroma
    # is unavailable or has no data it silently degrades to mock.
    rag_enabled: bool = False

    # Directory for Chroma's on-disk persistence.
    # Only used when rag_enabled=True.
    chroma_dir: str = "data/chroma"

    # Embedding model used for query vectorization at search time.
    embedding_model: str = "text-embedding-v4"

    # ── Retrieval mode ────────────────────────────────────────────

    # Controls the retrieval strategy used by KnowledgeRetriever.
    #   "semantic" — Chroma semantic search (default, fully functional).
    #   "bm25"     — Sparse keyword retrieval (placeholder shell, returns empty).
    #   "hybrid"   — Semantic + sparse fusion (placeholder; currently
    #                equivalent to semantic, reserved for future RRF).
    retrieval_mode: str = "semantic"

    _VALID_RETRIEVAL_MODES: set[str] = {"semantic", "bm25", "hybrid"}

    # ── Advisor mode configuration ────────────────────────────────

    # Primary mode selector for the investment advisor.
    #   "deepagent" — DeepAgent is the primary architecture.
    #       Falls back to pipeline only when DeepAgent is unavailable.
    #   "pipeline"  — Legacy hard-coded pipeline. No DeepAgent init.
    #   "auto"      — DeepAgent-first with fallback (semantic alias
    #       for deepagent in this build; reserved for future grayscale).
    advisor_mode: str = "deepagent"

    # Deprecated: FIN_AGENT_USE_DEEPAGENT_ADVISOR is still read for
    # backward compatibility.  true → "deepagent", false → "pipeline".
    # FIN_AGENT_ADVISOR_MODE takes precedence when both are set.
    use_deepagent_advisor: bool = False

    # ── MySQL metadata store configuration ──────────────────────────

    # When False (default), all MySQL operations are no-ops.
    # When True, the metadata store attempts to connect; failures are
    # reported but do not crash the application.
    mysql_enabled: bool = False

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str | None = None
    mysql_database: str = "fin_agent_knowledge"

    def __init__(self) -> None:
        # ── LLM env vars ─────────────────────────────────────────
        env_enabled = os.getenv("FIN_AGENT_LLM_ENABLED")
        if env_enabled is not None:
            self.llm_enabled = env_enabled.lower() in ("1", "true", "yes", "on")
        env_mock = os.getenv("FIN_AGENT_LLM_MOCK_MODE")
        if env_mock is not None:
            self.llm_mock_mode = env_mock.lower() in ("1", "true", "yes", "on")
        env_model = os.getenv("FIN_AGENT_LLM_MODEL")
        if env_model:
            self.llm_model = env_model

        # ── RAG env vars ─────────────────────────────────────────
        env_rag = os.getenv("FIN_AGENT_RAG_ENABLED")
        if env_rag is not None:
            self.rag_enabled = env_rag.lower() in ("1", "true", "yes", "on")

        env_chroma_dir = os.getenv("FIN_AGENT_CHROMA_DIR")
        if env_chroma_dir:
            self.chroma_dir = env_chroma_dir

        env_emb_model = os.getenv("FIN_AGENT_EMBEDDING_MODEL")
        if env_emb_model:
            self.embedding_model = env_emb_model

        # ── Retrieval mode env var ──────────────────────────────
        env_ret_mode = os.getenv("FIN_AGENT_RETRIEVAL_MODE")
        if env_ret_mode:
            mode = env_ret_mode.strip().lower()
            if mode in self._VALID_RETRIEVAL_MODES:
                self.retrieval_mode = mode
            # Invalid values are silently ignored → stays "semantic"

        env_cors = os.getenv("FIN_AGENT_CORS_ORIGINS")
        if env_cors:
            self.cors_origins = [
                item.strip()
                for item in env_cors.split(",")
                if item.strip()
            ]

        # ── Advisor mode env vars ────────────────────────────────────
        env_mode = os.getenv("FIN_AGENT_ADVISOR_MODE")
        if env_mode is not None and env_mode.strip():
            mode = env_mode.strip().lower()
            if mode in ("deepagent", "pipeline", "auto"):
                self.advisor_mode = mode

        # Backward compat: FIN_AGENT_USE_DEEPAGENT_ADVISOR
        # Only applies if FIN_AGENT_ADVISOR_MODE was NOT explicitly set.
        env_use_deep = os.getenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR")
        if env_use_deep is not None and env_mode is None:
            if env_use_deep.lower() in ("1", "true", "yes", "on"):
                self.advisor_mode = "deepagent"
                self.use_deepagent_advisor = True
            else:
                self.advisor_mode = "pipeline"
                self.use_deepagent_advisor = False

        # ── MySQL env vars ─────────────────────────────────────────
        env_mysql = os.getenv("FIN_AGENT_MYSQL_ENABLED")
        if env_mysql is not None:
            self.mysql_enabled = env_mysql.lower() in ("1", "true", "yes", "on")

        env_mysql_host = os.getenv("MYSQL_HOST")
        if env_mysql_host:
            self.mysql_host = env_mysql_host

        env_mysql_port = os.getenv("MYSQL_PORT")
        if env_mysql_port:
            try:
                self.mysql_port = int(env_mysql_port)
            except ValueError:
                pass

        env_mysql_user = os.getenv("MYSQL_USER")
        if env_mysql_user:
            self.mysql_user = env_mysql_user

        env_mysql_password = os.getenv("MYSQL_PASSWORD")
        if env_mysql_password:
            self.mysql_password = env_mysql_password

        env_mysql_db = os.getenv("MYSQL_DATABASE")
        if env_mysql_db:
            self.mysql_database = env_mysql_db


settings = Settings()

# ── Compliance constants ────────────────────────────────────────

RISK_NOTICE = (
    "投资有风险，入市需谨慎。"
    "本回答仅作信息参考，不构成投资决策依据，"
    "请根据自身风险承受能力独立判断。"
)

MAX_WARNINGS_BEFORE_BLOCK = 5
