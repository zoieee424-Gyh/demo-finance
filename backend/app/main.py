from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.routes.consultations import router as consultations_router
from app.api.routes.health import router as health_router
from app.api.routes.investment_advisor_debug import router as investment_advisor_debug_router
from app.api.routes.financial_report_debug import router as financial_report_debug_router
from app.api.routes.risk_control_debug import router as risk_control_debug_router
from app.api.routes.compliance_debug import router as compliance_debug_router
from app.api.routes.education_debug import router as education_debug_router
from app.api.routes.debug_stream import router as debug_stream_router
from app.api.routes.agent_query import router as agent_query_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="Agentic RAG financial service platform.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(consultations_router, prefix="/api")
    app.include_router(investment_advisor_debug_router, prefix="/api")
    app.include_router(financial_report_debug_router, prefix="/api")
    app.include_router(risk_control_debug_router, prefix="/api")
    app.include_router(compliance_debug_router, prefix="/api")
    app.include_router(education_debug_router, prefix="/api")
    app.include_router(debug_stream_router, prefix="/api")
    app.include_router(agent_query_router, prefix="/api")

    # ── Startup warmup via lifespan (no deprecation) ──────────
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(application: FastAPI):
        import logging
        logger = logging.getLogger("app.main")
        if settings.advisor_mode in ("deepagent", "auto"):
            logger.info("Warming up advisory agent (mode=%s) ...", settings.advisor_mode)
            from app.services.advisory_agent_factory import warmup_advisory_agent
            ok = warmup_advisory_agent(settings.advisor_mode)
            if ok:
                logger.info("Advisory agent warmup complete.")
            else:
                logger.warning("Advisory agent warmup failed — will lazy-init on first request.")
        else:
            logger.info("Advisor mode is 'pipeline' — skipping DeepAgent warmup.")
        yield

    app.router.lifespan_context = _lifespan

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app


app = create_app()


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.getenv("FIN_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("FIN_AGENT_PORT", "8010"))
    uvicorn.run(app, host=host, port=port)
