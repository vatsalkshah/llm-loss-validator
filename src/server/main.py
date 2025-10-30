"""FastAPI application entrypoint for the inference API."""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .generator import GenerationPipeline
from .middleware import APIKeyValidator, RequestLoggingMiddleware
from .model_loader import ModelLoader
from .schemas import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LLM Inference API", version="1.0.0")
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    loader = ModelLoader()
    pipeline = GenerationPipeline(loader)
    api_key_validator = APIKeyValidator()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/generate", response_model=ChatCompletionResponse)
    async def generate(
        request: ChatCompletionRequest,
        _: None = Depends(api_key_validator),
    ) -> ChatCompletionResponse:
        result = pipeline.generate(request)
        return result

    @app.post("/v1/generate/stream")
    async def generate_stream(
        request: ChatCompletionRequest,
        _: None = Depends(api_key_validator),
    ) -> StreamingResponse:
        stream = pipeline.generate_stream(request)

        def iter_sse():
            for chunk in stream:
                yield f"data: {chunk.json()}\n\n"

        return StreamingResponse(iter_sse(), media_type="text/event-stream")

    @app.exception_handler(Exception)
    async def default_exception_handler(_, exc: Exception):  # pragma: no cover - safety net
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="internal_error", detail=str(exc)).dict(),
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]
