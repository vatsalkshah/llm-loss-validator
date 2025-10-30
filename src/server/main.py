"""
FastAPI inference server for LLM chat completions.

This module provides an OpenAI-compatible inference API with support
for streaming and non-streaming responses, integrating with the existing
model cache and loading infrastructure.
"""

import uuid
import time
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
import torch

from server.config import config
from server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
    ChatMessageRole,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessageDelta,
    ErrorResponse,
)
from server.model_loader import ModelLoader, ModelNotCachedError
from server.generator import TextGenerator
from server.middleware import verify_api_key, api_key_header, RequestLoggingMiddleware
from core.model_cache import ModelCacheManager


# Global state
model_loader: Optional[ModelLoader] = None
cache_manager: Optional[ModelCacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    
    Initializes the model cache and loader on startup,
    and cleans up resources on shutdown.
    """
    global model_loader, cache_manager
    
    # Startup
    logger.info("Starting inference API server")
    logger.info(f"Config: host={config.host}, port={config.port}")
    logger.info(f"API Key Auth: {config.require_api_key}")
    logger.info(f"TLS Enabled: {config.tls_enabled}")
    
    # Validate TLS config if enabled
    if config.tls_enabled and not config.validate_tls_config():
        logger.error("TLS enabled but certificate/key files not found")
        raise ValueError("Invalid TLS configuration")
    
    # Initialize cache manager
    cache_manager = ModelCacheManager(
        cache_dir=config.cache_dir,
        hf_token=config.hf_token,
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize model loader
    model_loader = ModelLoader(
        cache_manager=cache_manager,
        hf_token=config.hf_token,
        device=device,
        torch_dtype="auto",
        reject_non_cached=config.reject_non_cached_models,
    )
    
    logger.info("Inference API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference API server")
    
    # Unload all models
    if model_loader:
        loaded_models = model_loader.list_loaded_models()
        logger.info(f"Unloading {len(loaded_models)} models")
        for model_key in loaded_models:
            # Parse model_id from cache_key
            model_id, revision = model_key.rsplit("@", 1) if "@" in model_key else (model_key, "main")
            model_loader.unload_model(model_id, revision)
    
    logger.info("Inference API server stopped")


# Create FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="OpenAI-compatible inference API for language models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "service": "LLM Inference API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models(api_key: Optional[str] = Depends(api_key_header)):
    """
    List available models.
    
    Returns both cached and loaded models.
    """
    await verify_api_key(api_key)
    
    if not cache_manager or not model_loader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    # Get cached models
    cached_models = cache_manager.list_models()
    loaded_models = model_loader.list_loaded_models()
    
    models = []
    for model_info in cached_models:
        model_id = model_info["model_id"]
        cache_key = f"{model_id}@{model_info['revision']}"
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(model_info["download_timestamp"]),
            "owned_by": "local",
            "cached": True,
            "loaded": cache_key in loaded_models,
        })
    
    return {"object": "list", "data": models}


async def generate_stream_response(
    request: ChatCompletionRequest,
    generator: TextGenerator,
):
    """
    Generate streaming response in SSE format.
    
    Args:
        request: Chat completion request
        generator: Text generator instance
    
    Yields:
        SSE-formatted chunks
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())
    
    # First chunk with role
    first_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatMessageDelta(role=ChatMessageRole.ASSISTANT, content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"
    
    # Stream generated content
    stream_iter = generator.generate_stream(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
    )
    
    try:
        for chunk in stream_iter:
            chunk_response = ChatCompletionStreamResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatMessageDelta(content=chunk["text"]),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk_response.model_dump_json()}\n\n"
        
        # Final chunk with finish_reason
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatMessageDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error during streaming generation: {e}")
        error_chunk = {
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(api_key_header),
):
    """
    Create a chat completion.
    
    This endpoint is OpenAI-compatible and supports both streaming
    and non-streaming responses.
    """
    await verify_api_key(api_key)
    
    if not model_loader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized"
        )
    
    # Validate max_tokens
    if request.max_tokens and request.max_tokens > config.max_gen_tokens_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_tokens exceeds limit of {config.max_gen_tokens_limit}"
        )
    
    # Check if model is cached (if rejection is enabled)
    if config.reject_non_cached_models:
        revision = "main"  # Could be extended to support revision in request
        if not cache_manager.is_model_cached(request.model, revision):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model} is not cached. Please cache the model first."
            )
    
    # Load model
    try:
        model, tokenizer = model_loader.load_model(
            model_id=request.model,
            revision="main",  # Could be extended
        )
    except ModelNotCachedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model} not found in cache"
        )
    except Exception as e:
        logger.error(f"Error loading model {request.model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )
    
    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        model_id=request.model,
    )
    
    # Handle streaming vs non-streaming
    if request.stream:
        return StreamingResponse(
            generate_stream_response(request, generator),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming generation
        try:
            result = generator.generate(
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )
            
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())
            
            response = ChatCompletionResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role=ChatMessageRole.ASSISTANT,
                            content=result["text"],
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["total_tokens"],
                ),
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}"
            )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn_config = {
        "app": "server.main:app",
        "host": config.host,
        "port": config.port,
        "workers": config.workers,
        "log_level": "info",
    }
    
    # Add TLS if enabled
    if config.tls_enabled:
        if config.validate_tls_config():
            uvicorn_config["ssl_certfile"] = config.tls_cert_path
            uvicorn_config["ssl_keyfile"] = config.tls_key_path
            logger.info("TLS enabled")
        else:
            logger.error("TLS configuration invalid, starting without TLS")
    
    uvicorn.run(**uvicorn_config)
