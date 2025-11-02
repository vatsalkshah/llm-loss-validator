from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import get_config
from src.core.model_registry import get_registry
from src.orchestrator import get_orchestrator


def create_app() -> FastAPI:
    app = FastAPI(title="Validator-Orchestrator API")
    orchestrator = get_orchestrator()

    @app.on_event("startup")
    async def _startup() -> None:
        await orchestrator.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await orchestrator.stop()

    @app.get("/")
    async def root():
        return {"ok": True}

    @app.get("/heartbeat")
    async def heartbeat():
        return orchestrator.heartbeat_snapshot()

    @app.get("/models")
    async def models():
        reg = get_registry()
        return {"loaded": reg.list_loaded(), "cached": reg.list_loaded()}

    @app.post("/inference")
    async def inference(body: Dict[str, Any]):
        model_id = body.get("model_id")
        prompt = body.get("prompt")
        gen = body.get("gen") or {}
        if not model_id or not prompt:
            return JSONResponse(status_code=400, content={"detail": "model_id and prompt are required"})
        text, usage = await orchestrator.run_inference(model_id, prompt, gen)
        return {"text": text, "usage": usage}

    @app.websocket("/ws/inference")
    async def ws_inference(ws: WebSocket):
        await ws.accept()
        try:
            init_msg = await ws.receive_text()
            payload = json.loads(init_msg)
            model_id = payload.get("model_id")
            prompt = payload.get("prompt")
            gen = payload.get("gen") or {}
            if not model_id or not prompt:
                await ws.send_text(json.dumps({"error": "model_id and prompt are required"}))
                await ws.close(code=1003)
                return

            idx = 0
            async for token in orchestrator.stream_inference(model_id, prompt, gen):
                await ws.send_text(json.dumps({"token": token, "idx": idx}))
                idx += 1
            await ws.send_text(json.dumps({"complete": True, "usage": {}}))
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            try:
                await ws.send_text(json.dumps({"error": str(e)}))
            finally:
                await ws.close(code=1011)

    return app
