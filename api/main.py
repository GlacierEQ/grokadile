"""Grokadile Inference API — FastAPI wrapper over runners.py for Vercel serverless deploy."""
import os
import sys
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Add repo root to path so runners.py / model.py are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    environment=os.getenv("ENV", "production"),
    release=os.getenv("MODEL_VERSION", "dev"),
)

app = FastAPI(
    title="Grokadile Inference API",
    version="1.0.0",
    description="JAX-based transformer inference endpoint",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.getenv("API_SECRET_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class InferResponse(BaseModel):
    output: str
    tokens_used: int
    model_version: str


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest, _key: str = Security(_verify_key)):
    """Run inference via runners.py generate()."""
    try:
        from runners import generate  # type: ignore

        output = generate(req.prompt, req.max_tokens, req.temperature)

        # Fire-and-forget: store to Supermemory + vector stores asynchronously
        try:
            from integrations.supermemory import store_inference_memory
            store_inference_memory(
                prompt=req.prompt,
                output=output,
                meta={"model_version": os.getenv("MODEL_VERSION", "dev")},
            )
        except Exception:
            pass  # Non-critical; Sentry will catch if needed

        return InferResponse(
            output=output,
            tokens_used=len(output.split()),
            model_version=os.getenv("MODEL_VERSION", "latest"),
        )
    except Exception as exc:
        sentry_sdk.capture_exception(exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODEL_VERSION", "latest"), "env": os.getenv("ENV", "production")}
