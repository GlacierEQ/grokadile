# Copyright 2024 GlacierEQ / Casey Barton
# FastAPI inference gateway — exposes MegatronBrain over HTTP
# Deploy: uvicorn api.main:app --reload
# Vercel: serverless via vercel.json

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Optional Sentry ──────────────────────────────────────────────────────────
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    _dsn = os.getenv("SENTRY_DSN")
    if _dsn:
        sentry_sdk.init(
            dsn=_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.2")),
        )
except ImportError:
    pass

logger = logging.getLogger("stealth_api")

app = FastAPI(
    title="Grokadile Stealth API",
    description="Grok-1 × Megatron-DeepSpeed multi-agent inference gateway",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy singleton brain ──────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_brain():
    """Singleton MegatronBrain — loaded once per worker process."""
    from stealth_brain import MegatronBrain  # local import to avoid cold-start cost
    brain = MegatronBrain(
        checkpoint_path=os.getenv("CKPT_PATH", "./checkpoints/"),
        tokenizer_path=os.getenv("TOKENIZER_PATH", "./tokenizer.model"),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        use_deepspeed=os.getenv("USE_DEEPSPEED", "1") == "1",
    )
    brain.initialize()
    return brain


# ── Request / Response models ─────────────────────────────────────────────────
class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt")
    agent_name: str = Field(default="STEALTH", description="Agent identity tag")
    mission: str = Field(default="", description="Agent mission context")
    max_len: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class InferResponse(BaseModel):
    agent: str
    response: str
    prompt_len: int
    response_len: int


class BroadcastRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    agents: Optional[list[str]] = Field(
        default=None,
        description="Subset of agents to query. None = all agents.",
    )
    max_len: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BroadcastResponse(BaseModel):
    results: dict[str, str]
    agent_count: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "grokadile-stealth-api"}


@app.get("/agents", tags=["Agents"])
async def list_agents():
    """Return all registered Stealth agent names and missions."""
    from stealth_terminal import STEALTH_AGENTS
    return {
        name: {"emoji": cfg["emoji"], "mission": cfg["mission"]}
        for name, cfg in STEALTH_AGENTS.items()
    }


@app.post("/infer", response_model=InferResponse, tags=["Inference"])
async def infer(
    req: InferRequest,
    brain=Depends(get_brain),
):
    """
    Route a single prompt through the Megatron brain as a named agent.
    """
    try:
        response = brain.think(
            prompt=req.prompt,
            agent_name=req.agent_name,
            mission=req.mission,
            max_len=req.max_len,
            temperature=req.temperature,
        )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))

    return InferResponse(
        agent=req.agent_name,
        response=response,
        prompt_len=len(req.prompt),
        response_len=len(response),
    )


@app.post("/broadcast", response_model=BroadcastResponse, tags=["Inference"])
async def broadcast(
    req: BroadcastRequest,
    brain=Depends(get_brain),
):
    """
    Broadcast a prompt to multiple Stealth agents simultaneously.
    Returns all responses in one payload.
    """
    from stealth_terminal import STEALTH_AGENTS
    import asyncio

    target_agents = {
        k: v for k, v in STEALTH_AGENTS.items()
        if req.agents is None or k in req.agents
    }
    if not target_agents:
        raise HTTPException(status_code=400, detail="No matching agents found.")

    async def _run(name: str, cfg: dict) -> tuple[str, str]:
        response = await asyncio.to_thread(
            brain.think,
            req.prompt,
            agent_name=name.upper(),
            mission=cfg["mission"],
            max_len=req.max_len,
            temperature=req.temperature,
        )
        return name, response

    try:
        pairs = await asyncio.gather(*[_run(n, c) for n, c in target_agents.items()])
    except Exception as exc:
        logger.exception("Broadcast error")
        raise HTTPException(status_code=500, detail=str(exc))

    return BroadcastResponse(
        results=dict(pairs),
        agent_count=len(pairs),
    )
