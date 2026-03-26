"""
Data Plane — the hot path that customers call for inference.

This is what the customer's app hits instead of calling OpenAI directly.
It authenticates the request, routes to the best model, runs inference,
and logs usage for billing.

Deployed separately from the control plane. Must be fast and stateless
(reads from cache, not DB on every request).
"""

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncSession

from modality.gateway.auth import authenticate
from modality.gateway.schemas import ChatMessage, ChatRequest, ChatResponse
from modality.providers.registry import get_provider
from modality.registry.database import get_db, init_db
from modality.registry.models import Customer, UsageLog
from modality.router.cache import RoutingCache
from modality.router.router import route_request

routing_cache = RoutingCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await routing_cache.refresh()
    yield


app = FastAPI(
    title="Modality — Data Plane",
    description="Customer-facing inference API with intelligent routing",
    version="0.1.0",
    lifespan=lifespan,
)


async def get_customer(request: Request, db: AsyncSession = Depends(get_db)) -> Customer:
    raw_key = request.headers.get("Authorization")
    return await authenticate(db, raw_key)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    customer: Customer = Depends(get_customer),
    db: AsyncSession = Depends(get_db),
):
    start = time.monotonic()

    # Route using cached model embeddings
    prompt = request.messages[-1].content if request.messages else ""
    decision = await route_request(
        prompt, db, customer_id=customer.id, cache=routing_cache
    )

    # Run inference
    provider = get_provider(decision.provider)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    result = await provider.create_completion(
        model=decision.provider_model_id,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    # Log usage asynchronously (non-blocking in prod, inline here for simplicity)
    usage_log = UsageLog(
        customer_id=customer.id,
        model_id=decision.model_id,
        provider=decision.provider,
        provider_model_id=decision.provider_model_id or "",
        is_fallback=decision.is_fallback,
        prompt_tokens=result.usage.get("prompt_tokens", 0),
        completion_tokens=result.usage.get("completion_tokens", 0),
        estimated_cost_usd=_estimate_cost(result.usage, decision.is_fallback),
        latency_ms=latency_ms,
    )
    db.add(usage_log)
    await db.commit()

    return ChatResponse(
        content=result.content,
        model=result.model,
        provider=decision.provider,
        is_fallback=decision.is_fallback,
        confidence=decision.confidence,
        routing_reason=decision.reason,
        usage=result.usage,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "plane": "data"}


def _estimate_cost(usage: dict, is_fallback: bool) -> float:
    """Rough cost estimate. In prod, pull actual rates from provider config."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    if is_fallback:
        # GPT-4o pricing (rough)
        return (prompt_tokens * 2.50 + completion_tokens * 10.00) / 1_000_000
    else:
        # Fine-tuned SLM pricing (rough — much cheaper)
        return (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
