"""
The router is the core of Modality. It takes an incoming request and decides
which fine-tuned SLM should handle it — or falls back to a large model.

Strategy:
1. Embed the incoming prompt using a cheap embedding model.
2. Compare against the domain embeddings of all active fine-tuned models.
3. Pick the best match above the confidence threshold.
4. If nothing matches, fall back to the configured large model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from modality.config import settings
from modality.providers.embeddings import get_embedding
from modality.registry.service import get_active_models
from modality.router.schemas import RouteDecision

if TYPE_CHECKING:
    from modality.router.cache import RoutingCache


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


async def route_request(
    prompt: str,
    db: AsyncSession,
    customer_id: int | None = None,
    cache: RoutingCache | None = None,
) -> RouteDecision:
    """Decide which model should handle this prompt.

    If a RoutingCache is provided (data plane), uses in-memory models.
    Otherwise falls back to querying the DB directly (control plane / tests).
    """

    if cache is not None:
        # Hot path — use cached models
        if customer_id is not None:
            candidates = cache.get_models_for_customer(customer_id)
        else:
            candidates = cache.models

        if not candidates:
            return _fallback("No active fine-tuned models available")

        prompt_embedding = await get_embedding(prompt)

        best = None
        best_score = -1.0
        for model in candidates:
            if model.domain_embedding is None:
                continue
            score = cosine_similarity(prompt_embedding, model.domain_embedding)
            if score > best_score:
                best_score = score
                best = model

        if best is not None and best_score >= settings.router_confidence_threshold:
            return RouteDecision(
                model_id=best.id,
                provider=best.provider,
                provider_model_id=best.provider_model_id,
                confidence=best_score,
                is_fallback=False,
                reason=f"Matched domain '{best.domain}' (score={best_score:.3f})",
            )

        return _fallback(
            f"Best score {best_score:.3f} below threshold {settings.router_confidence_threshold}",
            confidence=max(best_score, 0.0),
        )

    # Slow path — query DB directly
    active_models = await get_active_models(db)

    if customer_id is not None:
        active_models = [m for m in active_models if m.customer_id == customer_id]

    if not active_models:
        return _fallback("No active fine-tuned models available")

    prompt_embedding = await get_embedding(prompt)

    best_model = None
    best_score = -1.0

    for model in active_models:
        if model.domain_embedding is None:
            continue
        score = cosine_similarity(prompt_embedding, model.domain_embedding)
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is not None and best_score >= settings.router_confidence_threshold:
        return RouteDecision(
            model_id=best_model.id,
            provider=best_model.provider.value,
            provider_model_id=best_model.provider_model_id,
            confidence=best_score,
            is_fallback=False,
            reason=f"Matched domain '{best_model.domain}' (score={best_score:.3f})",
        )

    return _fallback(
        f"Best score {best_score:.3f} below threshold {settings.router_confidence_threshold}",
        confidence=max(best_score, 0.0),
    )


def _fallback(reason: str, confidence: float = 0.0) -> RouteDecision:
    return RouteDecision(
        model_id=None,
        provider=settings.fallback_provider,
        provider_model_id=settings.fallback_model,
        confidence=confidence,
        is_fallback=True,
        reason=reason,
    )
