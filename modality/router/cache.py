"""
In-memory cache of active models + their embeddings for the data plane.

The data plane should NOT hit the DB on every request. Instead, it loads
the routing table into memory and refreshes periodically (or on push from
the control plane).
"""

import asyncio
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from modality.registry.database import async_session
from modality.registry.models import FineTunedModel, ModelStatus


@dataclass
class CachedModel:
    id: int
    customer_id: int
    provider: str
    provider_model_id: str
    domain: str
    domain_embedding: list[float] | None


class RoutingCache:
    def __init__(self, refresh_interval_seconds: int = 30):
        self._models: list[CachedModel] = []
        self._refresh_interval = refresh_interval_seconds
        self._refresh_task: asyncio.Task | None = None

    @property
    def models(self) -> list[CachedModel]:
        return self._models

    async def refresh(self):
        """Pull all active models from DB into memory."""
        async with async_session() as db:
            result = await db.execute(
                select(FineTunedModel).where(FineTunedModel.status == ModelStatus.active)
            )
            models = result.scalars().all()

            self._models = [
                CachedModel(
                    id=m.id,
                    customer_id=m.customer_id,
                    provider=m.provider.value,
                    provider_model_id=m.provider_model_id or "",
                    domain=m.domain,
                    domain_embedding=m.domain_embedding,
                )
                for m in models
            ]

    async def start_background_refresh(self):
        """Start a background loop that refreshes the cache periodically."""
        async def _loop():
            while True:
                await asyncio.sleep(self._refresh_interval)
                try:
                    await self.refresh()
                except Exception:
                    pass  # log in prod — don't crash the refresh loop

        self._refresh_task = asyncio.create_task(_loop())

    def get_models_for_customer(self, customer_id: int) -> list[CachedModel]:
        return [m for m in self._models if m.customer_id == customer_id]
