"""API key authentication for the data plane."""

import hashlib

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from modality.registry.models import ApiKey, Customer

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def authenticate(
    db: AsyncSession,
    raw_key: str | None,
) -> Customer:
    """Validate an API key and return the associated customer."""
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Strip "Bearer " prefix if present
    if raw_key.startswith("Bearer "):
        raw_key = raw_key[7:]

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
        .options(selectinload(ApiKey.customer))
    )
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key.customer
