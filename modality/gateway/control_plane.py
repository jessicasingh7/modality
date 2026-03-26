"""
Control Plane — internal management API.

Used by your team and customer dashboards to:
- Onboard customers and issue API keys
- Upload training data and start fine-tuning
- Monitor jobs, view eval results
- Promote/demote models
- View usage and billing data

Deployed separately from the data plane. Does not need to be as fast.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from modality.finetune.data import ValidationError, validate_jsonl
from modality.finetune.pipeline import start_finetune_pipeline
from modality.gateway.schemas import (
    CustomerCreate,
    CustomerInfo,
    ApiKeyCreate,
    ApiKeyResponse,
    FineTuneRequest,
    FineTuneResponse,
    ModelInfo,
    UsageSummary,
)
from modality.registry.database import get_db, init_db
from modality.registry.models import ApiKey, Customer, FineTunedModel, UsageLog
from modality.registry.service import (
    demote_model,
    get_active_models,
    get_customer_models,
    get_or_create_customer,
    promote_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Modality — Control Plane",
    description="Internal management API for customers, models, and fine-tuning",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Customer management
# ---------------------------------------------------------------------------


@app.post("/customers", response_model=CustomerInfo)
async def create_customer(
    request: CustomerCreate,
    db: AsyncSession = Depends(get_db),
):
    customer = await get_or_create_customer(db, request.name, request.domain)
    return CustomerInfo(id=customer.id, name=customer.name, domain=customer.domain)


@app.post("/customers/{customer_id}/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    customer_id: int,
    request: ApiKeyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Issue a new API key for a customer. The raw key is only shown once."""
    raw_key, key_hash = ApiKey.generate()
    api_key = ApiKey(
        customer_id=customer_id,
        key_hash=key_hash,
        key_prefix=raw_key[:12] + "...",
        name=request.name,
    )
    db.add(api_key)
    await db.commit()

    return ApiKeyResponse(
        id=api_key.id,
        key=raw_key,  # only returned once
        key_prefix=api_key.key_prefix,
        name=api_key.name,
    )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


@app.post("/finetune", response_model=FineTuneResponse)
async def create_finetune(
    request: FineTuneRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        validate_jsonl(request.training_file_path)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    customer = await get_or_create_customer(db, request.customer_name, request.domain)

    job = await start_finetune_pipeline(
        db=db,
        customer=customer,
        training_file_path=request.training_file_path,
        base_model=request.base_model,
        provider_name=request.provider,
        domain_description=request.domain_description,
        hyperparameters=request.hyperparameters,
    )

    return FineTuneResponse(
        job_id=job.id,
        provider_job_id=job.provider_job_id,
        status=job.status.value,
        model_name=job.model.name,
    )


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@app.get("/models", response_model=list[ModelInfo])
async def list_models(db: AsyncSession = Depends(get_db)):
    models = await get_active_models(db)
    return [_model_to_info(m) for m in models]


@app.get("/customers/{customer_id}/models", response_model=list[ModelInfo])
async def list_customer_models(
    customer_id: int,
    db: AsyncSession = Depends(get_db),
):
    models = await get_customer_models(db, customer_id)
    return [_model_to_info(m) for m in models]


@app.post("/models/{model_id}/promote", response_model=ModelInfo)
async def promote(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await promote_model(db, model_id)
    return _model_to_info(model)


@app.post("/models/{model_id}/demote", response_model=ModelInfo)
async def demote(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await demote_model(db, model_id)
    return _model_to_info(model)


# ---------------------------------------------------------------------------
# Usage / billing
# ---------------------------------------------------------------------------


@app.get("/customers/{customer_id}/usage", response_model=UsageSummary)
async def get_usage(customer_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(
            func.count(UsageLog.id).label("total_requests"),
            func.sum(UsageLog.prompt_tokens).label("total_prompt_tokens"),
            func.sum(UsageLog.completion_tokens).label("total_completion_tokens"),
            func.sum(UsageLog.estimated_cost_usd).label("total_cost_usd"),
            func.count(UsageLog.id).filter(UsageLog.is_fallback.is_(True)).label("fallback_requests"),
        ).where(UsageLog.customer_id == customer_id)
    )
    row = result.one()
    total = row.total_requests or 0
    fallback = row.fallback_requests or 0

    return UsageSummary(
        customer_id=customer_id,
        total_requests=total,
        total_prompt_tokens=row.total_prompt_tokens or 0,
        total_completion_tokens=row.total_completion_tokens or 0,
        total_cost_usd=row.total_cost_usd or 0.0,
        fallback_requests=fallback,
        routed_requests=total - fallback,
        cost_savings_pct=_calc_savings(row) if total > 0 else 0.0,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "plane": "control"}


def _model_to_info(m: FineTunedModel) -> ModelInfo:
    return ModelInfo(
        id=m.id,
        name=m.name,
        provider=m.provider.value,
        base_model=m.base_model,
        domain=m.domain,
        status=m.status.value,
        eval_score=m.eval_score,
        cost_per_1k_input=m.cost_per_1k_input,
        cost_per_1k_output=m.cost_per_1k_output,
    )


def _calc_savings(row) -> float:
    """Estimate how much the customer saved by routing to SLMs vs. using fallback for everything."""
    if not row.total_cost_usd or not row.total_prompt_tokens:
        return 0.0
    actual_cost = row.total_cost_usd
    total_tokens = (row.total_prompt_tokens or 0) + (row.total_completion_tokens or 0)
    # What it would have cost at fallback rates (GPT-4o ~$5/M avg)
    hypothetical_cost = total_tokens * 5.0 / 1_000_000
    if hypothetical_cost == 0:
        return 0.0
    return max(0.0, (hypothetical_cost - actual_cost) / hypothetical_cost * 100)
