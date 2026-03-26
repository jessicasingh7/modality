"""
Fine-tuning pipeline orchestration.

Flow:
1. Customer uploads training data (JSONL)
2. We validate and upload it to the provider
3. Kick off a fine-tuning job
4. Poll for completion
5. When done, register the model and trigger evaluation
6. If eval passes, promote the model into the routing table
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from modality.config import settings
from modality.eval.evaluator import evaluate_model
from modality.providers.embeddings import get_embedding
from modality.providers.registry import get_provider
from modality.registry.models import (
    Customer,
    FineTunedModel,
    FineTuneJob,
    JobStatus,
    ModelStatus,
    Provider,
)
from modality.registry.service import promote_model


async def start_finetune_pipeline(
    db: AsyncSession,
    customer: Customer,
    training_file_path: str,
    base_model: str,
    provider_name: str,
    domain_description: str,
    hyperparameters: dict | None = None,
) -> FineTuneJob:
    """Kick off the full fine-tuning pipeline for a customer."""

    provider_enum = Provider(provider_name)
    provider = get_provider(provider_name)

    # 1. Create the model record
    model = FineTunedModel(
        customer_id=customer.id,
        name=f"{customer.name}-{customer.domain}-{base_model}",
        provider=provider_enum,
        base_model=base_model,
        status=ModelStatus.training,
        domain=customer.domain,
        domain_description=domain_description,
    )
    db.add(model)
    await db.flush()

    # 2. Upload training file to the provider
    file_id = await provider.upload_training_file(training_file_path)

    # 3. Start the fine-tuning job
    result = await provider.start_finetune(
        training_file=file_id,
        base_model=base_model,
        hyperparameters=hyperparameters,
    )

    # 4. Record the job
    job = FineTuneJob(
        model_id=model.id,
        provider=provider_enum,
        provider_job_id=result.provider_job_id,
        status=JobStatus.running,
        base_model=base_model,
        training_file=file_id,
        hyperparameters=hyperparameters,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    return job


async def poll_and_finalize_job(db: AsyncSession, job: FineTuneJob) -> FineTuneJob:
    """Check a running job's status and finalize if complete."""

    provider = get_provider(job.provider.value)
    result = await provider.check_finetune(job.provider_job_id)

    if result.status in ("succeeded", "completed"):
        job.status = JobStatus.succeeded
        job.finished_at = datetime.utcnow()

        # Update the model with the provider's model ID
        model = job.model
        model.provider_model_id = result.provider_model_id
        model.status = ModelStatus.evaluating

        # Generate domain embedding for routing
        if model.domain_description:
            model.domain_embedding = await get_embedding(model.domain_description)

        await db.commit()

        # Run evaluation
        eval_score = await evaluate_model(db, model)
        model.eval_score = eval_score

        if eval_score >= settings.eval_min_score:
            await promote_model(db, model.id)
        else:
            model.status = ModelStatus.inactive
            await db.commit()

    elif result.status == "failed":
        job.status = JobStatus.failed
        job.error = result.error
        job.finished_at = datetime.utcnow()
        job.model.status = ModelStatus.failed
        await db.commit()

    # Still running — no-op, caller should poll again later
    return job
