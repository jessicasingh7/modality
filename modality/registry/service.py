from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from modality.registry.models import Customer, FineTunedModel, ModelStatus


async def get_active_models(db: AsyncSession) -> list[FineTunedModel]:
    """Get all models that are active and available for routing."""
    result = await db.execute(
        select(FineTunedModel)
        .where(FineTunedModel.status == ModelStatus.active)
        .options(selectinload(FineTunedModel.customer))
    )
    return list(result.scalars().all())


async def get_customer_models(db: AsyncSession, customer_id: int) -> list[FineTunedModel]:
    result = await db.execute(
        select(FineTunedModel).where(FineTunedModel.customer_id == customer_id)
    )
    return list(result.scalars().all())


async def get_or_create_customer(db: AsyncSession, name: str, domain: str) -> Customer:
    result = await db.execute(select(Customer).where(Customer.name == name))
    customer = result.scalar_one_or_none()
    if customer:
        return customer
    customer = Customer(name=name, domain=domain)
    db.add(customer)
    await db.commit()
    await db.refresh(customer)
    return customer


async def promote_model(db: AsyncSession, model_id: int) -> FineTunedModel:
    """Promote a model to active status, making it available for routing."""
    result = await db.execute(select(FineTunedModel).where(FineTunedModel.id == model_id))
    model = result.scalar_one()
    model.status = ModelStatus.active
    await db.commit()
    await db.refresh(model)
    return model


async def demote_model(db: AsyncSession, model_id: int) -> FineTunedModel:
    result = await db.execute(select(FineTunedModel).where(FineTunedModel.id == model_id))
    model = result.scalar_one()
    model.status = ModelStatus.inactive
    await db.commit()
    await db.refresh(model)
    return model
