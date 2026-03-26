from pydantic import BaseModel


class RouteDecision(BaseModel):
    model_id: int | None
    provider: str
    provider_model_id: str | None
    confidence: float
    is_fallback: bool
    reason: str
