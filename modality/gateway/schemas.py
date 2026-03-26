from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    is_fallback: bool
    confidence: float
    routing_reason: str
    usage: dict


class FineTuneRequest(BaseModel):
    customer_name: str
    domain: str
    domain_description: str
    training_file_path: str
    base_model: str = "gpt-4o-mini-2024-07-18"
    provider: str = "openai"
    hyperparameters: dict | None = None


class FineTuneResponse(BaseModel):
    job_id: int
    provider_job_id: str
    status: str
    model_name: str


class ModelInfo(BaseModel):
    id: int
    name: str
    provider: str
    base_model: str
    domain: str
    status: str
    eval_score: float | None
    cost_per_1k_input: float | None
    cost_per_1k_output: float | None


class CustomerCreate(BaseModel):
    name: str
    domain: str


class CustomerInfo(BaseModel):
    id: int
    name: str
    domain: str


class ApiKeyCreate(BaseModel):
    name: str = "default"


class ApiKeyResponse(BaseModel):
    id: int
    key: str  # raw key — only shown once
    key_prefix: str
    name: str


class UsageSummary(BaseModel):
    customer_id: int
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    fallback_requests: int
    routed_requests: int
    cost_savings_pct: float
