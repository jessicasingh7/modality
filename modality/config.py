from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./modality.db"

    # Provider API keys
    openai_api_key: str = ""
    fireworks_api_key: str = ""
    together_api_key: str = ""

    # Fallback model (used when no fine-tuned model matches or confidence is low)
    fallback_model: str = "gpt-4o"
    fallback_provider: str = "openai"

    # Router settings
    router_confidence_threshold: float = 0.7
    router_embedding_model: str = "text-embedding-3-small"

    # Eval settings
    eval_min_score: float = 0.8  # minimum eval score to promote a model

    model_config = {"env_prefix": "MODALITY_", "env_file": ".env"}


settings = Settings()
