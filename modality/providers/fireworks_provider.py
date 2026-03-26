import httpx

from modality.config import settings
from modality.providers.base import BaseProvider, CompletionResult, FineTuneResult

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_FINETUNE_BASE = "https://api.fireworks.ai/v1"


class FireworksProvider(BaseProvider):
    def __init__(self):
        self.api_key = settings.fireworks_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def upload_training_file(self, file_path: str) -> str:
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"{FIREWORKS_FINETUNE_BASE}/datasets",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": f},
                )
            response.raise_for_status()
            return response.json()["id"]

    async def start_finetune(
        self,
        training_file: str,
        base_model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        hyperparameters: dict | None = None,
    ) -> FineTuneResult:
        body = {
            "dataset": training_file,
            "model": base_model,
        }
        if hyperparameters:
            body.update(hyperparameters)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FIREWORKS_FINETUNE_BASE}/fine-tuning/jobs",
                headers=self.headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        return FineTuneResult(
            provider_job_id=data["id"],
            provider_model_id=data.get("fine_tuned_model"),
            status=data["status"],
        )

    async def check_finetune(self, job_id: str) -> FineTuneResult:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{FIREWORKS_FINETUNE_BASE}/fine-tuning/jobs/{job_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()

        return FineTuneResult(
            provider_job_id=data["id"],
            provider_model_id=data.get("fine_tuned_model"),
            status=data["status"],
            error=data.get("error"),
        )

    async def cancel_finetune(self, job_id: str) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FIREWORKS_FINETUNE_BASE}/fine-tuning/jobs/{job_id}/cancel",
                headers=self.headers,
            )
            response.raise_for_status()

    async def create_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FIREWORKS_BASE}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        return CompletionResult(
            content=choice["message"]["content"],
            usage=data["usage"],
            model=data["model"],
        )
