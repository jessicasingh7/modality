import openai

from modality.config import settings
from modality.providers.base import BaseProvider, CompletionResult, FineTuneResult


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def upload_training_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            response = await self.client.files.create(file=f, purpose="fine-tune")
        return response.id

    async def start_finetune(
        self,
        training_file: str,
        base_model: str = "gpt-4o-mini-2024-07-18",
        hyperparameters: dict | None = None,
    ) -> FineTuneResult:
        params = {
            "training_file": training_file,
            "model": base_model,
        }
        if hyperparameters:
            params["hyperparameters"] = hyperparameters

        job = await self.client.fine_tuning.jobs.create(**params)
        return FineTuneResult(
            provider_job_id=job.id,
            provider_model_id=job.fine_tuned_model,
            status=job.status,
        )

    async def check_finetune(self, job_id: str) -> FineTuneResult:
        job = await self.client.fine_tuning.jobs.retrieve(job_id)
        return FineTuneResult(
            provider_job_id=job.id,
            provider_model_id=job.fine_tuned_model,
            status=job.status,
            error=str(job.error) if job.error else None,
        )

    async def cancel_finetune(self, job_id: str) -> None:
        await self.client.fine_tuning.jobs.cancel(job_id)

    async def create_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return CompletionResult(
            content=choice.message.content or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            model=response.model,
        )
