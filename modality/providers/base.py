from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FineTuneResult:
    provider_job_id: str
    provider_model_id: str | None  # populated once training completes
    status: str
    error: str | None = None


@dataclass
class CompletionResult:
    content: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int}
    model: str


class BaseProvider(ABC):
    @abstractmethod
    async def start_finetune(
        self,
        training_file: str,
        base_model: str,
        hyperparameters: dict | None = None,
    ) -> FineTuneResult:
        """Start a fine-tuning job. Returns the initial job status."""
        ...

    @abstractmethod
    async def check_finetune(self, job_id: str) -> FineTuneResult:
        """Check the status of a fine-tuning job."""
        ...

    @abstractmethod
    async def cancel_finetune(self, job_id: str) -> None:
        ...

    @abstractmethod
    async def upload_training_file(self, file_path: str) -> str:
        """Upload a JSONL training file. Returns the file ID."""
        ...

    @abstractmethod
    async def create_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Run inference against a model."""
        ...
