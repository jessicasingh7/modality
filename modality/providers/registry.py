from modality.providers.base import BaseProvider
from modality.providers.fireworks_provider import FireworksProvider
from modality.providers.openai_provider import OpenAIProvider

_providers: dict[str, BaseProvider] = {}


def get_provider(name: str) -> BaseProvider:
    if name not in _providers:
        match name:
            case "openai":
                _providers[name] = OpenAIProvider()
            case "fireworks":
                _providers[name] = FireworksProvider()
            case _:
                raise ValueError(f"Unknown provider: {name}")
    return _providers[name]
