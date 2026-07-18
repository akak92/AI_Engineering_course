import asyncio
import os
from abc import ABC, abstractmethod
from enum import Enum

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel, SecretStr

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class LLMConfig(BaseModel):
    """
    Almacena las API keys de cada proveedor como SecretStr para que nunca
    queden expuestas en texto plano en logs, prints o representaciones del objeto.
    """
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    def require_key_for(self, provider: "Provider") -> SecretStr:
        """
        Devuelve la API key necesaria para el proveedor solicitado.

        Lanza un ValueError si la key correspondiente no fue configurada,
        evitando instanciar un cliente sin credenciales válidas.
        """
        key: SecretStr | None = (
            self.openai_api_key if provider == Provider.OPENAI else self.anthropic_api_key
        )
        if key is None or not key.get_secret_value():
            raise ValueError(f"Falta configurar la API key para el proveedor '{provider.value}'.")
        return key

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, prompt: str) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    """Adapta la SDK de OpenAI a la interfaz común BaseLLMClient."""

    def __init__(self, api_key: SecretStr, model: str = "gpt-4o-mini") -> None:
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=api_key.get_secret_value())
        self._model: str = model

    async def chat(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

class AnthropicClient(BaseLLMClient):
    """Adapta la SDK de Anthropic a la interfaz común BaseLLMClient."""

    def __init__(self, api_key: SecretStr, model: str = "claude-3-5-sonnet-20241022") -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key.get_secret_value())
        self._model: str = model

    async def chat(self, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

class LLMFactory:
    @staticmethod
    def create_client(provider: Provider, config: LLMConfig) -> BaseLLMClient:
        """Selecciona e instancia el cliente adecuado según el proveedor solicitado."""
        api_key: SecretStr = config.require_key_for(provider)
        if provider == Provider.OPENAI:
            return OpenAIClient(api_key)
        if provider == Provider.ANTHROPIC:
            return AnthropicClient(api_key)
        raise ValueError(f"Proveedor no soportado: {provider}")

async def main() -> None:
    config: LLMConfig = LLMConfig(
        openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
    )

    provider: Provider = Provider.OPENAI
    try:
        client: BaseLLMClient = LLMFactory.create_client(provider, config)
        respuesta: str = await client.chat("Hola, ¿quién eres?")
        print(f"[{provider.value}] {respuesta}")
    except ValueError as e:
        print(f"Error de configuración: {e}")

if __name__ == "__main__":
    asyncio.run(main())