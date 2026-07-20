"""
Implementaciones concretas de BaseLLMClient para OpenAI y Anthropic.

Ambas clases:
- Usan siempre la versión asíncrona del SDK (AsyncOpenAI / AsyncAnthropic),
  nunca la síncrona, para no bloquear el event loop.
- Capturan los errores de red/autenticación/rate limit y los transforman
  en un ModelResponse(ok=False, error=...) en vez de dejarlos escapar.
"""

from collections.abc import AsyncGenerator

from anthropic import (
    AsyncAnthropic,
    APIConnectionError as AnthropicConnectionError,
    APIStatusError as AnthropicStatusError,
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    AsyncOpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIStatusError as OpenAIStatusError,
    RateLimitError as OpenAIRateLimitError,
)

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Role


class OpenAIClient(BaseLLMClient):
    """Adapta la SDK asíncrona de OpenAI a la interfaz común BaseLLMClient."""

    def __init__(self, api_key: str) -> None:
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=config.model,
                messages=self._to_openai_messages(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            texto: str = response.choices[0].message.content or ""
            return ModelResponse(provider=config.provider, model=config.model, content=texto)
        except OpenAIRateLimitError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Límite de tasa excedido (rate limit): {e}",
            )
        except OpenAIConnectionError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error de conexión con OpenAI: {e}",
            )
        except OpenAIStatusError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error de la API de OpenAI ({e.status_code}): {e}",
            )
        except Exception as e:  # noqa: BLE001 - último resguardo para no crashear el loop
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error inesperado en OpenAIClient: {e}",
            )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=config.model,
                messages=self._to_openai_messages(messages),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta: str | None = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (OpenAIRateLimitError, OpenAIConnectionError, OpenAIStatusError) as e:
            yield f"[ERROR OpenAI: {e}]"
        except Exception as e:  # noqa: BLE001
            yield f"[ERROR inesperado en OpenAIClient.stream: {e}]"


class AnthropicClient(BaseLLMClient):
    """Adapta la SDK asíncrona de Anthropic a la interfaz común BaseLLMClient."""

    def __init__(self, api_key: str) -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _split_system_and_messages(
        messages: list[ChatMessage],
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Anthropic recibe el "system prompt" como parámetro aparte, no como
        un mensaje más dentro de la lista. Acá separamos ambas cosas.
        """
        system_parts: list[str] = [m.content for m in messages if m.role == Role.SYSTEM]
        chat_messages: list[dict[str, str]] = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        return "\n".join(system_parts), chat_messages

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        system, chat_messages = self._split_system_and_messages(messages)
        try:
            response = await self._client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system,
                messages=chat_messages,
            )
            texto: str = response.content[0].text if response.content else ""
            return ModelResponse(provider=config.provider, model=config.model, content=texto)
        except AnthropicRateLimitError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Límite de tasa excedido (rate limit): {e}",
            )
        except AnthropicConnectionError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error de conexión con Anthropic: {e}",
            )
        except AnthropicStatusError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error de la API de Anthropic ({e.status_code}): {e}",
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                provider=config.provider, model=config.model, content="",
                ok=False, error=f"Error inesperado en AnthropicClient: {e}",
            )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        system, chat_messages = self._split_system_and_messages(messages)
        try:
            async with self._client.messages.stream(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system,
                messages=chat_messages,
            ) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicStatusError) as e:
            yield f"[ERROR Anthropic: {e}]"
        except Exception as e:  # noqa: BLE001
            yield f"[ERROR inesperado en AnthropicClient.stream: {e}]"
