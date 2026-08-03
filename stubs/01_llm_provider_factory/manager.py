"""
`AsyncLLMManager`: punto de entrada único para hablar con cualquier
proveedor de LLM soportado, sin que el resto de la aplicación conozca los
detalles de cada SDK (Factory + inicialización perezosa por proveedor).
"""

import os
from collections.abc import AsyncGenerator

from base_client import BaseLLMClient
from providers import AnthropicClient, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role


class AsyncLLMManager:
    """
    Selecciona e instancia el cliente adecuado (OpenAI o Anthropic) según el
    proveedor indicado en el `ModelConfig`, y expone una única interfaz
    (`generate`/`stream`) para el resto de la aplicación.
    """

    def __init__(
        self, openai_api_key: str | None = None, anthropic_api_key: str | None = None
    ) -> None:
        self._openai_api_key: str | None = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._anthropic_api_key: str | None = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self._clients: dict[Provider, BaseLLMClient] = {}

    def _get_client(self, provider: Provider) -> BaseLLMClient:
        """
        Crea (con inicialización perezosa, una única vez por proveedor) y
        devuelve el cliente solicitado, validando antes que exista la API
        key correspondiente.

        # TODO: al agregar un proveedor nuevo en `providers.py`/`schemas.py`,
        # sumá acá un `elif provider == Provider.<NUEVO>: ...`.
        """
        if provider in self._clients:
            return self._clients[provider]

        client: BaseLLMClient
        if provider == Provider.OPENAI:
            if not self._openai_api_key:
                raise ValueError("Falta configurar OPENAI_API_KEY para usar el proveedor 'openai'.")
            client = OpenAIClient(self._openai_api_key)
        elif provider == Provider.ANTHROPIC:
            if not self._anthropic_api_key:
                raise ValueError(
                    "Falta configurar ANTHROPIC_API_KEY para usar el proveedor 'anthropic'."
                )
            client = AnthropicClient(self._anthropic_api_key)
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")

        self._clients[provider] = client
        return client

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        """Genera una respuesta completa usando el proveedor indicado en `config`."""
        try:
            client: BaseLLMClient = self._get_client(config.provider)
        except ValueError as e:
            return ModelResponse(
                provider=config.provider, model=config.model, content="", ok=False, error=str(e)
            )
        return await client.generate(messages, config)

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """Genera la respuesta en modo streaming usando el proveedor indicado en `config`."""
        try:
            client: BaseLLMClient = self._get_client(config.provider)
        except ValueError as e:
            yield f"[ERROR: {e}]"
            return
        async for chunk in client.stream(messages, config):
            yield chunk


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    async def _demo() -> None:
        load_dotenv()
        manager = AsyncLLMManager()
        config = ModelConfig(provider=Provider.OPENAI, model="gpt-4o-mini")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="Sos un asistente conciso."),
            # TODO: reemplazá este mensaje de ejemplo por tu propio caso de uso.
            ChatMessage(role=Role.USER, content="¿Qué es un event loop en asyncio?"),
        ]
        response = await manager.generate(messages, config)
        print(response.model_dump_json(indent=2))

    asyncio.run(_demo())
