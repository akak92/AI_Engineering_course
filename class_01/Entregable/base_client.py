"""
Clase base abstracta para los clientes de LLM asíncronos.

Cualquier proveedor (OpenAI, Anthropic, etc.) que quiera integrarse al
Unified Async LLM Client debe heredar de BaseLLMClient e implementar
generate() y stream(). Esta es la pieza clave de la "intercambiabilidad":
el resto de la aplicación programa contra esta interfaz, nunca contra un
SDK concreto.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz común que deben implementar todos los clientes de proveedores."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        """
        Envía los mensajes al modelo y espera la respuesta completa.

        Debe implementarse con `await` sobre el SDK asíncrono del proveedor
        (nunca la versión síncrona, para no bloquear el event loop) y nunca
        debe dejar escapar excepciones de red/API: en su lugar, tiene que
        devolver un ModelResponse con ok=False y el detalle en error.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        """
        Devuelve un generador asíncrono que produce fragmentos de texto
        (chunks/tokens) conforme llegan desde la API del proveedor.

        Las implementaciones concretas deben declararse como
        `async def stream(...)` y usar `yield` dentro de un `async for`
        que recorra el stream nativo del SDK.
        """
        raise NotImplementedError
