"""
Script de prueba del Unified Async LLM Client.

Carga las variables de entorno desde `.env`, arma una pregunta corta y la
envía a cada proveedor configurado tanto en modo normal (respuesta completa)
como en modo streaming (fragmento por fragmento).

Uso:
    python main.py
"""

import asyncio

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider, Role

PREGUNTA = "¿Qué es la entropía?"

# Un modelo de referencia por proveedor. Ajusta estos nombres si tu cuenta
# tiene acceso a otros modelos.
MODELOS_POR_PROVEEDOR: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
}


async def probar_modo_normal(manager: AsyncLLMManager, provider: Provider) -> None:
    """Envía la pregunta en modo normal (respuesta completa, sin streaming)."""
    config = ModelConfig(provider=provider, model=MODELOS_POR_PROVEEDOR[provider])
    messages = [ChatMessage(role=Role.USER, content=PREGUNTA)]

    respuesta = await manager.generate(messages, config)
    print(f"\n--- {provider.value} | modo normal ---")
    if respuesta.ok:
        print(respuesta.content)
    else:
        print(f"[No se pudo obtener respuesta] {respuesta.error}")


async def probar_modo_streaming(manager: AsyncLLMManager, provider: Provider) -> None:
    """Envía la pregunta en modo streaming, imprimiendo cada fragmento conforme llega."""
    config = ModelConfig(provider=provider, model=MODELOS_POR_PROVEEDOR[provider])
    messages = [ChatMessage(role=Role.USER, content=PREGUNTA)]

    print(f"\n--- {provider.value} | modo streaming ---")
    async for fragmento in manager.stream(messages, config):
        print(fragmento, end="", flush=True)
    print()


async def main() -> None:
    load_dotenv()
    manager = AsyncLLMManager()

    for provider in (Provider.OPENAI, Provider.ANTHROPIC):
        await probar_modo_normal(manager, provider)
        await probar_modo_streaming(manager, provider)


if __name__ == "__main__":
    asyncio.run(main())
