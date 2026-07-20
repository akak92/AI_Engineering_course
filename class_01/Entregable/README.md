# Unified Async LLM Client

Cliente asíncrono unificado para hablar con distintos proveedores de LLM
(**OpenAI** y **Anthropic**) bajo una misma interfaz, con soporte de
**streaming** y validación de datos con **Pydantic**.

## Estructura del proyecto

```
Entregable/
├── schemas.py       # Modelos Pydantic: ChatMessage, ModelConfig, ModelResponse
├── base_client.py   # Clase abstracta BaseLLMClient (generate/stream)
├── providers.py     # OpenAIClient y AnthropicClient (implementaciones concretas)
├── manager.py       # AsyncLLMManager: selecciona el proveedor y expone la API única
├── main.py          # Script de prueba (modo normal + streaming)
├── .env.example      # Plantilla de variables de entorno
└── README.md
```

## Requisitos

- Python 3.12+
- Un entorno virtual con las dependencias instaladas:

```powershell
pip install openai anthropic pydantic python-dotenv
```

## Configuración

1. Copia `.env.example` a `.env`:

```powershell
Copy-Item .env.example .env
```

2. Completa `.env` con tus propias API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

No hace falta configurar ambas: si solo tienes key de un proveedor, el
otro simplemente devolverá un error controlado (sin crashear) al usarlo.

`.env` ya está incluido en `.gitignore`, por lo que tus credenciales nunca
se suben al repositorio.

## Ejecutar el script de prueba

Desde la raíz del proyecto, con el entorno virtual activado:

```powershell
python class_01/Entregable/main.py
```

El script:

1. Carga las variables de entorno desde `.env`.
2. Envía la pregunta `"¿Qué es la entropía?"` a OpenAI y a Anthropic.
3. Para cada proveedor, primero pide la respuesta en **modo normal**
   (`AsyncLLMManager.generate`) y luego en **modo streaming**
   (`AsyncLLMManager.stream`), imprimiendo cada fragmento de texto
   conforme llega.

## Uso básico en tu propio código

```python
import asyncio
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Provider, Role

async def main():
    manager = AsyncLLMManager()  # toma las keys de las variables de entorno
    config = ModelConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=0.7)
    messages = [ChatMessage(role=Role.USER, content="Hola, ¿quién eres?")]

    # Modo normal
    respuesta = await manager.generate(messages, config)
    print(respuesta.content if respuesta.ok else respuesta.error)

    # Modo streaming
    async for fragmento in manager.stream(messages, config):
        print(fragmento, end="", flush=True)

asyncio.run(main())
```

## Diseño y decisiones clave

- **Intercambiabilidad**: todo el código de la aplicación programa contra
  `BaseLLMClient` y `AsyncLLMManager`, nunca contra `AsyncOpenAI` o
  `AsyncAnthropic` directamente. Cambiar de proveedor es cambiar
  `config.provider`.
- **Asincronía real**: todas las llamadas usan `await` sobre los SDKs
  asíncronos oficiales, por lo que nunca bloquean el event loop.
- **Streaming**: `stream()` es un generador asíncrono (`async def` con
  `yield`) que recorre el stream nativo de cada SDK y expone únicamente
  fragmentos de texto (`str`), sin exponer detalles internos del proveedor.
- **Validación con Pydantic**: `ModelConfig` valida rangos (`temperature`
  entre 0 y 2, `max_tokens` positivo) antes de que la solicitud llegue a
  la API. `ChatMessage` valida que el contenido no esté vacío.
- **Manejo de errores**: `RateLimitError`, `APIConnectionError` y
  `APIStatusError` de cada SDK se capturan explícitamente. En modo normal
  se devuelve un `ModelResponse(ok=False, error=...)`; en modo streaming
  se emite un chunk `"[ERROR: ...]"` y se corta el generador. En ningún
  caso una excepción de red o de autenticación debería crashear el
  programa.
