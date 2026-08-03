# Template: Cliente LLM async intercambiable entre proveedores

**Patrón**: Factory + Strategy + Abstract Base Class.

## ¿Cuándo usar este template?

Cuando tu aplicación necesita hablar con **más de un proveedor de LLM**
(OpenAI, Anthropic, y potencialmente otros) y no querés que el resto del
código dependa del SDK concreto de ninguno de ellos — por ejemplo, para
poder cambiar de proveedor con una variable de entorno, hacer fallback si
uno está caído, o comparar respuestas entre proveedores.

## Estructura

```
schemas.py       # Role, Provider, ChatMessage, ModelConfig, ModelResponse (Pydantic)
base_client.py    # BaseLLMClient (ABC): contrato generate()/stream()
providers.py       # OpenAIClient, AnthropicClient (implementaciones concretas)
manager.py          # AsyncLLMManager: factory con inicialización perezosa
```

## Cómo adaptarlo a tu proyecto

1. **`schemas.py`**: agregá campos a `ModelConfig`/`ModelResponse` si tu
   caso de uso los necesita (ej. `top_p`, `stop_sequences`, `usage_tokens`).
2. **`providers.py`**: si necesitás un proveedor nuevo, copiá una de las
   clases existentes como base y adaptá `generate()`/`stream()` al SDK
   correspondiente. Mantené el mismo patrón: SDK asíncrono + captura de
   errores por tipo (rate limit, conexión, status) + fallback genérico.
3. **`manager.py`**: registrá el proveedor nuevo en `_get_client()`.
4. Llamá siempre a través de `AsyncLLMManager`, nunca instancies
   `OpenAIClient`/`AnthropicClient` directamente desde tu lógica de negocio
   — así mantenés la intercambiabilidad.

## Por qué está diseñado así

- **`BaseLLMClient` (ABC)**: el resto de tu aplicación programa contra esta
  interfaz. Si mañana cambiás de OpenAI a Anthropic (o agregás un tercer
  proveedor), el código que llama a `manager.generate(...)` no cambia.
- **`ModelResponse.ok`/`error` en vez de excepciones**: una falla de red o
  de rate limit de un proveedor externo es algo *esperable*, no
  excepcional. Modelarlo como parte del tipo de retorno obliga a quien
  llama a manejarlo explícitamente, en vez de que un `try/except` genérico
  en otra capa lo esconda.
- **Inicialización perezosa por proveedor (`_clients` cache)**: no se crea
  un cliente HTTP para Anthropic si tu app nunca lo usa.
- **SDKs siempre asíncronos**: usar la versión síncrona de un SDK dentro de
  una app `async` bloquea el event loop completo mientras dura la llamada
  de red — con muchas requests concurrentes esto degrada el throughput
  drásticamente.

## Ejecutar el demo

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # completá tu OPENAI_API_KEY
python manager.py
```
