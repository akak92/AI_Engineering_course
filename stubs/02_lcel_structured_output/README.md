# Template: Pipeline LCEL con salida estructurada (Pydantic) + reintentos

**Patrón**: `prompt | model.with_structured_output(schema).with_retry()`.

## ¿Cuándo usar este template?

Cuando necesitás que un LLM devuelva **datos validados** (no texto libre)
para que el resto de tu aplicación los consuma con garantías de tipo: un
formulario extraído de un texto, una clasificación, un resumen con campos
fijos, etc. Si en cambio solo necesitás texto plano, usá `StrOutputParser`
en lugar de `with_structured_output`.

## Estructura

```
schemas.py   # OutputSchema (Pydantic) — el "contrato" de la respuesta
chain.py      # get_model(), prompt, chain LCEL, process_input()
```

## Cómo adaptarlo a tu proyecto

1. **`schemas.py`**: reemplazá `OutputSchema` por el modelo real de tu
   dominio. Cada campo obligatorio sin default fuerza al modelo a
   completarlo; usá `Enum` para valores cerrados y `Field(description=...)`
   para guiar al modelo (esa descripción se envía como parte del schema).
2. **`chain.py`**: reescribí `SYSTEM_PROMPT` con las instrucciones reales de
   tu tarea, y renombrá la variable de plantilla `{texto}` si tu dominio
   usa otro nombre más claro (ej. `{transcripcion}`, `{codigo}`).
3. Llamá a `process_input(...)` (o renombralo, ej. `extract(...)`,
   `classify(...)`) desde el resto de tu app; siempre devuelve
   `OutputSchema | None` — nunca deja propagar una excepción del LLM.

## Por qué está diseñado así

- **`with_structured_output` en vez de parsear JSON a mano**: delega en el
  proveedor (tool calling / JSON mode) la garantía de formato, y en
  Pydantic la validación de contenido (rangos, enums, longitud mínima).
- **`.with_retry(stop_after_attempt=3)`**: un LLM puede devolver
  ocasionalmente un JSON incompleto (por ejemplo si `finish_reason` fue
  `length`) o la llamada puede fallar por un error transitorio de red.
  Reintentar automáticamente evita escribir ese boilerplate a mano en cada
  pipeline.
- **`process_input` devuelve `None` en error, no levanta excepción**: fuerza
  a quien lo llama a decidir explícitamente qué hacer si la extracción
  falló, en vez de que un error de LLM tire abajo todo el flujo.
- **Proveedor intercambiable (`LLM_PROVIDER`)**: mismo patrón que en
  `01_llm_provider_factory/` — permite cambiar de modelo sin tocar el
  prompt ni la lógica de negocio.

## Ejecutar el demo

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # completá tu OPENAI_API_KEY
python chain.py
```
