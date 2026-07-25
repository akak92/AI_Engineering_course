# Pipeline de Extracción de Entidades Técnicas

Pipeline que recibe un texto crudo (por ejemplo, un log de error o una
descripción de arquitectura) y devuelve un objeto **validado con Pydantic**
con las tecnologías mencionadas, el nivel de criticidad y un resumen
técnico, usando **LCEL** (`prompt | model.with_structured_output(schema)`)
con reintentos automáticos ante fallos de validación.

## Estructura del proyecto

```
Entregable/
├── schemas.py       # Provider, NivelCriticidad, TechExtraction (Pydantic)
├── chain.py         # Modelo, prompt, cadena LCEL y process_text()
├── main.py          # Mini-script de prueba
├── .env.example      # Plantilla de variables de entorno
└── README.md
```

## Requisitos

- Python 3.12+
- Dependencias:

```powershell
pip install langchain-core langchain-openai langchain-anthropic pydantic python-dotenv
```

## Configuración

1. Copia `.env.example` a `.env`:

```powershell
Copy-Item .env.example .env
```

2. Completa `.env` con tu API key y el proveedor a usar:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai   # o "anthropic"
```

Solo necesitas configurar la API key del proveedor que vayas a usar según
`LLM_PROVIDER`. `.env` ya está en `.gitignore`, así que tus credenciales
nunca se suben al repositorio.

## Ejecutar el script de prueba

```powershell
python class_02/Entregable/main.py
```

El script carga el `.env`, activa el logging para observar el proceso de
validación/reintentos, y ejecuta `process_text(...)` sobre un texto de
ejemplo (una descripción de arquitectura con FastAPI, Redis y PostgreSQL).

## Ejemplo de salida esperada

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Uso básico en tu propio código

```python
import asyncio
from chain import process_text

async def main():
    resultado = await process_text("Tu texto técnico acá...")
    if resultado:
        print(resultado.model_dump_json(indent=2))

asyncio.run(main())
```

## Diseño y decisiones clave

- **Esquema como contrato (`schemas.py`)**: `TechExtraction` define exactamente
  qué forma debe tener la respuesta (`tecnologias: list[str]` no vacía,
  `nivel_de_criticidad` restringido al enum `NivelCriticidad`, `resumen_tecnico`
  no vacío). Si el LLM devuelve algo que no cumple el contrato, Pydantic lo
  rechaza antes de que llegue al resto de la aplicación.
- **Intercambiabilidad de proveedor**: `get_model(provider)` en `chain.py`
  aplica el mismo patrón de la Pre-entrega 1 (`LLMFactory`/`AsyncLLMManager`):
  el resto del pipeline programa contra `BaseChatModel`, y el proveedor real
  (`ChatOpenAI` o `ChatAnthropic`) se elige según la variable de entorno
  `LLM_PROVIDER`.
- **Prompt modular**: `ChatPromptTemplate.from_messages` con roles
  `system`/`human`, sin f-strings hardcodeadas — la variable `{texto}` la
  gestiona LangChain al momento del `ainvoke`.
- **Salida estructurada en vez de `StrOutputParser`**: como el objetivo es un
  objeto validado (no texto plano), la cadena usa
  `model.with_structured_output(TechExtraction)` en lugar de un parser de
  texto: el modelo responde directamente con datos que calzan en el schema.
- **Resiliencia con `.with_retry()`**: si el LLM devuelve un JSON mal formado
  o incompleto (por ejemplo, si el `finish_reason` fue `length` y el JSON
  quedó cortado), la validación de Pydantic falla y `.with_retry()` reintenta
  automáticamente (hasta 3 intentos) antes de darse por vencido.
- **Manejo de errores en `process_text`**: si la extracción falla incluso
  después de los reintentos, se captura la excepción, se registra en el log
  como error y se devuelve `None` en vez de crashear el programa.
- **Logs de validación**: `process_text` loguea antes de la llamada (tamaño
  del texto, proveedor usado) y después (tecnologías y criticidad
  detectadas), para poder observar el comportamiento del pipeline y detectar
  reintentos en la consola.
