# Stubs / Templates reutilizables — AI Engineering Course

Templates genéricos y fuertemente tipados, extraídos de los patrones vistos
en `class_01` a `class_04`, pensados para copiar y pegar en **proyectos
nuevos** (no dependen de nada específico de este repo). Cada carpeta es
autocontenida: tiene su propio `README.md`, `requirements.txt` y
`.env.example` (cuando aplica).

## Índice

| Carpeta | Patrón | Basado en |
|---|---|---|
| [`01_llm_provider_factory/`](01_llm_provider_factory/) | Cliente LLM async intercambiable entre proveedores (Factory + Strategy + ABC) | `class_01/Entregable` |
| [`02_lcel_structured_output/`](02_lcel_structured_output/) | Pipeline LCEL con salida validada por Pydantic + reintentos | `class_02/Entregable` |
| [`03_local_rag_chromadb/`](03_local_rag_chromadb/) | RAG local end-to-end: chunking + ChromaDB + generación grounded | `class_03/Entregable` |
| [`04_cloud_rag_pinecone_hybrid/`](04_cloud_rag_pinecone_hybrid/) | RAG en la nube: Pinecone Serverless + recuperador híbrido (BM25 + vectorial) + evaluación | `class_04/Entregable` |

## Cómo usar estos templates

1. Copiá la carpeta del patrón que necesites a tu proyecto nuevo.
2. Instalá las dependencias: `pip install -r requirements.txt` (dentro de esa carpeta).
3. Copiá `.env.example` a `.env` y completá tus credenciales.
4. Buscá los comentarios `# TODO:` en el código — marcan las decisiones
   específicas de tu dominio (nombres de modelos, prompts, esquemas de
   datos, categorías, etc.) que tenés que adaptar. El resto (manejo de
   errores, reintentos, tipado, validación) ya está resuelto y probado.
5. Leé el `README.md` de cada carpeta: explica **qué problema resuelve el
   patrón**, cuándo usarlo y los pasos concretos de adaptación.

## Criterios de diseño aplicados en todos los stubs

- **Tipado estricto**: todas las funciones y clases usan type hints
  completos (incluyendo genéricos de `list[...]`, `dict[...]`, `| None`).
- **Pydantic para validación en los bordes**: cualquier dato que entra o
  sale del sistema (config, respuestas de LLM, resultados de evaluación)
  pasa por un modelo Pydantic, no por dicts sueltos.
- **Resiliencia explícita**: llamadas a APIs externas (LLMs, bases
  vectoriales) siempre están envueltas en manejo de errores que degrada
  con gracia (devuelve un resultado de error tipado) en vez de dejar
  propagar excepciones no controladas.
- **Configuración vía entorno**: nada de valores hardcodeados que deberían
  cambiar por proyecto (modelos, dimensiones, tamaños de chunk, pesos):
  todos son variables de entorno con defaults sensatos.
- **Idempotencia**: los pasos de "setup" (crear un índice, poblar una base
  vectorial) verifican el estado actual antes de actuar, para poder
  correrse muchas veces sin duplicar trabajo ni costo.
