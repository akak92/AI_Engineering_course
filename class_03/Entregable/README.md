# Pre-entrega 3: Sistema de Recuperación Semántica Local (RAG)

Sistema **End-to-End de RAG** (Retrieval-Augmented Generation) que recibe
una pregunta del usuario, busca los fragmentos más relevantes en una base
vectorial local (**ChromaDB**) poblada con documentos propios, y genera una
respuesta con un LLM (vía **LCEL**) que usa *exclusivamente* esa
información — respondiendo "no lo sé" si la respuesta no está presente en
el contexto recuperado.

El "cerebro" de este RAG son 4 documentos propios sobre **LangChain**,
**LangGraph** y **LangSmith** (`data/`).

## Estructura del proyecto

```
Entregable/
├── data/                 # "Cerebro" del RAG: 4 archivos .md de ejemplo
│   ├── 01_langchain.md
│   ├── 02_langgraph.md
│   ├── 03_langsmith.md
│   └── 04_integracion_stack.md
├── schemas.py             # Provider, RAGResponse (Pydantic)
├── ingest.py               # Módulo de Ingesta: chunking + persistencia en ChromaDB
├── retriever.py            # Capa de Recuperación: búsqueda por similitud en ChromaDB
├── rag_chain.py             # Generación Grounded: cadena LCEL + get_rag_response()
├── main.py                 # Script de prueba (pregunta respondible + pregunta trampa)
├── requirements.txt         # Dependencias con versiones fijadas
├── .env.example
└── README.md
```

## Cómo funciona (flujo end-to-end)

1. **Ingesta (`ingest.py`)**: lee los `.md`/`.txt` de `data/`, los limpia,
   los fragmenta con `RecursiveCharacterTextSplitter` contando tokens reales
   (`tiktoken`, chunks de 500 tokens con 50 de overlap) y los persiste en una
   colección de ChromaDB (`vectorstore/`, carpeta local). Si la colección ya
   tiene documentos, no vuelve a indexar (evita costo/tiempo innecesario).
2. **Recuperación (`retriever.py`)**: dada una pregunta, `SemanticRetriever`
   la embebe con el **mismo modelo de embeddings usado en la ingesta**
   (`OpenAIEmbeddingFunction`, `text-embedding-3-small`) y devuelve los
   `top_k` fragmentos más similares (por defecto 4, configurable).
3. **Generación grounded (`rag_chain.py`)**: los fragmentos recuperados se
   insertan como CONTEXTO en un prompt de "filtro de veracidad", y una
   cadena LCEL (`prompt | model.with_structured_output(RAGResponse)`)
   genera una respuesta validada con Pydantic, que incluye el texto de la
   respuesta y las referencias (archivos fuente) efectivamente usadas.

## Requisitos

- Python 3.12+
- Dependencias (ya instaladas en `env/` del repo, versiones fijadas en
  `requirements.txt`):

```powershell
pip install -r requirements.txt
```

## Configuración

1. Copiá `.env.example` a `.env`:

```powershell
Copy-Item .env.example .env
```

2. Completá `.env` con tu API key:

```
OPENAI_API_KEY=sk-...      # obligatoria: se usa para embeddings (ingesta y búsqueda) y, por defecto, para generación
ANTHROPIC_API_KEY=sk-ant-... # opcional: solo si LLM_PROVIDER=anthropic
LLM_PROVIDER=openai          # o "anthropic", para la etapa de generación
```

`OPENAI_API_KEY` es siempre necesaria (los embeddings usan OpenAI), sin
importar qué proveedor uses para generar la respuesta final. `.env` ya está
en `.gitignore`, así que tu key nunca se sube al repositorio.

## Ejecutar

### 1. Poblar la base vectorial (ingesta)

```powershell
cd class_03/Entregable
python ingest.py
```

Esto crea la carpeta local `vectorstore/` (también en `.gitignore`, no se
versiona) con los fragmentos de `data/` ya indexados. Si volvés a correrlo,
detecta que la colección ya tiene datos y no reindexa.

### 2. Correr las pruebas end-to-end

```powershell
python main.py
```

Ejecuta `DocumentIngestor().run()` (por si la base todavía no fue poblada) y
dos consultas contra `get_rag_response`:

- **Pregunta respondible**: "¿Qué es LCEL y cómo se componen los pasos de
  una cadena en LangChain?" → la respuesta está en `data/01_langchain.md`.
- **Pregunta trampa**: "¿Cuál es el precio mensual de la licencia
  empresarial de LangSmith?" → ese dato no existe en ningún documento; el
  modelo debe decir que no tiene esa información en vez de inventarla.

## Uso básico en tu propio código

```python
import asyncio
from rag_chain import get_rag_response

async def main():
    resultado = await get_rag_response("¿Qué es LangGraph?")
    print(resultado.respuesta)
    print(resultado.referencias)

asyncio.run(main())
```

## Errores comunes evitados en el diseño

- **Contexto infinito**: `top_k` limitado a 4 por defecto (rango 3-5
  sugerido), para no degradar la atención del modelo ni disparar el costo
  de tokens.
- **Embeddings no coincidentes**: `retriever.py` importa `COLLECTION_NAME`,
  `EMBEDDING_MODEL` y `PERSIST_DIR` directamente de `ingest.py`, así que la
  ingesta y la búsqueda usan siempre exactamente el mismo modelo de
  embeddings y la misma colección.
- **Falta de persistencia**: `DocumentIngestor.is_already_ingested()` evita
  reindexar si la colección ya tiene documentos, salvo que se pase
  `force=True`.
- **API keys expuestas**: `.env` está en `.gitignore`; solo se versiona
  `.env.example` sin valores reales.
