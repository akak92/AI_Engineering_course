# Pre-entrega 4: Sistema RAG Escalable en la Nube con Pinecone

**Módulo de Recuperación Escalable**: pipeline de ingesta a **Pinecone
Serverless**, un **recuperador híbrido** (BM25 léxico + vectorial semántico,
combinados con `EnsembleRetriever`) y un script de **evaluación** (Precision@5
/ Recall@5) sobre un golden set de preguntas.

El "cerebro" de este RAG son los mismos 4 documentos de `class_03` sobre
**LangChain**, **LangGraph** y **LangSmith** (`data/`), ahora indexados en la
nube con metadata avanzada (fuente, categoría, texto original).

## Estructura del proyecto

```
Entregable/
├── data/                    # 4 archivos .md de ejemplo (LangChain/LangGraph/LangSmith)
│   ├── 01_langchain.md
│   ├── 02_langgraph.md
│   ├── 03_langsmith.md
│   └── 04_integracion_stack.md
├── chunking.py               # Chunking compartido (Pinecone + BM25 usan los mismos chunks)
├── pinecone_setup.py          # Paso 1: crea el índice Serverless si no existe (idempotente)
├── ingest.py                   # Paso 2: embeddings + upsert en batches con metadata avanzada
├── rag_system.py                 # Paso 3: RAGSystem con EnsembleRetriever (BM25 + Pinecone)
├── golden_set.json                # Paso 4: 5 preguntas con documento fuente esperado
├── evaluate.py                     # Paso 4: calcula Precision@5 y Recall@5
├── requirements.txt
├── .env.example
└── README.md
```

## Cómo funciona (flujo end-to-end)

1. **Infraestructura (`pinecone_setup.py`)**: verifica si el índice
   `INDEX_NAME` existe; si no, lo crea en modo **Serverless** (`aws`,
   `us-east-1`) con dimensión **1536** (la de `text-embedding-3-small`) y
   métrica `cosine`. Es idempotente: correrlo de nuevo no recrea el índice.
2. **Ingesta (`ingest.py`)**: fragmenta `data/` con `chunking.py`
   (~650 tokens por chunk, 80 de overlap — dentro del rango 500-800
   recomendado), genera embeddings con OpenAI y sube los vectores a
   Pinecone **en batches**, dentro de un **namespace** dedicado
   (`PINECONE_NAMESPACE`), guardando en la metadata la fuente, la categoría,
   el índice de chunk y el **texto original** (para no depender de una base
   relacional adicional al recuperar).
3. **Recuperador híbrido (`rag_system.py`)**: `RAGSystem` combina:
   - Un `BM25Retriever` (léxico), reconstruido en memoria con **los mismos
     chunks** que se subieron a Pinecone.
   - Un retriever vectorial sobre `PineconeVectorStore` (semántico).
   - Ambos se fusionan con `EnsembleRetriever` (Reciprocal Rank Fusion),
     devolviendo el Top-5 combinado.
4. **Evaluación (`evaluate.py`)**: corre las 5 preguntas de `golden_set.json`
   contra `RAGSystem` y calcula Precision@5 y Recall@5 comparando la fuente
   de los documentos recuperados contra el `documento_id_esperado`.

## Requisitos

- Python 3.12+
- Cuenta de [Pinecone](https://www.pinecone.io/) (el plan gratuito Serverless alcanza)
- Dependencias (versiones fijadas en `requirements.txt`):

```powershell
pip install -r requirements.txt
```

## Configuración

1. Copiá `.env.example` a `.env`:

```powershell
Copy-Item .env.example .env
```

2. Completá `.env` con tus credenciales:

```
PINECONE_API_KEY=pcsk_...
OPENAI_API_KEY=sk-...
INDEX_NAME=rag-langchain-docs
```

`.env` ya está en `.gitignore` (a nivel de repo), así que tus keys nunca se
suben al repositorio.

## Cómo replicar el índice de Pinecone

### 1. Crear el índice Serverless

```powershell
cd class_04/Entregable
python pinecone_setup.py
```

Esto crea (si no existe) un índice llamado `INDEX_NAME` en modo Serverless
(`cloud="aws"`, `region="us-east-1"`, dimensión `1536`, métrica `cosine`) y
espera a que quede listo. Si el índice ya existe, el script no hace nada
(idempotente).

### 2. Poblar el índice (ingesta)

```powershell
python ingest.py
```

Fragmenta `data/`, genera embeddings y sube los vectores al namespace
`PINECONE_NAMESPACE` dentro del índice. Si el namespace ya tiene vectores
cargados, no vuelve a indexar (a menos que se llame con `force=True`).

### 3. Probar el recuperador híbrido

```powershell
python rag_system.py
```

Corre una consulta de ejemplo ("¿Qué es LangGraph?") y muestra los
documentos recuperados con su fuente.

### 4. Evaluar (Precision@5 / Recall@5)

```powershell
python evaluate.py
```

## Ejemplo de reporte de evaluación

```
=== Reporte de Evaluación (Precision@5 / Recall@5) ===

[OK] ¿Qué operador de LCEL se usa para encadenar declarativamente...?
    Esperado:   01_langchain.md
    Recuperado: ['01_langchain.md', '01_langchain.md', '04_integracion_stack.md', ...]
    Precision@5: 0.60

...

--- Resumen ---
Casos evaluados: 5
Recall@5:    100.00%  (el documento esperado aparece en el Top-5)
Precision@5: 68.00%  (fragmentos del Top-5 que pertenecen al documento esperado)
```

(Los valores exactos van a variar según la ejecución real contra tu índice.)

## Uso básico en tu propio código

```python
from rag_system import RAGSystem

system = RAGSystem()
docs = system.retrieve("¿Cómo se traza la ejecución de una cadena LLM?")
for doc in docs:
    print(doc.metadata["source"], "->", doc.page_content[:100])
```

## Errores comunes evitados en el diseño

- **Mismatch de dimensiones**: `EMBEDDING_DIMENSION=1536` en `.env`, igual a
  la salida de `text-embedding-3-small`; `pinecone_setup.py` crea el índice
  con esa dimensión explícitamente.
- **Ignorar el namespace**: toda la ingesta y la búsqueda usan
  `PINECONE_NAMESPACE`, evitando mezclar datos de distintos dominios en una
  misma consulta.
- **Chunking mal calibrado**: 650 tokens con 80 de overlap (rango 500-800
  sugerido), configurable vía `CHUNK_SIZE_TOKENS`/`CHUNK_OVERLAP_TOKENS`.
- **Falta de persistencia**: `pinecone_setup.py` e `ingest.py` son
  idempotentes (`has_index`, `is_already_ingested`) — no recrean el índice
  ni reindexan si ya existen datos.
- **API keys expuestas**: `.env` está en `.gitignore`; solo se versiona
  `.env.example` sin valores reales.
