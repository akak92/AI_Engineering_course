# Template: RAG en la nube con recuperador híbrido (Pinecone + BM25) y evaluación

**Patrón**: Infraestructura Serverless -> Ingesta con metadata avanzada ->
Recuperador híbrido (léxico + semántico, RRF) -> Evaluación (Precision@k / Recall@k).

## ¿Cuándo usar este template?

Cuando tu RAG necesita **escalar en la nube** (en vez de una base vectorial
local como Chroma) y tus consultas mezclan preguntas parafraseadas con
términos técnicos exactos / nombres propios, donde un recuperador
puramente semántico pierde precisión. Para un caso más simple, 100% local
y sin recuperador híbrido, ver `../03_local_rag_chromadb/`.

## Estructura

```
data/                  # Tus documentos .md/.txt (reemplazá el archivo de ejemplo)
chunking.py             # Chunking compartido (Pinecone y BM25 usan los mismos chunks)
pinecone_setup.py         # Crea el índice Serverless si no existe (idempotente)
ingest.py                  # Embeddings + upsert en batches con metadata avanzada
hybrid_retriever.py          # RAGSystem: EnsembleRetriever (BM25 + Pinecone)
golden_set.json                # Preguntas + documento fuente esperado
evaluate.py                     # Precision@k / Recall@k sobre el golden set
```

## Cómo adaptarlo a tu proyecto

1. Reemplazá `data/00_ejemplo_reemplazar.md` por tus documentos reales.
2. Completá `golden_set.json` con al menos 5 preguntas reales, cubriendo
   distintos documentos fuente (parafraseá — no copies literalmente
   palabras del documento, para probar recuperación real).
3. Corré `pinecone_setup.py` -> `ingest.py` -> `evaluate.py` en ese orden
   (ver README de comandos abajo).
4. Ajustá `BM25_WEIGHT`/`VECTOR_WEIGHT` en `.env` según qué tan léxico o
   semántico es tu dominio de preguntas.

## Por qué está diseñado así

- **Namespace dedicado**: separa tus datos de otros proyectos/entornos
  dentro del mismo índice de Pinecone; evita búsquedas ruidosas al mezclar
  dominios.
- **Metadata con el texto original**: cada vector guarda su propio texto en
  la metadata, evitando una consulta adicional a una base relacional para
  reconstruir el contenido de un resultado.
- **Chunking compartido (`chunking.py`)**: tanto el índice vectorial
  (Pinecone) como el índice léxico (BM25, reconstruido en memoria) se
  arman a partir de la misma función de chunking — así ambos recuperadores
  "ven" exactamente el mismo corpus con los mismos ids, y el
  `EnsembleRetriever` puede deduplicar correctamente.
- **`EnsembleRetriever` (Reciprocal Rank Fusion)**: combina los rankings de
  BM25 y Pinecone ponderados, en vez de simplemente concatenar resultados
  — un documento que aparece bien rankeado en ambos recuperadores sube más
  que uno que solo aparece en uno.
- **Idempotencia (`has_index`, `is_already_ingested`)**: correr los scripts
  de setup/ingesta muchas veces no duplica infraestructura ni gasta cuota
  de embeddings de más.
- **Evaluación con golden set**: Precision@k/Recall@k dan una métrica
  objetiva para comparar cambios (ej. distintos pesos de ensemble, distinto
  tamaño de chunk) en vez de juzgar "a ojo" si el RAG mejoró.

## Comandos

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env        # completá PINECONE_API_KEY y OPENAI_API_KEY

python pinecone_setup.py             # 1. crea el índice Serverless (idempotente)
python ingest.py                      # 2. puebla el namespace con tus documentos
python hybrid_retriever.py              # 3. prueba una consulta de ejemplo
python evaluate.py                       # 4. corre el golden set (Precision@k/Recall@k)
```

## Notas de compatibilidad de versiones

Este template fue verificado contra `pinecone==7.3.0` (la versión que
resuelve `pip` al instalar `langchain-pinecone==0.2.13`, que es distinta a
instalar `pinecone` solo). Si actualizás alguna de estas dependencias:

- Confirmá con `pip show pinecone` qué versión quedó activa.
- `pc.Index(name=...)` (con "I" mayúscula) es el método correcto en esta
  versión — no existe `pc.index()` en minúscula ni una clase `Index`
  pública para type hints en el paquete top-level.
- `EnsembleRetriever` vive en `langchain_classic.retrievers.ensemble` en
  esta combinación de versiones, no en `langchain_community` (donde vivía
  en versiones más viejas del ecosistema LangChain).
