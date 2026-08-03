# Template: RAG local end-to-end (chunking + ChromaDB + generación grounded)

**Patrón**: Ingesta -> Recuperación -> Generación grounded (LCEL + Pydantic).

## ¿Cuándo usar este template?

Cuando necesitás que un LLM responda preguntas basándose **exclusivamente**
en un conjunto propio de documentos (manuales, apuntes, normativas), sin
depender de infraestructura en la nube — todo corre local con ChromaDB.
Para un caso similar pero con base vectorial en la nube y recuperador
híbrido, ver `../04_cloud_rag_pinecone_hybrid/`.

## Estructura

```
data/           # Tus documentos .md/.txt (reemplazá el archivo de ejemplo)
chunking.py      # Chunking compartido (por tokens, con tiktoken)
ingest.py         # Chunking -> embeddings -> persistencia en ChromaDB
retriever.py       # Búsqueda por similitud sobre la misma colección
schemas.py          # RAGResponse (Pydantic): respuesta + referencias
rag_chain.py         # Prompt "filtro de veracidad" + cadena LCEL grounded
```

## Cómo adaptarlo a tu proyecto

1. Reemplazá `data/00_ejemplo_reemplazar.md` por tus documentos reales.
2. Si tus documentos tienen categorías/tipos, completá `CATEGORIES` en
   `chunking.py`.
3. Ajustá `SYSTEM_PROMPT` en `rag_chain.py` al tono/dominio de tu caso de
   uso (manteniendo las dos reglas: responder solo con el CONTEXTO, y decir
   explícitamente que no sabe si la info no está).
4. Corré `python ingest.py` una vez para poblar `vectorstore/` (se crea
   local, ya está en `.gitignore` de este template — ver abajo).
5. Llamá a `get_rag_response("tu pregunta")` desde tu aplicación.

## Por qué está diseñado así

- **Embeddings acoplados a la collection (`OpenAIEmbeddingFunction`)**: Chroma
  guarda la embedding function como parte de la configuración de la
  collection y la reutiliza automáticamente tanto al indexar como al
  consultar — así es estructuralmente imposible tener el error más común de
  un RAG (embeddings distintos para indexar vs. buscar).
- **Chunking por tokens reales (`tiktoken`), no por caracteres**: los
  límites de contexto de los LLMs se miden en tokens, no en caracteres; un
  chunk de "500 caracteres" puede ser mucho más o menos que 500 tokens
  según el idioma/contenido.
- **`is_already_ingested()` + `force=False`**: correr `ingest.py` muchas
  veces (ej. en cada deploy) no reindexa ni gasta cuota de embeddings si
  los documentos no cambiaron.
- **`top_k` acotado (3-5)**: pasar demasiados fragmentos al LLM degrada su
  atención ("Lost in the Middle") y sube el costo sin mejorar la respuesta.
- **Salida validada con Pydantic (`RAGResponse`)**: el LLM no solo devuelve
  texto, sino también qué fuentes usó — permite mostrar citas verificables
  en vez de confiar ciegamente en la respuesta.

## Ejecutar el demo

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # completá tu OPENAI_API_KEY
python ingest.py               # puebla la base vectorial local
python rag_chain.py             # corre una consulta de ejemplo
```

## Nota sobre `.gitignore`

Si este template se copia a un repo nuevo, agregá `vectorstore/` a su
`.gitignore` (la base vectorial es un artefacto generado, no código fuente).
