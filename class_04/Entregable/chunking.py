"""
Chunking compartido del sistema RAG.

Fragmenta los documentos de `data/` en `Chunk`s, contando tokens reales con
`tiktoken` vía `RecursiveCharacterTextSplitter`. Este módulo lo usan tanto
`ingest.py` (para subir los fragmentos a Pinecone) como `rag_system.py`
(para construir el índice léxico `BM25Retriever`), de modo que **ambos
recuperadores operen sobre exactamente los mismos fragmentos** con los
mismos ids.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"

# ~500-800 tokens es el punto medio sugerido: chunks muy chicos pierden
# contexto semántico, muy grandes diluyen la precisión del embedding.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "650"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

# Etiqueta de categoría por archivo (metadata avanzada para Pinecone).
CATEGORIES = {
    "01_langchain.md": "langchain",
    "02_langgraph.md": "langgraph",
    "03_langsmith.md": "langsmith",
    "04_integracion_stack.md": "integracion",
}


@dataclass
class Chunk:
    """Un fragmento de documento ya limpio y listo para embeber/indexar."""

    id: str
    text: str
    source: str
    category: str
    chunk_index: int


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk_documents(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Lee los `.md`/`.txt` de `data_dir`, los limpia y los fragmenta por tokens."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=lambda t: len(tokenizer.encode(t)),
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    archivos = sorted([*data_dir.glob("*.txt"), *data_dir.glob("*.md")])
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .txt/.md en {data_dir}")

    chunks: list[Chunk] = []
    for f in archivos:
        cleaned = _clean_text(f.read_text(encoding="utf-8"))
        category = CATEGORIES.get(f.name, "general")
        for i, text in enumerate(splitter.split_text(cleaned)):
            chunks.append(
                Chunk(
                    id=f"{f.name}::chunk_{i}",
                    text=text,
                    source=f.name,
                    category=category,
                    chunk_index=i,
                )
            )
    return chunks
