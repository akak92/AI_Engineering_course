"""
TEMPLATE: chunking compartido para un sistema RAG en la nube (Pinecone).

Igual que en `03_local_rag_chromadb/chunking.py`, pero pensado para que lo
usen dos consumidores distintos: `ingest.py` (sube los chunks a Pinecone) y
`hybrid_retriever.py` (reconstruye el índice léxico BM25 en memoria) — así
ambos recuperadores operan sobre exactamente el mismo corpus y los mismos
ids.

# TODO: reemplazar `CATEGORIES` por la taxonomía real de tu dominio (o
# eliminar el concepto de categoría si no lo necesitás).
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"

# ~500-800 tokens: punto medio entre perder contexto semántico (chunks muy
# chicos) y diluir la precisión del embedding (chunks muy grandes).
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "650"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "80"))

CATEGORIES: dict[str, str] = {
    # "manual_instalacion.md": "instalacion",
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
        raise FileNotFoundError(
            f"No se encontraron archivos .txt/.md en {data_dir}. "
            f"# TODO: agregá tus documentos fuente ahí."
        )

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
