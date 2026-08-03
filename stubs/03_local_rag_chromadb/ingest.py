"""
TEMPLATE: módulo de ingesta a ChromaDB (local, persistente).

Fragmenta `data/` (vía `chunking.py`), y persiste los fragmentos en una
colección de ChromaDB usando `OpenAIEmbeddingFunction`. Usar la embedding
function del propio ChromaDB (en vez de generar los vectores "a mano")
garantiza que se use el mismo modelo tanto para indexar como para
consultar — evita el error más común de un sistema RAG: embeddings no
coincidentes entre ingesta y búsqueda.

Uso:
    python ingest.py
"""

import os

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from chunking import Chunk, load_and_chunk_documents

load_dotenv()

DATA_DIR_NAME = "data"
PERSIST_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class DocumentIngestor:
    """Pipeline de ingesta: lee archivos -> limpia -> chunkea -> persiste en ChromaDB."""

    def __init__(
        self,
        persist_path: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_path = persist_path
        # Al quedar guardada como parte de la configuración de la
        # collection, Chroma reutiliza esta misma embedding function tanto
        # al indexar (`upsert`) como al consultar (`query_texts`).
        self.embedding_fn = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def is_already_ingested(self) -> bool:
        """Permite evitar re-indexar si la colección ya tiene documentos."""
        return self.collection.count() > 0

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict[str, str | int]:
        return {
            "source": chunk.source,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
        }

    def run(self, force: bool = False) -> int:
        """
        Ejecuta el pipeline completo: lectura -> limpieza -> chunking -> upsert.

        Si `force=False` (default) y la colección ya tiene documentos, no
        vuelve a indexar, optimizando tiempo y costo de ejecución.
        """
        if not force and self.is_already_ingested():
            print(
                f"La colección '{self.collection.name}' ya tiene "
                f"{self.collection.count()} fragmentos. Se omite la ingesta "
                f"(usá force=True para reindexar)."
            )
            return 0

        chunks = load_and_chunk_documents()
        print(f"{len(chunks)} fragmentos generados desde {DATA_DIR_NAME}/.")

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._build_metadata(c) for c in chunks]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Ingesta completa: {len(ids)} fragmentos persistidos en '{self.persist_path}'.")
        return len(ids)


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.run()
