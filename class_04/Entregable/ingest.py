"""
Pipeline de Ingesta a Pinecone (Paso 2).

Fragmenta los documentos de `data/` (vía `chunking.py`, compartido con el
recuperador léxico BM25 de `rag_system.py`), genera embeddings con OpenAI y
los sube a Pinecone en batches. Guarda el **texto original dentro de la
metadata** de cada vector (además de la fuente, la categoría y el índice de
chunk), para no depender de una base relacional adicional al recuperar.

Uso:
    python ingest.py
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from chunking import Chunk, load_and_chunk_documents
from pinecone_setup import ensure_index, get_index, get_pinecone_client

load_dotenv()

# Los namespaces separan datos dentro del mismo índice (ej. por entorno o
# tipo de contenido); evita que la búsqueda sea ruidosa/lenta al mezclar
# vectores de dominios distintos en una misma consulta.
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "langchain-docs")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class IngestionPipeline:
    """Chunking -> embeddings -> upsert en batches, con metadata avanzada."""

    def __init__(self, namespace: str = NAMESPACE) -> None:
        self.namespace = namespace
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        client = get_pinecone_client()
        ensure_index(client)
        self.index = get_index(client)

    def is_already_ingested(self) -> bool:
        """Evita reindexar si el namespace ya tiene vectores cargados."""
        stats = self.index.describe_index_stats()
        summary = stats.namespaces.get(self.namespace)
        return summary is not None and summary.vector_count > 0

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict[str, str | int]:
        return {
            "source": chunk.source,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
            # Texto original en la metadata: evita una consulta adicional a
            # una base relacional para reconstruir el contenido del chunk.
            "text": chunk.text,
        }

    def run(self, force: bool = False) -> int:
        """
        Ejecuta el pipeline completo: chunking -> embeddings -> upsert.

        Si `force=False` (default) y el namespace ya tiene vectores, no
        vuelve a indexar (optimiza tiempo y costo de ejecución).
        """
        if not force and self.is_already_ingested():
            print(
                f"El namespace '{self.namespace}' ya tiene vectores cargados. "
                f"Se omite la ingesta (usá force=True para reindexar)."
            )
            return 0

        chunks = load_and_chunk_documents()
        print(f"{len(chunks)} fragmentos generados desde data/.")

        total_upserted = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]

            try:
                embedded_vectors = self.embeddings.embed_documents([c.text for c in batch])
            except Exception as e:  # noqa: BLE001 - no perder toda la ingesta por un batch
                print(f"Error generando embeddings para el batch {i}-{i + len(batch)}: {e}")
                continue

            vectors = [
                {
                    "id": chunk.id,
                    "values": embedding,
                    "metadata": self._build_metadata(chunk),
                }
                for chunk, embedding in zip(batch, embedded_vectors)
            ]

            try:
                result = self.index.upsert(vectors=vectors, namespace=self.namespace)
                total_upserted += result.upserted_count
                print(f"Batch {i}-{i + len(batch)}: {result.upserted_count} vectores upsertados.")
            except Exception as e:  # noqa: BLE001 - continuar con el resto de los batches
                print(f"Error subiendo el batch {i}-{i + len(batch)} a Pinecone: {e}")
                continue

        print(
            f"Ingesta completa: {total_upserted} fragmentos persistidos en "
            f"el namespace '{self.namespace}'."
        )
        return total_upserted


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
