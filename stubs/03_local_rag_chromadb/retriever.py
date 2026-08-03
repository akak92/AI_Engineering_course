"""
TEMPLATE: capa de recuperación (retriever) sobre ChromaDB.

Se conecta a la misma colección poblada por `ingest.py`, reutilizando
exactamente el mismo `COLLECTION_NAME`/`EMBEDDING_MODEL`/`PERSIST_DIR`, para
garantizar que la consulta se embeba con el mismo modelo que se usó al
indexar los documentos.
"""

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from ingest import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR

load_dotenv()


@dataclass
class RetrievedFragment:
    """Un fragmento recuperado desde ChromaDB, con su fuente y distancia."""

    source: str
    text: str
    distance: float


class SemanticRetriever:
    """Busca los fragmentos más relevantes a una consulta en ChromaDB."""

    def __init__(
        self,
        persist_path: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedding_fn = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedFragment]:
        """
        Devuelve hasta `top_k` fragmentos más similares a `query`, ordenados
        por relevancia (menor distancia primero).

        # TODO: `top_k` entre 3 y 5 suele ser un buen default: pasar
        # demasiado contexto al LLM degrada su atención ("Lost in the
        # Middle") y sube el costo de tokens sin mejorar la respuesta.
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                f"La colección '{self.collection.name}' está vacía. "
                f"Corré `python ingest.py` antes de consultar."
            )

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        fragments: list[RetrievedFragment] = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i] or {}
            fragments.append(
                RetrievedFragment(
                    source=str(metadata.get("source", "desconocido")),
                    text=results["documents"][0][i],
                    distance=results["distances"][0][i],
                )
            )
        return fragments
