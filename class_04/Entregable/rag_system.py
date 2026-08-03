"""
Recuperador Híbrido (Paso 3).

`RAGSystem` encapsula un `EnsembleRetriever` que combina:

- **BM25** (léxico): bueno para términos técnicos exactos y nombres propios
  ("LangGraph", "BM25Retriever", "Serverless") que un embedding semántico a
  veces "difumina" al buscar por significado.
- **Pinecone** (semántico/vectorial): bueno para preguntas formuladas con
  vocabulario distinto al de los documentos, pero con el mismo significado.

Ambos recuperadores se construyen sobre exactamente el mismo corpus (los
chunks de `chunking.py`, los mismos que subió `ingest.py` a Pinecone), y se
combinan con Reciprocal Rank Fusion (RRF) vía `EnsembleRetriever`.

Uso:
    system = RAGSystem()
    docs = system.retrieve("¿Qué es LangGraph?")
"""

import os

from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from chunking import load_and_chunk_documents
from ingest import EMBEDDING_MODEL, NAMESPACE
from pinecone_setup import get_index, get_pinecone_client

load_dotenv()

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Pesos del ensemble para la fusión por rank recíproco (RRF). Por defecto,
# igual importancia para el recuperador léxico y el semántico.
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))


class RAGSystem:
    """Encapsula un `EnsembleRetriever` (BM25 + Pinecone) de búsqueda híbrida."""

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self.bm25_retriever: BaseRetriever = self._build_bm25_retriever(top_k)
        self.vector_retriever: BaseRetriever = self._build_vector_retriever(top_k)

        # Reciprocal Rank Fusion: combina los rankings de ambos
        # recuperadores ponderados por BM25_WEIGHT/VECTOR_WEIGHT.
        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )

    @staticmethod
    def _build_bm25_retriever(top_k: int) -> BM25Retriever:
        """
        Reconstruye el índice léxico BM25 en memoria a partir de los mismos
        chunks que `ingest.py` subió a Pinecone (mismo `chunking.py`), para
        que ambos recuperadores operen sobre el mismo corpus.
        """
        chunks = load_and_chunk_documents()
        return BM25Retriever.from_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "category": c.category, "chunk_index": c.chunk_index}
                for c in chunks
            ],
            ids=[c.id for c in chunks],
            k=top_k,
        )

    @staticmethod
    def _build_vector_retriever(top_k: int) -> BaseRetriever:
        """Conecta con el índice/namespace de Pinecone poblado por `ingest.py`."""
        client = get_pinecone_client()
        index = get_index(client)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=NAMESPACE,
        )
        return vectorstore.as_retriever(search_kwargs={"k": top_k})

    def retrieve(self, query: str) -> list[Document]:
        """Devuelve hasta `top_k` documentos combinando resultados léxicos y semánticos."""
        return self.retriever.invoke(query)[: self.top_k]


if __name__ == "__main__":
    system = RAGSystem()
    for doc in system.retrieve("¿Qué es LangGraph?"):
        print(f"[{doc.metadata.get('source')}] {doc.page_content[:100]}...")
