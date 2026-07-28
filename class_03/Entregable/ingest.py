"""
Módulo de Ingesta (Setup) del sistema RAG.

Lee documentos `.txt`/`.md` desde una carpeta (`data/` por defecto), los
fragmenta (chunking) contando tokens reales con `tiktoken` vía
`RecursiveCharacterTextSplitter` (mismo enfoque que `example_01.py`), y
persiste los fragmentos en una colección de ChromaDB usando
`OpenAIEmbeddingFunction`.

Usar la embedding function del propio ChromaDB (en vez de generar los
vectores "a mano") garantiza que se use el mismo modelo tanto para indexar
como para consultar, evitando el error más común de un sistema RAG:
embeddings no coincidentes entre ingesta y búsqueda.

Uso:
    python ingest.py
"""

import os
import re
from pathlib import Path

import chromadb
import tiktoken
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / os.getenv("VECTORSTORE_DIR", "vectorstore")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "langchain_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


class DocumentIngestor:
    """
    Pipeline de ingesta: lee archivos -> limpia -> chunkea (por tokens) ->
    persiste en ChromaDB.
    """

    def __init__(
        self,
        persist_path: Path = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        model_encoding: str = "cl100k_base",
    ) -> None:
        self.persist_path = persist_path
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        # La embedding function queda guardada como parte de la
        # configuración de la collection: Chroma la reutiliza
        # automáticamente tanto al indexar (`upsert`) como al consultar
        # (`query_texts`), así que el modelo nunca puede desincronizarse
        # entre ingesta y búsqueda.
        self.embedding_fn = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _load_documents(data_dir: Path) -> list[tuple[str, str]]:
        """Devuelve una lista de (nombre_archivo, contenido) para cada .txt/.md."""
        archivos = sorted([*data_dir.glob("*.txt"), *data_dir.glob("*.md")])
        if not archivos:
            raise FileNotFoundError(f"No se encontraron archivos .txt/.md en {data_dir}")
        return [(f.name, f.read_text(encoding="utf-8")) for f in archivos]

    def is_already_ingested(self) -> bool:
        """Permite evitar re-indexar si la colección ya tiene documentos."""
        return self.collection.count() > 0

    def run(self, data_dir: Path = DATA_DIR, force: bool = False) -> int:
        """
        Ejecuta el pipeline completo: lectura -> limpieza -> chunking -> upsert.

        Si `force=False` (default) y la colección ya tiene documentos, no
        vuelve a indexar, optimizando tiempo y costo de ejecución.
        """
        if not force and self.is_already_ingested():
            print(
                f"La colección '{self.collection.name}' ya tiene "
                f"{self.collection.count()} fragmentos en '{self.persist_path}'. "
                f"Se omite la ingesta (usá force=True para reindexar)."
            )
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for filename, raw_text in self._load_documents(data_dir):
            cleaned = self._clean_text(raw_text)
            chunks = self.splitter.split_text(cleaned)
            for i, chunk in enumerate(chunks):
                ids.append(f"{filename}::chunk_{i}")
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
            print(f"{filename}: {len(chunks)} fragmentos generados.")

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Ingesta completa: {len(ids)} fragmentos persistidos en '{self.persist_path}'.")
        return len(ids)


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    ingestor.run()
