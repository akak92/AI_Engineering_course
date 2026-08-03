"""
TEMPLATE: preparación de infraestructura de Pinecone (Serverless).

Verifica si el índice existe y lo crea si hace falta, con la dimensión
correspondiente al modelo de embeddings elegido.

# Nota de compatibilidad importante: si instalás `langchain-pinecone`, pip
# puede resolver una versión de `pinecone` distinta (más vieja) a la que
# tenías. Las funciones acá usan la API estable (`pc.has_index`,
# `pc.create_index`, `pc.describe_index`, `pc.Index`) que se mantiene igual
# entre versiones recientes del SDK — pero si algo falla, correr
# `pip show pinecone` y revisar el changelog de esa versión puntual.

Uso:
    python pinecone_setup.py
"""

import os
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException

if TYPE_CHECKING:
    from pinecone.db_data import _Index as Index

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "rag-hybrid-docs")
# TODO: si cambiás de modelo de embeddings, esta dimensión tiene que
# actualizarse junto con él (1536 = "text-embedding-3-small" de OpenAI), o
# la ingesta fallará por "mismatch de dimensiones".
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# "cosine" es la métrica recomendada para embeddings de OpenAI: no vienen
# normalizados a magnitud 1, y cosine mide similitud direccional sin que la
# magnitud del vector distorsione el ranking.
METRIC = "cosine"


def get_pinecone_client() -> Pinecone:
    """Instancia el cliente de Pinecone a partir de PINECONE_API_KEY."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PINECONE_API_KEY en el entorno (.env).")
    return Pinecone(api_key=api_key)


def ensure_index(
    pc: Pinecone,
    index_name: str = INDEX_NAME,
    dimension: int = EMBEDDING_DIMENSION,
) -> None:
    """
    Verifica si `index_name` existe; si no, lo crea en modo Serverless y
    espera (polling) a que quede listo. Es idempotente: si el índice ya
    existe, no hace nada.
    """
    if pc.has_index(index_name):
        print(f"El índice '{index_name}' ya existe. No se vuelve a crear.")
        return

    print(
        f"Creando índice serverless '{index_name}' "
        f"(dimensión={dimension}, metric={METRIC}, region={PINECONE_REGION})..."
    )
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    except PineconeApiException as e:
        raise RuntimeError(f"No se pudo crear el índice '{index_name}': {e}") from e

    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)
    print(f"Índice '{index_name}' listo.")


def get_index(pc: Pinecone, index_name: str = INDEX_NAME) -> "Index":
    """Devuelve el cliente de datos (data plane) para operar sobre el índice."""
    return pc.Index(name=index_name)


if __name__ == "__main__":
    client = get_pinecone_client()
    ensure_index(client)
    idx = get_index(client)
    print(f"Estadísticas del índice '{INDEX_NAME}': {idx.describe_index_stats()}")
