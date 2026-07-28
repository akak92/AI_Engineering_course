"""
Script de prueba del sistema RAG.

Corre la ingesta (si la colección todavía está vacía) y ejecuta dos
consultas contra `get_rag_response`:

1. Una pregunta cuya respuesta SÍ está en los documentos (`data/`).
2. Una "pregunta trampa" cuya respuesta NO está en los documentos, para
   verificar que el modelo responda que no tiene esa información en vez de
   alucinarla.

Uso:
    python main.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from ingest import DocumentIngestor
from rag_chain import get_rag_response
from schemas import RAGResponse

PREGUNTA_RESPONDIBLE = (
    "¿Qué es LCEL y cómo se componen los pasos de una cadena en LangChain?"
)
PREGUNTA_TRAMPA = (
    "¿Cuál es el precio mensual de la licencia empresarial de LangSmith?"
)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # Puebla la base vectorial si todavía no tiene documentos (no reindexa
    # si ya existe: ver `DocumentIngestor.is_already_ingested`).
    DocumentIngestor().run()

    casos = [
        ("Pregunta respondible (la respuesta está en los documentos)", PREGUNTA_RESPONDIBLE),
        ("Pregunta trampa (la respuesta NO está en los documentos)", PREGUNTA_TRAMPA),
    ]

    for titulo, pregunta in casos:
        print(f"\n=== {titulo} ===")
        print(f"Q: {pregunta}")
        resultado: RAGResponse = await get_rag_response(pregunta)
        print(f"A: {resultado.respuesta}")
        print(f"Referencias: {resultado.referencias if resultado.referencias else '(ninguna)'}")


if __name__ == "__main__":
    asyncio.run(main())
