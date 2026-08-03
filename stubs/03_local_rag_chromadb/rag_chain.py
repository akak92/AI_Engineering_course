"""
TEMPLATE: generación grounded (RAG local).

Cadena LCEL (`prompt | model.with_structured_output(RAGResponse)`) que
recibe los fragmentos recuperados por `SemanticRetriever`, arma un prompt de
"filtro de veracidad" y genera una respuesta validada con Pydantic,
incluyendo solo las referencias que el modelo dice haber usado.

Uso:
    from rag_chain import get_rag_response
    resultado = await get_rag_response("tu pregunta acá")
"""

import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from retriever import RetrievedFragment, SemanticRetriever
from schemas import Provider, RAGResponse

load_dotenv()

logger = logging.getLogger(__name__)

# top_k entre 3 y 5: evita el "Contexto Infinito" (degradación de atención /
# Lost in the Middle) y limita el costo de tokens.
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# TODO: adaptar el tono/dominio de estas instrucciones a tu caso de uso,
# manteniendo las dos reglas clave: (1) responder solo con el CONTEXTO, y
# (2) decir explícitamente que no tiene la información si no está presente.
SYSTEM_PROMPT = (
    "Sos un asistente técnico. Respondé ÚNICAMENTE basándote en el CONTEXTO "
    "que te paso a continuación, extraído de una base de documentos. Si la "
    "respuesta no está en el CONTEXTO, decí explícitamente que no tenés "
    "acceso a esa información; no completes con conocimiento general ni "
    "inventes datos que no estén presentes en el CONTEXTO.\n\n"
    "En 'referencias', incluí únicamente los nombres de archivo indicados "
    "en las etiquetas '[Fuente: ...]' del CONTEXTO que realmente hayas "
    "usado para responder. Si no usaste ningún fragmento, dejala vacía."
)


def get_model(provider: Provider) -> BaseChatModel:
    """Instancia el modelo de chat correspondiente al proveedor solicitado."""
    if provider == Provider.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if provider == Provider.ANTHROPIC:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    raise ValueError(f"Proveedor no soportado: {provider}")


provider: Provider = Provider(os.getenv("LLM_PROVIDER", Provider.OPENAI.value))
model: BaseChatModel = get_model(provider)

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}"),
])

structured_model: Runnable = model.with_structured_output(RAGResponse).with_retry(
    stop_after_attempt=3,
)
chain: Runnable = prompt | structured_model

retriever = SemanticRetriever()


def _build_context(fragments: list[RetrievedFragment]) -> str:
    """Arma el bloque de CONTEXTO a partir de los fragmentos recuperados."""
    if not fragments:
        return "(no se encontraron fragmentos relevantes en la base de documentos)"
    return "\n\n".join(f"[Fuente: {f.source}]\n{f.text}" for f in fragments)


async def get_rag_response(query: str, top_k: int = TOP_K) -> RAGResponse:
    """
    Ejecuta el flujo RAG completo:
    1. Recupera hasta `top_k` fragmentos relevantes para `query`.
    2. Arma el CONTEXTO con esos fragmentos, etiquetados por fuente.
    3. Llama al LLM de forma asíncrona vía la cadena LCEL.
    4. Devuelve un `RAGResponse` validado (respuesta + referencias).

    Si la generación falla incluso después de los reintentos de
    `.with_retry()`, se captura la excepción, se loguea, y se devuelve un
    `RAGResponse` de fallback en vez de dejarla propagar.
    """
    fragments = retriever.retrieve(query, top_k=top_k)
    contexto = _build_context(fragments)

    logger.info(
        "Query: %r -> %d fragmentos recuperados (top_k=%d)", query, len(fragments), top_k
    )
    for f in fragments:
        logger.debug("  - %s (distancia=%.4f)", f.source, f.distance)

    try:
        resultado: RAGResponse = await chain.ainvoke({"contexto": contexto, "pregunta": query})
    except Exception as e:  # noqa: BLE001 - último resguardo tras agotar los reintentos
        logger.error("Fallo la generación de la respuesta RAG: %s", e)
        return RAGResponse(
            respuesta="No pude generar una respuesta en este momento debido a un error interno.",
            referencias=[],
        )

    logger.info("Respuesta generada. Referencias citadas: %s", resultado.referencias)
    return resultado


if __name__ == "__main__":
    import asyncio

    from ingest import DocumentIngestor

    async def _demo() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
        DocumentIngestor().run()  # puebla la base si todavía está vacía
        # TODO: reemplazar por una pregunta real de tu dominio
        resultado = await get_rag_response("¿De qué tratan los documentos?")
        print(f"Respuesta: {resultado.respuesta}")
        print(f"Referencias: {resultado.referencias}")

    asyncio.run(_demo())
