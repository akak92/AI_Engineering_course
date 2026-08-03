"""
Esquemas (Pydantic) del template de RAG local.

`RAGResponse` es el contrato que debe cumplir la respuesta generada por el
LLM: si el modelo devuelve algo que no respeta esta forma, la validación de
Pydantic falla y `.with_retry()` (en `rag_chain.py`) reintenta la llamada.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Proveedores de LLM soportados para la etapa de generación."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RAGResponse(BaseModel):
    """Respuesta grounded generada a partir del contexto recuperado."""

    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta a la pregunta del usuario, basada exclusivamente en "
            "el CONTEXTO recibido. Si la respuesta no está en el CONTEXTO, "
            "debe decir explícitamente que no tiene acceso a esa información, "
            "en vez de inventar o completar con conocimiento general."
        ),
    )
    referencias: list[str] = Field(
        default_factory=list,
        description=(
            "Nombres de archivo (campo 'source' de los fragmentos, tal como "
            "aparecen en las etiquetas '[Fuente: ...]' del CONTEXTO) que se "
            "usaron efectivamente para responder. Lista vacía si no se usó "
            "ningún fragmento del CONTEXTO."
        ),
    )
