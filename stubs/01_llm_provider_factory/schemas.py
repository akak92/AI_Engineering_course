"""
Esquemas de datos (Pydantic) para un cliente LLM async multi-proveedor.

Definir esta capa primero evita el clásico "error de diccionarios anidados":
en vez de pasar dicts sueltos (`{"role": "user", "content": "..."}`) entre
funciones y confiar en que todos escriban las claves bien, se valida la
forma de los datos una sola vez, en el borde del sistema.

# TODO: agregar acá cualquier campo específico de tu dominio (ej. un
# `request_id` para trazabilidad, `metadata: dict[str, str]`, etc.).
"""

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Rol del emisor de un mensaje dentro de una conversación."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Provider(str, Enum):
    """
    Proveedores de LLM soportados por el manager.

    # TODO: agregar acá un valor por cada proveedor nuevo que integres
    # (ej. GOOGLE = "google", GROQ = "groq"), y después crear su cliente
    # concreto en `providers.py` y registrarlo en `manager.py`.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    """Un único mensaje dentro de una conversación."""

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje. No puede estar vacío.")


class ModelConfig(BaseModel):
    """
    Configuración de una solicitud a un modelo.

    Pydantic valida automáticamente los rangos declarados: si alguien
    intenta crear un `ModelConfig` con `temperature=5`, lanza un
    `ValidationError` antes de que la solicitud llegue a la API.
    """

    provider: Provider
    model: str = Field(description="Nombre del modelo a usar, ej. 'gpt-4o-mini'.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=8192)


class ModelResponse(BaseModel):
    """Respuesta normalizada de un modelo, sin importar el proveedor de origen."""

    provider: Provider
    model: str
    content: str
    ok: bool = True
    error: str | None = None
