"""
Esquemas de datos (Pydantic) para el Unified Async LLM Client.

Definir esta capa primero evita el clásico "error de diccionarios anidados":
en vez de pasar dicts sueltos (`{"role": "user", "content": "..."}`) entre
funciones y confiar en que todos escriban las claves bien, validamos la
forma de los datos una sola vez, en el borde del sistema.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Rol del emisor de un mensaje dentro de una conversación."""

    SYSTEM: str = "system"
    USER: str = "user"
    ASSISTANT: str = "assistant"


class Provider(str, Enum):
    """Proveedores de LLM soportados por el manager."""

    OPENAI: str = "openai"
    ANTHROPIC: str = "anthropic"


class ChatMessage(BaseModel):
    """Un único mensaje dentro de una conversación."""

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje. No puede estar vacío.")


class ModelConfig(BaseModel):
    """
    Configuración de una solicitud a un modelo.

    Pydantic valida automáticamente los rangos declarados: si alguien
    intenta crear un ModelConfig con temperature=5, lanzará un
    ValidationError antes de que la solicitud llegue a la API.
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
