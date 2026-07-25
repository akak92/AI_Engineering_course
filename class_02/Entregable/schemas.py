"""
Esquema de salida (Pydantic) para el Pipeline de Extracción de Entidades
Técnicas.

Define el "contrato" que debe cumplir cualquier respuesta del LLM: si el
modelo devuelve un JSON que no respeta esta forma (o le falta un campo
obligatorio), Pydantic lanza un ValidationError antes de que ese dato
llegue al resto de la aplicación.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Proveedores de LLM soportados por el pipeline."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class NivelCriticidad(str, Enum):
    """Nivel de criticidad de la arquitectura o el problema descrito en el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechExtraction(BaseModel):
    """Información técnica extraída y validada a partir de un texto crudo."""

    tecnologias: list[str] = Field(
        min_length=1,
        description="Lista de tecnologías, frameworks o herramientas mencionadas en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Nivel de criticidad de la arquitectura o el problema: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        min_length=1,
        description="Resumen técnico breve y claro del texto analizado.",
    )
