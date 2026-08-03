"""
TEMPLATE de esquema de salida (Pydantic) para una cadena LCEL con salida
estructurada.

# TODO: Este `OutputSchema` es un EJEMPLO. Reemplazalo por el modelo real
# que necesita tu caso de uso (los nombres/campos de acá son solo
# ilustrativos). Reglas a mantener:
# - Cada campo obligatorio sin default fuerza al LLM a completarlo.
# - Usá `Field(min_length=..., description=...)` para que la descripción
#   quede embebida en el JSON schema que ve el modelo (mejora la calidad
#   de la extracción) y para que Pydantic rechace strings vacíos.
# - Si un campo tiene un conjunto cerrado de valores válidos, modelalo como
#   `Enum` (ver `Provider`/`Criticidad` abajo), no como `str` libre.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Proveedores de LLM soportados para la etapa de generación."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Criticidad(str, Enum):
    """# TODO: ejemplo de enum para un campo con valores cerrados."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class OutputSchema(BaseModel):
    """
    # TODO: reemplazar por el "contrato" real de tu pipeline.

    Este es el modelo que fuerza `with_structured_output()`: si el LLM
    devuelve algo que no respeta esta forma, Pydantic lanza un
    `ValidationError` antes de que ese dato llegue al resto de tu
    aplicación, y `.with_retry()` (en `chain.py`) reintenta la llamada.
    """

    resultado_principal: str = Field(
        min_length=1,
        description="# TODO: describí acá qué debe contener este campo.",
    )
    nivel: Criticidad = Field(description="Nivel categórico, restringido al enum.")
    items: list[str] = Field(
        default_factory=list,
        description="Lista de ítems relacionados (vacía si no aplica).",
    )
