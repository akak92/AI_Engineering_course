"""
Pipeline de Extracción de Entidades Técnicas.

Recibe un texto crudo (ej. un log de error o una descripción de
arquitectura) y devuelve un objeto `TechExtraction` validado, usando una
cadena LCEL: prompt | model.with_structured_output(schema).
"""

import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from schemas import Provider, TechExtraction

# Carga el .env acá mismo (no solo en main.py): así, sin importar quién
# importe este módulo primero, LLM_PROVIDER y las API keys ya están
# disponibles antes de leerlas más abajo.
load_dotenv()

logger = logging.getLogger(__name__)


def get_model(provider: Provider) -> BaseChatModel:
    """
    Instancia el modelo de chat correspondiente al proveedor solicitado.

    Reutiliza la lógica de intercambiabilidad de la Pre-entrega 1
    (`LLMFactory`/`AsyncLLMManager`): el resto del pipeline programa contra
    `BaseChatModel`, sin importar si por debajo corre OpenAI o Anthropic.
    """
    if provider == Provider.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if provider == Provider.ANTHROPIC:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    raise ValueError(f"Proveedor no soportado: {provider}")


# 1. Configuración del modelo (elegido vía variable de entorno LLM_PROVIDER)
provider: Provider = Provider(os.getenv("LLM_PROVIDER", Provider.OPENAI.value))
model: BaseChatModel = get_model(provider)

# 2. Prompt Template modular (sin f-strings hardcodeados)
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un analista técnico. Analizá el texto que te da el usuario y "
        "extraé: las tecnologías mencionadas, el nivel de criticidad "
        "(baja, media o alta) y un resumen técnico breve. "
        "Respondé únicamente con los datos solicitados, sin explicaciones adicionales.",
    ),
    ("human", "Texto a analizar:\n{texto}"),
])

# 3. Salida estructurada + resiliencia
# .with_structured_output() obliga al modelo a devolver un TechExtraction
# válido (o falla la validación). .with_retry() reintenta automáticamente
# si el LLM devuelve un JSON mal formado/incompleto o hay un error transitorio.
structured_model: Runnable = model.with_structured_output(TechExtraction).with_retry(
    stop_after_attempt=3,
)

# 4. Cadena LCEL: prompt | model.with_structured_output(schema)
chain: Runnable = prompt | structured_model


async def process_text(text: str) -> TechExtraction | None:
    """
    Ejecuta el pipeline completo sobre un texto crudo y devuelve el objeto
    `TechExtraction` ya validado.

    Registra logs antes y después de la llamada para poder observar el
    proceso de validación. Si la extracción falla incluso después de los
    reintentos de `.with_retry()` (JSON persistentemente mal formado,
    error de red, etc.), se captura la excepción, se loguea y se devuelve
    `None` en vez de dejarla propagar.
    """
    logger.info(
        "Procesando texto (%d caracteres) con proveedor '%s'...", len(text), provider.value
    )
    try:
        resultado: TechExtraction = await chain.ainvoke({"texto": text})
    except Exception as e:  # noqa: BLE001 - último resguardo tras agotar los reintentos
        logger.error("Fallo la extracción/validación tras los reintentos: %s", e)
        return None

    logger.info(
        "Extracción validada OK -> tecnologias=%s | criticidad=%s",
        resultado.tecnologias,
        resultado.nivel_de_criticidad.value,
    )
    return resultado

