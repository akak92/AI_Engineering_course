"""
TEMPLATE: pipeline LCEL con salida estructurada y reintentos.

Cadena: `prompt | model.with_structured_output(OutputSchema)`, con
`.with_retry()` para tolerar JSON mal formado/incompleto o errores
transitorios de red, y `process_input()` como punto de entrada async con
logging y manejo de errores.

# TODO: adaptar `SYSTEM_PROMPT`, el nombre de la variable de plantilla
# (`{texto}`) y `OutputSchema` a tu caso de uso real.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from schemas import OutputSchema, Provider

load_dotenv()

logger = logging.getLogger(__name__)

# TODO: reemplazar por las instrucciones reales de tu tarea. Sé explícito
# sobre el formato esperado y sobre qué hacer si falta información (evita
# que el modelo "alucine" campos).
SYSTEM_PROMPT = (
    "Sos un asistente que extrae información estructurada del texto que te "
    "da el usuario. Respondé únicamente con los datos solicitados, sin "
    "explicaciones adicionales."
)


def get_model(provider: Provider) -> BaseChatModel:
    """
    Instancia el modelo de chat correspondiente al proveedor solicitado.

    El resto del pipeline programa contra `BaseChatModel`: cambiar de
    proveedor es cuestión de cambiar `LLM_PROVIDER` en el entorno, sin
    tocar `prompt`/`chain`.
    """
    if provider == Provider.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if provider == Provider.ANTHROPIC:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    raise ValueError(f"Proveedor no soportado: {provider}")


# 1. Modelo (elegido vía variable de entorno LLM_PROVIDER)
provider: Provider = Provider(os.getenv("LLM_PROVIDER", Provider.OPENAI.value))
model: BaseChatModel = get_model(provider)

# 2. Prompt modular: roles explícitos, variables de plantilla (nunca
# f-strings hardcodeadas: LangChain gestiona el reemplazo en `.ainvoke()`).
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{texto}"),  # TODO: renombrar la variable si tu dominio lo pide
])

# 3. Salida estructurada + resiliencia ante JSON mal formado/incompleto
structured_model: Runnable = model.with_structured_output(OutputSchema).with_retry(
    stop_after_attempt=3,
)

# 4. Cadena LCEL
chain: Runnable = prompt | structured_model


async def process_input(texto: str) -> OutputSchema | None:
    """
    Ejecuta el pipeline completo sobre `texto` y devuelve el objeto
    `OutputSchema` ya validado.

    Si la extracción falla incluso después de los reintentos de
    `.with_retry()` (JSON persistentemente mal formado, error de red,
    etc.), se captura la excepción, se loguea, y se devuelve `None` en vez
    de dejarla propagar y crashear el llamador.
    """
    logger.info("Procesando entrada (%d caracteres) con proveedor '%s'...", len(texto), provider.value)
    try:
        resultado: OutputSchema = await chain.ainvoke({"texto": texto})
    except Exception as e:  # noqa: BLE001 - último resguardo tras agotar los reintentos
        logger.error("Fallo el procesamiento/validación tras los reintentos: %s", e)
        return None

    logger.info("Procesamiento OK -> %s", resultado.model_dump())
    return resultado


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
        # TODO: reemplazar por un input real de tu dominio
        texto_ejemplo = "Ejemplo de texto de entrada para procesar."
        resultado = await process_input(texto_ejemplo)
        if resultado is None:
            print("No se pudo procesar la entrada (ver logs).")
            return
        print(resultado.model_dump_json(indent=2))

    asyncio.run(_demo())
