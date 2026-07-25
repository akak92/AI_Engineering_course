# Validación estructurada y resiliencia de cadenas

import asyncio
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_core.language_models import LanguageModelInput

# TODO 1: Define la clase Pydantic 'EntityExtraction' 
# Debe tener: topic (str), entities (Lista de str), y sentiment_score (float entre 0 y 1)
class EntityExtraction(BaseModel):
    topic: str = Field(description="Tema principal del texto.")
    entities: List[str] = Field(description="Lista de nombres propios, tecnologías o conceptos extraídas del texto.")
    sentiment_score: float = Field(description="Puntuación de 0 a 1 sobre qué tan positivo es el texto", ge=0.0, le=1.0)
    complexity_level: Optional[str] = Field(description="Nivel de complejidad del texto (opcional).",
                                            pattern="^(Low|Medium|High)$")  # Validación opcional para nivel de complejidad

async def run_validated_chain(text: str):
    llm: ChatOpenAI = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # TODO 2: Configura el modelo para usar la salida estructurada con Pydantic
    # Tip: Usa el método .with_structured_output()
    structured_llm: Runnable[LanguageModelInput, EntityExtraction] = llm.with_structured_output(EntityExtraction)
    
    # TODO 3: Agrega una estrategia de reintento con .with_retry() 
    # para que sea resiliente ante fallos de conexión (máximo 3 intentos).
    resilient_llm: Runnable[LanguageModelInput, EntityExtraction] = structured_llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )
    
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en análisis de datos técnicos. Extrae la información solicitada en formato estructurado."),
        ("human", "{input}")
    ])

    chain: RunnableSerializable[dict[str, Any], EntityExtraction] = prompt | resilient_llm
    
    # TODO 4: Une el prompt con el resilient_llm y ejecuta asíncronamente
    # No olvides manejar excepciones con try/except para capturar fallos de validación
    try:
        result: EntityExtraction = await chain.ainvoke({"input": text})
        print("=== Extracción exitosa ===")
        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error crítico en la cadena: : {type(e).__name__}")
        print(f"Detalles del error: {str(e)}")


if __name__ == "__main__":
    sample_text: str = "LangGraph es una extensión de LangChain para agentes cíclicos."
    asyncio.run(run_validated_chain(sample_text))