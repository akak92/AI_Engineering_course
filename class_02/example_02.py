# Ejemplo: Análisis de reseñas
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSerializable, RunnableParallel

model: ChatOpenAI = ChatOpenAI(model="gpt-4o")
parser: StrOutputParser = StrOutputParser()

# --- Chain 1: Sentiment
sentiment_chain: RunnableSerializable = (
    ChatPromptTemplate.from_template(
        "Clasifica el sentimiento de esta reseña en una palabra "
        "(positivo, negativo o neutral): {review}"
    )
    | model
    | parser
)

# --- Chain 2: Main theme
topic_chain: RunnableSerializable = (
    ChatPromptTemplate.from_template(
        "¿Cuál es el tema principal de esta reseña? Responde en 3 palabras {review}"
    )
    | model
    | parser
)

# Step 1: Corremos en paralelo
analysis: RunnableParallel = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

# Step 2: Sinstesis en una sola linea de ticket
ticket_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
    "sentimiento: {sentiment}\n tema: {topic}.\n " \
    "Genera un ticket de soporte en una sola línea para el equipo (máx. 15 palabras)."
)

# Step 3: Full chain
full_chain: RunnableSerializable = analysis | ticket_prompt | model | parser

async def main() -> None:
    review: str = (
        "El producto es excelente, pero el envío fue muy lento y el servicio al cliente no respondió."
    )
    response: str = await full_chain.ainvoke({"review": review})
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())