from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
import asyncio

# 1. Definición de import
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    "translate the following text to {language}: {text}")

# 2. El modelo
model: ChatOpenAI = ChatOpenAI(model="gpt-4o")

# 3. El parser de salida
parser: StrOutputParser = StrOutputParser()


# === Cadena ====
chain: RunnableSerializable = prompt | model | parser

# === Ejecución ====
async def main() -> None:
    response: str = await chain.ainvoke({"language": "Español", "text": "LCEL is powerfull"})
    print(response)


if __name__ == "__main__":
    asyncio.run(main())