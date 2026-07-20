"""
Ejercicio: Refactorización a LCEL Asíncrono

Migra una implementación imperativa basada en SDKs crudos a una arquitectura
declarativa usando LCEL (LangChain Expression Language): una cadena
ChatPromptTemplate -> modelo -> StrOutputParser, ejecutada de forma
asíncrona con `await chain.ainvoke(...)`.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

# 1. Configuración del modelo
model: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. Definición del template
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente experto que responde de forma clara y concisa."),
    ("human", "{pregunta}"),
])

# 3. Parser de salida
parser: StrOutputParser = StrOutputParser()

# 4. Construcción de la cadena (Pipeline)
chain: RunnableSerializable = prompt | model | parser

# 5. Ejecución asíncrona
async def main() -> None:
    response: str = await chain.ainvoke({"pregunta": "¿Qué es LCEL en LangChain?"})
    print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
