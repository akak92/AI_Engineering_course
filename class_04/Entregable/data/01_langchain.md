# LangChain: Framework para aplicaciones con LLMs

## ¿Qué es LangChain?

LangChain es un framework de código abierto para construir aplicaciones
impulsadas por modelos de lenguaje (LLMs). Provee abstracciones comunes
(prompts, modelos de chat, parsers de salida, retrievers, memoria) y una
forma declarativa de componerlas llamada **LCEL** (LangChain Expression
Language).

## LCEL: composición declarativa con el operador pipe

En LCEL, cualquier componente que implemente la interfaz `Runnable` puede
encadenarse con el operador `|`. Por ejemplo:

```python
chain = prompt | model | output_parser
```

Esto crea una cadena donde la salida de `prompt` (una lista de mensajes) se
pasa como entrada a `model`, y la salida de `model` (un `AIMessage`) se pasa
como entrada a `output_parser`. Todas las cadenas LCEL exponen los mismos
métodos: `invoke`, `batch`, `stream`, y sus versiones asíncronas `ainvoke`,
`abatch`, `astream`.

## Componentes principales

- **`ChatPromptTemplate`**: define la estructura del prompt con roles
  (`system`, `human`, `ai`) y variables de plantilla (`{variable}`), sin
  necesidad de f-strings hardcodeadas.
- **Modelos de chat (`BaseChatModel`)**: `ChatOpenAI`, `ChatAnthropic`, etc.
  Todos implementan la misma interfaz, lo que permite intercambiar
  proveedores sin tocar el resto de la cadena.
- **Output parsers**: `StrOutputParser` (extrae el texto plano de la
  respuesta) o `PydanticOutputParser` / `with_structured_output()` (fuerzan
  una salida validada contra un esquema Pydantic).
- **Retrievers**: componentes que, dado un texto de consulta, devuelven los
  documentos más relevantes desde una fuente (por ejemplo, una base
  vectorial como ChromaDB). Implementan `Runnable`, por lo que también
  pueden encadenarse con `|` dentro de una cadena LCEL.
- **Memoria y estado**: en aplicaciones conversacionales, LangChain permite
  inyectar el historial de la conversación como parte del prompt.

## RAG (Retrieval-Augmented Generation) con LangChain

Un patrón muy común es combinar un retriever con un modelo de lenguaje para
que las respuestas estén "ancladas" (grounded) en documentos reales, en
lugar de depender solo del conocimiento paramétrico del modelo. El flujo
típico es:

1. El usuario hace una pregunta.
2. Un retriever busca los fragmentos de texto más relevantes en una base
   vectorial (usando similitud coseno sobre embeddings).
3. Esos fragmentos se insertan como "contexto" dentro del prompt.
4. El modelo genera una respuesta basada exclusivamente en ese contexto,
   idealmente con instrucciones explícitas de no inventar información si la
   respuesta no está presente.

## Resiliencia: `.with_retry()`

Cuando se usa salida estructurada (`with_structured_output`), el LLM puede
devolver ocasionalmente un JSON mal formado o incompleto. El método
`.with_retry(stop_after_attempt=N)` reintenta automáticamente la llamada
ante fallos de validación o errores transitorios de red, sin necesidad de
escribir lógica de reintento manual.
