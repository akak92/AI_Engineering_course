# LangSmith: observabilidad y evaluación de aplicaciones LLM

## ¿Qué es LangSmith?

LangSmith es la plataforma de observabilidad, testing y monitoreo de
LangChain (y de aplicaciones LLM en general, incluso si no usan LangChain).
Su objetivo es responder una pregunta difícil de depurar con logs
tradicionales: **¿por qué el modelo respondió esto y no aquello?**

## Tracing (trazabilidad)

Cuando se activa LangSmith (normalmente configurando variables de entorno
como `LANGCHAIN_TRACING_V2=true` y `LANGCHAIN_API_KEY`), cada ejecución de
una cadena LCEL o un grafo de LangGraph queda registrada como un **trace**:
un árbol jerárquico que muestra cada paso intermedio (qué prompt exacto se
envió, qué modelo respondió, cuánto tardó, cuántos tokens consumió, si hubo
un reintento por `.with_retry()`, etc.). Esto es especialmente valioso en
sistemas RAG, donde conviene ver exactamente qué fragmentos recuperó el
retriever antes de que el modelo generara la respuesta final.

## Datasets y evaluación

LangSmith permite crear **datasets** de ejemplos (pares pregunta/respuesta
esperada, o pregunta/contexto/respuesta esperada) y correr **evaluadores**
sobre ellos, ya sea de forma automática (por ejemplo, otro LLM que juzga si
la respuesta es correcta o si está efectivamente "anclada" en el contexto
recuperado) o manual (un humano revisando y calificando cada resultado).
Esto convierte la evaluación de un sistema RAG en un proceso repetible y
medible, en lugar de una revisión ad-hoc.

## Monitoreo en producción

Además de la fase de desarrollo, LangSmith puede usarse para monitorear una
aplicación ya desplegada: detectar picos de latencia, respuestas que
activan el fallback de "no lo sé", tasas de error del LLM, o consultas de
usuarios que no encuentran buena cobertura en la base de documentos (una
señal de que faltan documentos en la ingesta).

## Por qué importa para un sistema RAG

En un pipeline RAG, los errores pueden ocurrir en distintas etapas:
mal chunking, embeddings no coincidentes entre indexado y consulta, un
`top_k` mal calibrado, o un prompt que no restringe bien al modelo. Sin
trazabilidad es difícil saber en cuál de esas etapas está el problema.
LangSmith permite inspeccionar cada etapa por separado: qué fragmentos trajo
el retriever, con qué score de similitud, y qué prompt final recibió el
modelo antes de responder.
