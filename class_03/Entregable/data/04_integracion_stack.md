# Cómo se integran LangChain, LangGraph y LangSmith

## El stack completo

Los tres proyectos son complementarios y suelen usarse juntos:

- **LangChain** aporta las abstracciones base: prompts, modelos de chat,
  parsers de salida y la composición declarativa vía LCEL (`prompt | model
  | parser`).
- **LangGraph** se construye sobre LangChain para orquestar flujos más
  complejos que una cadena lineal: grafos con estado, ciclos, decisiones
  condicionales y agentes que pueden usar herramientas.
- **LangSmith** observa la ejecución de cualquiera de los dos anteriores
  (cadenas LCEL o grafos de LangGraph), generando trazas, permitiendo crear
  datasets de evaluación y monitoreando el comportamiento en producción.

## Ejemplo de flujo combinado en un sistema RAG

1. **Ingesta (LangChain + ChromaDB)**: se usa `RecursiveCharacterTextSplitter`
   para fragmentar documentos y se persisten en una base vectorial.
2. **Cadena de recuperación y generación (LCEL)**: una cadena simple
   `retriever | prompt | model | parser` resuelve la mayoría de los casos
   de un RAG básico: buscar contexto, construir el prompt con ese contexto y
   generar una respuesta grounded (basada solo en lo recuperado).
3. **Orquestación más avanzada (LangGraph)**: si el sistema necesita decidir
   dinámicamente si debe reformular la búsqueda, hacer una segunda consulta,
   o verificar la respuesta antes de entregarla, esa lógica condicional se
   modela mejor como un grafo con nodos y aristas condicionales que como una
   cadena lineal.
4. **Observabilidad (LangSmith)**: en cualquiera de los dos casos anteriores,
   activar el tracing permite ver exactamente qué documentos recuperó el
   sistema para cada pregunta, con qué score de similitud, y si el modelo
   efectivamente basó su respuesta en ese contexto o "alucinó" información
   que no estaba presente.

## Buenas prácticas al combinar los tres

- Mantener el **mismo modelo de embeddings** para indexar documentos y para
  convertir la consulta del usuario en un vector de búsqueda; si difieren,
  la distancia vectorial pierde sentido y las recuperaciones son
  esencialmente aleatorias.
- Limitar el **`top_k`** de documentos recuperados (usualmente entre 3 y 5).
  Pasar demasiados fragmentos al modelo no solo incrementa el costo, sino
  que puede degradar la calidad de la respuesta por el efecto conocido como
  "Lost in the Middle": los modelos prestan menos atención a la información
  ubicada en el medio de un contexto muy largo.
- Instruir explícitamente al modelo, en el prompt de sistema, para que
  responda "no lo sé" o equivalente cuando la respuesta no esté presente en
  el contexto recuperado, en lugar de completar con conocimiento general que
  no puede verificarse.
- Usar LangSmith desde el principio del desarrollo, no solo en producción,
  para detectar temprano problemas de chunking, de recuperación o de
  prompts mal calibrados.
