# LangGraph: orquestación de agentes con estado

## ¿Qué es LangGraph?

LangGraph es una librería construida sobre LangChain para modelar flujos de
trabajo de agentes como **grafos con estado**, en lugar de cadenas lineales.
Mientras que LCEL es ideal para pipelines de un solo sentido (prompt ->
modelo -> parser), LangGraph está pensado para flujos con **ciclos**,
decisiones condicionales y múltiples pasos que dependen de un estado
compartido.

## Conceptos clave

- **`State`**: una estructura de datos (típicamente un `TypedDict` o un
  modelo Pydantic) que representa la información que fluye a través del
  grafo. Cada nodo puede leer y actualizar el estado.
- **Nodos (`nodes`)**: funciones (o cadenas LCEL) que reciben el estado
  actual, ejecutan alguna lógica (llamar a un LLM, invocar una herramienta,
  consultar una base de datos) y devuelven una actualización parcial del
  estado.
- **Aristas (`edges`)**: conexiones entre nodos. Pueden ser fijas (siempre
  ir de A a B) o **condicionales**, donde una función decide a qué nodo
  saltar según el contenido del estado (por ejemplo, "si la respuesta
  requiere una herramienta, ir al nodo de ejecución de herramientas; si no,
  terminar").
- **Ciclos**: a diferencia de un DAG (grafo acíclico dirigido) puro,
  LangGraph permite que el flujo vuelva a un nodo ya visitado, lo cual es
  esencial para patrones de agentes tipo ReAct (razonar -> actuar ->
  observar -> repetir).
- **Checkpointing**: LangGraph puede persistir el estado del grafo en cada
  paso (por ejemplo, en SQLite o Postgres), lo que permite pausar, retomar
  o inspeccionar la ejecución de un agente en cualquier punto, y habilita
  patrones de "human-in-the-loop" (pedir confirmación humana antes de
  continuar).

## Diferencia práctica con una cadena LCEL simple

Una cadena LCEL (`prompt | model | parser`) es determinística en su
estructura: siempre ejecuta los mismos pasos en el mismo orden. Un grafo de
LangGraph, en cambio, puede decidir dinámicamente su propio camino de
ejecución en tiempo real, según el estado. Por eso LangGraph se usa
típicamente para:

- Agentes que deciden qué herramienta usar en cada paso.
- Flujos de aprobación humana intercalados con pasos automáticos.
- Sistemas multi-agente donde distintos nodos representan distintos
  "roles" (por ejemplo, un nodo investigador y un nodo redactor).

## Relación con RAG

En un sistema RAG simple, una cadena LCEL lineal suele ser suficiente
(retriever -> prompt -> modelo -> parser). LangGraph se vuelve útil cuando
el RAG necesita lógica adicional, como decidir si es necesario reformular
la pregunta del usuario antes de buscar, si hace falta una segunda ronda de
búsqueda porque los documentos recuperados no fueron suficientes, o si la
respuesta debe pasar por un paso de verificación antes de entregarse al
usuario.
