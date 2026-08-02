import os
import asyncio
from pinecone import Pinecone, ServerlessSpec

# TODO: Asegurate de tener tu PINECONE_API_KEY en tu entorno de ejecución
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")

async def setup_vector_infraestructure(index_name: str, dimension: int) -> None:
    """
    Configura la infraestructura de Pinecone Serverless y realiza una carga inicial.
    """
    # 1. Inicializar el cliente de pinecone
    pc: Pinecone = Pinecone(api_key=PINECONE_API_KEY)

    # 2. rear el índice serverless si no existe
    if index_name not in pc.list_indexes().names():
        print(f"Creando índice serverless '{index_name}' con dimensión {dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            )
        )

    # Esperar a que el indice esté listo
    # verificamos
    while not pc.describe_index(index_name).status['ready']:
        await asyncio.sleep(1)

    # 3. conectar el indice
    index = pc.index(index_name)

    # 4. Operacion CRUD: upsert con namespace
    # Simulamos verctores (en un caso real vendrían de un modelo de embeddings)
    sample_vectors = [
        {"id": "vec1", "values": [0.1] * dimension, "metadata": {"source": "doc1", "topic" : "infra"}},
        {"id": "vec2", "values": [0.2] * dimension, "metadata": {"source": "doc2", "topic" : "security"}},
        {"id": "vec3", "values": [0.3] * dimension, "metadata": {"source": "doc3", "topic" : "devops"}},
    ]

    print(f"Upsertando {len(sample_vectors)} vectores en el índice '{index_name}'...")
    index.upsert(vectors=sample_vectors, namespace="dev-environment")

    # 5. Verificacion
    stats = index.describe_index_stats()
    print(f"Estadísticas del índice '{index_name}': {stats}")
    return index

if __name__ == "__main__":
    asyncio.run(setup_vector_infraestructure(index_name="my_serverless_index", dimension=1536))

'''
Próximos pasos

Agregá una función de búsqueda (`query`) sobre el namespace creado para completar el ciclo CRUD.
Implementá batching: dividí vectores en chunks y ejecutá múltiples upserts para escalabilidad.
Añadí comentarios que expliquen por qué elegiste cosine como métrica para el modelo de embeddings utilizado.
Incluí manejo de excepciones (`try/except`) alrededor de las operaciones de red para hacer el código más

'''