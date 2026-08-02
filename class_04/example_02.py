import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Mock del índice de Pinecone para testing
class MockPineconeIndex:
    async def upsert(self, vectors: List[Dict]):
        # Simula la inserción en la base de datos
        print(f"[Pinecone] Insertando batch de {len(vectors)} vectores...")
        return {"upserted_count": len(vectors)}

    async def query(self, vector: List[float], top_k: int, filter: Dict = None):
        # Simula una consulta con filtro
        print(f"[Pinecone] Consultando con filtro: {filter}")
        return {"matches": []}

class IngestionPipeline:
    def __init__(self, index: MockPineconeIndex):
        self.index = index

    def create_metadata(self, doc_text: str, category: str, author: str) -> Dict[str, Any]:
        """
        Genera metadatos enriquecidos para un documento
        """
        return {
            "text": doc_text[:100], # Guardamos snippet para referencia rapida
            "category": category,
            "author": author,
            "ingested_at": datetime.utcnow().isoformat(),
            "char_count": len(doc_text)
        }

    async def process_and_upsert_batches(self, documents: List[Dict], batch_size: int = 100):
        """
        Procesa documentos en batches para optimizar ingesta masiva.
        """
        total_upserted = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors_to_upsert = []
            for doc in batch:
                # Simulamos un embedding (dimensión 3 para ejemplo)
                dummy_embedding = [0.1, 0.2, 0.3]

                vectors_to_upsert.append({
                    "id": str(uuid.uuid4()),
                    "values": dummy_embedding,
                    "metadata": self.create_metadata(doc["text"], doc["category"], doc["author"])
                })

            result = await self.index.upsert(vectors=vectors_to_upsert)
            total_upserted += result["upserted_count"]

        return total_upserted

    async def search_by_category(self, query_vector: List[float], category: str):
        """
        Ejemplo de busqueda con filtrado por metadatos
        """
        return await self.index.query(
            vector=query_vector, 
            top_k=5, 
            filter={"category": category}
        )

# Flujo de ejecución
async def main():
    index = MockPineconeIndex()
    pipeline = IngestionPipeline(index)
    
    # Datos de ejemplo
    raw_docs = [
        {"text": "reporte financiero Q1", "category": "finance", "author": "Alice"},
        {"text": "Manual de ingeniería React", "category": "tech", "author": "Bob"}
    ]
    
    print("--- Iniciando ingesta masiva ---")
    count = await pipeline.process_and_upsert_batches(raw_docs, batch_size=2)
    print(f"Total de documentos upsertados: {count}")

    print("--- Realizando búsqueda por categoría 'finance' ---")
    search_results = await pipeline.search_by_category(query_vector=[0.1, 0.2, 0.3], category="finance")
    print(f"Resultados de búsqueda: {search_results}")

if __name__ == "__main__":
    asyncio.run(main())

"""
Correcciones

El método process_and_upsert_batches carece de manejo de errores; agregá try/except para capturar fallos de la API y evitar la pérdida de toda la ingesta.
En search_by_category usás filter={'category': category} en lugar de filter={'category': {'$eq': category}}; los operadores avanzados de Pinecone mejoran la expresividad y claridad.
Faltan validaciones de entrada: verificá que cada documento tenga 'text', 'category' y 'author' antes de procesarlo.
No se considera el aislamiento multi‑tenant; podrías incluir un campo 'user_id' en metadatos y filtrar por él para separar datos por usuario.
Próximos pasos

Explorá la documentación de Pinecone sobre filtros con $in, $gt y $and para dominar el filtrado avanzado.
Implementá una estrategia de reintentos con backoff exponencial para manejar errores transitorios de red.
Probá tu pipeline con documentos que tengan campos faltantes para ver cómo falla y luego añadí las validaciones necesarias.
Si el proyecto evoluciona a multi‑tenant, asegurate de que cada búsqueda incluya siempre el filtro de tenant correspondiente para no exponer datos de otros usuarios.
"""