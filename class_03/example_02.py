import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str):
        # 1. Configurar persistencia local
        self.client = chromadb.PersistentClient(path=persist_path)

        # 2. Configurar función de embedding (SentenceTransformer)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # 3. Obtener o crear la collection
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            embedding_function = self.embedding_fn
        )

    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        implementa el método upsert para evitar errores por ids duplicados
        """
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Upserted {len(ids)} documents successfully.")
        except Exception as e:
            print(f"Error during upsert: {e}")

    def semantic_search(self, query_text: str, n_results: int=3):
        """
        Realiza una búsqueda por similitud
        Retorna los dcumentos más cercanos al query_text (estructurados)
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        #Formatear la salida para que sea legible
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return formatted_results

    def delete_by_id(self, ids: List[str]):
        """
        Elimina documentos de la colección por sus IDs
        """
        try:
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} documents successfully.")
        except Exception as e:
            print(f"Error during deletion: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el gestor de memoria vectorial
    memory_manager = VectorMemoryManager(persist_path="./vector_db", collection_name="my_collection")

    # Datos de ejemplo
    sample_ids = ["doc1", "doc2"]
    sample_docs = [
        "Python 3.12 introduce mejoras en asyncio y tipos.",
        "Las bases de datos vectoriales son la clave para la memoria de largo plazo en IA."
    ]
    sample_meta = [{"source": "news"}, {"source": "tech_blog"}]

    memory_manager.upsert_documents(ids=sample_ids, documents=sample_docs, metadatas=sample_meta)

    # Busqueda
    search_query = "Mejoras en Python"
    matches = memory_manager.semantic_search(query_text=search_query)
    for match in matches:
        print(f"Match encontrado: {match['document']} (Score: {match['distance']})")
