"""
Script de demostración de procesamiento concurrente con asyncio.

Simula la generación de embeddings para un lote de textos, limitando la
concurrencia mediante un semáforo y aplicando un timeout a cada solicitud
individual. Al finalizar, reporta cuántos textos se procesaron con éxito
y cuáles fallaron (por ejemplo, por exceder el tiempo límite).
"""

from asyncio import Semaphore
import asyncio
import random
import time

async def get_embedding(text: str, sem: asyncio.Semaphore) -> dict:
    """
    Genera (simula) el embedding de un texto respetando un límite de concurrencia.

    Adquiere el semáforo antes de ejecutar y simula una latencia aleatoria
    entre 0.5 y 2.5 segundos. Si la operación supera los 2 segundos, se
    considera fallida por timeout.

    Args:
        text: Texto para el cual se generará el embedding.
        sem: Semáforo asyncio usado para limitar el número de tareas concurrentes.

    Returns:
        dict: Si tiene éxito, incluye "text", "dim" (dimensión del embedding)
        y "ok": True. Si falla por timeout, incluye "text", "ok": False
        y "error" con el mensaje de error.
    """
    async with sem:
        latency: float = random.uniform(0.5, 2.5)
        try:
            async with asyncio.timeout(2.0):
                await asyncio.sleep(latency)
                return {"text": text, "dim": 1536, "ok" : True}
        except TimeoutError:
            return {"text": text, "ok" : False, "error": "TimeoutError"}

async def embed_batch(texts: list[str]) -> list[dict]:
    """
    Procesa un lote de textos generando sus embeddings de forma concurrente.

    Limita la concurrencia a 3 tareas simultáneas mediante un semáforo y
    ejecuta todas las tareas en paralelo con asyncio.gather.

    Args:
        texts: Lista de textos a procesar.

    Returns:
        list[dict]: Lista de resultados (uno por texto), cada uno con el
        formato devuelto por get_embedding.
    """
    sem: Semaphore = asyncio.Semaphore(3)
    tasks: list = [get_embedding(t, sem) for t in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)

# Entry point for the script
async def main():
    """
    Punto de entrada del script.

    Genera una lista de textos de ejemplo, procesa su embedding en lote
    midiendo el tiempo total transcurrido, y muestra en consola un resumen
    con la cantidad de textos procesados exitosamente y el detalle de los
    que fallaron.
    """
    textos: list[str] = [f"documento_{i}" for i in range(8)]
    inicio: float = time.perf_counter()
    resultados: list[dict] = await embed_batch(textos)
    elapsed: float = time.perf_counter() - inicio

    ok: list[dict] = [r for r in resultados if isinstance(r, dict) and r.get("ok")]
    fallidos: list[dict] = [r for r in resultados if isinstance(r, dict) and not r.get("ok")]

    print(f"Procesados OK: {len(ok)}/ {len(textos)} en {elapsed:.2f}s")

    for f in fallidos:
        print(f"Fallido: {f.get('text')} - Error: {f.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())



