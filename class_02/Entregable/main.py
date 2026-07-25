"""
Mini-script de prueba del Pipeline de Extracción de Entidades Técnicas.

Carga las variables de entorno desde `.env`, configura el logging para ver
el proceso de validación/reintentos, y ejecuta `process_text` sobre un
texto de ejemplo.

Uso:
    python main.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from chain import process_text
from schemas import TechExtraction

TEXTO_EJEMPLO = (
    "Detectamos un cuello de botella en producción: la API construida con "
    "FastAPI está sufriendo timeouts bajo carga. Usamos Redis como caché de "
    "sesiones y PostgreSQL como base de datos principal, pero el pool de "
    "conexiones se agota rápido. Es urgente resolverlo antes del próximo pico "
    "de tráfico."
)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    resultado: TechExtraction | None = await process_text(TEXTO_EJEMPLO)

    print("\n--- Resultado ---")
    if resultado is None:
        print("No se pudo obtener una extracción válida (ver logs de error arriba).")
        return

    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
