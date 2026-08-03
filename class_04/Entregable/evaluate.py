"""
Script de Evaluación (Paso 4).

Carga un "Golden Set" de preguntas con su documento fuente esperado
(`golden_set.json`), ejecuta cada consulta contra `RAGSystem` y calcula dos
métricas sobre el Top-5 recuperado:

- **Recall@5**: ¿el documento esperado aparece entre los 5 fragmentos
  recuperados? (0 o 1 por consulta, promediado sobre todo el golden set).
- **Precision@5**: de los 5 fragmentos recuperados, ¿qué porcentaje
  pertenece al documento esperado?

Uso:
    python evaluate.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_system import RAGSystem

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


@dataclass
class EvalCase:
    """Un caso del golden set: pregunta + documento fuente esperado."""

    pregunta: str
    documento_id_esperado: str


@dataclass
class EvalResult:
    """Resultado de evaluar un `EvalCase` contra el `RAGSystem`."""

    pregunta: str
    documento_id_esperado: str
    fuentes_recuperadas: list[str]
    hit: bool
    precision_at_k: float


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalCase]:
    """Carga el golden set desde un JSON con pares pregunta/documento esperado."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data]


def evaluate_case(system: RAGSystem, case: EvalCase) -> EvalResult:
    """Ejecuta una consulta y calcula el acierto (Recall) y la Precision@k de ese caso."""
    docs = system.retrieve(case.pregunta)
    fuentes = [str(d.metadata.get("source", "desconocido")) for d in docs]

    aciertos = sum(1 for f in fuentes if f == case.documento_id_esperado)
    hit = aciertos > 0
    precision = aciertos / len(fuentes) if fuentes else 0.0

    return EvalResult(
        pregunta=case.pregunta,
        documento_id_esperado=case.documento_id_esperado,
        fuentes_recuperadas=fuentes,
        hit=hit,
        precision_at_k=precision,
    )


def run_evaluation(golden_set_path: Path = GOLDEN_SET_PATH) -> list[EvalResult]:
    """Corre todo el golden set contra un `RAGSystem` recién instanciado."""
    system = RAGSystem()
    cases = load_golden_set(golden_set_path)
    return [evaluate_case(system, case) for case in cases]


def print_report(results: list[EvalResult]) -> None:
    """Imprime el detalle por caso y el resumen de Recall@5 / Precision@5."""
    print("=== Reporte de Evaluación (Precision@5 / Recall@5) ===\n")
    for r in results:
        estado = "OK" if r.hit else "MISS"
        print(f"[{estado}] {r.pregunta}")
        print(f"    Esperado:   {r.documento_id_esperado}")
        print(f"    Recuperado: {r.fuentes_recuperadas}")
        print(f"    Precision@5: {r.precision_at_k:.2f}\n")

    n = len(results)
    recall_at_5 = sum(1 for r in results if r.hit) / n
    precision_at_5 = sum(r.precision_at_k for r in results) / n

    print("--- Resumen ---")
    print(f"Casos evaluados: {n}")
    print(f"Recall@5:    {recall_at_5:.2%}  (el documento esperado aparece en el Top-5)")
    print(f"Precision@5: {precision_at_5:.2%}  (fragmentos del Top-5 que pertenecen al documento esperado)")


if __name__ == "__main__":
    resultados = run_evaluation()
    print_report(resultados)
