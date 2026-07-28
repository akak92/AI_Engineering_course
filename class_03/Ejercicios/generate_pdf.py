"""
Genera el PDF entregable del ejercicio "Similitud Semantica vs. Coincidencia
de Palabras Clave" a partir del contenido definido en este script, usando
fpdf2. Incluye el analisis de oraciones, el codigo de similitud coseno con
scikit-learn y un diagrama de flujo dibujado con formas vectoriales
(equivalente al diagrama Mermaid del archivo .md).

Uso:
    python generate_pdf.py
"""

from fpdf import FPDF

OUTPUT_PATH = "ejercicio_01_similitud_semantica.pdf"

CODE_SNIPPET = '''from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Embeddings ya generados por un modelo (ej. text-embedding-3-small)
embedding_oracion_1 = np.array([[0.12, 0.08, -0.03]])  # vector oracion 1
embedding_oracion_2 = np.array([[0.11, 0.07, -0.02]])  # vector oracion 2

# 2. Similitud coseno entre ambos vectores
similitud = cosine_similarity(embedding_oracion_1, embedding_oracion_2)
print(f"Similitud coseno: {similitud[0][0]:.4f}")


# 3. Comparar una oracion base contra un conjunto de candidatas
def obtener_embedding(texto: str) -> list[float]:
    # Placeholder: aqui se llamaria al proveedor real
    # (ej. client.embeddings.create(model="text-embedding-3-small", input=texto))
    ...


oracion_base = "Publicamos cada componente en contenedores dentro de Kubernetes."
candidatas = [
    "El equipo automatizo la entrega continua mediante CI/CD.",
    "El servicio de micro-limpieza es excelente y llega puntual.",
]

vector_base = np.array([obtener_embedding(oracion_base)])
vectores_candidatas = np.array([obtener_embedding(c) for c in candidatas])
similitudes = cosine_similarity(vector_base, vectores_candidatas)

for oracion, score in zip(candidatas, similitudes[0]):
    print(f"[{score:.4f}] {oracion}")'''


class PDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, "Ejercicio: Similitud Semantica vs. Palabras Clave", align="L")
        self.cell(0, 8, f"Pagina {self.page_no()}", align="R")
        self.ln(12)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, "AI Engineering Course - Coderhouse", align="C")


def section_title(pdf: PDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 90)
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 40, 90)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)


def sub_title(pdf: PDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 40, 90)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def body_text(pdf: PDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)


def sentence_table(pdf: PDF, rows: list[tuple[str, str, str]], header: tuple[str, str, str],
                    col_widths: tuple[float, float, float]) -> None:
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_fill_color(20, 40, 90)
    pdf.set_text_color(255, 255, 255)
    for text, width in zip(header, col_widths):
        pdf.cell(width, 8, text, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    fill = False
    for num, sentence, keywords in rows:
        pdf.set_fill_color(240, 243, 250) if fill else pdf.set_fill_color(255, 255, 255)
        start_x = pdf.get_x()
        start_y = pdf.get_y()

        # Measure needed height using a clone-like approach: split lines manually
        line_height = 5
        available_width = col_widths[1] - 2
        pdf.set_xy(start_x + col_widths[0], start_y)
        lines_sentence = pdf.multi_cell(available_width, line_height, sentence, dry_run=True, output="LINES")
        available_width_kw = col_widths[2] - 2
        lines_keywords = pdf.multi_cell(available_width_kw, line_height, keywords, dry_run=True, output="LINES")
        row_height = max(len(lines_sentence), len(lines_keywords), 1) * line_height + 2

        pdf.set_xy(start_x, start_y)
        pdf.cell(col_widths[0], row_height, num, border=1, align="C", fill=True)
        pdf.set_xy(start_x + col_widths[0], start_y)
        pdf.multi_cell(col_widths[1], line_height, sentence, border=1, fill=True)
        pdf.set_xy(start_x + col_widths[0] + col_widths[1], start_y)
        pdf.multi_cell(col_widths[2], line_height, keywords, border=1, fill=True)

        pdf.set_xy(start_x, start_y + row_height)
        fill = not fill


def code_block(pdf: PDF, code: str) -> None:
    pdf.set_fill_color(30, 30, 36)
    pdf.set_text_color(220, 220, 220)
    pdf.set_font("Courier", "", 8.3)
    start_x = pdf.get_x()
    start_y = pdf.get_y()
    lines = code.split("\n")
    padding = 3
    line_height = 4.2

    for line in lines:
        # Manual page-break check before drawing each background strip
        if pdf.get_y() + line_height > pdf.h - pdf.b_margin:
            pdf.add_page()
            start_x = pdf.get_x()

    pdf.set_xy(start_x, start_y)
    total_height = len(lines) * line_height + 2 * padding
    pdf.rect(start_x, start_y, pdf.w - pdf.l_margin - pdf.r_margin, total_height, style="F")
    pdf.set_xy(start_x + padding, start_y + padding)
    for line in lines:
        pdf.set_x(start_x + padding)
        pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(start_y + total_height + 4)
    pdf.set_text_color(0, 0, 0)


def flow_box(pdf: PDF, x: float, y: float, w: float, h: float, text: str,
             fill_color: tuple[int, int, int], shape: str = "rect") -> None:
    pdf.set_fill_color(*fill_color)
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    if shape == "rounded":
        pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=3)
    elif shape == "diamond":
        pdf.polygon([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)], style="DF")
    else:
        pdf.rect(x, y, w, h, style="DF")
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(0, 0, 0)
    # Texto centrado vertical y horizontalmente, con un ancho reducido para
    # que no toque los bordes (especialmente en el rombo).
    text_w = w * (0.62 if shape == "diamond" else 0.88)
    line_height = 3.4
    lines = pdf.multi_cell(text_w, line_height, text, align="C", dry_run=True, output="LINES")
    text_h = len(lines) * line_height
    pdf.set_xy(x + (w - text_w) / 2, y + (h - text_h) / 2)
    pdf.multi_cell(text_w, line_height, text, align="C", border=0)


def arrow(pdf: PDF, x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    pdf.line(x1, y1, x2, y2)
    # arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 2.2
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(angle - 0.4), y2 - size * math.sin(angle - 0.4))
    p3 = (x2 - size * math.cos(angle + 0.4), y2 - size * math.sin(angle + 0.4))
    pdf.set_fill_color(60, 60, 60)
    pdf.polygon([p1, p2, p3], style="F")
    if label:
        pdf.set_font("Helvetica", "I", 7)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        pdf.set_xy(mid_x - 12, mid_y - 4)
        pdf.cell(24, 4, label, align="C")


def build_pdf() -> None:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 15, 18)

    # ---- Portada ----
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 40, 90)
    pdf.multi_cell(0, 12, "Similitud Semantica vs.\nCoincidencia de Palabras Clave", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8, "Evaluacion de un modelo de embeddings para discernir\nrelevancia semantica en el concepto de\n\"Despliegue de microservicios\"", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "AI Engineering Course - Coderhouse", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Ejercicio de la unidad (no es el entregable final)", align="C")
    pdf.set_text_color(0, 0, 0)

    # ---- Seccion 1: Objetivo ----
    pdf.add_page()
    section_title(pdf, "1. Objetivo")
    body_text(
        pdf,
        "Evaluar la capacidad de un modelo de embeddings para diferenciar entre:\n\n"
        "- Relevancia semantica real: oraciones que hablan del mismo concepto tecnico "
        "con vocabulario distinto (deberian tener alta similitud coseno).\n"
        "- Coincidencia superficial de palabras clave: oraciones que comparten terminos "
        "lexicos pero no el significado (deberian tener baja similitud coseno pese a "
        "compartir palabras).\n\n"
        "Concepto tecnico elegido: Despliegue de microservicios."
    )

    # ---- Seccion 2: Oraciones ----
    section_title(pdf, "2. Conjunto de oraciones")
    sub_title(pdf, "2.1 Oraciones semanticamente relevantes (mismo concepto, vocabulario distinto)")
    relevantes = [
        ("1", "Publicamos cada componente de la aplicacion de forma independiente en "
              "contenedores dentro del cluster de Kubernetes.", "contenedores, cluster, Kubernetes"),
        ("2", "El equipo automatizo la entrega continua de los servicios distribuidos "
              "mediante pipelines de CI/CD hacia el entorno productivo.", "CI/CD, pipelines, productivo"),
        ("3", "Cada modulo del sistema se empaqueta como una imagen Docker y se libera "
              "de manera autonoma sin afectar a los demas.", "Docker, imagen, autonoma"),
        ("4", "La orquestacion de los servicios se realiza escalando replicas dinamicamente "
              "segun la demanda en la nube.", "orquestacion, replicas, escalado"),
        ("5", "Lanzamos versiones nuevas de cada microservicio usando estrategias de rollout "
              "progresivo (canary) sin downtime.", "rollout, canary, downtime"),
    ]
    sentence_table(pdf, relevantes, ("#", "Oracion", "Vocabulario clave"), (10, 122, 44))
    pdf.ln(4)

    sub_title(pdf, "2.2 Oraciones trampa (comparten palabras clave, significado opuesto o irrelevante)")
    trampas = [
        ("T1", "El servicio de micro-limpieza es excelente y llega puntual todos los "
               "martes a la oficina.", "Comparte el prefijo \"micro-\" y la palabra \"servicio\", "
               "pero no tiene relacion con software."),
        ("T2", "El despliegue de las tropas se realizo en la frontera tras la orden del "
               "comando militar.", "Comparte la palabra \"despliegue\", pero el contexto es "
               "militar, no tecnico."),
    ]
    sentence_table(pdf, trampas, ("#", "Oracion", "Motivo de la trampa"), (10, 100, 66))
    pdf.ln(4)

    sub_title(pdf, "2.3 Resultado esperado")
    body_text(
        pdf,
        "Un buen modelo de embeddings deberia producir:\n\n"
        "- Similitud coseno ALTA (aprox. 0.6 - 0.9) entre las oraciones 1-5 entre si.\n"
        "- Similitud coseno BAJA (aprox. 0.0 - 0.3) entre las oraciones 1-5 y las oraciones "
        "trampa T1/T2, a pesar de la coincidencia lexica parcial (\"micro-\", \"despliegue\").\n\n"
        "Esto demuestra que el modelo captura el significado contextual y no solo la "
        "superposicion de tokens, a diferencia de un enfoque lexico (ej. TF-IDF o "
        "coincidencia exacta de palabras)."
    )

    # ---- Seccion 3: Codigo ----
    section_title(pdf, "3. Calculo de Similitud Coseno con scikit-learn")
    body_text(
        pdf,
        "scikit-learn provee la funcion cosine_similarity dentro del modulo "
        "sklearn.metrics.pairwise, que recibe matrices de vectores (embeddings) y "
        "devuelve la matriz de similitudes por pares.\n\n"
        "Formula: cos_sim(A, B) = (A . B) / (||A|| * ||B||)\n\n"
        "El resultado va de -1 (opuestos) a 1 (identicos en direccion), siendo 0 la "
        "ortogonalidad (sin relacion)."
    )
    sub_title(pdf, "3.1 Ejemplo en Python")
    code_block(pdf, CODE_SNIPPET)

    sub_title(pdf, "3.2 Notas de implementacion")
    body_text(
        pdf,
        "- cosine_similarity acepta matrices de forma (n_muestras, n_dimensiones); un solo "
        "vector debe pasarse como [[...]] o usando .reshape(1, -1).\n"
        "- Para busqueda semantica conviene calcular los embeddings de los documentos una "
        "sola vez y compararlos contra la query en una sola llamada vectorizada.\n"
        "- Si los embeddings estan normalizados (norma L2 = 1), la similitud coseno "
        "equivale al producto punto simple, mas rapido de calcular."
    )

    # ---- Seccion 4: Diagrama de flujo ----
    pdf.add_page()
    section_title(pdf, "4. Diagrama de flujo: proceso de busqueda semantica")
    body_text(pdf, "Equivalente visual del diagrama Mermaid definido en el archivo .md del ejercicio:")
    pdf.ln(2)

    # El diagrama completo se dibuja en una sola pasada, sin saltos de pagina
    # automaticos, para que las flechas queden perfectamente alineadas con
    # las cajas (rect/polygon no disparan salto de pagina, pero multi_cell si,
    # lo que desalinearia el dibujo si se deja el auto page-break activado).
    pdf.set_auto_page_break(auto=False)

    box_w, box_h, gap = 68, 11, 9
    center_x = pdf.w / 2
    main_x = center_x - box_w / 2
    top_y = pdf.get_y() + 3
    positions: dict[str, tuple[float, float, float, float]] = {}

    y = top_y
    flow_box(pdf, main_x, y, box_w, box_h, "Usuario ingresa una query", (210, 227, 252), "rounded")
    positions["A"] = (main_x, y, box_w, box_h)
    y += box_h + gap

    flow_box(pdf, main_x, y, box_w, box_h, "Generar embedding de la query\n(modelo de embeddings)", (222, 235, 250), "rect")
    positions["B"] = (main_x, y, box_w, box_h)
    y += box_h + gap

    diamond_h = box_h + 5
    flow_box(pdf, main_x, y, box_w, diamond_h, "Existe indice vectorial\nde documentos?", (255, 236, 179), "diamond")
    positions["C"] = (main_x, y, box_w, diamond_h)
    y += diamond_h + gap

    # D (rama "No") y E (rama "Si") en la misma fila, luego D se une a E.
    # D se dibuja mas angosta y pegada al margen izquierdo para que no quede
    # fuera de la pagina.
    hgap = 6
    side_w = main_x - pdf.l_margin - hgap
    d_x = pdf.l_margin
    e_x = main_x
    flow_box(pdf, d_x, y, side_w, box_h, "Generar embeddings\nde documentos", (222, 235, 250), "rect")
    positions["D"] = (d_x, y, side_w, box_h)
    flow_box(pdf, e_x, y, box_w, box_h, "Calcular similitud coseno\nquery vs. documentos", (222, 235, 250), "rect")
    positions["E"] = (e_x, y, box_w, box_h)
    y += box_h + gap

    flow_box(pdf, main_x, y, box_w, box_h, "Ordenar documentos por\nscore descendente", (222, 235, 250), "rect")
    positions["F"] = (main_x, y, box_w, box_h)
    y += box_h + gap

    flow_box(pdf, main_x, y, box_w, box_h, "Seleccionar Top-K\ndocumentos mas similares", (222, 235, 250), "rect")
    positions["G"] = (main_x, y, box_w, box_h)
    y += box_h + gap

    flow_box(pdf, main_x, y, box_w, box_h, "Devolver el documento\nmas parecido al usuario", (210, 246, 219), "rounded")
    positions["H"] = (main_x, y, box_w, box_h)

    # ---- Flechas ----
    ax, ay, aw, ah = positions["A"]
    bx, by, bw, bh = positions["B"]
    cx, cy, cw, ch = positions["C"]
    dx, dy, dw, dh = positions["D"]
    ex, ey, ew, eh = positions["E"]
    fx, fy, fw, fh = positions["F"]
    gx, gy, gw, gh = positions["G"]
    hx, hy, hw, hh = positions["H"]

    arrow(pdf, ax + aw / 2, ay + ah, bx + bw / 2, by)
    arrow(pdf, bx + bw / 2, by + bh, cx + cw / 2, cy)
    # C -> D (No, hacia la izquierda) y C -> E (Si, recto hacia abajo)
    arrow(pdf, cx + cw / 2, cy + ch, dx + dw / 2, dy, "No")
    arrow(pdf, cx + cw / 2, cy + ch, ex + ew / 2, ey, "Si")
    # D -> E (se unen antes de calcular similitud)
    arrow(pdf, dx + dw, dy + dh / 2, ex, ey + eh / 2)
    arrow(pdf, ex + ew / 2, ey + eh, fx + fw / 2, fy)
    arrow(pdf, fx + fw / 2, fy + fh, gx + gw / 2, gy)
    arrow(pdf, gx + gw / 2, gy + gh, hx + hw / 2, hy)

    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_y(hy + hh + 10)

    # ---- Seccion 5: Conclusion ----
    section_title(pdf, "5. Conclusion")
    body_text(
        pdf,
        "El ejercicio evidencia que la similitud coseno sobre embeddings permite capturar "
        "relevancia semantica real: las oraciones 1-5, pese a no compartir vocabulario, "
        "deberian agruparse por significado, mientras que las oraciones trampa T1/T2, pese "
        "a compartir palabras, deberian quedar claramente separadas por un score bajo. "
        "Esta es la base conceptual de los sistemas de busqueda semantica (RAG, recuperacion "
        "de documentos, etc.), que superan las limitaciones de la busqueda lexica tradicional "
        "(keyword matching)."
    )

    pdf.output(OUTPUT_PATH)
    print(f"PDF generado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
