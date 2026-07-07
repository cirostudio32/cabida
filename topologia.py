"""
topologia.py — Detector + selector de estrategia de planta.

Analiza la forma del lote y recomienda la topología arquitectónica adecuada:
  - spine       : doble crujía con pasillo central (lotes alargados)
  - claustro    : patio central + dptos perimetrales (lotes anchos y profundos)
  - tower       : núcleo central + dptos esquineros (lotes cuadrados pequeños)
  - L_plan     : planta en L (lotes esquineros)
  - irregular   : fallback (lotes complejos)

Por ahora solo `spine` está implementada en main._generate_geometry.
El resto se expone como recomendación para iteraciones futuras.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon


def _aspect_ratio(lote: Polygon) -> Tuple[float, float, float]:
    """Devuelve (long_len, short_len, ratio) del rectángulo mínimo rotado."""
    mrr = lote.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    d01 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    d12 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    long_len, short_len = max(d01, d12), min(d01, d12)
    ratio = long_len / short_len if short_len > 0 else float("inf")
    return long_len, short_len, ratio


def _compactness(lote: Polygon) -> float:
    """Compacidad isoperimétrica: 4πA / P². 1.0 = círculo, <0.7 = irregular."""
    if lote.length <= 0:
        return 0.0
    return float(4.0 * math.pi * lote.area / (lote.length ** 2))


def _mrr_area_ratio(lote: Polygon) -> float:
    """Área del lote / área de su rectángulo mínimo. <0.85 ≈ esquinero/L."""
    mrr_area = lote.minimum_rotated_rectangle.area
    if mrr_area <= 0:
        return 0.0
    return float(lote.area / mrr_area)


def analizar_lote(lote: Polygon, frente_m: float) -> Dict[str, Any]:
    """Devuelve métricas de forma del lote."""
    long_len, short_len, ratio = _aspect_ratio(lote)
    compactness = _compactness(lote)
    mrr_ratio = _mrr_area_ratio(lote)
    area = float(lote.area)
    # Esquinero/L solo si el lote es NO convexo (tiene vértice reflex).
    # Un trapezoide convexo con mrr_ratio bajo NO es esquinero.
    es_convexo = len(_detect_reflex_corners(lote)) == 0
    return {
        "area_m2": round(area, 2),
        "long_len_m": round(long_len, 2),
        "short_len_m": round(short_len, 2),
        "aspect_ratio": round(ratio, 3),
        "frente_m": round(frente_m, 2),
        "compactness": round(compactness, 3),
        "mrr_area_ratio": round(mrr_ratio, 3),
        "es_esquinero": (mrr_ratio < 0.85) and (not es_convexo),
        "es_compacto": compactness >= 0.75,
    }


def seleccionar_topologia(metricas: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica heurística sobre métricas y devuelve recomendación."""
    ratio = metricas["aspect_ratio"]
    short = metricas["short_len_m"]
    frente = metricas["frente_m"]
    area = metricas["area_m2"]
    esquinero = metricas["es_esquinero"]
    compacto = metricas["es_compacto"]

    candidatos: List[Tuple[str, float, str]] = []  # (nombre, score, motivo)

    if esquinero and frente >= 12 and short >= 10:
        candidatos.append((
            "L_plan", 0.85,
            "lote esquinero (mrr_area_ratio < 0.85), frente y fondo suficientes"
        ))

    # Hall compacto: patrón real Lima moderna entre medianeras.
    # Núcleo lateral + hall compacto; dptos frente/fondo a todo el ancho.
    if (not esquinero) and 7 <= short <= 45 and area <= 2200:
        candidatos.append((
            "hall_compacto", 0.90,
            "lote rectangular entre medianeras — núcleo lateral + hall compacto "
            "(patrón multifamiliar Lima moderna)"
        ))

    # Spine: doble crujía con pasillo central — fallback para lotes anchos
    # o cuando hall_compacto no logra generar.
    if short >= 8:
        candidatos.append((
            "spine", 0.80,
            "doble crujía con pasillo central — topología estándar multifamiliar"
        ))

    # Claustro: solo para lotes muy grandes y muchas unidades (≥14m×14m, ≥1200 m²).
    # En práctica Lima moderna no se usa para multifamiliar de mediana escala.
    if compacto and short >= 22 and frente >= 22 and area >= 1200:
        candidatos.append((
            "claustro", 0.65,
            "lote compacto muy grande (≥22×22m, ≥1200 m²) admite patio central"
        ))

    if ratio < 1.6 and short < 14 and area < 300:
        candidatos.append((
            "tower", 0.70,
            "lote cuadrado pequeño, núcleo central + dptos perimetrales"
        ))

    if short < 8:
        candidatos.append((
            "irregular", 0.40,
            "lote muy angosto (<8m); recomendación: revisar viabilidad"
        ))

    if not candidatos:
        candidatos.append((
            "spine", 0.55,
            "fallback genérico — doble crujía"
        ))

    candidatos.sort(key=lambda c: -c[1])
    nombre, score, motivo = candidatos[0]

    implementadas = {"spine", "claustro", "tower", "hall_compacto"}
    return {
        "recomendada": nombre,
        "confianza": score,
        "motivo": motivo,
        "implementada_actualmente": nombre in implementadas,
        "estrategia_usada": nombre if nombre in implementadas else "spine",
        "alternativas": [
            {"nombre": n, "score": s, "motivo": m}
            for n, s, m in candidatos[1:]
        ],
    }


def informe_topologia(lote: Polygon, frente_m: float) -> Dict[str, Any]:
    """Combina métricas + recomendación en un solo bloque."""
    metricas = analizar_lote(lote, frente_m)
    seleccion = seleccionar_topologia(metricas)
    return {"metricas_lote": metricas, "seleccion": seleccion}


def _detect_reflex_corners(poly: Polygon) -> List[int]:
    """Índices de vértices reflex (ángulo interior > 180°)."""
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:
        return []
    is_ccw = poly.exterior.is_ccw
    reflex: List[int] = []
    for i in range(n):
        prev = coords[(i - 1) % n]
        curr = coords[i]
        nxt = coords[(i + 1) % n]
        cross = (curr[0] - prev[0]) * (nxt[1] - curr[1]) - (curr[1] - prev[1]) * (nxt[0] - curr[0])
        if is_ccw and cross < 0:
            reflex.append(i)
        elif (not is_ccw) and cross > 0:
            reflex.append(i)
    return reflex


def find_main_rect(lote: Polygon) -> Optional[Polygon]:
    """Para lotes L-shape (1 reflex corner), devuelve el sub-rectángulo
    axis-aligned mayor inscrito (mitad del bbox cortada por el reflex).
    Para lotes rectangulares o complejos, devuelve None.
    """
    if not lote.is_valid:
        try:
            lote = lote.buffer(0)
        except Exception:
            return None
    reflex_idxs = _detect_reflex_corners(lote)
    if len(reflex_idxs) != 1:
        return None
    coords = list(lote.exterior.coords)[:-1]
    rx, ry = coords[reflex_idxs[0]]
    minx, miny, maxx, maxy = lote.bounds
    candidates = [
        Polygon([(minx, miny), (rx, miny), (rx, maxy), (minx, maxy)]),     # left
        Polygon([(rx, miny), (maxx, miny), (maxx, maxy), (rx, maxy)]),     # right
        Polygon([(minx, miny), (maxx, miny), (maxx, ry), (minx, ry)]),     # bottom
        Polygon([(minx, ry), (maxx, ry), (maxx, maxy), (minx, maxy)]),     # top
    ]
    # Filtrar los que están casi contenidos en el lote (tolerancia)
    valid = []
    for r in candidates:
        if r.area < 1.0:
            continue
        try:
            outside = r.difference(lote).area
        except Exception:
            continue
        if outside < 0.5:  # casi totalmente dentro
            valid.append(r)
    if not valid:
        return None
    return max(valid, key=lambda r: r.area)
