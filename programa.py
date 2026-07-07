"""
programa.py — Distribución interior de departamentos por tipología.

Modelo:
- Cada tipología define una secuencia de "bandas" desde el pasillo (t≈0)
  hacia la fachada / perímetro libre (t≈1).
- Cada banda tiene una profundidad relativa (depth_frac) y una lista de
  habitaciones con áreas objetivo.
- Las habitaciones de una banda se distribuyen en ancho (u∈[0,1])
  proporcional a su área objetivo (treemap 1D dentro de la banda).

Reglas arquitectónicas codificadas:
- Banda "circ" (entrada): muy delgada, junto al pasillo.
- Banda "wet" (cocina, baño, lavandería): cerca del pasillo / ducto de
  servicio. No requiere fachada (ventilación por ducto).
- Banda "day" (estar-comedor): intermedia, ventila parcialmente.
- Banda "study" (escritorio opcional): después de estar, antes de dorms.
- Banda "night" (dormitorios): en el perímetro libre, exige fachada
  o patio para ventilación e iluminación natural.

La profundidad efectiva de cada banda se cierra al final (t→1.0).
"""

import math
from typing import Tuple, Optional, Dict, Any, List

from shapely.geometry import Polygon


# Área mínima por habitación para emitir el polígono (filtro de ruido geométrico)
MIN_ROOM_AREA = 1.8


# Tabla única de áreas por tipología (m²) — fuente de verdad para
# clasificación (get_typology), optimizador de mix y distribución dirigida.
#   min    : área mínima viable de la tipología (= PROGRAMA area_min)
#   target : área objetivo de mercado usada para dimensionar anchos en mix dirigido
AREAS_TIPOLOGIA: Dict[str, Dict[str, float]] = {
    "1D":   {"min": 40.0, "target": 45.0},
    "1D+E": {"min": 50.0, "target": 55.0},
    "2D":   {"min": 60.0, "target": 68.0},
    "2D+E": {"min": 75.0, "target": 80.0},
    "3D":   {"min": 92.0, "target": 95.0},
}


PROGRAMA: Dict[str, Dict[str, Any]] = {
    "1D": {
        "area_min": 40.0,
        "bandas": [
            {"depth_frac": 0.10, "kind": "circ",
             "rooms": [{"nombre": "Circulación", "area": 3}]},
            {"depth_frac": 0.25, "kind": "wet",
             "rooms": [{"nombre": "Cocina", "area": 7},
                       {"nombre": "Baño", "area": 4}]},
            {"depth_frac": 0.30, "kind": "day",
             "rooms": [{"nombre": "Estar-Comedor", "area": 14}]},
            {"depth_frac": 0.35, "kind": "night",
             "rooms": [{"nombre": "Dormitorio", "area": 12}]},
        ],
    },
    "1D+E": {
        "area_min": 50.0,
        "bandas": [
            {"depth_frac": 0.09, "kind": "circ",
             "rooms": [{"nombre": "Circulación", "area": 3}]},
            {"depth_frac": 0.22, "kind": "wet",
             "rooms": [{"nombre": "Cocina", "area": 7},
                       {"nombre": "Baño", "area": 4}]},
            {"depth_frac": 0.27, "kind": "day",
             "rooms": [{"nombre": "Estar-Comedor", "area": 14}]},
            {"depth_frac": 0.15, "kind": "study",
             "rooms": [{"nombre": "Escritorio", "area": 5}]},
            {"depth_frac": 0.27, "kind": "night",
             "rooms": [{"nombre": "Dormitorio", "area": 12}]},
        ],
    },
    "2D": {
        "area_min": 60.0,
        "bandas": [
            {"depth_frac": 0.09, "kind": "circ",
             "rooms": [{"nombre": "Circulación", "area": 3}]},
            {"depth_frac": 0.22, "kind": "wet",
             "rooms": [{"nombre": "Cocina", "area": 7},
                       {"nombre": "Baño", "area": 4},
                       {"nombre": "Lavandería", "area": 3}]},
            {"depth_frac": 0.30, "kind": "day",
             "rooms": [{"nombre": "Estar-Comedor", "area": 18}]},
            {"depth_frac": 0.39, "kind": "night",
             "rooms": [{"nombre": "Dormitorio principal", "area": 12},
                       {"nombre": "Dormitorio secundario", "area": 10}]},
        ],
    },
    "2D+E": {
        "area_min": 75.0,
        "bandas": [
            {"depth_frac": 0.08, "kind": "circ",
             "rooms": [{"nombre": "Circulación", "area": 3}]},
            {"depth_frac": 0.22, "kind": "wet",
             "rooms": [{"nombre": "Cocina", "area": 8},
                       {"nombre": "Baño visita", "area": 3},
                       {"nombre": "Lavandería", "area": 3}]},
            {"depth_frac": 0.28, "kind": "day",
             "rooms": [{"nombre": "Estar-Comedor", "area": 20}]},
            {"depth_frac": 0.14, "kind": "study",
             "rooms": [{"nombre": "Escritorio", "area": 6}]},
            {"depth_frac": 0.28, "kind": "night",
             "rooms": [{"nombre": "Dormitorio principal", "area": 14},
                       {"nombre": "Dormitorio secundario", "area": 10}]},
        ],
    },
    "3D": {
        "area_min": 92.0,
        "bandas": [
            {"depth_frac": 0.07, "kind": "circ",
             "rooms": [{"nombre": "Circulación", "area": 4}]},
            {"depth_frac": 0.23, "kind": "wet",
             "rooms": [{"nombre": "Cocina", "area": 9},
                       {"nombre": "Baño visita", "area": 3},
                       {"nombre": "Baño compartido", "area": 4},
                       {"nombre": "Lavandería", "area": 4}]},
            {"depth_frac": 0.30, "kind": "day",
             "rooms": [{"nombre": "Estar-Comedor", "area": 24}]},
            {"depth_frac": 0.40, "kind": "night",
             "rooms": [{"nombre": "Dormitorio principal", "area": 14},
                       {"nombre": "Dormitorio 2", "area": 11},
                       {"nombre": "Dormitorio 3", "area": 10}]},
        ],
    },
}


def _safe_clip_to_lot(poly: Polygon, lot: Polygon) -> Optional[Polygon]:
    try:
        r = poly.intersection(lot)
    except Exception:
        return None
    if r.is_empty:
        return None
    if r.geom_type == "MultiPolygon":
        r = max(r.geoms, key=lambda g: g.area)
    if r.geom_type == "GeometryCollection":
        ps = [g for g in r.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not ps:
            return None
        r = max(ps, key=lambda g: g.area)
        if r.geom_type == "MultiPolygon":
            r = max(r.geoms, key=lambda g: g.area)
    if r.geom_type != "Polygon":
        return None
    return r


def strip_cell_polygon(
    corners: Tuple[Tuple[float, float], ...],
    u0: float,
    u1: float,
    t0: float,
    t1: float,
) -> Polygon:
    """Sub-celda bilineal sobre el strip hall→fachada.

    Convención corners = (p0, p1, p2, p3):
      p0, p1  borde interior (pasillo)
      p2, p3  borde exterior (fachada/perímetro libre)
    u parametriza el ancho (a lo largo del pasillo), t la profundidad.
    """
    p0, p1, p2, p3 = corners

    def pt(u: float, t: float) -> Tuple[float, float]:
        ix = (1 - u) * p0[0] + u * p1[0]
        iy = (1 - u) * p0[1] + u * p1[1]
        ox = (1 - u) * p3[0] + u * p2[0]
        oy = (1 - u) * p3[1] + u * p2[1]
        return ((1 - t) * ix + t * ox, (1 - t) * iy + t * oy)

    return Polygon([pt(u0, t0), pt(u1, t0), pt(u1, t1), pt(u0, t1)])


def clip_zone_polygon(
    quad: Polygon, unit: Polygon, lot: Polygon
) -> Optional[Polygon]:
    """Recorta una sub-celda contra la unidad y el lote real."""
    z = _safe_clip_to_lot(quad, lot)
    if z is None:
        return None
    try:
        z = z.intersection(unit)
    except Exception:
        return None
    if z.is_empty:
        return None
    if z.geom_type == "MultiPolygon":
        z = max(z.geoms, key=lambda g: g.area)
    if z.geom_type == "GeometryCollection":
        polys = [g for g in z.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        z = max(polys, key=lambda g: g.area)
        if z.geom_type == "MultiPolygon":
            z = max(z.geoms, key=lambda g: g.area)
    if z.geom_type != "Polygon":
        return None
    if z.area < MIN_ROOM_AREA:
        return None
    return z


def generate_interior_zones(
    corners: Tuple[Tuple[float, float], ...],
    typology: str,
    unit_poly: Polygon,
    lot: Polygon,
) -> List[Dict[str, Any]]:
    """Genera habitaciones por banda + treemap 1D por área dentro de cada banda.

    Cada zona devuelta:
        {"nombre": str, "geom": Polygon, "kind": str}
    """
    spec = PROGRAMA.get(typology, PROGRAMA["2D"])
    out: List[Dict[str, Any]] = []
    t_prev = 0.0

    for banda in spec["bandas"]:
        depth_frac = float(banda["depth_frac"])
        t_end = min(1.0, t_prev + depth_frac)
        if t_end - t_prev < 1e-3:
            continue
        kind = banda["kind"]
        rooms = banda["rooms"]
        total_area = sum(r["area"] for r in rooms) or 1.0

        u_prev = 0.0
        for i, room in enumerate(rooms):
            w_frac = room["area"] / total_area
            u_end = 1.0 if i == len(rooms) - 1 else min(1.0, u_prev + w_frac)
            quad = strip_cell_polygon(corners, u_prev, u_end, t_prev, t_end)
            geom = clip_zone_polygon(quad, unit_poly, lot)
            if geom is not None:
                out.append({
                    "nombre": room["nombre"],
                    "geom": geom,
                    "kind": kind,
                })
            u_prev = u_end
        t_prev = t_end

    return out
