"""
validators.py — Validadores arquitectónicos por unidad y por zona.

Chequeos:
  - Ventilación natural: zona principal (day/night/study) debe tocar
    perímetro libre (fachada o patio) con longitud ≥ MIN_VENT_LEN.
  - Baño / Lavandería: aceptan ventilación por ducto si lo intersectan.
  - Iluminación: perímetro libre / área ≥ LUX_RATIO_MIN (proxy).
  - Evacuación: distancia desde unidad a escalera ≤ EVAC_MAX_M.
"""

from typing import Any, Dict, Iterable, List, Optional

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


MIN_VENT_LEN = 0.60       # m de borde libre mínimo para ventilar/iluminar
LUX_RATIO_MIN = 0.04      # perímetro libre / área de zona (proxy iluminación)
EVAC_MAX_M = 30.0         # RNE A.010: distancia máx evacuación sin rociadores (45m con rociadores)
DUCTO_BUFFER = 0.10       # m de tolerancia para considerar contacto con ducto
POZO_BUFFER  = 0.12       # m de tolerancia para considerar contacto con pozo de luz


def _is_wet_service(nombre: str) -> bool:
    n = (nombre or "").upper()
    return ("BAÑO" in n) or ("BANO" in n) or ("LAVANDER" in n) or (n == "WC")


def facade_boundary(unit: Polygon, hall_buf: Optional[BaseGeometry]) -> BaseGeometry:
    """Frontera de la unidad que NO toca el hall (proxy de fachada/patio)."""
    border = unit.boundary
    if hall_buf is None or hall_buf.is_empty:
        return border
    try:
        return border.difference(hall_buf)
    except Exception:
        return border


def _intersection_length(zona: Polygon, line: BaseGeometry) -> float:
    if line is None or line.is_empty:
        return 0.0
    try:
        inter = zona.boundary.intersection(line)
    except Exception:
        return 0.0
    if inter.is_empty:
        return 0.0
    return float(getattr(inter, "length", 0.0))


def _touches_any_ducto(zona: Polygon, ductos: Iterable[Polygon]) -> bool:
    for d in ductos:
        try:
            if zona.intersects(d.buffer(DUCTO_BUFFER)):
                return True
        except Exception:
            continue
    return False


def _touches_any_pozo(zona: Polygon, pozos: Iterable[Polygon]) -> bool:
    for p in pozos:
        try:
            if zona.intersects(p.buffer(POZO_BUFFER)):
                return True
        except Exception:
            continue
    return False


def validar_zona(
    zona_geom: Polygon,
    nombre: str,
    kind: str,
    facade_line: BaseGeometry,
    ductos: Iterable[Polygon],
    pozos: Iterable[Polygon] = (),
) -> Dict[str, Any]:
    """Devuelve dict de validación para una habitación."""
    vent_len = _intersection_length(zona_geom, facade_line)
    needs_ducto = _is_wet_service(nombre)
    vent_ducto = _touches_any_ducto(zona_geom, ductos) if needs_ducto else False
    vent_pozo  = _touches_any_pozo(zona_geom, pozos) if kind == "night" else False

    if kind == "circ":
        ventila_ok = True
        ilumina_ok = True
    elif needs_ducto:
        ventila_ok = vent_len >= MIN_VENT_LEN or vent_ducto
        ilumina_ok = True
    elif kind == "night":
        ventila_ok = vent_len >= MIN_VENT_LEN or vent_pozo
        area = max(zona_geom.area, 0.01)
        ilumina_ok = (vent_len / area) >= LUX_RATIO_MIN or vent_pozo
    else:
        ventila_ok = vent_len >= MIN_VENT_LEN
        area = max(zona_geom.area, 0.01)
        ilumina_ok = (vent_len / area) >= LUX_RATIO_MIN

    return {
        "ventila": bool(ventila_ok),
        "ilumina": bool(ilumina_ok),
        "vent_len_m": round(vent_len, 3),
        "vent_por_ducto": bool(vent_ducto),
        "vent_por_pozo": bool(vent_pozo),
    }


def distancia_a_escalera(unit: Polygon, escalera: Optional[BaseGeometry]) -> float:
    """Distancia desde el centroide de la unidad al polígono de la escalera
    (aproximación euclidiana al punto de ingreso más cercano)."""
    if escalera is None or escalera.is_empty:
        return -1.0
    try:
        return float(unit.centroid.distance(escalera))
    except Exception:
        return -1.0


def validar_unidad(
    unit: Polygon,
    zonas: List[Dict[str, Any]],
    escalera: Optional[BaseGeometry],
    hall_buf: Optional[BaseGeometry],
    ductos: List[Polygon],
    pozos: List[Polygon] = (),
) -> Dict[str, Any]:
    """Anota cada zona con `validacion` y devuelve resumen por unidad."""
    facade_line = facade_boundary(unit, hall_buf)
    fallas: List[str] = []

    for z in zonas:
        v = validar_zona(z["geom"], z["nombre"], z.get("kind", ""), facade_line, ductos, pozos)
        z["validacion"] = v
        if not v["ventila"] and z.get("kind") in ("day", "night", "study"):
            fallas.append(f"{z['nombre']}: sin ventilación natural")
        if not v["ilumina"] and z.get("kind") in ("day", "night", "study"):
            fallas.append(f"{z['nombre']}: bajo ratio iluminación")

    evac = distancia_a_escalera(unit, escalera)
    evac_ok = (evac >= 0.0) and (evac <= EVAC_MAX_M)
    if not evac_ok and evac >= 0.0:
        fallas.append(f"evacuación {evac:.1f}m > {EVAC_MAX_M:.0f}m")

    ventila_principales = all(
        z["validacion"]["ventila"]
        for z in zonas
        if z.get("kind") in ("day", "night", "study")
    )

    return {
        "ventila_principales": ventila_principales,
        "distancia_evac_m": round(evac, 2),
        "evac_cumple": bool(evac_ok),
        "fallas": fallas,
    }
