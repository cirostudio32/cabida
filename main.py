# -*- coding: utf-8 -*-
"""
main.py — Motor de auditoría RNE + renderizado arquitectónico en Python.
FastAPI backend que genera geometría + empaqueta un objeto JSON normalizado
listo para consumo por motores WebGL (Three.js) o cualquier otro cliente.
"""

import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon
from shapely.ops import unary_union

def calc_poly_area(poly):
    """Calculate area with Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i]["x"] * poly[j]["y"]
        area -= poly[j]["x"] * poly[i]["y"]
    return abs(area) / 2


def get_typology(area: float) -> str:
    """Tipología multifamiliar urbano por área útil.
    Umbrales = área mínima de la siguiente tipología (AREAS_TIPOLOGIA)."""
    if area < AREAS_TIPOLOGIA["1D+E"]["min"]:
        return "1D"
    if area < AREAS_TIPOLOGIA["2D"]["min"]:
        return "1D+E"
    if area < AREAS_TIPOLOGIA["2D+E"]["min"]:
        return "2D"
    if area < AREAS_TIPOLOGIA["3D"]["min"]:
        return "2D+E"
    return "3D"


def dotacion_categoria_tipologia(typology: str) -> str:
    """Bucket IS.010 agua: 1D / 2D / 3D a partir de tipología extendida."""
    if typology == "1D":
        return "1D"
    if typology in ("1D+E", "2D"):
        return "2D"
    return "3D"


from zonificacion import validar_zonificacion, get_zona, HAB_PROMEDIO_POR_DEPTO
from programa import generate_interior_zones, AREAS_TIPOLOGIA
from validators import validar_unidad, EVAC_MAX_M
from topologia import informe_topologia, find_main_rect, analizar_lote, seleccionar_topologia

app = FastAPI()
# CORS abierto: el frontend Netlify llama a este backend (Render) cross-origin.
# API read-only sin datos sensibles. Si se restringe, añadir el dominio Netlify.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serve static files via explicit routes (API routes defined below will take priority)
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/styles.css")
async def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "styles.css"))

@app.get("/main.js")
async def serve_main_js():
    return FileResponse(os.path.join(BASE_DIR, "main.js"))

@app.get("/viewer3d.js")
async def serve_viewer3d_js():
    path = os.path.join(BASE_DIR, "viewer3d.js")
    print(f"Serving viewer3d.js from: {path}")
    return FileResponse(path)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN MAESTRA RNE (Reglamento Nacional de Edificaciones)
# ═══════════════════════════════════════════════════════════════
RNE = {
    "departamentos": {"min_multifamiliar": 40.0, "min_unipersonal": 16.0, "h_libre": 2.30},
    # RNE A.010: pozos que ventilan dormitorios exigen lado ≥ H/3 (no H/4 —
    # esa fracción es solo para pozos que sirven ambientes de servicio).
    "pozos_luz": {"min_abs": 2.20, "ratio_dorm": 1 / 3, "ratio_serv": 0.25},
    "circulacion_h": {"hall_ancho": 1.20, "interior": 0.90},
    "circulacion_v": {
        "esc_ancho": 1.20, "esc_largo": 5.60,
        # Criterio único de evacuación: validators.EVAC_MAX_M (30m sin rociadores)
        "evacuacion_max": EVAC_MAX_M,
        "h_max_sin_esc_prot": 15.0, "h_max_sin_ascensor": 12.0,
    },
    "ascensor": {"ancho": 2.00, "largo": 2.00},
    "estacionamientos": {"ancho_ind": 2.70, "largo": 5.00, "maniobra": 6.00},
    "instalaciones": {"aci_m3": 25.0, "agua_1d": 500.0, "agua_2d": 850.0, "agua_3d": 1200.0},
    "altura_piso": 2.80,
}

# ═══════════════════════════════════════════════════════════════
# NUEVO CRÉDITO MIVIVIENDA (Fondo Mivivienda) — topes referenciales.
# Valores en soles, indexados a UIT/decreto y actualizados ~anualmente por
# el Fondo Mivivienda. VERIFICAR VIGENCIA antes de comercializar: estos son
# valores de referencia (2024-2025), no una fuente normativa consultada en
# vivo por este motor.
# ═══════════════════════════════════════════════════════════════
MIVIVIENDA_TOPES_SOLES = {
    "rango1": 91_700.0,
    "rango2": 141_000.0,
    "rango3": 232_000.0,
    "rango4": 415_300.0,
}
MIVIVIENDA_AREA_MIN_M2 = 40.0  # Reglamento Operativo Mivivienda + RNE A.010


def _check_mivivienda(proyecto: "ProyectoInmobiliario", unidades_meta: list) -> Optional[dict]:
    """Checklist Mivivienda: área mínima + rango de precio por unidad
    (requiere proyecto.precios_tipologia en soles/m² para el precio;
    sin esos datos solo evalúa área mínima)."""
    if not getattr(proyecto, "acogido_mivivienda", False):
        return None
    precios = proyecto.precios_tipologia or {}
    detalle = []
    n_bajo_area = n_sin_precio = n_sobre_tope = 0
    for u in unidades_meta:
        area = u.get("area", 0.0)
        tip = u.get("tipologia", "")
        cumple_area = area >= MIVIVIENDA_AREA_MIN_M2
        if not cumple_area:
            n_bajo_area += 1
        precio_m2 = precios.get(tip)
        precio_total = rango = cumple_precio = None
        if precio_m2:
            precio_total = round(area * precio_m2, 2)
            for r, tope in MIVIVIENDA_TOPES_SOLES.items():
                if precio_total <= tope:
                    rango = r
                    break
            cumple_precio = rango is not None
            if not cumple_precio:
                n_sobre_tope += 1
        else:
            n_sin_precio += 1
        detalle.append({
            "tipologia": tip, "area": area,
            "cumple_area_min": cumple_area,
            "precio_total_estimado_soles": precio_total,
            "rango_mivivienda": rango,
            "cumple_precio": cumple_precio,
        })
    return {
        "aplica": True,
        "area_min_m2": MIVIVIENDA_AREA_MIN_M2,
        "topes_referenciales_soles": MIVIVIENDA_TOPES_SOLES,
        "nota": "Topes de precio referenciales — verificar decreto/UIT vigente del Fondo Mivivienda antes de comercializar.",
        "detalle_unidades": detalle,
        "n_unidades": len(detalle),
        "n_bajo_area_min": n_bajo_area,
        "n_sin_precio_definido": n_sin_precio,
        "n_sobre_tope_precio": n_sobre_tope,
        "cumple_global": n_bajo_area == 0 and n_sobre_tope == 0,
    }


class ProyectoInmobiliario(BaseModel):
    coordenadas_lote: List[Tuple[float, float]]
    area_bruta_terreno: float
    numero_pisos: int
    retiro_frontal: float
    zonificacion: str
    num_ascensores: int
    num_departamentos: int
    frente: Optional[float] = 10.0
    fondo: Optional[float] = 10.0
    derecha: Optional[float] = 20.0
    izquierda: Optional[float] = 20.0
    altura_piso: Optional[float] = 2.80
    pct_estac: Optional[float] = 30.0
    ciego_frente: Optional[bool] = False
    ciego_fondo: Optional[bool] = True
    ciego_derecha: Optional[bool] = True
    ciego_izquierda: Optional[bool] = True
    retiro_lateral: Optional[float] = 2.30
    retiro_posterior: Optional[float] = 2.30
    # Esquema de área libre: "muros_ciegos" | "patio_posterior" | "ducto_central"
    esquema_area_libre: Optional[str] = "muros_ciegos"
    # Si True, ignora num_departamentos y emite capacidad máxima viable.
    optimizar_densidad: Optional[bool] = False
    # Precios de venta por tipología (PEN/m²). Si se proveen, activa optimizador de mix.
    precios_tipologia: Optional[Dict[str, float]] = None
    frente_exterior: Optional[bool] = True       # True si el frente da a calle/espacio público
    fondo_exterior: Optional[bool] = False      # True si el fondo da a calle/espacio público
    derecha_exterior: Optional[bool] = False    # True si lado derecho da a exterior
    izquierda_exterior: Optional[bool] = False  # True si lado izquierdo da a exterior
    mix_tipologias: Optional[Dict[str, int]] = None  # Ej: {"1D": 1, "2D": 3, "3D": 2}
    area_libre_min_pct: Optional[float] = 0.0       # % mínimo de área libre del lote (ej: 20.0)
    # Overrides del certificado de parámetros municipal (None = usar tabla de zona)
    cus_maximo: Optional[float] = None
    altura_maxima_pisos: Optional[int] = None
    densidad_maxima_hab_ha: Optional[float] = None
    # True = reducir pisos automáticamente hasta cumplir altura/CUS/densidad.
    # False (default) = respetar los pisos solicitados y reportar incumplimientos.
    ajustar_pisos_normativa: Optional[bool] = False
    # Proyecto acogido al Nuevo Crédito Mivivienda (Fondo Mivivienda) →
    # activa mivivienda_check en la respuesta (área mín. y rango de precio
    # por unidad, usando precios_tipologia si se proveyó).
    acogido_mivivienda: Optional[bool] = False


# ═══════════════════════════════════════════════════════════════
# OPTIMIZADOR DE MIX TIPOLÓGICO
# ═══════════════════════════════════════════════════════════════

# Área mínima de referencia por tipología (m²) — fuente: AREAS_TIPOLOGIA
_AREA_TIPO: Dict[str, float] = {k: v["min"] for k, v in AREAS_TIPOLOGIA.items()}

def _optimizar_mix(
    strip_area_planta: float,
    depth_strip: float,
    precios: Dict[str, float],
    pisos: int,
) -> Dict[str, Any]:
    """Maximiza ingreso bruto dado área de strip disponible por planta.

    Enumera todas las combinaciones de 1 y 2 tipologías con conteos enteros.
    Devuelve mix óptimo, unidades por piso/edificio e ingreso bruto estimado.
    """
    if not precios or strip_area_planta <= 0 or depth_strip <= 0:
        return {}

    # Filtrar tipos con precio y área conocidos
    tipos = [
        (k, _AREA_TIPO[k], float(v))
        for k, v in precios.items()
        if k in _AREA_TIPO and v > 0
    ]
    if not tipos:
        return {}

    best: Dict[str, Any] = {"ingreso": -1.0}

    def evaluar(mix: Dict[str, int]) -> None:
        nonlocal best
        area_total = sum(cnt * _AREA_TIPO[t] for t, cnt in mix.items() if t in _AREA_TIPO)
        if area_total > strip_area_planta * 1.02:   # 2% holgura por redondeo
            return
        ingreso = sum(
            cnt * _AREA_TIPO[t] * precios[t]
            for t, cnt in mix.items()
            if t in _AREA_TIPO and t in precios
        ) * pisos
        n_total = sum(mix.values())
        if ingreso > best["ingreso"] and n_total > 0:
            best = {
                "mix": dict(mix),
                "ingreso": ingreso,
                "area_vendible_planta": r3(area_total),
                "unidades_planta": n_total,
            }

    # ── Combinaciones de 1 tipo ──
    for nombre, area, precio in tipos:
        n_max = min(int(strip_area_planta / area), 30)
        for n in range(1, n_max + 1):
            evaluar({nombre: n})

    # ── Combinaciones de 2 tipos ──
    for i, (n1, a1, _) in enumerate(tipos):
        for n2, a2, _ in tipos[i + 1:]:
            n1_max = min(int(strip_area_planta / a1), 20)
            for cnt1 in range(0, n1_max + 1):
                area_left = strip_area_planta - cnt1 * a1
                if area_left < 0:
                    break
                cnt2 = min(int(area_left / a2), 20)
                for c2 in range(0, cnt2 + 1):
                    m: Dict[str, int] = {}
                    if cnt1 > 0:
                        m[n1] = cnt1
                    if c2 > 0:
                        m[n2] = c2
                    if m:
                        evaluar(m)

    if best["ingreso"] < 0:
        return {}

    # Breakdown por tipología
    mix = best["mix"]
    breakdown = []
    for tipo, cnt in sorted(mix.items(), key=lambda x: -x[1]):
        area_t = _AREA_TIPO.get(tipo, 0)
        precio_t = precios.get(tipo, 0)
        breakdown.append({
            "tipologia": tipo,
            "unidades_planta": cnt,
            "unidades_edificio": cnt * pisos,
            "area_vendible_planta_m2": r3(cnt * area_t),
            "area_vendible_edificio_m2": r3(cnt * area_t * pisos),
            "ingreso_tipologia": r3(cnt * area_t * precio_t * pisos),
            "precio_m2": r3(precio_t),
        })

    return {
        "mix_recomendado": mix,
        "breakdown": breakdown,
        "unidades_totales_planta": best["unidades_planta"],
        "unidades_totales_edificio": best["unidades_planta"] * pisos,
        "area_vendible_planta_m2": best["area_vendible_planta"],
        "area_vendible_edificio_m2": r3(best["area_vendible_planta"] * pisos),
        "ingreso_bruto_estimado": r3(best["ingreso"]),
        "precios_usados": {k: v for k, v in precios.items() if k in _AREA_TIPO},
        "strip_area_disponible_planta_m2": r3(strip_area_planta),
        "nota": "Estimación basada en áreas mínimas RNE por tipología. Precios en PEN/m².",
    }


# ═══════════════════════════════════════════════════════════════
# HELPERS DE GEOMETRÍA
# ═══════════════════════════════════════════════════════════════

def r3(v: float) -> float:
    """Round to 3 decimals."""
    return round(v, 3)


MURO_T_NOMINAL = 0.15  # espesor nominal de muro (m) — descuento eje a eje


def area_neta_muros(ap, muro_t: float = MURO_T_NOMINAL) -> float:
    """Área neta interior: descuenta espesor de muro en ambos ejes del bbox.
    Misma fórmula para todas las topologías (spine/claustro/tower)."""
    bounds = ap.bounds
    bw = max(0.01, bounds[2] - bounds[0])
    bh = max(0.01, bounds[3] - bounds[1])
    return float(ap.area) * max(0.0, 1 - muro_t / bw) * max(0.0, 1 - muro_t / bh)


def poly_to_js(sp) -> list:
    """Shapely Polygon → [{x,y}, …] for JS."""
    if sp is None or sp.is_empty:
        return []
    if sp.geom_type == "MultiPolygon":
        sp = max(sp.geoms, key=lambda g: g.area)
    if not sp.is_valid:
        sp = sp.buffer(0)
        if hasattr(sp, 'geoms'):
            sp = max(sp.geoms, key=lambda g: g.area)
        if sp is None or sp.is_empty:
            return []
    pts = [{"x": r3(x), "y": r3(y)} for x, y in list(sp.exterior.coords)[:-1]]
    # Validate after rounding: rounding can re-introduce self-intersections.
    if len(pts) >= 3:
        _rp = Polygon([(p["x"], p["y"]) for p in pts])
        if not _rp.is_valid:
            _rp = _rp.buffer(0)
            if hasattr(_rp, 'geoms'):
                _rp = max(_rp.geoms, key=lambda g: g.area)
            if not _rp.is_empty:
                pts = [{"x": r3(x), "y": r3(y)} for x, y in list(_rp.exterior.coords)[:-1]]
    return pts


def safe_clip(poly, boundary):
    def _pick_poly(r):
        if r.is_empty:
            return None
        if r.geom_type == "GeometryCollection":
            ps = [g for g in r.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not ps:
                return None
            r = max(ps, key=lambda g: g.area)
        return r

    try:
        return _pick_poly(poly.intersection(boundary))
    except Exception:
        pass
    # Fallback: limpiar geometrías con buffer(0) y reintentar
    try:
        return _pick_poly(poly.buffer(0).intersection(boundary.buffer(0)))
    except Exception:
        return None  # No devolver el polígono original — podría estar fuera del lote


def make_rect(cx, cy, dx_l, dy_l, dx_s, dy_s, half_l, half_s):
    return Polygon([
        (cx - dx_l * half_l - dx_s * half_s, cy - dy_l * half_l - dy_s * half_s),
        (cx + dx_l * half_l - dx_s * half_s, cy + dy_l * half_l - dy_s * half_s),
        (cx + dx_l * half_l + dx_s * half_s, cy + dy_l * half_l + dy_s * half_s),
        (cx - dx_l * half_l + dx_s * half_s, cy - dy_l * half_l + dy_s * half_s),
    ])


def _interpolate(pA, pB, t):
    return {"x": pA["x"] + (pB["x"] - pA["x"]) * t, "y": pA["y"] + (pB["y"] - pA["y"]) * t}


def _get_cell(quad, u1, u2, v1, v2):
    def _gp(u, v):
        top = _interpolate(quad[0], quad[1], u)
        bot = _interpolate(quad[3], quad[2], u)
        return _interpolate(top, bot, v)
    return [_gp(u1, v1), _gp(u2, v1), _gp(u2, v2), _gp(u1, v2)]


# Alias: misma fórmula shoelace que calc_poly_area (unificado).
_calculate_poly_area = calc_poly_area


def _rect_min_side(shapely_poly) -> float:
    """Lado más corto del rectángulo envolvente mínimo (proxy de frente real)."""
    try:
        mrr = shapely_poly.minimum_rotated_rectangle
        c = list(mrr.exterior.coords)
        s0 = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
        s1 = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
        return min(s0, s1)
    except Exception:
        return 0.0


def _poly_width(poly):
    if len(poly) < 4:
        return 0
    dx = poly[1]["x"] - poly[0]["x"]
    dy = poly[1]["y"] - poly[0]["y"]
    return math.hypot(dx, dy)


def _centroid(pts: list) -> list:
    """Return centroid [x, y] of a [{x,y},...] or [[x,y],...] polygon."""
    if not pts:
        return [0, 0]
    if isinstance(pts[0], dict):
        cx = sum(p["x"] for p in pts) / len(pts)
        cy = sum(p["y"] for p in pts) / len(pts)
    else:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
    return [r3(cx), r3(cy)]


def _strip_segments_for_apartments(
    L_min: float, L_max: float, exclude_min: float, exclude_max: float
) -> List[Tuple[float, float]]:
    """
    Tramos libres a lo largo del eje largo del rectángulo mínimo rotado,
    excluyendo el hueco del núcleo o del patio (misma heurística que la repartición legacy).
    """
    if (exclude_max <= L_min + 0.5) or (exclude_min >= L_max - 0.5):
        return [(L_min, L_max)]
    e_min = max(L_min, exclude_min)
    e_max = min(L_max, exclude_max)
    s1 = max(0.0, e_min - L_min)
    s2 = max(0.0, L_max - e_max)
    if s1 + s2 < 0.1:
        return []
    out: List[Tuple[float, float]] = []
    if s1 > 2.0:
        out.append((L_min, e_min))
    if s2 > 2.0:
        out.append((e_max, L_max))
    return out


def _max_units_on_strips(segments: List[Tuple[float, float]], depth: float, min_area: float) -> int:
    """Cota superior de unidades si cada una exige ~depth * ancho >= min_area."""
    total = 0
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < 0.5 or depth <= 0 or min_area <= 0:
            continue
        total += max(0, int((seg_len * depth) / min_area))
    return total


MIN_PUERTA_M = 0.90  # RNE A.010: vano de puerta real, no esquina rozada


def _door_access(unit_poly, circ_poly, min_len: float = MIN_PUERTA_M, buf: float = 0.05) -> bool:
    """True solo si la unidad comparte una arista real (>=min_len) con la
    circulación (hall/corredor), no si apenas toca una esquina."""
    if unit_poly is None or circ_poly is None or unit_poly.is_empty or circ_poly.is_empty:
        return False
    try:
        shared = unit_poly.boundary.intersection(circ_poly.buffer(buf))
        return float(shared.length) >= min_len
    except Exception:
        return False


def _validate_adjacency(dpto_coords: list, hall_coords: list, tolerance: float = 0.50) -> bool:
    """
    Returns True if dpto_coords shares a real door-length edge with hall_coords
    (>=MIN_PUERTA_M), not just a touching corner.
    """
    if not dpto_coords or len(dpto_coords) < 3 or not hall_coords or len(hall_coords) < 3:
        return False
    try:
        dp_poly = Polygon(dpto_coords)
        hall_poly = Polygon(hall_coords)
        return _door_access(dp_poly, hall_poly, min_len=MIN_PUERTA_M, buf=max(tolerance, 0.05))
    except Exception:
        return False


def _departamento_outline_coords(entry: Any) -> list:
    """Extrae polígono contorno sea lista legacy o dict enriquecido."""
    if isinstance(entry, dict):
        return entry.get("contorno") or []
    return entry or []


# Generación interior delegada a programa.py: see generate_interior_zones.


# ═══════════════════════════════════════════════════════════════
# GRID ESTRUCTURAL DE COLUMNAS
# ═══════════════════════════════════════════════════════════════

def _compute_column_grid(cx, cy, dl_x, dl_y, ds_x, ds_y,
                         L_min, L_max, ds_rows, lote,
                         dl_positions=None):
    """Grilla de columnas 0.50×0.50m a lo largo dl.

    dl_positions: ejes dl explícitos (bordes de unidades + núcleo).
                  Si None, usa separación ~5.5m uniforme.
    ds_rows: posiciones ds donde van filas (±hw y ±exterior).
    """
    COL_HALF = 0.25  # 0.50×0.50m columna
    if dl_positions is None:
        TARGET_SPACING = 5.50
        span = L_max - L_min
        if span < 0.5:
            return []
        n_gaps = max(1, round(span / TARGET_SPACING))
        spacing = span / n_gaps
        dl_positions = [L_min + i * spacing for i in range(n_gaps + 1)]
    else:
        # merge positions closer than 2×COL_HALF to avoid overlapping columns
        dl_sorted = sorted(set(dl_positions))
        merged: List[float] = []
        for p in dl_sorted:
            if not merged or p - merged[-1] > 2 * COL_HALF + 0.05:
                merged.append(p)
        dl_positions = merged

    cols: List[Polygon] = []
    for ds_pos in ds_rows:
        for dl_pos in dl_positions:
            ccx = cx + dl_x * dl_pos + ds_x * ds_pos
            ccy = cy + dl_y * dl_pos + ds_y * ds_pos
            col = Polygon([
                (ccx - dl_x*COL_HALF - ds_x*COL_HALF, ccy - dl_y*COL_HALF - ds_y*COL_HALF),
                (ccx + dl_x*COL_HALF - ds_x*COL_HALF, ccy + dl_y*COL_HALF - ds_y*COL_HALF),
                (ccx + dl_x*COL_HALF + ds_x*COL_HALF, ccy + dl_y*COL_HALF + ds_y*COL_HALF),
                (ccx - dl_x*COL_HALF + ds_x*COL_HALF, ccy - dl_y*COL_HALF + ds_y*COL_HALF),
            ])
            clipped = safe_clip(col, lote)
            if clipped is not None and clipped.area > 0.05:
                cols.append(clipped)
    return cols


# ═══════════════════════════════════════════════════════════════
# AUTO-DUCTOS ADYACENTES A WET BANDS (cocina/baño/lavandería)
# ═══════════════════════════════════════════════════════════════

def _auto_ductos_wet(strip_records, cx, cy, dl_x, dl_y, ds_x, ds_y,
                     hw, lote, pozo_final):
    """Coloca ductos de ventilación en la pared compartida entre dptos adyacentes,
    a la profundidad del centro de la wet band (~t=0.21 desde el hall).

    El programa interior pone Baño/Lavandería en u-alto (borde derecho del dpto).
    La pared dl=nxt del dpto A coincide con dl=off del dpto B → ducto toca
    Baño/Lavandería de A y Cocina de B simultáneamente.

    Unidades aisladas (sin vecino adyacente) en el lado fondo reciben un ducto
    centrado en dl, a profundidad T_WET × depth en su muro posterior.
    """
    # Ducto de ventilación: 0.50×0.50m en muro medianero a ~75% del fondo
    # (zona húmeda: cocina/baño se ubican en sector posterior del dpto).
    DUCT_DIM = 0.50
    T_WET = 0.75
    half_d = DUCT_DIM / 2
    auto_ductos: List[Polygon] = []
    placed: List[tuple] = []

    def _place(wx, wy) -> bool:
        if any(math.hypot(wx - px, wy - py) < DUCT_DIM * 1.2 for px, py in placed):
            return False
        ducto = Polygon([
            (wx - dl_x*half_d - ds_x*half_d, wy - dl_y*half_d - ds_y*half_d),
            (wx + dl_x*half_d - ds_x*half_d, wy + dl_y*half_d - ds_y*half_d),
            (wx + dl_x*half_d + ds_x*half_d, wy + dl_y*half_d + ds_y*half_d),
            (wx - dl_x*half_d + ds_x*half_d, wy - dl_y*half_d + ds_y*half_d),
        ])
        clipped = safe_clip(ducto, lote)
        if clipped is not None and clipped.area >= 0.05:
            auto_ductos.append(clipped)
            placed.append((wx, wy))
            return True
        return False

    for sign_s in (1, -1):
        side = sorted(
            [r for r in strip_records if r.get("sign_s") == sign_s],
            key=lambda r: r.get("dl_off", 0)
        )
        paired_indices: set = set()
        for i in range(len(side) - 1):
            a, b = side[i], side[i + 1]
            if abs(a.get("dl_nxt", 0) - b.get("dl_off", 0)) > 0.15:
                continue  # gap = core/patio, no adyacentes
            wall_dl = (a["dl_nxt"] + b["dl_off"]) / 2
            depth_eff = a.get("depth_eff", hw)
            wet_ds = sign_s * (hw + depth_eff * T_WET)
            wx = cx + dl_x * wall_dl + ds_x * wet_ds
            wy = cy + dl_y * wall_dl + ds_y * wet_ds
            _place(wx, wy)
            paired_indices.add(i)
            paired_indices.add(i + 1)

        # Units isolated from all strip neighbors → ducto in wet zone.
        # For trapezoidal lots the outer wall may be closer at fondo, so try
        # progressively smaller T_WET depths until the ducto fits inside the lot.
        for i, r in enumerate(side):
            if i in paired_indices:
                continue
            dl_center = (r.get("dl_off", 0) + r.get("dl_nxt", 0)) / 2
            depth_eff = r.get("depth_eff", hw)
            for _t in (T_WET, 0.55, 0.35):
                wet_ds = sign_s * (hw + depth_eff * _t)
                wx = cx + dl_x * dl_center + ds_x * wet_ds
                wy = cy + dl_y * dl_center + ds_y * wet_ds
                if _place(wx, wy):
                    break

    return auto_ductos


# ═══════════════════════════════════════════════════════════════
# TOPOLOGÍA: CLAUSTRO (patio central + dptos perimetrales)
# ═══════════════════════════════════════════════════════════════

def _generate_claustro(proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
                       half_L, half_S, hw, num_dptos, num_asc,
                       nec_esc_prot, nec_ascensor, pozo_final, h_edif):
    """Claustro: patio central + corridors perimetrales + dptos en 4 alas."""
    retiro_lat_geo = float(proyecto.retiro_lateral or 2.30)
    retiro_pos_geo = float(proyecto.retiro_posterior or 2.30)
    retiro_lat_neg = retiro_lat_geo
    retiro_lat_pos = retiro_lat_geo
    retiro_fondo_geo = retiro_pos_geo
    _lc_c = list(lote.exterior.coords)
    _cl_dl = [(x - cx) * dl_x + (y - cy) * dl_y for x, y in _lc_c]
    _cl_ds = [(x - cx) * ds_x + (y - cy) * ds_y for x, y in _lc_c]
    L_min = min(_cl_dl) + retiro_lat_neg
    L_max = max(_cl_dl) - retiro_lat_pos
    S_max = max(_cl_ds) - retiro_fondo_geo
    S_min = min(_cl_ds)  # frente — retiro via lote clip

    MIN_FRENTE_DPTO = 3.00
    min_area_dpto = RNE["departamentos"]["min_multifamiliar"]

    # Patio central: ~30% of usable span, min = pozo_final, max = 9m
    span_l = L_max - L_min
    span_s = S_max - S_min
    ph_l = max(pozo_final, min(span_l * 0.30, 9.0))
    ph_s = max(pozo_final, min(span_s * 0.30, 9.0))
    # Clamp so each side can fit at least one apartment
    ph_l = min(ph_l, span_l / 2 - MIN_FRENTE_DPTO - hw)
    ph_s = min(ph_s, span_s / 2 - MIN_FRENTE_DPTO - hw)
    ph_l = max(ph_l, 1.5)
    ph_s = max(ph_s, 1.5)

    ring_l = ph_l + hw   # half-width of ring outer boundary along dl
    ring_s = ph_s + hw   # half-width of ring outer boundary along ds

    # Patio (inner void) and hall ring (outer rect used for rendering)
    patio_poly = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, ph_l, ph_s)
    patio_clipped = safe_clip(patio_poly, lote)
    ring_outer = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, ring_l, ring_s)
    hall_ring = safe_clip(ring_outer, lote)

    # Core at +dl corner, -ds side (near frente), inside ring corridor
    esc_w = RNE["circulacion_v"]["esc_ancho"]
    esc_half_l = 2.50 / 2
    esc_depth = esc_w * 2
    sc_l = ring_l - esc_half_l           # center along dl
    sc_s = -(ring_s - esc_depth / 2)     # center along ds (frente side)
    scx = cx + dl_x * sc_l + ds_x * sc_s
    scy = cy + dl_y * sc_l + ds_y * sc_s
    stair_poly = Polygon([
        (scx - dl_x*esc_half_l - ds_x*(esc_depth/2), scy - dl_y*esc_half_l - ds_y*(esc_depth/2)),
        (scx + dl_x*esc_half_l - ds_x*(esc_depth/2), scy + dl_y*esc_half_l - ds_y*(esc_depth/2)),
        (scx + dl_x*esc_half_l + ds_x*(esc_depth/2), scy + dl_y*esc_half_l + ds_y*(esc_depth/2)),
        (scx - dl_x*esc_half_l + ds_x*(esc_depth/2), scy - dl_y*esc_half_l + ds_y*(esc_depth/2)),
    ])

    asc_polys = []
    asc_l = RNE["ascensor"]["largo"]
    asc_w_dim = RNE["ascensor"]["ancho"]
    for i in range(num_asc):
        a_l = sc_l - esc_half_l - 0.20 - asc_l / 2 - i * (asc_l + 0.30)
        acx = cx + dl_x * a_l + ds_x * sc_s
        acy = cy + dl_y * a_l + ds_y * sc_s
        asc_polys.append(Polygon([
            (acx - dl_x*asc_l/2 - ds_x*asc_w_dim/2, acy - dl_y*asc_l/2 - ds_y*asc_w_dim/2),
            (acx + dl_x*asc_l/2 - ds_x*asc_w_dim/2, acy + dl_y*asc_l/2 - ds_y*asc_w_dim/2),
            (acx + dl_x*asc_l/2 + ds_x*asc_w_dim/2, acy + dl_y*asc_l/2 + ds_y*asc_w_dim/2),
            (acx - dl_x*asc_l/2 + ds_x*asc_w_dim/2, acy - dl_y*asc_l/2 + ds_y*asc_w_dim/2),
        ]))

    core_items = [stair_poly] + asc_polys
    core_clipped = safe_clip(unary_union(core_items).envelope if core_items else stair_poly, lote)

    vest_poly = None
    if nec_esc_prot:
        v_l = sc_l - esc_half_l - 0.75
        vcx = cx + dl_x * v_l + ds_x * sc_s
        vcy = cy + dl_y * v_l + ds_y * sc_s
        vest_poly = Polygon([
            (vcx - dl_x*0.75 - ds_x*(esc_depth/2), vcy - dl_y*0.75 - ds_y*(esc_depth/2)),
            (vcx + dl_x*0.75 - ds_x*(esc_depth/2), vcy + dl_y*0.75 - ds_y*(esc_depth/2)),
            (vcx + dl_x*0.75 + ds_x*(esc_depth/2), vcy + dl_y*0.75 + ds_y*(esc_depth/2)),
            (vcx - dl_x*0.75 + ds_x*(esc_depth/2), vcy - dl_y*0.75 + ds_y*(esc_depth/2)),
        ])

    hall_buf = None
    try:
        if hall_ring and not hall_ring.is_empty:
            hall_buf = hall_ring.buffer(0.40)
    except Exception:
        pass

    def place_strip(seg_min, seg_max, depth_near, depth_far, along_dl: bool):
        """Place apartments in one perimetral strip.
        along_dl=True: frente along dl axis (N/S strips).
        along_dl=False: frente along ds axis (E/W strips).
        depth_near/far: positions along the perpendicular axis.
        """
        units: List[Dict[str, Any]] = []
        seg_len = seg_max - seg_min
        depth = abs(depth_far - depth_near)
        if seg_len < MIN_FRENTE_DPTO or depth < 2.0:
            return units
        n_eff = min(int(seg_len * depth / min_area_dpto), int(seg_len // MIN_FRENTE_DPTO))
        while n_eff > 1 and (depth / (seg_len / n_eff)) > 3.0:
            n_eff -= 1
        if n_eff <= 0:
            return units
        w = seg_len / n_eff
        if w < MIN_FRENTE_DPTO - 0.01:
            return units
        for i in range(n_eff):
            off = seg_min + i * w
            nxt = off + w
            if along_dl:
                corners = [
                    (cx + dl_x*off + ds_x*depth_near, cy + dl_y*off + ds_y*depth_near),
                    (cx + dl_x*nxt + ds_x*depth_near, cy + dl_y*nxt + ds_y*depth_near),
                    (cx + dl_x*nxt + ds_x*depth_far,  cy + dl_y*nxt + ds_y*depth_far),
                    (cx + dl_x*off + ds_x*depth_far,  cy + dl_y*off + ds_y*depth_far),
                ]
            else:
                corners = [
                    (cx + dl_x*depth_near + ds_x*off, cy + dl_y*depth_near + ds_y*off),
                    (cx + dl_x*depth_near + ds_x*nxt, cy + dl_y*depth_near + ds_y*nxt),
                    (cx + dl_x*depth_far  + ds_x*nxt, cy + dl_y*depth_far  + ds_y*nxt),
                    (cx + dl_x*depth_far  + ds_x*off, cy + dl_y*depth_far  + ds_y*off),
                ]
            ap = safe_clip(Polygon(corners), lote)
            if ap is None or ap.area < min_area_dpto:
                continue
            try:
                cr = ap.area / ap.convex_hull.area if ap.convex_hull.area > 0 else 1.0
            except Exception:
                cr = 1.0
            if cr < 0.85:
                continue
            if hall_buf is not None:
                try:
                    if not ap.intersects(hall_buf):
                        continue
                except Exception:
                    pass
            units.append({"poly": ap, "corners": tuple(tuple(c) for c in corners),
                          "off": off, "nxt": nxt, "along_dl": along_dl,
                          "depth_near": depth_near, "depth_far": depth_far})
        return units

    raw_N = place_strip(L_min, L_max, ring_s, S_max, along_dl=True)
    raw_S = place_strip(L_min, L_max, -ring_s, S_min, along_dl=True)
    raw_E = place_strip(-ring_s, ring_s, ring_l, L_max, along_dl=False)
    raw_W = place_strip(-ring_s, ring_s, -ring_l, L_min, along_dl=False)
    all_raw: List[Dict[str, Any]] = raw_N + raw_S + raw_E + raw_W

    # Cap to requested num_departamentos
    if not proyecto.optimizar_densidad and len(all_raw) > num_dptos:
        all_raw = all_raw[:num_dptos]

    # ── Ductos entre unidades adyacentes en cada ala (wet band T_WET desde borde interior) ──
    DUCTO_SIZE = 0.50
    T_WET_CLAU = 0.15
    ductos_claustro: List[Any] = []
    for strip in [raw_N, raw_S, raw_E, raw_W]:
        for i in range(len(strip) - 1):
            u1, u2 = strip[i], strip[i + 1]
            if abs(u2["off"] - u1["nxt"]) > 0.05:
                continue  # not adjacent
            wall_pos = u1["nxt"]
            d_near, d_far = u1["depth_near"], u1["depth_far"]
            depth_span = d_far - d_near
            if abs(depth_span) < 0.5:
                continue
            wet_across = d_near + T_WET_CLAU * depth_span
            if u1["along_dl"]:
                wx = cx + dl_x * wall_pos + ds_x * wet_across
                wy = cy + dl_y * wall_pos + ds_y * wet_across
            else:
                wx = cx + dl_x * wet_across + ds_x * wall_pos
                wy = cy + dl_y * wet_across + ds_y * wall_pos
            d_poly = make_rect(wx, wy, dl_x, dl_y, ds_x, ds_y, DUCTO_SIZE / 2, DUCTO_SIZE / 2)
            d_clipped = safe_clip(d_poly, lote)
            if d_clipped and not d_clipped.is_empty and d_clipped.area > 0.05:
                ductos_claustro.append(d_clipped)

    # ── 4 franjas de corredor perimetral (ring corridor, 0.6 m de ancho) ──
    nc_ds = ph_s + hw / 2
    ec_dl = ph_l + hw / 2
    corridor_strips: List[Any] = []
    for (rcx, rcy, hl, hs) in [
        (cx + ds_x * nc_ds,  cy + ds_y * nc_ds,  ring_l, hw / 2),   # N
        (cx - ds_x * nc_ds,  cy - ds_y * nc_ds,  ring_l, hw / 2),   # S
        (cx + dl_x * ec_dl,  cy + dl_y * ec_dl,  hw / 2, ph_s),     # E
        (cx - dl_x * ec_dl,  cy - dl_y * ec_dl,  hw / 2, ph_s),     # W
    ]:
        c = safe_clip(make_rect(rcx, rcy, dl_x, dl_y, ds_x, ds_y, hl, hs), lote)
        if c and not c.is_empty:
            corridor_strips.append(c)

    departamentos_detalle: List[Dict[str, Any]] = []
    for rec in all_raw:
        ap = rec["poly"]
        corners = rec["corners"]
        area_gross = float(ap.area)
        area_m = area_neta_muros(ap)
        typ = get_typology(area_m)
        zonas_geom = generate_interior_zones(corners, typ, ap, lote)
        val_u = validar_unidad(unit=ap, zonas=zonas_geom,
                               escalera=stair_poly, hall_buf=hall_buf,
                               ductos=ductos_claustro)
        zonas_payload = [{
            "nombre": z["nombre"], "kind": z.get("kind", ""),
            "coords": poly_to_js(z["geom"]),
            "area_m2": r3(float(z["geom"].area)),
            "validacion": z.get("validacion", {}),
        } for z in zonas_geom]
        departamentos_detalle.append({
            "contorno": poly_to_js(ap), "tipologia": typ,
            "area_m2": r3(area_m), "area_gross_m2": r3(area_gross),
            "zonas": zonas_payload, "validacion": val_u,
        })

    cap_total = len(all_raw)
    topo_info = informe_topologia(lote, float(proyecto.frente or 0.0))

    geometry = {
        "hall": [],          # claustro no usa hall central; se envían corridors
        "corridors": [poly_to_js(c) for c in corridor_strips],
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote)),
        "ascensores": [poly_to_js(safe_clip(a, lote)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote)) if vest_poly else [],
        "patio": poly_to_js(patio_clipped),
        "ductos": [poly_to_js(d) for d in ductos_claustro],
        "esquema_area_libre": "claustro",
        "departamentos": departamentos_detalle,
        "cabida_multifamiliar": {
            "contexto": "Perú — claustro: patio central + dptos perimetrales",
            "area_min_dpto_m2": min_area_dpto,
            "profundidad_strip_estimada_m": r3(min(S_max - ring_s, L_max - ring_l)),
            "departamentos_solicitados_planta": num_dptos,
            "capacidad_maxima_estimada_planta": cap_total,
            "capacidad_lado_nucleo": cap_total,
            "capacidad_lado_patio": 0,
            "departamentos_generados_planta": len(departamentos_detalle),
            "nota": "Topología claustro: patio central + 4 alas perimetrales.",
        },
        "topologia": topo_info,
    }
    normativa = {
        "pozo_final": r3(pozo_final),
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": RNE["departamentos"]["min_multifamiliar"],
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_dptos,
            "departamentos_emitidos": len(departamentos_detalle),
            "capacidad_max_estimada_planta": cap_total,
        },
        "topologia": topo_info,
    }
    return geometry, normativa


# ═══════════════════════════════════════════════════════════════
# TOPOLOGÍA: TOWER (núcleo central + dptos esquineros)
# ═══════════════════════════════════════════════════════════════

def _reflex_vertex(lote: Polygon):
    """Vértice reflex (ángulo interior >180°) más pronunciado del lote, o
    None si es convexo. En un lote esquinero (L-shape) es la esquina
    interior — el lado opuesto (la esquina exterior, hacia calle) es la
    que P1 marca como premium; el núcleo debe alejarse de ahí."""
    coords = list(lote.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:
        return None
    area2 = sum(coords[i][0] * coords[(i + 1) % n][1] - coords[(i + 1) % n][0] * coords[i][1]
                for i in range(n))
    ccw = area2 > 0
    best = None
    best_turn = 1e-6
    for i in range(n):
        p0, p1, p2 = coords[i - 1], coords[i], coords[(i + 1) % n]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        turn = cross if ccw else -cross
        if turn < -best_turn:
            best, best_turn = p1, -turn
    return best


def _generate_tower(proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
                    half_L, half_S, hw, num_dptos, num_asc,
                    nec_esc_prot, nec_ascensor, pozo_final, h_edif):
    """Tower: central core + 4 corner quadrant apartments."""
    retiro_lat_geo = float(proyecto.retiro_lateral or 2.30)
    retiro_pos_geo = float(proyecto.retiro_posterior or 2.30)
    retiro_lat_neg = retiro_lat_geo
    retiro_lat_pos = retiro_lat_geo
    retiro_fondo_geo = retiro_pos_geo
    _lc_t = list(lote.exterior.coords)
    _ct_dl = [(x - cx) * dl_x + (y - cy) * dl_y for x, y in _lc_t]
    _ct_ds = [(x - cx) * ds_x + (y - cy) * ds_y for x, y in _lc_t]
    L_min = min(_ct_dl) + retiro_lat_neg
    L_max = max(_ct_dl) - retiro_lat_pos
    S_max = max(_ct_ds) - retiro_fondo_geo
    S_min = min(_ct_ds)

    MIN_FRENTE_DPTO = 3.00
    min_area_dpto = RNE["departamentos"]["min_multifamiliar"]

    esc_w = RNE["circulacion_v"]["esc_ancho"]
    esc_half_l = 2.50 / 2
    esc_depth = esc_w * 2
    asc_l = RNE["ascensor"]["largo"]
    asc_w_dim = RNE["ascensor"]["ancho"]

    # Core block half-dims (stair centered, ascensores along +dl)
    core_half_l = esc_half_l + (0.20 + asc_l) * num_asc if num_asc > 0 else esc_half_l
    core_half_s = esc_depth / 2
    hall_half_l = core_half_l + hw
    hall_half_s = core_half_s + hw

    stair_poly = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, esc_half_l, core_half_s)

    asc_polys = []
    for i in range(num_asc):
        a_l = esc_half_l + 0.20 + asc_l / 2 + i * (asc_l + 0.30)
        acx = cx + dl_x * a_l
        acy = cy + dl_y * a_l
        asc_polys.append(make_rect(acx, acy, dl_x, dl_y, ds_x, ds_y, asc_l / 2, asc_w_dim / 2))

    core_items = [stair_poly] + asc_polys
    core_clipped = safe_clip(unary_union(core_items).envelope if core_items else stair_poly, lote)

    vest_poly = None
    if nec_esc_prot:
        v_l = -(core_half_l + 0.75)
        vest_poly = make_rect(
            cx + dl_x * v_l, cy + dl_y * v_l, dl_x, dl_y, ds_x, ds_y, 0.75, core_half_s
        )

    hall_poly = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, hall_half_l, hall_half_s)
    hall_clipped = safe_clip(hall_poly, lote)
    hall_buf = None
    try:
        if hall_clipped and not hall_clipped.is_empty:
            hall_buf = hall_clipped.buffer(0.40)
    except Exception:
        pass

    # 4 quadrants: (l_near, l_far, s_near, s_far) in local dl/ds coordinates
    quadrants = [
        ( hall_half_l,  L_max,  hall_half_s,  S_max),   # NE
        (L_min, -hall_half_l,  hall_half_s,  S_max),    # NW
        ( hall_half_l,  L_max,  S_min, -hall_half_s),   # SE
        (L_min, -hall_half_l,  S_min, -hall_half_s),    # SW
    ]

    all_raw: List[Dict[str, Any]] = []
    for (l_near, l_far, s_near, s_far) in quadrants:
        q_dl = abs(l_far - l_near)
        q_ds = abs(s_far - s_near)
        if q_dl < MIN_FRENTE_DPTO or q_ds < 2.0:
            continue
        # Split along longer axis for denser fill.
        # Convención corners (programa.py): p0,p1 = borde interior (hall),
        # p2,p3 = borde exterior — el lado hall es el de menor |coordenada|.
        quadrant_units: List[Dict[str, Any]] = []
        if q_dl >= q_ds:
            n = max(1, min(int(q_dl // MIN_FRENTE_DPTO), int(q_dl * q_ds / min_area_dpto)))
            w = q_dl / n
            l_lo = min(l_near, l_far)
            # Lado hall = menor |s| (el cuadrante colinda con el hall central)
            if abs(s_near) <= abs(s_far):
                s_hall, s_out = s_near, s_far
            else:
                s_hall, s_out = s_far, s_near
            for i in range(n):
                lo, hi = l_lo + i * w, l_lo + (i + 1) * w
                corners = [
                    (cx + dl_x*lo + ds_x*s_hall, cy + dl_y*lo + ds_y*s_hall),
                    (cx + dl_x*hi + ds_x*s_hall, cy + dl_y*hi + ds_y*s_hall),
                    (cx + dl_x*hi + ds_x*s_out,  cy + dl_y*hi + ds_y*s_out),
                    (cx + dl_x*lo + ds_x*s_out,  cy + dl_y*lo + ds_y*s_out),
                ]
                ap = safe_clip(Polygon(corners), lote)
                if ap is None or ap.area < min_area_dpto:
                    continue
                try:
                    cr = ap.area / ap.convex_hull.area if ap.convex_hull.area > 0 else 1.0
                except Exception:
                    cr = 1.0
                if cr < 0.85:
                    continue
                quadrant_units.append({
                    "poly": ap, "corners": tuple(tuple(c) for c in corners),
                    "split_dl": True, "off": lo, "nxt": hi,
                    "hall_pos": s_hall, "out_pos": s_out,
                })
        else:
            n = max(1, min(int(q_ds // MIN_FRENTE_DPTO), int(q_dl * q_ds / min_area_dpto)))
            w = q_ds / n
            s_lo_base = min(s_near, s_far)
            # Lado hall = menor |l|
            if abs(l_near) <= abs(l_far):
                l_hall, l_out = l_near, l_far
            else:
                l_hall, l_out = l_far, l_near
            for i in range(n):
                s0, s1 = s_lo_base + i * w, s_lo_base + (i + 1) * w
                corners = [
                    (cx + dl_x*l_hall + ds_x*s0, cy + dl_y*l_hall + ds_y*s0),
                    (cx + dl_x*l_hall + ds_x*s1, cy + dl_y*l_hall + ds_y*s1),
                    (cx + dl_x*l_out  + ds_x*s1, cy + dl_y*l_out  + ds_y*s1),
                    (cx + dl_x*l_out  + ds_x*s0, cy + dl_y*l_out  + ds_y*s0),
                ]
                ap = safe_clip(Polygon(corners), lote)
                if ap is None or ap.area < min_area_dpto:
                    continue
                try:
                    cr = ap.area / ap.convex_hull.area if ap.convex_hull.area > 0 else 1.0
                except Exception:
                    cr = 1.0
                if cr < 0.85:
                    continue
                quadrant_units.append({
                    "poly": ap, "corners": tuple(tuple(c) for c in corners),
                    "split_dl": False, "off": s0, "nxt": s1,
                    "hall_pos": l_hall, "out_pos": l_out,
                })
        all_raw.extend(quadrant_units)

    if not proyecto.optimizar_densidad and len(all_raw) > num_dptos:
        all_raw = all_raw[:num_dptos]

    # ── Ductos en paredes medianeras entre unidades adyacentes (wet band cerca hall) ──
    DUCTO_SIZE_TW = 0.50
    T_WET_TW = 0.20
    ductos_tower: List[Any] = []
    _adj = sorted(all_raw, key=lambda r: (r.get("split_dl", True), r.get("hall_pos", 0), r.get("off", 0)))
    for i in range(len(_adj) - 1):
        a, b = _adj[i], _adj[i + 1]
        if a.get("split_dl") != b.get("split_dl"):
            continue
        if abs(a.get("hall_pos", 0) - b.get("hall_pos", 1e9)) > 0.05:
            continue  # distinto cuadrante
        if abs(a.get("nxt", 0) - b.get("off", 1e9)) > 0.05:
            continue  # no adyacentes
        wall = a["nxt"]
        wet = a["hall_pos"] + T_WET_TW * (a["out_pos"] - a["hall_pos"])
        if a.get("split_dl"):
            wx = cx + dl_x * wall + ds_x * wet
            wy = cy + dl_y * wall + ds_y * wet
        else:
            wx = cx + dl_x * wet + ds_x * wall
            wy = cy + dl_y * wet + ds_y * wall
        d_poly = make_rect(wx, wy, dl_x, dl_y, ds_x, ds_y, DUCTO_SIZE_TW / 2, DUCTO_SIZE_TW / 2)
        d_clipped = safe_clip(d_poly, lote)
        if d_clipped is not None and not d_clipped.is_empty and d_clipped.area > 0.05:
            ductos_tower.append(d_clipped)

    departamentos_detalle: List[Dict[str, Any]] = []
    for rec in all_raw:
        ap = rec["poly"]
        corners = rec["corners"]
        area_gross = float(ap.area)
        area_m = area_neta_muros(ap)
        typ = get_typology(area_m)
        zonas_geom = generate_interior_zones(corners, typ, ap, lote)
        val_u = validar_unidad(unit=ap, zonas=zonas_geom,
                               escalera=stair_poly, hall_buf=hall_buf,
                               ductos=ductos_tower)
        zonas_payload = [{
            "nombre": z["nombre"], "kind": z.get("kind", ""),
            "coords": poly_to_js(z["geom"]),
            "area_m2": r3(float(z["geom"].area)),
            "validacion": z.get("validacion", {}),
        } for z in zonas_geom]
        departamentos_detalle.append({
            "contorno": poly_to_js(ap), "tipologia": typ,
            "area_m2": r3(area_m), "area_gross_m2": r3(area_gross),
            "zonas": zonas_payload, "validacion": val_u,
        })

    cap_total = len(all_raw)
    topo_info = informe_topologia(lote, float(proyecto.frente or 0.0))

    geometry = {
        "hall": poly_to_js(hall_clipped),
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote)),
        "ascensores": [poly_to_js(safe_clip(a, lote)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote)) if vest_poly else [],
        "patio": [],
        "ductos": [poly_to_js(d) for d in ductos_tower],
        "esquema_area_libre": "tower",
        "departamentos": departamentos_detalle,
        "cabida_multifamiliar": {
            "contexto": "Perú — tower: núcleo central + dptos esquineros",
            "area_min_dpto_m2": min_area_dpto,
            "profundidad_strip_estimada_m": r3(min(L_max - hall_half_l, S_max - hall_half_s)),
            "departamentos_solicitados_planta": num_dptos,
            "capacidad_maxima_estimada_planta": cap_total,
            "capacidad_lado_nucleo": cap_total,
            "capacidad_lado_patio": 0,
            "departamentos_generados_planta": len(departamentos_detalle),
            "nota": "Topología tower: núcleo central + 4 cuadrantes perimetrales.",
        },
        "topologia": topo_info,
    }
    normativa = {
        "pozo_final": r3(pozo_final),
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": RNE["departamentos"]["min_multifamiliar"],
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_dptos,
            "departamentos_emitidos": len(departamentos_detalle),
            "capacidad_max_estimada_planta": cap_total,
        },
        "topologia": topo_info,
    }
    return geometry, normativa


def _erode_lote(lote, r_lat: float, r_pos: float):
    """Erosiona el polígono real del lote lado por lado: cada borde se
    desplaza hacia adentro según su retiro (laterales → r_lat, fondo →
    r_pos; frente ya viene neto). Funciona en lotes trapezoidales e
    irregulares — el retiro sigue al borde real, no al bounding box."""
    try:
        cxx, cyy = lote.centroid.x, lote.centroid.y
        coords = list(lote.exterior.coords)
        poly = lote
        K = 10000.0
        for i in range(len(coords) - 1):
            (x1, y1), (x2, y2) = coords[i], coords[i + 1]
            ex, ey = x2 - x1, y2 - y1
            L = math.hypot(ex, ey)
            if L < 0.01:
                continue
            ex, ey = ex / L, ey / L
            nx, ny = -ey, ex
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if (cxx - mx) * nx + (cyy - my) * ny < 0:
                nx, ny = -nx, -ny  # normal hacia adentro
            if abs(ex) > abs(ey):
                r = 0.0 if my < cyy else r_pos  # frente neto / fondo
            else:
                r = r_lat
            if r <= 0:
                continue
            px, py = x1 + nx * r, y1 + ny * r
            qx, qy = x2 + nx * r, y2 + ny * r
            half = Polygon([
                (px - ex * K, py - ey * K),
                (qx + ex * K, qy + ey * K),
                (qx + ex * K + nx * K, qy + ey * K + ny * K),
                (px - ex * K + nx * K, py - ey * K + ny * K),
            ])
            poly = poly.intersection(half)
            if poly.is_empty:
                return None
        if hasattr(poly, "geoms"):
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly if (poly and not poly.is_empty) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# TOPOLOGÍA COSTILLAS — corredor central + dptos a ambos lados
# (calca del patrón real ref 9 Max Gonzales / ref 1 Campodónico:
#  corredor angosto centrado, escalera y ascensores enfrentados en el
#  corredor, franjas de pozo continuas en AMBAS medianeras, bloques
#  frente/fondo a todo el ancho, wet cores contra el corredor)
# ═══════════════════════════════════════════════════════════════

def _generate_costillas(proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
                        half_L, half_S, hw,
                        num_dptos, num_asc, nec_esc_prot, nec_ascensor,
                        pozo_final, h_edif):
    # Calibrado contra DXF reales (referencias/1-6.dxf):
    # crujía de dpto 7.7-8.2m constante, corredor 1.80m, anchos tipológicos
    # 1D 5.2 / 1D+E 6.25 / 2D 6.9 / 2D+E 8.35, pozos al mínimo normativo.
    CORR_W = 1.80
    DEPTH_T, DEPTH_MIN, DEPTH_MAX = 8.20, 6.50, 8.60  # crujía real
    ESC_W, ESC_L = 2.50, float(RNE["circulacion_v"].get("esc_largo", 5.60))
    ASC_W = 2.00
    asc_l = RNE["ascensor"]["largo"]
    min_area_dpto = RNE["departamentos"]["min_multifamiliar"]

    r_lat = float(proyecto.retiro_lateral or 0.0)
    r_pos = float(proyecto.retiro_posterior or 0.0)
    # Retiros sobre el polígono REAL (bordes inclinados incluidos)
    lote_util = _erode_lote(lote, r_lat, r_pos) or lote
    bx0, by0, bx1, by1 = lote_util.bounds
    ya, yb = by0, by1
    D_avail = yb - ya
    # E2: bbox completo — cada elemento se recorta con safe_clip(x, lote_util).
    # Elimina triángulos de esquina en lotes trapezoidales y asimétricos.
    xa, xb = bx0, bx1
    W = xb - xa
    if W < 13.0 or D_avail < 16.0:
        return None, None  # angosto/corto → otros esquemas

    # E1: % área libre = solo validación (opción a) — nunca recorta D_use
    D_use = D_avail

    POZO_REQ = max(2.20, pozo_final)
    fz_req = POZO_REQ
    _solape = 1.20

    # Columnas de dptos con crujía real (≤8.2m); el pozo toma SOLO lo
    # requerido por norma — el excedente de ancho vuelve a las columnas.
    col_w = min(DEPTH_T, (W - CORR_W - 2 * 2.20) / 2)
    fz = (W - CORR_W - 2 * col_w) / 2
    if fz < fz_req:
        # intentar conformidad H/4 reduciendo crujía hasta el mínimo real
        _col_alt = (W - CORR_W - 2 * fz_req) / 2
        if _col_alt >= DEPTH_MIN:
            col_w = _col_alt
            fz = fz_req
    elif fz > fz_req:
        # pozo no más grande que lo normativo: devolver ancho a las columnas
        col_w = min(DEPTH_MAX, col_w + (fz - fz_req))
        fz = (W - CORR_W - 2 * col_w) / 2
    franja_conf = (fz + 1e-6) >= POZO_REQ
    if col_w < 4.0:
        return None, None
    ucl = xa + fz + col_w        # borde izquierdo del corredor
    ucr = ucl + CORR_W           # borde derecho del corredor

    # Alturas del núcleo en cada columna (al arranque de la zona costillas)
    nuc_l_h = ESC_L + (1.0 if nec_esc_prot else 0.0) * 0.0  # vestíbulo embebido
    nuc_r_h = (num_asc * (asc_l + 0.20)) if num_asc > 0 else 0.0

    # Filas: ancho de fachada tipológico real (1D+E 6.25 / 2D 6.9), nunca <5.2
    h_fila = max(5.2, (min_area_dpto * 1.13) / col_w)

    def _block_cap(width_avail, depth_blk):
        """Máx unidades en bloque frente/fondo: frente tipológico mínimo
        5.2m (calibración DXF Lima) y área neta del bloque. Tope duro 2 —
        el corredor central (stub ucl..ucr) solo da puerta a las unidades
        que abarcan ese eje; una 3ra unidad lateral quedaría sin acceso
        real y se descarta en _door_access (huecos silenciosos). H2 fix:
        antes el corte binario usaba W/2 con la franja de pozo restada,
        subestimando la mitad neta y forzando n=1 (un solo dpto >100m²)
        en lotes de 15-20m que sí caben en 2 unidades tipológicas."""
        if depth_blk <= 3.5 or width_avail <= 0:
            return 1
        by_frente = int(width_avail // 5.2)
        by_area = int((width_avail * depth_blk) / (min_area_dpto * 1.05))
        return max(1, min(2, by_frente, by_area))

    # ── Búsqueda: nº de filas por lado que mejor cumple num_dptos ──
    # Bloques frente/fondo con crujía real 8.2m; el excedente de profundidad
    # va al patio posterior (área libre), no a inflar unidades.
    best = None
    # F6: crujía calibrada (DXF Lima 7.7-8.2m) — el sobrante de profundidad
    # pasa a patio posterior, no infla los dptos. Escalación: si con la
    # crujía calibrada no se alcanza num_dptos, se relaja el cap.
    for _cap in (DEPTH_T, DEPTH_MAX, float("inf")):
        for n_m_try in range(0, 9):
            _a, _b_ = (n_m_try + 1) // 2, n_m_try // 2
            for (fl, fr) in {(_a, _b_), (_b_, _a)}:
                Dm_try = max(nuc_l_h + fl * h_fila, nuc_r_h + fr * h_fila, ESC_L)
                if Dm_try > D_use - 2 * DEPTH_MIN + 1e-9:
                    continue
                Df_try = Db_try = min(_cap, (D_use - Dm_try) / 2)
                n_f = _block_cap(W, Df_try)
                n_b = _block_cap(W, Db_try)
                tot = min(num_dptos, n_f + n_b + n_m_try)
                # max logrado; luego menor sobre-capacidad; luego núcleo más compacto
                key = (tot, -(n_f + n_b + n_m_try), -Dm_try)
                if best is None or key > best[0]:
                    best = (key, fl, fr, Df_try, Db_try, n_f, n_b)
        if best is not None and best[0][0] >= num_dptos:
            break
    if best is None:
        return None, None
    _, filas_l, filas_r, Df, Db, n_f_cap, n_b_cap = best
    n_m = filas_l + filas_r
    # Profundidad de zona media = lo necesario; el resto queda como patio
    Dm = max(nuc_l_h + filas_l * h_fila, nuc_r_h + filas_r * h_fila, ESC_L) + 0.20

    # A4: la construcción reparte Dm completo entre filas_l/filas_r por
    # igual (no descuenta nuc_l_h/nuc_r_h, el núcleo se recorta después).
    # Si un lado tiene menos filas que el otro, hereda la Dm del lado que
    # manda y su única fila se estira (>75m², fuera de rango calibrado).
    # Solo se repone ESE lado (el que no fija Dm) con más filas de tamaño
    # h_fila real; el lado que ya fija Dm queda intacto para no alterar
    # el manejo de bordes inclinados/trapezoidales.
    term_l = nuc_l_h + filas_l * h_fila
    term_r = nuc_r_h + filas_r * h_fila
    if filas_l > 0 and term_l < term_r - 1e-6:
        filas_l = max(filas_l, int(Dm // h_fila))
    if filas_r > 0 and term_r < term_l - 1e-6:
        filas_r = max(filas_r, int(Dm // h_fila))
    n_m = filas_l + filas_r

    yf0 = ya + Df       # arranque zona costillas (línea frente)
    yb0 = yf0 + Dm      # remate zona costillas (línea fondo)
    y_end = yb0 + Db
    patio_depth = yb - y_end

    # Ajuste de unidades pedidas en bloques
    n_total = max(1, min(num_dptos, n_f_cap + n_b_cap + n_m))
    # H2: reparto ALTERNADO frente/fondo (antes siempre llenaba n_b hasta
    # su cap antes de tocar n_f -- con caps >2 dejaba el bloque frente
    # entero sin partir mientras el fondo absorbía todo el remanente).
    n_f, n_b = 1, 1
    _rem = n_total - 2 - n_m
    _turn = 0
    while _rem > 0:
        if _turn == 0 and n_f < n_f_cap:
            n_f += 1; _rem -= 1
        elif _turn == 1 and n_b < n_b_cap:
            n_b += 1; _rem -= 1
        elif n_f < n_f_cap:
            n_f += 1; _rem -= 1
        elif n_b < n_b_cap:
            n_b += 1; _rem -= 1
        else:
            break
        _turn = 1 - _turn

    # E2: si borde inclinado del lote corta un bloque fondo/frente dejando dptos
    # demasiado pequeños, bajar a n_b=1 / n_f=1 (un dpto a todo ancho).
    def _y_top_at(poly, x):
        """Máx y de un polígono al corte vertical x."""
        from shapely.geometry import LineString as _LS
        try:
            ln = _LS([(x, poly.bounds[1] - 1), (x, poly.bounds[3] + 1)])
            inter = poly.boundary.intersection(ln)
            ys = ([p.y for p in inter.geoms] if hasattr(inter, "geoms")
                  else ([inter.y] if not inter.is_empty else []))
            return max(ys) if ys else poly.bounds[3]
        except Exception:
            return poly.bounds[3]

    def _shrink_for_slope(n, y_top_of_x, y0):
        """H2: bloque partido en n a lo ancho (xa..xb); si el borde
        inclinado del lote deja algún segmento bajo área mínima, reduce n
        (generaliza el viejo chequeo binario n==2 a cualquier N)."""
        while n > 1:
            step = (xb - xa) / n
            ok = True
            for i in range(n):
                xm = xa + (i + 0.5) * step
                d = max(0.0, y_top_of_x(xm) - y0)
                if d * step < min_area_dpto * 1.05:
                    ok = False
                    break
            if ok:
                break
            n -= 1
        return n

    if n_b > 1:
        n_b = _shrink_for_slope(n_b, lambda x: _y_top_at(lote_util, x), yb0)

    # ── Núcleo: escalera (izq) y ascensores (der) enfrentados al corredor ──
    stair_poly = Polygon([
        (ucl - ESC_W, yf0), (ucl, yf0),
        (ucl, yf0 + ESC_L), (ucl - ESC_W, yf0 + ESC_L),
    ])
    vest_poly = None
    if nec_esc_prot:
        vest_poly = Polygon([
            (ucl - 1.50, yf0), (ucl, yf0),
            (ucl, yf0 + 1.50), (ucl - 1.50, yf0 + 1.50),
        ])
    asc_polys = []
    for i in range(num_asc):
        a0 = yf0 + i * (asc_l + 0.20)
        asc_polys.append(Polygon([
            (ucr, a0), (ucr + ASC_W, a0),
            (ucr + ASC_W, a0 + asc_l), (ucr, a0 + asc_l),
        ]))
    core_items = [stair_poly] + asc_polys
    core_clipped = safe_clip(unary_union(core_items).envelope, lote_util)

    # ── Hall: corredor central + ensanche frente a escalera Y ascensores
    # (G3 — nodo distribuidor en ambos frentes de núcleo, no solo pasillo;
    # dimensionado por nuc_l_h/nuc_r_h reales, válido para cualquier lote) ──
    hall_parts = [Polygon([(ucl, yf0), (ucr, yf0), (ucr, yb0), (ucl, yb0)])]
    _ens_l = min(nuc_l_h + 0.8, Dm)
    hall_parts.append(Polygon([
        (ucl - 0.70, yf0), (ucl, yf0),
        (ucl, yf0 + _ens_l), (ucl - 0.70, yf0 + _ens_l),
    ]))
    if num_asc > 0:
        _ens_r = min(nuc_r_h + 0.8, Dm)
        hall_parts.append(Polygon([
            (ucr, yf0), (ucr + 0.70, yf0),
            (ucr + 0.70, yf0 + _ens_r), (ucr, yf0 + _ens_r),
        ]))
    hall_clipped = safe_clip(unary_union(hall_parts), lote_util)
    hall_buf = hall_clipped.buffer(0.40) if hall_clipped and not hall_clipped.is_empty else None

    def _split_block(n):
        """H2: partición N-way del bloque frente/fondo (xa..xb). N=1 no
        parte; N=2 mantiene el corte por el eje del corredor (compat
        visual); N>=3 reparte uniforme por ancho tipológico."""
        if n <= 1:
            return [(xa, xb)]
        if n == 2:
            _mid = (ucl + ucr) / 2
            return [(xa, _mid), (_mid, xb)]
        step = (xb - xa) / n
        return [(xa + i * step, xa + (i + 1) * step) for i in range(n)]

    # ── Specs de unidades ──
    units_spec = []
    # Bloque frente (puertas al arranque del corredor)
    _splits_f = _split_block(n_f)
    for (u0, u1) in _splits_f:
        units_spec.append({
            "corners": ((u0, yf0), (u1, yf0), (u1, ya), (u0, ya)),
            "lado": "frente", "fachada": bool(proyecto.frente_exterior),
        })
    # Filas izquierda (wet al corredor, dormitorios a franja izq). Arrancan en
    # yf0 (no tras el núcleo): la franja junto a escalera que el núcleo no
    # ocupa en X se suma a la primera fila en vez de quedar remanente muerto
    # (G2); el propio núcleo se recorta después vía difference(core_union).
    _v = yf0
    _h_l = (yb0 - _v) / filas_l if filas_l > 0 else 0.0
    for j in range(filas_l):
        r0, r1 = _v + j * _h_l, _v + (j + 1) * _h_l
        units_spec.append({
            "corners": ((ucl, r0), (ucl, r1), (xa + fz, r1), (xa + fz, r0)),
            "lado": "intermedio", "fachada": False,
        })
    # Filas derecha
    _v = yf0
    _h_r = (yb0 - _v) / filas_r if filas_r > 0 else 0.0
    for j in range(filas_r):
        r0, r1 = _v + j * _h_r, _v + (j + 1) * _h_r
        units_spec.append({
            "corners": ((ucr, r0), (ucr, r1), (xb - fz, r1), (xb - fz, r0)),
            "lado": "intermedio", "fachada": False,
        })
    # Bloque fondo (puertas al remate del corredor)
    _splits_b = _split_block(n_b)
    for (u0, u1) in _splits_b:
        units_spec.append({
            "corners": ((u0, yb0), (u1, yb0), (u1, y_end), (u0, y_end)),
            "lado": "fondo",
            "fachada": bool(proyecto.fondo_exterior) or patio_depth >= 2.5,
        })

    # ── Franjas de pozo en medianeras: acotadas SOLO a las filas intermedias
    # que ventilan (zona media ± solape) — nunca invaden bloques frente/fondo,
    # que ventilan a calle/patio y no requieren pozo lateral (G1, RNE A.010:
    # el pozo se dimensiona frente a los ambientes que sirve, no de punta a punta) ──
    pozos_final, cumple_final = [], []
    franja_izq = Polygon([(xa, yf0 - _solape), (xa + fz, yf0 - _solape),
                          (xa + fz, yb0 + _solape), (xa, yb0 + _solape)])
    franja_der = Polygon([(xb - fz, yf0 - _solape), (xb, yf0 - _solape),
                          (xb, yb0 + _solape), (xb - fz, yb0 + _solape)])
    for fp_raw in (franja_izq, franja_der):
        fp = safe_clip(fp_raw, lote_util)
        if fp is not None and not fp.is_empty and fp.area > 0.5:
            pozos_final.append(fp)
            cumple_final.append(franja_conf)
    pozos_union = unary_union(pozos_final).buffer(0.0) if pozos_final else None

    # E3: hall = solo pasillo compacto. Remanentes van a "remanentes_zona_media"
    # (incluidos en footprint pero NO en pct_circ). E4 los convierte en dptos.
    remanentes_zona_media: List[Polygon] = []
    try:
        zona_media = Polygon([(xa + fz, yf0), (xb - fz, yf0),
                              (xb - fz, yb0), (xa + fz, yb0)])
        _ocupado = unary_union(
            [Polygon(s["corners"]) for s in units_spec if s["lado"] == "intermedio"]
            + core_items + hall_parts
            + ([vest_poly] if vest_poly else [])
        ).buffer(0.0)
        _resto = zona_media.difference(_ocupado)
        if pozos_union is not None:
            _resto = _resto.difference(pozos_union)
        _piezas = list(_resto.geoms) if hasattr(_resto, "geoms") else [_resto]
        remanentes_zona_media = [p for p in _piezas if not p.is_empty and p.area >= 0.5]
    except Exception:
        pass

    # ── Construcción + validación de unidades ──
    ductos: List[Polygon] = []
    departamentos_detalle: List[Dict[str, Any]] = []
    sin_acceso = 0
    core_union = unary_union(core_items).buffer(0.0)
    for spec in units_spec:
        corners = spec["corners"]
        ap = safe_clip(Polygon(corners), lote_util)
        if ap is None or ap.is_empty:
            continue
        for sub in (pozos_union, core_union):
            if sub is not None and ap.intersects(sub):
                d_ = ap.difference(sub)
                if hasattr(d_, "geoms"):
                    d_ = max(d_.geoms, key=lambda g: g.area)
                ap = d_
        if hall_clipped is not None:
            ap = ap.difference(hall_clipped.buffer(0.0))
            if hasattr(ap, "geoms"):
                ap = max(ap.geoms, key=lambda g: g.area)
        if ap.is_empty or ap.area < min_area_dpto * 0.85:
            continue
        if hall_clipped is not None and not _door_access(ap, hall_clipped):
            sin_acceso += 1
            continue
        area_gross = float(ap.area)
        area_m = area_neta_muros(ap)
        typ = get_typology(area_m)
        zonas_geom = generate_interior_zones(corners, typ, ap, lote)
        val_u = validar_unidad(
            unit=ap, zonas=zonas_geom,
            escalera=stair_poly, hall_buf=hall_buf, ductos=ductos,
            pozos=[p for p, ok in zip(pozos_final, cumple_final) if ok],
        )
        val_u["fachada_exterior"] = bool(spec["fachada"])
        val_u["distancia_escalera_m"] = val_u.get("distancia_evac_m", 0.0)
        val_u["dist_esc_cumple"] = val_u.get("evac_cumple", True)
        zonas_payload = [{
            "nombre": z["nombre"], "kind": z.get("kind", ""),
            "coords": poly_to_js(z["geom"]),
            "area_m2": r3(float(z["geom"].area)),
            "validacion": z.get("validacion", {}),
        } for z in zonas_geom]
        departamentos_detalle.append({
            "contorno": poly_to_js(ap),
            "tipologia": typ,
            "area_m2": r3(area_m),
            "area_gross_m2": r3(area_gross),
            "lado": spec["lado"],
            "es_reducida": bool(_rect_min_side(ap) < 5.2 or area_m < min_area_dpto * 1.05),
            "zonas": zonas_payload,
            "validacion": val_u,
        })

    if len(departamentos_detalle) < 2:
        return None, None

    patio_poly = None
    if (yb - y_end) > 0.30:
        # patio dentro del lote útil (la banda de retiro posterior queda
        # fuera: el patio cuenta como área diseñada, el retiro no)
        patio_poly = safe_clip(Polygon([
            (xa, y_end), (xb, y_end), (xb, yb), (xa, yb),
        ]), lote_util)

    # Columnas en intersecciones de ejes
    COL_H = 0.25
    u_lines = sorted({xa, xa + fz, ucl, ucr, xb - fz, xb})
    v_lines = sorted({ya, yf0, yb0, y_end} |
                     {c[1] for s in units_spec for c in s["corners"]})
    columnas = []
    avoid = pozos_union.buffer(0.05) if pozos_union is not None else None
    for ux in u_lines:
        for vy in v_lines:
            cpol = Polygon([(ux - COL_H, vy - COL_H), (ux + COL_H, vy - COL_H),
                            (ux + COL_H, vy + COL_H), (ux - COL_H, vy + COL_H)])
            cc = safe_clip(cpol, lote_util)
            if cc is None or cc.is_empty or cc.area < 0.06:
                continue
            if avoid is not None and cc.intersects(avoid):
                continue
            columnas.append(cc)

    cap_total = n_f_cap + n_b_cap + n_m
    topo_info = informe_topologia(lote, float(proyecto.frente or 0.0))
    topo_info["seleccion"]["recomendada"] = "costillas"

    geometry = {
        "hall": poly_to_js(hall_clipped),
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote_util)),
        "ascensores": [poly_to_js(safe_clip(a, lote_util)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote_util)) if vest_poly else [],
        "patio": poly_to_js(patio_poly) if patio_poly else [],
        "ductos": [poly_to_js(d) for d in ductos],
        "remanentes_zona_media": [poly_to_js(r) for r in remanentes_zona_media],
        "pozos_luz": [poly_to_js(p) for p in pozos_final],
        "pozos_luz_cumple": list(cumple_final),
        "columnas": [poly_to_js(c) for c in columnas],
        "esquema_area_libre": "costillas",
        "departamentos": departamentos_detalle,
        "cabida_multifamiliar": {
            "contexto": "Perú — costillas: corredor central + franjas de pozo en medianeras (patrón Lima entre medianeras)",
            "area_min_dpto_m2": min_area_dpto,
            "profundidad_strip_estimada_m": r3(Df),
            "departamentos_solicitados_planta": num_dptos,
            "capacidad_maxima_estimada_planta": cap_total,
            "capacidad_lado_nucleo": n_f_cap + n_m,
            "capacidad_lado_patio": n_b_cap,
            "departamentos_generados_planta": len(departamentos_detalle),
            "nota": "Corredor central 1.50m; escalera y ascensores enfrentados; dormitorios ventilan a franjas de pozo laterales.",
        },
        "topologia": topo_info,
    }
    _n_ok = sum(1 for ok in cumple_final if ok)
    normativa = {
        "pozo_final": r3(pozo_final),
        "pozos_luz_check": {
            "dimension_requerida_m": r3(POZO_REQ),
            "colocados": len(pozos_final),
            "conformes": _n_ok,
            "no_conformes": len(pozos_final) - _n_ok,
            "ok": (len(pozos_final) - _n_ok) == 0,
            "nota": "Franjas de pozo continuas en ambas medianeras (ancho {:.2f}m vs H/4 = {:.2f}m).".format(fz, POZO_REQ),
        },
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": min_area_dpto,
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_dptos,
            "departamentos_emitidos": len(departamentos_detalle),
            "capacidad_max_estimada_planta": cap_total,
            "descartados_sin_acceso": sin_acceso,
        },
        "topologia": topo_info,
    }
    return geometry, normativa


# ═══════════════════════════════════════════════════════════════
# TOPOLOGÍA COSTILLAS — DOS NÚCLEOS (P1: ancho >24m o fondo >38m)
# Dos torres tipo-costillas independientes con patio central y junta
# constructiva. Reusa _generate_costillas dos veces sobre sub-lotes ya
# eroded (retiros reales aplicados una sola vez, sobre el lote completo)
# para no duplicar la lógica de generación por fila/pozo/hall.
# ═══════════════════════════════════════════════════════════════
DOS_NUCLEOS_W_MIN = 24.0
PATIO_CENTRAL_GAP = 3.0  # ancho del patio/junta entre torres


def _generate_costillas_dos_nucleos(proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
                                     half_L, half_S, hw,
                                     num_dptos, num_asc, nec_esc_prot, nec_ascensor,
                                     pozo_final, h_edif):
    r_lat = float(proyecto.retiro_lateral or 0.0)
    r_pos = float(proyecto.retiro_posterior or 0.0)
    lote_util = _erode_lote(lote, r_lat, r_pos) or lote
    bx0, by0, bx1, by1 = lote_util.bounds
    W = bx1 - bx0
    if W < DOS_NUCLEOS_W_MIN:
        return None, None

    mid_x = (bx0 + bx1) / 2
    xl_cut = mid_x - PATIO_CENTRAL_GAP / 2
    xr_cut = mid_x + PATIO_CENTRAL_GAP / 2
    K = 10000.0
    half_izq = Polygon([(bx0 - K, by0 - K), (xl_cut, by0 - K), (xl_cut, by1 + K), (bx0 - K, by1 + K)])
    half_der = Polygon([(xr_cut, by0 - K), (bx1 + K, by0 - K), (bx1 + K, by1 + K), (xr_cut, by1 + K)])
    lote_l = safe_clip(half_izq, lote_util)
    lote_r = safe_clip(half_der, lote_util)
    if lote_l is None or lote_l.is_empty or lote_r is None or lote_r.is_empty:
        return None, None
    Wl = lote_l.bounds[2] - lote_l.bounds[0]
    Wr = lote_r.bounds[2] - lote_r.bounds[0]
    if Wl < 13.0 or Wr < 13.0:
        return None, None  # cada torre necesita el mínimo de costillas por sí sola

    # Retiros ya aplicados sobre lote_util antes del corte: cada sub-torre
    # recibe retiro 0 para no erosionar de nuevo (incluyendo el corte central,
    # que es junta constructiva, no medianera con retiro).
    try:
        proyecto_sub = proyecto.model_copy(update={"retiro_lateral": 0.0, "retiro_posterior": 0.0})
    except AttributeError:  # pydantic v1
        proyecto_sub = proyecto.copy(update={"retiro_lateral": 0.0, "retiro_posterior": 0.0})

    nd_l = round(num_dptos * Wl / (Wl + Wr))
    nd_l = max(1, min(num_dptos - 1, nd_l))
    nd_r = num_dptos - nd_l
    asc_l_n = 1 if nec_ascensor else 0
    asc_r_n = 1 if nec_ascensor else 0

    args_l = (proyecto_sub, lote_l, cx, cy, dl_x, dl_y, ds_x, ds_y, half_L, half_S, hw,
              nd_l, asc_l_n, nec_esc_prot, nec_ascensor, pozo_final, h_edif)
    args_r = (proyecto_sub, lote_r, cx, cy, dl_x, dl_y, ds_x, ds_y, half_L, half_S, hw,
              nd_r, asc_r_n, nec_esc_prot, nec_ascensor, pozo_final, h_edif)
    g_l, n_l = _generate_costillas(*args_l)
    g_r, n_r = _generate_costillas(*args_r)
    if g_l is None or g_r is None:
        return None, None

    patio_central = safe_clip(Polygon([
        (xl_cut, by0), (xr_cut, by0), (xr_cut, by1), (xl_cut, by1),
    ]), lote_util)

    geometry = {
        "hall": g_l["hall"],
        "halls": [g_l["hall"], g_r["hall"]],
        "core": g_l["core"],
        "cores": [g_l["core"], g_r["core"]],
        "escalera": g_l["escalera"],
        "escaleras": [g_l["escalera"], g_r["escalera"]],
        "ascensores": g_l["ascensores"] + g_r["ascensores"],
        "ascensores_por_nucleo": [g_l["ascensores"], g_r["ascensores"]],
        "vestibulo": g_l["vestibulo"] or g_r["vestibulo"],
        "vestibulos": [g_l["vestibulo"], g_r["vestibulo"]],
        "patio": g_l["patio"] or g_r["patio"],
        "patio_central": poly_to_js(patio_central) if patio_central else [],
        # Un núcleo completo (hall+escalera+ascensores+vestíbulo+core) por
        # torre -- consumido por primer_piso/sótano/azotea para no tratar
        # las dos torres como si compartieran un único núcleo global.
        "nucleos": [
            {"hall": g_l["hall"], "escalera": g_l["escalera"],
             "ascensores": g_l["ascensores"], "vestibulo": g_l["vestibulo"],
             "core": g_l["core"]},
            {"hall": g_r["hall"], "escalera": g_r["escalera"],
             "ascensores": g_r["ascensores"], "vestibulo": g_r["vestibulo"],
             "core": g_r["core"]},
        ],
        "ductos": g_l["ductos"] + g_r["ductos"],
        "remanentes_zona_media": g_l["remanentes_zona_media"] + g_r["remanentes_zona_media"],
        "pozos_luz": g_l["pozos_luz"] + g_r["pozos_luz"],
        "pozos_luz_cumple": g_l["pozos_luz_cumple"] + g_r["pozos_luz_cumple"],
        "columnas": g_l["columnas"] + g_r["columnas"],
        "esquema_area_libre": "costillas_dos_nucleos",
        "departamentos": g_l["departamentos"] + g_r["departamentos"],
        "cabida_multifamiliar": {
            "contexto": "Perú — costillas dos núcleos: dos torres independientes con patio central y junta constructiva (P1, ancho >24m)",
            "area_min_dpto_m2": g_l["cabida_multifamiliar"]["area_min_dpto_m2"],
            "departamentos_solicitados_planta": num_dptos,
            "capacidad_maxima_estimada_planta": (g_l["cabida_multifamiliar"]["capacidad_maxima_estimada_planta"]
                                                  + g_r["cabida_multifamiliar"]["capacidad_maxima_estimada_planta"]),
            "departamentos_generados_planta": len(g_l["departamentos"]) + len(g_r["departamentos"]),
            "nota": "Dos torres tipo-costillas separadas por patio central (junta constructiva); cada una con núcleo propio.",
        },
        "topologia": g_l["topologia"],
    }
    normativa = {
        "pozo_final": r3(pozo_final),
        "pozos_luz_check": {
            "colocados": len(geometry["pozos_luz"]),
            "conformes": sum(1 for ok in geometry["pozos_luz_cumple"] if ok),
            "no_conformes": sum(1 for ok in geometry["pozos_luz_cumple"] if not ok),
            "ok": all(geometry["pozos_luz_cumple"]),
            "nota": "Dos torres, franjas de pozo continuas por torre en ambas medianeras.",
        },
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": n_l["area_min_dpto"],
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_dptos,
            "departamentos_emitidos": len(geometry["departamentos"]),
            "capacidad_max_estimada_planta": geometry["cabida_multifamiliar"]["capacidad_maxima_estimada_planta"],
            "descartados_sin_acceso": (n_l["cabida_planta"]["descartados_sin_acceso"]
                                        + n_r["cabida_planta"]["descartados_sin_acceso"]),
        },
        "topologia": g_l["topologia"],
    }
    return geometry, normativa


# ═══════════════════════════════════════════════════════════════
# TOPOLOGÍA HALL COMPACTO — patrón Lima moderna entre medianeras
# ═══════════════════════════════════════════════════════════════
# Planta típica real: bloque de dptos al frente (fachada a calle),
# núcleo (escalera+ascensor) adosado a una medianera a media profundidad,
# hall de distribución compacto, dptos intermedios ventilando a pozos
# en medianeras, bloque de dptos al fondo con patio posterior.
# Sustituye al corredor central continuo (spine) en lotes rectangulares.

def _generate_hall_compacto(proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
                            half_L, half_S, hw,
                            num_dptos, num_asc, nec_esc_prot, nec_ascensor,
                            pozo_final, h_edif):
    HALL_W = 1.20      # ancho mínimo hall distribución (RNE A.010 ≥1.20)
    CORR_W = 1.50      # corredor hacia bloque fondo (RNE A.010 ≥1.20)
    CORE_W = 2.50      # ancho columna de núcleo (escalera 2×1.20 + muro)
    # Caja de escalera de evacuación real: dos tramos de 1.20m + descansos
    ESC_L  = float(RNE["circulacion_v"].get("esc_largo", 5.60))
    MURO_T = 0.15
    min_area_dpto = RNE["departamentos"]["min_multifamiliar"]
    asc_l = RNE["ascensor"]["largo"]

    # Retiros sobre el polígono REAL del lote (bordes inclinados incluidos);
    # si el usuario define retiro, se respeta SIEMPRE. Retiro 0 = medianera.
    r_lat = float(proyecto.retiro_lateral or 0.0)
    r_pos = float(proyecto.retiro_posterior or 0.0)
    _lu = _erode_lote(lote, r_lat, r_pos)
    if _lu is not None:
        lote = _lu  # todo el generador recorta contra el lote erosionado
    bx0, by0, bx1, by1 = lote.bounds
    xa, xb = bx0, bx1
    ya, yb = by0, by1
    W = xb - xa
    D_avail = yb - ya
    if W < 6.5 or D_avail < 10.0:
        return None, None

    # E1: % área libre = solo validación (opción a) — nunca recorta D_use
    D_use = D_avail

    # ── Pozos de luz: dimensión best-effort vs requerida H/4 ──
    # (se calcula antes del dimensionado para descontar su mordida en filas)
    POZO_REQ = max(2.20, pozo_final)
    pw = min(POZO_REQ, max(2.20, W * 0.28))
    pv = min(POZO_REQ, 4.0)
    pozo_conf = (min(pw, pv) + 1e-6) >= POZO_REQ
    _pozo_loss = pw * pv / 2.0  # mordida típica de pozo de junta por unidad
    # Pozo franja lateral continua (patrón real: franja en medianera a lo largo
    # de la zona media, área >> normativa, unidades intermedias rematan en ella)
    pz_franja = max(2.20, min(POZO_REQ, max(2.20, W * 0.25)))
    franja_conf = (pz_franja + 1e-6) >= POZO_REQ
    _solape = 1.20  # muesca de la franja sobre bloques frente/fondo
    _bite_blk = pz_franja * _solape  # mordida de la franja en esquina de bloque

    # ── Núcleo: escalera + ascensores en columna a la medianera izquierda ──
    Lc = ESC_L + (num_asc * (asc_l + 0.20) if num_asc > 0 else 0.0)
    DF_MIN, DF_MAX = 5.5, 11.0
    DF_CAL = 8.20  # F6: crujía calibrada (DXF Lima 7.7-8.2m); sobrante → patio

    # Lote poco profundo: núcleo posterior (escalera al fondo, patrón micro-lote)
    nucleo_posterior = D_use < DF_MIN + Lc
    CORE_ROW_D = 2.50  # profundidad de la fila de núcleo posterior
    n_m_sel = 0  # filas intermedias seleccionadas (solo esquema con bloque fondo)

    Wm_est = W - CORE_W - CORR_W

    def _block_cap_d(width_avail, depth_blk, bite_area):
        """Máx unidades en bloque full-width: frente mín 5.2m (calibración
        DXF Lima — ancho tipológico 1D, REGLAS_DISENO.md) y área neta
        del bloque (descontando mordidas de pozo, que se compensan con
        ancho extra en las unidades de borde)."""
        if depth_blk <= 3.5 or width_avail <= 0:
            return 0
        by_frente = int(width_avail // 5.2)
        by_area = int((width_avail * depth_blk - bite_area) / (min_area_dpto * 1.05))
        return max(0, min(4, by_frente, by_area))

    if nucleo_posterior:
        Df = D_use - CORE_ROW_D - HALL_W
        if Df < 4.2 or W * Df < min_area_dpto * 1.05:
            return None, None
        db_exists = False
        Dm = HALL_W + CORE_ROW_D
        Db = 0.0
    elif D_use >= DF_MIN * 2 + max(Lc, HALL_W * 2):
        # ── Búsqueda del mejor reparto frente/medio/fondo para num_dptos ──
        # Con filas intermedias, el pozo es franja lateral continua (patrón
        # real): las filas rematan en la franja, sin mordidas internas.
        db_exists = True
        best = None
        max_rows = max(0, int((D_use - 2 * DF_MIN - HALL_W * 2) // 3.0))
        # F6: crujía calibrada — sobrante de profundidad pasa a patio
        # posterior (y_end < yb), no infla dptos ni la zona media. Escalación:
        # si con la crujía calibrada no se alcanza num_dptos, se relaja el cap
        # (honrar nd pedido manda sobre la calibración).
        for _cap in (DF_CAL, 9.5, float("inf")):
            for n_m_try in range(0, max_rows + 1):
                Wm_eff = Wm_est - (pz_franja if n_m_try > 0 else 0.0)
                if n_m_try > 0 and Wm_eff < 4.5:
                    break
                h_fila = max(3.0, min_area_dpto * 1.05 / max(Wm_eff, 3.0))
                Dm_try = max(Lc, HALL_W * 2 + n_m_try * h_fila)
                if Dm_try > D_use - 2 * DF_MIN + 1e-9:
                    continue
                Df_try = Db_try = min(_cap, (D_use - Dm_try) / 2)
                Dm_eff = Dm_try
                _mid_d = Dm_eff - HALL_W * 2
                if n_m_try > 0:
                    _base = _mid_d / n_m_try
                    if _base < 3.0 or _base * Wm_eff < min_area_dpto * 1.05 - 1e-6:
                        continue
                _bite = _bite_blk if n_m_try > 0 else _pozo_loss
                nf = _block_cap_d(W, Df_try, _bite)   # muesca de franja / junta derecha
                nb = _block_cap_d(W, Db_try, _bite)
                tot = min(num_dptos, nf + nb + n_m_try)
                # max dptos logrados; a igualdad, mínimas filas medias (corredor más corto)
                key = (tot, -n_m_try)
                if best is None or key > best[0]:
                    best = (key, n_m_try, Df_try, Db_try, nf, nb, Dm_try)
            if best is not None and best[0][0] >= num_dptos:
                break
        if best is None:
            return None, None
        _, n_m_sel, Df, Db, n_f_cap, n_b_cap, Dm = best
    else:
        db_exists = False
        Dm_min = max(Lc, HALL_W + 3.2)
        Df = min(DF_CAL, D_use - Dm_min)   # F6: crujía calibrada
        Dm = Dm_min
        Db = 0.0

    vm0 = ya + Df          # inicio zona media (línea de hall)
    vm1 = vm0 + Dm         # fin zona media (línea de bloque fondo)
    y_end = vm1 + Db       # fin de edificación (= yb cuando D_use = D_avail)

    # ── Geometría del núcleo ──
    if nucleo_posterior:
        # Fila horizontal al fondo: escalera (4.0×2.5) + ascensores en línea
        _row_v0 = vm0 + HALL_W
        stair_poly = Polygon([
            (xa, _row_v0), (xa + ESC_L, _row_v0),
            (xa + ESC_L, _row_v0 + CORE_ROW_D), (xa, _row_v0 + CORE_ROW_D),
        ])
        asc_polys = []
        for i in range(num_asc):
            a0 = xa + ESC_L + 0.20 + i * (asc_l + 0.20)
            asc_polys.append(Polygon([
                (a0, _row_v0), (a0 + asc_l, _row_v0),
                (a0 + asc_l, _row_v0 + 2.00), (a0, _row_v0 + 2.00),
            ]))
        _row_len = ESC_L + (num_asc * (asc_l + 0.20) if num_asc > 0 else 0.0)
        core_col = Polygon([(xa, _row_v0), (xa + min(_row_len + 0.5, W), _row_v0),
                            (xa + min(_row_len + 0.5, W), _row_v0 + CORE_ROW_D),
                            (xa, _row_v0 + CORE_ROW_D)])
        core_clipped = safe_clip(core_col, lote)
        vest_poly = None
        if nec_esc_prot:
            vest_poly = Polygon([
                (xa, _row_v0), (xa + 1.50, _row_v0),
                (xa + 1.50, _row_v0 + 1.50), (xa, _row_v0 + 1.50),
            ])
        # Hall: banda completa entre bloque frente y fila de núcleo
        hall_parts = [Polygon([(xa, vm0), (xb, vm0),
                               (xb, vm0 + HALL_W), (xa, vm0 + HALL_W)])]
    else:
        stair_poly = Polygon([
            (xa, vm0), (xa + CORE_W, vm0),
            (xa + CORE_W, vm0 + ESC_L), (xa, vm0 + ESC_L),
        ])
        asc_polys = []
        for i in range(num_asc):
            a0 = vm0 + ESC_L + 0.20 + i * (asc_l + 0.20)
            # Cabina alineada al borde del corredor (puerta con acceso directo)
            asc_polys.append(Polygon([
                (xa + CORE_W - 2.00, a0), (xa + CORE_W, a0),
                (xa + CORE_W, a0 + asc_l), (xa + CORE_W - 2.00, a0 + asc_l),
            ]))
        core_col = Polygon([(xa, vm0), (xa + CORE_W, vm0),
                            (xa + CORE_W, vm1), (xa, vm1)])
        core_clipped = safe_clip(core_col, lote)
        vest_poly = None
        if nec_esc_prot:
            vest_poly = Polygon([
                (xa + CORE_W - 1.50, vm0), (xa + CORE_W, vm0),
                (xa + CORE_W, vm0 + 1.50), (xa + CORE_W - 1.50, vm0 + 1.50),
            ])

        # ── Hall compacto + corredor + hall fondo ──
        # El hall remata antes del pozo (franja lateral o pozo de junta)
        _pz_lado = pz_franja if (db_exists and n_m_sel > 0) else pw
        hall_x1 = xb - _pz_lado - 0.10
        hall_parts = [Polygon([(xa + CORE_W, vm0), (hall_x1, vm0),
                               (hall_x1, vm0 + HALL_W), (xa + CORE_W, vm0 + HALL_W)])]
        if db_exists:
            hall_parts.append(Polygon([  # corredor al fondo
                (xa + CORE_W, vm0 + HALL_W), (xa + CORE_W + CORR_W, vm0 + HALL_W),
                (xa + CORE_W + CORR_W, vm1), (xa + CORE_W, vm1),
            ]))
            # Hall fondo: cubre todas las puertas del bloque fondo hasta la franja
            _hf_x1 = hall_x1
            hall_parts.append(Polygon([
                (xa + CORE_W, vm1 - HALL_W), (_hf_x1, vm1 - HALL_W),
                (_hf_x1, vm1), (xa + CORE_W, vm1),
            ]))
        else:
            # Sin bloque fondo: stub de corredor frente al núcleo para dar
            # acceso a la puerta del ascensor (queda más allá del hall)
            hall_parts.append(Polygon([
                (xa + CORE_W, vm0 + HALL_W), (xa + CORE_W + CORR_W, vm0 + HALL_W),
                (xa + CORE_W + CORR_W, vm0 + Lc), (xa + CORE_W, vm0 + Lc),
            ]))
    hall_clipped = safe_clip(unary_union(hall_parts), lote)
    hall_buf = hall_clipped.buffer(0.40) if hall_clipped and not hall_clipped.is_empty else None

    # ── Distribución de unidades ──
    units_spec = []  # (corners p0..p3 [p0p1=lado puerta], lado, fachada_ext)

    def _split_u(u0, u1, n, extra_last=0.0, extra_first=0.0):
        """Divide [u0,u1] en n tramos. Las unidades de borde junto a pozos
        reciben ancho extra (compensan la mordida) y las demás se reducen
        proporcionalmente — todas terminan con área útil similar."""
        span = u1 - u0
        base = (span - extra_last - extra_first) / n
        outs = []
        cur = u0
        for i in range(n):
            w = base + (extra_first if i == 0 else 0.0) + (extra_last if i == n - 1 else 0.0)
            outs.append((cur, cur + w))
            cur += w
        return outs

    # Capacidades: para db_exists vienen de la búsqueda; resto se calcula aquí
    mid_v0 = vm0 + HALL_W
    mid_v1 = (vm1 - HALL_W) if db_exists else vm1
    mid_depth = max(0.0, mid_v1 - mid_v0)
    # stub/corredor frente al núcleo existe en ambos casos (acceso ascensor)
    xm0 = xa + (0.0 if nucleo_posterior else CORE_W + CORR_W)
    Wm = xb - xm0
    if db_exists:
        n_m_cap = n_m_sel
    else:
        n_f_cap = _block_cap_d(W, Df, 1)
        n_b_cap = 0
        # unidades lado a lado en u, puerta al hall (borde frontal)
        # 2×_pozo_loss: typically 2 junction pozos (vm0 + y_end) — conservative
        n_m_cap = min(int(Wm / 3.5),
                      int((Wm * mid_depth - 2 * _pozo_loss) / (min_area_dpto * 1.05))) if mid_depth >= 3.2 else 0
        n_m_cap = max(0, min(4, n_m_cap))

    # Asignación exacta de num_dptos dentro de la capacidad.
    # Frente se mantiene premium (unidades más grandes): el excedente
    # se reparte primero a intermedio y fondo.
    n_total = max(1, min(num_dptos, n_f_cap + n_b_cap + n_m_cap))
    n_f = 1
    n_b = 1 if n_b_cap > 0 else 0
    n_m = 0
    _rem = n_total - n_f - n_b
    _ciclo = ("m", "b", "f")
    _i = 0
    _stall = 0
    while _rem > 0 and _stall < 3:
        _k = _ciclo[_i % 3]
        _i += 1
        if _k == "m" and n_m < n_m_cap:
            n_m += 1; _rem -= 1; _stall = 0
        elif _k == "b" and 0 < n_b < n_b_cap:
            n_b += 1; _rem -= 1; _stall = 0
        elif _k == "f" and n_f < n_f_cap:
            n_f += 1; _rem -= 1; _stall = 0
        else:
            _stall += 1

    # ¿Pozo franja lateral? (hay filas intermedias en esquema con bloque fondo)
    usa_franja = db_exists and n_m > 0

    # Bloque frente: fachada a calle, puerta al hall (borde posterior).
    # La unidad derecha compensa la mordida del pozo/franja de la junta frente.
    _xf = ((_bite_blk if usa_franja else _pozo_loss) / Df) if (n_f > 1 and Df > 0) else 0.0
    for (u0, u1) in _split_u(xa, xb, n_f, extra_last=_xf):
        units_spec.append({
            "corners": ((u0, vm0), (u1, vm0), (u1, ya), (u0, ya)),
            "lado": "frente", "fachada": bool(proyecto.frente_exterior),
        })
    # Zona media: filas uniformes que rematan en la franja lateral
    # (wet core a la puerta/corredor, dormitorios asomados al pozo — patrón real)
    if n_m > 0 and db_exists:
        _xm1 = xb - pz_franja  # borde derecho de filas = inicio de franja
        _base_h = mid_depth / n_m
        r0 = mid_v0
        for j in range(n_m):
            r1 = r0 + _base_h
            units_spec.append({
                "corners": ((xm0, r0), (xm0, r1), (_xm1, r1), (_xm1, r0)),
                "lado": "intermedio", "fachada": False,
            })
            r0 = r1
    elif n_m > 0:
        _xm = (_pozo_loss / max(mid_depth, 1.0)) if n_m > 1 else 0.0
        for (u0, u1) in _split_u(xm0, xb, n_m, extra_last=_xm):
            units_spec.append({
                "corners": ((u0, mid_v0), (u1, mid_v0), (u1, mid_v1), (u0, mid_v1)),
                "lado": "intermedio", "fachada": False,
            })
    # Bloque fondo: puerta al hall fondo, fachada al patio posterior si existe.
    patio_depth = yb - y_end
    _xbd = ((_bite_blk if usa_franja else _pozo_loss) / Db) if (n_b > 1 and Db > 0) else 0.0
    _xbi = (_pozo_loss / Db) if (n_b > 1 and Db > 0 and db_exists and W >= 10.0 and not usa_franja) else 0.0
    for (u0, u1) in (_split_u(xa, xb, n_b, extra_last=_xbd, extra_first=_xbi) if n_b > 0 else []):
        units_spec.append({
            "corners": ((u0, vm1), (u1, vm1), (u1, y_end), (u0, y_end)),
            "lado": "fondo",
            "fachada": bool(proyecto.fondo_exterior) or patio_depth >= 2.5,
        })

    # ── Pozos de luz ──
    pozos_shared: List[Polygon] = []
    pozos_cumple: List[bool] = []
    if usa_franja:
        # Franja lateral continua en medianera derecha a lo largo de la zona
        # media, con muescas sobre los bloques frente/fondo (patrón ref. real:
        # área de pozo en proyecto ≈ 2× la normativa).
        pozos_shared.append(Polygon([
            (xb - pz_franja, vm0 - _solape), (xb, vm0 - _solape),
            (xb, vm1 + _solape), (xb - pz_franja, vm1 + _solape),
        ]))
        pozos_cumple.append(franja_conf)
    else:
        # Pozos puntuales a caballo de cada junta entre bloques
        juntas = [vm0]
        if db_exists:
            juntas.append(vm1)
        elif (not nucleo_posterior) and patio_depth < 1.0:
            juntas.append(y_end - pv / 2)  # esquina posterior si no hay patio
        for jv in juntas:
            pozos_shared.append(Polygon([
                (xb - pw, jv - pv / 2), (xb, jv - pv / 2),
                (xb, jv + pv / 2), (xb - pw, jv + pv / 2),
            ]))
            pozos_cumple.append(pozo_conf)
        if db_exists and W >= 10.0:
            # pozo en medianera izquierda en la junta media/fondo
            pozos_shared.append(Polygon([
                (xa, vm1 - pv / 2), (xa + pw, vm1 - pv / 2),
                (xa + pw, vm1 + pv / 2), (xa, vm1 + pv / 2),
            ]))
            pozos_cumple.append(pozo_conf)
        if db_exists and n_m == 0 and mid_depth > 1.2:
            # Patio de luz interior: la zona media sin filas se convierte en
            # patio central (elimina espacio muerto, ventila los tres bloques)
            _mid_b = (xa + xb) / 2 + 1.5
            _patio_int = unary_union([
                Polygon([(xm0, mid_v0), (xb, mid_v0), (xb, mid_v1), (xm0, mid_v1)]),
                Polygon([(_mid_b, mid_v1), (xb, mid_v1), (xb, vm1), (_mid_b, vm1)]),
            ])
            pozos_shared.append(_patio_int)
            pozos_cumple.append(
                (min(xb - xm0, vm1 - mid_v0) + 1e-6) >= POZO_REQ
            )

    pozos_clip = [safe_clip(p, lote) for p in pozos_shared]
    pozos_final, cumple_final = [], []
    for p, ok in zip(pozos_clip, pozos_cumple):
        if p is not None and not p.is_empty and p.area > 0.5:
            pozos_final.append(p)
            cumple_final.append(ok)
    pozos_union = unary_union(pozos_final).buffer(0.0) if pozos_final else None
    if pozos_union is not None and hall_clipped is not None:
        try:
            _h = hall_clipped.difference(pozos_union)
            if hasattr(_h, "geoms"):
                _h = max(_h.geoms, key=lambda g: g.area)
            if not _h.is_empty:
                hall_clipped = _h
                hall_buf = hall_clipped.buffer(0.40)
        except Exception:
            pass

    # ── Construcción + validación de unidades ──
    ductos: List[Polygon] = []
    departamentos_detalle: List[Dict[str, Any]] = []
    sin_acceso = 0
    for spec in units_spec:
        corners = spec["corners"]
        ap = safe_clip(Polygon(corners), lote)
        if ap is None or ap.is_empty:
            continue
        if pozos_union is not None and ap.intersects(pozos_union):
            diff = ap.difference(pozos_union)
            if hasattr(diff, "geoms"):
                diff = max(diff.geoms, key=lambda g: g.area)
            if not diff.is_empty and diff.area >= min_area_dpto * 0.35:
                ap = diff
        if hall_clipped is not None:
            ap = ap.difference(hall_clipped.buffer(0.0))
            if hasattr(ap, "geoms"):
                ap = max(ap.geoms, key=lambda g: g.area)
        if ap.is_empty or ap.area < min_area_dpto * 0.85:
            continue
        # Garantía de acceso: la unidad debe tener vano de puerta real al
        # hall/corredor (>=MIN_PUERTA_M), no solo tocar una esquina.
        if hall_clipped is not None and not _door_access(ap, hall_clipped):
            sin_acceso += 1
            continue
        area_gross = float(ap.area)
        area_m = area_neta_muros(ap)
        typ = get_typology(area_m)
        zonas_geom = generate_interior_zones(corners, typ, ap, lote)
        val_u = validar_unidad(
            unit=ap, zonas=zonas_geom,
            escalera=stair_poly, hall_buf=hall_buf, ductos=ductos,
            pozos=[p for p, ok in zip(pozos_final, cumple_final) if ok],
        )
        val_u["fachada_exterior"] = bool(spec["fachada"])
        val_u["distancia_escalera_m"] = val_u.get("distancia_evac_m", 0.0)
        val_u["dist_esc_cumple"] = val_u.get("evac_cumple", True)
        zonas_payload = [{
            "nombre": z["nombre"], "kind": z.get("kind", ""),
            "coords": poly_to_js(z["geom"]),
            "area_m2": r3(float(z["geom"].area)),
            "validacion": z.get("validacion", {}),
        } for z in zonas_geom]
        departamentos_detalle.append({
            "contorno": poly_to_js(ap),
            "tipologia": typ,
            "area_m2": r3(area_m),
            "area_gross_m2": r3(area_gross),
            "lado": spec["lado"],
            "es_reducida": bool(_rect_min_side(ap) < 5.2 or area_m < min_area_dpto * 1.05),
            "zonas": zonas_payload,
            "validacion": val_u,
        })

    if len(departamentos_detalle) < 1:
        return None, None

    # ── Patio posterior (área libre) ──
    patio_poly = None
    if patio_depth > 0.30:
        patio_poly = safe_clip(Polygon([
            (xa, y_end), (xb, y_end), (xb, yb), (xa, yb),
        ]), lote)

    # ── Columnas en intersecciones de ejes de bloques ──
    COL_H = 0.25
    u_lines = sorted({xa, xb, xm0, xa + CORE_W} |
                     {c[0] for s in units_spec for c in s["corners"]})
    v_lines = sorted({ya, vm0, mid_v0, mid_v1, vm1, y_end})
    columnas = []
    avoid = pozos_union.buffer(0.05) if pozos_union is not None else None
    for ux in u_lines:
        for vy in v_lines:
            cpol = Polygon([(ux - COL_H, vy - COL_H), (ux + COL_H, vy - COL_H),
                            (ux + COL_H, vy + COL_H), (ux - COL_H, vy + COL_H)])
            cc = safe_clip(cpol, lote)
            if cc is None or cc.is_empty or cc.area < 0.06:
                continue
            if avoid is not None and cc.intersects(avoid):
                continue
            columnas.append(cc)

    cap_total = n_f_cap + n_b_cap + n_m_cap
    topo_info = informe_topologia(lote, float(proyecto.frente or 0.0))
    topo_info["seleccion"]["recomendada"] = "hall_compacto"

    geometry = {
        "hall": poly_to_js(hall_clipped),
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote)),
        "ascensores": [poly_to_js(safe_clip(a, lote)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote)) if vest_poly else [],
        "patio": poly_to_js(patio_poly) if patio_poly else [],
        "ductos": [poly_to_js(d) for d in ductos],
        "pozos_luz": [poly_to_js(p) for p in pozos_final],
        "pozos_luz_cumple": list(cumple_final),
        "columnas": [poly_to_js(c) for c in columnas],
        "esquema_area_libre": "hall_compacto",
        "departamentos": departamentos_detalle,
        "cabida_multifamiliar": {
            "contexto": "Perú — hall compacto: núcleo lateral + bloques frente/medio/fondo (patrón Lima entre medianeras)",
            "area_min_dpto_m2": min_area_dpto,
            "profundidad_strip_estimada_m": r3(Df),
            "departamentos_solicitados_planta": num_dptos,
            "capacidad_maxima_estimada_planta": cap_total,
            "capacidad_lado_nucleo": n_f_cap + n_m_cap,
            "capacidad_lado_patio": n_b_cap,
            "departamentos_generados_planta": len(departamentos_detalle),
            "nota": "Hall compacto + corredor corto; dptos frente/fondo a todo el ancho, intermedios ventilan a pozos en medianera.",
        },
        "topologia": topo_info,
    }
    _n_ok = sum(1 for ok in cumple_final if ok)
    normativa = {
        "pozo_final": r3(pozo_final),
        "pozos_luz_check": {
            "dimension_requerida_m": r3(POZO_REQ),
            "colocados": len(pozos_final),
            "conformes": _n_ok,
            "no_conformes": len(pozos_final) - _n_ok,
            "ok": (len(pozos_final) - _n_ok) == 0,
            "nota": "Pozo de luz mínimo H/4 = {:.2f}m en medianeras, a caballo de juntas entre bloques.".format(POZO_REQ),
        },
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": min_area_dpto,
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_dptos,
            "departamentos_emitidos": len(departamentos_detalle),
            "capacidad_max_estimada_planta": cap_total,
            "descartados_sin_acceso": sin_acceso,
        },
        "topologia": topo_info,
    }
    return geometry, normativa


# ═══════════════════════════════════════════════════════════════
# E5 — EVALUACIÓN INTERNA (self-critique)
# ═══════════════════════════════════════════════════════════════

def _evaluar_diseno(geometry: dict, lote_poly: Polygon, r_lat: float, r_pos: float,
                    num_dptos: int, pozo_req: float = 0.0) -> dict:
    """Evalúa el diseño generado con las mismas métricas que el arnés.
    Retorna {"score": 0-100, "defectos": [...], "metricas": {...}}.
    'defectos' = lista de dicts con {tipo, severidad, descripcion}.
    """
    defectos: list = []
    metricas: dict = {}

    def _geom_polys(key: str) -> list:
        """Polígonos para 'key' (singular) + variante plural 'keys' (lista) —
        topologías multi-núcleo (p.ej. costillas_dos_nucleos) emiten varias
        instancias del mismo elemento (halls/cores/escaleras)."""
        polys: list = []
        for pts in [geometry.get(key, [])] + list(geometry.get(key + "s", [])):
            if pts and len(pts) >= 3:
                try:
                    p = Polygon([(c["x"], c["y"]) for c in pts])
                    if not p.is_empty:
                        polys.append(p)
                except Exception:
                    pass
        return polys

    # ── Footprint ──────────────────────────────────────────────
    fp_parts: list = []
    # "patio"/"patio_central" cuentan como área diseñada (área libre intencional, no hueco)
    for key in ("hall", "core", "vestibulo", "escalera", "patio", "patio_central"):
        fp_parts.extend(_geom_polys(key))
    for grp in ("corridors", "ascensores"):
        for item in geometry.get(grp, []):
            if item and len(item) >= 3:
                try:
                    p = Polygon([(c["x"], c["y"]) for c in item])
                    if not p.is_empty:
                        fp_parts.append(p)
                except Exception:
                    pass
    for r in geometry.get("remanentes_zona_media", []):
        if r and len(r) >= 3:
            try:
                p = Polygon([(c["x"], c["y"]) for c in r])
                if not p.is_empty:
                    fp_parts.append(p)
            except Exception:
                pass
    for dpto in geometry.get("departamentos", []):
        pts = _departamento_outline_coords(dpto)
        if pts and len(pts) >= 3:
            try:
                p = Polygon([(c["x"], c["y"]) for c in pts])
                if not p.is_empty:
                    fp_parts.append(p)
            except Exception:
                pass
    footprint: Optional[Polygon] = None
    if fp_parts:
        try:
            u = unary_union(fp_parts)
            if hasattr(u, "geoms"):
                u = max(u.geoms, key=lambda g: g.area)
            footprint = u if (u and not u.is_empty) else None
        except Exception:
            pass

    # ── Lote útil ──────────────────────────────────────────────
    lote_util = _erode_lote(lote_poly, r_lat, r_pos) or lote_poly

    # ── nd_delta ───────────────────────────────────────────────
    nd_emit = len(geometry.get("departamentos", []))
    nd_delta = abs(nd_emit - num_dptos)
    metricas["nd_emitidos"] = nd_emit
    metricas["nd_pedidos"] = num_dptos
    metricas["nd_delta"] = nd_delta
    if nd_delta > 2:
        defectos.append({"tipo": "nd_delta", "severidad": "critico",
                         "descripcion": f"Δ{nd_delta}: emitidos {nd_emit}/{num_dptos}"})
    elif nd_delta > 1:
        defectos.append({"tipo": "nd_delta", "severidad": "menor",
                         "descripcion": f"Δ{nd_delta}: emitidos {nd_emit}/{num_dptos}"})

    # ── Huecos ─────────────────────────────────────────────────
    huecos_sin_pz = 0.0
    if footprint is not None:
        try:
            diff = lote_util.difference(footprint.buffer(0.05))
            total_h = float(diff.area)
            pz_area = 0.0
            for p_pts in geometry.get("pozos_luz", []):
                if p_pts and len(p_pts) >= 3:
                    try:
                        pp = Polygon([(c["x"], c["y"]) for c in p_pts])
                        pz_area += diff.intersection(pp.buffer(0.1)).area
                    except Exception:
                        pass
            huecos_sin_pz = max(0.0, total_h - pz_area)
        except Exception:
            huecos_sin_pz = 0.0
    metricas["huecos_sin_pz"] = round(huecos_sin_pz, 2)
    if huecos_sin_pz > lote_util.area * 0.05:
        defectos.append({"tipo": "huecos", "severidad": "critico",
                         "descripcion": f"Huecos {huecos_sin_pz:.1f}m² (>{lote_util.area*0.05:.1f}m²=5% edificable)"})
    elif huecos_sin_pz > 10.0:
        defectos.append({"tipo": "huecos", "severidad": "menor",
                         "descripcion": f"Huecos {huecos_sin_pz:.1f}m²"})

    # ── pct_circ ───────────────────────────────────────────────
    pct_circ = 0.0
    if footprint is not None and footprint.area > 0:
        circ_pts: list = []
        for key in ("hall", "vestibulo"):
            circ_pts.extend(_geom_polys(key))
        for item in geometry.get("corridors", []):
            if item and len(item) >= 3:
                try:
                    p = Polygon([(c["x"], c["y"]) for c in item])
                    if not p.is_empty:
                        circ_pts.append(p)
                except Exception:
                    pass
        if circ_pts:
            try:
                circ_area = float(unary_union(circ_pts).area)
                pct_circ = round(circ_area / footprint.area * 100, 1)
            except Exception:
                pass
    metricas["pct_circ"] = pct_circ
    if pct_circ > 15.0:
        defectos.append({"tipo": "pct_circ", "severidad": "critico",
                         "descripcion": f"Circulación {pct_circ}% > 15%"})
    elif pct_circ > 12.0:
        defectos.append({"tipo": "pct_circ", "severidad": "menor",
                         "descripcion": f"Circulación {pct_circ}% > 12%"})

    # ── Eficiencia ─────────────────────────────────────────────
    # denominador = área techada: el patio (área libre diseñada) integra el
    # footprint para huecos/retiros pero no es construido — se descuenta
    efi = 0.0
    if footprint is not None and footprint.area > 0:
        vendible = 0.0
        for dpto in geometry.get("departamentos", []):
            area = dpto.get("area_m2") if isinstance(dpto, dict) else 0.0
            if area:
                vendible += float(area)
        techada = footprint
        patio_pts = geometry.get("patio", [])
        if patio_pts and len(patio_pts) >= 3:
            try:
                techada = footprint.difference(
                    Polygon([(c["x"], c["y"]) for c in patio_pts]).buffer(0))
            except Exception:
                pass
        if techada.area > 0:
            efi = round(vendible / techada.area * 100, 1)
    metricas["eficiencia"] = efi
    if efi < 60.0 and efi > 0:
        defectos.append({"tipo": "eficiencia", "severidad": "critico",
                         "descripcion": f"Eficiencia {efi}% < 60%"})
    elif efi < 70.0 and efi > 0:
        defectos.append({"tipo": "eficiencia", "severidad": "menor",
                         "descripcion": f"Eficiencia {efi}% < 70%"})

    # ── Acceso ─────────────────────────────────────────────────
    dptos = geometry.get("departamentos", [])
    acc_ok = 0
    circ_polys: list = _geom_polys("hall")
    for item in geometry.get("corridors", []):
        if item and len(item) >= 3:
            try:
                p = Polygon([(c["x"], c["y"]) for c in item])
                circ_polys.append(p)
            except Exception:
                pass
    circ_union = unary_union(circ_polys) if circ_polys else None
    for dpto in dptos:
        pts = _departamento_outline_coords(dpto)
        if not pts or len(pts) < 3:
            continue
        try:
            dp = Polygon([(c["x"], c["y"]) for c in pts])
            if circ_union and _door_access(dp, circ_union):
                acc_ok += 1
        except Exception:
            pass
    acc_pct = round(acc_ok / max(len(dptos), 1) * 100, 1)
    metricas["acceso_pct"] = acc_pct
    metricas["acceso_ok"] = acc_ok
    metricas["acceso_total"] = len(dptos)
    if acc_pct < 100.0 and len(dptos) > 0:
        sevv = "critico" if acc_pct < 80.0 else "menor"
        defectos.append({"tipo": "acceso", "severidad": sevv,
                         "descripcion": f"Acceso {acc_ok}/{len(dptos)} dptos ({acc_pct}%)"})

    # ── Pozos de luz: lado ≥ H/3 si sirven dormitorios (RNE A.010) ──
    pozos_luz = geometry.get("pozos_luz", [])
    pozos_bajo_norma = 0
    for pz in pozos_luz:
        if not pz or len(pz) < 3:
            continue
        try:
            pxs = [c["x"] if isinstance(c, dict) else c[0] for c in pz]
            pys = [c["y"] if isinstance(c, dict) else c[1] for c in pz]
            lado_min = min(max(pxs) - min(pxs), max(pys) - min(pys))
            if pozo_req > 0 and lado_min + 1e-6 < pozo_req:
                pozos_bajo_norma += 1
        except Exception:
            pass
    metricas["pozos_bajo_norma"] = pozos_bajo_norma
    metricas["pozo_req_m"] = round(pozo_req, 2)
    if pozos_bajo_norma > 0:
        defectos.append({"tipo": "pozo_luz", "severidad": "menor",
                         "descripcion": f"{pozos_bajo_norma} pozo(s) < {pozo_req:.2f}m "
                                        f"(RNE H/3 para dormitorios) — lote no da el ancho requerido"})

    # ── Frente real (calibración DXF Lima, REGLAS_DISENO.md: mín 5.2m) ──
    frente_bajo = 0
    for dpto in dptos:
        pts = _departamento_outline_coords(dpto)
        if not pts or len(pts) < 3:
            continue
        try:
            dp = Polygon([(c["x"], c["y"]) for c in pts])
            if _rect_min_side(dp) + 1e-6 < 5.2:
                frente_bajo += 1
        except Exception:
            pass
    metricas["frente_bajo_norma"] = frente_bajo
    if frente_bajo > 0:
        # Severidad "menor" siempre: el lado mínimo del rectángulo envolvente
        # puede salir chico por una muesca de pozo o un lote trapezoidal sin
        # que la unidad sea realmente angosta (falso positivo geométrico) —
        # se reporta para revisión, no bloquea el score/checklist.
        defectos.append({"tipo": "frente_angosto", "severidad": "menor",
                         "descripcion": f"{frente_bajo} dpto(s) con frente < 5.2m "
                                        f"(calibración real Lima; revisar si es muesca de pozo o dpto tubo real)"})

    # ── Score ──────────────────────────────────────────────────
    s = 0
    # nd_delta: 25 pts  (Δ0=25, Δ1=20, Δ2=12 crédito parcial, Δ≥3=0)
    s += 25 if nd_delta == 0 else (20 if nd_delta == 1 else (12 if nd_delta == 2 else 0))
    # eficiencia: 20 pts
    s += 20 if efi >= 80 else (15 if efi >= 70 else (8 if efi >= 60 else 0))
    # pct_circ: 20 pts  (≤10=20, ≤12=18, >12=0)
    s += 20 if pct_circ <= 10 else (18 if pct_circ <= 12 else 0)
    # huecos: 20 pts
    s += 20 if huecos_sin_pz <= 1 else (15 if huecos_sin_pz <= 5 else (8 if huecos_sin_pz <= 10 else 0))
    # acceso: 10 pts
    s += 10 if acc_pct >= 100 else (7 if acc_pct >= 80 else 0)
    # eficiencia y acceso combos extras: 5 pts bonus para compensar
    s += 5 if (nd_delta <= 1 and acc_pct >= 100 and pct_circ <= 12) else 0

    score = min(100, s)
    metricas["score"] = score

    n_criticos = sum(1 for d in defectos if d["severidad"] == "critico")
    return {
        "score": score,
        "defectos": defectos,
        "n_criticos": n_criticos,
        "metricas": metricas,
    }


# ═══════════════════════════════════════════════════════════════
# GENERACIÓN DE GEOMETRÍA (núcleo, idéntico al anterior)
# ═══════════════════════════════════════════════════════════════

# Variables que solo el generador "spine" (fallback genérico) implementa
# de verdad; costillas/hall_compacto/claustro/tower las ignoran en silencio.
_HONRA_MIX = {"spine"}
_HONRA_ESQUEMA = {"spine"}
_HONRA_PRECIOS = {"spine"}
_HONRA_OPTIMIZAR = {"spine", "claustro", "tower"}


def _variables_ignoradas(proyecto: ProyectoInmobiliario, topo_usada: str) -> list:
    """Avisa qué variables enviadas por el usuario no tienen efecto en la
    topología efectivamente usada, en vez de ignorarlas en silencio."""
    avisos = []
    if proyecto.mix_tipologias and topo_usada not in _HONRA_MIX:
        avisos.append(f"mix_tipologias: sin efecto en topología '{topo_usada}' (solo aplica en spine)")
    if proyecto.esquema_area_libre and proyecto.esquema_area_libre != "muros_ciegos" and topo_usada not in _HONRA_ESQUEMA:
        avisos.append(f"esquema_area_libre='{proyecto.esquema_area_libre}': sin efecto en topología '{topo_usada}' (solo aplica en spine)")
    if proyecto.optimizar_densidad and topo_usada not in _HONRA_OPTIMIZAR:
        avisos.append(f"optimizar_densidad: sin efecto en topología '{topo_usada}' (solo aplica en spine/claustro/tower)")
    if proyecto.precios_tipologia and topo_usada not in _HONRA_PRECIOS:
        avisos.append(f"precios_tipologia: sin efecto en topología '{topo_usada}' (solo aplica en spine)")
    if proyecto.ciego_frente:
        avisos.append("ciego_frente: sin implementación en el motor, no afecta ningún resultado")
    return avisos


def _generate_geometry(proyecto: ProyectoInmobiliario):
    """Core geometry generation — shared between audit and render endpoints."""
    lote = Polygon(proyecto.coordenadas_lote)
    if not lote.is_valid:
        lote = lote.buffer(0)

    h_edif = proyecto.numero_pisos * (proyecto.altura_piso or RNE["altura_piso"])
    num_dptos = max(2, proyecto.num_departamentos)
    num_asc = max(0, proyecto.num_ascensores)

    pozo_final = max(RNE["pozos_luz"]["min_abs"], h_edif * RNE["pozos_luz"]["ratio_dorm"])
    nec_ascensor = h_edif > RNE["circulacion_v"]["h_max_sin_ascensor"]
    nec_esc_prot = h_edif > RNE["circulacion_v"]["h_max_sin_esc_prot"]

    # RNE: ascensor obligatorio sobre h_max_sin_ascensor — forzar mínimo 1
    ascensor_forzado = False
    if nec_ascensor and num_asc == 0:
        num_asc = 1
        ascensor_forzado = True

    # Para lotes L-shape, spine se orienta al ala mayor (no al bbox del lote
    # completo). En lotes rectangulares, fallback al rectángulo mínimo rotado.
    main_rect = find_main_rect(lote)
    if main_rect is not None:
        mrr = main_rect
        spine_centroid = main_rect.centroid
    else:
        mrr = lote.minimum_rotated_rectangle
        spine_centroid = lote.centroid
    mc = list(mrr.exterior.coords)
    d01 = math.hypot(mc[1][0] - mc[0][0], mc[1][1] - mc[0][1])
    d12 = math.hypot(mc[2][0] - mc[1][0], mc[2][1] - mc[1][1])

    if d01 >= d12:
        long_len, short_len = d01, d12
        ang = math.atan2(mc[1][1] - mc[0][1], mc[1][0] - mc[0][0])
    else:
        long_len, short_len = d12, d01
        ang = math.atan2(mc[2][1] - mc[1][1], mc[2][0] - mc[1][0])

    cx, cy = spine_centroid.x, spine_centroid.y
    dl_x, dl_y = math.cos(ang), math.sin(ang)
    ds_x, ds_y = -dl_y, dl_x
    # Normalize dl so sign_s=-1 always faces frente (small y = street side).
    # Horizontal (|dl_x|>|dl_y|): ensure dl_x>0 so +ds points toward fondo (large y).
    # Vertical (|dl_y|>=|dl_x|): ensure dl_y>0 so L_min (izquierda end) = frente end.
    if abs(dl_x) > abs(dl_y):
        if dl_x < 0:
            dl_x, dl_y = -dl_x, -dl_y
            ds_x, ds_y = -ds_x, -ds_y
    elif dl_y < 0:
        dl_x, dl_y = -dl_x, -dl_y
        ds_x, ds_y = -ds_x, -ds_y

    half_L = long_len / 2
    half_S = short_len / 2
    hw = RNE["circulacion_h"]["hall_ancho"] / 2

    # ── Topología: branch a claustro si aplica ──
    _topo_m = analizar_lote(lote, float(proyecto.frente or 0.0))
    _topo_s = seleccionar_topologia(_topo_m)

    # P1 — esquinero: núcleo hacia medianera interior, esquina-calle
    # premium. Sin dato explícito de qué arista es calle en coordenadas_lote,
    # se infiere del vértice reflex (esquina interior del L): el núcleo se
    # desplaza hacia ahí, agrandando el cuadrante opuesto (la esquina
    # exterior hacia calle) para el dpto premium. Aplica antes de armar
    # _args para que TODAS las topologías (spine/tower/etc.) hereden el bias.
    if _topo_m.get("es_esquinero"):
        _rv = _reflex_vertex(lote)
        if _rv is not None:
            _vx, _vy = _rv[0] - cx, _rv[1] - cy
            _b_dl = _vx * dl_x + _vy * dl_y
            _b_ds = _vx * ds_x + _vy * ds_y
            _bn = math.hypot(_b_dl, _b_ds)
            if _bn > 1e-6:
                _b_dl, _b_ds = _b_dl / _bn, _b_ds / _bn
                _BIAS = 0.30
                cx = cx + dl_x * (_b_dl * half_L * _BIAS) + ds_x * (_b_ds * half_S * _BIAS)
                cy = cy + dl_y * (_b_dl * half_L * _BIAS) + ds_y * (_b_ds * half_S * _BIAS)

    _args = (proyecto, lote, cx, cy, dl_x, dl_y, ds_x, ds_y,
             half_L, half_S, hw,
             max(2, proyecto.num_departamentos), num_asc,
             nec_esc_prot, nec_ascensor, pozo_final, h_edif)
    # ── E5: evaluador interno — se invoca sobre el resultado antes de retornar ──
    _r_lat_ev = float(proyecto.retiro_lateral or 0.0)
    _r_pos_ev = float(proyecto.retiro_posterior or 0.0)
    _nd_req = max(2, proyecto.num_departamentos)

    def _emit(g, n, topo_usada):
        """Adjunta evaluación E5 + avisos de variables ignoradas y retorna."""
        n["ascensor_forzado"] = ascensor_forzado
        n["variables_ignoradas"] = _variables_ignoradas(proyecto, topo_usada)
        # Topologías de núcleo único no arman "nucleos" -- sintetizar una
        # lista de un elemento para que primer_piso/sótano/azotea puedan
        # iterar sobre "nucleos" sin importar si hay 1 o 2 torres.
        if "nucleos" not in g:
            g["nucleos"] = [{
                "hall": g.get("hall", []), "escalera": g.get("escalera", []),
                "ascensores": g.get("ascensores", []), "vestibulo": g.get("vestibulo", []),
                "core": g.get("core", []),
            }]
        try:
            g["evaluacion"] = _evaluar_diseno(g, lote, _r_lat_ev, _r_pos_ev, _nd_req, pozo_final)
        except Exception:
            g["evaluacion"] = {"score": 0, "defectos": [], "n_criticos": 0, "metricas": {}}
        return g, n

    if _topo_s["recomendada"] == "hall_compacto":
        # 0º: dos núcleos (P1: ancho útil >24m) — dos torres costillas con
        # patio central y junta constructiva, cada una con núcleo propio.
        _r_lat_dn = float(proyecto.retiro_lateral or 0.0)
        _r_pos_dn = float(proyecto.retiro_posterior or 0.0)
        _lu_dn = _erode_lote(lote, _r_lat_dn, _r_pos_dn) or lote
        _W_dn = _lu_dn.bounds[2] - _lu_dn.bounds[0]
        if _W_dn >= DOS_NUCLEOS_W_MIN:
            _g, _n = _generate_costillas_dos_nucleos(*_args)
            if _g is not None:
                return _emit(_g, _n, "costillas_dos_nucleos")
        # 1º: costillas (corredor central, patrón ref 9) para lotes ≥13m ancho
        _g, _n = _generate_costillas(*_args)
        if _g is not None:
            return _emit(_g, _n, "costillas")
        # 2º: hall compacto con núcleo lateral (lotes angostos/cortos)
        _g, _n = _generate_hall_compacto(*_args)
        if _g is not None:
            return _emit(_g, _n, "hall_compacto")
        # No factible: lote demasiado angosto para cualquier topología multifamiliar.
        _r_lat = float(proyecto.retiro_lateral or 0.0)
        _r_pos = float(proyecto.retiro_posterior or 0.0)
        _lu = _erode_lote(lote, _r_lat, _r_pos)
        _W_util = round((_lu.bounds[2] - _lu.bounds[0]) if _lu else short_len * 2, 2)
        raise ValueError(
            f"Lote inviable: ancho útil {_W_util}m < 6.5m mínimo para topología multifamiliar. "
            f"Aumentar frente o reducir retiros laterales."
        )
    if _topo_s["recomendada"] == "claustro":
        _g, _n = _generate_claustro(*_args)
        return _emit(_g, _n, "claustro")
    if _topo_s["recomendada"] == "tower":
        _g, _n = _generate_tower(*_args)
        return _emit(_g, _n, "tower")

    hall_poly = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, half_L, hw)
    hall_clipped = safe_clip(hall_poly, lote)

    esc_w = RNE["circulacion_v"]["esc_ancho"]
    esc_half_l = 2.50 / 2
    esc_depth = esc_w * 2

    stair_poly = Polygon([
        (cx - dl_x * esc_half_l + ds_x * hw,             cy - dl_y * esc_half_l + ds_y * hw),
        (cx + dl_x * esc_half_l + ds_x * hw,             cy + dl_y * esc_half_l + ds_y * hw),
        (cx + dl_x * esc_half_l + ds_x * (hw + esc_depth), cy + dl_y * esc_half_l + ds_y * (hw + esc_depth)),
        (cx - dl_x * esc_half_l + ds_x * (hw + esc_depth), cy - dl_y * esc_half_l + ds_y * (hw + esc_depth)),
    ])

    asc_polys = []
    asc_w = RNE["ascensor"]["ancho"]
    asc_l = RNE["ascensor"]["largo"]
    for i in range(num_asc):
        offset = esc_half_l + 0.20 + asc_l / 2 + i * (asc_l + 0.30)
        ac = (cx + dl_x * offset, cy + dl_y * offset)
        asc_poly = Polygon([
            (ac[0] - dl_x * asc_l / 2 + ds_x * hw,           ac[1] - dl_y * asc_l / 2 + ds_y * hw),
            (ac[0] + dl_x * asc_l / 2 + ds_x * hw,           ac[1] + dl_y * asc_l / 2 + ds_y * hw),
            (ac[0] + dl_x * asc_l / 2 + ds_x * (hw + asc_w), ac[1] + dl_y * asc_l / 2 + ds_y * (hw + asc_w)),
            (ac[0] - dl_x * asc_l / 2 + ds_x * (hw + asc_w), ac[1] - dl_y * asc_l / 2 + ds_y * (hw + asc_w)),
        ])
        asc_polys.append(asc_poly)

    core_items = [stair_poly] + asc_polys
    core_envelope = unary_union(core_items).envelope if core_items else stair_poly
    core_clipped = safe_clip(core_envelope, lote)

    vest_poly = None
    if nec_esc_prot:
        # Vestíbulo de presurización: antecámara en la entrada de la escalera.
        # Ocupa la primera sección del shaft (ds: hw → hw+1.50m), mismo ancho que la escalera.
        # Así queda contenido dentro del núcleo y no protrude como polígono externo.
        _vest_depth = 1.50
        vest_poly = Polygon([
            (cx - dl_x * esc_half_l + ds_x * hw,
             cy - dl_y * esc_half_l + ds_y * hw),
            (cx + dl_x * esc_half_l + ds_x * hw,
             cy + dl_y * esc_half_l + ds_y * hw),
            (cx + dl_x * esc_half_l + ds_x * (hw + _vest_depth),
             cy + dl_y * esc_half_l + ds_y * (hw + _vest_depth)),
            (cx - dl_x * esc_half_l + ds_x * (hw + _vest_depth),
             cy - dl_y * esc_half_l + ds_y * (hw + _vest_depth)),
        ])

    # ── Esquema de área libre ──────────────────────────────────────
    # Tres esquemas, todos cumplen pozo mín = H/4 (pozo_final):
    #   muros_ciegos  : patio lateral pequeño + ductos en muros ciegos (default)
    #   patio_posterior: patio único grande al fondo, sin ductos pequeños
    #   ducto_central : ducto único central dimensionado a pozo_final
    esquema_al = (proyecto.esquema_area_libre or "muros_ciegos").lower().strip()
    if esquema_al not in ("muros_ciegos", "patio_posterior", "ducto_central"):
        esquema_al = "muros_ciegos"

    ductos: List[Polygon] = []
    patio_clipped = None
    patio_long_half = 0.0  # mitad de la dimensión del patio sobre el eje largo

    if esquema_al == "patio_posterior":
        # Patio rectangular al fondo (lado opuesto al núcleo).
        # Ancho a lo largo del eje largo, profundidad ≥ pozo_final.
        patio_long = max(pozo_final * 2.2, min(half_L * 1.4, short_len * 0.75))
        patio_long_half = patio_long / 2
        patio_depth = max(pozo_final, 3.0)
        patio_poly = Polygon([
            (cx - dl_x * patio_long_half - ds_x * hw,
             cy - dl_y * patio_long_half - ds_y * hw),
            (cx + dl_x * patio_long_half - ds_x * hw,
             cy + dl_y * patio_long_half - ds_y * hw),
            (cx + dl_x * patio_long_half - ds_x * (hw + patio_depth),
             cy + dl_y * patio_long_half - ds_y * (hw + patio_depth)),
            (cx - dl_x * patio_long_half - ds_x * (hw + patio_depth),
             cy - dl_y * patio_long_half - ds_y * (hw + patio_depth)),
        ])
        patio_clipped = safe_clip(patio_poly, lote)

    elif esquema_al == "ducto_central":
        # Ducto único central de lado = pozo_final (mínimo normativo H/4).
        dim = max(pozo_final, 2.40)
        patio_long_half = dim / 2
        ducto_poly = Polygon([
            (cx - dl_x * patio_long_half - ds_x * hw,
             cy - dl_y * patio_long_half - ds_y * hw),
            (cx + dl_x * patio_long_half - ds_x * hw,
             cy + dl_y * patio_long_half - ds_y * hw),
            (cx + dl_x * patio_long_half - ds_x * (hw + dim),
             cy + dl_y * patio_long_half - ds_y * (hw + dim)),
            (cx - dl_x * patio_long_half - ds_x * (hw + dim),
             cy - dl_y * patio_long_half - ds_y * (hw + dim)),
        ])
        patio_clipped = safe_clip(ducto_poly, lote)

    else:  # muros_ciegos (default)
        # Sin patio estático — frente da a la calle (ventilación por fachada).
        # Ventilación de zona húmeda vía _auto_ductos_wet en muros medianeros.
        patio_clipped = None
        patio_long_half = 0.0

    core_min_L = -esc_half_l - 0.20
    core_max_L = (esc_half_l + 0.20 + asc_l / 2 + (num_asc - 1) * (asc_l + 0.30) + asc_l / 2) if num_asc > 0 else esc_half_l + 0.20
    # Fondo side only needs to clear the stair shaft — ascensores are on the frente (+ds) side
    # so fondo units can extend behind them without collision.
    stair_only_max_L = esc_half_l + 0.20
    patio_min_L = -patio_long_half - 0.20
    patio_max_L = patio_long_half + 0.20

    # ── Retiros aplicados a geometría ──
    # Retiro medido desde el límite real del lote (proyección de vértices sobre ejes dl/ds)
    # en lugar de desde el MRR — garantiza separación uniforme en lotes irregulares.
    # HORIZONTAL (|dl_x|>|dl_y|): L=izq/der, S=frente/fondo.
    # VERTICAL   (|dl_y|>|dl_x|): L=frente/fondo, S=izq/der.
    _lc = list(lote.exterior.coords)
    _lot_dl_vals = [(x - cx) * dl_x + (y - cy) * dl_y for x, y in _lc]
    _lot_ds_vals = [(x - cx) * ds_x + (y - cy) * ds_y for x, y in _lc]
    _lot_dl_max_geo = max(_lot_dl_vals)
    _lot_dl_min_geo = min(_lot_dl_vals)
    _lot_ds_max_geo = max(_lot_ds_vals)

    retiro_lat_geo = float(proyecto.retiro_lateral or 0.0)
    retiro_pos_geo = float(proyecto.retiro_posterior or 0.0)
    _is_horiz = abs(dl_x) > abs(dl_y)
    if _is_horiz:
        retiro_lat_neg = retiro_lat_geo   # extremo izq (DL-)
        retiro_lat_pos = retiro_lat_geo   # extremo der (DL+)
        retiro_fondo_geo = retiro_pos_geo  # extremo fondo (DS)
    else:
        retiro_lat_neg = 0.0              # frente: en lote polygon
        retiro_lat_pos = retiro_pos_geo   # extremo fondo (DL+)
        retiro_fondo_geo = retiro_lat_geo  # DS laterales (izq/der)
    L_min_useful = _lot_dl_min_geo + retiro_lat_neg
    L_max_useful = _lot_dl_max_geo - retiro_lat_pos
    half_S_useful = max(hw + 0.5, _lot_ds_max_geo - retiro_fondo_geo)

    # ── Dimensiones mínimas dpto (RNE A.020) ──
    MIN_FRENTE_DPTO = 3.00         # ancho mínimo de unidad en colindancia con hall
    MAX_FRENTE_DPTO = 8.00         # ancho máximo — unidad más ancha = muy grande en Lima
    MAX_RATIO_PROF_FRENTE = 3.0    # profundidad / frente; bloque alargado inviable
    MURO_T = 0.15                  # espesor nominal de muro (eje a eje); net = gross - MURO_T por dimensión

    # ── Multifamiliar (referencia RNE — Lima / uso nacional mismo umbral de área mín.) ──
    min_area_dpto = RNE["departamentos"]["min_multifamiliar"]
    depth_strip = max(0.35, half_S_useful - hw)
    # Aplicar restricción de % área libre mínimo si el usuario la definió.
    _pct_al = float(proyecto.area_libre_min_pct or 0.0)
    if _pct_al > 0:
        _L_span = max(L_max_useful - L_min_useful, 0.1)
        _max_depth_al = (lote.area * (1 - _pct_al / 100) / _L_span - 2 * hw) / 2
        depth_strip = min(depth_strip, max(0.35, _max_depth_al))
    # Filtro estricto: dpto sub-mínimo se descarta (no factor permisivo).
    min_poly_area = min_area_dpto

    segs_core_side = _strip_segments_for_apartments(L_min_useful, L_max_useful, core_min_L, core_max_L)
    # muros_ciegos: fondo side has no patio; exclude stair shaft only (ascensores are on frente side)
    _fondo_excl_min = core_min_L if esquema_al == "muros_ciegos" else patio_min_L
    _fondo_excl_max = stair_only_max_L if esquema_al == "muros_ciegos" else patio_max_L
    segs_patio_side = _strip_segments_for_apartments(L_min_useful, L_max_useful, _fondo_excl_min, _fondo_excl_max)
    cap_core = _max_units_on_strips(segs_core_side, depth_strip, min_area_dpto)
    cap_patio = _max_units_on_strips(segs_patio_side, depth_strip, min_area_dpto)
    cap_total = cap_core + cap_patio
    num_pedido_planta = num_dptos
    # Modo maximizar densidad: ignora num_departamentos y emite cap_total.
    if proyecto.optimizar_densidad and cap_total > 0:
        num_dptos = cap_total

    hall_buf = None
    try:
        if hall_clipped and not hall_clipped.is_empty:
            hall_buf = hall_clipped.buffer(0.40)
    except Exception:
        hall_buf = None

    strip_units_raw: List[Dict[str, Any]] = []

    _MIX_AREA: Dict[str, float] = {k: v["target"] for k, v in AREAS_TIPOLOGIA.items()}

    def _proportional_types(seq, n):
        """Select n types from seq proportionally by count, ordered smallest-first."""
        from collections import Counter
        cnt = Counter(seq)
        total_cnt = len(seq)
        alloc = {t: max(0, round(n * c / total_cnt)) for t, c in cnt.items()}
        diff = n - sum(alloc.values())
        if diff > 0:
            largest = max(alloc, key=lambda t: _MIX_AREA.get(t, 68.0))
            alloc[largest] += diff
        elif diff < 0:
            smallest = min(alloc, key=lambda t: _MIX_AREA.get(t, 68.0))
            alloc[smallest] = max(0, alloc[smallest] + diff)
        result = []
        for t in sorted(alloc, key=lambda t: _MIX_AREA.get(t, 68.0)):
            result.extend([t] * alloc[t])
        return result

    def distribute_units(L_min, L_max, exclude_min, exclude_max, num_units, sign_s, tipo_seq=None):
        units: List[Dict[str, Any]] = []
        if num_units <= 0:
            return units
        if (exclude_max <= L_min + 0.5) or (exclude_min >= L_max - 0.5):
            segments = [(L_min, L_max, num_units)]
        else:
            e_min = max(L_min, exclude_min)
            e_max = min(L_max, exclude_max)
            s1 = max(0, e_min - L_min)
            s2 = max(0, L_max - e_max)
            if s1 + s2 < 0.1:
                return []
            n1 = int(round(num_units * (s1 / (s1 + s2)))) if s1 > 2.0 else 0
            n2 = num_units - n1 if s2 > 2.0 else 0
            if n2 == 0 and s2 > max(3.0, s1 / max(1, n1) if n1 else 0) and num_units >= 2:
                n2 = 1
                n1 -= 1
            if n1 == 0 and s1 > max(3.0, s2 / max(1, n2) if n2 else 0) and num_units >= 2:
                n1 = 1
                n2 -= 1
            segments = []
            if n1 > 0:
                segments.append((L_min, e_min, n1))
            if n2 > 0:
                segments.append((e_max, L_max, n2))

        tipo_offset = 0  # tracks position in pre-sized tipo_seq across segments
        for seg_start, seg_end, n in segments:
            if n <= 0:
                continue
            seg_len = seg_end - seg_start
            depth_eff = half_S_useful - hw
            if depth_eff < 0.5:
                continue

            if tipo_seq:
                # Slice the pre-assigned type list for this segment.
                # tipo_seq is already sized to num_units for this side so the floor total
                # matches the requested mix exactly — no re-proportionalization needed.
                seg_typs = list(tipo_seq[tipo_offset:tipo_offset + n])
                tipo_offset += n
                if not seg_typs:
                    n_eff = 0
                    var_widths = []
                else:
                    targets = [_MIX_AREA.get(t, 68.0) / max(depth_eff, 1.0) for t in seg_typs]
                    min_mix_w = max(MIN_FRENTE_DPTO, min_poly_area / max(depth_eff, 1.0))
                    var_widths = [max(min_mix_w, min(t, MAX_FRENTE_DPTO)) for t in targets]
                    # Scale down proportionally before dropping any type
                    if var_widths and sum(var_widths) > seg_len + 0.05:
                        scale = seg_len / sum(var_widths)
                        var_widths = [max(min_mix_w, w * scale) for w in var_widths]
                    # Only drop if still over budget (all units at floor)
                    while len(var_widths) > 1 and sum(var_widths) > seg_len + 0.05:
                        max_i = max(range(len(var_widths)), key=lambda i: var_widths[i])
                        var_widths.pop(max_i)
                        seg_typs.pop(max_i)
                    # Distribute leftover space; cap per-unit at 1.35× tipo target area
                    remain = seg_len - sum(var_widths)
                    if remain > 0.01 and var_widths:
                        extra = remain / len(var_widths)
                        var_widths = [
                            min(w + extra,
                                MAX_FRENTE_DPTO,
                                _MIX_AREA.get(seg_typs[i], 68.0) * 1.35 / max(depth_eff, 1.0))
                            for i, w in enumerate(var_widths)
                        ]
                    n_eff = len(var_widths)
            else:
                # Equal-width distribution (existing behavior)
                max_n_seg = int((seg_len * depth_strip) / min_area_dpto) if seg_len > 0.1 else 0
                # Límite adicional por frente mínimo RNE A.020 (3.00m por unidad)
                max_by_frente = int(seg_len // MIN_FRENTE_DPTO) if seg_len > 0.1 else 0
                n_eff = min(n, max_n_seg, max_by_frente)
                # Si profundidad/frente excede ratio máximo, bajar n_eff hasta cumplir
                while n_eff > 1 and (depth_strip / (seg_len / n_eff)) > MAX_RATIO_PROF_FRENTE:
                    n_eff -= 1
                if n_eff <= 0:
                    continue
                w = seg_len / n_eff
                # Frente máximo: sube n_eff si unidad quedaría demasiado ancha, respetando n.
                if w > MAX_FRENTE_DPTO and max_n_seg > n_eff:
                    n_eff = min(n, max_n_seg, math.ceil(seg_len / MAX_FRENTE_DPTO))
                    w = seg_len / n_eff
                if w < MIN_FRENTE_DPTO - 0.01:
                    continue
                var_widths = [seg_len / n_eff] * n_eff
                seg_typs = [None] * n_eff

            off = seg_start
            for i, w_i in enumerate(var_widths):
                nxt = off + w_i
                typ_override = seg_typs[i] if tipo_seq and i < len(seg_typs) else None
                if sign_s > 0:
                    corners = [
                        (cx + dl_x * off + ds_x * hw, cy + dl_y * off + ds_y * hw),
                        (cx + dl_x * nxt + ds_x * hw, cy + dl_y * nxt + ds_y * hw),
                        (cx + dl_x * nxt + ds_x * (hw + depth_eff), cy + dl_y * nxt + ds_y * (hw + depth_eff)),
                        (cx + dl_x * off + ds_x * (hw + depth_eff), cy + dl_y * off + ds_y * (hw + depth_eff)),
                    ]
                else:
                    corners = [
                        (cx + dl_x * off - ds_x * hw, cy + dl_y * off - ds_y * hw),
                        (cx + dl_x * nxt - ds_x * hw, cy + dl_y * nxt - ds_y * hw),
                        (cx + dl_x * nxt - ds_x * (hw + depth_eff), cy + dl_y * nxt - ds_y * (hw + depth_eff)),
                        (cx + dl_x * off - ds_x * (hw + depth_eff), cy + dl_y * off - ds_y * (hw + depth_eff)),
                    ]
                off = nxt  # advance offset for variable-width distribution
                ap = safe_clip(Polygon(corners), lote)
                if ap is None or ap.area < min_poly_area:
                    continue
                # Convexidad: rechaza L-shapes generados por lotes irregulares.
                try:
                    hull_area = ap.convex_hull.area
                    convex_ratio = (ap.area / hull_area) if hull_area > 0 else 1.0
                except Exception:
                    convex_ratio = 1.0
                if convex_ratio < 0.85:
                    continue
                # Adyacencia al hall (sin medir longitud — corners[0..1] dan frente teórico ≥ MIN_FRENTE_DPTO).
                if hall_buf is not None:
                    try:
                        if not ap.intersects(hall_buf):
                            continue
                    except Exception:
                        pass
                ct = tuple(tuple(c) for c in corners)
                # depth_actual: profundidad REAL de la unidad clippeada al lote
                # (distinto de depth_eff teórico cuando el lote es irregular/trapezoidal)
                try:
                    _ap_ds = [(x - cx) * ds_x + (y - cy) * ds_y for x, y in ap.exterior.coords]
                    if sign_s > 0:
                        _depth_actual = max(_ap_ds) - hw
                    else:
                        _depth_actual = (-min(_ap_ds)) - hw
                    _depth_actual = max(0.5, _depth_actual)
                except Exception:
                    _depth_actual = depth_eff
                units.append({"poly": ap, "corners": ct,
                              "dl_off": off - w_i, "dl_nxt": nxt, "sign_s": sign_s,
                              "depth_eff": depth_eff, "depth_actual": _depth_actual,
                              "tipo_override": typ_override})
        return units

    if cap_total > 0:
        num_efectivo = min(num_dptos, cap_total)
        if cap_core > 0 and cap_patio > 0:
            dptos_a = min(cap_core, max(1, int(round(num_efectivo * cap_core / cap_total))))
            dptos_b = num_efectivo - dptos_a
        elif cap_core > 0:
            dptos_a = num_efectivo
            dptos_b = 0
        else:
            dptos_a = 0
            dptos_b = num_efectivo
        dptos_a = min(dptos_a, cap_core)
        dptos_b = min(dptos_b, cap_patio)
        diff = num_efectivo - dptos_a - dptos_b
        while diff > 0:
            if dptos_a < cap_core:
                dptos_a += 1
                diff -= 1
            elif dptos_b < cap_patio:
                dptos_b += 1
                diff -= 1
            else:
                break
        while diff < 0:
            if dptos_a > 0:
                dptos_a -= 1
                diff += 1
            elif dptos_b > 0:
                dptos_b -= 1
                diff += 1
            else:
                break
    else:
        dptos_a = max(1, num_dptos // 2)
        dptos_b = max(1, num_dptos - dptos_a)

    # Build typology sequences from mix if provided
    tipo_seq_a = None
    tipo_seq_b = None
    if proyecto.mix_tipologias:
        mix = {k: v for k, v in proyecto.mix_tipologias.items() if v > 0}
        if mix:
            flat = []
            for typ in sorted(mix.keys(), key=lambda t: _MIX_AREA.get(t, 68.0)):
                flat.extend([typ] * mix[typ])
            # Pre-assign exact per-side slices so the floor total matches mix exactly.
            # Smaller types go to fondo side (tipo_seq_a), larger to frente side (tipo_seq_b).
            tipo_seq_a = flat[:dptos_a] if flat else None
            tipo_seq_b = flat[dptos_a:] if flat else None
            if not tipo_seq_a:
                tipo_seq_a = None
            if not tipo_seq_b:
                tipo_seq_b = None

    strip_units_raw.extend(distribute_units(L_min_useful, L_max_useful, core_min_L, core_max_L, dptos_a, 1, tipo_seq=tipo_seq_a))
    # Fondo side (-ds) only needs to clear the stair shaft in dl — ascensores are at +ds
    # so fondo units behind the elevator range are physically valid.
    # patio/ducto_central schemes use their own exclusion geometry.
    excl_min_b = patio_min_L if esquema_al != "muros_ciegos" else core_min_L
    excl_max_b = patio_max_L if esquema_al != "muros_ciegos" else stair_only_max_L
    strip_units_raw.extend(distribute_units(L_min_useful, L_max_useful, excl_min_b, excl_max_b, dptos_b, -1, tipo_seq=tipo_seq_b))

    # ── Hall trimming: solo abarca el ancho real de las unidades, no de extremo a extremo ──
    if strip_units_raw:
        _act_L_min = min(min(u['dl_off'] for u in strip_units_raw), core_min_L)
        _act_L_max = max(max(u['dl_nxt'] for u in strip_units_raw), core_max_L)
        _act_half  = (_act_L_max - _act_L_min) / 2
        _act_ctr   = (_act_L_min + _act_L_max) / 2
        _hall_trimmed = make_rect(
            cx + dl_x * _act_ctr, cy + dl_y * _act_ctr,
            dl_x, dl_y, ds_x, ds_y, _act_half, hw
        )
        hall_clipped = safe_clip(_hall_trimmed, lote)
        try:
            if hall_clipped and not hall_clipped.is_empty:
                hall_buf = hall_clipped.buffer(0.40)
        except Exception:
            pass

    # ── Pozos de luz ──────────────────────────────────────────────────────────────
    # Dimensión normativa requerida: H/4 (pozo_final), sin cap artificial.
    # Si el pozo colocado queda menor que la requerida (límites geométricos),
    # se marca cumple=False y NO da crédito de ventilación a dormitorios.
    POZO_REQ = max(2.20, pozo_final)
    pozos_shared: List[Polygon] = []
    pozos_cumple: List[bool] = []

    def _prect_pozo(dl_a, dl_b, dsn, dsf):
        return Polygon([
            (cx + dl_x*dl_a + ds_x*dsn, cy + dl_y*dl_a + ds_y*dsn),
            (cx + dl_x*dl_b + ds_x*dsn, cy + dl_y*dl_b + ds_y*dsn),
            (cx + dl_x*dl_b + ds_x*dsf, cy + dl_y*dl_b + ds_y*dsf),
            (cx + dl_x*dl_a + ds_x*dsf, cy + dl_y*dl_a + ds_y*dsf),
        ])

    def _subtract_pozo_from(idx, cp) -> bool:
        """Resta el pozo de la unidad. True si la unidad sobrevive con área viable."""
        try:
            np_ = strip_units_raw[idx]['poly'].difference(cp)
            if hasattr(np_, 'geoms'):
                np_ = max(np_.geoms, key=lambda g: g.area)
            if not np_.is_valid:
                np_ = np_.buffer(0)
                if hasattr(np_, 'geoms'):
                    np_ = max(np_.geoms, key=lambda g: g.area)
            # Umbral reducido: aceptar fragmento si es > min viable (evita polígonos fantasma)
            if not np_.is_empty and np_.area >= min_poly_area * 0.35:
                strip_units_raw[idx]['poly'] = np_
                strip_units_raw[idx]['has_pozo'] = True
                return True
        except Exception:
            pass
        return False

    def _place_shared_pozos(sign):
        """Place shared pozos between adjacent contiguous pairs in strip `sign`."""
        _sorted = sorted(
            [(i, u) for i, u in enumerate(strip_units_raw) if u['sign_s'] == sign],
            key=lambda x: x[1]['dl_off']
        )
        for _ji in range(len(_sorted) - 1):
            _ia, _ua = _sorted[_ji]
            _ib, _ub = _sorted[_ji + 1]
            _shared_dl = _ua['dl_nxt']
            if abs(_shared_dl - _ub['dl_off']) > 0.15:
                continue
            # Usar profundidad real clippeada: min de ambas unidades del par
            _dep_a = _ua.get('depth_actual', _ua.get('depth_eff', depth_strip))
            _dep_b = _ub.get('depth_actual', _ub.get('depth_eff', depth_strip))
            _dep = min(_dep_a, _dep_b)
            # Espacio disponible dejando ≥0.40m de profundidad y ≥0.50m de frente por unidad
            _avail_ds = _dep - 0.40
            _avail_dl = (_ub['dl_nxt'] - 0.50) - (_ua['dl_off'] + 0.50)
            # Si la dimensión normativa (H/4) cabe completa → pozo conforme.
            # Si no cabe → pozo compacto best-effort (≤2.80m) marcado no-conforme:
            # no destruye unidades sin ganar conformidad.
            _fits = min(_avail_ds, _avail_dl) + 0.01 >= POZO_REQ
            _dim = POZO_REQ if _fits else min(POZO_REQ, 2.80)
            _pozo_ds = min(_dim, _avail_ds)
            if _pozo_ds < RNE["pozos_luz"]["min_abs"]:  # mínimo absoluto normativo 2.10m
                continue
            _ph = _dim / 2
            _dl_a = max(_ua['dl_off'] + 0.50, _shared_dl - _ph)
            _dl_b = min(_ub['dl_nxt'] - 0.50, _shared_dl + _ph)
            if _dl_b - _dl_a < RNE["pozos_luz"]["min_abs"]:
                continue
            _ds_near = sign * (hw + _dep - _pozo_ds)
            _ds_far  = sign * (hw + _dep)
            _cp = safe_clip(_prect_pozo(_dl_a, _dl_b, _ds_near, _ds_far), lote)
            if _cp is None or _cp.area <= 0.3:
                continue
            # Solo emitir el pozo si AMBAS unidades sobreviven la sustracción
            # (evita pozos fantasma solapados sobre unidades intactas)
            _ok_a = _subtract_pozo_from(_ia, _cp)
            _ok_b = _subtract_pozo_from(_ib, _cp)
            if not (_ok_a and _ok_b):
                continue
            # Conformidad: ambas dimensiones ≥ requerida (H/4)
            _cumple = min(_dl_b - _dl_a, _pozo_ds) + 0.01 >= POZO_REQ
            pozos_shared.append(_cp)
            pozos_cumple.append(_cumple)

    # sign_s > 0 strip always gets pozos (fondo strip in horiz, one lateral in vert)
    _place_shared_pozos(1)

    # sign_s < 0 strip: add pozos when its exterior face is a medianera (not open to street)
    _is_horiz = abs(dl_x) > abs(dl_y)
    if not _is_horiz:
        # Vertical building: sign_s=-1 → -ds direction (one lateral side)
        if ds_x <= 0:
            _neg_open = bool(proyecto.derecha_exterior) or not bool(proyecto.ciego_derecha)
        else:
            _neg_open = bool(proyecto.izquierda_exterior) or not bool(proyecto.ciego_izquierda)
        if not _neg_open:
            _place_shared_pozos(-1)

    # Individual pozos for isolated units with no adjacent pair when retiro=0 on their DS face.
    # These units' exterior (ds) face is a medianera — dormitorios cannot ventilate outward.
    _retiro_lat_eff = float(proyecto.retiro_lateral or 0.0)
    _retiro_pos_eff = float(proyecto.retiro_posterior or 0.0)
    if _is_horiz:
        # Fondo strip (sign_s=+1): ds exterior = fondo face → isolated unit needs pozo if retiro_pos=0
        _solo_signs = [(1, _retiro_pos_eff == 0.0)]
    else:
        # Both lateral strips: ds exterior = lateral face → needs pozo if retiro_lat=0
        _solo_signs = [(1, _retiro_lat_eff == 0.0), (-1, _retiro_lat_eff == 0.0)]
    for _sign, _solo_needed in _solo_signs:
        if not _solo_needed:
            continue
        for _idx, _u in enumerate(strip_units_raw):
            if _u.get('sign_s') != _sign or _u.get('has_pozo'):
                continue
            # Usar profundidad real clippeada de la unidad
            _dep = _u.get('depth_actual', _u.get('depth_eff', depth_strip))
            _avail_ds = _dep - 0.40
            _avail_dl = (_u['dl_nxt'] - 0.50) - (_u['dl_off'] + 0.50)
            _fits = min(_avail_ds, _avail_dl) + 0.01 >= POZO_REQ
            _dim = POZO_REQ if _fits else min(POZO_REQ, 2.80)
            _pozo_ds = min(_dim, _avail_ds)
            if _pozo_ds < RNE["pozos_luz"]["min_abs"]:  # mínimo absoluto normativo 2.10m
                continue
            _dl_ctr = (_u['dl_off'] + _u['dl_nxt']) / 2
            _ph = _dim / 2
            _dl_a = max(_u['dl_off'] + 0.50, _dl_ctr - _ph)
            _dl_b = min(_u['dl_nxt'] - 0.50, _dl_ctr + _ph)
            if _dl_b - _dl_a < RNE["pozos_luz"]["min_abs"]:
                continue
            _ds_near = _sign * (hw + _dep - _pozo_ds)
            _ds_far  = _sign * (hw + _dep)
            _cp = safe_clip(_prect_pozo(_dl_a, _dl_b, _ds_near, _ds_far), lote)
            if _cp is None or _cp.area <= 0.3:
                continue
            if not _subtract_pozo_from(_idx, _cp):
                continue
            _cumple = min(_dl_b - _dl_a, _pozo_ds) + 0.01 >= POZO_REQ
            pozos_shared.append(_cp)
            pozos_cumple.append(_cumple)

    # ── L-shaped frente units: wrap behind elevator + stair core ─────────────
    # Dead space exists behind core where units don't reach:
    #   • Behind elevator (dl=1.45..core_max_L, ds=hw+asc_w..full)
    #   • Behind stair shaft (dl=core_min_L..stair_only_max_L, ds=hw+esc_depth..full)
    # Adjacent frente units get L-arms to use these areas.
    if num_asc > 0 and esquema_al == "muros_ciegos":
        _asc_dl_start   = esc_half_l + 0.20   # 1.45m — elevator starts here in dl
        _ds_full        = half_S_useful        # full ds extent of frente strip
        # Elevator arm: ds starts PAST elevator depth (asc_w), not past stair depth
        _behind_asc_ds  = hw + asc_w           # dead zone starts here (behind elevator)
        # Stair arm: ds starts PAST stair depth
        _behind_esc_ds  = hw + esc_depth       # dead zone starts here (behind stair)

        if _ds_full - _behind_asc_ds > 1.0:
            # Right unit (dl_off ≈ core_max_L) wraps behind elevator
            _rgt = [(i, u) for i, u in enumerate(strip_units_raw)
                    if u['sign_s'] > 0 and u['dl_off'] >= core_max_L - 0.15]
            if _rgt:
                _ir, _ur = min(_rgt, key=lambda x: x[1]['dl_off'])
                _asc_arm = Polygon([
                    (cx + dl_x*_asc_dl_start + ds_x*_behind_asc_ds, cy + dl_y*_asc_dl_start + ds_y*_behind_asc_ds),
                    (cx + dl_x*core_max_L    + ds_x*_behind_asc_ds, cy + dl_y*core_max_L    + ds_y*_behind_asc_ds),
                    (cx + dl_x*core_max_L    + ds_x*_ds_full,       cy + dl_y*core_max_L    + ds_y*_ds_full),
                    (cx + dl_x*_asc_dl_start + ds_x*_ds_full,       cy + dl_y*_asc_dl_start + ds_y*_ds_full),
                ])
                _asc_arm_c = safe_clip(_asc_arm, lote)
                if _asc_arm_c is not None and _asc_arm_c.area > 1.0:
                    try:
                        _np = _ur['poly'].union(_asc_arm_c)
                        if hasattr(_np, 'geoms'):
                            _np = max(_np.geoms, key=lambda g: g.area)
                        if not _np.is_valid:
                            _np = _np.buffer(0)
                            if hasattr(_np, 'geoms'):
                                _np = max(_np.geoms, key=lambda g: g.area)
                        strip_units_raw[_ir]['poly'] = _np
                    except Exception:
                        pass

        if _ds_full - _behind_esc_ds > 1.0:
            # Left unit (dl_nxt ≈ core_min_L) wraps behind stair shaft
            _lft = [(i, u) for i, u in enumerate(strip_units_raw)
                    if u['sign_s'] > 0 and u['dl_nxt'] <= core_min_L + 0.15]
            if _lft:
                _il, _ul = max(_lft, key=lambda x: x[1]['dl_nxt'])
                _stair_arm = Polygon([
                    (cx + dl_x*core_min_L       + ds_x*_behind_esc_ds, cy + dl_y*core_min_L       + ds_y*_behind_esc_ds),
                    (cx + dl_x*stair_only_max_L + ds_x*_behind_esc_ds, cy + dl_y*stair_only_max_L + ds_y*_behind_esc_ds),
                    (cx + dl_x*stair_only_max_L + ds_x*_ds_full,       cy + dl_y*stair_only_max_L + ds_y*_ds_full),
                    (cx + dl_x*core_min_L       + ds_x*_ds_full,       cy + dl_y*core_min_L       + ds_y*_ds_full),
                ])
                _stair_arm_c = safe_clip(_stair_arm, lote)
                if _stair_arm_c is not None and _stair_arm_c.area > 1.0:
                    try:
                        _np = _ul['poly'].union(_stair_arm_c)
                        if hasattr(_np, 'geoms'):
                            _np = max(_np.geoms, key=lambda g: g.area)
                        if not _np.is_valid:
                            _np = _np.buffer(0)
                            if hasattr(_np, 'geoms'):
                                _np = max(_np.geoms, key=lambda g: g.area)
                        strip_units_raw[_il]['poly'] = _np
                    except Exception:
                        pass

    # ── Pasada seguridad: forzar subtract de pozos en unidades que aún los contengan ──
    # Itera por pozo individual (no union) con buffer 0.05m para cubrir floating-point boundary.
    # Umbral de overlap bajado a 0.01 m² para capturar casos de boundary exacto.
    for _pz in pozos_shared:
        try:
            _pz_buf = _pz.buffer(0.05)
        except Exception:
            _pz_buf = _pz
        for _rec in strip_units_raw:
            try:
                if not _rec['poly'].intersects(_pz_buf):
                    continue
                _ov = _rec['poly'].intersection(_pz_buf)
                if _ov.is_empty or _ov.area < 0.01:
                    continue
                _np = _rec['poly'].difference(_pz_buf)
                if hasattr(_np, 'geoms'):
                    _np = max(_np.geoms, key=lambda g: g.area)
                if not _np.is_valid:
                    _np = _np.buffer(0)
                    if hasattr(_np, 'geoms'):
                        _np = max(_np.geoms, key=lambda g: g.area)
                if not _np.is_empty and _np.area >= min_poly_area * 0.35:
                    _rec['poly'] = _np
            except Exception:
                pass

    # ── Pasada 1: generar zonas interiores (sin validar aún) ──
    apt_pre: List[Dict[str, Any]] = []
    for rec in strip_units_raw:
        ap = rec["poly"]
        corners = rec["corners"]
        area_gross = float(ap.area)
        # 2.1 Espesor de muros: área neta interior (helper común a todas las topologías)
        area_m = area_neta_muros(ap, MURO_T)
        # 2.4 Coherencia: área teórica = polígono corners antes del clip por lote
        try:
            corners_poly = Polygon(corners)
            area_teorica = float(corners_poly.area)
            ratio_clip = area_gross / max(area_teorica, 0.01)
        except Exception:
            area_teorica = area_gross
            ratio_clip = 1.0
        # Respect mix override when area matches; fall back to area-based when significantly off
        if rec.get("tipo_override") and ratio_clip >= 0.8:
            _t_ov = rec["tipo_override"]
            _max_ok = _MIX_AREA.get(_t_ov, 68.0) * 1.55
            typ = _t_ov if area_m <= _max_ok else get_typology(area_m)
        else:
            typ = get_typology(area_m)
        zonas_geom = generate_interior_zones(corners, typ, ap, lote)
        apt_pre.append({"poly": ap, "corners": corners, "area_m": area_m,
                        "area_gross": area_gross, "area_teorica": area_teorica,
                        "ratio_clip": ratio_clip,
                        "typ": typ, "zonas": zonas_geom, "sign_s": rec.get("sign_s", 0),
                        "dl_off": rec.get("dl_off", None), "dl_nxt": rec.get("dl_nxt", None)})

    # ── Auto-ductos antes de validar (así validators los ven) ──
    try:
        auto_d = _auto_ductos_wet(
            strip_units_raw, cx, cy, dl_x, dl_y, ds_x, ds_y, hw, lote, pozo_final
        )
        # Los pozos de luz ya proveen ventilación: no duplicar ducto húmedo en misma zona
        if pozos_shared:
            try:
                _pz_union = unary_union(pozos_shared).buffer(0.05)
                auto_d = [d for d in auto_d if not d.intersects(_pz_union)]
            except Exception:
                pass
        ductos.extend(auto_d)
    except Exception as e:
        import logging; logging.getLogger("uvicorn").warning(f"AUTO-DUCTOS ERROR: {e}")

    # ── Pasada 2: validar con ductos completos (estáticos + auto) ──
    # Pre-compute actual lot extents in dl-space so end-unit detection uses the
    # real lot boundary (not the MRR half-length, which overshoots on trapezoids).
    try:
        _lot_dl_pts = [dl_x * (x - cx) + dl_y * (y - cy)
                       for x, y in lote.exterior.coords]
        _lot_dl_min = min(_lot_dl_pts)
        _lot_dl_max = max(_lot_dl_pts)
    except Exception:
        _lot_dl_min = _lot_dl_min_geo
        _lot_dl_max = _lot_dl_max_geo

    # Pre-compute which surviving unit is the frente-end / fondo-end for each strip.
    # Using min/max dl_off per strip is more robust than comparing against L_min_useful:
    # it handles cases where the theoretical first unit is rejected (L-shaped lot clipping).
    _strip_frente_end: dict = {}  # sign_s → index in apt_pre of frente-end unit
    _strip_fondo_end:  dict = {}  # sign_s → index in apt_pre of fondo-end unit
    for _ss in (1, -1):
        _idxs = [(i, d) for i, d in enumerate(apt_pre) if d.get("sign_s") == _ss]
        if not _idxs:
            continue
        _ff = min(_idxs, key=lambda x: x[1].get("dl_off") or 0)
        _fn = max(_idxs, key=lambda x: x[1].get("dl_nxt") or 0)
        _strip_frente_end[_ss] = _ff[0]
        _strip_fondo_end[_ss]  = _fn[0]

    departamentos_detalle: List[Dict[str, Any]] = []
    for _apt_idx, d in enumerate(apt_pre):
        ap = d["poly"]
        corners = d["corners"]
        area_m = d["area_m"]
        typ = d["typ"]
        zonas_geom = d["zonas"]
        sign_s_unit = d.get("sign_s", 0)
        validacion_unidad = validar_unidad(
            unit=ap,
            zonas=zonas_geom,
            escalera=stair_poly,
            hall_buf=hall_buf,
            ductos=ductos,
            # Solo pozos con dimensión normativa completa dan crédito de ventilación
            pozos=[p for p, ok in zip(pozos_shared, pozos_cumple) if ok],
        )
        # Determine fachada_exterior for all four sides using polygon extents along dl axis.
        # Use corner projections (not centroid) so large units that touch the building end
        # are correctly identified regardless of unit depth.
        try:
            cen_x, cen_y = ap.centroid.x, ap.centroid.y
            dl_cen = dl_x * (cen_x - cx) + dl_y * (cen_y - cy)
            dl_pts = [dl_x * (x - cx) + dl_y * (y - cy) for x, y in ap.exterior.coords]
            dl_unit_min = min(dl_pts)
            dl_unit_max = max(dl_pts)
            # Use pre-computed strip end-unit indices: which surviving unit has the
            # smallest dl_off (frente end) / largest dl_nxt (fondo end) per strip.
            # More robust than L_min_useful ± tol: works even when the theoretical
            # first unit is rejected by the convexity filter.
            _ss = d.get("sign_s", 0)
            is_izquierda_end = (_strip_frente_end.get(_ss) == _apt_idx)
            is_derecha_end   = (_strip_fondo_end.get(_ss)  == _apt_idx)
        except Exception:
            cen_x = cen_y = dl_cen = 0.0
            is_derecha_end = is_izquierda_end = False
        # fachada_exterior: depends on building orientation (dl direction).
        # Horizontal building (|dl_x| > |dl_y|): ds is the frente-fondo axis.
        #   sign_s < 0 → -ds → toward frente (street) → always exterior.
        #   sign_s > 0 → +ds → toward fondo → exterior only if fondo_exterior.
        # Vertical building (|dl_y| > |dl_x|): ds is the lateral axis (left/right).
        #   +ds maps to LEFT (ds_x<0) or RIGHT (ds_x>0) of lote.
        #   Exterior depends on ciego_izquierda/ciego_derecha.
        #   Units at frente end (near L_min, is_izquierda_end) are always exterior
        #   because their end wall faces the retiro/street.
        _is_horiz = abs(dl_x) > abs(dl_y)
        if _is_horiz:
            _strip_ext = (sign_s_unit < 0) or (sign_s_unit > 0 and bool(proyecto.fondo_exterior))
            _end_ext = (
                (is_derecha_end and bool(proyecto.derecha_exterior))
                or (is_izquierda_end and bool(proyecto.izquierda_exterior))
            )
        else:
            # ds_x<0 → +ds is leftward (izquierda); ds_x>0 → +ds is rightward (derecha).
            if ds_x <= 0:
                _pos_ciego = bool(proyecto.ciego_izquierda)
                _pos_ext   = bool(proyecto.izquierda_exterior)
                _neg_ciego = bool(proyecto.ciego_derecha)
                _neg_ext   = bool(proyecto.derecha_exterior)
            else:
                _pos_ciego = bool(proyecto.ciego_derecha)
                _pos_ext   = bool(proyecto.derecha_exterior)
                _neg_ciego = bool(proyecto.ciego_izquierda)
                _neg_ext   = bool(proyecto.izquierda_exterior)
            if sign_s_unit > 0:
                _strip_ext = not _pos_ciego or _pos_ext
            else:
                _strip_ext = not _neg_ciego or _neg_ext
            _frente_end = is_izquierda_end
            _fondo_end  = is_derecha_end
            _end_ext = (_frente_end and bool(proyecto.frente_exterior)) or (_fondo_end and bool(proyecto.fondo_exterior))
        validacion_unidad["fachada_exterior"] = _strip_ext or _end_ext
        # 2.3 Distancia a escalera — criterio único: validar_unidad (EVAC_MAX_M)
        validacion_unidad["distancia_escalera_m"] = validacion_unidad.get("distancia_evac_m", 0.0)
        validacion_unidad["dist_esc_cumple"] = validacion_unidad.get("evac_cumple", True)
        # 2.4 Coherencia lotes irregulares
        ratio_clip = d.get("ratio_clip", 1.0)
        es_reducida = ratio_clip < 0.80
        validacion_unidad["es_unidad_reducida"] = es_reducida
        validacion_unidad["ratio_area_clip"] = r3(ratio_clip)
        zonas_payload = []
        for z in zonas_geom:
            v = z.get("validacion", {})
            zonas_payload.append({
                "nombre": z["nombre"],
                "kind": z.get("kind", ""),
                "coords": poly_to_js(z["geom"]),
                "area_m2": r3(float(z["geom"].area)),
                "validacion": v,
            })
        # lado: for horizontal dl, sign_s<0=frente. For vertical dl, determined by cen along dl.
        if _is_horiz:
            lado = "frente" if sign_s_unit < 0 else "fondo"
        else:
            lado = "frente" if dl_cen < 0 else "fondo"
        departamentos_detalle.append({
            "contorno": poly_to_js(ap),
            "tipologia": typ,
            "area_m2": r3(area_m),
            "area_gross_m2": r3(d.get("area_gross", area_m)),
            "lado": lado,
            "es_reducida": es_reducida,
            "zonas": zonas_payload,
            "validacion": validacion_unidad,
        })

    # ── #6 fase B + #7 L_plan: brazo + dptos secundarios en ala chica ──
    brazo_poly = None
    connector_poly = None
    secondary_count = 0
    if main_rect is not None:
        try:
            diff_geom = lote.difference(main_rect)
            small_geom = None
            if not diff_geom.is_empty:
                if diff_geom.geom_type == "Polygon":
                    small_geom = diff_geom
                elif diff_geom.geom_type == "MultiPolygon":
                    small_geom = max(diff_geom.geoms, key=lambda g: g.area)
            if small_geom is not None and small_geom.area >= min_area_dpto * 1.2:
                sr_b = small_geom.bounds  # axis-aligned bbox
                sr_w = sr_b[2] - sr_b[0]
                sr_h = sr_b[3] - sr_b[1]

                if sr_h >= sr_w:
                    # Brazo vertical (long Y); dptos a izq/der del brazo
                    brazo_cx = (sr_b[0] + sr_b[2]) / 2
                    brazo_poly = Polygon([
                        (brazo_cx - hw, sr_b[1]), (brazo_cx + hw, sr_b[1]),
                        (brazo_cx + hw, sr_b[3]), (brazo_cx - hw, sr_b[3])
                    ])
                    # Connector horizontal entre main hall (cx) y brazo
                    conn_y = (sr_b[1] + sr_b[3]) / 2
                    conn_x0 = min(cx, brazo_cx)
                    conn_x1 = max(cx, brazo_cx)
                    connector_poly = Polygon([
                        (conn_x0, conn_y - hw), (conn_x1, conn_y - hw),
                        (conn_x1, conn_y + hw), (conn_x0, conn_y + hw)
                    ])
                    brazo_length = sr_b[3] - sr_b[1]
                    for side in (-1, 1):
                        if side < 0:
                            x_inner = brazo_cx - hw
                            x_outer = sr_b[0]
                        else:
                            x_inner = brazo_cx + hw
                            x_outer = sr_b[2]
                        depth = abs(x_outer - x_inner)
                        if depth < 2.5:
                            continue
                        n_max_seg = int((brazo_length * depth) / min_area_dpto) if brazo_length > 0.1 else 0
                        n_max_fr = int(brazo_length // MIN_FRENTE_DPTO)
                        n_eff = min(n_max_seg, n_max_fr)
                        if n_eff <= 0:
                            continue
                        w_each = brazo_length / n_eff
                        for i in range(n_eff):
                            y0 = sr_b[1] + i * w_each
                            y1 = y0 + w_each
                            x_lo = min(x_inner, x_outer)
                            x_hi = max(x_inner, x_outer)
                            corners_sec = [
                                (x_lo, y0), (x_hi, y0),
                                (x_hi, y1), (x_lo, y1),
                            ]
                            ap_sec = safe_clip(Polygon(corners_sec), lote)
                            if ap_sec is None or ap_sec.area < min_poly_area:
                                continue
                            try:
                                hull_a = ap_sec.convex_hull.area
                                cr = ap_sec.area / hull_a if hull_a > 0 else 1.0
                            except Exception:
                                cr = 1.0
                            if cr < 0.85:
                                continue
                            area_sec = float(ap_sec.area)
                            typ_sec = get_typology(area_sec)
                            ct_sec = tuple(tuple(c) for c in corners_sec)
                            zonas_sec = generate_interior_zones(ct_sec, typ_sec, ap_sec, lote)
                            val_sec = validar_unidad(
                                unit=ap_sec, zonas=zonas_sec,
                                escalera=stair_poly, hall_buf=hall_buf, ductos=ductos,
                            )
                            zonas_payload_sec = [{
                                "nombre": z["nombre"],
                                "kind": z.get("kind", ""),
                                "coords": poly_to_js(z["geom"]),
                                "area_m2": r3(float(z["geom"].area)),
                                "validacion": z.get("validacion", {}),
                            } for z in zonas_sec]
                            departamentos_detalle.append({
                                "contorno": poly_to_js(ap_sec),
                                "tipologia": typ_sec,
                                "area_m2": r3(area_sec),
                                "zonas": zonas_payload_sec,
                                "validacion": val_sec,
                            })
                            secondary_count += 1
                else:
                    # Brazo horizontal (long X); dptos arriba/abajo del brazo
                    brazo_cy = (sr_b[1] + sr_b[3]) / 2
                    brazo_poly = Polygon([
                        (sr_b[0], brazo_cy - hw), (sr_b[2], brazo_cy - hw),
                        (sr_b[2], brazo_cy + hw), (sr_b[0], brazo_cy + hw)
                    ])
                    conn_x = (sr_b[0] + sr_b[2]) / 2
                    conn_y0 = min(cy, brazo_cy)
                    conn_y1 = max(cy, brazo_cy)
                    connector_poly = Polygon([
                        (conn_x - hw, conn_y0), (conn_x + hw, conn_y0),
                        (conn_x + hw, conn_y1), (conn_x - hw, conn_y1)
                    ])
                    brazo_length = sr_b[2] - sr_b[0]
                    for side in (-1, 1):
                        if side < 0:
                            y_inner = brazo_cy - hw
                            y_outer = sr_b[1]
                        else:
                            y_inner = brazo_cy + hw
                            y_outer = sr_b[3]
                        depth = abs(y_outer - y_inner)
                        if depth < 2.5:
                            continue
                        n_max_seg = int((brazo_length * depth) / min_area_dpto) if brazo_length > 0.1 else 0
                        n_max_fr = int(brazo_length // MIN_FRENTE_DPTO)
                        n_eff = min(n_max_seg, n_max_fr)
                        if n_eff <= 0:
                            continue
                        w_each = brazo_length / n_eff
                        for i in range(n_eff):
                            x0 = sr_b[0] + i * w_each
                            x1 = x0 + w_each
                            y_lo = min(y_inner, y_outer)
                            y_hi = max(y_inner, y_outer)
                            corners_sec = [
                                (x0, y_lo), (x1, y_lo),
                                (x1, y_hi), (x0, y_hi),
                            ]
                            ap_sec = safe_clip(Polygon(corners_sec), lote)
                            if ap_sec is None or ap_sec.area < min_poly_area:
                                continue
                            try:
                                hull_a = ap_sec.convex_hull.area
                                cr = ap_sec.area / hull_a if hull_a > 0 else 1.0
                            except Exception:
                                cr = 1.0
                            if cr < 0.85:
                                continue
                            area_sec = float(ap_sec.area)
                            typ_sec = get_typology(area_sec)
                            ct_sec = tuple(tuple(c) for c in corners_sec)
                            zonas_sec = generate_interior_zones(ct_sec, typ_sec, ap_sec, lote)
                            val_sec = validar_unidad(
                                unit=ap_sec, zonas=zonas_sec,
                                escalera=stair_poly, hall_buf=hall_buf, ductos=ductos,
                            )
                            zonas_payload_sec = [{
                                "nombre": z["nombre"],
                                "kind": z.get("kind", ""),
                                "coords": poly_to_js(z["geom"]),
                                "area_m2": r3(float(z["geom"].area)),
                                "validacion": z.get("validacion", {}),
                            } for z in zonas_sec]
                            departamentos_detalle.append({
                                "contorno": poly_to_js(ap_sec),
                                "tipologia": typ_sec,
                                "area_m2": r3(area_sec),
                                "zonas": zonas_payload_sec,
                                "validacion": val_sec,
                            })
                            secondary_count += 1
        except Exception:
            pass

    # Extender hall con brazo + connector (L-shaped hall)
    if brazo_poly is not None:
        try:
            parts = [p for p in (hall_clipped, safe_clip(brazo_poly, lote), safe_clip(connector_poly, lote) if connector_poly else None) if p is not None and not p.is_empty]
            if parts:
                hall_clipped = unary_union(parts)
        except Exception:
            pass

    # 3.3 Grilla estructural: columnas en ejes de unidades + núcleo
    depth_eff_col = half_S_useful - hw
    col_ds_rows = [-(hw + depth_eff_col), -hw, hw, hw + depth_eff_col]
    _col_dl_set: set = {L_min_useful, L_max_useful, core_min_L, core_max_L}
    for _sr in strip_units_raw:
        if _sr.get('dl_off') is not None:
            _col_dl_set.add(_sr['dl_off'])
        if _sr.get('dl_nxt') is not None:
            _col_dl_set.add(_sr['dl_nxt'])
    columnas = _compute_column_grid(
        cx, cy, dl_x, dl_y, ds_x, ds_y,
        L_min_useful, L_max_useful, col_ds_rows, lote,
        dl_positions=sorted(_col_dl_set),
    )

    geometry = {
        "hall": poly_to_js(hall_clipped),
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote)),
        "ascensores": [poly_to_js(safe_clip(a, lote)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote)) if vest_poly else [],
        "patio": poly_to_js(patio_clipped),
        "ductos": [poly_to_js(d) for d in ductos],
        "pozos_luz": [poly_to_js(p) for p in pozos_shared],
        "pozos_luz_cumple": list(pozos_cumple),
        "columnas": [poly_to_js(c) for c in columnas],
        "esquema_area_libre": esquema_al,
        "departamentos": departamentos_detalle,
        "cabida_multifamiliar": {
            "contexto": "Perú — edificación multifamiliar (área mínima RNE como referencia de cabida)",
            "area_min_dpto_m2": min_area_dpto,
            "profundidad_strip_estimada_m": r3(depth_strip),
            "departamentos_solicitados_planta": num_pedido_planta,
            "capacidad_maxima_estimada_planta": cap_total,
            "capacidad_lado_nucleo": cap_core,
            "capacidad_lado_patio": cap_patio,
            "departamentos_generados_planta": len(departamentos_detalle),
            "nota": "Capacidad estimada con el rectángulo mínimo del lote; el recorte al polígono real puede reducir áreas útiles.",
        },
    }

    topo_info = informe_topologia(lote, proyecto.frente or 0.0)
    geometry["topologia"] = topo_info

    # Área útil de strip disponible por planta (ambos lados del hall)
    strip_area_planta = sum(
        (seg_end - seg_start) * depth_strip
        for segs in (segs_core_side, segs_patio_side)
        for seg_start, seg_end in segs
    )

    _n_pozos_ok = sum(1 for ok in pozos_cumple if ok)
    normativa = {
        "pozo_final": r3(pozo_final),
        "pozos_luz_check": {
            "dimension_requerida_m": r3(POZO_REQ),
            "colocados": len(pozos_shared),
            "conformes": _n_pozos_ok,
            "no_conformes": len(pozos_shared) - _n_pozos_ok,
            "ok": (len(pozos_shared) - _n_pozos_ok) == 0,
            "nota": (
                "Pozo de luz mínimo H/4 = {:.2f}m. Pozos no conformes no acreditan "
                "ventilación de dormitorios; en edificios altos considerar esquema "
                "patio_posterior o dormitorios a fachada.".format(POZO_REQ)
            ),
        },
        "ascensor_obligatorio": nec_ascensor,
        "ascensor_forzado": ascensor_forzado,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": RNE["departamentos"]["min_multifamiliar"],
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
        "cabida_planta": {
            "departamentos_pedidos": num_pedido_planta,
            "departamentos_emitidos": len(departamentos_detalle),
            "capacidad_max_estimada_planta": cap_total,
        },
        "topologia": topo_info,
    }

    if proyecto.precios_tipologia:
        try:
            mix = _optimizar_mix(
                strip_area_planta, depth_strip,
                proyecto.precios_tipologia, proyecto.numero_pisos
            )
            if mix:
                normativa["mix_optimo"] = mix
        except Exception:
            pass

    normativa["variables_ignoradas"] = _variables_ignoradas(proyecto, "spine")

    # E5: evaluación interna (spine fallback path)
    try:
        geometry["evaluacion"] = _evaluar_diseno(
            geometry, lote, _r_lat_ev, _r_pos_ev, _nd_req, pozo_final)
    except Exception:
        geometry["evaluacion"] = {"score": 0, "defectos": [], "n_criticos": 0, "metricas": {}}
    return geometry, normativa


def _generate_primer_piso(proyecto: ProyectoInmobiliario, geometry: dict):
    coords = proyecto.coordenadas_lote
    if len(coords) != 4:
        # lote no cuadrilátero: bbox real como fallback (_get_cell exige quad)
        lote_sh = Polygon(coords)
        if not lote_sh.is_valid:
            lote_sh = lote_sh.buffer(0)
        bx0, by0, bx1, by1 = lote_sh.bounds
        coords = [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]
    p1, p2, p3, p4 = ({"x": x, "y": y} for x, y in coords)

    # coordenadas_lote ya viene neto de retiro frontal (mismo contrato que
    # _erode_lote y la planta típica: "frente ya viene neto") — mismo marco
    # real que usan escalera/ascensores/departamentos, sin reconstrucción
    # sintética independiente.
    techada_poly = [p1, p2, p3, p4]

    r_lat = float(proyecto.retiro_lateral or 0.0)
    frente_neto = max(1.0, math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"]))
    fondo_izq   = max(1.0, math.hypot(p4["x"] - p1["x"], p4["y"] - p1["y"]))
    fondo_der   = max(1.0, math.hypot(p3["x"] - p2["x"], p3["y"] - p2["y"]))

    u_left  = 0.0 if proyecto.ciego_izquierda else r_lat / frente_neto
    u_right = 1.0 if proyecto.ciego_derecha   else 1.0 - r_lat / frente_neto
    v_bot   = 1.0 if proyecto.ciego_fondo     else 1.0 - r_lat / ((fondo_izq + fondo_der) / 2)

    lote_neto = _get_cell(techada_poly, u_left, u_right, 0, v_bot)
    b_w = max(1.0, _poly_width(lote_neto))
    b_d = max(1.0, math.hypot(
        lote_neto[2]["x"] - lote_neto[1]["x"],
        lote_neto[2]["y"] - lote_neto[1]["y"],
    ))

    def uw(m): return max(0.0, min(1.0, m / b_w))
    def vd(m): return max(0.0, min(1.0, m / b_d))

    # ── Normative sizing ────────────────────────────────────────
    HAS_SOTANO = (proyecto.pct_estac or 0) > 0
    # Rampa vehicular: RNE A.010 art. 33 → 3.00m libre, solo si hay sótano y lote suficiente
    RAMPA_W = 3.00 if HAS_SOTANO and b_w >= 8.0 else 0.0
    # G7: mismo largo que la rampa real del sótano (RNE art.66, pendiente max
    # 15%) — antes esta huella cruzaba TODO el fondo (vd=1.0) desalineada del
    # tramo real que dibuja _generate_sotano (regresión visual si no coincide).
    RAMPA_PENDIENTE_MAX = 0.15
    _h_nivel = float(proyecto.altura_piso or 2.80)
    RAMPA_L = min(b_d, max(4.5, _h_nivel / RAMPA_PENDIENTE_MAX))
    GAP = 0.10

    # Cuarto de basura: RNE A.010 art. 40 → 0.03m²/m² área techada/piso, min 6m²
    area_techada_floor = b_w * b_d
    req_basura = max(6.0, area_techada_floor * 0.03)
    BASURA_W = min(3.50, max(2.50, math.sqrt(req_basura)))
    BASURA_D = max(2.50, req_basura / BASURA_W)

    # Cuarto de tableros: RNE EM.010 → min 1.50 × 2.00m
    TABL_W = 1.50
    TABL_D = max(2.00, BASURA_D)

    # SSHH accesible: RNE A.120 art. 15 → 1 ud. por planta, min 1.80 × 2.00m
    SSHH_W = 2.00
    SSHH_D = 2.00

    # Lobby: RNE A.010 art. 21 → min 2.40m ancho; ~20-25% del ancho neto
    LOBBY_W = max(2.40, min(5.00, b_w * 0.25))
    LOBBY_D = max(3.50, b_d * 0.20)

    # G5: lobby centrado en el eje real de CADA núcleo (escalera+ascensores)
    # -- no en lo que sobre tras empaquetar servicios, y no en el promedio
    # de todas las torres (eso deja el lobby flotando en el patio central
    # de esquemas dos-núcleos). Un lobby+puerta por núcleo real.
    _nucleos_pp = geometry.get("nucleos") or [{
        "escalera": geometry.get("escalera") or [],
        "ascensores": geometry.get("ascensores") or [],
    }]
    dx = p2["x"] - p1["x"]
    n_nuc = max(1, len(_nucleos_pp))
    LOBBY_W_EACH = LOBBY_W if n_nuc == 1 else max(2.40, LOBBY_W / n_nuc)
    PUERTA_W = max(1.20, min(2.00, LOBBY_W_EACH * 0.5))

    lobbies, puertas, lobby_bands = [], [], []
    for _nuc in _nucleos_pp:
        esc_pts = _nuc.get("escalera") or []
        asc_pts_all = [p for a in (_nuc.get("ascensores") or []) for p in a]
        nuc_pts = list(esc_pts) + asc_pts_all
        if nuc_pts and abs(dx) > 1e-6:
            ncx = (min(p["x"] for p in nuc_pts) + max(p["x"] for p in nuc_pts)) / 2.0
            u_nuc = max(0.0, min(1.0, (ncx - p1["x"]) / dx))
        else:
            u_nuc = 0.5
        u0 = max(0.0, min(1.0 - uw(LOBBY_W_EACH), u_nuc - uw(LOBBY_W_EACH) / 2.0))
        u1 = min(1.0, u0 + uw(LOBBY_W_EACH))

        depth = LOBBY_D
        if esc_pts and len(esc_pts) >= 3:
            exs = [p["x"] for p in esc_pts]
            eys = [p["y"] for p in esc_pts]
            x0 = p1["x"] + dx * u0
            x1 = p1["x"] + dx * u1
            if x1 > min(exs) and x0 < max(exs):
                # escalera cae en el ancho del lobby → extender profundidad
                # hasta tocarla (RNE A.010: acceso continuo hall→núcleo)
                front_y = min(p1["y"], p2["y"])
                depth = max(LOBBY_D, (max(eys) - front_y) + 0.30)
        lobbies.append(_get_cell(lote_neto, u0, u1, 0, vd(depth)))

        pu_c = (u0 + u1) / 2.0
        pu0 = max(0.0, pu_c - uw(PUERTA_W) / 2.0)
        pu1 = min(1.0, pu0 + uw(PUERTA_W))
        puertas.append(_get_cell(lote_neto, pu0, pu1, 0, vd(0.20)))
        lobby_bands.append((u0, u1))

    lobby, puerta = lobbies[0], puertas[0]
    lobby_u0, lobby_u1 = lobby_bands[0]

    # Resto de servicios: empaquetados alrededor de la banda del lobby
    # (salta la banda si el cursor la cruza) — deja lobby siempre en su eje.
    u = 0.0

    def _place(w):
        nonlocal u
        if u < lobby_u0 and u + uw(w) > lobby_u0:
            u = lobby_u1 + uw(GAP)
        u0_, u1_ = u, u + uw(w)
        if u1_ > 1.0:
            return None, None
        u = u1_ + uw(GAP)
        return u0_, u1_

    # 1. Rampa (full depth → sótano)
    rampa = []
    if RAMPA_W > 0:
        u0, u1 = _place(RAMPA_W)
        if u0 is not None:
            rampa = _get_cell(lote_neto, u0, u1, 0, vd(RAMPA_L))

    # 2. Cuarto de basura
    basura = []
    u0, u1 = _place(BASURA_W)
    if u0 is not None:
        basura = _get_cell(lote_neto, u0, u1, 0, vd(BASURA_D))

    # 3. Cuarto de tableros
    tableros = []
    u0, u1 = _place(TABL_W)
    if u0 is not None:
        tableros = _get_cell(lote_neto, u0, u1, 0, vd(TABL_D))

    # 4. SSHH accesible
    servicios = []
    u0, u1 = _place(SSHH_W)
    if u0 is not None:
        servicios = _get_cell(lote_neto, u0, u1, 0, vd(SSHH_D))

    # 5. Comercio: remanente a la derecha del lobby (o del último servicio
    # si desbordó ahí), solo si da un ancho útil
    com_u0 = max(u, lobby_u1 + uw(GAP))
    com2 = []
    if (1.0 - com_u0) * b_w > 1.50:
        com2 = _get_cell(lote_neto, com_u0, 1.0, 0, vd(LOBBY_D))

    return {
        "rampa":     rampa,
        "basura":    basura,
        "tableros":  tableros,
        "servicios": servicios,
        "lobby":     lobby,
        "puerta":    puerta,
        "lobbies":   lobbies,
        "puertas":   puertas,
        "comercios": [c for c in [com2] if c],
    }


def _generate_azotea(proyecto: ProyectoInmobiliario, geometry: dict, normativa: dict):
    """Planta azotea: caja escalera, cuarto máquinas, tanque elevado (RNE IS.010).

    Alineada al núcleo REAL de la planta típica: la caja de escalera es la
    proyección de la escalera generada, el cuarto de máquinas envuelve los
    ascensores reales y el tanque se apoya junto al núcleo.
    """
    num_dptos_planta = len(geometry.get("departamentos") or []) or (proyecto.num_departamentos or 4)
    num_dptos_total = num_dptos_planta * (proyecto.numero_pisos or 1)

    # Un núcleo por torre (dos-núcleos = dos torres = dos cajas de escalera +
    # dos cuartos de máquinas propios, no un cuarto de máquinas que abarque
    # ambas torres cruzando el patio central).
    _nucleos_az = geometry.get("nucleos") or [{
        "escalera": geometry.get("escalera") or [],
        "ascensores": geometry.get("ascensores") or [],
    }]

    # ── Tanque elevado: RNE IS.010 art. 2.4.2 → vol = 1/3 dotación diaria
    # (200 L/hab, 3 hab/dpto promedio), repartido entre los núcleos que sí
    # tienen caja de escalera.
    vol_litros = num_dptos_total * 3 * 200 / 3
    vol_m3 = vol_litros / 1000.0
    h_tanque = 2.00

    def _bbox(pts_lists):
        xs = [p["x"] for pts in pts_lists for p in pts]
        ys = [p["y"] for pts in pts_lists for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def _rect_cell(x0, y0, x1, y1):
        return [{"x": r3(x0), "y": r3(y0)}, {"x": r3(x1), "y": r3(y0)},
                {"x": r3(x1), "y": r3(y1)}, {"x": r3(x0), "y": r3(y1)}]

    try:
        lote_sh = Polygon(proyecto.coordenadas_lote)
        if not lote_sh.is_valid:
            lote_sh = lote_sh.buffer(0)
    except Exception:
        lote_sh = None

    _validos = [n for n in _nucleos_az if (n.get("escalera") or []) and len(n["escalera"]) >= 3]
    if _validos:
        n_nuc = len(_validos)
        vol_m3_each = vol_m3 / n_nuc
        base_area_each = vol_m3_each / h_tanque
        TANK_W = max(2.00, min(4.00, math.sqrt(base_area_each)))
        TANK_D = max(2.00, min(5.00, base_area_each / TANK_W))

        cajas_escalera, cuartos_maquinas, tanques_elevados = [], [], []
        area_cm_total = 0.0
        for _nuc in _validos:
            esc_pts = _nuc["escalera"]
            asc_list = [a for a in (_nuc.get("ascensores") or []) if a and len(a) >= 3]
            caja_escalera = [dict(p) for p in esc_pts]
            core_parts = [esc_pts] + asc_list

            cuarto_maquinas = []
            area_cm = 0.0
            if asc_list:
                ax0, ay0, ax1, ay1 = _bbox(asc_list)
                CM_W = max(3.00, ax1 - ax0)
                CM_D = max(3.50, ay1 - ay0)
                acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
                cuarto_maquinas = _rect_cell(acx - CM_W / 2, acy - CM_D / 2,
                                             acx + CM_W / 2, acy + CM_D / 2)
                area_cm = CM_W * CM_D

            cx0, cy0, cx1, cy1 = _bbox(core_parts)
            tcx = (cx0 + cx1) / 2
            ty0 = cy1 + 0.30
            tank_rect = _rect_cell(tcx - TANK_W / 2, ty0, tcx + TANK_W / 2, ty0 + TANK_D)
            if lote_sh is not None:
                try:
                    tank_poly = Polygon([(p["x"], p["y"]) for p in tank_rect])
                    if tank_poly.difference(lote_sh).area > 0.10 * tank_poly.area:
                        ty1 = cy0 - 0.30
                        tank_rect = _rect_cell(tcx - TANK_W / 2, ty1 - TANK_D,
                                               tcx + TANK_W / 2, ty1)
                except Exception:
                    pass

            cajas_escalera.append(caja_escalera)
            cuartos_maquinas.append(cuarto_maquinas)
            tanques_elevados.append(tank_rect)
            area_cm_total += area_cm

        return {
            "caja_escalera":    cajas_escalera[0],
            "cuarto_maquinas":  cuartos_maquinas[0],
            "tanque_elevado":   tanques_elevados[0],
            "cajas_escalera":   cajas_escalera,
            "cuartos_maquinas": cuartos_maquinas,
            "tanques_elevados": tanques_elevados,
            "vol_tanque_m3":    r3(vol_m3),
            "area_cm_m2":       r3(area_cm_total),
        }

    # ── Fallback legacy (sin escalera en geometry): marco real del lote
    # (G6, mismo patrón F1/F4) en vez de quad sintético frente/fondo/
    # derecha/izquierda ──
    coords = proyecto.coordenadas_lote
    if len(coords) != 4:
        lote_sh = Polygon(coords)
        if not lote_sh.is_valid:
            lote_sh = lote_sh.buffer(0)
        bx0, by0, bx1, by1 = lote_sh.bounds
        coords = [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]
    p1, p2, p3, p4 = ({"x": x, "y": y} for x, y in coords)
    techada_poly = [p1, p2, p3, p4]

    retiro_lat  = float(proyecto.retiro_lateral or 2.30)
    frente_neto = max(1.0, math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"]))
    fondo_neto  = max(1.0, (
        math.hypot(p4["x"] - p1["x"], p4["y"] - p1["y"]) +
        math.hypot(p3["x"] - p2["x"], p3["y"] - p2["y"])
    ) / 2)
    u_left  = 0.0 if proyecto.ciego_izquierda else retiro_lat / frente_neto
    u_right = 1.0 if proyecto.ciego_derecha   else 1.0 - retiro_lat / frente_neto
    v_bot   = 1.0 if proyecto.ciego_fondo     else 1.0 - retiro_lat / fondo_neto

    lote_neto = _get_cell(techada_poly, u_left, u_right, 0, v_bot)
    b_w = max(1.0, _poly_width(lote_neto))
    b_d = max(1.0, math.hypot(
        lote_neto[2]["x"] - lote_neto[1]["x"],
        lote_neto[2]["y"] - lote_neto[1]["y"],
    ))

    def uw(m): return max(0.0, min(1.0, m / b_w))
    def vd(m): return max(0.0, min(1.0, m / b_d))

    ESC_W = 2.50
    ESC_D = 2.50
    esc_u0 = max(0.0, 0.5 - uw(ESC_W) / 2)
    esc_u1 = min(1.0, esc_u0 + uw(ESC_W))
    esc_v0, esc_v1 = 0.38, min(1.0, 0.38 + vd(ESC_D))
    caja_escalera = _get_cell(lote_neto, esc_u0, esc_u1, esc_v0, esc_v1)

    num_asc_fb = max(1, proyecto.num_ascensores or 1)
    CM_W = max(3.00, num_asc_fb * 2.50)
    CM_D = 3.50
    cm_u0 = min(1.0, esc_u1 + uw(0.10))
    cm_u1 = min(1.0, cm_u0 + uw(CM_W))
    cm_v0, cm_v1 = esc_v0, min(1.0, esc_v0 + vd(CM_D))
    cuarto_maquinas = _get_cell(lote_neto, cm_u0, cm_u1, cm_v0, cm_v1)

    tank_u0 = max(0.0, 0.5 - uw(TANK_W) / 2)
    tank_u1 = min(1.0, tank_u0 + uw(TANK_W))
    tank_v0 = max(0.0, 1.0 - vd(TANK_D))
    tanque_elevado = _get_cell(lote_neto, tank_u0, tank_u1, tank_v0, 1.0)

    return {
        "caja_escalera":   caja_escalera,
        "cuarto_maquinas": cuarto_maquinas,
        "tanque_elevado":  tanque_elevado,
        "vol_tanque_m3":   r3(vol_m3),
        "area_cm_m2":      r3(CM_W * CM_D),
    }


def _generate_sotano(proyecto: ProyectoInmobiliario, geometry: dict, normativa: dict):
    # G6: marco real del lote (mismo patrón F1/F4) — antes reconstruía un
    # trapecio sintético desde frente/fondo/derecha/izquierda, desalineado
    # del marco real que usan planta típica/primer piso (rampa PB↔S1 no
    # coincidía en lotes no perfectamente rectangulares).
    coords = proyecto.coordenadas_lote
    if len(coords) != 4:
        lote_sh = Polygon(coords)
        if not lote_sh.is_valid:
            lote_sh = lote_sh.buffer(0)
        bx0, by0, bx1, by1 = lote_sh.bounds
        coords = [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]
    p1, p2, p3, p4 = ({"x": x, "y": y} for x, y in coords)

    inset = 0.30
    slab = [
        {"x": p1["x"] + inset, "y": p1["y"] + inset},
        {"x": p2["x"] - inset, "y": p2["y"] + inset},
        {"x": p3["x"] - inset, "y": p3["y"] - inset},
        {"x": p4["x"] + inset, "y": p4["y"] - inset},
    ]

    xs = [p["x"] for p in slab]
    ys = [p["y"] for p in slab]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # G7: largo real de rampa por pendiente RNE A.010 art.66 (max 15% recta),
    # no la profundidad completa del lote — antes cruzaba todo el fondo
    # (ej. 27m) desperdiciando losa que podía ser plazas/cisternas.
    RAMPA_W = 3.0
    RAMPA_PENDIENTE_MAX = 0.15
    h_nivel = float(proyecto.altura_piso or 2.80)
    rampa_largo = min(max_y - min_y, max(4.5, h_nivel / RAMPA_PENDIENTE_MAX))
    rampa = [
        {"x": round(min_x, 3), "y": round(min_y, 3)},
        {"x": round(min_x + RAMPA_W, 3), "y": round(min_y, 3)},
        {"x": round(min_x + RAMPA_W, 3), "y": round(min_y + rampa_largo, 3)},
        {"x": round(min_x, 3), "y": round(min_y + rampa_largo, 3)},
    ]
    rampa_poly = Polygon([(p["x"], p["y"]) for p in rampa])

    # ── Núcleo: continuidad vertical real — reutiliza escalera+ascensores+
    # vestíbulo de la planta típica (mismo marco de coordenadas para el caso
    # de lote rectangular estándar; ver limitación conocida para lotes
    # irregulares en la nota F1) en vez de dejar el sótano sin salida ──
    # Reserva TODOS los núcleos (dos-núcleos = dos escaleras+ascensores
    # propios) -- reservar solo el singular "escalera"/"ascensores" dejaba
    # el núcleo de la torre derecha sin reservar y plazas podían invadirlo.
    nucleo_raw = []
    for _nuc in (geometry.get("nucleos") or [{
        "escalera": geometry.get("escalera") or [],
        "ascensores": geometry.get("ascensores") or [],
        "vestibulo": geometry.get("vestibulo") or [],
    }]):
        esc = _nuc.get("escalera") or []
        if len(esc) >= 3:
            nucleo_raw.append(esc)
        for a in (_nuc.get("ascensores") or []):
            if a and len(a) >= 3:
                nucleo_raw.append(a)
        vest = _nuc.get("vestibulo") or []
        if len(vest) >= 3:
            nucleo_raw.append(vest)

    nucleo_shapes = []
    for pts in nucleo_raw:
        try:
            sp = Polygon([(p["x"], p["y"]) for p in pts]).buffer(0)
            if not sp.is_empty:
                nucleo_shapes.append(sp)
        except Exception:
            pass

    # ── Zona reservada: rampa + núcleo (con margen de acceso) — nada se
    # coloca encima. Antes las plazas se trazaban en grilla ciega sobre
    # toda la losa e invadían la rampa (hallazgo: 6 plazas bajo rampa). ──
    reserved_parts = [rampa_poly] + [sp.buffer(0.6) for sp in nucleo_shapes]
    reserved = unary_union(reserved_parts)

    dptos = geometry.get("departamentos", [])
    num_dptos = len(dptos) if dptos else proyecto.num_departamentos
    total_dptos = num_dptos * proyecto.numero_pisos
    pct_estac = proyecto.pct_estac or 30
    req_estac = math.ceil(total_dptos * (pct_estac / 100))

    stall_w = normativa.get("estacionamiento_ancho", 2.70)
    stall_d = 5.00
    aisle_w = 6.00

    dot = normativa.get("dotaciones", RNE["instalaciones"])
    agua_1d = dot.get("agua_1d", 500) / 1000
    agua_2d = dot.get("agua_2d", 850) / 1000
    agua_3d = dot.get("agua_3d", 1200) / 1000
    aci_m3 = dot.get("aci_m3", 25)

    dom = 0
    for dpto in dptos:
        coords = _departamento_outline_coords(dpto)
        if coords and len(coords) >= 3:
            area = calc_poly_area(coords)
            typ_full = get_typology(area)
            if isinstance(dpto, dict) and dpto.get("tipologia"):
                typ_full = dpto["tipologia"]
            cat = dotacion_categoria_tipologia(typ_full)
            m3 = agua_1d if cat == "1D" else (agua_2d if cat == "2D" else agua_3d)
            dom += m3 * proyecto.numero_pisos

    total_cist = dom + aci_m3
    cuarto_maq = max(15, total_cist * 0.12)

    # ── Cisternas: recinto de ancho fijo realista (antes: franja de todo
    # el ancho de losa, ~0.3-0.7m — no construible) en la esquina opuesta
    # a la rampa, sumado a la zona reservada ──
    cisternas = []
    if total_cist > 0:
        area_dom_a = dom / 2
        area_dom_b = dom / 2
        area_aci = aci_m3
        area_maq = cuarto_maq
        total_cist_area = area_dom_a + area_dom_b + area_aci + area_maq
        cist_w = max(3.5, min(6.0, max_x - min_x - RAMPA_W - 1.0))
        cist_h = total_cist_area / cist_w if cist_w > 0 else 0
        cx0 = max_x - cist_w
        cy0 = max_y - cist_h
        zones = [
            {"label": f"CIST. CONS. A\n{area_dom_a:.1f} m3", "area": area_dom_a, "fill": "#bfdbfe", "stroke": "#2563eb"},
            {"label": f"CIST. CONS. B\n{area_dom_b:.1f} m3", "area": area_dom_b, "fill": "#93c5fd", "stroke": "#1d4ed8"},
            {"label": f"CIST. ACI\n{area_aci:.1f} m3", "area": area_aci, "fill": "#fca5a5", "stroke": "#dc2626"},
            {"label": f"CTO. MAQ.\n{area_maq:.1f} m2", "area": area_maq, "fill": "#fef3c7", "stroke": "#d97706"},
        ]
        cursor_y = cy0
        for zone in zones:
            zone_h = zone["area"] / cist_w if cist_w > 0 else 0
            poly = [
                {"x": round(cx0, 3), "y": round(cursor_y, 3)},
                {"x": round(max_x, 3), "y": round(cursor_y, 3)},
                {"x": round(max_x, 3), "y": round(cursor_y + zone_h, 3)},
                {"x": round(cx0, 3), "y": round(cursor_y + zone_h, 3)},
            ]
            cisternas.append({"poly": poly, "label": zone["label"], "fill": zone["fill"], "stroke": zone["stroke"]})
            cursor_y += zone_h
        cist_shape = Polygon([(cx0, cy0), (max_x, cy0), (max_x, max_y), (cx0, max_y)])
        reserved = unary_union([reserved, cist_shape.buffer(0.3)])

    # ── Plazas: grid-scan de dos filas por pasillo (patrón original),
    # descartando celdas que invaden rampa/núcleo/cisternas en vez de
    # trazar ciego sobre toda la losa ──
    def _cell_free(poly_pts):
        try:
            sp = Polygon([(p["x"], p["y"]) for p in poly_pts])
            return reserved.intersection(sp).area < 0.05
        except Exception:
            return True

    def _scan_row(y):
        row = []
        x_cursor = min_x
        while x_cursor + stall_w <= max_x:
            stall = [
                {"x": round(x_cursor, 3), "y": round(y, 3)},
                {"x": round(x_cursor + stall_w, 3), "y": round(y, 3)},
                {"x": round(x_cursor + stall_w, 3), "y": round(y + stall_d, 3)},
                {"x": round(x_cursor, 3), "y": round(y + stall_d, 3)},
            ]
            if _cell_free(stall):
                row.append(stall)
            x_cursor += stall_w
        return row

    def _scan_nivel(need):
        """Llena una losa (mismo footprint) con hasta `need` plazas."""
        stalls = []
        aisles = []
        stall_num = 1
        remaining = need
        y_cursor = min_y

        while remaining > 0 and y_cursor + stall_d <= max_y:
            row1 = _scan_row(y_cursor)
            for stall in row1:
                if remaining <= 0:
                    break
                stalls.append({"id": f"E-{stall_num:02d}", "poly": stall})
                stall_num += 1; remaining -= 1

            if row1:
                aisle_y = y_cursor + stall_d
                if aisle_y + aisle_w < max_y:
                    aisle = [
                        {"x": round(min_x, 3), "y": round(aisle_y, 3)},
                        {"x": round(max_x, 3), "y": round(aisle_y, 3)},
                        {"x": round(max_x, 3), "y": round(aisle_y + aisle_w, 3)},
                        {"x": round(min_x, 3), "y": round(aisle_y + aisle_w, 3)},
                    ]
                    aisles.append(aisle)
                    y_cursor = aisle_y + aisle_w
                else:
                    break
                if remaining > 0 and y_cursor + stall_d <= max_y:
                    row2 = _scan_row(y_cursor)
                    for stall in row2:
                        if remaining <= 0:
                            break
                        stalls.append({"id": f"E-{stall_num:02d}", "poly": stall})
                        stall_num += 1; remaining -= 1
                    y_cursor += stall_d
            else:
                y_cursor += 1.0
        return stalls, aisles

    nucleo_out = [poly_to_js(sp) for sp in nucleo_shapes]

    # ── Niveles: si un sótano no cabe la demanda de req_estac, se agregan
    # niveles adicionales (mismo footprint, rampa/núcleo continúan verticales)
    # hasta cubrir el déficit — antes se reportaba count < req_estac sin
    # ninguna acción (hallazgo: 12/21 plazas, "cumple" nunca se corregía). ──
    MAX_NIVELES = 6
    niveles = []
    cubiertas = 0
    nivel_idx = 1
    while cubiertas < req_estac and nivel_idx <= MAX_NIVELES:
        stalls_n, aisles_n = _scan_nivel(req_estac - cubiertas)
        if not stalls_n and nivel_idx > 1:
            break  # nivel adicional no aporta plazas: parar (lote no da más)
        niveles.append({
            "name": f"S{nivel_idx}",
            "slab": slab,
            "stalls": stalls_n,
            "aisles": aisles_n,
            "count": len(stalls_n),
            "rampa": rampa,
            "nucleo": nucleo_out,
            "cisternas": cisternas if nivel_idx == 1 else [],
        })
        cubiertas += len(stalls_n)
        if not stalls_n:
            break
        nivel_idx += 1

    s1 = niveles[0]
    return {
        "name": s1["name"],
        "slab": s1["slab"],
        "stalls": s1["stalls"],
        "aisles": s1["aisles"],
        "count": s1["count"],
        "rampa": s1["rampa"],
        "nucleo": s1["nucleo"],
        "cisternas": s1["cisternas"],
        "req_estac": req_estac,
        "cisterna_total_m3": round(total_cist, 1),
        "cisterna_domestico": round(dom, 1),
        "cisterna_aci": round(aci_m3, 1),
        "cisterna_maq": round(cuarto_maq, 1),
        "niveles": niveles,
        "num_niveles": len(niveles),
        "count_total": cubiertas,
        "deficit": max(0, req_estac - cubiertas),
    }


# ═══════════════════════════════════════════════════════════════
# EMPAQUETADO JSON — Estructura normalizada para Three.js / WebGL
# ═══════════════════════════════════════════════════════════════

def _build_webgl_payload(
    proyecto: ProyectoInmobiliario,
    geometry: dict,
    normativa: dict,
    primer_piso: dict,
    sotano: dict,
    azotea: dict = None,
) -> dict:
    """
    Construye el objeto JSON completo con coordenadas normalizadas (centradas en 0,0)
    y estructura semántica lista para consumo WebGL / Three.js.
    """

    # ── Polígono REAL del lote recibido (frente ya neto por convención del
    # caller) — única fuente para dibujar lote/retiro/torre. Antes se
    # reconstruía un trapecio sintético desde frente/fondo/derecha/izquierda,
    # que solo coincidía con el lote real en el caso rectangular por defecto
    # y desalineaba la torre en lotes de mapa/dibujados/irregulares. ──
    frente = proyecto.frente or 10
    fondo_val = proyecto.fondo or 10
    derecha = proyecto.derecha or 20
    izquierda = proyecto.izquierda or 20

    lote_real = Polygon(proyecto.coordenadas_lote)
    if not lote_real.is_valid:
        lote_real = lote_real.buffer(0)
    lote_pts = [{"x": x, "y": y} for x, y in list(lote_real.exterior.coords)[:-1]]

    # Centroide del lote real para normalizar a (0,0)
    cx_norm = lote_real.centroid.x
    cy_norm = lote_real.centroid.y

    def norm(pts):
        """[{x,y},...] → [[x-cx, y-cy],...] normalized + rounded."""
        if not pts:
            return []
        return [[r3(p["x"] - cx_norm), r3(p["y"] - cy_norm)] for p in pts]

    def norm_cell(cell):
        """_get_cell result [{x,y},...] → [[x,y],...] normalized."""
        if not cell:
            return []
        return [[r3(p["x"] - cx_norm), r3(p["y"] - cy_norm)] for p in cell]

    # ── Lote y retiro ──
    lote_coords = norm(lote_pts)

    # Banda de retiro frontal: extrusión hacia afuera del borde de frente
    # real (el lote recibido ya viene neto de este retiro) — no un
    # interpolado sobre corners sintéticos.
    retiro_pts = []
    r_front = float(proyecto.retiro_frontal or 0.0)
    if r_front > 0 and len(lote_pts) >= 3:
        coords = [(p["x"], p["y"]) for p in lote_pts]
        n = len(coords)
        best = None
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if best is None or my < best[0]:
                ex, ey = x2 - x1, y2 - y1
                L = math.hypot(ex, ey) or 1.0
                ex, ey = ex / L, ey / L
                inx, iny = -ey, ex
                if (cx_norm - mx) * inx + (cy_norm - my) * iny < 0:
                    inx, iny = -inx, -iny  # inx,iny apunta hacia adentro
                best = (my, (x1, y1), (x2, y2), (-inx, -iny))
        if best is not None:
            _, (x1, y1), (x2, y2), (onx, ony) = best
            band = [
                (x1, y1), (x2, y2),
                (x2 + onx * r_front, y2 + ony * r_front),
                (x1 + onx * r_front, y1 + ony * r_front),
            ]
            retiro_pts = norm([{"x": x, "y": y} for x, y in band])

    # ── Área del terreno ──
    area_terreno = r3(lote_real.area)

    # ── Unidades (departamentos) ──
    hall_coords_norm = norm(geometry.get("hall", []))
    hall_raw = geometry.get("hall", [])
    hall_poly_sh = None
    if len(hall_raw) >= 3:
        try:
            hall_poly_sh = Polygon([(p["x"], p["y"]) for p in hall_raw]).buffer(0.05)
        except Exception:
            pass
    dptos_raw = geometry.get("departamentos", [])
    unidades = []
    typ_hab_map = {"1D": 2, "1D+E": 3, "2D": 3, "2D+E": 4, "3D": 5}
    for i, raw in enumerate(dptos_raw):
        dpto_pts = _departamento_outline_coords(raw)
        if not dpto_pts or len(dpto_pts) < 3:
            continue
        area_gross = r3(_calculate_poly_area(dpto_pts))
        # Use stored net interior area (with muro deduction) if available; fallback to gross polygon area
        area = r3(raw.get("area_m2", area_gross)) if isinstance(raw, dict) else area_gross
        typ = ""
        if isinstance(raw, dict) and raw.get("tipologia"):
            typ = raw["tipologia"]
        if not typ:
            typ = get_typology(area)
        hab = typ_hab_map.get(typ, 3)
        coords_norm = norm(dpto_pts)
        zonas_norm = []
        if isinstance(raw, dict):
            for z in raw.get("zonas") or []:
                if not isinstance(z, dict):
                    continue
                zc = z.get("coords") or []
                if len(zc) >= 3:
                    zonas_norm.append({
                        "nombre": z.get("nombre", ""),
                        "kind": z.get("kind", ""),
                        "coords": norm(zc),
                        "area_m2": z.get("area_m2", 0.0),
                        "validacion": z.get("validacion", {}),
                    })
        if hall_poly_sh is not None and len(dpto_pts) >= 3:
            try:
                apt_poly = Polygon([(p["x"], p["y"]) for p in dpto_pts])
                colinda_hall = bool(hall_poly_sh.intersects(apt_poly))
            except Exception:
                colinda_hall = _validate_adjacency(coords_norm, hall_coords_norm, tolerance=0.60)
        else:
            colinda_hall = _validate_adjacency(coords_norm, hall_coords_norm, tolerance=0.60)
        raw_val = raw.get("validacion", {}) if isinstance(raw, dict) else {}
        unidades.append({
            "id": f"X{i + 1:02d}",
            "type": "apartment",
            "coords": coords_norm,
            "zonas": zonas_norm,
            "metadata": {
                "area": area,
                "area_gross": r3(raw.get("area_gross_m2", area_gross)) if isinstance(raw, dict) else area_gross,
                "tipologia": typ,
                "habitantes": hab,
                "lado": raw.get("lado", "fondo") if isinstance(raw, dict) else "fondo",
                "es_reducida": raw.get("es_reducida", False) if isinstance(raw, dict) else False,
            },
            "validacion": {
                "colinda_hall": colinda_hall,
                "cumple_area_min": area >= RNE["departamentos"]["min_multifamiliar"],
                "fachada_exterior": raw_val.get("fachada_exterior", False),
                # Fallback a distancia_evac_m: claustro/tower no anotan distancia_escalera_m
                "distancia_escalera_m": raw_val.get("distancia_escalera_m", raw_val.get("distancia_evac_m", 0.0)),
                "dist_esc_cumple": raw_val.get("dist_esc_cumple", raw_val.get("evac_cumple", True)),
                "es_unidad_reducida": raw_val.get("es_unidad_reducida", False),
                "ratio_area_clip": raw_val.get("ratio_area_clip", 1.0),
                "arquitectonica": raw_val,
            }
        })

    # ── Área vendible y eficiencia (huella real) ──
    # area_techada_planta = área de la HUELLA del edificio (unión de polígonos
    # techados: dptos gross + hall + corridors + núcleo), no la suma de áreas
    # netas. Incluye muros → CUS y eficiencia realistas. Pozos de luz ya están
    # sustraídos de las unidades; el patio no se incluye (sin techar).
    total_vendible = sum(u["metadata"]["area"] for u in unidades)
    hall_area = r3(_calculate_poly_area(geometry.get("hall", [])))
    core_area = r3(_calculate_poly_area(geometry.get("core", [])))
    corridors_area = r3(sum(
        _calculate_poly_area(c) for c in geometry.get("corridors", []) if c
    ))
    area_comun = r3(hall_area + core_area + corridors_area)

    def _sh_poly(pts):
        """[{x,y},...] → shapely Polygon válido o None."""
        if not pts or len(pts) < 3:
            return None
        try:
            pl = Polygon([(p["x"], p["y"]) for p in pts])
            if not pl.is_valid:
                pl = pl.buffer(0)
            return pl if (pl and not pl.is_empty) else None
        except Exception:
            return None

    _fp_parts = []
    for _pts in (
        [geometry.get("hall", []), geometry.get("core", []), geometry.get("vestibulo", [])]
        + list(geometry.get("corridors", []))
        + [_departamento_outline_coords(d) for d in dptos_raw]
    ):
        _pl = _sh_poly(_pts)
        if _pl is not None:
            _fp_parts.append(_pl)
    area_techada = r3(total_vendible + area_comun)  # fallback: suma de netas
    if _fp_parts:
        try:
            _fp_union = unary_union(_fp_parts)
            if _fp_union and not _fp_union.is_empty:
                area_techada = r3(float(_fp_union.area))
        except Exception:
            pass
    eficiencia = r3((total_vendible / area_techada * 100) if area_techada > 0 else 0)

    # ── Núcleo ──
    esc_presurizada = normativa.get("esc_protegida_obligatoria", False)
    escalera_pts = norm(geometry.get("escalera", []))
    asc_list = [norm(a) for a in geometry.get("ascensores", []) if a and len(a) >= 3]
    vest_pts = norm(geometry.get("vestibulo", []))

    # ── Técnico: patios, ductos, corridors ──
    patio_pts = norm(geometry.get("patio", []))
    pozo_final = normativa.get("pozo_final", 2.2)
    ductos_list = [norm(d) for d in geometry.get("ductos", []) if d and len(d) >= 3]
    _pz_raw = geometry.get("pozos_luz", [])
    _pz_cumple = geometry.get("pozos_luz_cumple", [])
    pozos_luz_pairs = [
        (norm(p), bool(_pz_cumple[i]) if i < len(_pz_cumple) else True)
        for i, p in enumerate(_pz_raw)
        if p and len(p) >= 3
    ]
    corridor_list = [norm(c) for c in geometry.get("corridors", []) if c and len(c) >= 3]

    patio_w = 0
    if len(patio_pts) >= 4:
        patio_w = math.hypot(
            patio_pts[1][0] - patio_pts[0][0],
            patio_pts[1][1] - patio_pts[0][1]
        )

    # ── Anotaciones (cotas y etiquetas) ──
    anotaciones = []
    # Cotas del terreno — solo si el lote real es un cuadrilátero (frente/
    # derecha/fondo/izquierda solo tienen sentido en ese caso; en lotes
    # irregulares se omiten en vez de rotular con lados que no existen)
    side_labels = []
    if len(lote_coords) == 4:
        side_labels = [
            (lote_coords[0], lote_coords[1], f"{frente:.1f}m (Fte)", "cota"),
            (lote_coords[1], lote_coords[2], f"{derecha:.1f}m (Der)", "cota"),
            (lote_coords[2], lote_coords[3], f"{fondo_val:.1f}m (Fdo)", "cota"),
            (lote_coords[3], lote_coords[0], f"{izquierda:.1f}m (Izq)", "cota"),
        ]
    for pa, pb, txt, clase in side_labels:
        mid = [r3((pa[0] + pb[0]) / 2), r3((pa[1] + pb[1]) / 2)]
        anotaciones.append({"pos": mid, "texto": txt, "clase": clase})

    # Etiquetas de departamentos
    for u in unidades:
        if u["coords"]:
            cen = _centroid(u["coords"])
            anotaciones.append({
                "pos": cen,
                "texto": f"{u['metadata']['tipologia']} · DPTO {u['id']} · {u['metadata']['area']:.1f}m²",
                "clase": "etiqueta"
            })

    # ── Sótano: estacionamientos (coordenadas normalizadas) ──
    stalls_norm = []
    for st in sotano.get("stalls", []):
        coords_n = norm(st.get("poly", []))
        if coords_n:
            stalls_norm.append({"id": st["id"], "coords": coords_n})

    aisles_norm = [norm(a) for a in sotano.get("aisles", []) if a and len(a) >= 3]

    cisternas_norm = []
    for c in sotano.get("cisternas", []):
        coords_n = norm(c.get("poly", []))
        if coords_n:
            cisternas_norm.append({
                "coords": coords_n,
                "label": c.get("label", ""),
                "fill": c.get("fill", "#bfdbfe"),
                "stroke": c.get("stroke", "#2563eb"),
            })

    primer_piso_norm = {
        "comercios": [norm_cell(c) for c in primer_piso.get("comercios", [])],
        "servicios": norm_cell(primer_piso.get("servicios", [])),
        "lobby":     norm_cell(primer_piso.get("lobby", [])),
        "puerta":    norm_cell(primer_piso.get("puerta", [])),
        "rampa":     norm_cell(primer_piso.get("rampa", [])),
        "basura":    norm_cell(primer_piso.get("basura", [])),
        "tableros":  norm_cell(primer_piso.get("tableros", [])),
        # Un lobby+puerta por núcleo real (dos-núcleos = dos accesos).
        "lobbies": [norm_cell(c) for c in primer_piso.get("lobbies", [])],
        "puertas": [norm_cell(c) for c in primer_piso.get("puertas", [])],
    }

    sotano_norm = {
        "name": sotano.get("name", "S1"),
        "slab": norm(sotano.get("slab", [])),
        "stalls": stalls_norm,
        "aisles": aisles_norm,
        "cisternas": cisternas_norm,
        "rampa": norm(sotano.get("rampa", [])),
        "nucleo": [norm(p) for p in sotano.get("nucleo", []) if p and len(p) >= 3],
        "req_estac": sotano.get("req_estac", 0),
        "count": sotano.get("count", 0),
        "cisterna_total_m3": sotano.get("cisterna_total_m3", 0),
        "cisterna_domestico": sotano.get("cisterna_domestico", 0),
        "cisterna_aci": sotano.get("cisterna_aci", 0),
        "cisterna_maq": sotano.get("cisterna_maq", 0),
        "num_niveles": sotano.get("num_niveles", 1),
        "count_total": sotano.get("count_total", sotano.get("count", 0)),
        "deficit": sotano.get("deficit", max(0, sotano.get("req_estac", 0) - sotano.get("count", 0))),
        "niveles": [
            {
                "name": nv.get("name"),
                "stalls": [
                    {"id": st.get("id"), "coords": norm(st.get("poly", []))}
                    for st in nv.get("stalls", [])
                ],
                "aisles": [norm(a) for a in nv.get("aisles", [])],
                "count": nv.get("count", 0),
            }
            for nv in sotano.get("niveles", [])
        ],
    }

    return {
        "metadata_proyecto": {
            "terreno_area": area_terreno,
            "area_vendible_planta": r3(total_vendible),
            "area_comun_planta": area_comun,
            "area_techada_planta": area_techada,
            "eficiencia_total": eficiencia,
            "pisos": proyecto.numero_pisos,
            "altura_piso": proyecto.altura_piso or 2.80,
            "h_edificio": r3(proyecto.numero_pisos * (proyecto.altura_piso or 2.80)),
            "num_departamentos_planta": len(unidades),
            "num_departamentos_total": len(unidades) * proyecto.numero_pisos,
        },
        "normativa": {
            "pozo_luz_minimo": normativa.get("pozo_final"),
            "pozos_luz_check": normativa.get("pozos_luz_check"),
            "ascensor_obligatorio": normativa.get("ascensor_obligatorio"),
            "esc_protegida_obligatoria": normativa.get("esc_protegida_obligatoria"),
            "evacuacion_max_m": normativa.get("evacuacion_max"),
            "area_min_dpto_m2": normativa.get("area_min_dpto"),
            "estacionamiento_ancho_m": normativa.get("estacionamiento_ancho"),
            "dotaciones": normativa.get("dotaciones"),
            "variables_ignoradas": normativa.get("variables_ignoradas", []),
        },
        "geometria": {
            "lote": {"type": "polygon", "coords": lote_coords},
            "retiros": [{"id": "frontal", "coords": retiro_pts}] if retiro_pts else [],
            "unidades": unidades,
            "circulacion": {
                "hall": {"coords": hall_coords_norm},
                # "halls": una entrada por torre (dos-núcleos); torre única
                # -> fallback a la misma hall singular.
                "halls": [{"coords": norm(h)} for h in geometry.get("halls", [geometry.get("hall", [])])],
                "corridors": [{"coords": c} for c in corridor_list],
                "pasillos": [],
            },
            "nucleo": {
                "escaleras": {
                    "coords": escalera_pts,
                    "tipo": "presurizada" if esc_presurizada else "abierta",
                },
                "ascensores": [{"coords": a} for a in asc_list],
                "vestibulo": {"coords": vest_pts},
                "core_envelope": {"coords": norm(geometry.get("core", []))},
                # "nucleos": un núcleo completo (hall+escalera+ascensores+
                # vestíbulo+core) por torre real -- consumido por el
                # frontend para no dibujar ascensores/escaleras huérfanos
                # cuando hay más de una torre.
                "nucleos": [
                    {
                        "hall": {"coords": norm(n.get("hall", []))},
                        "escalera": {"coords": norm(n.get("escalera", []))},
                        "ascensores": [{"coords": norm(a)} for a in n.get("ascensores", [])],
                        "vestibulo": {"coords": norm(n.get("vestibulo", []))},
                        "core": {"coords": norm(n.get("core", []))},
                    }
                    for n in geometry.get("nucleos", [])
                ],
            },
            "tecnico": {
                "patios": (
                    [{
                        "coords": patio_pts,
                        "cumple_minimo": patio_w >= pozo_final,
                        "dimension_minima_requerida": r3(pozo_final),
                        "dimension_actual": r3(patio_w),
                    }] if patio_pts else []
                ) + (
                    [{
                        "coords": norm(geometry["patio_central"]),
                        "cumple_minimo": True,
                        "dimension_minima_requerida": r3(pozo_final),
                        "dimension_actual": r3(PATIO_CENTRAL_GAP),
                        "tipo": "patio_central",
                    }] if geometry.get("patio_central") else []
                ),
                "ductos": [{"coords": d} for d in ductos_list],
                "pozos_luz": [{"coords": p, "cumple": ok} for p, ok in pozos_luz_pairs],
                "columnas": [
                    {"coords": norm(c)}
                    for c in geometry.get("columnas", [])
                    if c and len(c) >= 3
                ],
            },
            "primer_piso": primer_piso_norm,
            "sotano": sotano_norm,
            "azotea": {
                "caja_escalera":   norm_cell((azotea or {}).get("caja_escalera", [])),
                "cuarto_maquinas": norm_cell((azotea or {}).get("cuarto_maquinas", [])),
                "tanque_elevado":  norm_cell((azotea or {}).get("tanque_elevado", [])),
                "vol_tanque_m3":   (azotea or {}).get("vol_tanque_m3", 0),
                "area_cm_m2":      (azotea or {}).get("area_cm_m2", 0),
                # Una caja/cuarto/tanque por núcleo real (dos-núcleos).
                "cajas_escalera":   [norm_cell(c) for c in (azotea or {}).get("cajas_escalera", [])],
                "cuartos_maquinas": [norm_cell(c) for c in (azotea or {}).get("cuartos_maquinas", [])],
                "tanques_elevados": [norm_cell(c) for c in (azotea or {}).get("tanques_elevados", [])],
            },
        },
        "anotaciones": anotaciones,
        # 4.1 Cuadro de áreas completo
        "cuadro_areas": _build_cuadro_areas(
            area_terreno=area_terreno,
            area_techada=area_techada,
            total_vendible=total_vendible,
            area_comun=area_comun,
            eficiencia=eficiencia,
            num_pisos=proyecto.numero_pisos,
        ),
        # 4.2 Cuadro de unidades (planta típica)
        "cuadro_unidades": [
            {
                "id": u["id"],
                "tipologia": u["metadata"]["tipologia"],
                "area_neta_m2": u["metadata"]["area"],
                "area_gross_m2": u["metadata"].get("area_gross", u["metadata"]["area"]),
                "lado": u["metadata"].get("lado", "fondo"),
                "fachada_exterior": u["validacion"].get("fachada_exterior", False),
                "dist_escalera_m": u["validacion"].get("distancia_escalera_m", 0.0),
                "dist_esc_cumple": u["validacion"].get("dist_esc_cumple", True),
                "es_reducida": u["metadata"].get("es_reducida", False),
                "cumple_area_min": u["validacion"].get("cumple_area_min", True),
            }
            for u in unidades
        ],
    }


def _build_cuadro_areas(
    area_terreno: float,
    area_techada: float,
    total_vendible: float,
    area_comun: float,
    eficiencia: float,
    num_pisos: int,
) -> dict:
    """Tabla resumen de áreas para cabida (PLAN2 §4.1).

    area_techada = huella real del edificio (incluye muros y residuos
    geométricos); area_muros_otros = techada − vendible neta − común.
    """
    area_techada_total = r3(area_techada * num_pisos)
    area_vendible_total = r3(total_vendible * num_pisos)
    area_libre = r3(max(0.0, area_terreno - area_techada))
    pct_area_libre = r3((area_libre / area_terreno * 100) if area_terreno > 0 else 0)
    cos_real = r3(area_techada / area_terreno if area_terreno > 0 else 0)
    cus_real = r3(area_techada_total / area_terreno if area_terreno > 0 else 0)
    area_muros_otros = r3(max(0.0, area_techada - total_vendible - area_comun))
    return {
        "area_terreno_m2": r3(area_terreno),
        "area_libre_planta_m2": area_libre,
        "pct_area_libre": pct_area_libre,
        "area_techada_planta_m2": r3(area_techada),
        "area_techada_total_m2": area_techada_total,
        "area_vendible_planta_m2": r3(total_vendible),
        "area_vendible_total_m2": area_vendible_total,
        "area_comun_planta_m2": r3(area_comun),
        "area_muros_otros_m2": area_muros_otros,
        "eficiencia_pct": eficiencia,
        "cos_real": cos_real,
        "cus_real": cus_real,
        "num_pisos": num_pisos,
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════════════════════════

@app.post("/auditoria-rne")
async def validar_arquitectura(
    proyecto: ProyectoInmobiliario,
):
    """
    Endpoint principal.
    Retorna un payload JSON normalizado con estructura semántica lista
    para Three.js / WebGL.

    Generación conforme: el número de pisos se ajusta automáticamente al
    máximo permitido por altura, CUS y densidad de la zona efectiva
    (tabla + overrides del certificado). El ajuste se reporta en `clamps`.
    """
    # ── Zona efectiva: tabla + overrides del certificado ──
    _alt_piso = proyecto.altura_piso or RNE["altura_piso"]
    zona_overrides: Dict[str, Any] = {}
    if proyecto.cus_maximo and proyecto.cus_maximo > 0:
        zona_overrides["cus"] = float(proyecto.cus_maximo)
    if proyecto.altura_maxima_pisos and proyecto.altura_maxima_pisos > 0:
        zona_overrides["altura_max_m"] = float(proyecto.altura_maxima_pisos) * _alt_piso
    if proyecto.densidad_maxima_hab_ha and proyecto.densidad_maxima_hab_ha > 0:
        zona_overrides["densidad_max_hab_ha"] = float(proyecto.densidad_maxima_hab_ha)
    if proyecto.area_libre_min_pct and proyecto.area_libre_min_pct > 0:
        zona_overrides["area_libre_min"] = float(proyecto.area_libre_min_pct) / 100.0

    zona_eff = dict(get_zona(proyecto.zonificacion))
    for k, v in zona_overrides.items():
        if k in zona_eff:
            zona_eff[k] = v

    # ── Clamp 1 (pre-generación): pisos por altura máxima ──
    ajustar = bool(proyecto.ajustar_pisos_normativa)
    pisos_solicitados = proyecto.numero_pisos
    pisos_max_altura = max(1, int(zona_eff["altura_max_m"] / _alt_piso + 1e-6))
    pisos_eff = min(pisos_solicitados, pisos_max_altura) if ajustar else pisos_solicitados
    limitado_por = set()
    if pisos_eff < pisos_solicitados:
        limitado_por.add("altura")

    # ── Clamps 2-3 (iterativos): pisos por CUS y densidad ──
    # La huella y el mix de tipologías se conocen tras generar; al bajar pisos
    # cambia pozo_final (H/4) y puede cambiar la huella → iterar hasta converger.
    pisos_max_cus = pisos_eff
    pisos_max_dens = pisos_eff
    for _clamp_it in range(3):
        try:
            proyecto_eff = proyecto.model_copy(update={"numero_pisos": pisos_eff})
        except AttributeError:  # pydantic v1
            proyecto_eff = proyecto.copy(update={"numero_pisos": pisos_eff})

        try:
            geometry, normativa = _generate_geometry(proyecto_eff)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        primer_piso_data = _generate_primer_piso(proyecto_eff, geometry)
        sotano_data      = _generate_sotano(proyecto_eff, geometry, normativa)
        azotea_data      = _generate_azotea(proyecto_eff, geometry, normativa)
        webgl_payload = _build_webgl_payload(
            proyecto_eff, geometry, normativa, primer_piso_data, sotano_data, azotea_data
        )

        meta = webgl_payload["metadata_proyecto"]
        _terreno = meta["terreno_area"]
        _huella = meta["area_techada_planta"]
        _tip_planta = [u["metadata"]["tipologia"] for u in webgl_payload["geometria"]["unidades"]]
        _hab_planta = sum(HAB_PROMEDIO_POR_DEPTO.get(t, 3) for t in _tip_planta)
        _area_ha = _terreno / 10_000.0 if _terreno > 0 else 0.0

        pisos_max_cus = (
            max(1, int(zona_eff["cus"] * _terreno / _huella + 1e-6))
            if _huella > 0 else pisos_eff
        )
        pisos_max_dens = (
            max(1, int(zona_eff["densidad_max_hab_ha"] * _area_ha / _hab_planta + 1e-6))
            if _hab_planta > 0 else pisos_eff
        )
        _nuevo = max(1, min(pisos_eff, pisos_max_cus, pisos_max_dens))
        if ajustar and _nuevo < pisos_eff:
            if pisos_max_cus < pisos_eff:
                limitado_por.add("cus")
            if pisos_max_dens < pisos_eff:
                limitado_por.add("densidad")
            pisos_eff = _nuevo
            continue
        break

    clamps_info = {
        "ajuste_automatico": ajustar,
        "pisos_solicitados": pisos_solicitados,
        "pisos_efectivos": pisos_eff,
        "limitado_por": sorted(limitado_por),
        "pisos_max_altura": pisos_max_altura,
        "pisos_max_cus": pisos_max_cus,
        "pisos_max_densidad": pisos_max_dens,
        "pisos_max_normativa": max(1, min(pisos_max_altura, pisos_max_cus, pisos_max_dens)),
    }

    # ── Chequeo de zonificación (CUS, altura, área libre, retiros, frente, densidad) ──
    area_terreno_m2 = meta["terreno_area"]
    area_techada_planta = meta["area_techada_planta"]
    area_techada_total = area_techada_planta * pisos_eff
    altura_edif = meta["h_edificio"]
    area_libre_planta = max(0.0, area_terreno_m2 - area_techada_planta)

    retiro_lat_dev = proyecto.retiro_lateral if proyecto.retiro_lateral is not None else 2.30
    retiro_pos_dev = proyecto.retiro_posterior if proyecto.retiro_posterior is not None else 2.30
    # Exención por medianería: lado ciego = colindancia → retiro no exigible en ese lado.
    # El check usa el menor retiro de los lados NO ciegos; si ambos son ciegos, no aplica.
    _lat_libres = [
        retiro_lat_dev
        for ciego in (proyecto.ciego_izquierda, proyecto.ciego_derecha)
        if not ciego
    ]
    retiro_lat_aplica = len(_lat_libres) > 0
    retiro_lat_min_aplic = min(_lat_libres) if _lat_libres else 0.0
    retiro_pos_aplica = not bool(proyecto.ciego_fondo)
    retiro_pos_aplic = retiro_pos_dev if retiro_pos_aplica else 0.0

    tipologias_total = [
        u["metadata"]["tipologia"]
        for u in webgl_payload["geometria"]["unidades"]
    ] * pisos_eff
    num_unidades_total = meta["num_departamentos_total"]

    zon_check = validar_zonificacion(
        zona_codigo=proyecto.zonificacion,
        area_terreno_m2=area_terreno_m2,
        frente_m=proyecto.frente or 0.0,
        area_techada_total_m2=area_techada_total,
        altura_edificio_m=altura_edif,
        area_libre_planta_m2=area_libre_planta,
        retiro_frontal_m=proyecto.retiro_frontal,
        retiro_lateral_min_m=retiro_lat_min_aplic,
        retiro_posterior_m=retiro_pos_aplic,
        num_unidades_total=num_unidades_total,
        tipologias=tipologias_total,
        retiro_lateral_aplica=retiro_lat_aplica,
        retiro_posterior_aplica=retiro_pos_aplica,
        overrides=zona_overrides or None,
    )
    zon_check["area_techada_total_m2"] = r3(area_techada_total)
    zon_check["area_techada_planta_m2"] = r3(area_techada_planta)
    zon_check["area_libre_planta_m2"] = r3(area_libre_planta)

    webgl_payload["zonificacion_check"] = zon_check
    webgl_payload["clamps"] = clamps_info
    webgl_payload["normativa"]["zonificacion"] = {
        "codigo": zon_check["zona"],
        "nombre": zon_check["zona_nombre"],
        "parametros": zon_check["parametros_zona"],
    }
    normativa["zonificacion_check"] = zon_check
    normativa["clamps"] = clamps_info

    # ── Mivivienda: área mín. + rango de precio por unidad (opt-in) ──
    mivivienda_check = _check_mivivienda(
        proyecto, [u["metadata"] for u in webgl_payload["geometria"]["unidades"]]
    )
    if mivivienda_check is not None:
        webgl_payload["mivivienda_check"] = mivivienda_check
        normativa["mivivienda_check"] = mivivienda_check

    # ── Compatibilidad hacia atrás (campos que el JS actual consume) ──
    # Poda de campos que ningún consumidor (main.js / viewer3d.js) lee:
    #  - columnas, remanentes_zona_media, esquema_area_libre, pozos_luz_cumple
    #  - evaluacion: redundante con el campo top-level "diseno" (abajo)
    _GG_DEAD = {"columnas", "remanentes_zona_media", "esquema_area_libre",
                "pozos_luz_cumple", "evaluacion"}
    geometria_generada = {k: v for k, v in geometry.items() if k not in _GG_DEAD}

    response = {
        **webgl_payload,
        # Backward-compat fields consumed by the existing main.js
        "status": "Auditoría RNE — Spine & Ribs (WebGL Mode)",
        "geometria_generada": geometria_generada,
        "normativa_estricta": normativa,
        "primer_piso": primer_piso_data,
        "sotano": sotano_data,
        # E5: auto-crítica del diseño
        "diseno": geometry.get("evaluacion", {"score": 0, "defectos": [], "n_criticos": 0, "metricas": {}}),
    }

    return response