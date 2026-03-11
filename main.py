"""
main.py — Motor de auditoría RNE + renderizado arquitectónico en Python.
FastAPI backend que genera geometría Y renderiza la planta con matplotlib.
"""

import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon
from shapely.ops import unary_union

from renderer import (
    render_planta_tipica,
    render_primer_piso,
    render_sotano,
    calc_poly_area as renderer_calc_area,
    get_typology,
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN MAESTRA RNE (Reglamento Nacional de Edificaciones)
# ═══════════════════════════════════════════════════════════════
RNE = {
    "departamentos": {"min_multifamiliar": 40.0, "min_unipersonal": 16.0, "h_libre": 2.30},
    "pozos_luz": {"min_abs": 2.10, "ratio_dorm": 0.25},
    "circulacion_h": {"hall_ancho": 1.20, "interior": 0.90},
    "circulacion_v": {
        "esc_ancho": 1.20, "esc_largo": 5.60,
        "evacuacion_max": 45.0,
        "h_max_sin_esc_prot": 15.0, "h_max_sin_ascensor": 12.0,
    },
    "ascensor": {"ancho": 2.00, "largo": 2.00},
    "estacionamientos": {"ancho_ind": 2.70, "largo": 5.00, "maniobra": 6.00},
    "instalaciones": {"aci_m3": 25.0, "agua_1d": 500.0, "agua_2d": 850.0, "agua_3d": 1200.0},
    "altura_piso": 2.80,
}


class ProyectoInmobiliario(BaseModel):
    coordenadas_lote: List[Tuple[float, float]]
    area_bruta_terreno: float
    numero_pisos: int
    retiro_frontal: float
    zonificacion: str
    num_ascensores: int
    num_departamentos: int
    # Parámetros adicionales para renderizado
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


def poly_to_js(sp):
    """Shapely Polygon → [{x,y}, …] for JS."""
    if sp is None or sp.is_empty:
        return []
    if sp.geom_type == "MultiPolygon":
        sp = max(sp.geoms, key=lambda g: g.area)
    return [{"x": round(x, 3), "y": round(y, 3)} for x, y in list(sp.exterior.coords)[:-1]]


def safe_clip(poly, boundary):
    try:
        r = poly.intersection(boundary)
        if r.is_empty:
            return None
        if r.geom_type == "GeometryCollection":
            ps = [g for g in r.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not ps:
                return None
            r = max(ps, key=lambda g: g.area)
        return r
    except Exception:
        return poly


def make_rect(cx, cy, dx_l, dy_l, dx_s, dy_s, half_l, half_s):
    """Create a rectangle from center, direction vectors and half-extents."""
    return Polygon([
        (cx - dx_l * half_l - dx_s * half_s, cy - dy_l * half_l - dy_s * half_s),
        (cx + dx_l * half_l - dx_s * half_s, cy + dy_l * half_l - dy_s * half_s),
        (cx + dx_l * half_l + dx_s * half_s, cy + dy_l * half_l + dy_s * half_s),
        (cx - dx_l * half_l + dx_s * half_s, cy - dy_l * half_l + dy_s * half_s),
    ])


def _interpolate(pA, pB, t):
    """Interpolate between two points."""
    return {
        "x": pA["x"] + (pB["x"] - pA["x"]) * t,
        "y": pA["y"] + (pB["y"] - pA["y"]) * t,
    }


def _get_cell(quad, u1, u2, v1, v2):
    """Get a cell from a quad polygon (bilinear interpolation)."""
    def _gp(u, v):
        top = _interpolate(quad[0], quad[1], u)
        bot = _interpolate(quad[3], quad[2], u)
        return _interpolate(top, bot, v)
    return [_gp(u1, v1), _gp(u2, v1), _gp(u2, v2), _gp(u1, v2)]


def _calculate_poly_area(poly):
    """Calculate area with Shoelace formula for [{x,y}...] format."""
    n = len(poly)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i]["x"] * poly[j]["y"]
        area -= poly[j]["x"] * poly[i]["y"]
    return abs(area) / 2


def _poly_width(poly):
    if len(poly) < 4:
        return 0
    dx = poly[1]["x"] - poly[0]["x"]
    dy = poly[1]["y"] - poly[0]["y"]
    return math.hypot(dx, dy)


def _generate_geometry(proyecto: ProyectoInmobiliario):
    """Core geometry generation — shared between audit and render endpoints."""
    lote = Polygon(proyecto.coordenadas_lote)
    if not lote.is_valid:
        lote = lote.buffer(0)

    h_edif = proyecto.numero_pisos * (proyecto.altura_piso or RNE["altura_piso"])
    num_dptos = max(2, proyecto.num_departamentos)
    num_asc = max(0, proyecto.num_ascensores)

    # ── 1. NORMATIVOS ──
    pozo_final = max(RNE["pozos_luz"]["min_abs"], h_edif * RNE["pozos_luz"]["ratio_dorm"])
    nec_ascensor = h_edif > RNE["circulacion_v"]["h_max_sin_ascensor"]
    nec_esc_prot = h_edif > RNE["circulacion_v"]["h_max_sin_esc_prot"]

    # ── 2. ORIENTACIÓN (MRR) ──
    mrr = lote.minimum_rotated_rectangle
    mc = list(mrr.exterior.coords)
    d01 = math.hypot(mc[1][0] - mc[0][0], mc[1][1] - mc[0][1])
    d12 = math.hypot(mc[2][0] - mc[1][0], mc[2][1] - mc[1][1])

    if d01 >= d12:
        long_len, short_len = d01, d12
        ang = math.atan2(mc[1][1] - mc[0][1], mc[1][0] - mc[0][0])
    else:
        long_len, short_len = d12, d01
        ang = math.atan2(mc[2][1] - mc[1][1], mc[2][0] - mc[1][0])

    cx, cy = lote.centroid.x, lote.centroid.y
    dl_x, dl_y = math.cos(ang), math.sin(ang)
    ds_x, ds_y = -dl_y, dl_x

    half_L = long_len / 2
    half_S = short_len / 2
    hw = RNE["circulacion_h"]["hall_ancho"] / 2

    # ── 3. HALL ──
    hall_poly = make_rect(cx, cy, dl_x, dl_y, ds_x, ds_y, half_L, hw)
    hall_clipped = safe_clip(hall_poly, lote)

    # ── 4. CORE ──
    esc_w = RNE["circulacion_v"]["esc_ancho"]
    esc_half_l = 2.50 / 2
    esc_depth = esc_w * 2

    esc_center_s = hw + esc_depth / 2
    stair_poly = Polygon([
        (cx - dl_x * esc_half_l + ds_x * hw,           cy - dl_y * esc_half_l + ds_y * hw),
        (cx + dl_x * esc_half_l + ds_x * hw,           cy + dl_y * esc_half_l + ds_y * hw),
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
            (ac[0] - dl_x * asc_l / 2 + ds_x * hw,              ac[1] - dl_y * asc_l / 2 + ds_y * hw),
            (ac[0] + dl_x * asc_l / 2 + ds_x * hw,              ac[1] + dl_y * asc_l / 2 + ds_y * hw),
            (ac[0] + dl_x * asc_l / 2 + ds_x * (hw + asc_w),    ac[1] + dl_y * asc_l / 2 + ds_y * (hw + asc_w)),
            (ac[0] - dl_x * asc_l / 2 + ds_x * (hw + asc_w),    ac[1] - dl_y * asc_l / 2 + ds_y * (hw + asc_w)),
        ])
        asc_polys.append(asc_poly)

    core_items = [stair_poly] + asc_polys
    core_envelope = unary_union(core_items).envelope if core_items else stair_poly
    core_clipped = safe_clip(core_envelope, lote)

    vest_poly = None
    if nec_esc_prot:
        vest_poly = Polygon([
            (cx - dl_x * esc_half_l - dl_x * 1.50 + ds_x * hw,
             cy - dl_y * esc_half_l - dl_y * 1.50 + ds_y * hw),
            (cx - dl_x * esc_half_l + ds_x * hw,
             cy - dl_y * esc_half_l + ds_y * hw),
            (cx - dl_x * esc_half_l + ds_x * (hw + esc_depth),
             cy - dl_y * esc_half_l + ds_y * (hw + esc_depth)),
            (cx - dl_x * esc_half_l - dl_x * 1.50 + ds_x * (hw + esc_depth),
             cy - dl_y * esc_half_l - dl_y * 1.50 + ds_y * (hw + esc_depth)),
        ])

    # ── 5. PATIO DE LUCES ──
    patio_dim = min(pozo_final, short_len * 0.35)
    patio_half = patio_dim / 2
    patio_poly = Polygon([
        (cx - dl_x * patio_half - ds_x * hw,              cy - dl_y * patio_half - ds_y * hw),
        (cx + dl_x * patio_half - ds_x * hw,              cy + dl_y * patio_half - ds_y * hw),
        (cx + dl_x * patio_half - ds_x * (hw + patio_dim), cy + dl_y * patio_half - ds_y * (hw + patio_dim)),
        (cx - dl_x * patio_half - ds_x * (hw + patio_dim), cy - dl_y * patio_half - ds_y * (hw + patio_dim)),
    ])
    patio_clipped = safe_clip(patio_poly, lote)

    # ── 6. DUCTOS ──
    duct_dim = 1.50
    duct_half = duct_dim / 2
    ductos = []
    for sign in [-1, 1]:
        duct_offset_l = half_L * 0.45 * sign
        dc = (cx + dl_x * duct_offset_l, cy + dl_y * duct_offset_l)
        d_poly = Polygon([
            (dc[0] - dl_x * duct_half - ds_x * hw,              dc[1] - dl_y * duct_half - ds_y * hw),
            (dc[0] + dl_x * duct_half - ds_x * hw,              dc[1] + dl_y * duct_half - ds_y * hw),
            (dc[0] + dl_x * duct_half - ds_x * (hw + duct_dim), dc[1] + dl_y * duct_half - ds_y * (hw + duct_dim)),
            (dc[0] - dl_x * duct_half - ds_x * (hw + duct_dim), dc[1] - dl_y * duct_half - ds_y * (hw + duct_dim)),
        ])
        clipped = safe_clip(d_poly, lote)
        if clipped:
            ductos.append(clipped)

    # ── 7. DEPARTAMENTOS ──
    dptos_a = max(1, num_dptos // 2)
    dptos_b = max(1, num_dptos - dptos_a)
    apartments = []

    def distribute_units(L_min, L_max, exclude_min, exclude_max, num_units, sign_s):
        units = []
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

            if n2 == 0 and s2 > max(3.0, s1/max(1,n1) if n1 else 0) and num_units >= 2:
                n2 = 1; n1 -= 1
            if n1 == 0 and s1 > max(3.0, s2/max(1,n2) if n2 else 0) and num_units >= 2:
                n1 = 1; n2 -= 1

            segments = []
            if n1 > 0: segments.append((L_min, e_min, n1))
            if n2 > 0: segments.append((e_max, L_max, n2))

        for seg_start, seg_end, n in segments:
            if n <= 0: continue
            w = (seg_end - seg_start) / n
            for i in range(n):
                off = seg_start + i * w
                nxt = off + w
                if sign_s > 0:
                    corners = [
                        (cx + dl_x * off + ds_x * hw,       cy + dl_y * off + ds_y * hw),
                        (cx + dl_x * nxt + ds_x * hw,        cy + dl_y * nxt + ds_y * hw),
                        (cx + dl_x * nxt + ds_x * half_S,    cy + dl_y * nxt + ds_y * half_S),
                        (cx + dl_x * off + ds_x * half_S,    cy + dl_y * off + ds_y * half_S),
                    ]
                else:
                    corners = [
                        (cx + dl_x * off - ds_x * hw,       cy + dl_y * off - ds_y * hw),
                        (cx + dl_x * nxt - ds_x * hw,        cy + dl_y * nxt - ds_y * hw),
                        (cx + dl_x * nxt - ds_x * half_S,    cy + dl_y * nxt - ds_y * half_S),
                        (cx + dl_x * off - ds_x * half_S,    cy + dl_y * off - ds_y * half_S),
                    ]
                ap = safe_clip(Polygon(corners), lote)
                if ap and ap.area >= 5:
                    units.append(ap)
        return units

    core_min_L = -esc_half_l - 0.20
    core_max_L = (esc_half_l + 0.20 + asc_l / 2 + (num_asc - 1) * (asc_l + 0.30) + asc_l/2) if num_asc > 0 else esc_half_l + 0.20

    apartments.extend(distribute_units(-half_L, half_L, core_min_L, core_max_L, dptos_a, 1))
    patio_min_L = -patio_half - 0.20
    patio_max_L = patio_half + 0.20
    apartments.extend(distribute_units(-half_L, half_L, patio_min_L, patio_max_L, dptos_b, -1))

    # ── BUILD GEOMETRY DICT ──
    geometry = {
        "hall": poly_to_js(hall_clipped),
        "core": poly_to_js(core_clipped),
        "escalera": poly_to_js(safe_clip(stair_poly, lote)),
        "ascensores": [poly_to_js(safe_clip(a, lote)) for a in asc_polys],
        "vestibulo": poly_to_js(safe_clip(vest_poly, lote)) if vest_poly else [],
        "patio": poly_to_js(patio_clipped),
        "ductos": [poly_to_js(d) for d in ductos],
        "departamentos": [poly_to_js(a) for a in apartments],
    }

    normativa = {
        "pozo_final": round(pozo_final, 2),
        "ascensor_obligatorio": nec_ascensor,
        "esc_protegida_obligatoria": nec_esc_prot,
        "evacuacion_max": RNE["circulacion_v"]["evacuacion_max"],
        "area_min_dpto": RNE["departamentos"]["min_multifamiliar"],
        "dotaciones": RNE["instalaciones"],
        "estacionamiento_ancho": RNE["estacionamientos"]["ancho_ind"],
    }

    return geometry, normativa


def _generate_primer_piso(proyecto: ProyectoInmobiliario, geometry: dict):
    """Generate first floor elements (lobby, commerce, ramp)."""
    frente = proyecto.frente or 10
    fondo_val = proyecto.fondo or 10
    derecha = proyecto.derecha or 20
    izquierda = proyecto.izquierda or 20
    retiro_frontal = proyecto.retiro_frontal

    p1 = {"x": -frente / 2, "y": 0}
    p2 = {"x": frente / 2, "y": 0}
    p3 = {"x": fondo_val / 2, "y": derecha}
    p4 = {"x": -fondo_val / 2, "y": izquierda}
    polygon_pts = [p1, p2, p3, p4]

    rt_y = min(retiro_frontal, derecha, izquierda)
    if rt_y > 0 and izquierda > 0 and derecha > 0:
        u_l = rt_y / izquierda
        u_r = rt_y / derecha
        pr3 = _interpolate(p2, p3, u_r)
        pr4 = _interpolate(p1, p4, u_l)
        techada_poly = [pr4, pr3, p3, p4]
    else:
        techada_poly = [p1, p2, p3, p4]

    # Apply lateral setbacks
    retiro_lat = 2.30
    frente_neto = max(1, frente)
    fondo_neto = max(1, (derecha + izquierda) / 2)

    u_left = 0 if (proyecto.ciego_izquierda) else retiro_lat / frente_neto
    u_right = 1.0 if (proyecto.ciego_derecha) else 1.0 - (retiro_lat / frente_neto)
    v_bottom = 1.0 if (proyecto.ciego_fondo) else 1.0 - (retiro_lat / fondo_neto)

    lote_neto = _get_cell(techada_poly, u_left, u_right, 0, v_bottom)

    b_w = max(1, _poly_width(lote_neto))
    b_d = max(1, math.hypot(
        lote_neto[2]["x"] - lote_neto[1]["x"],
        lote_neto[2]["y"] - lote_neto[1]["y"]
    ))

    rampa_u = 3.00 / b_w
    rampa_v = min(1.0, 20.00 / b_d)
    rampa = _get_cell(lote_neto, 0, rampa_u, 0, rampa_v)

    u_lobby_start = max(rampa_u + 0.02, 0.38)
    u_lobby_end = min(1.0, 0.62)

    servicios = _get_cell(lote_neto, 0, rampa_u, rampa_v + 0.01, rampa_v + 0.16)
    lobby = _get_cell(lote_neto, u_lobby_start, u_lobby_end, 0, 0.3)
    com1 = _get_cell(lote_neto, rampa_u + 0.01, u_lobby_start, 0, 0.3)
    com2 = _get_cell(lote_neto, u_lobby_end, 1.0, 0, 0.3)

    return {
        "comercios": [com1, com2],
        "servicios": servicios,
        "lobby": lobby,
        "rampa": rampa,
    }


def _generate_sotano(proyecto: ProyectoInmobiliario, geometry: dict, normativa: dict):
    """Generate basement level with parking stalls."""
    frente = proyecto.frente or 10
    fondo_val = proyecto.fondo or 10
    derecha = proyecto.derecha or 20
    izquierda = proyecto.izquierda or 20

    p1 = {"x": -frente / 2, "y": 0}
    p2 = {"x": frente / 2, "y": 0}
    p3 = {"x": fondo_val / 2, "y": derecha}
    p4 = {"x": -fondo_val / 2, "y": izquierda}
    polygon_pts = [p1, p2, p3, p4]

    # Inset for basement
    inset = 0.30
    slab = [
        {"x": p1["x"] + inset, "y": p1["y"] + inset},
        {"x": p2["x"] - inset, "y": p2["y"] + inset},
        {"x": p3["x"] - inset, "y": p3["y"] - inset},
        {"x": p4["x"] + inset, "y": p4["y"] - inset},
    ]

    dptos = geometry.get("departamentos", [])
    num_dptos = len(dptos) if dptos else proyecto.num_departamentos
    total_dptos = num_dptos * proyecto.numero_pisos
    pct_estac = proyecto.pct_estac or 30
    req_estac = math.ceil(total_dptos * (pct_estac / 100))

    stall_w = normativa.get("estacionamiento_ancho", 2.70)
    stall_d = 5.00
    aisle_w = 6.00

    # Generate parking stalls in a grid
    xs = [p["x"] for p in slab]
    ys = [p["y"] for p in slab]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    stalls = []
    aisles = []
    stall_num = 1
    remaining = req_estac
    y_cursor = min_y

    while remaining > 0 and y_cursor + stall_d <= max_y:
        # Top row
        x_cursor = min_x
        row_placed = False
        while x_cursor + stall_w <= max_x and remaining > 0:
            stall = [
                {"x": round(x_cursor, 3), "y": round(y_cursor, 3)},
                {"x": round(x_cursor + stall_w, 3), "y": round(y_cursor, 3)},
                {"x": round(x_cursor + stall_w, 3), "y": round(y_cursor + stall_d, 3)},
                {"x": round(x_cursor, 3), "y": round(y_cursor + stall_d, 3)},
            ]
            stalls.append({"id": f"E-{stall_num:02d}", "poly": stall})
            stall_num += 1
            remaining -= 1
            row_placed = True
            x_cursor += stall_w

        if row_placed:
            # Aisle
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

            # Bottom row (sharing aisle)
            if remaining > 0 and y_cursor + stall_d <= max_y:
                x_cursor = min_x
                while x_cursor + stall_w <= max_x and remaining > 0:
                    stall = [
                        {"x": round(x_cursor, 3), "y": round(y_cursor, 3)},
                        {"x": round(x_cursor + stall_w, 3), "y": round(y_cursor, 3)},
                        {"x": round(x_cursor + stall_w, 3), "y": round(y_cursor + stall_d, 3)},
                        {"x": round(x_cursor, 3), "y": round(y_cursor + stall_d, 3)},
                    ]
                    stalls.append({"id": f"E-{stall_num:02d}", "poly": stall})
                    stall_num += 1
                    remaining -= 1
                    x_cursor += stall_w
                y_cursor += stall_d
        else:
            y_cursor += 1.0

    # Cisterna calculation
    dot = normativa.get("dotaciones", RNE["instalaciones"])
    agua_1d = dot.get("agua_1d", 500) / 1000
    agua_2d = dot.get("agua_2d", 850) / 1000
    agua_3d = dot.get("agua_3d", 1200) / 1000
    aci_m3 = dot.get("aci_m3", 25)

    dom = 0
    for dpto in dptos:
        if dpto and len(dpto) >= 3:
            area = renderer_calc_area(dpto)
            typ = get_typology(area)
            m3 = agua_1d if typ == "1D" else (agua_2d if typ == "2D" else agua_3d)
            dom += m3 * proyecto.numero_pisos

    total_cist = dom + aci_m3
    cuarto_maq = max(15, total_cist * 0.12)

    # Rampa in basement
    rampa = [
        {"x": round(min_x, 3), "y": round(min_y, 3)},
        {"x": round(min_x + 3.0, 3), "y": round(min_y, 3)},
        {"x": round(min_x + 3.0, 3), "y": round(max_y, 3)},
        {"x": round(min_x, 3), "y": round(max_y, 3)},
    ]

    # ── CISTERNA SUBDIVISION (IS.010) ──
    cisternas = []
    if total_cist > 0:
        cist_depth = 2.5  # tank depth in meters
        area_dom_a = (dom / 2) / cist_depth
        area_dom_b = (dom / 2) / cist_depth
        area_aci = aci_m3 / cist_depth
        area_maq = cuarto_maq
        total_cist_area = area_dom_a + area_dom_b + area_aci + area_maq

        # Place cisterns at the bottom of the slab, full width
        cist_width = max_x - min_x
        if cist_width > 0 and total_cist_area > 0:
            cist_height = total_cist_area / cist_width
            cist_top = max_y - cist_height

            # Subdivide horizontally (stacking from bottom up)
            zones = [
                {"label": f"CIST. CONS. A\n{dom/2:.1f} m3", "area": area_dom_a,
                 "fill": "#bfdbfe", "stroke": "#2563eb"},
                {"label": f"CIST. CONS. B\n{dom/2:.1f} m3", "area": area_dom_b,
                 "fill": "#93c5fd", "stroke": "#1d4ed8"},
                {"label": f"CIST. ACI\n{aci_m3:.1f} m3", "area": area_aci,
                 "fill": "#fca5a5", "stroke": "#dc2626"},
                {"label": f"CTO. MAQ.\n{cuarto_maq:.1f} m2", "area": area_maq,
                 "fill": "#fef3c7", "stroke": "#d97706"},
            ]

            cursor_y = cist_top
            for zone in zones:
                zone_h = zone["area"] / cist_width if cist_width > 0 else 1
                poly = [
                    {"x": round(min_x, 3), "y": round(cursor_y, 3)},
                    {"x": round(max_x, 3), "y": round(cursor_y, 3)},
                    {"x": round(max_x, 3), "y": round(cursor_y + zone_h, 3)},
                    {"x": round(min_x, 3), "y": round(cursor_y + zone_h, 3)},
                ]
                cisternas.append({
                    "poly": poly,
                    "label": zone["label"],
                    "fill": zone["fill"],
                    "stroke": zone["stroke"],
                })
                cursor_y += zone_h

    return {
        "name": "S1",
        "slab": slab,
        "stalls": stalls,
        "aisles": aisles,
        "count": len(stalls),
        "rampa": rampa,
        "cisternas": cisternas,
        "req_estac": req_estac,
        "cisterna_total_m3": round(total_cist, 1),
        "cisterna_domestico": round(dom, 1),
        "cisterna_aci": round(aci_m3, 1),
        "cisterna_maq": round(cuarto_maq, 1),
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/auditoria-rne")
async def validar_arquitectura(proyecto: ProyectoInmobiliario):
    """Original audit endpoint — now also generates rendered images."""
    geometry, normativa = _generate_geometry(proyecto)

    # Build params dict for renderer
    render_params = {
        "frente": proyecto.frente,
        "fondo": proyecto.fondo,
        "derecha": proyecto.derecha,
        "izquierda": proyecto.izquierda,
        "retiro_frontal": proyecto.retiro_frontal,
        "pisos": proyecto.numero_pisos,
        "altura_piso": proyecto.altura_piso or 2.80,
    }

    data = {
        "geometria_generada": geometry,
        "normativa_estricta": normativa,
    }

    # Render planta típica
    img_tipica = render_planta_tipica(data, render_params)

    # Render primer piso
    primer_piso_data = _generate_primer_piso(proyecto, geometry)
    img_primer = render_primer_piso(data, render_params, primer_piso_data)

    # Render sótano
    sotano_data = _generate_sotano(proyecto, geometry, normativa)
    img_sotano = render_sotano(data, render_params, sotano_data)

    return {
        "status": "Auditoría RNE — Spine & Ribs",
        "geometria_generada": geometry,
        "normativa_estricta": normativa,
        "primer_piso": primer_piso_data,
        "sotano": sotano_data,
        "renders": {
            "planta_tipica": img_tipica,
            "primer_piso": img_primer,
            "sotano": img_sotano,
        }
    }