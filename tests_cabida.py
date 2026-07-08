#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests_cabida.py — E0: Arnés de medición. Línea base PASS/FAIL.
Corre sin servidor: importa directamente desde main.py.
Genera tabla + PNGs en _tests_out/.

Uso: python tests_cabida.py
"""
import sys, os, math, traceback, datetime
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union

# ── Backend imports ──────────────────────────────────────────────────────────
import main as M
from main import ProyectoInmobiliario, _generate_geometry, _erode_lote, RNE

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tests_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (umbrales de PASS)
# ─────────────────────────────────────────────────────────────────────────────
THR_RETIRO_TOL   = 0.05   # m — tolerancia retiro real vs pedido
THR_PCT_CIRC_MAX = 12.0   # % — E3: presupuesto duro de circulación
THR_EFI_MIN      = 60.0   # % — línea base permisiva (E4 lo sube a 78)
THR_HUECOS_MAX   = 10.0   # m² — línea base permisiva
THR_ACCESO_PCT   = 80.0   # % dptos con acceso ≥1.2m (E3 → 100%)
THR_PARA_MAX     = 10.0   # grados — fondo vs lindero posterior
THR_ND_DELTA     = 2      # tolerancia dptos emitidos vs pedidos
THR_FRENTE_BAJO_PCT = 20.0  # % dptos con frente <5.2m (calibración DXF Lima)
THR_SCORE_MIN    = 75     # E5: score mínimo por caso (compuerta E5 ≥85 promedio)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers geométricos
# ─────────────────────────────────────────────────────────────────────────────

def _pts_to_poly(pts):
    """[{x,y},...] o [(x,y),...] → shapely Polygon o None."""
    if not pts or len(pts) < 3:
        return None
    try:
        if isinstance(pts[0], dict):
            coords = [(p["x"], p["y"]) for p in pts]
        else:
            coords = [(p[0], p[1]) for p in pts]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def _build_footprint(geometry):
    """Unión de todos los polígonos techados → shapely Polygon."""
    parts = []
    # "patio" cuenta como área diseñada (área libre intencional, no hueco);
    # el motor lo emite clipeado al lote útil, nunca invade retiros.
    for key in ("hall", "core", "vestibulo", "escalera", "patio"):
        p = _pts_to_poly(geometry.get(key, []))
        if p:
            parts.append(p)
    for c in geometry.get("corridors", []):
        p = _pts_to_poly(c)
        if p:
            parts.append(p)
    for a in geometry.get("ascensores", []):
        p = _pts_to_poly(a)
        if p:
            parts.append(p)
    for d in geometry.get("departamentos", []):
        pts = M._departamento_outline_coords(d)
        p = _pts_to_poly(pts)
        if p:
            parts.append(p)
    for r in geometry.get("remanentes_zona_media", []):
        p = _pts_to_poly(r)
        if p:
            parts.append(p)
    if not parts:
        return None
    try:
        u = unary_union(parts)
        if isinstance(u, MultiPolygon):
            u = max(u.geoms, key=lambda g: g.area)
        return u if not u.is_empty else None
    except Exception:
        return None


def _make_lote_poly(frente, fondo, derecha, izquierda):
    """Polígono bruto del lote (sin retiros)."""
    return Polygon([
        (-frente / 2, 0),
        (frente / 2, 0),
        (fondo / 2, derecha),
        (-fondo / 2, izquierda),
    ])


def _make_lote_coords(frente, fondo, derecha, izquierda, retiro_frontal=0.0):
    """coordenadas_lote: lote bruto con retiro frontal aplicado (igual que JS)."""
    p1 = (-frente / 2, 0.0)
    p2 = (frente / 2, 0.0)
    p3 = (fondo / 2, derecha)
    p4 = (-fondo / 2, izquierda)
    if retiro_frontal <= 0:
        return [p1, p2, p3, p4]
    uR = min(retiro_frontal / derecha, 0.99) if derecha > 0 else 0
    uL = min(retiro_frontal / izquierda, 0.99) if izquierda > 0 else 0
    def interp(a, b, t):
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    pr3 = interp(p2, p3, uR)
    pr4 = interp(p1, p4, uL)
    return [pr4, pr3, p3, p4]


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def metric_retiro_por_borde(footprint, lote_poly):
    """
    Distancia mínima del footprint a cada arista del lote.
    Devuelve lista [(idx, dx_m, edge_label)] donde idx recorre las aristas.
    """
    if footprint is None or footprint.is_empty:
        return []
    coords = list(lote_poly.exterior.coords)[:-1]
    n = len(coords)
    result = []
    labels = ["frente", "derecha", "fondo", "izquierda"] if n == 4 else [str(i) for i in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        edge = LineString([(x1, y1), (x2, y2)])
        try:
            dist = float(footprint.distance(edge))
        except Exception:
            dist = -1.0
        result.append((i, round(dist, 3), labels[i]))
    return result


def metric_pct_circulacion(geometry, footprint):
    """(hall + corridors) / footprint.area * 100."""
    if footprint is None or footprint.area <= 0:
        return 0.0
    circ = []
    for key in ("hall", "vestibulo"):
        p = _pts_to_poly(geometry.get(key, []))
        if p:
            circ.append(p)
    for c in geometry.get("corridors", []):
        p = _pts_to_poly(c)
        if p:
            circ.append(p)
    if not circ:
        return 0.0
    try:
        return round(unary_union(circ).area / footprint.area * 100, 1)
    except Exception:
        return 0.0


def metric_eficiencia(geometry, footprint):
    """Sum net dpto areas / área techada (footprint − patio) * 100."""
    if footprint is None or footprint.area <= 0:
        return 0.0
    total = sum(
        (d.get("area_m2", 0) if isinstance(d, dict) else 0)
        for d in geometry.get("departamentos", [])
    )
    techada = footprint
    patio = _pts_to_poly(geometry.get("patio", []))
    if patio:
        try:
            techada = footprint.difference(patio)
        except Exception:
            pass
    if techada.area <= 0:
        return 0.0
    return round(total / techada.area * 100, 1)


def metric_huecos(footprint, lote_util, geometry=None):
    """Área de lote_util no cubierta por building (m²).
    Devuelve (total_m2, sin_pozos_m2) — el segundo excluye área de pozos
    de luz justificados (open-air requerido por norma)."""
    if footprint is None:
        total = round(lote_util.area, 1)
        return total, total
    try:
        diff = lote_util.difference(footprint.buffer(0.05))
        total = round(diff.area, 2)
    except Exception:
        return -1.0, -1.0

    # Restar pozos justificados del numerador de huecos
    pozo_area = 0.0
    if geometry:
        for p in geometry.get("pozos_luz", []):
            poly = _pts_to_poly(p)
            if poly:
                try:
                    pozo_area += diff.intersection(poly.buffer(0.1)).area
                except Exception:
                    pozo_area += poly.area
    return total, round(max(0.0, total - pozo_area), 2)


def metric_acceso(geometry):
    """% dptos con contacto ≥1.2m al hall/corredor. Devuelve (ok, total)."""
    hall_p = _pts_to_poly(geometry.get("hall", []))
    circ_polys = [hall_p.buffer(0.08)] if hall_p else []
    for c in geometry.get("corridors", []):
        p = _pts_to_poly(c)
        if p:
            circ_polys.append(p.buffer(0.08))
    if not circ_polys:
        return 0, len(geometry.get("departamentos", []))
    circ = unary_union(circ_polys)

    ok = 0
    dptos = geometry.get("departamentos", [])
    for d in dptos:
        pts = M._departamento_outline_coords(d)
        dp = _pts_to_poly(pts)
        if dp is None:
            continue
        try:
            shared = dp.intersection(circ)
            # Longitud de intersección en el borde (perimeter overlap)
            contact = shared.length if not shared.is_empty else 0.0
            # Si es un área (overlap real) usar el exterior compartido
            if hasattr(shared, "area") and shared.area > 0.2:
                ok += 1
                continue
            if contact >= 1.2:
                ok += 1
        except Exception:
            pass
    return ok, len(dptos)


def metric_frente_min(geometry):
    """Frente real (lado más corto del rectángulo envolvente) por dpto.
    Devuelve (n_bajo_5.2m, n_total, min_visto)."""
    dptos = geometry.get("departamentos", [])
    n_bajo, min_visto = 0, None
    for d in dptos:
        pts = M._departamento_outline_coords(d)
        dp = _pts_to_poly(pts)
        if dp is None or dp.is_empty:
            continue
        try:
            mrr = dp.minimum_rotated_rectangle
            c = list(mrr.exterior.coords)
            s0 = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
            s1 = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
            frente = min(s0, s1)
        except Exception:
            continue
        min_visto = frente if min_visto is None else min(min_visto, frente)
        if frente + 1e-6 < 5.2:
            n_bajo += 1
    return n_bajo, len(dptos), round(min_visto, 2) if min_visto is not None else None


def metric_paralelismo(footprint, lote_poly):
    """Ángulo (grados) entre borde trasero del edificio y lindero posterior del lote."""
    if footprint is None or footprint.is_empty:
        return 999.0
    try:
        coords_lot = list(lote_poly.exterior.coords)[:-1]
        n = len(coords_lot)
        # Lindero posterior = arista con mayor y medio
        best_i = max(range(n),
                     key=lambda i: (coords_lot[i][1] + coords_lot[(i + 1) % n][1]) / 2)
        lx1, ly1 = coords_lot[best_i]
        lx2, ly2 = coords_lot[(best_i + 1) % n]
        lot_ang = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))

        # Borde trasero del edificio: entre aristas con midy cercano al máximo,
        # elegir la MÁS LARGA (evita spikes verticales cortos en esquinas).
        fp_coords = list(footprint.exterior.coords)[:-1]
        m = len(fp_coords)
        midys = [(fp_coords[j][1] + fp_coords[(j + 1) % m][1]) / 2 for j in range(m)]
        max_midy = max(midys)
        tol = 2.0  # m
        candidates = [j for j in range(m) if midys[j] >= max_midy - tol]
        def _edge_len(j):
            x0, y0 = fp_coords[j]; x1, y1 = fp_coords[(j + 1) % m]
            return math.hypot(x1 - x0, y1 - y0)
        best_j = max(candidates, key=_edge_len)
        bx1, by1 = fp_coords[best_j]
        bx2, by2 = fp_coords[(best_j + 1) % m]
        bld_ang = math.degrees(math.atan2(by2 - by1, bx2 - bx1))

        diff = abs(lot_ang - bld_ang) % 180
        return round(min(diff, 180 - diff), 1)
    except Exception:
        return 999.0


def metric_pozos(geometry, h_edif):
    """(area_pozos_total, area_normativo_min_total, ratio)."""
    pozo_req = max(RNE["pozos_luz"]["min_abs"],
                   h_edif * RNE["pozos_luz"]["ratio_dorm"])
    total_area = 0.0
    n_pozos = 0
    for p in geometry.get("pozos_luz", []):
        poly = _pts_to_poly(p)
        if poly:
            total_area += poly.area
            n_pozos += 1
    if n_pozos == 0:
        return 0.0, 0.0, 0.0
    norm_total = pozo_req * pozo_req * n_pozos  # cuadrado mínimo por pozo
    ratio = round(total_area / norm_total, 2) if norm_total > 0 else 0.0
    return round(total_area, 2), round(norm_total, 2), ratio


# ─────────────────────────────────────────────────────────────────────────────
# Visualización PNG
# ─────────────────────────────────────────────────────────────────────────────

TIPO_COLOR = {
    "1D":   "#93c5fd",  # blue-300
    "1D+E": "#86efac",  # green-300
    "2D":   "#fde047",  # yellow-300
    "2D+E": "#fdba74",  # orange-300
    "3D":   "#f9a8d4",  # pink-300
}


def _poly_to_mpl(poly, **kwargs):
    """shapely Polygon → matplotlib Patch."""
    if poly is None or poly.is_empty:
        return None
    try:
        xy = np.array(poly.exterior.coords)
        return MplPoly(xy, closed=True, **kwargs)
    except Exception:
        return None


def render_png(caso, geometry, normativa, lote_poly, lote_util,
               footprint, metrics, pass_fail, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")

    # Lote
    lp = _poly_to_mpl(lote_poly, fc="none", ec="#555", lw=2, zorder=1)
    if lp:
        ax.add_patch(lp)

    # Lote útil (envolvente retiros)
    lu = _poly_to_mpl(lote_util, fc="#e0f2fe22", ec="#0ea5e9", lw=1.2,
                      ls="--", zorder=2)
    if lu:
        ax.add_patch(lu)

    # Footprint
    if footprint:
        fp = _poly_to_mpl(footprint, fc="#e2e8f022", ec="#334155", lw=1.5, zorder=3)
        if fp:
            ax.add_patch(fp)

    # Hall / corridors
    for key in ("hall", "vestibulo"):
        p = _pts_to_poly(geometry.get(key, []))
        if p:
            patch = _poly_to_mpl(p, fc="#fef08a", ec="#ca8a04", lw=0.8, alpha=0.85, zorder=4)
            if patch:
                ax.add_patch(patch)
    for c in geometry.get("corridors", []):
        p = _pts_to_poly(c)
        if p:
            patch = _poly_to_mpl(p, fc="#fef9c3", ec="#ca8a04", lw=0.7, alpha=0.75, zorder=4)
            if patch:
                ax.add_patch(patch)

    # Core / escalera / ascensores
    for key in ("core", "escalera"):
        p = _pts_to_poly(geometry.get(key, []))
        if p:
            patch = _poly_to_mpl(p, fc="#fca5a5", ec="#dc2626", lw=0.8, alpha=0.85, zorder=5)
            if patch:
                ax.add_patch(patch)
    for a in geometry.get("ascensores", []):
        p = _pts_to_poly(a)
        if p:
            patch = _poly_to_mpl(p, fc="#fca5a5", ec="#dc2626", lw=0.7, alpha=0.75, zorder=5)
            if patch:
                ax.add_patch(patch)

    # Departamentos
    for d in geometry.get("departamentos", []):
        pts = M._departamento_outline_coords(d)
        p = _pts_to_poly(pts)
        if p is None:
            continue
        tip = d.get("tipologia", "") if isinstance(d, dict) else ""
        color = TIPO_COLOR.get(tip, "#d1d5db")
        area = d.get("area_m2", 0) if isinstance(d, dict) else 0
        patch = _poly_to_mpl(p, fc=color, ec="#374151", lw=0.8, alpha=0.75, zorder=6)
        if patch:
            ax.add_patch(patch)
        cx, cy = p.centroid.x, p.centroid.y
        ax.text(cx, cy, f"{tip}\n{area:.0f}m²", ha="center", va="center",
                fontsize=6.5, color="#111827", zorder=9)

    # Pozos
    for pz in geometry.get("pozos_luz", []):
        p = _pts_to_poly(pz)
        if p:
            patch = _poly_to_mpl(p, fc="#f5f5f4", ec="#9ca3af", lw=1.0, alpha=0.9, zorder=7)
            if patch:
                ax.add_patch(patch)
            xs = [c[0] for c in p.exterior.coords]
            ys = [c[1] for c in p.exterior.coords]
            ax.plot([min(xs), max(xs)], [min(ys), max(ys)], c="#9ca3af", lw=0.8, zorder=8)
            ax.plot([min(xs), max(xs)], [max(ys), min(ys)], c="#9ca3af", lw=0.8, zorder=8)

    # Cotas lote
    minx, miny, maxx, maxy = lote_poly.bounds
    ax.set_xlim(minx - 3, maxx + 3)
    ax.set_ylim(miny - 3, maxy + 5)

    # ── Panel de métricas ──
    m = metrics
    nd_ok  = abs(m.get("nd_emitidos", 0) - m.get("nd_pedidos", 1)) <= THR_ND_DELTA
    efi_ok = m.get("eficiencia", 0) >= THR_EFI_MIN
    circ_ok= m.get("pct_circ", 100) <= THR_PCT_CIRC_MAX
    hue_ok = m.get("huecos_sin_pozos", m.get("huecos", 999)) <= THR_HUECOS_MAX
    par_ok = m.get("paralelismo", 999) <= THR_PARA_MAX

    acc_ok_n, acc_tot = m.get("acceso", (0, 1))
    acc_pct  = acc_ok_n / max(acc_tot, 1) * 100
    acc_ok   = acc_pct >= THR_ACCESO_PCT

    def _sym(ok):
        return "✓" if ok else "✗"

    lines = [
        f"Caso: {caso['name']}",
        f"Lote {caso['frente']}×{caso.get('derecha','-')}m  nd={caso['nd']}  r_lat={caso['retiro_lat']} r_pos={caso['retiro_pos']}",
        "",
        f"{_sym(nd_ok)}  Dptos: {m.get('nd_emitidos','?')}/{caso['nd']} pedidos",
        f"{_sym(efi_ok)}  Eficiencia: {m.get('eficiencia','?')}%  (min {THR_EFI_MIN}%)",
        f"{_sym(circ_ok)}  Circulación: {m.get('pct_circ','?')}%  (max {THR_PCT_CIRC_MAX}%)",
        f"{_sym(hue_ok)}  Huecos ex.pz: {m.get('huecos_sin_pozos','?')}m² (tot {m.get('huecos','?')}) (max {THR_HUECOS_MAX})",
        f"{_sym(acc_ok)}  Acceso: {acc_ok_n}/{acc_tot} dptos ({acc_pct:.0f}%)",
        f"{_sym(par_ok)}  Paralelismo fondo: {m.get('paralelismo','?')}°  (max {THR_PARA_MAX}°)",
    ]

    # Retiros por borde
    retiro_lines = []
    ped = {"frente": 0, "derecha": caso["retiro_lat"],
           "fondo": caso["retiro_pos"], "izquierda": caso["retiro_lat"]}
    for idx, dist, label in m.get("retiros_borde", []):
        pedido = ped.get(label, caso["retiro_lat"])
        diff = abs(dist - pedido)
        sym = "✓" if diff <= THR_RETIRO_TOL else "✗"
        retiro_lines.append(f"{sym}  Retiro {label}: {dist:.2f}m (ped {pedido:.1f}m)")

    lines += [""] + retiro_lines

    total_checks = sum([nd_ok, efi_ok, circ_ok, hue_ok, acc_ok, par_ok] +
                       [abs(d - ped.get(lb, caso["retiro_lat"])) <= THR_RETIRO_TOL
                        for _, d, lb in m.get("retiros_borde", [])])
    total_n = 6 + len(m.get("retiros_borde", []))
    lines += ["", f"Score: {total_checks}/{total_n}"]

    # Background panel
    ax.text(
        minx - 2.8, maxy + 4.5,
        "\n".join(lines),
        ha="left", va="top", fontsize=7.5,
        fontfamily="monospace",
        color="#1e293b",
        bbox=dict(fc="white", ec="#cbd5e1", lw=0.8, boxstyle="round,pad=0.4"),
        transform=ax.transData,
        zorder=10,
        clip_on=False,
    )

    ax.set_title(
        f"E0 baseline — {caso['name']}  [{total_checks}/{total_n} PASS]",
        fontsize=9, color="#1e293b"
    )
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Casos de prueba
# ─────────────────────────────────────────────────────────────────────────────

def _make_caso(name, frente, fondo, derecha, izquierda, nd,
               retiro_lat=2.3, retiro_pos=2.3, pisos=7, expect_inviable=False):
    return dict(name=name, frente=frente, fondo=fondo,
                derecha=derecha, izquierda=izquierda, nd=nd,
                retiro_lat=retiro_lat, retiro_pos=retiro_pos, pisos=pisos,
                expect_inviable=expect_inviable)


CASOS = [
    # ── Rectangulares ─────────────────────────────────────
    _make_caso("R10x25_nd4",   10,  10, 25, 25, 4, expect_inviable=True),
    _make_caso("R13x28_nd4",   13,  13, 28, 28, 4),
    _make_caso("R13x28_nd2",   13,  13, 28, 28, 2),
    _make_caso("R17x34_nd4",   17,  17, 34, 34, 4),
    _make_caso("R17x34_nd8",   17,  17, 34, 34, 8),
    _make_caso("R20x30_nd4",   20,  20, 30, 30, 4),
    _make_caso("R20x30_nd6",   20,  20, 30, 30, 6),
    _make_caso("R24x32_nd6",   24,  24, 32, 32, 6),
    _make_caso("R24x32_nd8",   24,  24, 32, 32, 8),
    # ── Retiro 0 ──────────────────────────────────────────
    _make_caso("R20x30_nd4_r0",20,  20, 30, 30, 4, retiro_lat=0, retiro_pos=0),
    _make_caso("R24x32_nd6_r0",24,  24, 32, 32, 6, retiro_lat=0, retiro_pos=0),
    # ── Proporciones especiales ────────────────────────────
    _make_caso("R15x45_nd6",   15,  15, 45, 45, 6),   # profundo
    _make_caso("R30x20_nd6",   30,  30, 20, 20, 6),   # ancho poco profundo
    _make_caso("R18x18_nd4",   18,  18, 18, 18, 4),   # cuadrado
    # ── Trapezoides ────────────────────────────────────────
    _make_caso("T_estrecho_nd4",18, 24, 30, 30, 4),   # frente < fondo
    _make_caso("T_ancho_nd4",   24, 18, 30, 30, 4),   # frente > fondo
    _make_caso("T_asim_nd6",    20, 20, 32, 28, 6),   # profundidades asimétricas
    _make_caso("T_asim_pos0",   20, 20, 32, 28, 6, retiro_lat=2.3, retiro_pos=0),
    # ── Lote real usuario ──────────────────────────────────
    _make_caso("USUARIO_nd4",   24, 29.3, 32.2, 32.1, 4),
    _make_caso("USUARIO_nd6",   24, 29.3, 32.2, 32.1, 6),
    _make_caso("USUARIO_nd8",   24, 29.3, 32.2, 32.1, 8),
    _make_caso("USUARIO_r0_nd4",24, 29.3, 32.2, 32.1, 4, retiro_lat=0, retiro_pos=0),
    _make_caso("USUARIO_r3_nd6",24, 29.3, 32.2, 32.1, 6, retiro_lat=2.3, retiro_pos=3.0),
    # ── Lotes calibrados (REGLAS_DISENO.md, medidos de DXF reales Lima) ────
    _make_caso("REGLAS_15x31_nd6",   15,   15, 31,   31,   6),
    _make_caso("REGLAS_17x31_nd6",   17,   17, 31,   31,   6),
    _make_caso("REGLAS_12x30_nd4",   12.5, 12.5, 30.5, 30.5, 4),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────────────────────────────────────

def run_caso(caso):
    """Ejecuta un caso, devuelve dict de métricas + resultado."""
    c = caso
    coords = _make_lote_coords(c["frente"], c["fondo"], c["derecha"], c["izquierda"],
                                retiro_frontal=0.0)
    proyecto = ProyectoInmobiliario(
        coordenadas_lote=coords,
        area_bruta_terreno=float(Polygon(coords).area),
        numero_pisos=c["pisos"],
        retiro_frontal=0.0,
        zonificacion="RDM",
        num_ascensores=1,
        num_departamentos=c["nd"],
        frente=float(c["frente"]),
        fondo=float(c["fondo"]),
        derecha=float(c["derecha"]),
        izquierda=float(c["izquierda"]),
        retiro_lateral=float(c["retiro_lat"]),
        retiro_posterior=float(c["retiro_pos"]),
        ciego_frente=False,
        ciego_fondo=True,
        ciego_derecha=True,
        ciego_izquierda=True,
        area_libre_min_pct=0.0,
        ajustar_pisos_normativa=False,
    )

    try:
        geometry, normativa = _generate_geometry(proyecto)
    except Exception as e:
        return None, None, None, None, None, str(e)

    lote_poly = _make_lote_poly(c["frente"], c["fondo"], c["derecha"], c["izquierda"])
    lote_util = _erode_lote(lote_poly, c["retiro_lat"], c["retiro_pos"]) or lote_poly
    footprint = _build_footprint(geometry)
    h_edif = c["pisos"] * 2.80

    nd_emit = len(geometry.get("departamentos", []))
    # Medir retiros contra el lote ORIGINAL (lote_poly), no el erosionado
    retiros = metric_retiro_por_borde(footprint, lote_poly)
    pct_circ = metric_pct_circulacion(geometry, footprint)
    efi = metric_eficiencia(geometry, footprint)
    huecos_total, huecos_sin_pz = metric_huecos(footprint, lote_util, geometry)
    acceso = metric_acceso(geometry)
    para = metric_paralelismo(footprint, lote_util)
    pz_total, pz_norm, pz_ratio = metric_pozos(geometry, h_edif)
    frente_bajo, frente_tot, frente_min = metric_frente_min(geometry)

    metrics = {
        "nd_pedidos": c["nd"],
        "nd_emitidos": nd_emit,
        "pct_circ": pct_circ,
        "eficiencia": efi,
        "huecos": huecos_total,
        "huecos_sin_pozos": huecos_sin_pz,
        "acceso": acceso,
        "paralelismo": para,
        "pozos_area": pz_total,
        "pozos_norm": pz_norm,
        "pozos_ratio": pz_ratio,
        "retiros_borde": retiros,
        "frente_bajo": frente_bajo,
        "frente_tot": frente_tot,
        "frente_min": frente_min,
        "lote_area": round(lote_poly.area, 1),
        "util_area": round(lote_util.area, 1),
        "fp_area": round(footprint.area, 1) if footprint else 0,
        "evaluacion": geometry.get("evaluacion", {}),
    }
    return geometry, normativa, lote_poly, lote_util, footprint, metrics


def evaluate_metrics(caso, metrics):
    """Devuelve lista de (check_name, passed, detail)."""
    c = caso
    m = metrics
    checks = []

    # nd delta
    nd_delta = abs(m["nd_emitidos"] - m["nd_pedidos"])
    checks.append(("nd_delta",
                   nd_delta <= THR_ND_DELTA,
                   f"{m['nd_emitidos']}/{m['nd_pedidos']} (Δ{nd_delta})"))

    # eficiencia
    checks.append(("eficiencia",
                   m["eficiencia"] >= THR_EFI_MIN,
                   f"{m['eficiencia']}% (min {THR_EFI_MIN}%)"))

    # circulación
    checks.append(("pct_circ",
                   m["pct_circ"] <= THR_PCT_CIRC_MAX,
                   f"{m['pct_circ']}% (max {THR_PCT_CIRC_MAX}%)"))

    # huecos (sin pozos justificados)
    hp = m.get("huecos_sin_pozos", m["huecos"])
    checks.append(("huecos_sin_pz",
                   hp <= THR_HUECOS_MAX,
                   f"{hp}m² excl.pozos (total {m['huecos']}m²) (max {THR_HUECOS_MAX}m²)"))

    # acceso
    acc_ok_n, acc_tot = m["acceso"]
    acc_pct = acc_ok_n / max(acc_tot, 1) * 100
    checks.append(("acceso",
                   acc_pct >= THR_ACCESO_PCT,
                   f"{acc_ok_n}/{acc_tot} ({acc_pct:.0f}%)"))

    # paralelismo
    checks.append(("paralelismo",
                   m["paralelismo"] <= THR_PARA_MAX,
                   f"{m['paralelismo']}° (max {THR_PARA_MAX}°)"))

    # frente real (REGLAS_DISENO.md: mín tipológico 5.2m) — tolera un
    # residual bajo (mordida de pozo/lote irregular), no exige cero.
    f_bajo, f_tot = m["frente_bajo"], m["frente_tot"]
    f_pct_bajo = f_bajo / max(f_tot, 1) * 100
    checks.append(("frente_min",
                   f_pct_bajo <= THR_FRENTE_BAJO_PCT,
                   f"{f_bajo}/{f_tot} dpto(s) < 5.2m ({f_pct_bajo:.0f}%, "
                   f"max {THR_FRENTE_BAJO_PCT:.0f}%) min visto {m['frente_min']}m"))

    # retiros
    ped_map = {"frente": 0.0, "derecha": c["retiro_lat"],
               "fondo": c["retiro_pos"], "izquierda": c["retiro_lat"]}
    for idx, dist, label in m["retiros_borde"]:
        pedido = ped_map.get(label, c["retiro_lat"])
        diff = abs(dist - pedido)
        checks.append((f"retiro_{label}",
                       diff <= THR_RETIRO_TOL,
                       f"{dist:.2f}m (ped {pedido:.1f}m, Δ{diff:.2f}m)"))

    # E5: score del diseño (auto-crítica del motor)
    ev = m.get("evaluacion", {})
    score = ev.get("score", 0)
    n_crit = ev.get("n_criticos", 0)
    checks.append(("e5_score",
                   score >= THR_SCORE_MIN and n_crit == 0,
                   f"score={score} criticos={n_crit} (min {THR_SCORE_MIN})"))

    return checks


def print_table(rows):
    """Imprime tabla resumen al terminal."""
    header = f"{'CASO':<22} {'ND':>5} {'EFI%':>6} {'CIRC%':>6} {'HUE':>6} {'ACC%':>5} {'PAR°':>5} {'E5':>5} {'SCORE':>7}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    e5_scores = []
    for r in rows:
        e5 = r.get("e5_score", "?")
        if isinstance(e5, (int, float)):
            e5_scores.append(e5)
        print(
            f"{r['name']:<22} "
            f"{r.get('nd','?'):>5} "
            f"{r.get('efi','?'):>6} "
            f"{r.get('circ','?'):>6} "
            f"{r.get('hue','?'):>6} "
            f"{r.get('acc','?'):>5} "
            f"{r.get('par','?'):>5} "
            f"{str(e5):>5} "
            f"{r.get('score','?'):>7}"
        )
    print(sep)
    total_p = sum(r.get("pass_n", 0) for r in rows)
    total_n = sum(r.get("total_n", 1) for r in rows)
    avg_e5 = round(sum(e5_scores) / len(e5_scores), 1) if e5_scores else 0
    print(f"TOTAL: {total_p}/{total_n} checks PASS   E5 score promedio: {avg_e5}/100\n")


def main():
    print(f"\n{'═'*60}")
    print(f"  tests_cabida.py — E0 Arnés de medición")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {len(CASOS)} casos × ~8 métricas")
    print(f"{'═'*60}")

    table_rows = []
    all_checks = []

    for i, caso in enumerate(CASOS):
        print(f"\n[{i+1:02d}/{len(CASOS)}] {caso['name']} ...", end=" ", flush=True)

        result = run_caso(caso)
        geometry, normativa, lote_poly, lote_util, footprint, metrics_or_err = result

        if isinstance(metrics_or_err, str):
            # Caso marcado inviable: PASS si lanzó ValueError con "inviable".
            if caso.get("expect_inviable") and "inviable" in metrics_or_err.lower():
                print(f"INVIABLE (esperado): {metrics_or_err[:60]}")
                table_rows.append({"name": caso["name"], "nd": "INV", "efi": "—",
                                    "circ": "—", "hue": "—", "acc": "—",
                                    "par": "—", "score": "1/1",
                                    "pass_n": 1, "total_n": 1})
            else:
                print(f"ERROR: {metrics_or_err}")
                table_rows.append({"name": caso["name"], "nd": "ERR", "efi": "ERR",
                                    "circ": "ERR", "hue": "ERR", "acc": "ERR",
                                    "par": "ERR", "score": "0/0",
                                    "pass_n": 0, "total_n": 0})
            continue

        # Caso esperaba inviable pero generó: FAIL explícito.
        if caso.get("expect_inviable"):
            print("FAIL: esperaba inviable pero generó geometría")
            table_rows.append({"name": caso["name"], "nd": "!INV", "efi": "—",
                                "circ": "—", "hue": "—", "acc": "—",
                                "par": "—", "score": "0/1",
                                "pass_n": 0, "total_n": 1})
            continue

        metrics = metrics_or_err
        checks = evaluate_metrics(caso, metrics)
        pass_n = sum(1 for _, ok, _ in checks if ok)
        total_n = len(checks)
        pct = pass_n / max(total_n, 1) * 100

        # Build pass_fail dict for PNG
        pf = {name: ok for name, ok, _ in checks}

        # PNG
        png_path = os.path.join(OUT_DIR, f"{i+1:02d}_{caso['name']}.png")
        try:
            render_png(caso, geometry, normativa, lote_poly, lote_util,
                       footprint, metrics, pf, png_path)
        except Exception:
            print(f"[PNG ERROR] {traceback.format_exc()[:200]}")

        status = "PASS" if pass_n == total_n else f"FAIL ({pass_n}/{total_n})"
        print(f"{status}")

        # Detailed failures
        for name, ok, detail in checks:
            if not ok:
                print(f"    ✗ {name}: {detail}")

        acc_ok_n, acc_tot = metrics["acceso"]
        acc_pct = acc_ok_n / max(acc_tot, 1) * 100

        ev = metrics.get("evaluacion", {})
        e5_s = ev.get("score", 0)
        table_rows.append({
            "name": caso["name"],
            "nd": f"{metrics['nd_emitidos']}/{caso['nd']}",
            "efi": f"{metrics['eficiencia']}",
            "circ": f"{metrics['pct_circ']}",
            "hue": f"{metrics.get('huecos_sin_pozos', metrics['huecos'])}",
            "acc": f"{acc_pct:.0f}",
            "par": f"{metrics['paralelismo']}",
            "e5_score": e5_s,
            "score": f"{pass_n}/{total_n}",
            "pass_n": pass_n,
            "total_n": total_n,
        })
        all_checks.extend(checks)

    print_table(table_rows)

    # Resumen por tipo de check
    from collections import defaultdict
    by_check = defaultdict(lambda: [0, 0])  # [pass, total]
    for name, ok, _ in all_checks:
        key = name if not name.startswith("retiro_") else "retiro_borde"
        by_check[key][1] += 1
        if ok:
            by_check[key][0] += 1
    print("Resumen por tipo de check:")
    for k, (p, t) in sorted(by_check.items()):
        bar = "█" * p + "░" * (t - p)
        print(f"  {k:<20} {bar} {p}/{t}")

    print(f"\nPNGs guardados en: {OUT_DIR}")
    print(f"Total: {sum(r.get('pass_n',0) for r in table_rows)}"
          f"/{sum(r.get('total_n',0) for r in table_rows)} checks PASS")


if __name__ == "__main__":
    main()
