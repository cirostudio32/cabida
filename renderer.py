"""
renderer.py — Motor de renderizado arquitectónico en Python (matplotlib).
Genera imágenes PNG de planta típica, primer piso y sótanos.
"""

import math
import io
import base64
from typing import List, Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # No-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.patheffects as pe
import numpy as np


# ═══════════════════════════════════════════════════════════════
# PALETA DE COLORES — Diseño arquitectónico profesional
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "background":       "#f8fafc",
    "grid":             "#e2e8f0",
    "lindero_fill":     (1, 1, 1, 0.4),
    "lindero_stroke":   "#94a3b8",
    "retiro_fill":      (0.94, 0.27, 0.27, 0.06),
    "retiro_stroke":    "#ef4444",
    "lote_neto_stroke": "#0696D7",
    "techada_fill":     (1, 1, 1, 0.6),
    "techada_stroke":   "#94a3b8",
    # Departamentos
    "dpto_1D":          "#dbeafe",  # azul claro
    "dpto_2D":          "#d1fae5",  # verde claro
    "dpto_3D":          "#fef3c7",  # amarillo claro
    "dpto_stroke":      "#334155",
    # Core / Circulación
    "core_fill":        "#e2e8f0",
    "core_stroke":      "#334155",
    "hall_fill":        "#e2e8f0",
    "hall_stroke":      "#94a3b8",
    "escalera_fill":    "#f8fafc",
    "escalera_pres":    "#fef3c7",
    "escalera_stroke":  "#334155",
    "ascensor_fill":    "#f1f5f9",
    "ascensor_stroke":  "#334155",
    "vestibulo_fill":   "#fef9c3",
    "vestibulo_stroke": "#d97706",
    # Patio & Ductos
    "patio_fill":       "#f0f9ff",
    "patio_stroke":     "#0369a1",
    "ducto_fill":       "#fff7ed",
    "ducto_stroke":     "#ea580c",
    # Primer piso
    "comercio_fill":    "#ecfdf5",
    "comercio_stroke":  "#10b981",
    "servicios_fill":   "#e2e8f0",
    "servicios_stroke": "#475569",
    "lobby_fill":       "#fef3c7",
    "lobby_stroke":     "#d97706",
    "rampa_fill":       "#fee2e2",
    "rampa_stroke":     "#dc2626",
    # Sótano
    "slab_fill":        "#f1f5f9",
    "slab_stroke":      "#64748b",
    "stall_fill":       "#ffffff",
    "stall_stroke":     "#475569",
    "aisle_fill":       "#e2e8f0",
    "aisle_stroke":     "#94a3b8",
    # Cisternas
    "cist_a_fill":      "#bfdbfe",
    "cist_a_stroke":    "#2563eb",
    "cist_b_fill":      "#93c5fd",
    "cist_b_stroke":    "#1d4ed8",
    "cist_aci_fill":    "#fca5a5",
    "cist_aci_stroke":  "#dc2626",
    "cist_maq_fill":    "#fef3c7",
    "cist_maq_stroke":  "#d97706",
    # Cotas
    "cota_line":        "#475569",
    "cota_text":        "#1e293b",
    "label_primary":    "#1e293b",
    "label_secondary":  "#475569",
    "label_detail":     "#64748b",
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0):
    """Convert hex color to RGBA tuple."""
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def poly_centroid(poly: List[Dict[str, float]]) -> Tuple[float, float]:
    """Calculate centroid of a polygon."""
    if not poly:
        return (0, 0)
    cx = sum(p["x"] for p in poly) / len(poly)
    cy = sum(p["y"] for p in poly) / len(poly)
    return (cx, cy)


def poly_to_mpl(poly: List[Dict[str, float]]) -> np.ndarray:
    """Convert [{x,y},...] to numpy array for matplotlib."""
    if not poly:
        return np.array([])
    return np.array([[p["x"], p["y"]] for p in poly])


def calc_poly_area(poly: List[Dict[str, float]]) -> float:
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


def poly_width_height(poly: List[Dict[str, float]]) -> Tuple[float, float]:
    """Get width and height from a 4-point polygon."""
    if len(poly) < 4:
        return (0, 0)
    w = math.hypot(poly[1]["x"] - poly[0]["x"], poly[1]["y"] - poly[0]["y"])
    h = math.hypot(poly[2]["x"] - poly[1]["x"], poly[2]["y"] - poly[1]["y"])
    return (w, h)


def get_typology(area: float) -> str:
    """Get apartment typology name from area."""
    if area <= 55:
        return "1D"
    if area <= 80:
        return "2D"
    return "3D"


def get_typology_color(typology: str) -> str:
    """Get color for a typology."""
    if typology == "1D":
        return COLORS["dpto_1D"]
    if typology == "2D":
        return COLORS["dpto_2D"]
    return COLORS["dpto_3D"]


# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE DIBUJO
# ═══════════════════════════════════════════════════════════════

def _draw_poly(ax, poly, fill_color, stroke_color, label=None,
               linestyle='-', linewidth=1.5, decor='none', zorder=2,
               alpha_fill=0.85, fontsize=8):
    """Draw a polygon with fill, stroke, optional label, and decoration."""
    if not poly or len(poly) < 3:
        return

    coords = poly_to_mpl(poly)

    # Fill
    if fill_color:
        if isinstance(fill_color, str):
            fc = hex_to_rgba(fill_color, alpha_fill)
        else:
            fc = fill_color
        patch = MplPolygon(coords, closed=True, facecolor=fc,
                           edgecolor='none', zorder=zorder)
        ax.add_patch(patch)

    # Stroke
    if stroke_color:
        if isinstance(stroke_color, str):
            sc = hex_to_rgba(stroke_color)
        else:
            sc = stroke_color
        patch_s = MplPolygon(coords, closed=True, facecolor='none',
                             edgecolor=sc, linewidth=linewidth,
                             linestyle=linestyle, zorder=zorder + 0.5)
        ax.add_patch(patch_s)

    # Decorations
    if decor == 'xBox' or decor == 'asc':
        # Diagonal cross
        ax.plot([coords[0, 0], coords[2, 0]], [coords[0, 1], coords[2, 1]],
                color='#64748b', linewidth=0.8, zorder=zorder + 0.3)
        ax.plot([coords[1, 0], coords[3, 0]], [coords[1, 1], coords[3, 1]],
                color='#64748b', linewidth=0.8, zorder=zorder + 0.3)
    elif decor == 'stairs' and len(coords) >= 4:
        steps = 6
        for k in range(1, steps):
            u = k / steps
            pt1x = coords[0, 0] + (coords[3, 0] - coords[0, 0]) * u
            pt1y = coords[0, 1] + (coords[3, 1] - coords[0, 1]) * u
            pt2x = coords[1, 0] + (coords[2, 0] - coords[1, 0]) * u
            pt2y = coords[1, 1] + (coords[2, 1] - coords[1, 1]) * u
            ax.plot([pt1x, pt2x], [pt1y, pt2y],
                    color='#94a3b8', linewidth=0.7, zorder=zorder + 0.3)

    # Label
    if label:
        cx, cy = poly_centroid(poly)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            offset_y = (i - (len(lines) - 1) / 2) * (fontsize * 0.18)
            ax.text(cx, cy - offset_y, line,  # Inverted Y
                    fontsize=fontsize, fontweight='bold',
                    ha='center', va='center',
                    color=COLORS["label_primary"],
                    zorder=zorder + 1,
                    path_effects=[pe.withStroke(linewidth=2.5,
                                               foreground='white')])


def _draw_cota(ax, p_a, p_b, text, offset_dist=0.6, fontsize=7):
    """Draw an AutoCAD-style dimension line between two points."""
    dx = p_b["x"] - p_a["x"]
    dy = p_b["y"] - p_a["y"]
    length = math.hypot(dx, dy)
    if length < 0.1:
        return

    nx, ny = dx / length, dy / length
    # Perpendicular
    ox, oy = -ny * offset_dist, nx * offset_dist

    c1x = p_a["x"] + ox
    c1y = p_a["y"] + oy
    c2x = p_b["x"] + ox
    c2y = p_b["y"] + oy

    # Main dimension line
    ax.plot([c1x, c2x], [c1y, c2y], color=COLORS["cota_line"],
            linewidth=0.8, zorder=8)

    # Tick marks
    tick = 0.15
    ax.plot([c1x - tick * ny, c1x + tick * ny],
            [c1y + tick * nx, c1y - tick * nx],
            color=COLORS["cota_line"], linewidth=0.8, zorder=8)
    ax.plot([c2x - tick * ny, c2x + tick * ny],
            [c2y + tick * nx, c2y - tick * nx],
            color=COLORS["cota_line"], linewidth=0.8, zorder=8)

    # Extension lines
    ax.plot([p_a["x"], c1x], [p_a["y"], c1y],
            color=COLORS["cota_line"], linewidth=0.5, linestyle='--', zorder=7)
    ax.plot([p_b["x"], c2x], [p_b["y"], c2y],
            color=COLORS["cota_line"], linewidth=0.5, linestyle='--', zorder=7)

    # Text
    mx = (c1x + c2x) / 2
    my = (c1y + c2y) / 2
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90 or angle < -90:
        angle += 180

    ax.text(mx, my, text, fontsize=fontsize,
            ha='center', va='center', rotation=angle,
            color=COLORS["cota_text"], fontweight='medium', zorder=9,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.85))


def _draw_poly_dimensions(ax, poly, offset=0.8):
    """Draw width and height dimensions for a 4-point polygon."""
    if not poly or len(poly) < 4:
        return
    w, h = poly_width_height(poly)
    if w > 0.5:
        _draw_cota(ax, poly[0], poly[1], f"{w:.2f}m", offset)
    if h > 0.5:
        _draw_cota(ax, poly[1], poly[2], f"{h:.2f}m", offset)


def _draw_complex_label(ax, poly, typology, unit_id, area_text, fontsize_typo=14):
    """Draw apartment label: typology + ID + area."""
    cx, cy = poly_centroid(poly)

    ax.text(cx, cy + 0.4, typology, fontsize=fontsize_typo, fontweight='black',
            ha='center', va='center', color=COLORS["label_primary"], zorder=10,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax.text(cx, cy - 0.3, f"DPTO {unit_id}", fontsize=7, fontweight='bold',
            ha='center', va='center', color=COLORS["label_secondary"], zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.text(cx, cy - 0.9, area_text, fontsize=6.5, fontweight='medium',
            ha='center', va='center', color=COLORS["label_detail"], zorder=10,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def _draw_lindero_cotas(ax, polygon_pts, params):
    """Draw terrain boundary dimension labels."""
    if len(polygon_pts) < 4:
        return

    labels = [
        (0, 1, f'{params.get("frente", 0):.1f}m (Fte)', -1.5),
        (1, 2, f'{params.get("derecha", 0):.1f}m (Der)', -2.0),
        (2, 3, f'{params.get("fondo", 0):.1f}m (Fdo)', -1.5),
        (3, 0, f'{params.get("izquierda", 0):.1f}m (Izq)', -2.0),
    ]

    for i, j, text, offset in labels:
        _draw_cota(ax, polygon_pts[i], polygon_pts[j], text, offset)


# ═══════════════════════════════════════════════════════════════
# RENDERIZADORES PRINCIPALES
# ═══════════════════════════════════════════════════════════════

def render_planta_tipica(data: Dict[str, Any], params: Dict[str, Any],
                         width_px: int = 2400, height_px: int = 1800) -> str:
    """
    Render the typical floor plan as a base64 PNG.

    Args:
        data: geometry data from /auditoria-rne response
        params: input parameters (frente, derecha, fondo, izquierda, etc.)
        width_px, height_px: output image dimensions in pixels

    Returns:
        Base64-encoded PNG string
    """
    dpi = 200
    fig, ax = plt.subplots(1, 1,
                           figsize=(width_px / dpi, height_px / dpi),
                           dpi=dpi)
    ax.set_facecolor(COLORS["background"])
    ax.set_aspect('equal')

    geo = data.get("geometria_generada", {})
    normativa = data.get("normativa_estricta", {})

    # Compute polygon from params
    frente = params.get("frente", 10)
    fondo_val = params.get("fondo", 10)
    derecha = params.get("derecha", 20)
    izquierda = params.get("izquierda", 20)
    retiro_frontal = params.get("retiro_frontal", 0)
    pisos = params.get("pisos", 15)
    altura_piso = params.get("altura_piso", 2.80)

    # Terrain polygon
    p1 = {"x": -frente / 2, "y": 0}
    p2 = {"x": frente / 2, "y": 0}
    p3 = {"x": fondo_val / 2, "y": derecha}
    p4 = {"x": -fondo_val / 2, "y": izquierda}
    polygon_pts = [p1, p2, p3, p4]

    # Retiro
    rt_y = min(retiro_frontal, derecha, izquierda)
    if rt_y > 0:
        u_l = rt_y / izquierda if izquierda > 0 else 0
        u_r = rt_y / derecha if derecha > 0 else 0
        pr3 = {"x": p2["x"] + (p3["x"] - p2["x"]) * u_r,
               "y": p2["y"] + (p3["y"] - p2["y"]) * u_r}
        pr4 = {"x": p1["x"] + (p4["x"] - p1["x"]) * u_l,
               "y": p1["y"] + (p4["y"] - p1["y"]) * u_l}
        retiro_poly = [p1, p2, pr3, pr4]
    else:
        retiro_poly = []

    # ── DRAW GRID ──
    all_xs = [p["x"] for p in polygon_pts]
    all_ys = [p["y"] for p in polygon_pts]
    margin = 3.0
    x_min, x_max = min(all_xs) - margin, max(all_xs) + margin
    y_min, y_max = min(all_ys) - margin, max(all_ys) + margin

    # Grid lines (1m spacing)
    grid_step = 1.0
    for gx in np.arange(math.floor(x_min), math.ceil(x_max) + 1, grid_step):
        ax.axvline(x=gx, color=COLORS["grid"], linewidth=0.3, zorder=0)
    for gy in np.arange(math.floor(y_min), math.ceil(y_max) + 1, grid_step):
        ax.axhline(y=gy, color=COLORS["grid"], linewidth=0.3, zorder=0)

    # ── LINDEROS (terrain boundary) ──
    _draw_poly(ax, polygon_pts, COLORS["lindero_fill"], COLORS["lindero_stroke"],
               linestyle='--', linewidth=1.5, zorder=1, alpha_fill=0.3)

    # ── RETIRO ──
    if retiro_poly:
        _draw_poly(ax, retiro_poly, COLORS["retiro_fill"], COLORS["retiro_stroke"],
                   label="RETIRO\nMUNICIPAL", linewidth=0.8, zorder=1.5, fontsize=7)

    # ── DEPARTAMENTOS ──
    dptos = geo.get("departamentos", [])
    for i, dpto_coords in enumerate(dptos):
        if not dpto_coords or len(dpto_coords) < 3:
            continue
        area = calc_poly_area(dpto_coords)
        typology = get_typology(area)
        color = get_typology_color(typology)
        unit_id = f"X{i + 1:02d}"

        _draw_poly(ax, dpto_coords, color, COLORS["dpto_stroke"],
                   linewidth=1.2, zorder=3, alpha_fill=0.75)
        _draw_complex_label(ax, dpto_coords, typology, unit_id,
                            f"{area:.2f} m²")

    # ── HALL ──
    hall = geo.get("hall", [])
    if hall and len(hall) >= 3:
        _draw_poly(ax, hall, COLORS["hall_fill"], COLORS["hall_stroke"],
                   linewidth=1.2, zorder=4, alpha_fill=0.6)

    # ── CORE ──
    core = geo.get("core", [])
    if core and len(core) >= 3:
        _draw_poly(ax, core, COLORS["core_fill"], COLORS["core_stroke"],
                   linewidth=1.2, zorder=5, alpha_fill=0.5)

    # ── ESCALERA ──
    esc_presurizada = normativa.get("esc_protegida_obligatoria", False)
    escalera = geo.get("escalera", [])
    if escalera and len(escalera) >= 3:
        esc_fill = COLORS["escalera_pres"] if esc_presurizada else COLORS["escalera_fill"]
        esc_label = "ESC\nPRES. [P]" if esc_presurizada else "ESC\n\u21cb"
        _draw_poly(ax, escalera, esc_fill, COLORS["escalera_stroke"],
                   label=esc_label, linewidth=1.5, decor='stairs', zorder=6)
        _draw_poly_dimensions(ax, escalera)

    # ── VESTÍBULO ──
    vestibulo = geo.get("vestibulo", [])
    if vestibulo and len(vestibulo) >= 3:
        _draw_poly(ax, vestibulo, COLORS["vestibulo_fill"],
                   COLORS["vestibulo_stroke"], label="VEST.\nPREVIO",
                   linewidth=1.5, zorder=6)
        _draw_poly_dimensions(ax, vestibulo)

    # ── ASCENSOR(ES) ──
    ascensores = geo.get("ascensores", [])
    for asc in ascensores:
        if asc and len(asc) >= 3:
            _draw_poly(ax, asc, COLORS["ascensor_fill"],
                       COLORS["ascensor_stroke"], label="ASC\n[X]",
                       linewidth=1.5, decor='asc', zorder=6)
            _draw_poly_dimensions(ax, asc)

    # ── PATIO DE LUCES ──
    patio = geo.get("patio", [])
    if patio and len(patio) >= 3:
        pozo_final = normativa.get("pozo_final", 2.2)
        _draw_poly(ax, patio, COLORS["patio_fill"], COLORS["patio_stroke"],
                   label="PATIO DE\nLUCES", linewidth=1.5, decor='xBox', zorder=5)
        _draw_poly_dimensions(ax, patio)
        # Annotate minimum
        cx, cy = poly_centroid(patio)
        ax.text(cx, cy - 1.2, f"d≥{pozo_final:.1f}m (H/4)", fontsize=6,
                ha='center', va='center', color=COLORS["patio_stroke"],
                zorder=10, path_effects=[pe.withStroke(linewidth=2,
                                                       foreground='white')])

    # ── DUCTOS DE SERVICIO ──
    ductos = geo.get("ductos", [])
    for d in ductos:
        if d and len(d) >= 3:
            _draw_poly(ax, d, COLORS["ducto_fill"], COLORS["ducto_stroke"],
                       label="DUCTO\nSERV.", linewidth=1.2, decor='xBox', zorder=5)
            _draw_poly_dimensions(ax, d)

    # ── COTAS DEL TERRENO ──
    _draw_lindero_cotas(ax, polygon_pts, params)

    # ── AXIS CONFIG ──
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()  # Y increases downward like a plan view
    ax.set_xlabel("metros", fontsize=8, color='#64748b')
    ax.set_ylabel("metros", fontsize=8, color='#64748b')
    ax.tick_params(labelsize=7, colors='#94a3b8')

    # Title
    H = pisos * altura_piso
    area_terreno = calc_poly_area(polygon_pts)
    ax.set_title(
        f"PLANTA TÍPICA — {len(dptos)} dptos/piso · {pisos} pisos · "
        f"H={H:.1f}m · Terreno: {area_terreno:.1f}m²",
        fontsize=10, fontweight='bold', color='#1e293b', pad=12
    )

    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        spine.set_linewidth(0.5)

    plt.tight_layout()

    # ── EXPORT as base64 ──
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=COLORS["background"], edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_primer_piso(data: Dict[str, Any], params: Dict[str, Any],
                       primer_piso_data: Dict[str, Any],
                       width_px: int = 2400, height_px: int = 1800) -> str:
    """Render the first floor plan as a base64 PNG."""
    dpi = 200
    fig, ax = plt.subplots(1, 1,
                           figsize=(width_px / dpi, height_px / dpi),
                           dpi=dpi)
    ax.set_facecolor(COLORS["background"])
    ax.set_aspect('equal')

    geo = data.get("geometria_generada", {})
    normativa = data.get("normativa_estricta", {})
    pp = primer_piso_data

    # Terrain
    frente = params.get("frente", 10)
    fondo_val = params.get("fondo", 10)
    derecha = params.get("derecha", 20)
    izquierda = params.get("izquierda", 20)

    p1 = {"x": -frente / 2, "y": 0}
    p2 = {"x": frente / 2, "y": 0}
    p3 = {"x": fondo_val / 2, "y": derecha}
    p4 = {"x": -fondo_val / 2, "y": izquierda}
    polygon_pts = [p1, p2, p3, p4]

    all_xs = [p["x"] for p in polygon_pts]
    all_ys = [p["y"] for p in polygon_pts]
    margin = 3.0
    x_min, x_max = min(all_xs) - margin, max(all_xs) + margin
    y_min, y_max = min(all_ys) - margin, max(all_ys) + margin

    # Grid
    for gx in np.arange(math.floor(x_min), math.ceil(x_max) + 1, 1.0):
        ax.axvline(x=gx, color=COLORS["grid"], linewidth=0.3, zorder=0)
    for gy in np.arange(math.floor(y_min), math.ceil(y_max) + 1, 1.0):
        ax.axhline(y=gy, color=COLORS["grid"], linewidth=0.3, zorder=0)

    # Linderos
    _draw_poly(ax, polygon_pts, COLORS["lindero_fill"], COLORS["lindero_stroke"],
               linestyle='--', linewidth=1.5, zorder=1, alpha_fill=0.3)

    # Comercios
    for com in pp.get("comercios", []):
        if com and len(com) >= 3:
            _draw_poly(ax, com, COLORS["comercio_fill"], COLORS["comercio_stroke"],
                       label="COMERCIOS\nY USOS LOCALES", linewidth=1.2, zorder=3)

    # Servicios
    servicios = pp.get("servicios", [])
    if servicios and len(servicios) >= 3:
        _draw_poly(ax, servicios, COLORS["servicios_fill"],
                   COLORS["servicios_stroke"], label="SSHH/\nBASURA",
                   linewidth=1.2, zorder=3)

    # Lobby
    lobby = pp.get("lobby", [])
    if lobby and len(lobby) >= 3:
        _draw_poly(ax, lobby, COLORS["lobby_fill"], COLORS["lobby_stroke"],
                   label="LOBBY RECP.", linewidth=1.2, zorder=3)

    # Rampa
    rampa = pp.get("rampa", [])
    if rampa and len(rampa) >= 3:
        _draw_poly(ax, rampa, COLORS["rampa_fill"], COLORS["rampa_stroke"],
                   label="RAMPA\nVEHIC.\n3.00m", linewidth=1.2, zorder=3)

    # Core elements
    esc_presurizada = normativa.get("esc_protegida_obligatoria", False)
    escalera = geo.get("escalera", [])
    if escalera and len(escalera) >= 3:
        esc_fill = COLORS["escalera_pres"] if esc_presurizada else COLORS["escalera_fill"]
        _draw_poly(ax, escalera, esc_fill, COLORS["escalera_stroke"],
                   label="ESC.", linewidth=1.5, decor='stairs', zorder=6)

    ascensores = geo.get("ascensores", [])
    for asc in ascensores:
        if asc and len(asc) >= 3:
            _draw_poly(ax, asc, COLORS["ascensor_fill"],
                       COLORS["ascensor_stroke"], label="ASC\n[X]",
                       linewidth=1.5, decor='asc', zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_title("PRIMER PISO — Comercio, Lobby y Acceso Vehicular",
                 fontsize=10, fontweight='bold', color='#1e293b', pad=12)
    ax.tick_params(labelsize=7, colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        spine.set_linewidth(0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=COLORS["background"], edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_sotano(data: Dict[str, Any], params: Dict[str, Any],
                  sotano_data: Dict[str, Any],
                  width_px: int = 2400, height_px: int = 1800) -> str:
    """Render a basement level as a base64 PNG."""
    dpi = 200
    fig, ax = plt.subplots(1, 1,
                           figsize=(width_px / dpi, height_px / dpi),
                           dpi=dpi)
    ax.set_facecolor(COLORS["background"])
    ax.set_aspect('equal')

    geo = data.get("geometria_generada", {})
    normativa = data.get("normativa_estricta", {})
    sot = sotano_data

    # Slab
    slab = sot.get("slab", [])
    if slab and len(slab) >= 3:
        _draw_poly(ax, slab, COLORS["slab_fill"], COLORS["slab_stroke"],
                   linewidth=1.5, zorder=1)

    # Core
    core = geo.get("core", [])
    if core and len(core) >= 3:
        _draw_poly(ax, core, COLORS["core_fill"], COLORS["core_stroke"],
                   label="NÚCLEO", linewidth=1, zorder=4)

    esc_presurizada = normativa.get("esc_protegida_obligatoria", False)
    escalera = geo.get("escalera", [])
    if escalera and len(escalera) >= 3:
        esc_fill = COLORS["escalera_pres"] if esc_presurizada else COLORS["escalera_fill"]
        _draw_poly(ax, escalera, esc_fill, COLORS["escalera_stroke"],
                   label="ESC.", linewidth=1.5, decor='stairs', zorder=5)

    ascensores = geo.get("ascensores", [])
    for asc in ascensores:
        if asc and len(asc) >= 3:
            _draw_poly(ax, asc, COLORS["ascensor_fill"],
                       COLORS["ascensor_stroke"], label="ASC\n[X]",
                       linewidth=1.5, decor='asc', zorder=5)

    vestibulo = geo.get("vestibulo", [])
    if vestibulo and len(vestibulo) >= 3:
        _draw_poly(ax, vestibulo, COLORS["vestibulo_fill"],
                   COLORS["vestibulo_stroke"], label="VEST.\nPREVIO",
                   linewidth=1.5, zorder=5)

    # Rampa
    rampa = sot.get("rampa", [])
    if rampa and len(rampa) >= 3:
        _draw_poly(ax, rampa, COLORS["rampa_fill"], COLORS["rampa_stroke"],
                   label="VACÍO RAMPA", linewidth=1, zorder=4)

    # Aisles
    for aisle in sot.get("aisles", []):
        if aisle and len(aisle) >= 3:
            _draw_poly(ax, aisle, COLORS["aisle_fill"], COLORS["aisle_stroke"],
                       label="PASILLO 6.00m", linewidth=0.8, zorder=2, fontsize=6)

    # Stalls
    for stall in sot.get("stalls", []):
        poly = stall.get("poly", [])
        stall_id = stall.get("id", "")
        if poly and len(poly) >= 3:
            _draw_poly(ax, poly, COLORS["stall_fill"], COLORS["stall_stroke"],
                       label=stall_id, linewidth=0.7, zorder=3, fontsize=5.5)

    # Cisterns
    for cist in sot.get("cisternas", []):
        poly = cist.get("poly", [])
        if poly and len(poly) >= 3:
            _draw_poly(ax, poly, cist.get("fill", "#bfdbfe"),
                       cist.get("stroke", "#2563eb"),
                       label=cist.get("label", ""), linewidth=1.2, zorder=4,
                       fontsize=6)

    # Auto-fit
    all_polys = [slab] + [s.get("poly", []) for s in sot.get("stalls", [])] + \
                [a for a in sot.get("aisles", [])]
    all_xs = []
    all_ys = []
    for poly in all_polys:
        if poly:
            all_xs.extend([p["x"] for p in poly])
            all_ys.extend([p["y"] for p in poly])
    if not all_xs:
        all_xs = [0, 10]
        all_ys = [0, 10]

    margin = 2.0
    ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
    ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    ax.invert_yaxis()

    level_name = sot.get("name", "S1")
    stall_count = sot.get("count", 0)
    ax.set_title(
        f"SÓTANO {level_name} — {stall_count} estacionamientos",
        fontsize=10, fontweight='bold', color='#1e293b', pad=12
    )
    ax.tick_params(labelsize=7, colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        spine.set_linewidth(0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=COLORS["background"], edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
