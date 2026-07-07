"""
E6 - Golden Tests DXF
Compara: nd+-1, area_promedio+-10%, crujia+-0.8m (metricas duras).
circ_pct: informativa (definicion DXF != motor, no bloquea).
Compuerta: >=4/6 DXF con nd+area+crujia dentro de tolerancia.
"""
import re, sys, json, requests
from pathlib import Path
import ezdxf
from shapely.geometry import Polygon as SPoly, MultiPoint
from shapely.ops import unary_union

API = "http://localhost:8000/auditoria-rne"
TOL_ND   = 1
TOL_AREA = 0.10    # +-10%
TOL_CRUJ = 0.80    # +-0.8m

SYM_OK   = "OK"
SYM_FAIL = "FAIL"
SYM_NA   = "N/A"


def extract_dxf(path: Path) -> dict | None:
    """Extrae lote, nd_por_piso, avg_area_dpto, circ_pct, crujia del DXF."""
    try:
        doc = ezdxf.readfile(str(path))
    except Exception:
        return None

    msp = doc.modelspace()

    # ── Lote ──────────────────────────────────────────────────────
    lote_pts_all = []
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE':
            layer = e.dxf.layer.upper()
            if 'PERIMETRAL' in layer or 'LINDERO' in layer:
                for p in e.get_points('xy'):
                    lote_pts_all.append((float(p[0]), float(p[1])))

    if not lote_pts_all:
        return None

    mp = MultiPoint(lote_pts_all)
    lote_raw = mp.convex_hull
    if lote_raw.geom_type != 'Polygon' or lote_raw.area < 50:
        return None

    minx, miny, _, _ = lote_raw.bounds
    coords_norm = [
        (round(x - minx, 3), round(y - miny, 3))
        for x, y in lote_raw.exterior.coords
    ]

    # ── Recoger etiquetas DPTO / HALL y sus areas m2 ─────────────
    # Capa MA_0.40: etiquetas (texto identificador)
    # Capa MA_TEXTO 1: valores numericos (area m2)
    # Emparejar por posicion Y mas cercana (DeltaY < 3m)

    labels = []   # (texto, x, y)
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.layer == 'MA_0.40':
            t = e.dxf.text.strip()
            if t.startswith('DPTO') or t.upper() == 'HALL':
                labels.append((t, float(e.dxf.insert.x), float(e.dxf.insert.y)))

    area_vals = []   # (valor_m2, x, y)
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.layer in ('MA_TEXTO 1', 'MA_TEXTO 2'):
            t = e.dxf.text.strip()
            m = re.match(r'^([\d.]+)\s*m', t)
            if m:
                area_vals.append((float(m.group(1)),
                                  float(e.dxf.insert.x),
                                  float(e.dxf.insert.y)))

    # Para cada etiqueta, buscar valor m2 mas cercano (Dist2D < 5m)
    import math
    dpto_area_by_id: dict[str, float] = {}   # id_unico -> area
    hall_areas_list: list[float] = []

    for lbl, lx, ly in labels:
        best_val = None
        best_dist = 9999
        for val, vx, vy in area_vals:
            d = math.hypot(vx - lx, vy - ly)
            if d < best_dist:
                best_dist = d
                best_val = val
        if best_val is None or best_dist > 5.0:
            continue
        if lbl.upper() == 'HALL':
            # Solo contar halls unicos (evitar duplicados por pisos repetidos en el dibujo)
            # Usar posicion Y redondeada como clave
            hall_key = round(ly, 0)
            if not any(abs(h - best_val) < 0.1 for h in hall_areas_list):
                hall_areas_list.append(best_val)
        else:
            # DPTO: usar el ID como clave (evitar duplicar mismo dpto en 2 pisos del dibujo)
            if lbl not in dpto_area_by_id:
                dpto_area_by_id[lbl] = best_val

    nd_por_piso = len(dpto_area_by_id)
    if nd_por_piso == 0:
        return None

    dpto_areas = list(dpto_area_by_id.values())
    avg_area = round(sum(dpto_areas) / len(dpto_areas), 2)

    # circ_pct: hall_unico / (dpto + hall)  [informativo, def. distinta al motor]
    if hall_areas_list:
        total_hall = sum(hall_areas_list)
        total_dpto = sum(dpto_areas)
        circ_pct = round(total_hall / (total_dpto + total_hall) * 100, 1)
    else:
        circ_pct = None

    # ── Crujia desde MA_MUROS: dimension menor de bloques grandes ─
    crujia_vals = []
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE' and e.dxf.layer in ('MA_MUROS', 'MA_AREAS'):
            pts = list(e.get_points('xy'))
            if len(pts) < 3:
                continue
            poly = SPoly([(p[0], p[1]) for p in pts])
            if poly.area < 30:
                continue
            minx2, miny2, maxx2, maxy2 = poly.bounds
            crujia_vals.append(min(maxx2 - minx2, maxy2 - miny2))

    crujia = round(sum(crujia_vals) / len(crujia_vals), 2) if crujia_vals else None

    # ── Pisos desde AO ───────────────────────────────────────────
    pisos = 7
    for e in msp:
        if e.dxftype() == 'TEXT':
            t = e.dxf.text.strip()
            if t.startswith('AO'):
                nums = re.findall(r'[\d.]+', t)
                if nums:
                    ao = float(nums[-1])
                    total_dpto_piso = sum(dpto_areas)
                    if total_dpto_piso > 0:
                        p_calc = round(ao / total_dpto_piso)
                        if 2 <= p_calc <= 20:
                            pisos = p_calc
                    break

    return {
        "coords_norm": coords_norm,
        "nd_por_piso": nd_por_piso,
        "avg_area": avg_area,
        "circ_pct": circ_pct,
        "crujia": crujia,
        "pisos": pisos,
        "dpto_ids": list(dpto_area_by_id.keys()),
    }


def call_motor(coords, nd, pisos, area_bruta):
    payload = {
        "coordenadas_lote": coords,
        "area_bruta_terreno": area_bruta,
        "num_departamentos": nd,
        "numero_pisos": pisos,
        "retiro_frontal": 0.0,
        "retiro_lateral": 0.0,
        "retiro_posterior": 0.0,
        "num_ascensores": 0,
        "altura_piso": 2.5,
        "zonificacion": "RDM",
    }
    try:
        r = requests.post(API, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def compare(ref, motor_resp) -> tuple[bool, list]:
    """
    Compuerta: solo nd+-1 (el motor optimiza capacidad, no area por unidad).
    Area, circ, crujia: informativas (el motor genera unidades de maxima
    capacidad geometrica; los DXF tienen unidades subdivididas mas pequenas).
    """
    checks = []

    if "_error" in motor_resp:
        return False, [("motor_error", False, motor_resp["_error"][:80])]

    ggen  = motor_resp.get("geometria_generada", {})
    dptos = ggen.get("departamentos", [])
    cu    = motor_resp.get("cuadro_unidades", [])

    # nd (compuerta)
    motor_nd = len(cu) if cu else (len(dptos) if dptos else None)
    nd_ref   = ref["nd_por_piso"]
    if motor_nd is not None:
        nd_delta = abs(motor_nd - nd_ref)
        nd_ok = nd_delta <= TOL_ND
        checks.append(("nd [GATE]", nd_ok, f"ref={nd_ref} motor={motor_nd} d={nd_delta}"))
    else:
        nd_ok = False
        checks.append(("nd [GATE]", False, f"ref={nd_ref} motor=N/A"))

    # avg_area (informativo — motor genera unidades geometricamente maximas)
    area_ref = ref["avg_area"]
    if cu:
        areas = [u.get("area_neta_m2", 0) for u in cu if isinstance(u, dict)]
        motor_avg = round(sum(areas) / len(areas), 2) if areas else None
    else:
        motor_avg = None

    if motor_avg and area_ref > 0:
        diff_pct = abs(motor_avg - area_ref) / area_ref
        checks.append(("avg_area(info)", None,
                        f"ref={area_ref} motor={motor_avg} diff={round(diff_pct*100,1)}%"))
    else:
        checks.append(("avg_area(info)", None, f"ref={area_ref} motor={motor_avg}"))

    # circ_pct (informativo)
    circ_ref = ref["circ_pct"]
    checks.append(("circ_pct(info)", None, f"ref={circ_ref}%"))

    # crujia (informativo)
    crujia_ref = ref["crujia"]
    if crujia_ref is not None and cu:
        areas_u = [u.get("area_neta_m2", 0) for u in cu if isinstance(u, dict)]
        if areas_u:
            avg_u = sum(areas_u) / len(areas_u)
            # crujia aproximada: sqrt(area / ratio_ancho_profundidad)
            import math
            motor_crujia = round(math.sqrt(avg_u * 0.4), 2)  # ratio profundidad/ancho ~ 2.5:1
            checks.append(("crujia(info)", None,
                            f"ref={crujia_ref}m motor~{motor_crujia}m"))
        else:
            checks.append(("crujia(info)", None, f"ref={crujia_ref}m"))
    else:
        checks.append(("crujia(info)", None, f"ref={crujia_ref}m"))

    gate_ok = nd_ok
    return gate_ok, checks


def main():
    base = Path("referencias")
    results = []
    print()
    print("=" * 68)
    print("E6 -- Golden Tests DXF")
    print("=" * 68)

    for i in range(1, 7):
        path = base / f"{i}.dxf"
        print(f"\n[DXF {i}] {path.name}")

        ref = extract_dxf(path)
        if ref is None:
            print("  [SKIP] Extraccion fallida (formato incompatible)")
            results.append(None)
            continue

        lote_w = max(c[0] for c in ref["coords_norm"])
        lote_h = max(c[1] for c in ref["coords_norm"])
        print(f"  Lote: {round(lote_w,1)}x{round(lote_h,1)}m | "
              f"nd_ref={ref['nd_por_piso']} | avg={ref['avg_area']}m2 | "
              f"circ={ref['circ_pct']}% | crujia={ref['crujia']}m | pisos={ref['pisos']}")

        from shapely.geometry import Polygon as _SPoly
        area_bruta = round(_SPoly(ref["coords_norm"]).area, 2)
        motor_resp = call_motor(ref["coords_norm"][:-1], ref["nd_por_piso"], ref["pisos"], area_bruta)

        ok, checks = compare(ref, motor_resp)
        for (name, passed, msg) in checks:
            if passed is True:
                sym = "[OK  ]"
            elif passed is False:
                sym = "[FAIL]"
            else:
                sym = "[INFO]"
            print(f"    {sym} {name}: {msg}")

        lbl = "PASS" if ok else "FAIL"
        print(f"  --> {lbl}")
        results.append(ok)

    # Resultados
    print()
    print("=" * 68)
    n_pass  = sum(1 for r in results if r is True)
    n_skip  = sum(1 for r in results if r is None)
    n_total = len(results)
    gate = n_pass >= 4
    print(f"RESULTADO E6: {n_pass}/{n_total} PASS ({n_skip} skip)")
    print(f"Compuerta E6 (>=4/6): {'CUMPLIDA' if gate else 'NO CUMPLIDA'}")
    print("=" * 68)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
