"""Extrae perfil de huella de los DXF de referencia -> huellas_ref.json.

No reconstruye polígonos por-dpto (MA_AREAS no es limpio; requeriría
polygonizar MA_MUROS, frágil). Extrae lo que SÍ es fiable y sirve para
calibrar/validar el motor:
  - lote (W x H, área)
  - huella global: bbox + rectangularidad (detecta barra vs L/U)
  - pozos/ductos: dims + posición relativa al lote (near-medianera vs centro)
  - tipologías: áreas de dpto desde labels MA_TEXTO 1
  - dpto/piso

Uso: python _scripts/extraer_huellas.py  ->  huellas_ref.json
"""
import ezdxf, re, json
from shapely.geometry import Polygon

REFS = [1, 2, 3, 4, 6]  # 5.dxf usa layers distintos (levantamiento 11MB), se omite


def _lote(msp):
    lote = None
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE' and 'PERIMETRAL' in e.dxf.layer.upper():
            pg = Polygon([(p[0], p[1]) for p in e.get_points('xy')])
            if pg.is_valid and (lote is None or pg.area > lote.area):
                lote = pg
    return lote


def _perfil(i):
    doc = ezdxf.readfile(f'referencias/{i}.dxf')
    msp = doc.modelspace()
    lote = _lote(msp)
    lb = lote.bounds
    LW, LH = lb[2] - lb[0], lb[3] - lb[1]

    huella_bbox = None   # mayor polígono MA_AREAS = huella construida del piso
    pozos = []           # rectángulos chicos = pozos/ductos
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE' and e.dxf.layer == 'MA_AREAS':
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
            if len(pts) < 3:
                continue
            pg = Polygon(pts)
            if not pg.is_valid:
                continue
            b = pg.bounds
            bw, bh = b[2] - b[0], b[3] - b[1]
            rect = pg.area / (bw * bh) if bw * bh else 0
            if pg.area > 20:  # huella de bloque
                if huella_bbox is None or pg.area > huella_bbox['area']:
                    huella_bbox = {'area': round(pg.area, 1), 'w': round(bw, 2),
                                   'h': round(bh, 2), 'rectangularidad': round(rect, 2)}
            elif 1.0 < pg.area <= 20:  # pozo/ducto
                c = pg.centroid
                pozos.append({'area': round(pg.area, 1), 'w': round(bw, 2),
                              'h': round(bh, 2),
                              'pos_x': round((c.x - lb[0]) / LW, 2),
                              'pos_y': round((c.y - lb[1]) / LH, 2),
                              # distancia mínima a medianera lateral (0=pegado)
                              'd_medianera': round(min((c.x - lb[0]) / LW,
                                                       (lb[2] - c.x) / LW), 2)})

    # áreas de tipología desde labels
    areas = []
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.layer == 'MA_TEXTO 1':
            m = re.match(r'^([\d.]+)\s*m', e.dxf.text.strip())
            if m:
                v = float(m.group(1))
                if 15 < v < 200:
                    areas.append(round(v, 1))
    dptos = {e.dxf.text.strip() for e in msp
             if e.dxftype() == 'TEXT' and e.dxf.layer == 'MA_0.40'
             and e.dxf.text.strip().startswith('DPTO')}

    # Núcleo/circulación vertical: los DXF no tienen capa dedicada, pero el label
    # HALL marca el punto de circulación. Su posición relativa (x_rel) dice si el
    # núcleo real es lateral (cerca de medianera) o central. Esto dicta el
    # criterio de ubicación del núcleo en el motor (ver _generate_costillas).
    nucleos = []
    for e in msp:
        if e.dxftype() in ('TEXT', 'MTEXT'):
            t = (e.dxf.text if e.dxftype() == 'TEXT' else e.text).strip().upper()
            if any(k in t for k in ('HALL', 'ESCAL', 'ASCEN', 'NUCLE', 'VEST')):
                try:
                    ins = e.dxf.insert
                except Exception:
                    continue
                x_rel = round((ins[0] - lb[0]) / LW, 2)
                nucleos.append({
                    'label': t[:16],
                    'x_rel': x_rel,
                    'y_rel': round((ins[1] - lb[1]) / LH, 2),
                    'd_medianera': round(min(x_rel, 1 - x_rel), 2),  # 0=pegado
                })
    # clasificación lateral vs central (umbral 0.35 desde medianera)
    if nucleos:
        d_min = min(n['d_medianera'] for n in nucleos)
        nucleo_tipo = 'lateral' if d_min < 0.35 else 'central'
    else:
        nucleo_tipo = None

    return {
        'dxf': i,
        'lote': {'w': round(LW, 2), 'h': round(LH, 2), 'area': round(lote.area, 1)},
        'huella': huella_bbox,
        'pozos': pozos,
        'tipologias_area': sorted(set(areas)),
        'dptos_piso': len(dptos),
        'nucleos': nucleos,
        'nucleo_tipo': nucleo_tipo,
    }


def main():
    perfiles = [_perfil(i) for i in REFS]
    out = {'fuente': 'referencias/*.dxf', 'perfiles': perfiles}
    with open('huellas_ref.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # --- self-check: patrón esperado de pozos (pegados a medianera) ---
    todos_pozos = [p for pf in perfiles for p in pf['pozos']]
    assert todos_pozos, "no se extrajo ningún pozo"
    cerca_medianera = sum(1 for p in todos_pozos if p['d_medianera'] < 0.25)
    frac = cerca_medianera / len(todos_pozos)
    assert frac >= 0.5, f"esperado pozos cerca de medianera, got {frac:.0%}"
    print(f"OK: {len(perfiles)} perfiles, {len(todos_pozos)} pozos, "
          f"{frac:.0%} pegados a medianera -> huellas_ref.json")


if __name__ == '__main__':
    main()
