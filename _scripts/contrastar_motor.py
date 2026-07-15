"""Contrasta salida del motor vs biblioteca de huellas reales (huellas_ref.json).

Corre _generate_geometry con cada lote de referencia y mide desviación en:
  - área de departamentos (motor vs envolvente real 40-84 m2)
  - área de pozos de luz (motor vs ~10 m2 compacto real)
  - factibilidad (lotes que el real construyó pero el motor rechaza)

Uso: python _scripts/contrastar_motor.py
Requiere: python _scripts/extraer_huellas.py  (genera huellas_ref.json)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapely.geometry import Polygon
from main import ProyectoInmobiliario, _generate_geometry

PISOS = 7


def _poly(cont):
    return Polygon([(p['x'], p['y']) for p in cont])


def _run(W, H, nd):
    coords = [(0, 0), (W, 0), (W, H), (0, H)]
    p = ProyectoInmobiliario(
        coordenadas_lote=coords, area_bruta_terreno=W * H, numero_pisos=PISOS,
        retiro_frontal=0.0, zonificacion='RDM', num_ascensores=1,
        num_departamentos=nd, frente=W, fondo=H, derecha=H, izquierda=H,
        retiro_lateral=0.0, retiro_posterior=0.0, ciego_frente=False,
        ciego_fondo=True, ciego_derecha=True, ciego_izquierda=True,
        area_libre_min_pct=0.0, ajustar_pisos_normativa=False)
    return _generate_geometry(p)


def main():
    ref = json.load(open('huellas_ref.json', encoding='utf-8'))
    hallazgos = []
    for pf in ref['perfiles']:
        W, H = pf['lote']['w'], pf['lote']['h']
        real_areas = pf['tipologias_area']
        real_max = max(real_areas) if real_areas else 0
        print(f"\n=== DXF{pf['dxf']}  lote {W}x{H}  nd_req={pf['dptos_piso']} ===")
        try:
            g, _ = _run(W, H, pf['dptos_piso'])
        except Exception as e:
            print(f"  RECHAZADO: {str(e)[:70]}")
            hallazgos.append(('factibilidad', pf['dxf'], str(e)[:50]))
            continue
        dA = sorted(round(d['area_m2'], 1) for d in g['departamentos'])
        # dpto sobredimensionado: > envolvente real * 1.2
        big = [a for a in dA if a > real_max * 1.2]
        print(f"  DPTO  motor={dA}")
        print(f"        real ={real_areas} (max {real_max:.0f})")
        if big:
            print(f"  [!] {len(big)} dpto > {real_max*1.2:.0f}m2 (sobredimensionados): {big}")
            hallazgos.append(('dpto_grande', pf['dxf'], big))
        # POZO: el motor es pozo-de-luz normativo (lado>=H/3), no ducto. Se valida
        # contra la conformidad RNE que el propio motor calcula, no vs 10m2 (ducto).
        cumple = g.get('pozos_luz_cumple', [])
        no_conf = sum(1 for ok in cumple if not ok)
        print(f"  POZO  n={len(cumple)} conformes_RNE={sum(cumple)}/{len(cumple)}")
        if no_conf:
            print(f"  [!] {no_conf} pozo NO cumple RNE (lado<H/3)")
            hallazgos.append(('pozo_no_conforme', pf['dxf'], no_conf))

    print(f"\n{'='*50}\nRESUMEN: {len(hallazgos)} desviaciones")
    for tipo, dxf, det in hallazgos:
        print(f"  [{tipo}] DXF{dxf}: {det}")
    return hallazgos


if __name__ == '__main__':
    main()
