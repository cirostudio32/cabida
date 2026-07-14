"""
E7 - Paridad geometry -> payload webgl.

El bug real (jul-2026): _generate_geometry arma bien las dos torres en
esquemas dos-nucleos (halls/escaleras/nucleos plural), pero
_build_webgl_payload y las plantas derivadas (primer_piso/sotano/azotea)
solo leian las claves singulares ("hall","escalera","ascensores") o
promediaban ambas torres -> nucleos huerfanos, lobby flotando en el
patio central, ascensores sin escalera. Este check corre el caso dorado
dos-nucleos end-to-end (igual que la API real) y verifica que CADA nucleo
del payload tenga hall+escalera+ascensores no vacios, y que cada
lobby/puerta de primer_piso caiga dentro del bbox de su propio nucleo
(no en el promedio/patio central).
"""
import sys
from shapely.geometry import Polygon
import main
import tests_cabida as tc


def _bbox(pts):
    xs = [p[0] if isinstance(p, (list, tuple)) else p["x"] for p in pts]
    return min(xs), max(xs)


def run():
    caso = next(c for c in tc.CASOS if c["name"] == "ANCHO_40x30_nd10")
    coords = tc._make_lote_coords(caso["frente"], caso["fondo"], caso["derecha"],
                                   caso["izquierda"], retiro_frontal=0.0)
    proyecto = main.ProyectoInmobiliario(
        coordenadas_lote=coords, area_bruta_terreno=float(Polygon(coords).area),
        numero_pisos=caso["pisos"], retiro_frontal=0.0, zonificacion="RDM",
        num_ascensores=1, num_departamentos=caso["nd"],
        frente=float(caso["frente"]), fondo=float(caso["fondo"]),
        derecha=float(caso["derecha"]), izquierda=float(caso["izquierda"]),
        retiro_lateral=float(caso["retiro_lat"]), retiro_posterior=float(caso["retiro_pos"]),
        ciego_frente=False, ciego_fondo=True, ciego_derecha=True, ciego_izquierda=True,
        area_libre_min_pct=0.0, ajustar_pisos_normativa=False,
    )
    geometry, normativa = main._generate_geometry(proyecto)
    assert len(geometry.get("nucleos", [])) == 2, (
        f"caso dorado dos-nucleos debe generar 2 nucleos, genero {len(geometry.get('nucleos', []))}"
    )

    primer_piso = main._generate_primer_piso(proyecto, geometry)
    sotano = main._generate_sotano(proyecto, geometry, normativa)
    azotea = main._generate_azotea(proyecto, geometry, normativa)
    payload = main._build_webgl_payload(proyecto, geometry, normativa, primer_piso, sotano, azotea)
    g = payload["geometria"]

    fails = []

    nucleos = g["nucleo"]["nucleos"]
    if len(nucleos) != 2:
        fails.append(f"payload.nucleo.nucleos debe tener 2 entradas, tiene {len(nucleos)}")

    for i, nuc in enumerate(nucleos):
        hall = nuc["hall"]["coords"]
        esc = nuc["escalera"]["coords"]
        asc = nuc["ascensores"]
        if not hall:
            fails.append(f"nucleo[{i}]: hall vacio -> torre sin acceso")
        if not esc:
            fails.append(f"nucleo[{i}]: escalera vacia -> torre sin evacuacion")
        if not asc:
            fails.append(f"nucleo[{i}]: sin ascensores")
        if hall and esc and asc:
            hx = _bbox(hall)
            ex = _bbox(esc)
            # Escalera y hall del MISMO nucleo deben solaparse en X (adyacentes),
            # no estar en semiplanos opuestos del lote (eso es el bug de
            # nucleos divididos / ascensores huerfanos).
            overlap = min(hx[1], ex[1]) - max(hx[0], ex[0])
            if overlap < -1.0:
                fails.append(
                    f"nucleo[{i}]: hall X{hx} y escalera X{ex} no son adyacentes (torre partida)"
                )

    # Lobby/puerta: cada lobby debe caer dentro del bbox X del nucleo mas
    # cercano, nunca en el punto medio entre ambas torres (patio central).
    pp = g.get("primer_piso") or {}
    lobbies_list = pp.get("lobbies") or ([pp["lobby"]] if pp.get("lobby") else [])
    if len(lobbies_list) != 2:
        fails.append(f"primer_piso.lobbies debe tener 2 entradas (una por torre), tiene {len(lobbies_list)}")
    for i, lob in enumerate(lobbies_list):
        if not lob:
            fails.append(f"lobby[{i}] vacio")
            continue
        lx = _bbox(lob)
        matched = any(
            min(lx[1], _bbox(nuc["hall"]["coords"])[1]) - max(lx[0], _bbox(nuc["hall"]["coords"])[0]) > -1.0
            for nuc in nucleos if nuc["hall"]["coords"]
        )
        if not matched:
            fails.append(f"lobby[{i}] X{lx} no se solapa con ningun hall real -> lobby flotando")

    # Azotea: una caja de escalera + cuarto de maquinas por nucleo real.
    az = payload["geometria"]["azotea"]
    if len(az.get("cajas_escalera", [])) != 2:
        fails.append(f"azotea.cajas_escalera debe tener 2, tiene {len(az.get('cajas_escalera', []))}")
    if len(az.get("cuartos_maquinas", [])) != 2:
        fails.append(f"azotea.cuartos_maquinas debe tener 2, tiene {len(az.get('cuartos_maquinas', []))}")

    # Sotano: nucleo reservado debe cubrir AMBAS torres (>=4 piezas: 2 escaleras + 2+ ascensores).
    sot_nuc = payload["geometria"]["sotano"]["nucleo"]
    if len(sot_nuc) < 4:
        fails.append(f"sotano.nucleo debe reservar >=4 piezas (2 escaleras+2 ascensores), tiene {len(sot_nuc)}")

    if fails:
        print("E7 FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("E7 PASS: 2 nucleos completos, lobbies alineados, azotea/sotano cubren ambas torres.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
