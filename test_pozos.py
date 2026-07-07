import urllib.request, json, sys
sys.path.insert(0, ".")
from shapely.geometry import Polygon as SP

def api_test(label, lot, pisos, retlat, retpos, ndptos, cieg_iz, cieg_der, cieg_fondo):
    payload = json.dumps({
        "coordenadas_lote": lot, "area_bruta_terreno": 700,
        "numero_pisos": pisos, "retiro_frontal": 3.2,
        "retiro_lateral": retlat, "retiro_posterior": retpos,
        "zonificacion": "RDM", "num_ascensores": 1, "num_departamentos": ndptos,
        "ciego_izquierda": cieg_iz, "ciego_derecha": cieg_der, "ciego_fondo": cieg_fondo,
        "esquema_area_libre": "muros_ciegos",
        "frente_exterior": True, "fondo_exterior": False,
        "izquierda_exterior": False, "derecha_exterior": False
    }).encode()
    req = urllib.request.Request("http://localhost:8000/auditoria-rne", data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    geo = data.get("geometria", {})
    units = geo.get("unidades", [])
    tec = geo.get("tecnico", {})
    pozos = tec.get("pozos_luz", [])
    overlaps = 0
    for u in units:
        uc = u.get("coords", [])
        if len(uc) < 3: continue
        try:
            up = SP([(c[0],c[1]) for c in uc])
            for pz in pozos:
                pc = pz.get("coords", [])
                if len(pc) < 3: continue
                pp = SP([(c[0],c[1]) for c in pc])
                if up.intersection(pp).area > 0.01:
                    overlaps += 1
        except: pass
    status = "OK" if overlaps == 0 else "FAIL"
    print(label + ": dptos=" + str(len(units)) + " pozos=" + str(len(pozos)) + " overlaps=" + str(overlaps) + " -> " + status)

api_test("M1 HORIZ 40x20 ret2.3", [[0,0],[0,20],[40,20],[40,0]], 8, 2.3, 2.3, 8, True, True, True)
api_test("M2 VERT 20x35 lat=0",   [[0,0],[0,35],[20,35],[20,0]], 8, 0.0, 2.3, 6, True, True, True)
api_test("M3 HORIZ 30x20 pos=0",  [[0,0],[0,20],[30,20],[30,0]], 8, 2.3, 0.0, 6, True, True, True)
