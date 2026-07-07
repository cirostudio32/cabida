import requests

coords = [(0,0),(30,0),(30,62.5),(0,62.5)]
payload = {
    "coordenadas_lote": coords, "area_bruta_terreno": 1875.0,
    "num_departamentos": 8, "numero_pisos": 3,
    "retiro_frontal": 0.0, "retiro_lateral": 0.0, "retiro_posterior": 0.0,
    "num_ascensores": 0, "altura_piso": 2.5, "zonificacion": "RDM",
}
r = requests.post("http://localhost:8000/auditoria-rne", json=payload, timeout=30)
data = r.json()

ggen = data.get("geometria_generada", {})
dptos = ggen.get("departamentos", [])
print(f"dptos en ggen: {len(dptos)}")
for d in dptos[:4]:
    if isinstance(d, dict):
        print(f"  id={d.get('id')} area={d.get('area_m2')} lado={d.get('lado')}")

print()
unidades = data["geometria"]["unidades"]
for u in unidades[:4]:
    md = u.get("metadata", {})
    print(f"  id={u['id']} area={md.get('area')} tipo={md.get('tipologia')} lado={md.get('lado')}")

# Check evaluacion
print()
ev = ggen.get("evaluacion", {})
print("evaluacion keys:", list(ev.keys()) if isinstance(ev, dict) else ev)
