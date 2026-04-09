import requests
import json

payload = {
    "coordenadas_lote": [[0,0],[20,0],[20,20],[0,20]],
    "area_bruta_terreno": 400,
    "numero_pisos": 5,
    "retiro_frontal": 0,
    "zonificacion": "RDA",
    "num_ascensores": 1,
    "num_departamentos": 4,
    "frente": 20,
    "fondo": 20,
    "derecha": 20,
    "izquierda": 20,
    "altura_piso": 2.8,
    "pct_estac": 30,
    "ciego_frente": False,
    "ciego_fondo": True,
    "ciego_derecha": True,
    "ciego_izquierda": True
}

try:
    res = requests.post('http://127.0.0.1:8000/auditoria-rne', json=payload)
    if res.status_code == 200:
        data = res.json()
        print("Status Code: 200")
        print("Geometria keys:", data.get("geometria", {}).keys())
        unidades = data.get("geometria", {}).get("unidades", [])
        print("Number of unidades in geometria:", len(unidades))
        print("geometria_generada keys:", data.get("geometria_generada", {}).keys())
        deptos = data.get("geometria_generada", {}).get("departamentos", [])
        print("Number of departamentos in geometria_generada:", len(deptos))
    else:
        print(f"Error {res.status_code}: {res.text}")
except Exception as e:
    print("Failed to request:", e)
