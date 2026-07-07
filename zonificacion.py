"""
zonificacion.py — Tabla de parámetros urbanísticos por zona + validador.

Referencia genérica para Perú (RNE + ordenanzas municipales típicas de Lima).
Los valores son orientativos para cabida preliminar; no sustituyen
la ordenanza vigente del distrito. Cada proyecto debe verificar la
zonificación real en el certificado de parámetros municipal.

Campos por zona:
  cus               Coeficiente de Edificación máximo (area techada / area terreno)
  altura_max_m      Altura máxima edificable, metros
  area_libre_min    Fracción mínima del lote sin techar (0–1)
  retiro_frontal    Retiro frontal mínimo, metros
  retiro_lateral    Retiro lateral mínimo (cuando aplica), metros
  retiro_posterior  Retiro posterior mínimo, metros
  frente_min_m      Frente normativo mínimo del lote, metros
  densidad_max_hab_ha  Densidad neta máxima, habitantes / hectárea
  area_min_lote_m2  Área mínima del lote para acogerse a la zona
"""

from typing import Dict, Any, List, Optional


ZONIFICACION: Dict[str, Dict[str, Any]] = {
    "RDB": {
        "nombre": "Residencial Densidad Baja",
        "cus": 1.5,
        "altura_max_m": 9.0,
        "area_libre_min": 0.40,
        "retiro_frontal": 3.0,
        "retiro_lateral": 3.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 8.0,
        "densidad_max_hab_ha": 1300,
        "area_min_lote_m2": 160.0,
    },
    "RDM": {
        "nombre": "Residencial Densidad Media",
        "cus": 2.1,
        "altura_max_m": 15.0,
        "area_libre_min": 0.35,
        "retiro_frontal": 3.0,
        "retiro_lateral": 2.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 8.0,
        "densidad_max_hab_ha": 1800,
        "area_min_lote_m2": 160.0,
    },
    "RDA": {
        "nombre": "Residencial Densidad Alta",
        "cus": 3.5,
        "altura_max_m": 21.0,
        "area_libre_min": 0.30,
        "retiro_frontal": 3.0,
        "retiro_lateral": 3.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 10.0,
        "densidad_max_hab_ha": 2250,
        "area_min_lote_m2": 200.0,
    },
    "RDMA": {
        "nombre": "Residencial Densidad Muy Alta",
        "cus": 5.0,
        "altura_max_m": 45.0,
        "area_libre_min": 0.30,
        "retiro_frontal": 3.0,
        "retiro_lateral": 3.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 10.0,
        "densidad_max_hab_ha": 2900,
        "area_min_lote_m2": 450.0,
    },
    "CZ": {
        "nombre": "Comercio Zonal",
        "cus": 4.0,
        "altura_max_m": 24.0,
        "area_libre_min": 0.30,
        "retiro_frontal": 3.0,
        "retiro_lateral": 0.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 10.0,
        "densidad_max_hab_ha": 2250,
        "area_min_lote_m2": 250.0,
    },
    "CV": {
        "nombre": "Comercio Vecinal",
        "cus": 2.5,
        "altura_max_m": 12.0,
        "area_libre_min": 0.30,
        "retiro_frontal": 3.0,
        "retiro_lateral": 0.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 8.0,
        "densidad_max_hab_ha": 1800,
        "area_min_lote_m2": 160.0,
    },
    "CM": {
        "nombre": "Comercio Metropolitano",
        "cus": 6.0,
        "altura_max_m": 48.0,
        "area_libre_min": 0.30,
        "retiro_frontal": 3.0,
        "retiro_lateral": 0.0,
        "retiro_posterior": 3.0,
        "frente_min_m": 15.0,
        "densidad_max_hab_ha": 2900,
        "area_min_lote_m2": 450.0,
    },
}

DEFAULT_ZONA = "RDA"

HAB_PROMEDIO_POR_DEPTO: Dict[str, int] = {
    "1D": 2,
    "1D+E": 3,
    "2D": 3,
    "2D+E": 4,
    "3D": 5,
}


def get_zona(codigo: str) -> Dict[str, Any]:
    """Devuelve los parámetros de la zona. Fallback a DEFAULT_ZONA."""
    if not codigo:
        return ZONIFICACION[DEFAULT_ZONA]
    key = codigo.strip().upper()
    return ZONIFICACION.get(key, ZONIFICACION[DEFAULT_ZONA])


def _flag(ok: bool, msg_ok: str, msg_fail: str) -> Dict[str, Any]:
    return {"ok": bool(ok), "mensaje": msg_ok if ok else msg_fail}


def validar_zonificacion(
    zona_codigo: str,
    area_terreno_m2: float,
    frente_m: float,
    area_techada_total_m2: float,
    altura_edificio_m: float,
    area_libre_planta_m2: float,
    retiro_frontal_m: float,
    retiro_lateral_min_m: float,
    retiro_posterior_m: float,
    num_unidades_total: int,
    tipologias: Optional[List[str]] = None,
    retiro_lateral_aplica: bool = True,
    retiro_posterior_aplica: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Valida cumplimiento de parámetros urbanísticos.

    Args:
        zona_codigo: clave de la zona (e.g. "RDA")
        area_terreno_m2: área bruta del lote
        frente_m: frente del lote
        area_techada_total_m2: suma de área techada todos los pisos
        altura_edificio_m: altura total (m) sobre nivel terreno
        area_libre_planta_m2: m² libres en planta de primer piso (sin techar)
        retiro_frontal_m: retiro frontal aplicado
        retiro_lateral_min_m: el menor de los dos retiros laterales aplicados
        retiro_posterior_m: retiro posterior aplicado
        num_unidades_total: total de departamentos en todo el edificio
        tipologias: lista de tipologías por depto (para estimar habitantes)
        retiro_lateral_aplica: False si ambos laterales colindan con medianera
            (muro ciego) — el retiro lateral no es exigible en colindancia.
        retiro_posterior_aplica: False si el fondo colinda con medianera.
        overrides: valores del certificado de parámetros que reemplazan la
            tabla de zona (claves de ZONIFICACION, ej. "cus", "altura_max_m",
            "densidad_max_hab_ha", "area_libre_min").

    Returns:
        dict con todos los chequeos individuales + flag global.
    """
    zona = dict(get_zona(zona_codigo))
    parametros_certificado = []
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in zona:
                zona[k] = v
                parametros_certificado.append(k)
    cus_max = zona["cus"]
    h_max = zona["altura_max_m"]
    al_min_frac = zona["area_libre_min"]
    rf_min = zona["retiro_frontal"]
    rl_min = zona["retiro_lateral"]
    rp_min = zona["retiro_posterior"]
    frente_min = zona["frente_min_m"]
    dens_max = zona["densidad_max_hab_ha"]
    area_lote_min = zona["area_min_lote_m2"]

    cus_calc = (area_techada_total_m2 / area_terreno_m2) if area_terreno_m2 > 0 else 0.0
    al_frac_calc = (area_libre_planta_m2 / area_terreno_m2) if area_terreno_m2 > 0 else 0.0

    if tipologias:
        hab_total = sum(HAB_PROMEDIO_POR_DEPTO.get(t, 3) for t in tipologias)
    else:
        hab_total = num_unidades_total * 3

    area_ha = area_terreno_m2 / 10_000.0 if area_terreno_m2 > 0 else 0.0
    densidad_calc = (hab_total / area_ha) if area_ha > 0 else 0.0

    checks = {
        "cus": {
            **_flag(
                cus_calc <= cus_max + 1e-6,
                f"CUS {cus_calc:.2f} ≤ {cus_max:.2f}",
                f"CUS {cus_calc:.2f} excede máximo {cus_max:.2f}",
            ),
            "valor": round(cus_calc, 3),
            "maximo": cus_max,
        },
        "altura": {
            **_flag(
                altura_edificio_m <= h_max + 1e-6,
                f"Altura {altura_edificio_m:.2f}m ≤ {h_max:.1f}m",
                f"Altura {altura_edificio_m:.2f}m excede máximo {h_max:.1f}m",
            ),
            "valor": round(altura_edificio_m, 2),
            "maximo": h_max,
        },
        "area_libre": {
            **_flag(
                al_frac_calc + 1e-6 >= al_min_frac,
                f"Área libre {al_frac_calc * 100:.1f}% ≥ {al_min_frac * 100:.0f}%",
                f"Área libre {al_frac_calc * 100:.1f}% bajo mínimo {al_min_frac * 100:.0f}%",
            ),
            "valor_pct": round(al_frac_calc * 100, 2),
            "minimo_pct": round(al_min_frac * 100, 2),
        },
        "retiro_frontal": {
            **_flag(
                retiro_frontal_m + 1e-6 >= rf_min,
                f"Retiro frontal {retiro_frontal_m:.2f}m ≥ {rf_min:.1f}m",
                f"Retiro frontal {retiro_frontal_m:.2f}m bajo mínimo {rf_min:.1f}m",
            ),
            "valor": round(retiro_frontal_m, 2),
            "minimo": rf_min,
        },
        "retiro_lateral": {
            **(
                _flag(True,
                      "Colindancia con medianera — retiro lateral no exigible",
                      "")
                if not retiro_lateral_aplica else
                _flag(
                    rl_min <= 0 or retiro_lateral_min_m + 1e-6 >= rl_min,
                    f"Retiro lateral {retiro_lateral_min_m:.2f}m ≥ {rl_min:.1f}m",
                    f"Retiro lateral {retiro_lateral_min_m:.2f}m bajo mínimo {rl_min:.1f}m",
                )
            ),
            "valor": round(retiro_lateral_min_m, 2),
            "minimo": rl_min,
            "obligatorio": rl_min > 0 and retiro_lateral_aplica,
        },
        "retiro_posterior": {
            **(
                _flag(True,
                      "Colindancia con medianera — retiro posterior no exigible",
                      "")
                if not retiro_posterior_aplica else
                _flag(
                    retiro_posterior_m + 1e-6 >= rp_min,
                    f"Retiro posterior {retiro_posterior_m:.2f}m ≥ {rp_min:.1f}m",
                    f"Retiro posterior {retiro_posterior_m:.2f}m bajo mínimo {rp_min:.1f}m",
                )
            ),
            "valor": round(retiro_posterior_m, 2),
            "minimo": rp_min,
            "obligatorio": retiro_posterior_aplica,
        },
        "frente_lote": {
            **_flag(
                frente_m + 1e-6 >= frente_min,
                f"Frente {frente_m:.2f}m ≥ {frente_min:.1f}m",
                f"Frente {frente_m:.2f}m bajo mínimo {frente_min:.1f}m",
            ),
            "valor": round(frente_m, 2),
            "minimo": frente_min,
        },
        "area_lote": {
            **_flag(
                area_terreno_m2 + 1e-6 >= area_lote_min,
                f"Lote {area_terreno_m2:.1f}m² ≥ {area_lote_min:.0f}m²",
                f"Lote {area_terreno_m2:.1f}m² bajo mínimo {area_lote_min:.0f}m²",
            ),
            "valor": round(area_terreno_m2, 2),
            "minimo": area_lote_min,
        },
        "densidad": {
            **_flag(
                densidad_calc <= dens_max + 1e-6,
                f"Densidad {densidad_calc:.0f} hab/ha ≤ {dens_max} hab/ha",
                f"Densidad {densidad_calc:.0f} hab/ha excede {dens_max} hab/ha",
            ),
            "valor": round(densidad_calc, 1),
            "maximo": dens_max,
            "habitantes_estimados": hab_total,
        },
    }

    cumple_todo = all(c["ok"] for c in checks.values())

    return {
        "zona": zona_codigo or DEFAULT_ZONA,
        "zona_nombre": zona["nombre"],
        "parametros_zona": zona,
        "parametros_certificado": parametros_certificado,
        "checks": checks,
        "cumple_zonificacion": cumple_todo,
        "incumplimientos": [k for k, v in checks.items() if not v["ok"]],
    }
