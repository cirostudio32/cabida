# Plan de mejora — ArchGEN32 cabida pre-financiera

Objetivo: layout más realista, cumpliendo normativa, optimizado para evaluación financiera previa.

## Hecho (sesiones previas)
- [x] Zonificación parametrizada (`zonificacion.py`): CUS, altura, área libre, retiros, frente, densidad. Validador en endpoint.
- [x] Programa interior por tipología (`programa.py`) — calculado, oculto en cabida pre-financiera (toggle `window.SHOW_INTERIOR_ZONES`).
- [x] Validators arquitectónicos (`validators.py`): ventilación, iluminación, evacuación por dpto.
- [x] Topología selector (`topologia.py`): detector + recomendación (spine | claustro | L_plan | tower).
- [x] Esquema área libre — 3 modos (`muros_ciegos` | `patio_posterior` | `ducto_central`).
- [x] Visual viewer3d: color por tipología, label `TIPO/DPTO/m²`, hall con área, "LÍMITE DE PROPIEDAD" en bordes.

## Pendiente — paquete normativo
- [x] **#1 Retiros laterales aplicados a geometría**. Hoy solo se validan. Reducir strips útiles cuando muro no ciego.
- [x] **#2 Frente mínimo dpto ≥ 3.00m** (RNE A.020) + ratio profundidad/frente ≤ 3.0. Limitar n_dptos por strip.
- [ ] **#4 Núcleo según frente**: detectar dirección frente automáticamente y reorientar núcleo↔patio para liberar fachada vendible. **Bloqueado**: requiere refactor mayor (núcleo lateral en lugar de spine central).

## Pendiente — salto cualitativo
- [x] **#3 Clipping limpio en lotes irregulares**. Convex ratio ≥ 0.85 + adyacencia hall. Descarta L-shapes degenerados post-clip.
- [x] **#5 Densidad mejorada**. Flag `optimizar_densidad` ignora num_departamentos y emite cap_total. Filtro estricto 40m² (no 36).
- [x] **#6 Hall en L / U** para lotes irregulares.
  - [x] Fase A: spine se orienta a ala mayor del L.
  - [x] Fase B: brazo + connector + dptos en ala chica. Lotes L pasan de 2 → 5 dptos viables.
- [~] **#7 Topologías nuevas implementadas**.
  - [x] L_plan completo (con #6 fase B).
  - [x] claustro (patio central + dptos perimetrales) — lotes compactos grandes.
  - [x] tower (núcleo central + esquineros) — lotes cuadrados pequeños.
- [x] **#8 Ductos auto-adyacentes a wet bands** (cocina/baño). Paredes compartidas entre dptos al nivel wet band (t≈0.21). Spine topology.
- [x] **#8b Validators re-feed**: two-pass — generar zonas → auto-ductos → validar. Ductos no duplicados.

## Pendiente — refinamientos
- [x] **#9 Grid estructural columnas c/5–8m**. Grilla 0.40×0.40m a ~5.5m eje dl, filas en bordes hall + exterior. Render viewer3d (gris oscuro sobre dptos).
- [ ] **Estacionamientos** con columnas, giros, plaza accesible, pendiente rampa.
- [x] **Frontend KPIs**: panel zonificación check (verde/rojo), métricas topología recomendada, fallas de validación por dpto.
- [x] **Optimizador de mix tipológico**: dado lote + zona + precios m²/tipo, sugerir mix que maximiza ingreso bruto. Enumeración 1-2 tipos, inputs PEN/m² en sidebar, resultado en normativa tab.

## Convenciones de orientación (importante)
Frontend genera polígono con frente en `y=0`:
- `p1 = (-frente/2, 0)`, `p2 = (frente/2, 0)` → edge frente
- `p3 = (fondo/2, derecha)`, `p4 = (-fondo/2, izquierda)` → laterales y fondo

Rotated rect del lote, asumiendo lote rectangular:
- `ds` (eje corto) = dirección frente↔fondo
- `dl` (eje largo) = dirección laterales (derecha↔izquierda)
- `-ds` típicamente apunta al frente (y mínimo)

Mapeo ciegos → ejes:
- `ciego_frente` ↔ muro -ds
- `ciego_fondo` ↔ muro +ds
- `ciego_derecha` ↔ muro +dl
- `ciego_izquierda` ↔ muro -dl

## Archivos clave
- [main.py](main.py) — endpoint + `_generate_geometry` (lógica spine)
- [zonificacion.py](zonificacion.py) — tabla zonas + validador urbanístico
- [programa.py](programa.py) — distribución interior por tipología
- [validators.py](validators.py) — chequeos arquitectónicos
- [topologia.py](topologia.py) — selector de estrategia
- [renderer.py](renderer.py) — render matplotlib (debug)
- [viewer3d.js](viewer3d.js) — render Three.js (UI)
- [main.js](main.js) — orquestador frontend
- [index.html](index.html) — UI inputs
