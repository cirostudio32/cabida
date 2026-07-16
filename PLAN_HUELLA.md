# PLAN — Rework de generación de huella (tamaño, forma, núcleo/hall)

**Fecha:** 2026-07-15 · **Alcance:** `_generate_costillas` (main.py:1238+) — la topología dominante
(lotes rectangulares y trapezoidales entre medianeras). Extensible después a hall_compacto/dos_nucleos.

---

## 1. DIAGNÓSTICO — por qué salen dptos de 35–44 m²

Auditoría del código actual, con línea exacta:

| # | Defecto | Ubicación | Efecto observado |
|---|---------|-----------|------------------|
| D1 | Filtro de emisión acepta unidades con **área BRUTA ≥ 0.85×40 = 34 m²** (`ap.area < min_area_dpto * 0.85 → continue`). Tras descuento de muros (`area_neta_muros`, ~-8%), la neta cae a **31–38 m²** y se emite igual (solo se marca `es_reducida`). | main.py:1704 | Los dptos de 35 m² que ves. La unidad nace del clip del borde inclinado / resta de pozo-núcleo-hall, queda chica, y el motor la publica en vez de re-balancear. |
| D2 | `_split_block` parte el bloque frente/fondo **uniforme en X** (`step = (xb-xa)/n`), sin considerar que el clip al borde inclinado deja áreas desiguales. | main.py:1598-1608 | En trapezoide: fondo 68/84/97 m² (el corte igual en X da áreas distintas tras el clip). Sin re-balanceo post-clip. |
| D3 | `_shrink_for_slope` chequea área *estimada* (d×step) **antes** del clip real y solo reduce n; nunca **mueve** los cortes. | main.py:1482-1498 | Deja pasar configuraciones donde una unidad clipada queda chica (interactúa con D1). |
| D4 | **Núcleo partido**: escalera pegada a medianera izq (`ucl-ESC_W`) + ascensores a medianera der (`ucr+ASC_W`), enfrentados a través del corredor. | main.py:1539-1556 | Mochetas ilógicas. **Ningún** DXF real lo hace: núcleo SIEMPRE es un solo paquete — lateral (DXF1/3/4, lotes ≤30 m) o central (DXF2/6, lotes ≥34 m). Además roba frente tipológico en AMBAS medianeras. |
| D5 | Vestíbulo previo se emite siempre que `nec_esc_prot` (1.5×1.5). | main.py:1543-1548 | RNE A.010 art. 26: escalera **presurizada NO requiere vestíbulo previo** — m² regalados en edificios >5 pisos (el caso típico, 7 pisos). |
| D6 | Hall = corredor 1.80 de punta a punta de la zona media + ensanches 0.70 en ambos lados. | main.py:1564-1595 | Sobre-circulación cuando hay pocas filas intermedias; el corredor no se recorta a la última puerta servida. |
| D7 | Franjas de pozo `fz` **continuas** (2.2 m × todo el largo de la zona media) en AMBAS medianeras. | main.py:1276-1292, 1649-1663 | Real: ductos compactos 1.8×5.6 (~10 m²) en pares frente/fondo pegados a UNA medianera. La franja continua resta ancho a las columnas de dptos en toda la profundidad. |
| D8 | La búsqueda optimiza `tot` (nº unidades) pero **no** hay término de equilibrio de áreas ni banda objetivo de tamaño. | main.py:1344-1405 | Mezclas 44+97 m² en la misma planta: el motor "cumple" el conteo con unidades desbalanceadas. |

**Regla de oro que falta:** el motor jamás debe emitir una unidad fuera de banda; si no cabe, debe
**fusionar** (n−1 unidades más grandes) o **reportar capacidad real menor**, nunca publicar la aberración.

---

## 2. BASE NORMATIVA (RNE vigente)

- **A.020 Vivienda (RM 188-2021):** dpto multifamiliar mín **40 m²** (16 m² solo "vivienda de uso
  colectivo", no aplica aquí). Ascensor obligatorio cuando el acceso al 5º piso está a >12 m del ingreso.
  Muros entre viviendas y hacia circulación común: RF 60 min (espesor ya modelado, 0.15).
- **A.010 (RM 191-2021):** residencial >15 m de altura puede tener **una sola escalera de evacuación**
  cumpliendo requisitos (detección de humo, alarma) — valida el esquema de núcleo único. Escalera
  **presurizada: sin vestíbulo previo** (elimina D5). Recorrido de evacuación máx **45 m sin rociadores /
  60 m con** (el motor usa 30 m conservador — mantener como default configurable, es más estricto que norma).
  Pozos de luz: lado ≥ H/3 (ya implementado, mantener honestidad de `pozos_luz_cumple`).
- **A.130:** ascensores NO son medio de evacuación; ancho de escape ≥1.20 (banda distribuidora ya usa 1.20).

## 3. BASE EMPÍRICA (referencias DXF + antecedentes)

De `huellas_ref.json` (5 proyectos construidos):

- **Tipologías reales: 39.6–84.2 m², mediana ≈ 52 m².** Banda sana: **[40, 88] neta**, objetivo 45–75.
- **Núcleo:** lateral si lote angosto (W ≤ ~30 m: DXF1 x_rel 0.94, DXF3 0.12, DXF4 0.22), central si
  ancho (W ≥ ~34 m: DXF2 0.39, DXF6 0.48). **Nunca partido en dos medianeras.**
- **Ductos:** 1.8×5.6 ≈ 10 m², en **pares frente/fondo de la zona media**, pegados a medianera
  (d_medianera 0.09–0.24). No franjas continuas.
- **Huella:** rectangularidad 0.19–0.46 → forma peine/L/U, con vacíos discretos; no barra llena.
- Antecedentes Lima (GGR Golf, Vértice Point): entre medianeras se **minimiza circulación común** y se
  hace ventilar sala/dorm a fachada o patio, wet-cores a ductos.

---

## 4. PLAN DE IMPLEMENTACIÓN (6 fases, cada una committeable y medible)

**Gate de cada fase (obligatorio antes de commit):**
`python _scripts/contrastar_motor.py` (baseline: 5 desviaciones, sin dpto >1.2×real_max nuevo) ·
`python tests_cabida.py` (≥203/205) · `tests_e6_golden.py` (≥5/6) · `tests_e7_payload_paridad.py` (PASS) ·
payload real trapezoidal vía **HTTP** (server sin `--reload`, reiniciar tras cada edit) ·
render visual de 20×30, 25×28 trapezoidal, 15×25 angosto.

### F1 — Gate duro de tamaño + fusión (mata los 35–44 m²) — el más urgente
1. Nueva constante `AREA_MIN_NETA = 40.0` (RNE) y helper `_neta_ok(ap) = area_neta_muros(ap) >= 40`.
2. En la emisión (main.py:1689-1736): construir TODAS las unidades del bloque primero (post-clip,
   post-restas), y si alguna queda con neta <40 → **fusionar con la vecina** del mismo bloque
   (re-split a n−1) y reintentar; solo si n=1 sigue bajo mín → descartar y descontar del conteo.
3. Reemplazar el filtro `0.85×min` bruto por neta ≥ 40 estricta. `es_reducida` deja de existir como
   estado publicable: o cumple o se fusiona.
4. Reporte honesto: si el lote solo da 5 sanas y pidieron 6 → `advertencias: ["capacidad real: 5
   unidades ≥40m²; la 6ª violaría RNE A.020"]` (campo ya existe en payload).
   - *Riesgo:* baja el conteo en lotes tight → puede tocar tests golden. Mitigación: fusión antes que descarte.

### F2 — Split equi-área en bloques (forma correcta en trapezoide)
1. Reescribir `_split_block(n)` → `_split_block_equiarea(n, y0, y1)`: clipar el bloque completo al
   lote, y buscar cortes x₁..xₙ₋₁ tales que las n piezas clipadas tengan **área igual** (bisección
   sobre área acumulada; shapely `clip_by_rect` en 20-30 iteraciones, costo trivial).
2. Constraint de forma: cada pieza con `min_side ≥ 5.2` (frente tipológico); si un corte lo viola,
   mover el corte al mínimo 5.2 y compensar en las demás (áreas casi-iguales, forma válida).
3. `_shrink_for_slope` pasa a operar sobre las áreas **clipadas reales** (ya no estimación d×step).
   - *Resultado esperado en tu lote:* fondo 83/83/83 en vez de 68/84/97.

### F3 — Núcleo único compacto (lateral/central por criterio DXF) — el estructural grande
1. Criterio de posición (de huellas_ref): `W_util < 32 → 'lateral'` (pegado a la medianera del lado
   con MENOS frente aprovechable — o a la del muro ciego más largo); `W_util ≥ 32 → 'central'`.
2. **Lateral:** paquete único `escalera(2.5×5.6) + ascensores(2.0×asc_l) adosados en L o en línea`,
   pegado a una medianera al arranque de la zona media; hall-nodo (≈1.8×3.0) frente al paquete
   conectado al corredor. La medianera opuesta queda LIBRE → +2.5 m de frente tipológico para filas.
3. **Central:** paquete único en el eje del corredor (como DXF2/6), corredor rodea un solo lado.
4. Cambios concretos: `nuc_l_h`/`nuc_r_h` → un solo `nuc_h`; `Dm = max(nuc_h, filas×h_fila)`;
   reparto `filas_l/filas_r` asimétrico (el lado del núcleo pierde el tramo del paquete, el opuesto
   gana una fila); emisión `stair_poly/asc_polys` contiguos; ensanche de hall solo en el lado del núcleo.
5. Eliminar `vest_poly` cuando la escalera es presurizada (`nec_esc_prot` y h_edif >5 pisos → A.010
   art. 26 no exige vestíbulo); mantenerlo solo para escalera protegida no presurizada.
   - *Es la fase más invasiva:* tocar búsqueda + emisión + hall. Hacerla en 2 commits (lateral primero,
     central después). Medir mochetas: contar vértices cóncavos del footprint (debe bajar).

### F4 — Hall mínimo (recorte a lo servido)
1. Corredor termina en la **última puerta** que sirve (trim ya existente en `_generate_hall_compacto`
   — portar la lógica), no de yf0 a yb0 fijo.
2. Ancho: 1.50 m si sirve ≤4 unidades por tramo, 1.80 si más (A.010 escape ≥1.20 + holgura).
3. Métrica de gate: `pct_circ` (ya existe en tests) no debe subir; objetivo <12% de la huella.

### F5 — Ductos compactos en vez de franjas continuas
1. Sustituir franjas `fz` continuas por **pares de ductos 1.8×5.6** pegados a la medianera del lado
   con filas intermedias, en las posiciones frente/fondo de la zona media (patrón DXF exacto).
2. El ancho liberado (2.2 − 0 en tramos sin ducto) vuelve a `col_w` → filas intermedias más profundas
   → más área por unidad (empuja hacia la banda 45–75).
3. **Honestidad RNE intacta:** los ductos ventilan cocinas/baños; si una fila intermedia tiene
   dormitorios sin fachada, se mantiene el pozo normativo (lado ≥ H/3) SOLO frente a ese tramo y
   `pozos_luz_cumple` sigue reportando no-conformidad cuando el lote no da. No se maquilla.
4. Requiere F3 primero (la posición de ductos depende del lado del núcleo).

### F6 — Gate de calidad de salida (candado final, anti-regresión permanente)
1. Función `_validar_planta(geometry)` que corre SIEMPRE antes de devolver: toda unidad neta ∈
   [40, 98.6], `min_side ≥ 4.8`, acceso a hall verificado, evac ≤ límite, sin solapes unidad-unidad
   ni unidad-núcleo (>0.05 m²).
2. Si falla → reintenta con `num_dptos−1` (hasta −2); si aún falla → error explícito con motivo, no
   planta rota.
3. Añadir a `contrastar_motor.py` un check de **equilibrio**: `max_area/min_area ≤ 1.6` por planta
   (las refs cumplen: 84/40=2.1 global pero por bloque ≤1.6) + check mochetas (vértices cóncavos).
4. Test nuevo `tests_trapezoide.py`: 4 lotes sintéticos (recto, cuña suave 5°, cuña fuerte 12°,
   esquina cortada) × 3 conteos — todas las unidades en banda, fondo a dist=0 de la línea real.

### Orden y dependencias
```
F1 (tamaño/fusión)  ──►  F2 (equi-área)  ──►  F3 (núcleo único)  ──►  F4 (hall)  ──►  F5 (ductos)
                                                      │
F6 (gate salida) — se implementa junto a F1 y se endurece en cada fase
```
F1+F2 eliminan los 35–44 m² (tu bloqueo actual). F3–F5 son el rework estructural ya acordado
(mochetas, franjas, simetría). Cada fase = 1-2 commits, gates completos, verificación HTTP en vivo.

### Riesgos conocidos
- `_generate_costillas` es la función más calibrada: cambios de búsqueda mueven TODOS los renders.
  Por eso el orden va de menos invasivo (filtro de emisión) a más (núcleo).
- F3 puede romper `tests_e6_golden` (esperan geometría actual) → actualizar goldens conscientemente,
  con render antes/después lado a lado para aprobación visual.
- DXF3 (13 m) sigue rechazado por `W < 13.0` (main.py:1266): F3-lateral podría habilitarlo (núcleo en
  una sola medianera cabe en 13 m). Marcar como stretch-goal de F3, no bloqueante.
