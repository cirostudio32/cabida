# Plan de reconstrucción del motor de cabida — "diseñar como arquitecto"

Decisión confirmada: **% Área Libre Mín = solo validación** (check rojo si no
cumple; nunca recorta/mutila el diseño).

Regla general de trabajo: ninguna etapa se da por cerrada sin pasar su
**compuerta de verificación** (tests automáticos + render visual). Cada etapa
termina con commit lógico y reporte al usuario con evidencia (PNG + tabla).

---

## ETAPA 0 — Arnés de pruebas y medición
**Objetivo:** poder afirmar con números si un cambio mejora o rompe.

**Acciones:**
1. Crear `tests_cabida.py` (standalone, corre con `python tests_cabida.py`):
   - Suite de lotes sintéticos (12): rectangulares 10×25, 13×28, 17×34, 20×30,
     24×32; trapezoidales (frente < fondo y frente > fondo, ±3m de sesgo);
     profundo 15×45; ancho 30×20; cuadrado 18×18; irregular 5 vértices;
     el lote real del usuario (24 / 32.1 / 29.3 / 32.2).
   - Cada lote × nd ∈ {2, 4, 6, 8, 10} × retiros {0, 2, 3} (muestra, ~40 casos).
2. Por caso, medir del payload (no del código):
   - `retiro_real_por_borde`: distancia mínima edificio→cada lindero (shapely).
   - `pct_circulacion` = (hall+corredores) / techada.
   - `eficiencia` = vendible / techada.
   - `dptos_emitidos` vs pedidos; áreas min/max/promedio; tipologías.
   - `area_pozos` vs mínimo normativo (ratio).
   - `huecos`: área del lote edificable no asignada a nada (debe ser ~0).
   - `acceso`: cada dpto con frente de puerta ≥1.20m de contacto real al hall.
   - `paralelismo`: ángulo entre el borde fondo del edificio y el lindero
     posterior (debe ser ≈0°).
3. Salida: tabla resumen + PNG por caso en `_tests_out/` + contador
   PASS/FAIL por criterio.

**Compuerta E0:** el arnés corre completo y reporta; los FAIL actuales quedan
documentados como línea base (se espera que muchos fallen — eso es lo que
vamos a levantar).

---

## ETAPA 1 — Envolvente edificable exacta (retiros)
**Objetivo:** retiro N ⇒ el edificio queda EXACTAMENTE a N de ese lindero;
retiro 0 ⇒ toca el lindero. El % área libre deja de recortar profundidad.

**Acciones:**
1. `_erode_lote` pasa a ser LA única fuente de envolvente para todas las
   topologías (costillas, hall_compacto, spine, tower, claustro): se aplica
   antes del dispatch en `_generate_geometry`, no dentro de cada generador.
2. Eliminar todo recorte de `D_use` por `% área libre` (queda solo el check
   en zonificación). Eliminar lógicas duplicadas de retiro por bbox.
3. Clasificación de bordes robusta: frente = borde más cercano/paralelo al
   eje del frente declarado (no "el horizontal de menor y"); laterales y
   fondo por adyacencia recorriendo el polígono desde el frente. Funciona
   con lotes rotados o con frente arriba.

**Compuerta E1 (automática, en el arnés):**
- Para TODOS los casos: |retiro_real − retiro_pedido| ≤ 0.05m en cada borde.
- Caso retiro 0: distancia edificio-lindero = 0 (±0.05).
- % área libre alto (40) NO cambia la geometría vs % libre 0 (misma huella);
  solo cambia el check.
**Compuerta visual:** PNG del lote del usuario con retiro posterior 0 y 3 —
el borde del edificio pegado / a 3m del lindero inclinado.

---

## ETAPA 2 — Marco orientado al lote (paralelismo y cortes reales)
**Objetivo:** crujías y bloques paralelos al frente y fondo REALES; unidades
de borde absorben linderos inclinados; cero triángulos muertos.

**Acciones:**
1. Marco local (u,v): u = dirección del frente del lote; v = perpendicular
   hacia el fondo. Toda la generación pasa a (u,v) y se transforma al final
   (matriz de rotación), en vez de asumir ejes X/Y.
2. Cortes de bloques: el corte del bloque fondo se hace PARALELO al lindero
   posterior erosionado (no horizontal): se construye con offset del borde
   real. Frente igual con el frente real.
3. Las unidades extremas toman el polígono resultante (muro perimetral
   inclinado); las divisiones interiores siguen rectas en (u,v).
4. Ancho de cálculo por profundidad: en cada corte v se mide el ancho real
   del polígono (línea de intersección), no el bbox ni el promedio.

**Compuerta E2:**
- `paralelismo` fondo-edificio vs lindero ≤ 2° en todos los casos.
- `huecos` ≤ 2% del área edificable en lotes trapezoidales.
- Áreas de unidades de borde dentro de ±15% de sus vecinas (absorción
  razonable, no unidades deformes).
**Visual:** trapezoide del usuario nd=4 y nd=8: bloque fondo siguiendo la
inclinación del lindero.

---

## ETAPA 3 — Núcleo y circulación con presupuesto duro
**Objetivo:** hall con sentido arquitectónico; nunca más "banda gris".

**Acciones:**
1. Presupuesto de circulación: `area_circ_max = max(14 m², 3.0 × nd)` para
   hall+corredor (núcleo vertical aparte). Tope global: circulación ≤12%
   del techado. El generador DEBE cumplirlo o iterar.
2. Hall = rectángulo compacto frente a escalera+ascensores (lado mayor =
   frente de los accesos). Corredor: solo si algún dpto no alcanza el hall;
   ancho 1.20-1.80; largo mínimo necesario (hasta la puerta más lejana).
3. Posición del núcleo: candidatos (centro geométrico, arrimado a medianera
   izq/der, a 1/3 del fondo) → se elige el que minimiza largo de corredor
   con todos los dptos servidos. Determinista y general.
4. Eliminar la heurística "hall absorbe remanentes". Remanentes van a
   dptos (etapa 4); si un residuo no es asignable (<9m² o sin fachada),
   se convierte en ducto/depósito, no en hall.

**Compuerta E3:**
- `pct_circulacion` ≤ 12% en todos los casos; hall ≤ max(14, 3×nd)+20%.
- `acceso`: 100% de dptos con contacto de puerta ≥1.20m al hall/corredor.
- Evacuación ≤30m se mantiene.
**Visual:** caso del usuario nd=4: hall ~14-16m² en cruz/T pegado al núcleo,
nada de banda.

---

## ETAPA 4 — Subdivisión total del área en dptos
**Objetivo:** el 100% del área edificable − núcleo − circulación termina en
departamentos (o ducto/pozo justificado). Cero vacíos sin nombre.

**Acciones:**
1. Particionador: sobre el polígono restante, cortes paralelos al frente
   (crujías de 7-9m) y cortes perpendiculares por anchos tipológicos
   (1D 5.2 / 1D+E 6.25 / 2D 6.9 / 2D+E 8.35 / 3D 9.0 — calibrado DXF),
   eligiendo la combinación que (a) emite exactamente nd si cabe,
   (b) maximiza área vendible, (c) áreas dentro de rangos tipológicos.
2. Si nd < capacidad: los dptos crecen (hasta ~110-130m²) absorbiendo el
   área — nunca generar vacíos para "respetar" áreas chicas.
3. Si nd > capacidad: emitir capacidad y reportar faltante (ya existe).
4. Pozos: pasada final — para cada dpto, si su banda de dormitorios no toca
   fachada/retiro/patio: insertar pozo mínimo normativo (H/4 × H/4) adosado
   a medianera, compartido entre vecinos cuando sea posible, restándolo
   con compensación de área. Ductos wet 0.6×3.3 pareados entre cocinas/baños
   espalda-espalda (patrón DXF).
5. Mix: respetar `mix_tipologias` si el usuario lo define (pendiente actual).

**Compuerta E4:**
- `huecos` ≈ 0 (≤1% del edificable) en TODOS los casos.
- `eficiencia` ≥ 78% en lotes ≥300m².
- `area_pozos` ≤ 1.3× el mínimo normativo total.
- dptos_emitidos == nd cuando capacidad lo permite; áreas dentro del rango
  de su tipología (±10%).
- 100% dormitorios con ventilación (fachada, patio o pozo conforme).

---

## ETAPA 5 — Bucle de auto-crítica (el "arquitecto")
**Objetivo:** el motor evalúa su propio diseño y lo corrige antes de emitir.

**Acciones:**
1. Función `evaluar_planta(payload) → score + lista de defectos` usando las
   mismas métricas del arnés (única fuente de verdad).
2. Loop generar→evaluar→ajustar (≤4 iteraciones): defecto dirige el ajuste
   (corredor largo → mover núcleo; dpto sin ventilación → pozo; circulación
   alta → fusionar hall; hueco → reasignar a dpto vecino).
3. El payload expone `diseño.score` y `diseño.defectos` para que el usuario
   vea qué auto-correcciones hizo y qué quedó imperfecto.

**Compuerta E5:** sobre la suite completa: score promedio ≥85/100; ningún
caso con defecto "crítico" (retiro violado, dpto sin acceso, hueco >1%).

---

## ETAPA 6 — Tests dorados contra los DXF reales
**Objetivo:** validar contra realidad construida, no contra mi criterio.

**Acciones:**
1. Por cada DXF (1-6): extraer lote, nd real, áreas, % circulación
   (script ya probado con ezdxf).
2. Generar cabida sobre ese mismo lote con los mismos pisos y nd.
3. Comparar: nd ±1; área promedio ±10%; % circulación ±3 puntos;
   crujía ±0.8m.

**Compuerta E6:** ≥4 de 6 DXF dentro de tolerancia (los esquineros pueden
quedar fuera hasta la etapa de esquineros).

---

## Orden y dependencias
E0 → E1 → E2 → E3 → E4 → E5 → E6.
E1+E2 corrigen retiros/paralelismo (captura del usuario).
E3 corrige el hall. E4 corrige distribución/pozos/vacíos.
Cada etapa: implementar → correr arnés → mostrar tabla + PNGs al usuario →
OK del usuario → siguiente.

## Fuera de alcance de este plan (cola posterior)
Esquineros L con doble frente; 2 núcleos/torres en lotes >28m; ductos
telescópicos IS.010; estacionamientos por ratio; primer piso conectado.
