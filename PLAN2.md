# PLAN2 — Hoja de Ruta: Motor de Cabida para Decisión de Compraventa

**Objetivo:** Cabida arquitectónica que permita evaluar la viabilidad económica de un terreno
antes de la compra. Salida: cuadro de áreas, mix de tipologías, eficiencia, estimado de ingresos
por ratios de venta (fase futura).

---

## FASE 1 — Nuevas Variables de Input (UI + Backend)

### 1.1 Fachada exterior por lado
**Qué es:** El usuario define qué lados del lote tienen salida a calle o espacio público
(fachada exterior) y cuáles dan a medianería privada (fachada interior).

**Por qué importa:** Las unidades con fachada exterior valen más (vista, ventilación directa,
iluminación). El motor puede asignarles tipologías de mayor área o precio.

**Variables a agregar en frontend:**
- `frente_exterior: bool` (siempre True — da a la calle)
- `fondo_exterior: bool` (True si da a otro frente, False si colinda con predio)
- `derecha_exterior: bool`
- `izquierda_exterior: bool`

**Impacto en motor:** Determina qué lado del strip (+ds / -ds) recibe tipologías premium.
Si `fondo_exterior = True`, las unidades del lado fondo pueden ser más grandes.

**Estado:** PENDIENTE

---

### 1.2 Mix de tipologías objetivo
**Qué es:** El usuario define cuántas unidades por tipología quiere en la planta típica.
Complementa (o reemplaza) el parámetro actual `dptos_planta`.

**Opciones de input:**
- Modo libre: solo ingresa `dptos_planta` (motor decide mix por área disponible)
- Modo dirigido: ingresa `mix = {"1D": 1, "2D": 3, "3D": 2}` — el motor intenta honrarlo

**Impacto en motor:** `distribute_units()` respeta las proporciones del mix.
Si no puede cumplir un tipo, lo escala al tipo más cercano disponible.

**Estado:** PENDIENTE

---

### 1.3 Parámetros municipales variables
**Qué es:** Cada municipalidad de Lima tiene retiros y parámetros distintos.
El usuario ingresa los valores preliminares que conoce del certificado de parámetros.

**Variables a agregar:**
- `retiro_frontal` (ya existe)
- `retiro_lateral` — actualmente hardcoded a 2.30m si no es ciego
- `retiro_posterior` — actualmente hardcoded a 2.30m si no es ciego
- `altura_maxima_pisos` (ya existe como `pisos`)
- `cos_maximo` — coeficiente de ocupación de suelo (área techada / área terreno)
- `cus_maximo` — coeficiente de usos de suelo (área total / área terreno)

**Nota:** No modelamos servidumbres por ahora.

**Estado:** PENDIENTE — `retiro_lateral` y `retiro_posterior` ya tienen lógica pero no son
inputs de usuario; hay que exponerlos en el formulario.

---

## FASE 2 — Mejoras de Lógica de Distribución

### 2.1 Espesor de muros en el cálculo de áreas
**Problema actual:** Las unidades se calculan de eje a eje de muro (sin descontar espesor).
Una cabida real trabaja en metros libres interiores.

**Solución:**
- Muro medianero entre unidades: 0.15m → descontar 0.075m de cada lado
- Muro perimetral (fachada): 0.15–0.20m → descontar del frente/fondo
- En la distribución: reducir `depth_eff` y el ancho de cada unidad en la mitad del muro

**Impacto:** Áreas netas bajarán ~3–5% respecto al valor actual (más realistas).

**Estado:** PENDIENTE

---

### 2.2 Pozos de luz normativos para dormitorios
**Problema actual:** Solo se generan ductos 0.50×0.50m para zonas húmedas.
Los dormitorios requieren ventilación a fachada exterior o a pozo de luz (d ≥ H/4, min 2.20m).

**Solución:**
- Unidades con dormitorios que NO dan a fachada exterior: requieren pozo de luz
- Pozo de luz: cuadrado de lado = max(H/4, 2.20m), compartido entre unidades adyacentes
- El pozo se coloca en el muro posterior (lado fondo), en el eje medianero entre dos unidades
- Validar que cada dormitorio "ve" un pozo o fachada

**Estado:** PENDIENTE

---

### 2.3 Distancia máxima a escalera (RNE A.010)
**Problema actual:** No se valida que todos los dptos estén a ≤25m de la escalera.

**Solución:** Post-generación, calcular la distancia desde el centro de cada unidad
al núcleo de escalera. Marcar las unidades que superan 25m como no conformes.
Si hay unidades lejanas en lotes largos (>50m), considerar doble núcleo.

**Estado:** PENDIENTE

---

### 2.4 Coherencia de unidades en lotes irregulares
**Problema actual:** En lotes trapezoidales, las unidades de esquina se recortan y quedan
con tipologías menores (ej. X07=53m², X08=89m²), generando incoherencia visual.

**Solución:**
- Si una unidad clipeada tiene área < 80% del área teórica calculada (seg_len × depth_eff),
  marcarla como "reducida" y ajustar tipología automáticamente.
- Opción: fusionar la unidad reducida con la adyacente si su frente es <3.5m.

**Estado:** PENDIENTE

---

## FASE 3 — Plantas Diferenciadas

### 3.1 Planta baja real
**Problema actual:** La planta baja es una réplica de la planta típica.

**Contenido correcto de planta baja:**
- Lobby de acceso + hall de distribución
- Cuarto de basura (mínimo 3m² por RNE)
- Cuarto de limpieza
- Cuarto de tableros eléctricos
- Rampa vehicular (si hay sótano)
- Comercio en fachada (opcional, según zonificación)
- Sin unidades residenciales si el retiro lo impide

**Estado:** PENDIENTE — existe código básico de primer piso pero no modela estos espacios

---

### 3.2 Planta azotea / nivel técnico
**Contenido:**
- Tanque elevado (capacidad = dotación diaria × 1/3)
- Cuarto de máquinas de ascensor
- Área de expansión de escalera (caja continua)
- Área libre techada (lavandería comunal, opcional)

**Estado:** PENDIENTE

---

### 3.3 Grilla estructural
**Por qué importa:** Sin columnas no hay cabida constructiva real.
Las áreas de departamentos deben respetar los ejes de columna.

**Solución:**
- Grilla ortogonal cada 5.00–6.00m sobre el eje dl
- Columnas 0.50×0.50m en intersecciones
- Los límites de unidades se alinean a la grilla (no a ancho libre arbitrario)

**Estado:** PENDIENTE — columnas existen en el payload pero no se generan automáticamente

---

## FASE 4 — Cuadro de Áreas y Viabilidad Económica

### 4.1 Cuadro de áreas completo
**Salida requerida:**
| Elemento | Área (m²) |
|---|---|
| Área de terreno | |
| Área libre (RNE) | |
| Área techada por piso | |
| Área techada total | |
| Área vendible (dptos) | |
| Área común (hall + núcleo) | |
| Eficiencia (vendible/techada) | |
| COS real | |
| CUS real | |

**Estado:** PARCIAL — algunos datos existen en el payload pero no se muestran en tabla completa

---

### 4.2 Cuadro de unidades
**Salida requerida por cada unidad:**
| ID | Tipología | Área (m²) | Lado | Fachada | Colinda hall |
|---|---|---|---|---|---|

**Estado:** PARCIAL — existe en el panel lateral pero incompleto

---

### 4.3 Estimado de ingresos (FASE FUTURA)
- Precio por m² según tipología y orientación (usuario ingresa ratio $/m²)
- Ingreso bruto estimado = Σ (área_i × precio_i)
- Costo estimado de obra = área techada total × ratio_construcción (usuario ingresa)
- Margen bruto = ingreso − costo − terreno
- VAN preliminar (sin financiamiento)

**Estado:** NO IMPLEMENTADO — requiere nuevo módulo de viabilidad económica

---

## ORDEN DE IMPLEMENTACIÓN PROPUESTO

| Prioridad | Tarea | Fase | Complejidad |
|---|---|---|---|
| 1 | Retiro lateral/posterior como input de usuario | 1.3 | Baja |
| 2 | Fachada exterior por lado | 1.1 | Baja |
| 3 | Mix de tipologías objetivo | 1.2 | Media |
| 4 | Espesor de muros en áreas | 2.1 | Media |
| 5 | Coherencia en lotes irregulares | 2.4 | Media |
| 6 | Validación distancia a escalera | 2.3 | Baja |
| 7 | Pozos de luz para dormitorios | 2.2 | Alta |
| 8 | Cuadro de áreas completo | 4.1 | Media |
| 9 | Cuadro de unidades completo | 4.2 | Baja |
| 10 | Planta baja real | 3.1 | Alta |
| 11 | Grilla estructural | 3.3 | Alta |
| 12 | Planta azotea | 3.2 | Media |
| 13 | Estimado de ingresos | 4.3 | Media |

---

## ESTADO ACTUAL DEL MOTOR (línea base)

- Topología: spine (doble crujía) ✅
- Hall 1.20m RNE ✅
- Núcleo (escalera + ascensor + vestíbulo) ✅
- Ductos húmedos 0.50m ✅ (fix z-level pendiente de verificar)
- Tipología por área ✅
- RNE pozo de luz mínimo ✅
- Eficiencia básica ✅
- Lote irregular (trapezoidal) ✅ parcial
- Planta baja ✅ básica
- Sótanos y estacionamientos ✅ básico
- Muros con espesor ❌
- Mix de tipologías dirigido ❌
- Fachada exterior variable ❌
- Parámetros municipales como input ❌
- Pozos de luz para dormitorios ❌
- Grilla estructural automática ❌
- Cuadro de áreas completo ❌
- Viabilidad económica ❌
