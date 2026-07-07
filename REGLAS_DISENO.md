# Reglas de diseño — extraídas de referencias (planos reales, Lima)

## CALIBRACIÓN DURA (medida de los DXF referencias/1-6.dxf con ezdxf+shapely)
- Profundidad (crujía) de dpto: **7.7-8.2m constante** en todos los proyectos.
- Anchos tipológicos de fachada: **1D 5.2 / 1D+E 6.25 / 2D 6.75-6.97 / 2D+E 8.35-8.57m**.
- Áreas reales: 1D 40.2-40.7 / 1D+E 45.4-51.4 / 2D 49.3-54.2 / 2D+E 59.4-67.3 m².
- Corredor/pasadizo: **1.80m** de ancho.
- Hall común por planta: 50-84 m² (incluye corredor).
- Ductos entre wet-cores: 3.35×0.60 y 3.15×0.40 m (chicos, pareados).
- Pozos: al MÍNIMO normativo; área libre sale de retiros + patio posterior.
- Lotes medidos: 15×31 (2 torres dibujadas), 17×31 aprox/torre, 12.5-13×30.5
  (esquinero doble), 25×23.5 esquinero.


Catálogo de patrones que el motor de cabida debe respetar. Fuente: planos reales
de edificios multifamiliares en Lima (Av. Campodónico, Av. Brasil, Mariscal
Cáceres, Fco. de Zela, Alm. Guisse, Simón Bolívar, Villarán, Max Gonzales,
Santa Rosa, San Martín, Pardo de Zela).

## P1 — Posición del núcleo (escalera + ascensores SIEMPRE contiguos)
| Lote | Núcleo | Ref |
|---|---|---|
| Angosto <14m | Embebido al centro entre dos columnas de dptos, o a medianera | 8, 4 |
| Medio 14-24m entre medianeras | Centrado en ancho, a media profundidad | 1, 9 |
| Esquinero | Hacia medianera interior; la esquina-calle SIEMPRE para dptos premium | 5, 6, 11, 12 |
| Ancho >24m o fondo >38m | DOS núcleos / dos torres con patio central y junta constructiva | 2, 3, 4, 7 |
| Ancho poco profundo | Fila única a calle + cabecera al fondo, núcleo central | 12 |

## P2 — Hall de distribución
- 30-67 m² para 6-9 dptos por núcleo (≈ 5-7 m²/dpto servido).
- Sin corredores >15m. Pasadizo 1.2-1.5m solo donde imprescindible.
- Estado motor: hall 2.2m + corredor 1.5m + hall fondo. ✔ implementado

## P3 — Capacidad por núcleo
- 6-10 dptos/planta por núcleo. Pedidos mayores → segundo núcleo (pendiente).

## P4 — Pozos de luz (ref 9 con cotas, lote 24×35)
- FRANJAS LATERALES CONTINUAS en medianeras a lo largo de la zona media,
  no pozos chicos dispersos. Ref 9: 2 franjas ~3×16m; rotula
  "área pozo normativo 47.20 m² / en proyecto 96.87 m²" (≈2× norma).
- Unidades intermedias rematan en la franja (dentado/muescas para ventanas).
- Representación: vacío blanco humo + X de arista a arista. ✔ implementado
- Estado motor: franja derecha continua cuando hay filas intermedias,
  con muescas (solape 1.2m) sobre bloques frente/fondo. ✔ implementado

## P5 — Mix por ubicación
- 3D / 2D+E → esquinas y fachada a calle.
- 1D / 1D+E → interiores junto al núcleo.
- 2D → fondo y laterales.
- Áreas reales Lima: 1D 37-44 · 1D+E 41-57 · 2D 46-56 · 2D+E 54-70 · 3D 56-73.
  Casi ninguna unidad >75 m².

## P6 — Proporciones de unidad
- Profundidad 7-10m, frente 5-8m. Unidades de borde junto a pozo compensan
  mordida con ancho extra (áreas finales parejas). ✔ implementado

## P7 — Wet cores espalda con espalda
- Cocinas/baños contra el corredor/núcleo o muro común entre vecinos
  (comparten ducto); fachadas y pozos libres para dormitorios/salas.
- Estado motor: bandas wet junto a puerta (programa.py). Ducto común explícito
  entre vecinos: pendiente.

## P8 — Ductos / montantes (IS.010)
- Ductos de ventilación se dimensionan por aparatos servidos: pisos altos
  sirven menos → sección puede reducirse al subir (telescópico). Pendiente
  (requiere geometría por piso).

## Pendientes priorizados
1. Dos núcleos / dos torres para lotes anchos o muy profundos (P1, P3).
2. Split del dpto intermedio único cuando sale >75 m² (P5/P6).
3. Ducto común wet-core entre vecinos (P7).
4. Ductos telescópicos por piso (P8).
5. Esquineros: núcleo a medianera interior + premium a esquina (P1).
