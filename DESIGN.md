---
name: ArchGEN32
description: Calculadora de cabida inmobiliaria pre-financiera para arquitectos e inversionistas.
colors:
  navy-accent: "#232741"
  navy-accent-deep: "#14172a"
  navy-soft: "rgba(35, 39, 65, 0.07)"
  ink-navy: "#14172a"
  neutral-bg: "#f5f6f8"
  surface: "#ffffff"
  text-primary: "#1d2033"
  text-secondary: "#868ba0"
  border: "#ececf1"
  border-strong: "#dcdde5"
  legend-1d: "#fef08a"
  legend-1d-e: "#fed7aa"
  legend-2d: "#d9f99d"
  legend-2d-e: "#bae6fd"
  legend-3d: "#e9d5ff"
  legend-circulacion: "#e5e7eb"
typography:
  title:
    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
    fontSize: "1.3rem"
    fontWeight: 700
    letterSpacing: "-0.01em"
  heading:
    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 700
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
    fontSize: "0.85rem"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
    fontSize: "0.68rem"
    fontWeight: 700
    letterSpacing: "0.04em"
rounded:
  sm: "9px"
  md: "12px"
  lg: "16px"
  pill: "999px"
spacing:
  xs: "0.35rem"
  sm: "0.55rem"
  md: "0.85rem"
  lg: "1.1rem"
components:
  button-primary:
    backgroundColor: "{colors.navy-accent}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    padding: "0.7rem 1.1rem"
  button-primary-hover:
    backgroundColor: "{colors.navy-accent-deep}"
  button-tool:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.sm}"
    padding: "0.45rem 0.75rem"
  button-tool-hover:
    backgroundColor: "{colors.navy-soft}"
    textColor: "{colors.navy-accent-deep}"
  kpi-card:
    backgroundColor: "#fbfbfe"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "0.7rem 0.85rem"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    padding: "0.55rem 0.65rem"
---

# Design System: ArchGEN32

## 1. Overview

**Creative North Star: "El Salón de Inversión"**

ArchGEN32 se siente como el salón privado donde se decide si un terreno vale la inversión: sobrio, financiero, boutique. Es un tema **claro y aireado**: superficies blancas sobre un fondo gris-frío muy claro (`#f5f6f8`), con **navy near-black (`#232741`) como único acento de acción** — toggles activos, botón primario, foco, tabs. El navbar es blanco con borde fino inferior (no oscuro): el aire y el whitespace hacen el trabajo que antes hacía la decoración. Tipografía Inter sin adornos.

Este sistema rechaza explícitamente la grilla cruda de una hoja de cálculo y el gris industrial de un software CAD — ver anti-referencias en PRODUCT.md. Los datos (áreas, % eficiencia, cumplimiento normativo) siempre llevan más peso visual que cualquier decoración.

**Key Characteristics:**
- Acento monocromo navy near-black (`#232741`) que carga la autoridad visual; casi todo lo demás es tinta y neutros fríos.
- Fondo gris-frío claro (`#f5f6f8`) y tarjetas blancas puras (`#ffffff`) — contraste suave, aireado.
- Bordes finos (1–1.5px, `#ececf1`) + sombras muy suaves; separación por aire, no por peso.
- Radios generosos y consistentes (9–16px): suave y contemporáneo (pill solo en badges/toggle).

## 2. Colors

Paleta **monocroma navy** sobre neutros fríos: un único acento de acción (navy near-black), más una familia de colores de leyenda para tipologías de departamento en planos/mapas.

### Primary
- **Navy de Acción** (`#232741`): color de acción y de dato — botón primario, toggle activo, foco de inputs, borde activo de tabs, valores KPI destacados. Su hover/profundidad es `#14172a`. Es el único color con carga de "acción".

### Neutral
- **Papel Frío** (`#f5f6f8`): fondo general de la app (`body`, workspace).
- **Blanco Puro** (`#ffffff`): superficie de navbar, tarjetas, paneles, inputs — donde vive el dato.
- **Tinta de Texto** (`#1d2033`): texto primario / headings.
- **Gris Azulado** (`#868ba0`): texto secundario, labels, hints.
- **Borde Fino** (`#ececf1`) / **Borde Firme** (`#dcdde5`): división de superficies; el firme se reserva para inputs y hover states.

### Named Rules
**La Regla del Acento Único.** El navy near-black (`#232741`) es el único color con carga semántica de "acción" o "importancia" en toda la interfaz. Verde (`--color-success-border` `#059669`) y rojo (`--color-danger-border` `#dc2626`) existen solo como semántica de estado (cumple/no cumple normativa), nunca como decoración.

**Sin dorados/beige.** Ningún callout, KPI o superficie usa tonos cálidos (ámbar/crema/dorado); los informativos van en neutro frío. Restricción explícita del usuario.

### Status Tokens
Semántica de cumplimiento reutilizada en badges, `.compliance-item`, etiquetas del canvas 3D y texto de resultado (`.status-ok`/`.status-fail`/`.status-warn` en `styles.css`):
- **Success**: `--color-success-bg` (`#f0fdf4`), `--color-success-border` (`#059669`), `--color-success-text` (`#166534`).
- **Danger**: `--color-danger-bg` (`#fef2f2`), `--color-danger-border` (`#dc2626`), `--color-danger-text` (`#991b1b`).
- **Warn**: `--color-warn-bg` (`#fef3c7`), `--color-warn-border` (`#f59e0b`), `--color-warn-text` (`#92400e`).
- **Slate (neutral, etiquetas 3D)**: `--color-slate-800` (`#1e293b`), `--color-slate-600` (`#475569`), `--color-slate-500` (`#94a3b8`), `--color-slate-400` (`#64748b`).

## 3. Typography

**Body/UI Font:** Inter (con fallback `system-ui, -apple-system, sans-serif`) — única familia tipográfica en todo el sistema.

**Character:** Una sola familia sans-serif geométrica-humanista que prioriza legibilidad de datos sobre expresividad; el peso (600–700) y el letter-spacing negativo (-0.01em) en títulos hacen el trabajo que normalmente haría una fuente display.

### Hierarchy
- **Title** (700, 1.2–1.3rem): valor KPI principal (`.area-display strong`, `.kpi-value`) — el número que el usuario vino a buscar.
- **Heading** (700, 0.88–0.92rem, -0.01em): títulos de panel y toolbar (`.panel h3`, `.toolbar h2`).
- **Body** (400–600, 0.82–0.85rem): texto de inputs, botones, contenido de tabla.
- **Label** (700, 0.65–0.74rem, uppercase, +0.03–0.05em): etiquetas de KPI, headers de tabla, hints — siempre en mayúsculas con tracking abierto para diferenciarse del dato.

### Named Rules
**La Regla Sin Display.** No hay tipografía "hero" ni display: es un producto de trabajo, no una landing. La jerarquía se logra con peso y tamaño de escala pequeña, nunca con tipografía ornamental.

## 4. Elevation

Sistema plano con acentos puntuales: la mayoría de superficies (accordion panels, cards, toolbar) descansan sin sombra o con `shadow-xs` casi imperceptible, y solo ganan una sombra más marcada (`shadow-sm`) en `:hover` como señal de interactividad. La excepción es el floating panel (ductos) y los dropdowns, que llevan `shadow-md` de forma permanente porque flotan sobre el canvas y necesitan separarse del fondo en todo momento, no solo al interactuar.

### Shadow Vocabulary
- **xs** (`0 1px 2px rgba(23,23,60,0.05)`): reposo — accordion panels, toolbar, tarjetas de tabla.
- **sm** (`0 2px 6px rgba(23,23,60,0.06), 0 1px 2px rgba(23,23,60,0.05)`): hover de paneles y accordions — señal de interactividad, no de jerarquía permanente.
- **md** (`0 8px 24px -6px rgba(23,23,60,0.14), 0 2px 6px -2px rgba(23,23,60,0.08)`): elementos flotantes sobre el canvas (floating-panel de ductos) — permanente, no solo hover.
- **lg** (`0 16px 40px -10px rgba(23,23,60,0.20)`): reservada para overlays de mayor prioridad (definida pero de uso puntual).

### Named Rules
**La Regla del Reposo Plano.** Ninguna superficie de contenido normal lleva sombra en reposo salvo `shadow-xs`; la sombra crece solo como respuesta a hover o a la necesidad de flotar sobre otro contenido (canvas 3D).

## 5. Components

### Buttons
- **Shape:** radio `sm` (`9px`, `--radius-sm`) — suave, nunca completamente redondeado salvo badges/pills.
- **Primary:** fondo navy sólido (`#232741`), texto blanco, `padding: 0.7rem 1.1rem`, sombra suave neutra (`0 4px 14px -6px rgba(20,23,42,0.45)`); en hover profundiza a `#14172a`.
- **Accent (Ver Mapa / 3D activo):** fill navy sólido con sombra neutra — señala la acción de vista primaria.
- **Tool (Ghost):** fondo blanco, borde `border-color`, texto secundario, `gap` + icono SVG; en hover pasa a `navy-soft` con texto/borde navy. Estados `:active` (translateY 1px) y `:focus-visible` (halo `navy-soft`). Variante `icon-only` con footprint cuadrado.

### Toggle Switch
- **Style:** track de 44×24px, gris `#d1d5db` en reposo, navy (`navy-accent`) al activarse; thumb blanco circular con sombra sutil que se traslada 20px.
- **Focus:** anillo de foco de 3px, gris cuando inactivo y `navy-soft` cuando activo.

### KPI Cards / Chips
- **Style:** fondo casi blanco, borde fino, radio `sm`. Label en mayúsculas gris, valor en `title` weight 700 (navy). `.status-check` usa pills verdes/rojas para cumple/no-cumple — nunca ámbar/dorado.

### Cards / Containers
- **Corner Style:** `radius-md` (12px) para paneles y toolbar; `radius-sm` (9px) para tarjetas internas (KPI, filas de tabla, ítems de normativa).
- **Background:** blanco puro sobre fondo `neutral-bg`.
- **Shadow Strategy:** ver Elevación — sombras muy suaves; `shadow-xs` en reposo, `shadow-sm` en hover.
- **Border:** 1px `border` en casi todos los contenedores; separación primaria, la sombra es secundaria y sutil.

### Inputs / Fields
- **Style:** borde 1.5px `border-strong`, radio `sm`, fondo blanco.
- **Focus:** borde navy + halo de 3.5px en `navy-soft` — sin sombra dura, coherente con la filosofía de elevación ambiental.

### Navigation (Navbar)
- **Style:** franja blanca (`#ffffff`) de 60px con borde inferior fino (`border-color`) — clara y aireada, sin glow ni zona atmosférica.
- **Brand:** logo + wordmark directamente sobre el navbar blanco (sin píldora), chip `BETA` en `navy-soft`, crédito de autor como texto muted a la derecha.
- **Actions:** botones ghost/fill sobre blanco; `sync-status` como pill gris claro con borde fino y punto verde.
- **Tabs (sidebar y workspace):** texto gris secundario en reposo, navy en hover/activo, con un borde inferior de 2px navy (`box-shadow: inset 0 -2px 0`) marcando el estado activo — nunca un fondo sólido de color fuerte.

### Floating Panel (Ductos)
[Signature Component] Panel flotante sobre el canvas 3D: fondo blanco 92% translúcido con `backdrop-filter: blur(8px)`, `shadow-md` permanente (única excepción a la regla de reposo plano, justificada por flotar sobre contenido 3D activo).

## 6. Do's and Don'ts

### Do:
- **Do** usar el navy (`#232741`/`#14172a`) como único acento de acción/importancia en toda la interfaz — botón primario, toggle activo, foco, valores clave.
- **Do** mantener bordes finos (1–1.5px, `#ececf1`/`#dcdde5`) + sombras muy suaves como separador primario entre superficies.
- **Do** usar mayúsculas + tracking abierto solo en labels/headers de tabla (0.65–0.74rem) para diferenciarlos del dato.
- **Do** dejar respirar el layout: aire y whitespace hacen el trabajo, no la decoración.

### Don't:
- **Don't** convertir la interfaz en una grilla de hoja de cálculo: nunca añadir líneas de grilla densas ni cabeceras de tabla sin jerarquía tipográfica.
- **Don't** adoptar la estética CAD genérica: nada de grises industriales, iconografía diminuta ni menús anidados profundos.
- **Don't** introducir dorado/beige en ningún componente — color eliminado explícitamente del sistema (navbar incluido).
- **Don't** añadir sombras oscuras o duras en reposo a tarjetas/paneles de contenido; la sombra solo crece en hover o cuando el elemento flota sobre el canvas 3D.
- **Don't** introducir una segunda familia tipográfica o tipografía display; Inter cubre toda la jerarquía con peso y tamaño.
