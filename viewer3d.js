/**
 * viewer3d.js  —  Motor WebGL / Three.js para ArchGEN32
 * Motor ÚNICO de renderizado. El Canvas 2D antiguo queda desactivado.
 *
 * API pública:
 *   const v = new Viewer3D(container);
 *   v.renderProyecto(geometria, metadata);  ← re-renderiza (limpia antes)
 *   v.setMode('3d' | '2d');                 ← toggle volumetría
 *   v.setView('tipica' | 'primero' | 'sotano');
 *   v.dispose();
 */

import * as THREE from 'three';
import { OrbitControls }          from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { Line2 }        from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

// ═══════════════════════════════════════════════════════════
//  PALETA DE COLORES — PLANO ARQUITECTÓNICO (fondo blanco)
// ═══════════════════════════════════════════════════════════
const C = {
  bg:           0xf8fafc,   // fondo blanco hueso
  grid:         0xe2e8f0,   // grid gris muy claro

  lote:         0xef4444,   // lindero rojo
  retiro:       0xef4444,

  // Paleta legacy (fallback) — actualmente no usada
  apt_palette: [
    0xdbeafe, 0xfce7f3, 0xd1fae5, 0xe0e7ff, 0xfef9c3,
    0xffe4e6, 0xf3e8ff, 0xecfdf5, 0xfef3c7, 0xe0f2fe,
  ],
  // Color por tipología — paleta consistente alineada a referencia profesional
  apt_by_typology: {
    "1D":   0xfce7f3,  // rosa pastel
    "1D+E": 0xe0e7ff,  // lavanda pastel
    "2D":   0xe0f2fe,  // celeste pastel
    "2D+E": 0xd1fae5,  // verde pastel
    "3D":   0xfef9c3,  // amarillo pastel
  },
  apt_stroke:   0x1e3a5f,  // azul oscuro tipo muro arquitectónico

  // Zonas interiores — versión ligeramente más oscura del color del apto
  zone_alpha:   0.18,       // solo para darkening relativo
  zona_circ:    0xf1f5f9,
  zona_cocina:  0xfff7ed,
  zona_bano:    0xe0f2fe,
  zona_lav:     0xfef9c3,
  zona_esc:     0xfae8ff,
  zona_dorm:    0xdbeafe,
  zona_estar:   0xecfdf5,
  zona_stroke:  0x64748b,

  hall:         0xcbd5e1,
  hall_stroke:  0x334155,
  core:         0xe2e8f0,
  esc_abierta:  0xf1f5f9,
  esc_pres:     0xfef3c7,
  esc_stroke:   0x334155,
  ascensor:     0xdbeafe,
  asc_stroke:   0x334155,
  vestibulo:    0xfef9c3,
  vest_stroke:  0xd97706,
  patio:        0xe0f2fe,
  patio_stroke: 0x0369a1,
  ducto:        0xffedd5,
  ducto_stroke: 0xea580c,
  pozo_luz:        0x6ee7b7,
  pozo_luz_stroke: 0x047857,
  columna:      0x475569,
  columna_stroke: 0x1e293b,
  stall:        0xffffff,
  aisle:        0xe2e8f0,
};

// ═══════════════════════════════════════════════════════════
//  HELPERS GEOMÉTRICOS
// ═══════════════════════════════════════════════════════════

/** Área polígono [[x,y],...] por shoelace, m². */
function polyArea(coords) {
  if (!coords || coords.length < 3) return 0;
  let a = 0;
  for (let i = 0, n = coords.length; i < n; i++) {
    const j = (i + 1) % n;
    a += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1];
  }
  return Math.abs(a) / 2;
}

/**  [[x,y],...] → THREE.Shape  (plano XY, Y invertido = vista planta top-down) */
function toShape(coords) {
  if (!coords || coords.length < 3) return null;
  const s = new THREE.Shape();
  s.moveTo(coords[0][0], -coords[0][1]);
  for (let i = 1; i < coords.length; i++) s.lineTo(coords[i][0], -coords[i][1]);
  s.closePath();
  return s;
}

/** [[x,y],...] → THREE.Vector3[] cerrado */
function toLoop(coords, z = 0.05) {
  if (!coords || coords.length < 2) return [];
  const pts = coords.map(([x, y]) => new THREE.Vector3(x, -y, z));
  pts.push(pts[0].clone());
  return pts;
}

/** centroid [[x,y],...] → [cx, cy] */
function cent(coords) {
  const n = coords.length;
  return [
    coords.reduce((s, p) => s + p[0], 0) / n,
    coords.reduce((s, p) => s + p[1], 0) / n,
  ];
}

/** bounding box   { minX, maxX, minY, maxY, cx, cy, w, h } */
function bbox(coords) {
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return { minX, maxX, minY, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY };
}

// ═══════════════════════════════════════════════════════════
//  CLASE VIEWER3D
// ═══════════════════════════════════════════════════════════
export class Viewer3D {

  constructor(container) {
    this.container   = container;
    this.mode        = '2d';
    this._geometria  = null;
    this._metadata   = null;
    this._activeView = 'tipica';
    this._groups     = {};          // name → THREE.Group
    this._labels     = [];          // CSS2DObject[] para limpiar DOM
    this._disposed   = false;
    this._frameLogged = false;       // Debug flag
    this._sotanoLevelIdx = 0;        // Nivel de sótano mostrado en 2D (S1=0, S2=1...)

    this._buildRenderer();
    this._buildScene();
    this._buildCameras();
    // Apply correct pixel size AFTER cameras are built
    this._applySize();
    requestAnimationFrame(() => this._applySize());
    this._buildControls();
    this._buildGrid();
    this._loop();
    this._buildNavButtons();
    this._buildNavCube();

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);
  }

  // ─────────────────────────────────────────────────────────
  //  INICIALIZACIÓN
  // ─────────────────────────────────────────────────────────

  _buildRenderer() {
    // Append to container first so we can measure its true rendered dimensions
    // (important when container is in a flex/grid layout that isn't laid out yet at constructor time)

    // WebGL canvas — let CSS control display size; Three.js only controls the pixel buffer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(C.bg, 1);
    this.renderer.shadowMap.enabled = false;    // off for perf in 2D mode

    const canvas = this.renderer.domElement;
    Object.assign(canvas.style, {
      position: 'absolute', inset: '0',
      width: '100%', height: '100%',
      zIndex: '20', display: 'block',
    });
    this.container.appendChild(canvas);

    // CSS2D overlay
    this.labelRenderer = new CSS2DRenderer();
    const lblEl = this.labelRenderer.domElement;
    Object.assign(lblEl.style, {
      position: 'absolute', inset: '0',
      width: '100%', height: '100%',
      zIndex: '21', pointerEvents: 'none', overflow: 'hidden',
    });
    this.container.appendChild(lblEl);

    // Note: _applySize() will be called after _buildCameras() in constructor
  }

  /** Read real container pixel size and push it to renderer + cameras */
  _applySize() {
    const W = this.container.offsetWidth;
    const H = this.container.offsetHeight;
    if (!W || !H) return;

    // updateStyle = false → do NOT override our CSS width/height on the canvas element
    this.renderer.setSize(W, H, false);
    this.labelRenderer.setSize(W, H);

    const aspect = W / H;
    this._frustum = this._frustum || 60;
    this.camOrtho.left   = -this._frustum * aspect / 2;
    this.camOrtho.right  =  this._frustum * aspect / 2;
    this.camOrtho.top    =  this._frustum / 2;
    this.camOrtho.bottom = -this._frustum / 2;
    this.camOrtho.updateProjectionMatrix();

    this.camPersp.aspect = aspect;
    this.camPersp.updateProjectionMatrix();
  }



  _buildScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(C.bg);

    // Iluminación plana — sin sombras, colores puros para estilo de plano
    this.scene.add(new THREE.AmbientLight(0xffffff, 2.5));
    const sun = new THREE.DirectionalLight(0xffffff, 0.3);
    sun.position.set(30, 60, 80);
    this.scene.add(sun);
  }

  _buildCameras() {
    const { clientWidth: W, clientHeight: H } = this.container;
    const aspect = W / Math.max(H, 1);

    // Ortográfica — vista planta 2D
    const F = 60;
    this.camOrtho = new THREE.OrthographicCamera(
      -F * aspect / 2, F * aspect / 2, F / 2, -F / 2, 0.1, 2000
    );
    this.camOrtho.position.set(0, 0, 200);
    this.camOrtho.lookAt(0, 0, 0);
    this._frustum = F;

    // Perspectiva — volumetría 3D  (Z-up: edificio sube en Z)
    this.camPersp = new THREE.PerspectiveCamera(45, aspect, 0.1, 2000);
    this.camPersp.up.set(0, 0, 1);   // Z es "arriba" para orbitar correctamente
    this.camPersp.position.set(0, -80, 120);
    this.camPersp.lookAt(0, 0, 0);

    this.camera = this.camOrtho;
  }

  _buildControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this._apply2DControls();
    this.controls.update();
  }

  _apply2DControls() {
    const c = this.controls;
    c.enableRotate        = false;
    c.enableDamping       = true;
    c.dampingFactor       = 0.12;
    c.screenSpacePanning  = true;
    c.zoomSpeed           = 1.2;
    c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    c.touches      = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
  }

  _apply3DControls() {
    const c = this.controls;
    c.enableRotate        = true;
    c.enableDamping       = true;
    c.dampingFactor       = 0.08;
    c.screenSpacePanning  = false;
    c.minDistance         = 5;
    c.maxDistance         = 600;
    c.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    c.touches      = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  }

  _buildGrid() {
    // Grid de fondo tipo AutoCAD — muy sutil, líneas tenues
    const g = new THREE.GridHelper(400, 200, C.grid, C.grid);
    g.rotation.x = Math.PI / 2;
    g.position.z = -0.3;
    g.material.opacity = 0.25;
    g.material.transparent = true;
    this.scene.add(g);
    this._grid = g;

    // Inyectar estilos CSS2D para etiquetas de plano arquitectónico (solo la primera vez)
    if (!document.getElementById('cad-label-styles')) {
      const st = document.createElement('style');
      st.id = 'cad-label-styles';
      st.textContent = `
        .three-label {
          font-family: 'Inter', system-ui, sans-serif;
          font-size: 10px;
          font-weight: 600;
          color: #1e293b;
          text-align: center;
          pointer-events: none;
          white-space: pre-line;
          line-height: 1.4;
        }
        .three-label.label-nucleo {
          color: #475569;
          font-size: 8px;
          font-weight: 500;
        }
        .three-label.label-apt {
          color: #0f172a;
          font-size: 11px;
          font-weight: 700;
        }
        .three-label.label-warn  { color: #b45309; }
        .three-label.label-ext   { color: #92400e; }  /* fachada exterior: ámbar oscuro */
        .three-label.label-int   { color: #334155; }  /* fachada interior: gris pizarra */
        .three-label.label-ducto { color: #ea580c; font-size: 8px; }
        .three-label.label-pozo  { color: #0f172a; font-size: 7px; font-weight: 600; background: transparent; border: none; box-shadow: none; backdrop-filter: none; padding: 0; }
        .three-label.label-stall { color: #64748b; font-size: 7px; font-weight: 400; }
        .three-label.label-cisterna { color: #1d4ed8; font-size: 8px; }
        .three-label.label-comercio { color: #059669; font-size: 9px; }
        .three-label.label-lobby    { color: #d97706; font-size: 9px; }
        .three-label.label-rampa    { color: #dc2626; font-size: 8px; }
        .three-label.label-serv     { color: #475569; font-size: 8px; }
        .three-label.label-cota     { color: #334155; font-size: 9px; font-weight: 600; background: rgba(255,255,255,0.85); padding: 1px 4px; border-radius: 3px; }
      `;
      document.head.appendChild(st);
    }
  }

  _loop() {
    const tick = () => {
      if (this._disposed) return;
      this._raf = requestAnimationFrame(tick);
      this.controls.update();
      
      // Sincronizar visibilidad de etiquetas con sus grupos padres
      this._labels.forEach(lbl => {
        if (lbl.parent && lbl.parent.visible !== undefined) {
          lbl.visible = lbl.parent.visible;
        }
      });
      
      this.renderer.render(this.scene, this.camera);
      this.labelRenderer.render(this.scene, this.camera);
      if (this._navCube && this.mode === '3d') this._syncNavCube();
    };
    tick();
  }

  // ─────────────────────────────────────────────────────────
  //  RESIZE
  // ─────────────────────────────────────────────────────────

  _resize() { this._applySize(); }



  // ─────────────────────────────────────────────────────────
  //  MODO 2D / 3D
  // ─────────────────────────────────────────────────────────

  /** Alterna entre planta técnica ('2d') y volumetría ('3d') */
  setMode(mode) {
    this.mode = mode;
    if (mode === '3d') {
      this.camera = this.camPersp;
      this._apply3DControls();
      this._grid.rotation.set(0, 0, 0);
      this._grid.position.set(0, -0.5, 0);
      if (this._geometria?.lote?.coords) this._fit3D(this._geometria.lote.coords);
    } else {
      this.camera = this.camOrtho;
      this._apply2DControls();
      this._grid.rotation.x = Math.PI / 2;
      this._grid.position.set(0, 0, -0.2);
      if (this._geometria?.lote?.coords) this._fit2D(this._geometria.lote.coords);
    }
    this.controls.object = this.camera;
    this.controls.update();

    if (this._navOverlay) this._navOverlay.style.display = mode === '3d' ? 'flex' : 'none';
    if (this._navCube)    this._navCube.canvas.style.display = mode === '3d' ? 'block' : 'none';

    // Re-render la escena con la nueva profundidad de extrusión
    if (this._geometria) this.renderProyecto(this._geometria, this._metadata);
  }

  // ─────────────────────────────────────────────────────────
  //  RENDER PRINCIPAL
  // ─────────────────────────────────────────────────────────

  /**
   * Punto de entrada único. Limpia la escena anterior y dibuja
   * todo el proyecto usando el payload normalizado del backend.
   */
  renderProyecto(geometria, metadata = {}) {
    this._geometria = geometria;
    this._metadata  = metadata;

    // Limpieza total de la escena anterior
    this.clearScene();

    const is3D    = this.mode === '3d';
    const pisos   = metadata.pisos || 1;
    const altPiso = metadata.altura_piso || 2.80;
    const depth3D = pisos * altPiso;

    // Extrusión: thin slab en 2D, edificio completo en 3D
    const extrude = (base = true) => ({
      depth: is3D ? (base ? depth3D : depth3D) : 0.05,
      bevelEnabled: false,
    });
    const thin = { depth: 0.04, bevelEnabled: false };

    // pozoHoles: fallback visual si backend falló en sustraer el pozo (aplica en 2D y 3D).
    // El guard pointInPolygon en _mesh() evita geometría inválida cuando ya fue sustraído.
    const pozoHoles = (geometria.tecnico?.pozos_luz || [])
      .filter(p => p.coords?.length >= 3)
      .map(p => p.coords);

    // ── 1. LOTE ───────────────────────────────────────────────────────
    const loteCoords = geometria.lote?.coords;
    if (loteCoords?.length >= 3) {
      const g = this._group('lote');
      this._mesh(g, loteCoords, 0xffffff, 1.0, thin, -0.08);
      this._outline(g, loteCoords, C.lote, 3, 0.1);
      // "LÍMITE DE PROPIEDAD" en los bordes superior y derecho del bbox,
      // fuera del lote, replicando la convención profesional.
      const xs = loteCoords.map(c => c[0]);
      const ys = loteCoords.map(c => c[1]);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const pad = 0.8;
      this._labelAt(g, (xMin + xMax) / 2, yMin - pad, 'LÍMITE DE PROPIEDAD', 'label-lote', 0.4);
      this._labelAt(g, xMax + pad, (yMin + yMax) / 2, 'LÍMITE DE PROPIEDAD', 'label-lote label-lote-vert', 0.4);

      // Cotas: longitud de cada arista con nombre de lado (Fte/Fdo/Der/Izq)
      const n = loteCoords.length;
      const pcx = loteCoords.reduce((s, c) => s + c[0], 0) / n;
      const pcy = loteCoords.reduce((s, c) => s + c[1], 0) / n;
      // Heurística de nombre por posición del punto medio relativo al centroide
      const sideLabel = (mx, my) => {
        const dx = mx - pcx, dy = my - pcy;
        if (Math.abs(dy) >= Math.abs(dx)) return dy < 0 ? 'Fte' : 'Fdo';
        return dx > 0 ? 'Der' : 'Izq';
      };
      for (let i = 0; i < n; i++) {
        const [x1, y1] = loteCoords[i];
        const [x2, y2] = loteCoords[(i + 1) % n];
        const len = Math.hypot(x2 - x1, y2 - y1);
        if (len < 1.0) continue;
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        // Normal perpendicular al borde, apuntando al exterior del polígono
        let nx = -(y2 - y1), ny = (x2 - x1);
        const nm = Math.hypot(nx, ny) || 1;
        nx /= nm; ny /= nm;
        if (nx * (mx - pcx) + ny * (my - pcy) < 0) { nx = -nx; ny = -ny; }
        const off = 1.8;
        this._labelAt(g, mx + nx * off, my + ny * off,
          `${len.toFixed(1)}m (${sideLabel(mx, my)})`, 'label-cota', 0.4);
      }
    }

    // ── 2. RETIROS ───────────────────────────────────────────
    (geometria.retiros || []).forEach(r => {
      if (r.coords?.length < 3) return;
      const g = this._group('retiros');
      this._mesh(g, r.coords, C.retiro, 0.08, thin, 0.02);
      this._outline(g, r.coords, C.retiro, 1, 0.1);
      this._label(g, r.coords, 'RETIRO', 'label-nucleo', 0.15);
    });

    // ── 3. HALL / CIRCULACIÓN ────────────────────────────────
    const corridors = geometria.circulacion?.corridors || [];
    if (corridors.length > 0) {
      // Claustro: render 4 narrow ring-corridor strips individually
      let totalCorrArea = 0;
      corridors.forEach(corr => {
        if (!corr.coords?.length) return;
        const g = this._group('circulacion');
        this._mesh(g, corr.coords, C.hall, 0.92, thin, 0.06);
        this._outline(g, corr.coords, C.hall_stroke, 2, 0.1);
        totalCorrArea += polyArea(corr.coords);
      });
      if (corridors[0]?.coords?.length) {
        this._label(this._group('circulacion'), corridors[0].coords,
          `CORREDOR\n${totalCorrArea.toFixed(1)} m²`, 'label-nucleo', 0.3);
      }
    } else {
      const hallCoords = geometria.circulacion?.hall?.coords;
      if (hallCoords?.length >= 3) {
        const g = this._group('circulacion');
        this._mesh(g, hallCoords, C.hall, 0.92, extrude(), 0.03, pozoHoles);
        this._outline(g, hallCoords, C.hall_stroke, 2, 0.1);
        const hallArea = polyArea(hallCoords).toFixed(2);
        this._label(g, hallCoords, `HALL\n${hallArea} m²`, 'label-nucleo', is3D ? depth3D + 0.5 : 0.3);
      }
    }

    // ── 4. NÚCLEO ────────────────────────────────────────────
    const nuc = geometria.nucleo || {};
    const pres = nuc.escaleras?.tipo === 'presurizada';

    if (nuc.core_envelope?.coords?.length >= 3) {
      this._mesh(this._group('nucleo'), nuc.core_envelope.coords, C.core, 0.85, extrude(), 0.02, pozoHoles);
    }

    // Escalera
    if (nuc.escaleras?.coords?.length >= 3) {
      const g = this._group('nucleo');
      this._mesh(g, nuc.escaleras.coords, pres ? C.esc_pres : C.esc_abierta, 0.96, extrude(), 0.1);
      this._outline(g, nuc.escaleras.coords, C.esc_stroke, 2, 0.15);
      this._stairLines(g, nuc.escaleras.coords);
      this._label(g, nuc.escaleras.coords, pres ? 'ESC PRES.\n🔒' : 'ESC\n⇋', 'label-nucleo', is3D ? depth3D + 0.5 : 0.25);
    }

    // Ascensores
    (nuc.ascensores || []).forEach((asc, i) => {
      if (!asc.coords?.length) return;
      const g = this._group('nucleo');
      this._mesh(g, asc.coords, C.ascensor, 0.92, extrude(), 0.1);
      this._outline(g, asc.coords, C.asc_stroke, 1.5, 0.15);
      this._cross(g, asc.coords);
      this._label(g, asc.coords, `ASC ${i + 1}`, 'label-nucleo', is3D ? depth3D + 0.5 : 0.25);
    });

    // Vestíbulo previo
    if (nuc.vestibulo?.coords?.length >= 3) {
      const g = this._group('nucleo');
      this._mesh(g, nuc.vestibulo.coords, C.vestibulo, 0.92, extrude(), 0.1);
      this._outline(g, nuc.vestibulo.coords, C.vest_stroke, 1.5, 0.15);
      this._label(g, nuc.vestibulo.coords, 'VEST.\nPREVIO', 'label-nucleo', is3D ? depth3D + 0.5 : 0.25);
    }

    // ── 5. TÉCNICO ───────────────────────────────────────────
    const tec = geometria.tecnico || {};

    (tec.patios || []).forEach(p => {
      if (!p.coords?.length) return;
      const g = this._group('tecnico');
      this._mesh(g, p.coords, C.patio, 0.88, thin, 0.05);
      this._outline(g, p.coords, C.patio_stroke, 1.8, 0.12);
      this._cross(g, p.coords, C.patio_stroke, 0.35);
      const txt = `PATIO\n≥${(p.dimension_minima_requerida || 0).toFixed(1)}m`;
      this._label(g, p.coords, txt, p.cumple_minimo ? 'label-ok' : 'label-fail', 0.2);
    });

    // Ductos sobre apartamentos (zAptBase=0.15 + aptExtrudeDepth=0.06 → top=0.21)
    const zDucto = is3D ? depth3D + 0.10 : 0.30;
    (tec.ductos || []).forEach(d => {
      if (!d.coords?.length) return;
      const g = this._group('tecnico');
      this._mesh(g, d.coords, C.ducto, 0.96, thin, zDucto);
      this._outline(g, d.coords, C.ducto_stroke, 1.8, zDucto + 0.04);
      this._cross(g, d.coords, C.ducto_stroke, zDucto + 0.04);
      this._label(g, d.coords, 'DUCTO\nSERV.', 'label-ducto', zDucto + 0.06);
    });

    // Pozos de luz — convención de plano: vacío blanco humo con X
    // de arista a arista (diagonales). Sub-norma (< H/4) mantiene borde rojo.
    (tec.pozos_luz || []).forEach(p => {
      if (!p.coords?.length) return;
      const g = this._group('pozos');
      const subNorma = p.cumple === false;
      const fillCol   = subNorma ? 0xfee2e2 : 0xf5f5f4;   // blanco humo
      const strokeCol = subNorma ? 0xdc2626 : 0x9ca3af;   // gris claro
      const lbl = subNorma ? 'POZO < MÍN' : 'POZO DE LUZ';
      if (is3D) {
        // 3D: pozo = vacío puro. shape.holes ya corta el hueco en dptos/hall/core.
        // Solo marcar la boca en cubierta — ningún mesh dentro del shaft.
        const zRim = depth3D + 0.01;
        this._outline(g, p.coords, strokeCol, 2.0, zRim + 0.06);
        this._cross(g, p.coords, strokeCol, 0.9, zRim + 0.08);
        this._label(g, p.coords, lbl, 'label-pozo', zRim + 0.12);
      } else {
        const zPozo = 0.35;
        this._mesh(g, p.coords, fillCol, 1.0, thin, zPozo);
        this._outline(g, p.coords, strokeCol, 1.5, zPozo + 0.02);
        this._cross(g, p.coords, strokeCol, 0.9, zPozo + 0.04);
        this._label(g, p.coords, lbl, 'label-pozo', zPozo + 0.08);
      }
    });


    // ── 6. UNIDADES (DEPARTAMENTOS) + divisiones interiores ──────────
    const zAptBase = 0.15;
    const aptExtrudeDepth = is3D ? depth3D : 0.06;
    const zZonasTop = zAptBase + aptExtrudeDepth + 0.01;

    (geometria.unidades || []).forEach((u, idx) => {
      if (!u.coords?.length) return;
      const g   = this._group('unidades');

      const typ = u.metadata?.tipologia || '?D';
      const aptColor = C.apt_by_typology[typ] != null
        ? C.apt_by_typology[typ]
        : C.apt_palette[idx % C.apt_palette.length];
      const isFachadaExt = !!u.validacion?.fachada_exterior;
      const esReducida   = !!u.metadata?.es_reducida;
      this._mesh(g, u.coords, aptColor, 0.90, extrude(), zAptBase, pozoHoles);
      this._outline(g, u.coords, C.apt_stroke, 4.2, zAptBase + 0.07);
      if (isFachadaExt) {
        this._outline(g, u.coords, 0xf59e0b, 3.5, zAptBase + 0.09); // gold = exterior fachada
      }
      if (is3D) {
        // Perímetro superior (techo del volumen extruido)
        this._outline(g, u.coords, C.apt_stroke, 1.8, zAptBase + depth3D + 0.02);
        // Líneas de división entre pisos
        if (pisos > 1) {
          for (let f = 1; f < pisos; f++) {
            this._outline(g, u.coords, 0x475569, 1.0, zAptBase + altPiso * f);
          }
        }
      }

      // Zonas interiores ocultas por defecto (cabida pre-financiera).
      if (typeof window !== 'undefined' && window.SHOW_INTERIOR_ZONES) {
        (u.zonas || []).forEach(z => {
          if (!z.coords || z.coords.length < 3) return;
          const zCol = this._zoneColor(z.nombre);
          this._mesh(g, z.coords, zCol, 0.80, thin, zZonasTop);
          this._outline(g, z.coords, C.zona_stroke, 0.6, zZonasTop + 0.01);
          this._label(g, z.coords, z.nombre, 'label-zona', zZonasTop + 0.05);
        });
      }

      const area = (u.metadata?.area ?? polyArea(u.coords)).toFixed(2);
      const ok   = u.validacion?.colinda_hall !== false;
      const id   = u.id || `X${String(idx + 1).padStart(2, '0')}`;
      const fachadaMark = isFachadaExt ? '\n<span class="fach-text">◆ FACH. EXT.</span>' : '\n<span class="fach-text">○ FACH. INT.</span>';
      const reducidaMark = esReducida ? ' ⚠' : '';
      const txt  = `${typ}\nDPTO ${id}\n${area} m²${fachadaMark}${reducidaMark}`;
      const labelClass = `label-apt${ok ? '' : ' label-warn'}${isFachadaExt ? ' label-ext' : ' label-int'}`;
      this._label(g, u.coords, txt, labelClass, zAptBase + 0.32);
    });

    // ── 7. PRIMER PISO ───────────────────────────────────────
    const pp = geometria.primer_piso;
    if (pp) {
      const primerExtrude = is3D ? { depth: altPiso, bevelEnabled: false } : thin;
      const zPP = is3D ? -altPiso : 0.03;

      (pp.comercios || []).forEach(c => {
        if (!c?.length) return;
        const g = this._group('primer_piso');
        this._mesh(g, c, 0xd1fae5, 0.88, primerExtrude, zPP);
        this._outline(g, c, 0x10b981, 1.4, zPP + 0.1);
        this._label(g, c, 'COMERCIO', 'label-comercio', zPP + 0.2);
      });
      if (pp.lobby?.length >= 3) {
        const g = this._group('primer_piso');
        this._mesh(g, pp.lobby, 0xfef3c7, 0.9, primerExtrude, zPP);
        this._outline(g, pp.lobby, 0xd97706, 1.5, zPP + 0.1);
        this._label(g, pp.lobby, 'LOBBY', 'label-lobby', zPP + 0.2);
      }
      if (pp.rampa?.length >= 3) {
        const g = this._group('primer_piso');
        this._mesh(g, pp.rampa, 0xfee2e2, 0.88, primerExtrude, zPP);
        this._outline(g, pp.rampa, 0xdc2626, 1.4, zPP + 0.1);
        this._label(g, pp.rampa, 'RAMPA\n3m', 'label-rampa', zPP + 0.2);
      }
      if (pp.servicios?.length >= 3) {
        const g = this._group('primer_piso');
        this._mesh(g, pp.servicios, 0xe2e8f0, 0.88, primerExtrude, zPP);
        this._outline(g, pp.servicios, 0x475569, 1.2, zPP + 0.1);
        this._label(g, pp.servicios, 'SSHH\nPCD', 'label-serv', zPP + 0.2);
      }
      if (pp.basura?.length >= 3) {
        const g = this._group('primer_piso');
        this._mesh(g, pp.basura, 0xfef9c3, 0.88, primerExtrude, zPP);
        this._outline(g, pp.basura, 0xca8a04, 1.4, zPP + 0.1);
        this._label(g, pp.basura, 'BASURA', 'label-serv', zPP + 0.2);
      }
      if (pp.tableros?.length >= 3) {
        const g = this._group('primer_piso');
        this._mesh(g, pp.tableros, 0xdbeafe, 0.88, primerExtrude, zPP);
        this._outline(g, pp.tableros, 0x1d4ed8, 1.4, zPP + 0.1);
        this._label(g, pp.tableros, 'TABL.\nELEC.', 'label-serv', zPP + 0.2);
      }
    }

    // ── 8. SÓTANO ────────────────────────────────────────────
    // 2D: solo dibuja el nivel seleccionado (this._sotanoLevelIdx) — apilar
    // todos en top-view los superpone en la misma x,y (ilegible). 3D: apila
    // todos los niveles reales, cada uno visible por su profundidad propia.
    const sot = geometria.sotano;
    if (sot) {
      const niveles = (sot.niveles && sot.niveles.length)
        ? sot.niveles
        : [{ name: sot.name, stalls: sot.stalls, aisles: sot.aisles, count: sot.count }];
      const selIdx = Math.min(this._sotanoLevelIdx || 0, niveles.length - 1);
      const drawNivel = (nv, i) => {
        const zNv = is3D ? -altPiso * 1.5 * (i + 1) : -0.05;
        const g = this._group('sotano');
        if (sot.slab?.length >= 3) {
          this._mesh(g, sot.slab, 0xe2e8f0, 0.82, thin, zNv);
          this._outline(g, sot.slab, 0x64748b, 1.5, zNv + 0.05);
        }
        (nv.aisles || []).forEach(a => {
          if (!a?.length) return;
          this._mesh(this._group('sotano'), a, 0xe2e8f0, 0.6, thin, zNv + 0.02);
        });
        (nv.stalls || []).forEach(st => {
          if (!st.coords?.length) return;
          this._mesh(g, st.coords, 0xffffff, 0.92, thin, zNv + 0.03);
          this._outline(g, st.coords, 0x475569, 0.9, zNv + 0.06);
          this._label(g, st.coords, st.id, 'label-stall', zNv + 0.1);
        });
        if (i === 0) {
          (sot.cisternas || []).forEach(c => {
            if (!c.coords?.length) return;
            const col = parseInt((c.fill || '#bfdbfe').replace('#', ''), 16);
            const strk = parseInt((c.stroke || '#2563eb').replace('#', ''), 16);
            this._mesh(g, c.coords, col, 0.92, thin, zNv + 0.04);
            this._outline(g, c.coords, strk, 1.2, zNv + 0.08);
            this._label(g, c.coords, c.label || '', 'label-cisterna', zNv + 0.1);
          });
        }
      };
      if (is3D) {
        niveles.forEach(drawNivel);
      } else {
        drawNivel(niveles[selIdx], selIdx);
      }
    }

    // ── 9. AZOTEA ────────────────────────────────────────────
    const az = geometria.azotea;
    if (az) {
      const azExtrude = is3D ? { depth: altPiso * 0.5, bevelEnabled: false } : thin;
      const zAz = is3D ? depth3D + altPiso * 0.1 : 0.20;

      if (az.caja_escalera?.length >= 3) {
        const g = this._group('azotea');
        this._mesh(g, az.caja_escalera, 0xe0e7ff, 0.90, azExtrude, zAz);
        this._outline(g, az.caja_escalera, 0x4338ca, 1.6, zAz + 0.05);
        this._label(g, az.caja_escalera, 'CAJA\nESC.', 'label-serv', zAz + 0.15);
      }
      if (az.cuarto_maquinas?.length >= 3) {
        const g = this._group('azotea');
        this._mesh(g, az.cuarto_maquinas, 0xfce7f3, 0.90, azExtrude, zAz);
        this._outline(g, az.cuarto_maquinas, 0xbe185d, 1.6, zAz + 0.05);
        const cmTxt = `CUARTO\nMÁQ.\n${(az.area_cm_m2 || 0).toFixed(1)}m²`;
        this._label(g, az.cuarto_maquinas, cmTxt, 'label-serv', zAz + 0.15);
      }
      if (az.tanque_elevado?.length >= 3) {
        const g = this._group('azotea');
        this._mesh(g, az.tanque_elevado, 0xdbeafe, 0.90, azExtrude, zAz);
        this._outline(g, az.tanque_elevado, 0x1d4ed8, 2.0, zAz + 0.05);
        const tankTxt = `TANQUE\n${(az.vol_tanque_m3 || 0).toFixed(1)}m³`;
        this._label(g, az.tanque_elevado, tankTxt, 'label-cisterna', zAz + 0.15);
      }
    }

    // Sincronizar visibilidad según la vista activa
    this._syncVis();

    // Auto-fit cámara
    this._fit2D(loteCoords);
    if (is3D) this._fit3D(loteCoords);
  }

  // ─────────────────────────────────────────────────────────
  //  LIMPIEZA DE ESCENA  (pública para uso externo)
  // ─────────────────────────────────────────────────────────

  /** Elimina todos los objetos creados por renderProyecto (geometrías, materiales y labels del DOM). */
  clearScene() {
    // Liberar CSS2DObject → elimina los <div> del DOM
    for (const lbl of this._labels) {
      if (lbl.element && lbl.element.parentNode) {
        lbl.element.parentNode.removeChild(lbl.element);
      }
    }
    this._labels = [];

    // Liberar grupos + geometría/materiales de GPU
    for (const group of Object.values(this._groups)) {
      group.traverse(obj => {
        if (obj.isMesh || obj.isLine) {
          obj.geometry?.dispose();
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material?.dispose();
        }
      });
      this.scene.remove(group);
    }
    this._groups = {};
  }

  // ─────────────────────────────────────────────────────────
  //  CONTROL DE VISTAS
  // ─────────────────────────────────────────────────────────

  setView(view) {
    this._activeView = view;
    this._syncVis();
  }

  /** Cambia el nivel de sótano mostrado en 2D (S1=0, S2=1...) y re-dibuja. */
  setSotanoLevel(idx) {
    this._sotanoLevelIdx = idx || 0;
    if (this._geometria) {
      this.renderProyecto(this._geometria, this._metadata);
      this._syncVis();
    }
  }

  _syncVis() {
    const v = this._activeView || 'tipica';
    const hidden = {
      tipica:   ['primer_piso', 'sotano', 'azotea', 'tecnico'],
      primero:  ['unidades', 'sotano', 'azotea'],
      sotano:   ['unidades', 'primer_piso', 'azotea', 'circulacion', 'nucleo', 'retiros', 'tecnico', 'pozos'],
      azotea:   ['unidades', 'primer_piso', 'sotano', 'pozos'],
    }[v] || [];
    for (const [name, grp] of Object.entries(this._groups)) {
      grp.visible = !hidden.includes(name);
    }
  }

  // ─────────────────────────────────────────────────────────
  //  PRIMITIVAS
  // ─────────────────────────────────────────────────────────

  /** Malla rellena — MeshStandardMaterial para reaccionar a la luz
   *  holes: array de [[x,y],...] que se recortan de la forma (pozos de luz) */
  _mesh(group, coords, color, opacity = 1, extCfg = {}, zBase = 0, holes = []) {
    const shape = toShape(coords);
    if (!shape) return null;
    // Cortar pozos como agujeros solo si el centroide del pozo está DENTRO del polígono.
    // Evita geometría inválida cuando el backend ya sustrajo el pozo (centroide fuera).
    for (const h of (holes || [])) {
      if (!h || h.length < 3) continue;
      // Centroide del pozo en espacio Three (y negado)
      const hcx = h.reduce((s, p) => s + p[0], 0) / h.length;
      const hcy = -(h.reduce((s, p) => s + p[1], 0) / h.length);
      // Ray-casting point-in-polygon
      let inside = false;
      for (let i = 0, j = coords.length - 1; i < coords.length; j = i++) {
        const xi = coords[i][0], yi = -coords[i][1];
        const xj = coords[j][0], yj = -coords[j][1];
        if ((yi > hcy) !== (yj > hcy) && hcx < (xj - xi) * (hcy - yi) / (yj - yi) + xi)
          inside = !inside;
      }
      if (!inside) continue;  // pozo no dentro del polígono — ya fue sustraído o no aplica
      // Cortar el hole exactamente en las coords del pozo (sin expandir — earcut requiere
      // que el hole esté estrictamente dentro del shape para no generar triángulos inválidos).
      const path = new THREE.Path();
      path.moveTo(h[0][0], -h[0][1]);
      for (let i = 1; i < h.length; i++) path.lineTo(h[i][0], -h[i][1]);
      path.closePath();
      shape.holes.push(path);
    }
    const geo = new THREE.ExtrudeGeometry(shape, {
      depth: extCfg.depth ?? 0.04,
      bevelEnabled: extCfg.bevelEnabled ?? false,
    });
    const mat = new THREE.MeshStandardMaterial({
      color,
      transparent: opacity < 1,
      opacity,
      roughness: 0.8,
      metalness: 0.0,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.z = zBase;
    mesh.receiveShadow = true;
    mesh.castShadow    = true;
    group.add(mesh);
    return mesh;
  }

  /** Contorno de línea — usa Line2 para respetar linewidth en WebGL */
  _outline(group, coords, color, lw = 1.5, z = 0.1) {
    const pts = toLoop(coords, z);
    if (pts.length < 2) return;
    const positions = [];
    for (const pt of pts) positions.push(pt.x, pt.y, pt.z);
    const geo = new LineGeometry();
    geo.setPositions(positions);
    const mat = new LineMaterial({
      color,
      linewidth: lw,
      resolution: new THREE.Vector2(
        this.container.offsetWidth  || 800,
        this.container.offsetHeight || 600,
      ),
    });
    group.add(new Line2(geo, mat));
  }

  /** Cruz decorativa — usa Line2 para respetar linewidth en WebGL.
      X de arista a arista: pares de vértices opuestos del polígono (no bbox —
      un pozo clippeado/irregular tenía la X desalineada de sus esquinas reales). */
  _cross(group, coords, color = 0x94a3b8, opacity = 0.5, z = 0.35) {
    if (!coords || coords.length < 3) return;
    const W = this.container.offsetWidth  || 800;
    const H = this.container.offsetHeight || 600;
    // Ordenar vértices por ángulo desde centroide → parear opuestos.
    // Robusto para rectángulos normales, clippeados y polígonos irregulares.
    const cx = coords.reduce((s, p) => s + p[0], 0) / coords.length;
    const cy = coords.reduce((s, p) => s + p[1], 0) / coords.length;
    const sorted = [...coords].sort((a, b) =>
      Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)
    );
    const n = sorted.length;
    const half = Math.floor(n / 2);
    for (const [i, j] of [[0, half], [1 % n, (half + 1) % n]]) {
      const pos = [sorted[i][0], -sorted[i][1], z, sorted[j][0], -sorted[j][1], z];
      const geo = new LineGeometry();
      geo.setPositions(pos);
      const mat = new LineMaterial({
        color, linewidth: 1.5,
        transparent: opacity < 1, opacity,
        resolution: new THREE.Vector2(W, H),
      });
      group.add(new Line2(geo, mat));
    }
  }

  /** Líneas de peldaño para escaleras */
  _stairLines(group, coords) {
    if (!coords || coords.length < 4) return;
    const mat = new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.65 });
    const steps = 7;
    for (let k = 1; k < steps; k++) {
      const t = k / steps;
      const lerp = (a, b) => a + (b - a) * t;
      const p1 = new THREE.Vector3(
        lerp(coords[0][0], coords[3 % coords.length][0]),
        -lerp(coords[0][1], coords[3 % coords.length][1]), 0.35);
      const p2 = new THREE.Vector3(
        lerp(coords[1][0], coords[2][0]),
        -lerp(coords[1][1], coords[2][1]), 0.35);
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([p1, p2]), mat));
    }
  }

  /** Etiqueta CSS2D */
  _label(group, coords, text, cssClass = '', z = 0.5) {
    if (!coords?.length || !text?.trim()) return;
    const [cx, cy] = cent(coords);
    const div = document.createElement('div');
    div.className = `three-label ${cssClass}`.trim();
    div.innerHTML = text.replace(/\n/g, '<br>');
    const obj = new CSS2DObject(div);
    obj.position.set(cx, -cy, z);
    group.add(obj);  // Añadir al grupo para que herede visibilidad
    this._labels.push(obj);
    return obj;
  }

  /** Etiqueta CSS2D en un punto explícito (no usa centroide). */
  _labelAt(group, x, y, text, cssClass = '', z = 0.5) {
    if (!text?.trim()) return;
    const div = document.createElement('div');
    div.className = `three-label ${cssClass}`.trim();
    div.innerHTML = text.replace(/\n/g, '<br>');
    const obj = new CSS2DObject(div);
    obj.position.set(x, -y, z);
    group.add(obj);
    this._labels.push(obj);
    return obj;
  }

  // ─────────────────────────────────────────────────────────
  //  CONVERSIÓN COORDENADAS (para herramienta de dibujo)
  // ─────────────────────────────────────────────────────────

  /** Pixel de pantalla → coordenadas de mapa [x, y] en metros. Solo válido en modo 2D. */
  screenToWorld(clientX, clientY) {
    const rect = this.container.getBoundingClientRect();
    const v = new THREE.Vector3(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
      0
    );
    v.unproject(this.camOrtho);
    return [v.x, -v.y]; // Three.js y invertido vs. coords de mapa
  }

  /** Coordenadas de mapa [x, y] → pixel de pantalla {x, y} relativo al container. */
  worldToScreen(mapX, mapY) {
    const rect = this.container.getBoundingClientRect();
    const v = new THREE.Vector3(mapX, -mapY, 0).project(this.camOrtho);
    return { x: (v.x + 1) / 2 * rect.width, y: (1 - v.y) / 2 * rect.height };
  }

  // ─────────────────────────────────────────────────────────
  //  GRUPOS
  // ─────────────────────────────────────────────────────────

  _group(name) {
    if (!this._groups[name]) {
      const g = new THREE.Group();
      g.name = name;
      this.scene.add(g);
      this._groups[name] = g;
    }
    return this._groups[name];
  }

  // ─────────────────────────────────────────────────────────
  //  CÁMARA AUTO-FIT
  // ─────────────────────────────────────────────────────────

  _fit2D(coords) {
    if (!coords?.length) return;
    const b    = bbox(coords);
    const W    = this.container.clientWidth;
    const H    = this.container.clientHeight;
    const asp  = W / Math.max(H, 1);
    const newF = Math.max(b.w / asp, b.h) * 1.30;   // 30% margen
    this._frustum = newF;

    this.camOrtho.left   = -newF * asp / 2;
    this.camOrtho.right  =  newF * asp / 2;
    this.camOrtho.top    =  newF / 2;
    this.camOrtho.bottom = -newF / 2;
    this.camOrtho.updateProjectionMatrix();

    this.camOrtho.position.set(b.cx, -b.cy, 200);
    this.camOrtho.lookAt(b.cx, -b.cy, 0);
    this.controls.target.set(b.cx, -b.cy, 0);
    this.controls.update();
  }

  _fit3D(coords) {
    if (!coords?.length) return;
    const b    = bbox(coords);
    const size = Math.max(b.w, b.h);
    const meta = this._metadata || {};
    const H    = (meta.pisos || 1) * (meta.altura_piso || 2.8);

    // Isométrica: desde frente-derecha-arriba
    this.camPersp.position.set(
      b.cx + size * 0.9,
      -b.cy - size * 0.9,
      H + size * 0.7
    );
    this.camPersp.lookAt(b.cx, -b.cy, H / 2);
    this.controls.target.set(b.cx, -b.cy, H / 2);
    this.controls.update();
  }

  // ─────────────────────────────────────────────────────────
  //  UTILIDADES
  // ─────────────────────────────────────────────────────────

  _aptColor(typ) {
    // Colores de departamentos para fondo oscuro — semi-opacos, tipo CAD
    if (typ === '1D')   return C.apt_1D;
    if (typ === '1D+E') return C.apt_1DE;
    if (typ === '2D')   return C.apt_2D;
    if (typ === '2D+E') return C.apt_2DE;
    return C.apt_3D;
  }

  _zoneColor(nombre) {
    const s = String(nombre || '');
    const u = s.toUpperCase();
    if (u.includes('CIRCUL')) return C.zona_circ;
    if (u.includes('COCINA')) return C.zona_cocina;
    if (u.includes('BAÑO') || u.includes('BANO') || u === 'WC') return C.zona_bano;
    if (u.includes('LAVANDER')) return C.zona_lav;
    if (u.includes('ESCRITORIO') || s.includes('(+E)')) return C.zona_esc;
    if (u.includes('DORM')) return C.zona_dorm;
    if (u.includes('ESTAR') || u.includes('COMEDOR')) return C.zona_estar;
    return C.zona_circ;
  }

  // ─────────────────────────────────────────────────────────
  //  ICONOS DE NAVEGACIÓN 3D
  // ─────────────────────────────────────────────────────────

  _buildNavButtons() {
    // Solo zoom + reset (la rotación la gestiona el nav cube)
    const overlay = document.createElement('div');
    overlay.id = 'nav3d-overlay';
    Object.assign(overlay.style, {
      position: 'absolute', bottom: '16px', right: '16px',
      zIndex: '35', display: 'none',
      flexDirection: 'column', gap: '4px',
      pointerEvents: 'auto', userSelect: 'none',
    });

    const BASE_BTN = [
      'width:32px', 'height:32px', 'border-radius:7px',
      'border:1px solid rgba(0,0,0,0.18)', 'background:rgba(255,255,255,0.93)',
      'cursor:pointer', 'font-size:15px', 'display:flex',
      'align-items:center', 'justify-content:center',
      'box-shadow:0 1px 4px rgba(0,0,0,0.18)',
      'color:#1e293b', 'transition:background 0.12s',
      'padding:0', 'outline:none',
    ].join(';');

    const mkBtn = (icon, title, action) => {
      const b = document.createElement('button');
      b.title = title; b.innerHTML = icon; b.style.cssText = BASE_BTN;
      b.addEventListener('click', e => { e.stopPropagation(); action(); });
      b.addEventListener('mouseenter', () => b.style.background = 'rgba(226,232,240,0.97)');
      b.addEventListener('mouseleave', () => b.style.background = 'rgba(255,255,255,0.93)');
      return b;
    };

    overlay.appendChild(mkBtn('+', 'Acercar',      () => this._zoom(0.8)));
    overlay.appendChild(mkBtn('−', 'Alejar',        () => this._zoom(1.25)));
    overlay.appendChild(mkBtn('⌂', 'Vista inicial', () => this._resetView()));

    this.container.appendChild(overlay);
    this._navOverlay = overlay;
  }

  // ─────────────────────────────────────────────────────────
  //  CUBO DE NAVEGACIÓN (View Cube)
  // ─────────────────────────────────────────────────────────

  _buildNavCube() {
    const SZ = 90;
    const canvas = document.createElement('canvas');
    canvas.width  = SZ * (window.devicePixelRatio || 1);
    canvas.height = SZ * (window.devicePixelRatio || 1);
    Object.assign(canvas.style, {
      position: 'absolute', top: '10px', right: '10px',
      width: SZ + 'px', height: SZ + 'px',
      zIndex: '36', cursor: 'pointer', display: 'none',
      borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.30)',
    });

    const gRend = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    gRend.setPixelRatio(window.devicePixelRatio || 1);
    gRend.setSize(SZ, SZ, false);
    gRend.setClearColor(0x1e293b, 0.78);

    const gScene = new THREE.Scene();
    gScene.add(new THREE.AmbientLight(0xffffff, 2.0));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(2, 2, 4); gScene.add(dl);

    // Textura de cara
    const _faceTex = (label, bg) => {
      const c = document.createElement('canvas');
      c.width = c.height = 128;
      const ctx = c.getContext('2d');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, 128, 128);
      ctx.strokeStyle = 'rgba(148,163,184,0.9)';
      ctx.lineWidth = 6; ctx.strokeRect(3, 3, 122, 122);
      ctx.fillStyle = '#f1f5f9';
      ctx.font = 'bold 24px Inter,system-ui,sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, 64, 64);
      return new THREE.CanvasTexture(c);
    };

    // Orden BoxGeometry: +X, -X, +Y, -Y, +Z, -Z  (Z-up: +Z=TOP)
    const FACES = [
      { label: 'DER',    bg: '#1e3a5f', view: 'right'  },  // +X
      { label: 'IZQ',    bg: '#1e3a5f', view: 'left'   },  // -X
      { label: 'FONDO',  bg: '#14532d', view: 'back'   },  // +Y
      { label: 'FRENTE', bg: '#14532d', view: 'front'  },  // -Y
      { label: 'TOP',    bg: '#3b0764', view: 'top'    },  // +Z
      { label: 'BASE',   bg: '#1c1917', view: 'bottom' },  // -Z
    ];
    const mats = FACES.map(f =>
      new THREE.MeshStandardMaterial({ map: _faceTex(f.label, f.bg), side: THREE.DoubleSide })
    );

    const cube  = new THREE.Mesh(new THREE.BoxGeometry(1.55, 1.55, 1.55), mats);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.57, 1.57, 1.57)),
      new THREE.LineBasicMaterial({ color: 0x94a3b8, linewidth: 1 })
    );
    gScene.add(cube); gScene.add(edges);

    const gCam = new THREE.PerspectiveCamera(36, 1, 0.1, 50);
    gCam.up.set(0, 0, 1);

    // ── Drag-to-orbit + click-to-snap ──────────────────────
    const raycaster = new THREE.Raycaster();
    let _drag = null;

    canvas.addEventListener('pointerdown', e => {
      canvas.setPointerCapture(e.pointerId);
      _drag = { x: e.clientX, y: e.clientY, totalMove: 0, lastE: e };
      e.preventDefault();
    });

    canvas.addEventListener('pointermove', e => {
      if (!_drag) return;
      const dx = e.clientX - _drag.x;
      const dy = e.clientY - _drag.y;
      _drag.totalMove += Math.abs(dx) + Math.abs(dy);
      // Arrastrar cubo = orbitar cámara principal
      // dx>0 (derecha) → cámara gira a la derecha del edificio
      // dy>0 (abajo)   → cámara baja (más frontal)
      this._orbit(dx * 0.013, dy * 0.013);
      _drag.x = e.clientX;
      _drag.y = e.clientY;
      _drag.lastE = e;
    });

    canvas.addEventListener('pointerup', e => {
      if (!_drag) return;
      const wasDrag = _drag.totalMove > 5;
      const lastE = _drag.lastE;
      _drag = null;
      if (wasDrag) return;          // fue drag → no hacer snap
      // Click: determinar cara y hacer snap
      const rect = canvas.getBoundingClientRect();
      const mx = ((lastE.clientX - rect.left) / rect.width)  * 2 - 1;
      const my = -((lastE.clientY - rect.top)  / rect.height) * 2 + 1;
      raycaster.setFromCamera(new THREE.Vector2(mx, my), gCam);
      const hits = raycaster.intersectObject(cube);
      if (!hits.length) return;
      const fi = Math.floor(hits[0].faceIndex / 2);
      this._snapView(FACES[fi]?.view || 'front');
    });

    this.container.appendChild(canvas);
    this._navCube = { canvas, renderer: gRend, scene: gScene, camera: gCam, cube, edges };
  }

  _syncNavCube() {
    const nc = this._navCube;
    if (!nc) return;
    // Dirección cámara principal → posiciona cámara del gizmo en el mismo ángulo
    const dir = this.camPersp.position.clone()
      .sub(this.controls.target).normalize().multiplyScalar(5);
    nc.camera.position.copy(dir);
    nc.camera.up.copy(this.camPersp.up);
    nc.camera.lookAt(0, 0, 0);
    nc.renderer.render(nc.scene, nc.camera);
  }

  _snapView(face) {
    if (!this._geometria?.lote?.coords) return;
    const b   = bbox(this._geometria.lote.coords);
    const meta = this._metadata || {};
    const H   = (meta.pisos || 1) * (meta.altura_piso || 2.8);
    const s   = Math.max(b.w, b.h);
    const cx = b.cx, cy = -b.cy, cz = H / 2;
    const d  = s * 1.9;
    const positions = {
      top:    [cx,     cy,     cz + d],
      bottom: [cx,     cy,     cz - d * 0.6],
      front:  [cx,     cy - d, cz * 0.6],
      back:   [cx,     cy + d, cz * 0.6],
      right:  [cx + d, cy,     cz * 0.6],
      left:   [cx - d, cy,     cz * 0.6],
    };
    const [px, py, pz] = positions[face] || positions.front;
    this.camPersp.position.set(px, py, pz);
    this.camPersp.lookAt(cx, cy, cz);
    this.controls.target.set(cx, cy, cz);
    this.controls.update();
  }

  _orbit(dAzimuth, dPolar) {
    // Z-up convention (edificio sube en Z).
    // dAzimuth → gira alrededor del eje Z mundial (izq/der, como AutoCAD 3D).
    // dPolar   → inclina la cámara cambiando su elevación.
    const target = this.controls.target.clone();
    const pos = this.camPersp.position.clone().sub(target);

    if (dAzimuth !== 0) {
      // Rotación alrededor de Z
      const cos = Math.cos(dAzimuth), sin = Math.sin(dAzimuth);
      const nx = pos.x * cos - pos.y * sin;
      const ny = pos.x * sin + pos.y * cos;
      pos.x = nx;
      pos.y = ny;
    }

    if (dPolar !== 0) {
      // Eje horizontal perpendicular a la dirección de la cámara y a Z-up
      const zUp = new THREE.Vector3(0, 0, 1);
      const right = pos.clone().cross(zUp).normalize();
      if (right.lengthSq() > 0.001) {
        const prevZ = pos.z;
        pos.applyAxisAngle(right, dPolar);
        // Evitar pasar por el cénit o bajar demasiado
        const minZ = 1.0;
        const maxZ = pos.length() * 0.97;
        if (pos.z < minZ || pos.z > maxZ) pos.z = prevZ; // revert vertical si sale de rango
      }
    }

    this.camPersp.position.copy(target.clone().add(pos));
    this.camPersp.lookAt(target);
    this.controls.update();
  }

  _zoom(scale) {
    const target = this.controls.target.clone();
    const pos = this.camPersp.position.clone().sub(target);
    const newLen = pos.length() * scale;
    if (newLen < 5 || newLen > 600) return;
    pos.setLength(newLen);
    this.camPersp.position.copy(target.clone().add(pos));
    this.controls.update();
  }

  _resetView() {
    if (this._geometria?.lote?.coords) this._fit3D(this._geometria.lote.coords);
  }

  // ─────────────────────────────────────────────────────────
  //  LIMPIEZA FINAL
  // ─────────────────────────────────────────────────────────

  dispose() {
    this._disposed = true;
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.clearScene();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
    if (this._navOverlay) { this._navOverlay.remove(); this._navOverlay = null; }
    if (this._navCube) {
      this._navCube.renderer.dispose();
      this._navCube.canvas.remove();
      this._navCube = null;
    }
  }
}
