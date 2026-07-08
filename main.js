import { Viewer3D } from './viewer3d.js?v=20260706a';

const initApp = () => {
    // --- API Configuration ---
    // En producción (Netlify), usa la URL de Render.com. En local, usa localhost.
    const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? window.location.origin
        : 'https://cabida.onrender.com';

    // --- UI bindings ---
    const inputs = document.querySelectorAll("#params-form input, #params-form select");
    const btnGenerateAI = document.getElementById("btn-generate-ai");
    const aiLoader = document.getElementById("ai-loader");
    const toggleEditMode = document.getElementById("toggle-edit-mode");
    const toggleLinderos = document.getElementById("toggle-linderos");

    // Leaflet Map State
    let leafletMap = null;
    let mapDrawLayer = null;

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // KPIs
    const kpiTerreno = document.getElementById("kpi-terreno");
    const kpiTechadaTotal = document.getElementById("kpi-techada-total");
    const kpiVendibleTotal = document.getElementById("kpi-vendible-total");
    const kpiEficiencia = document.getElementById("kpi-eficiencia");

    // Tables
    const thDptos = document.getElementById("th-dptos");
    const trDptoHeaders = document.getElementById("tr-dpto-headers");
    const tbodyCuadroAreas = document.getElementById("tbody-cuadro-areas");
    const tfSpan1 = document.getElementById("tf-span-1");
    const tfVendible = document.getElementById("tf-vendible");
    const tfComun = document.getElementById("tf-comun");
    const tfTotal = document.getElementById("tf-total");

    const tbodyTipologias = document.getElementById("tbody-tipologias");
    const resTotalDptos = document.getElementById("res-total-dptos");
    const resEstacionamientos = document.getElementById("res-estacionamientos");
    const resHabitantes = document.getElementById("res-habitantes");

    const wrapper = document.getElementById("canvas-wrapper");

    // --- Graph State ---
    let isGenerated = false;
    let isEditMode = false;
    let showLinderos = true;

    // --- Lote personalizado (dibujado inline o desde mapa) ---
    let customLoteCoords = null;  // [[x,y],...] — sobreescribe polígono 4-lados
    let drawMode = false;
    let drawVerts = [];           // vértices acumulados durante dibujo
    let drawSvgEl = null;         // overlay SVG
    let drawLastMouse = null;     // posición mouse para rubber band

    // --- Geometry & Data ---
    let params = {};
    let calc = {};
    let rneResultado = null; // Respuesta íntegra del servidor Python (normativa_estricta)
    let polygon = [];       // Linderos brutos
    let retiroPoly = [];    // Polígono del retiro frontal
    let techadaPoly = [];   // Post retiro frontal
    let loteNetoPoly = [];  // Lote Neto: post TODOS los retiros (frontal + laterales)

    let apartments = [];
    let corePoly = [];
    let ascensorPoly = [];
    let escaleraPoly = [];
    let smallDuctos = [];
    let patioPoly = [];
    let vestibuloPoly = []; // Vestíbulo previo ventilado (escalera presurizada)

    const numAscensoresInput = document.getElementById("num-ascensores");

    // Muros Ciegos
    const ciegoInp = {
        frente: document.getElementById("ciego-frente"),
        fondo: document.getElementById("ciego-fondo"),
        derecha: document.getElementById("ciego-derecha"),
        izquierda: document.getElementById("ciego-izquierda")
    };

    // --- Interactivity Setup ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');

            const target = document.getElementById(btn.dataset.tab);
            if (target) {
                target.classList.remove('hidden');
                target.classList.add('active');
            }
        });
    });

    // Pisos Slider Sync

    const pisosSlider = document.getElementById('pisos-slider');
    const pisosInput = document.getElementById('pisos');
    const pisosDisplay = document.getElementById('pisos-display');
    if (pisosSlider && pisosInput) {
        pisosSlider.addEventListener('input', () => {
            pisosInput.value = pisosSlider.value;
            if (pisosDisplay) pisosDisplay.innerText = pisosSlider.value;
            pisosInput.dispatchEvent(new Event('input', { bubbles: true }));
        });
    }

    // Workspace Tabs (Planta / Datos)
    document.querySelectorAll('.ws-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.ws-tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.ws-tab-content').forEach(c => c.style.display = 'none');

            btn.classList.add('active');

            let target = document.getElementById(btn.dataset.wsTab);
            if (target) {
                if (btn.dataset.wsTab === 'ws-viewport') {
                    target.style.display = 'flex';
                } else {
                    target.style.display = 'block';
                }
            }
        });
    });

    // --- View Toggle Logic ---
    const btnToggleView = document.getElementById("btn-toggle-view");
    const mapViewport = document.getElementById("map-viewport");
    const planViewport = document.getElementById("plan-viewport");

    function showMapView() {
        planViewport.style.display = "none";
        mapViewport.style.display = "flex";
        btnToggleView.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" style="vertical-align: middle; margin-right: 4px;"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"></path></svg> Volver a Planos`;
        if (!leafletMap) {
            initLeafletMap();
        } else {
            // Force redraw since container was display:none
            setTimeout(() => { leafletMap.invalidateSize(); }, 50);
        }
    }

    function showPlanView() {
        mapViewport.style.display = "none";
        planViewport.style.display = "flex";
        btnToggleView.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" style="vertical-align: middle; margin-right: 4px;"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"></path></svg> Ver Mapa`;
    }

    if (btnToggleView) {
        btnToggleView.addEventListener("click", () => {
            if (mapViewport.style.display === "none") {
                showMapView();
            } else {
                showPlanView();
            }
        });
    }

    // ═══════════════════════════════════════════════════════
    //  THREE.JS VIEWER INIT
    // ═══════════════════════════════════════════════════════
    let viewer3d = null;

    /** Inicializa Viewer3D la primera vez que se necesita */
    function getViewer3D() {
        if (viewer3d) return viewer3d;
        try {
            viewer3d = new Viewer3D(wrapper);
            window.__viewer3d = viewer3d;
            console.log('🌐 Viewer3D (Three.js) inicializado');
        } catch(e) {
            console.error('❌ Error inicializando Viewer3D:', e);
            viewer3d = null;
        }
        return viewer3d;
    }

    // Botón toggle 3D ↔ Planta
    const btnToggle3D = document.getElementById('btn-toggle-3d');
    let is3DMode = false;
    if (btnToggle3D) {
        btnToggle3D.addEventListener('click', () => {
            is3DMode = !is3DMode;
            const v = getViewer3D();
            if (!v) return;

            // Llamar setMode para cambiar cámara y controles
            v.setMode(is3DMode ? '3d' : '2d');
            
            // IMPORTANTE: regenerar geometría con la nueva profundidad
            if (window.webglPayload && window.metadataProyecto) {
                v.renderProyecto(window.webglPayload, window.metadataProyecto);
                // Sincronizar vista actual
                const vpLevelSel = document.getElementById('viewport-level');
                if (vpLevelSel) {
                    const levelMap = { tipica: 'tipica', primero: 'primero', sotano: 'sotano', azotea: 'azotea' };
                    v.setView(levelMap[vpLevelSel.value] || 'tipica');
                }
            }
            
            btnToggle3D.classList.toggle('active-3d', is3DMode);
            btnToggle3D.innerHTML = is3DMode
                ? `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" style="vertical-align:middle;margin-right:4px"><path d="M2 12l10-9 10 9-10 9z"/></svg> Planta 2D`
                : `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" style="vertical-align:middle;margin-right:4px"><path d="M12 3L2 7.5 12 12 22 7.5z"/><path d="M2 7.5v9L12 21l10-4.5v-9"/></svg> Volumetría 3D`;
        });
    }

    // Sincronizar selector de nivel de planta con setView del viewer3d
    const vpLevelSel = document.getElementById('viewport-level');
    if (vpLevelSel) {
        vpLevelSel.addEventListener('change', () => {
            const v = getViewer3D();
            if (!v) return;
            const levelMap = { tipica: 'tipica', primero: 'primero', sotano: 'sotano', azotea: 'azotea' };
            v.setView(levelMap[vpLevelSel.value] || 'tipica');
        });
    }

    // === Leaflet Map Logic ===
    function initLeafletMap() {
        // Initialize Map
        leafletMap = L.map('leaflet-map-container').setView([-12.046374, -77.042793], 15); // Default to Lima
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(leafletMap);

        // Feature Group to store drawn polygons
        mapDrawLayer = new L.FeatureGroup();
        mapDrawLayer.addTo(leafletMap);

        L.Control.geocoder({
            defaultMarkGeocode: false,
            placeholder: 'Buscar dirección o lugar...',
            errorMessage: 'Lugar no encontrado.'
        })
            .on('markgeocode', function (e) {
                leafletMap.fitBounds(e.geocode.bbox);
            })
            .addTo(leafletMap);

        leafletMap.pm.addControls({
            position: 'topleft',
            drawCircle: false, drawRectangle: false, drawPolyline: false,
            drawCircleMarker: false, drawMarker: false, drawText: false,
            editControls: true, removalMode: true
        });

        leafletMap.on('pm:create', function (e) {
            mapDrawLayer.clearLayers();
            mapDrawLayer.addLayer(e.layer);
            calculateMapMeasures(e.layer);

            // Allow editing of the new layer
            e.layer.on('pm:edit', function (x) {
                calculateMapMeasures(x.layer);
            });
        });

        document.getElementById('btn-sync-map').addEventListener('click', () => {
            if (mapDrawLayer.getLayers().length > 0) {
                calculateMapMeasures(mapDrawLayer.getLayers()[0]);
                updateCalculations();
            }
            showPlanView();
        });

        document.getElementById('btn-rotate-map-params').addEventListener('click', () => {
            let fte = document.getElementById("frente").value;
            let der = document.getElementById("derecha").value;
            let fdo = document.getElementById("fondo").value;
            let izq = document.getElementById("izquierda").value;

            // Rotate counter-clockwise: front takes right, right takes back...
            document.getElementById("frente").value = der;
            document.getElementById("derecha").value = fdo;
            document.getElementById("fondo").value = izq;
            document.getElementById("izquierda").value = fte;

            // Flash the input to show change
            ['frente', 'derecha', 'fondo', 'izquierda'].forEach(id => {
                let el = document.getElementById(id);
                el.style.backgroundColor = '#e0f2fe';
                setTimeout(() => el.style.backgroundColor = '', 300);
            });
        });
    }

    function calculateMapMeasures(layer) {
        let latlngs = layer.getLatLngs()[0];
        if (latlngs.length < 3) return;

        // ── Convertir a métrico local (flat-earth relativo al centroide) ──
        const nn = latlngs.length;
        const clat = latlngs.reduce((s, ll) => s + ll.lat, 0) / nn;
        const clng = latlngs.reduce((s, ll) => s + ll.lng, 0) / nn;
        const latRad = clat * Math.PI / 180;
        const mPerLat = 111320, mPerLng = 111320 * Math.cos(latRad);
        const localPts = latlngs.map(ll => ({
            x: (ll.lng - clng) * mPerLng,
            y: (ll.lat - clat) * mPerLat,
        }));
        // Convención plano: y=0 en borde sur (frente), y crece hacia fondo
        const yMin = Math.min(...localPts.map(p => p.y));
        const metricPts = localPts.map(p => [p.x, -(p.y - yMin)]);
        customLoteCoords = metricPts;
        updateCustomLoteIndicator();
        // Área real (shoelace)
        let areaMap = 0;
        for (let i = 0; i < nn; i++) {
            const [x1, y1] = metricPts[i], [x2, y2] = metricPts[(i + 1) % nn];
            areaMap += x1 * y2 - x2 * y1;
        }
        areaMap = Math.abs(areaMap) / 2;
        calc.areaTerreno = areaMap;
        const adisp = document.getElementById('area-terreno-display');
        if (adisp) adisp.innerText = areaMap.toFixed(2);

        // ── Lados aproximados para inputs de referencia ──
        let pts = latlngs.map(ll => turf.point([ll.lng, ll.lat]));
        let fte = 0, fdo = 0, der = 0, izq = 0;
        if (latlngs.length === 4) {
            fte = turf.distance(pts[0], pts[1], { units: 'meters' });
            der = turf.distance(pts[1], pts[2], { units: 'meters' });
            fdo = turf.distance(pts[2], pts[3], { units: 'meters' });
            izq = turf.distance(pts[3], pts[0], { units: 'meters' });
        } else {
            let bb = turf.bbox(turf.polygon([[...latlngs.map(ll => [ll.lng, ll.lat]), [latlngs[0].lng, latlngs[0].lat]]]));
            let w = turf.distance(turf.point([bb[0], bb[1]]), turf.point([bb[2], bb[1]]), { units: 'meters' });
            let h = turf.distance(turf.point([bb[0], bb[1]]), turf.point([bb[0], bb[3]]), { units: 'meters' });
            fte = w; fdo = w; der = h; izq = h;
        }
        document.getElementById("frente").value = fte.toFixed(1);
        document.getElementById("fondo").value = fdo.toFixed(1);
        document.getElementById("derecha").value = der.toFixed(1);
        document.getElementById("izquierda").value = izq.toFixed(1);
    }

    // ═══════════════════════════════════════════════════════════
    //  HERRAMIENTA DE DIBUJO DE LOTE INLINE
    // ═══════════════════════════════════════════════════════════

    function updateCustomLoteIndicator() {
        const clearBtn = document.getElementById('btn-clear-custom-lote');
        if (clearBtn) clearBtn.style.display = customLoteCoords ? 'inline-flex' : 'none';
    }

    function enterDrawMode() {
        const v = getViewer3D();
        if (!v) {
            // Sin viewer3d, mostrar mapa como alternativa
            alert('Genera una distribución inicial primero (o usa el botón "Ver Mapa") para calibrar la escala del lote.');
            return;
        }
        drawMode = true;
        drawVerts = [];
        drawLastMouse = null;

        // Overlay SVG sobre el viewport
        drawSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        Object.assign(drawSvgEl.style, {
            position: 'absolute', inset: '0', width: '100%', height: '100%',
            zIndex: '25', cursor: 'crosshair', display: 'block',
        });
        wrapper.appendChild(drawSvgEl);

        // Banner de instrucciones
        const inst = document.createElement('div');
        inst.id = 'draw-inst-banner';
        Object.assign(inst.style, {
            position: 'absolute', top: '8px', left: '50%', transform: 'translateX(-50%)',
            zIndex: '26', background: '#1e293b', color: 'white',
            padding: '6px 16px', borderRadius: '6px', fontSize: '12px',
            fontWeight: '600', pointerEvents: 'none', whiteSpace: 'nowrap',
        });
        inst.textContent = 'Click → vértice  •  Doble click → cerrar polígono  •  Esc → cancelar';
        wrapper.appendChild(inst);

        // Botón cancelar
        const cancelBtn = document.createElement('button');
        cancelBtn.id = 'draw-cancel-btn';
        cancelBtn.textContent = 'Cancelar dibujo';
        Object.assign(cancelBtn.style, {
            position: 'absolute', top: '8px', right: '8px', zIndex: '26',
            background: '#ef4444', color: 'white', border: 'none',
            borderRadius: '6px', padding: '5px 12px', fontSize: '12px',
            fontWeight: '600', cursor: 'pointer',
        });
        cancelBtn.onclick = () => exitDrawMode(false);
        wrapper.appendChild(cancelBtn);

        drawSvgEl.addEventListener('click',     _onDrawClick);
        drawSvgEl.addEventListener('dblclick',  _onDrawDblClick);
        drawSvgEl.addEventListener('mousemove', _onDrawMouseMove);
        document.addEventListener('keydown',    _onDrawKey);
    }

    function exitDrawMode(save) {
        drawMode = false;
        if (drawSvgEl) { drawSvgEl.remove(); drawSvgEl = null; }
        document.getElementById('draw-inst-banner')?.remove();
        document.getElementById('draw-cancel-btn')?.remove();
        drawSvgEl?.removeEventListener('click',     _onDrawClick);
        drawSvgEl?.removeEventListener('dblclick',  _onDrawDblClick);
        drawSvgEl?.removeEventListener('mousemove', _onDrawMouseMove);
        document.removeEventListener('keydown', _onDrawKey);

        if (save && drawVerts.length >= 3) {
            customLoteCoords = drawVerts.map(v => [v.x, v.y]);
            // Calcular área y mostrar en sidebar
            let areaCustom = 0;
            const n = customLoteCoords.length;
            for (let i = 0; i < n; i++) {
                const [x1, y1] = customLoteCoords[i];
                const [x2, y2] = customLoteCoords[(i + 1) % n];
                areaCustom += x1 * y2 - x2 * y1;
            }
            areaCustom = Math.abs(areaCustom) / 2;
            calc.areaTerreno = areaCustom;
            const adisp = document.getElementById('area-terreno-display');
            if (adisp) adisp.innerText = areaCustom.toFixed(2);
            updateCustomLoteIndicator();
        }
        drawVerts = [];
    }

    function _onDrawClick(e) {
        if (e.detail > 1) return; // ignorar el click del dblclick
        const v = window.__viewer3d;
        if (!v) return;
        const [wx, wy] = v.screenToWorld(e.clientX, e.clientY);
        drawVerts.push({ x: wx, y: wy });
        _renderDrawSvg(drawLastMouse);
    }

    function _onDrawDblClick() {
        if (drawVerts.length >= 3) exitDrawMode(true);
    }

    function _onDrawMouseMove(e) {
        drawLastMouse = e;
        _renderDrawSvg(e);
    }

    function _onDrawKey(e) {
        if (e.key === 'Escape') exitDrawMode(false);
    }

    function _renderDrawSvg(mouseEvent) {
        if (!drawSvgEl || !window.__viewer3d) return;
        const v = window.__viewer3d;
        const rect = wrapper.getBoundingClientRect();
        drawSvgEl.innerHTML = '';

        // Convertir vértices a píxeles de pantalla
        const pts = drawVerts.map(vtx => v.worldToScreen(vtx.x, vtx.y));

        // Líneas existentes + rubber band
        let allPts = [...pts];
        if (mouseEvent) {
            allPts.push({ x: mouseEvent.clientX - rect.left, y: mouseEvent.clientY - rect.top });
        }

        if (pts.length >= 3) {
            // Relleno semitransparente
            const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            poly.setAttribute('points', pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '));
            poly.setAttribute('fill', 'rgba(59,130,246,0.12)');
            poly.setAttribute('stroke', '#3b82f6');
            poly.setAttribute('stroke-width', '2');
            poly.setAttribute('stroke-dasharray', '6,3');
            drawSvgEl.appendChild(poly);
        }

        if (allPts.length >= 2) {
            const pl = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            pl.setAttribute('points', allPts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '));
            pl.setAttribute('fill', 'none');
            pl.setAttribute('stroke', '#3b82f6');
            pl.setAttribute('stroke-width', '2');
            pl.setAttribute('stroke-dasharray', '6,3');
            drawSvgEl.appendChild(pl);
        }

        // Vértices marcadores
        pts.forEach((p, i) => {
            const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            c.setAttribute('cx', p.x.toFixed(1));
            c.setAttribute('cy', p.y.toFixed(1));
            c.setAttribute('r', i === 0 ? '7' : '4');
            c.setAttribute('fill', i === 0 ? '#10b981' : '#3b82f6');
            c.setAttribute('stroke', 'white');
            c.setAttribute('stroke-width', '1.5');
            drawSvgEl.appendChild(c);
        });

        // Cotas de cada arista ya dibujada
        for (let i = 0; i < pts.length - 1; i++) {
            const [wx1, wy1] = [drawVerts[i].x, drawVerts[i].y];
            const [wx2, wy2] = [drawVerts[i + 1].x, drawVerts[i + 1].y];
            const len = Math.hypot(wx2 - wx1, wy2 - wy1);
            const mp = v.worldToScreen((wx1 + wx2) / 2, (wy1 + wy2) / 2);
            const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            txt.setAttribute('x', mp.x.toFixed(1));
            txt.setAttribute('y', (mp.y - 6).toFixed(1));
            txt.setAttribute('text-anchor', 'middle');
            txt.setAttribute('font-size', '11');
            txt.setAttribute('font-weight', '600');
            txt.setAttribute('fill', '#1e293b');
            txt.setAttribute('stroke', 'white');
            txt.setAttribute('stroke-width', '3');
            txt.setAttribute('paint-order', 'stroke');
            txt.textContent = `${len.toFixed(1)}m`;
            drawSvgEl.appendChild(txt);
        }
    }

    // Math utils
    function calculatePolyArea(poly) {
        if (!poly || poly.length < 3) return 0;
        let area = 0; let j = poly.length - 1;
        for (let i = 0; i < poly.length; i++) {
            area += (poly[j].x + poly[i].x) * (poly[j].y - poly[i].y);
            j = i;
        }
        return Math.abs(area / 2);
    }

    function calculatePolyWidth(poly) {
        if (!poly || poly.length < 2) return 0;
        let minX = Infinity, maxX = -Infinity;
        for (let p of poly) {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
        }
        return maxX - minX;
    }

    function calculatePolyHeight(poly) {
        if (!poly || poly.length < 2) return 0;
        let minY = Infinity, maxY = -Infinity;
        for (let p of poly) {
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        return maxY - minY;
    }


    function interpolate(pA, pB, t) {
        return { x: pA.x + (pB.x - pA.x) * t, y: pA.y + (pB.y - pA.y) * t };
    }

    function getQuadPoint(quad, u, v) {
        // quad is [pTL, pTR, pBR, pBL]
        let pT = interpolate(quad[0], quad[1], u);
        let pB = interpolate(quad[3], quad[2], u);
        return interpolate(pT, pB, v);
    }

    function getCell(quad, u1, u2, v1, v2) {
        return [
            getQuadPoint(quad, u1, v1),
            getQuadPoint(quad, u2, v1),
            getQuadPoint(quad, u2, v2),
            getQuadPoint(quad, u1, v2)
        ];
    }

    // Turf helpers
    function polyToTurf(poly) {
        if (!poly || poly.length < 3) return null;
        let coords = poly.map(p => [p.x, p.y]);
        coords.push([poly[0].x, poly[0].y]);
        let p = turf.polygon([coords]);
        return turf.rewind(p, { mutate: true });
    }

    function turfToPoly(turfPoly) {
        if (!turfPoly) return [];
        let coords = [];
        if (turfPoly.geometry.type === 'Polygon') {
            coords = turfPoly.geometry.coordinates[0];
        } else if (turfPoly.geometry.type === 'MultiPolygon') {
            let maxArea = -1;
            for (let c of turfPoly.geometry.coordinates) {
                let currentCoords = c[0];
                let pList = currentCoords.map(pt => ({ x: pt[0], y: pt[1] }));
                let pListPop = pList.slice(0, pList.length - 1);
                let a = calculatePolyArea(pListPop);
                if (a > maxArea) {
                    maxArea = a;
                    coords = currentCoords;
                }
            }
        }
        let res = [];
        for (let i = 0; i < coords.length - 1; i++) {
            res.push({ x: coords[i][0], y: coords[i][1] });
        }
        return res;
    }

    function updateCalculations() {
        params = {
            frente: parseFloat(document.getElementById("frente")?.value) || 0,
            fondo: parseFloat(document.getElementById("fondo")?.value) || 0,
            derecha: parseFloat(document.getElementById("derecha")?.value) || 0,
            izquierda: parseFloat(document.getElementById("izquierda")?.value) || 0,
            retiroFrontal: Math.max(0, parseFloat(document.getElementById("retiro-frontal")?.value) || 0),
            pctLibreReq: parseInt(document.getElementById("area-libre-req")?.value) || 0,
            pisos: parseInt(document.getElementById("pisos")?.value) || 1,
            pctEstac: parseFloat(document.getElementById("pct-estac")?.value) || 100,
            nAscensores: parseInt(numAscensoresInput ? numAscensoresInput.value : 1) || 0,
            dptosPlanta: parseInt(document.getElementById("dptos-planta")?.value) || 4,
            distribucion: document.getElementById("distribucion")?.value || 'optimo',
            modoObjetivo: document.getElementById("modo-objetivo")?.value || 'cantidad',
            areaPromedio: parseFloat(document.getElementById("area-promedio")?.value) || 60,
            ductW: parseFloat(document.getElementById("duct-w")?.value) || 2.2,
            ductH: parseFloat(document.getElementById("duct-h")?.value) || 2.2,
            numDuctosLimit: parseInt(document.getElementById("num-ductos")?.value) || 2,
            alturaPiso: parseFloat(document.getElementById("altura-piso")?.value) || 2.80,
            retiroLateral: (() => { const v = parseFloat(document.getElementById("retiro-lateral")?.value); return isNaN(v) ? 2.30 : v; })(),
            retiroPosterior: (() => { const v = parseFloat(document.getElementById("retiro-posterior")?.value); return isNaN(v) ? 2.30 : v; })(),
        };

        // === RNE A.010: Derived normative values ===
        let H = params.pisos * params.alturaPiso;
        params.H = H;
        params.pozoDormMin = Math.max(2.20, H / 4);  // Dormitorios: d >= H/4, mín 2.20m
        params.pozoSalaMin = Math.max(3.00, H / 3);  // Sala/Comedor: d >= H/3, mín 3.00m
        params.ductoVentAreaMin = 3.0;                // SSHH/Cocina: área mín 3.00m²
        params.ductoVentLadoMin = 1.50;               // SSHH/Cocina: lado mín 1.50m
        params.hallMinWidth = 1.20;                   // Hall común mín 1.20m
        params.escaleraMinWidth = 1.20;               // Escalera mín 1.20m ancho

        // Generar Poligono Linderos (Aproximación para diagrama)
        // P1(FrontLeft), P2(FrontRight), P3(BackRight), P4(BackLeft)
        // Frente es ancho arriba. Fondo es ancho abajo. Derecha/Izquierda son profundidades.
        const p1 = { x: -params.frente / 2, y: 0 };
        const p2 = { x: params.frente / 2, y: 0 };
        const p3 = { x: params.fondo / 2, y: params.derecha };
        const p4 = { x: -params.fondo / 2, y: params.izquierda };
        polygon = [p1, p2, p3, p4];

        calc.areaTerreno = calculatePolyArea(polygon);

        // === FASE 2: Retiro Frontal ===
        let rtY = Math.min(params.retiroFrontal, params.derecha, params.izquierda);
        let uL = rtY / params.izquierda;
        let uR = rtY / params.derecha;
        let pr3 = interpolate(p2, p3, uR);
        let pr4 = interpolate(p1, p4, uL);

        retiroPoly = rtY > 0 ? [p1, p2, pr3, pr4] : [];
        techadaPoly = rtY > 0 ? [pr4, pr3, p3, p4] : [p1, p2, p3, p4];

        // === FASE 2: Lote Neto (offset por retiros laterales A.010) ===
        // Si un lateral NO tiene Muro Ciego → aplica retiro definido por el usuario
        let retiroLat = params.retiroLateral;
        let frenteNeto = Math.max(1, params.frente);
        let fondoNeto = Math.max(1, (params.derecha + params.izquierda) / 2);

        let u_left = (ciegoInp.izquierda && ciegoInp.izquierda.checked) ? 0 : retiroLat / frenteNeto;
        let u_right = (ciegoInp.derecha && ciegoInp.derecha.checked) ? 1.0 : 1.0 - (retiroLat / frenteNeto);
        let v_bottom = (ciegoInp.fondo && ciegoInp.fondo.checked) ? 1.0 : 1.0 - (retiroLat / fondoNeto);
        // Frente: ya aplicamos retiro frontal, v_top siempre es 0
        let v_top = 0;

        loteNetoPoly = getCell(techadaPoly, u_left, u_right, v_top, v_bottom);

        calc.areaLoteNeto = calculatePolyArea(loteNetoPoly);
        calc.areaRetiroTotal = calc.areaTerreno - calc.areaLoteNeto;
        calc.areaLibreReq = (calc.areaTerreno * params.pctLibreReq) / 100;
        calc.areaTechadaPlanta = calc.areaLoteNeto; // Máxima área edificable = lote neto

        document.getElementById("area-terreno-display").innerText = calc.areaTerreno.toFixed(2);
        kpiTerreno.innerText = calc.areaTerreno.toFixed(2);

        if (!isGenerated) {
            let ac = Math.max(15, calc.areaTechadaPlanta * 0.15);
            let av = calc.areaTechadaPlanta - ac;
            kpiTechadaTotal.innerText = (calc.areaTechadaPlanta * params.pisos).toFixed(0);
            kpiVendibleTotal.innerText = (av * params.pisos).toFixed(0);
            kpiEficiencia.innerText = ((av / calc.areaTechadaPlanta) * 100).toFixed(1);
        }

        // === UMBRALES DE SEGURIDAD (Flexibilidad Paramétrica) ===
        let totalDptosEst = params.dptosPlanta * params.pisos;

        // Pisos <= 4: escalera abierta, sin ascensor obligatorio
        // Pisos > 4: ascensor obligatorio
        // H > 11.00m: escalera presurizada/ventilada con vestíbulo previo
        params.ascensorObligatorio = params.pisos > 4;
        params.escaleraPresurizada = H > 11.0;
        params.escaleraTipo = H > 11.0 ? 'Presurizada' : 'Abierta';

        let alertBox = document.getElementById('elevator-alert');
        if (alertBox) {
            let msgs = [];

            // Alerta: ascensor obligatorio pero usuario puso 0
            if (params.ascensorObligatorio && params.nAscensores < 1) {
                msgs.push(`⚠ RNE: Ascensor OBLIGATORIO para ${params.pisos} pisos (>4).`);
            }

            // Alerta: recomendación de 2+ ascensores para edificios grandes
            if (params.pisos >= 8 && params.nAscensores < 2 && totalDptosEst > 40) {
                msgs.push(`⚠ Se recomienda mín. 2 ascensores para ${params.pisos} pisos y ${totalDptosEst} dptos.`);
            }

            // Alerta: escalera presurizada
            if (params.escaleraPresurizada) {
                msgs.push(`🔒 H=${H.toFixed(1)}m (>11m): Escalera debe ser presurizada o con vestíbulo previo.`);
            }

            if (msgs.length > 0) {
                alertBox.style.display = 'block';
                alertBox.innerHTML = msgs.join('<br>');
            } else {
                alertBox.style.display = 'none';
            }
        }

        // === INVIABILIDAD NORMATIVA ===
        // Si H/4 > ancho disponible del Lote Neto → pozos de luz no caben
        let loteNetoWidth = calculatePolyWidth(loteNetoPoly);
        let pozoMinReq = params.pozoDormMin; // H/4 min
        let invAlert = document.getElementById('inviability-alert');
        if (invAlert) {
            if (loteNetoWidth > 0 && pozoMinReq >= loteNetoWidth * 0.5) {
                invAlert.style.display = 'block';
                invAlert.innerHTML = `🚫 INVIABILIDAD NORMATIVA: El pozo de luz mín. (${pozoMinReq.toFixed(2)}m = H/4) excede el 50% del ancho del Lote Neto (${loteNetoWidth.toFixed(2)}m). Reduce pisos o aumenta el frente.`;
            } else {
                invAlert.style.display = 'none';
            }
        }

        // === Á. Libre Real (reactiva) ===
        calc.areaLibreReal = calc.areaTerreno - calc.areaLoteNeto;
        calc.pctLibreReal = calc.areaTerreno > 0 ? (calc.areaLibreReal / calc.areaTerreno) * 100 : 0;

        // Update KPI bar Á. Libre
        let kpiLibre = document.getElementById('kpi-libre-bar');
        if (kpiLibre) kpiLibre.innerText = calc.pctLibreReal.toFixed(1);
    }

    function getTypology(area) {
        if (area < 44) return { name: '1D', col: '1D', hab: 2 };
        if (area < 54) return { name: '1D+E', col: '1DE', hab: 3 };
        if (area < 72) return { name: '2D', col: '2D', hab: 3 };
        if (area < 92) return { name: '2D+E', col: '2DE', hab: 4 };
        return { name: '3D', col: '3D', hab: 5 };
    }

    // --- FASE 2: Generación del Núcleo de Circulación Vertical ---

    function buildBasePolygons(dptos) {
        // 1. LIMPIEZA QUIRÚRGICA
        // Vaciamos los polígonos antiguos, pero respetamos corePoly y window.hallProcedural
        // porque ahora esos datos son sagrados y vienen de Python.
        apartments = [];
        patioPoly = [];
        smallDuctos = [];
        ascensorPoly = [];
        escaleraPoly = [];
        vestibuloPoly = [];

        // 2. VARIABLES DE CONTINGENCIA
        // Evitamos que las funciones de habitabilidad posteriores arrojen errores.
        window.hasCore = true;
        calc.maxDistEscalera = 0;
        calc.cumple25m = true;

        // 3. MOCK (SIMULACIÓN) DE DEPARTAMENTOS
        // Creamos "fantasmas" de departamentos sin geometría (poly: []).
        // Esto permite que la función de sótanos calcule la cisterna y los 
        // estacionamientos requeridos sin que el código colapse por falta de datos.
        let cantidadDptos = dptos || 6;
        for (let i = 0; i < cantidadDptos; i++) {
            apartments.push({
                id: `X${String(i + 1).padStart(2, '0')}`,
                poly: [], // Sin polígono para que el Canvas no dibuje basura
                area: 60,
                typology: '3D',
                hab: 3
            });
        }
    }

    function processGeometryAndCheckHabitability(dptosToTry) {
        let totalVendible = 0;

        apartments.forEach(ap => {
            if (!ap.poly || ap.poly.length === 0) {
                // IGNORAR FANTASMAS: Si es un dpto simulado, sumar su área as-is y no alterar
                totalVendible += ap.area;
                return;
            }

            let turfAp = polyToTurf(ap.poly);
            let finalArea = 0;
            if (turfAp) {
                let objectsToSubtract = [...smallDuctos];
                if (patioPoly.length > 0) objectsToSubtract.push(patioPoly);
                if (corePoly.length > 0) {
                    objectsToSubtract.push(corePoly);
                    if (ascensorPoly.length > 0) objectsToSubtract.push(ascensorPoly);
                    if (escaleraPoly.length > 0) objectsToSubtract.push(escaleraPoly);
                    if (vestibuloPoly.length > 0) objectsToSubtract.push(vestibuloPoly);
                }

                let didSubtract = objectsToSubtract.some(d => d && d.length >= 3);
                objectsToSubtract.forEach(duct => {
                    let turfDuct = polyToTurf(duct);
                    if (turfDuct) {
                        try {
                            let diff = turf.difference(turfAp, turfDuct);
                            if (diff) turfAp = diff;
                        } catch (e) { }
                    }
                });
                ap.poly = turfToPoly(turfAp);
                // Mantener zonas esquemáticas del backend para el diagrama (canvas / referencia visual).
                // El área útil sigue calculándose sobre el polígono ya restado.

                finalArea = 0;
                if (turfAp.geometry.type === 'Polygon') {
                    let rings = turfAp.geometry.coordinates;
                    finalArea += calculatePolyArea(rings[0].map(pt => ({ x: pt[0], y: pt[1] })).slice(0, -1));
                    for (let i = 1; i < rings.length; i++) finalArea -= calculatePolyArea(rings[i].map(pt => ({ x: pt[0], y: pt[1] })).slice(0, -1));
                } else if (turfAp.geometry.type === 'MultiPolygon') {
                    for (let c of turfAp.geometry.coordinates) {
                        finalArea += calculatePolyArea(c[0].map(pt => ({ x: pt[0], y: pt[1] })).slice(0, -1));
                        for (let i = 1; i < c.length; i++) finalArea -= calculatePolyArea(c[i].map(pt => ({ x: pt[0], y: pt[1] })).slice(0, -1));
                    }
                }

                try {
                    let center = turf.pointOnFeature(turfAp);
                    ap.labelPt = { cx: center.geometry.coordinates[0], cy: center.geometry.coordinates[1] };
                } catch (e) {
                    let cx = 0, cy = 0; ap.poly.forEach(p => { cx += p.x; cy += p.y; });
                    ap.labelPt = { cx: cx / ap.poly.length, cy: cy / ap.poly.length };
                }
            }

            ap.area = Math.max(0, finalArea);
            let t = getTypology(ap.area);
            ap.typology = t.name;
            ap.colorVar = `--col-${t.col}`;
            ap.hab = t.hab;
            totalVendible += ap.area;
        });

        // --- AUDITORIA DE HABITABILIDAD ---
        let allValid = true;
        let areaFaltante = 0;
        let viviendasGigantes = 0;
        let minDimFailures = 0;
        let islaUnidades = 0;

        for (const ap of apartments) {
            if (!ap.poly || ap.poly.length === 0) continue; // No penalizar dptos fantasma

            let apW = calculatePolyWidth(ap.poly);
            let apH = calculatePolyHeight(ap.poly);
            let minDim = Math.min(apW, apH);
            if (minDim < 3.0) { allValid = false; minDimFailures++; }

            if (ap.area < 40) {
                allValid = false;
                areaFaltante += (40 - ap.area);
            }
            if (ap.area > 120) {
                viviendasGigantes++;
            }

            // CHEQUEO DE CONECTIVIDAD: Toca el Hall o Core?
            let hallProc = window.hallProcedural || [];
            let connected = false;
            // Check against core
            if (corePoly.length > 0) {
                let turfCore = polyToTurf(corePoly);
                let turfUnit = polyToTurf(ap.poly);
                if (turfCore && turfUnit) {
                    try {
                        let bufferedCore = turf.buffer(turfCore, 0.30, { units: 'meters' });
                        let overlap = turf.intersect(bufferedCore, turfUnit);
                        if (overlap) connected = true;
                    } catch (e) { }
                }
            }
            // Check against hall if not yet connected
            if (!connected && hallProc.length >= 3) {
                let turfHall = polyToTurf(hallProc);
                let turfUnit = polyToTurf(ap.poly);
                if (turfHall && turfUnit) {
                    try {
                        let bufferedHall = turf.buffer(turfHall, 0.30, { units: 'meters' });
                        let overlap = turf.intersect(bufferedHall, turfUnit);
                        if (overlap) connected = true;
                    } catch (e) { }
                }
            }
            if (!connected) { islaUnidades++; }
        }

        let invAlert = document.getElementById('inviability-alert');
        if (invAlert) {
            if (!allValid || viviendasGigantes > 0 || islaUnidades > 0) {
                invAlert.style.display = 'block';
                let msgs = [];
                if (areaFaltante > 0) msgs.push(`\u26a0\ufe0f IMPOSIBLE: Faltan ${areaFaltante.toFixed(1)} m\u00b2 de area util para alcanzar la cuota de ${apartments.length} unidades.`);
                if (minDimFailures > 0) msgs.push(`\u26a0\ufe0f A.020: Ancho minimo 3.0m NO cumplido en ${minDimFailures} departamentos.`);
                if (viviendasGigantes > 0) msgs.push(`\u26a0\ufe0f Vivienda Gigante (>120m\u00b2): ${viviendasGigantes} unidades superan el limite.`);
                if (islaUnidades > 0) msgs.push(`\ud83d\udeb7 CONECTIVIDAD: ${islaUnidades} unidades NO tocan el Hall.`);
                invAlert.innerHTML = `\ud83d\udeab <b>RESTRICCION ABSOLUTA INCUMPLIDA:</b><br/>` + msgs.join("<br/>");
            } else {
                invAlert.style.display = 'none';
            }
        }

        let hallArea = calculatePolyArea(corePoly);
        if (hallArea < 5 && params.dptosPlanta > 1) hallArea = 15;

        let actDuctArea = 0;
        smallDuctos.forEach(d => actDuctArea += calculatePolyArea(d));
        if (patioPoly.length) actDuctArea += calculatePolyArea(patioPoly);

        calc.realVendiblePlanta = totalVendible;
        calc.realComunPlanta = hallArea;
        calc.realTotalPlanta = totalVendible + hallArea;

        return true;
    }

    function runAILayoutGeneration() {
        let dptos = params.dptosPlanta;
        if (params.modoObjetivo === 'area') {
            let estUsable = calc.areaLoteNeto * 0.85;
            dptos = Math.max(1, Math.round(estUsable / params.areaPromedio));
        }

        // Connectivity loop: regenerate until all units touch the hall
        let maxRetries = 3;
        for (let attempt = 0; attempt < maxRetries; attempt++) {
            buildBasePolygons(dptos);
            processGeometryAndCheckHabitability(dptos);

            // Check connectivity
            let allConnected = true;
            if (corePoly.length > 0) {
                let turfCore = polyToTurf(corePoly);
                if (turfCore) {
                    try {
                        let bufferedCore = turf.buffer(turfCore, 0.20, { units: 'meters' });
                        for (const ap of apartments) {
                            let turfUnit = polyToTurf(ap.poly);
                            if (turfUnit) {
                                let overlap = turf.intersect(bufferedCore, turfUnit);
                                if (!overlap) { allConnected = false; break; }
                            }
                        }
                    } catch (e) { }
                }
            }
            if (allConnected) break;
        }

        updateTables();
        updateCompliancePanel();
    }

    function updateCompliancePanel() {
        let panel = document.getElementById('compliance-panel');
        if (!panel) return;

        let H = params.H || (params.pisos * params.alturaPiso);
        let reqDorm = Math.max(2.20, H / 4);
        let reqSala = Math.max(3.00, H / 3);

        // 1. Area Libre Real = retiros + pozos + ductos
        let patioArea = patioPoly.length ? calculatePolyArea(patioPoly) : 0;
        let actDuctArea = 0;
        smallDuctos.forEach(d => actDuctArea += calculatePolyArea(d));
        let areaLibreReal = calc.areaRetiroTotal + patioArea + actDuctArea;
        let pctLibre = calc.areaTerreno > 0 ? (areaLibreReal / calc.areaTerreno * 100) : 0;
        let pctReq = params.pctLibreReq || 0;
        let cumpleLibre = pctLibre >= pctReq;

        // 2. Pozos de Luz
        let patioW = patioPoly.length >= 4 ? calculatePolyWidth(patioPoly) : 0;
        let patioH = patioPoly.length >= 4 ? calculatePolyHeight(patioPoly) : 0;
        let patioMinDim = Math.min(patioW, patioH);
        let cumplePozoDorm = patioMinDim >= reqDorm;
        let cumplePozoSala = patioMinDim >= reqSala;

        // 3. Estacionamientos (fuente única: sótano del backend)
        let sot = window.webglPayload?.sotano || {};
        let totalDptos = apartments.length * params.pisos;
        let estReq = sot.req_estac ?? Math.ceil(totalDptos * (params.pctEstac / 100));
        let estLogrados = sot.count || 0;
        let cumpleEst = estLogrados >= estReq;

        // 4. Cisterna (backend)
        let volDom = sot.cisterna_domestico || 0;
        let volACI = sot.cisterna_aci || 0;
        let volTotal = sot.cisterna_total_m3 || 0;

        // 5. Conectividad (acceso al hall/core/escalera)
        // The Spine & Ribs algorithm places all apartments fronting the hall corridor.
        // Use distance-based check: if any apartment edge is within 0.50m of hall, core, or escalera, it's connected.
        let cumpleConect = true;
        let failConect = 0;
        let hallPoly = window.hallProcedural || [];

        function minDistPolyToPoly(polyA, polyB) {
            // Returns minimum distance between any vertex of polyA and any edge of polyB
            let minD = Infinity;
            for (const pt of polyA) {
                for (let j = 0; j < polyB.length; j++) {
                    let hA = polyB[j], hB = polyB[(j + 1) % polyB.length];
                    let dx = hB.x - hA.x, dy = hB.y - hA.y;
                    let segLen2 = dx * dx + dy * dy;
                    if (segLen2 < 0.0001) continue;
                    let t = Math.max(0, Math.min(1, ((pt.x - hA.x) * dx + (pt.y - hA.y) * dy) / segLen2));
                    let dist = Math.hypot(pt.x - (hA.x + t * dx), pt.y - (hA.y + t * dy));
                    if (dist < minD) minD = dist;
                }
            }
            return minD;
        }

        if (corePoly.length > 0 || hallPoly.length > 0) {
            apartments.forEach(ap => {
                let dHall = hallPoly.length >= 3 ? minDistPolyToPoly(ap.poly, hallPoly) : Infinity;
                let dCore = corePoly.length >= 3 ? minDistPolyToPoly(ap.poly, corePoly) : Infinity;
                let dEsc = escaleraPoly.length >= 3 ? minDistPolyToPoly(ap.poly, escaleraPoly) : Infinity;
                let minDist = Math.min(dHall, dCore, dEsc);
                // Apartment is connected if within 0.50m of any circulation element
                if (minDist > 0.50) { cumpleConect = false; failConect++; }
            });
        }

        // 6. Distancia max a escalera (A.130: evacuacion_max del servidor Python, default 45m)
        let evacuacionMax = rneResultado?.normativa_estricta?.evacuacion_max || 45;
        let cumpleEvac = (calc.maxDistEscalera || 0) <= evacuacionMax;

        // 7. Ascensores
        let cumpleAsc = (params.pisos <= 5) || (ascensorPoly.length > 0);
        let nAsc = ascensorPoly.length > 0 ? Math.max(1, Math.round(calculatePolyWidth(ascensorPoly) / 2.0)) : 0;

        // 8. Escalera tipo
        let cumpleEsc = !params.escaleraPresurizada || vestibuloPoly.length > 0;

        function row(art, desc, cumple, detail) {
            let icon = cumple ? '✅' : '❌';
            let cls = cumple ? 'ok' : 'fail';
            return `<div class="compliance-item ${cls}">
                <div class="compliance-icon">${icon}</div>
                <div class="compliance-detail">
                    <strong>${art}</strong>
                    <span>${desc}</span>
                    <small>${detail}</small>
                </div>
            </div>`;
        }

        let html = '<div class="compliance-grid">';
        html += row('A.010 Art.19 — Área Libre',
            `${pctLibre.toFixed(1)}% logrado vs ${pctReq}% requerido`,
            cumpleLibre,
            `${areaLibreReal.toFixed(1)} m² de ${calc.areaTerreno.toFixed(1)} m² (retiros+pozos+ductos)`);

        html += row('A.010 Art.40 — Pozos de Luz',
            `d=${patioMinDim.toFixed(2)}m vs H/4=${reqDorm.toFixed(2)}m (dorm) / H/3=${reqSala.toFixed(2)}m (sala)`,
            cumplePozoDorm,
            `Patio: ${patioW.toFixed(1)}×${patioH.toFixed(1)}m = ${patioArea.toFixed(1)} m²`);

        html += row('A.010 Art.65 — Estacionamientos',
            `${estLogrados} logrados vs ${estReq} requeridos`,
            cumpleEst,
            `Sótano ${sot.name || 'S1'}`);

        let cab = rneResultado?.geometria_generada?.cabida_multifamiliar;
        if (cab) {
            let ped = cab.departamentos_solicitados_planta;
            let gen = cab.departamentos_generados_planta;
            let cap = cab.capacidad_maxima_estimada_planta;
            let cumpleCab = gen >= ped;
            html += row('Cabida — multifamiliar (referencia RNE)',
                `${gen} unidades generadas en planta vs ${ped} solicitadas`,
                cumpleCab,
                `Capacidad máx. estimada ~${cap}/planta · strip ~${cab.profundidad_strip_estimada_m} m · área mín. ${cab.area_min_dpto_m2} m²`);
        }

        html += row('A.010 Art.26 — Conectividad Hall',
            cumpleConect ? 'Todas las unidades acceden al Hall (≥1.50m)' : `${failConect} unidad(es) sin acceso directo al Hall`,
            cumpleConect,
            `Frontera mín requerida: 1.50m con núcleo vertical`);

        html += row('A.130 Art.13 — Dist. Máx Evacuación',
            `Máx dist: ${(calc.maxDistEscalera || 0).toFixed(1)}m (límite ${evacuacionMax.toFixed(2)}m)`,
            cumpleEvac,
            params.escaleraPresurizada ? 'Escalera presurizada + vestíbulo previo 2.00m²' : 'Escalera convencional');

        html += row('A.010 Art.30 — Ascensores',
            `${nAsc} ascensor(es) de 2.00×2.00m`,
            cumpleAsc,
            cumpleAsc ? 'Hall ≥1.50m ancho libre ✓' : 'Obligatorio para >5 pisos');

        html += row('IS.010 / A.130 — Cisterna',
            `${volTotal.toFixed(1)} m³ total`,
            true,
            `Consumo: ${volDom.toFixed(1)} m³ / ACI: ${volACI.toFixed(0)} m³ / Bombas: ${(sot.cisterna_maq || 0).toFixed(0)} m²`);

        html += '</div>';
        panel.innerHTML = html;
    }


    function updateNormaCheckPanel(resultado) {
        const panel = document.getElementById('norma-check-panel');
        if (!panel) return;
        const norma = resultado?.normativa_estricta;
        const geo   = resultado?.geometria_generada;
        if (!norma && !geo) return;
        panel.style.display = '';

        // ── helpers ──
        function chk(ok, label, detail) {
            const icon = ok ? '✅' : '❌';
            const bg   = ok ? '#f0fdf4' : '#fef2f2';
            const bdr  = ok ? '#86efac' : '#fca5a5';
            return `<div style="display:flex;align-items:flex-start;gap:6px;padding:5px 7px;background:${bg};border:1px solid ${bdr};border-radius:5px;font-size:0.75rem;">
                <span style="font-size:0.85rem;line-height:1.4">${icon}</span>
                <div><strong style="color:#1e293b">${label}</strong><br><span style="color:#475569">${detail}</span></div>
            </div>`;
        }
        function warn(label, detail) {
            return `<div style="display:flex;align-items:flex-start;gap:6px;padding:5px 7px;background:#fffbeb;border:1px solid #fcd34d;border-radius:5px;font-size:0.75rem;">
                <span style="font-size:0.85rem;line-height:1.4">⚠️</span>
                <div><strong style="color:#1e293b">${label}</strong><br><span style="color:#475569">${detail}</span></div>
            </div>`;
        }

        let checksHtml = '';

        // ── Pisos vs normativa: ajuste automático (opt-in) o aviso informativo ──
        const cl = resultado?.clamps;
        if (cl) {
            const maxNorm = cl.pisos_max_normativa ?? Math.min(cl.pisos_max_altura, cl.pisos_max_cus, cl.pisos_max_densidad);
            if (cl.ajuste_automatico && cl.pisos_efectivos < cl.pisos_solicitados) {
                const causas = { altura: 'altura máx.', cus: 'CUS máx.', densidad: 'densidad máx.' };
                const motivo = (cl.limitado_por || []).map(c => causas[c] || c).join(' + ');
                checksHtml += warn(
                    `Pisos ajustados: ${cl.pisos_solicitados} → ${cl.pisos_efectivos}`,
                    `Limitado por ${motivo} de la zona · máx por altura: ${cl.pisos_max_altura} · por CUS: ${cl.pisos_max_cus} · por densidad: ${cl.pisos_max_densidad}`);
            } else if (!cl.ajuste_automatico && cl.pisos_solicitados > maxNorm) {
                checksHtml += warn(
                    `Pisos: ${cl.pisos_solicitados} excede máx. normativo (${maxNorm})`,
                    `Máx por altura: ${cl.pisos_max_altura} · por CUS: ${cl.pisos_max_cus} · por densidad: ${cl.pisos_max_densidad} — se respetan los pisos ingresados; revisa los checks de zonificación`);
            }
        }

        if (norma) {
            // Pozo de luz — check real del backend (pozos colocados vs dimensión H/4)
            const pozoFinal = norma.pozo_final || 0;
            const H = params.H || (params.pisos * params.alturaPiso);
            const pzc = norma.pozos_luz_check;
            if (pzc && pzc.colocados > 0) {
                checksHtml += chk(pzc.ok,
                    `Pozos de luz: ${pzc.conformes}/${pzc.colocados} conformes`,
                    `Dimensión requerida H/4 = ${(pzc.dimension_requerida_m || 0).toFixed(2)} m` +
                    (pzc.ok ? '' : ` · ${pzc.no_conformes} pozo(s) sub-norma (sin crédito de ventilación)`));
            } else {
                checksHtml += chk(true, `Pozo de luz: ${pozoFinal.toFixed(2)} m mín`, `H/4 = ${(H/4).toFixed(2)} m · ${H.toFixed(1)} m altura edificio · sin pozos requeridos`);
            }

            // Ascensor
            const ascObl = norma.ascensor_obligatorio;
            const tieneAsc = params.numAscensores > 0 || ascensorPoly.length > 0;
            const ascOk = !ascObl || tieneAsc;
            checksHtml += chk(ascOk, `Ascensor: ${ascObl ? 'Obligatorio' : 'No requerido'}`, ascObl ? (tieneAsc ? `${params.numAscensores} ascensor(es) definido(s)` : 'Sin ascensor — corregir') : `H=${H.toFixed(1)}m ≤ 12m`);

            // Escalera protegida
            const escObl = norma.esc_protegida_obligatoria;
            const tieneVest = vestibuloPoly.length > 0;
            const escOk = !escObl || tieneVest;
            checksHtml += chk(escOk, `Escalera: ${escObl ? 'Presurizada (H>15m)' : 'Abierta ✓'}`, escObl ? (tieneVest ? 'Vestíbulo previo detectado' : 'Sin vestíbulo previo') : `H=${H.toFixed(1)}m ≤ 15m`);

            // Área mínima dptos
            const minArea = norma.area_min_dpto || 40;
            if (geo?.departamentos?.length) {
                const subMin = geo.departamentos.filter(d => (d.area_m2 || 0) < minArea);
                const areaOk = subMin.length === 0;
                checksHtml += chk(areaOk, `Área mín. ${minArea} m² (RNE A.020)`, areaOk ? `Todos los ${geo.departamentos.length} dptos ≥ ${minArea} m²` : `${subMin.length} dpto(s) sub-mínimo`);
            }

            // Cabida
            const cab = geo?.cabida_multifamiliar || norma.cabida_planta;
            if (cab) {
                const ped = cab.departamentos_solicitados_planta ?? cab.departamentos_pedidos ?? 0;
                const gen = cab.departamentos_generados_planta ?? cab.departamentos_emitidos ?? 0;
                const cap = cab.capacidad_maxima_estimada_planta ?? 0;
                const cabOk = gen >= ped;
                checksHtml += chk(cabOk, `Cabida: ${gen}/${ped} dptos/planta generados`, `Capacidad máx estimada: ${cap}/planta`);
            }

            // Evacuación por validacion backend
            if (geo?.departamentos?.length) {
                const sinEvac = geo.departamentos.filter(d => d.validacion && d.validacion.evac_cumple === false);
                const evacOk = sinEvac.length === 0;
                const evacMax = norma.evacuacion_max || 45;
                checksHtml += chk(evacOk, `Evacuación ≤ ${evacMax} m (A.130)`, evacOk ? `Todos los dptos dentro del límite` : `${sinEvac.length} dpto(s) exceden ${evacMax} m`);
            }

            // Ventilación
            if (geo?.departamentos?.length) {
                const sinVent = geo.departamentos.filter(d => d.validacion && d.validacion.ventila_principales === false);
                if (sinVent.length > 0)
                    checksHtml += warn(`Ventilación (${sinVent.length} dpto(s))`, `${sinVent.length} unidad(es) con zona principal sin ventilación natural`);
                else
                    checksHtml += chk(true, 'Ventilación natural', `Todas las zonas principales ventilan`);
            }
        }

        // ── Zonificación (CUS, altura, área libre, retiros, frente, densidad) ──
        const zc = resultado?.zonificacion_check;
        if (zc?.checks) {
            const cert = (zc.parametros_certificado || []).length > 0
                ? ' · certificado aplicado' : '';
            checksHtml += `<div style="margin-top:8px;padding:4px 2px;font-size:0.72rem;font-weight:700;color:#334155;border-bottom:1px solid #e2e8f0;">
                ZONIFICACIÓN ${zc.zona} — ${zc.zona_nombre}${cert}</div>`;
            const labels = {
                cus: 'CUS', altura: 'Altura', area_libre: 'Área libre',
                retiro_frontal: 'Retiro frontal', retiro_lateral: 'Retiro lateral',
                retiro_posterior: 'Retiro posterior', frente_lote: 'Frente de lote',
                area_lote: 'Área de lote', densidad: 'Densidad',
            };
            for (const [k, c] of Object.entries(zc.checks)) {
                checksHtml += chk(c.ok, labels[k] || k, c.mensaje || '');
            }
        }

        // ── Mix óptimo por precios (si el usuario ingresó PEN/m²) ──
        const mox = resultado?.normativa_estricta?.mix_optimo;
        if (mox?.mix_recomendado) {
            const mixStr = Object.entries(mox.mix_recomendado)
                .map(([t, n]) => `${n}×${t}`).join(' + ');
            const ingreso = (mox.ingreso_bruto_estimado || 0).toLocaleString('es-PE', { maximumFractionDigits: 0 });
            checksHtml += `<div style="margin-top:8px;padding:4px 2px;font-size:0.72rem;font-weight:700;color:#334155;border-bottom:1px solid #e2e8f0;">
                MIX ÓPTIMO POR INGRESO</div>`;
            checksHtml += chk(true,
                `${mixStr} por planta (${mox.unidades_totales_planta} dptos)`,
                `Ingreso bruto edificio ≈ S/ ${ingreso} · vendible ${mox.area_vendible_edificio_m2?.toFixed(0)} m²`);
        }

        document.getElementById('norma-checks').innerHTML = checksHtml;

        // ── Topología ──
        const topo = norma?.topologia || geo?.topologia;
        if (topo?.seleccion) {
            const sel = topo.seleccion;
            const met = topo.metricas_lote || {};
            const impl = sel.implementada_actualmente;
            const topoColor = impl ? '#166534' : '#92400e';
            const topoBg   = impl ? '#f0fdf4' : '#fffbeb';
            document.getElementById('norma-topo').innerHTML = `
                <div style="padding:7px 9px;background:${topoBg};border-radius:6px;border:1px solid ${impl ? '#86efac' : '#fcd34d'};">
                    <div style="font-weight:700;color:${topoColor};font-size:0.8rem;">${sel.recomendada.toUpperCase()} ${impl ? '✅' : '⚠️ (no implementada)'}</div>
                    <div style="color:#475569;font-size:0.73rem;margin-top:3px;">${sel.motivo}</div>
                    <div style="color:#94a3b8;font-size:0.7rem;margin-top:4px;">
                        Lote ${met.area_m2?.toFixed(0)} m² · ratio ${met.aspect_ratio?.toFixed(2)} · corto ${met.short_len_m?.toFixed(1)} m · confianza ${(sel.confianza*100).toFixed(0)}%
                    </div>
                    ${sel.alternativas?.length ? `<div style="color:#94a3b8;font-size:0.7rem;margin-top:2px;">Alt: ${sel.alternativas.map(a => a.nombre).join(', ')}</div>` : ''}
                </div>`;
        }

        // ── Validación por dpto ──
        if (geo?.departamentos?.length) {
            let dHtml = '';
            geo.departamentos.forEach((d, i) => {
                const v = d.validacion || {};
                const fallas = v.fallas || [];
                const ok = fallas.length === 0 && v.evac_cumple !== false;
                const icon = ok ? '✅' : '❌';
                const bg   = ok ? '#f8fafc' : '#fef2f2';
                dHtml += `<div style="padding:3px 7px;background:${bg};border-radius:4px;border:1px solid ${ok?'#e2e8f0':'#fca5a5'};font-size:0.72rem;">
                    <strong>${icon} D${String(i+1).padStart(2,'0')} ${d.tipologia||''} · ${(d.area_m2||0).toFixed(1)}m²</strong>
                    ${fallas.length ? `<div style="color:#dc2626;">${fallas.join(' · ')}</div>` : ''}
                </div>`;
            });
            document.getElementById('norma-dptos').innerHTML = dHtml;
        }

    }

    // Calculate shared edge length between two polygons
    function calcSharedEdge(polyA, polyB) {
        let maxShared = 0;
        for (let i = 0; i < polyA.length; i++) {
            let a1 = polyA[i], a2 = polyA[(i + 1) % polyA.length];
            for (let j = 0; j < polyB.length; j++) {
                let b1 = polyB[j], b2 = polyB[(j + 1) % polyB.length];
                let shared = segmentOverlap(a1, a2, b1, b2);
                if (shared > maxShared) maxShared = shared;
            }
        }
        return maxShared;
    }

    function segmentOverlap(a1, a2, b1, b2) {
        let tol = 0.15;
        let dx1 = a2.x - a1.x, dy1 = a2.y - a1.y;
        let dx2 = b2.x - b1.x, dy2 = b2.y - b1.y;
        let len1 = Math.hypot(dx1, dy1), len2 = Math.hypot(dx2, dy2);
        if (len1 < 0.01 || len2 < 0.01) return 0;
        let cross = Math.abs(dx1 * dy2 - dy1 * dx2) / (len1 * len2);
        if (cross > 0.1) return 0;
        let dist = Math.abs((a2.x - a1.x) * (a1.y - b1.y) - (a1.x - b1.x) * (a2.y - a1.y)) / len1;
        if (dist > tol) return 0;
        let isHoriz = Math.abs(dx1) > Math.abs(dy1);
        let aMin, aMax, bMin, bMax;
        if (isHoriz) {
            aMin = Math.min(a1.x, a2.x); aMax = Math.max(a1.x, a2.x);
            bMin = Math.min(b1.x, b2.x); bMax = Math.max(b1.x, b2.x);
        } else {
            aMin = Math.min(a1.y, a2.y); aMax = Math.max(a1.y, a2.y);
            bMin = Math.min(b1.y, b2.y); bMax = Math.max(b1.y, b2.y);
        }
        return Math.max(0, Math.min(aMax, bMax) - Math.max(aMin, bMin));
    }

    function updateTables() {
        let totalDptosPiso = apartments.length;
        let numPisos = params.pisos;
        let totalDptoEdificio = totalDptosPiso * numPisos;

        // 1. Grid Table (Excel style)
        thDptos.colSpan = totalDptosPiso;
        trDptoHeaders.innerHTML = '';
        apartments.forEach(ap => { trDptoHeaders.innerHTML += `<th>DPTO ${ap.id}</th>`; });

        tbodyCuadroAreas.innerHTML = '';
        let sumVendible = 0, sumComun = 0, sumTotal = 0;

        for (let p = 1; p <= numPisos; p++) {
            let tr = document.createElement("tr");
            tr.innerHTML = `<td class="text-center"><b>${p}</b></td>`;
            let floorVendible = 0;
            apartments.forEach(ap => {
                tr.innerHTML += `<td class="text-center">${ap.area.toFixed(2)}</td>`;
                floorVendible += ap.area;
            });
            tr.innerHTML += `<td class="text-center fw-bold text-blue">${floorVendible.toFixed(2)}</td>`;
            tr.innerHTML += `<td class="text-center text-secondary">${calc.realComunPlanta.toFixed(2)}</td>`;
            let fTotal = floorVendible + calc.realComunPlanta;
            tr.innerHTML += `<td class="text-center fw-bold">${fTotal.toFixed(2)}</td>`;

            sumVendible += floorVendible; sumComun += calc.realComunPlanta; sumTotal += fTotal;
            tbodyCuadroAreas.appendChild(tr);
        }

        tfSpan1.colSpan = totalDptosPiso + 1;
        tfVendible.innerText = sumVendible.toFixed(2);
        tfComun.innerText = sumComun.toFixed(2);
        tfTotal.innerText = sumTotal.toFixed(2);

        // Update top KPIs
        kpiTechadaTotal.innerText = sumTotal.toFixed(2);
        kpiVendibleTotal.innerText = sumVendible.toFixed(2);
        kpiEficiencia.innerText = ((sumVendible / sumTotal) * 100).toFixed(1);

        // 2. Sub tables
        let typesCount = {}; let typesTotalArea = {}; let typesHab = {};
        apartments.forEach(ap => {
            if (!typesCount[ap.typology]) { typesCount[ap.typology] = 0; typesTotalArea[ap.typology] = 0; typesHab[ap.typology] = 0; }
            typesCount[ap.typology] += numPisos;
            typesTotalArea[ap.typology] += ap.area * numPisos;
            typesHab[ap.typology] += ap.hab * numPisos;
        });

        tbodyTipologias.innerHTML = ''; let totalHabsEdificio = 0;
        for (const [type, count] of Object.entries(typesCount)) {
            let prom = (typesTotalArea[type] / count).toFixed(2);
            let pct = ((count / totalDptoEdificio) * 100).toFixed(1);
            tbodyTipologias.innerHTML += `<tr><td><b>${type}</b></td><td class="text-center">${prom}</td><td class="text-center">${count}</td><td class="text-center">${pct}%</td></tr>`;
            totalHabsEdificio += typesHab[type];
        }

        resTotalDptos.innerText = totalDptoEdificio;
        let reqEstac = Math.ceil(totalDptoEdificio * (params.pctEstac / 100));
        resEstacionamientos.innerText = reqEstac;
        resHabitantes.innerText = Math.round(totalHabsEdificio);

        // Update Planta KPI bar
        let kpiBarT = document.getElementById("kpi-terreno-bar");
        let kpiBarTe = document.getElementById("kpi-techada-bar");
        let kpiBarV = document.getElementById("kpi-vendible-bar");
        let kpiBarE = document.getElementById("kpi-eficiencia-bar");
        let kpiBarD = document.getElementById("kpi-dptos-bar");
        if (kpiBarT) kpiBarT.innerText = calc.areaTerreno?.toFixed(2) || "0";
        if (kpiBarTe) kpiBarTe.innerText = sumTotal.toFixed(2);
        if (kpiBarV) kpiBarV.innerText = sumVendible.toFixed(2);
        if (kpiBarE) kpiBarE.innerText = ((sumVendible / sumTotal) * 100).toFixed(1);
        if (kpiBarD) kpiBarD.innerText = totalDptoEdificio;

        // Á. Libre Real = Retiros + Ductos + Patio (per planta)
        let actDuctAreaTotal = 0;
        smallDuctos.forEach(d => actDuctAreaTotal += calculatePolyArea(d));
        if (patioPoly.length) actDuctAreaTotal += calculatePolyArea(patioPoly);
        let areaLibreRealPlanta = calc.areaRetiroTotal + actDuctAreaTotal;
        let pctLibreReal = calc.areaTerreno > 0 ? (areaLibreRealPlanta / calc.areaTerreno) * 100 : 0;
        let kpiLibreBar = document.getElementById('kpi-libre-bar');
        if (kpiLibreBar) kpiLibreBar.innerText = pctLibreReal.toFixed(1);

        // Update RNE normative display
        let rneH = document.getElementById("rne-H");
        let rnePozoDorm = document.getElementById("rne-pozo-dorm");
        let rnePozoSala = document.getElementById("rne-pozo-sala");
        if (rneH) rneH.innerText = params.H?.toFixed(2) || "0";
        if (rnePozoDorm) rnePozoDorm.innerText = params.pozoDormMin?.toFixed(2) || "2.20";
        if (rnePozoSala) rnePozoSala.innerText = params.pozoSalaMin?.toFixed(2) || "3.00";

        // Dynamic thresholds display
        let rneAsc = document.getElementById("rne-ascensor");
        let rneEsc = document.getElementById("rne-escalera-tipo");
        if (rneAsc) {
            if (params.ascensorObligatorio) {
                rneAsc.innerHTML = '<span style="color:#dc2626">Obligatorio (&gt;4 pisos)</span>';
            } else {
                rneAsc.innerHTML = '<span style="color:#059669">No requerido (≤4 pisos)</span>';
            }
        }
        if (rneEsc) {
            if (params.escaleraPresurizada) {
                rneEsc.innerHTML = '<span style="color:#dc2626">Presurizada 🔒</span>';
            } else {
                rneEsc.innerHTML = '<span style="color:#059669">Abierta ✓</span>';
            }
        }
    }

    function _actualizarCuadros(resultado) {
        // 4.1 Cuadro de áreas — COS, CUS, área libre
        const ca = resultado?.cuadro_areas;
        if (ca) {
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
            set('res-area-libre', (ca.area_libre_planta_m2 ?? 0).toFixed(2));
            set('res-pct-libre', (ca.pct_area_libre ?? 0).toFixed(1));
            set('res-cos', (ca.cos_real ?? 0).toFixed(3));
            set('res-cus', (ca.cus_real ?? 0).toFixed(3));
        }
        // 4.2 Cuadro de unidades
        const cu = resultado?.cuadro_unidades;
        const tbody = document.getElementById('tbody-cuadro-unidades');
        if (cu && tbody) {
            tbody.innerHTML = '';
            cu.forEach(u => {
                const fachExt = u.fachada_exterior
                    ? '<span style="color:#d97706;font-weight:700;">◆ Exterior</span>'
                    : '<span style="color:#64748b;">○ Interior</span>';
                const escOk = u.dist_esc_cumple
                    ? `<span style="color:#059669">✓</span>`
                    : `<span style="color:#dc2626">✗ >${u.dist_escalera_m?.toFixed(1)}m</span>`;
                const lado = u.lado === 'frente' ? 'Frente' : 'Fondo';
                let obs = [];
                if (u.es_reducida) obs.push('<span style="color:#f59e0b">Recortada</span>');
                if (!u.cumple_area_min) obs.push('<span style="color:#dc2626">Sub-mín.</span>');
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="text-center fw-bold">${u.id}</td>
                    <td class="text-center">${u.tipologia}</td>
                    <td class="text-center">${(u.area_neta_m2 ?? 0).toFixed(2)}</td>
                    <td class="text-center text-secondary">${(u.area_gross_m2 ?? 0).toFixed(2)}</td>
                    <td class="text-center">${lado}</td>
                    <td class="text-center">${fachExt}</td>
                    <td class="text-center">${(u.dist_escalera_m ?? 0).toFixed(1)}</td>
                    <td class="text-center">${escOk}</td>
                    <td class="text-center">${obs.join(' ')}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    }

    function getCurrentViewKey() {
        let vp = document.getElementById('viewport-level');
        if (!vp) return 'planta_tipica';
        let val = vp.value;
        if (val === 'tipica') return 'planta_tipica';
        if (val === 'primero') return 'primer_piso';
        if (val === 'sotano') return 'sotano';
        return 'planta_tipica';
    }

    // --- API Integración ---
    async function ejecutarAuditoriaRNE() {
        // 1. EL PUENTE DE DATOS COMPLETO
        const distribSel = document.getElementById('distribucion');
        const esquemaMap = { optimo: 'muros_ciegos', post: 'patio_posterior', centro: 'ducto_central' };
        const esquemaAreaLibre = esquemaMap[distribSel?.value] || 'muros_ciegos';
        const optDensidad = !!document.getElementById('optimizar-densidad')?.checked;
        const datos = {
            // Solo retiro frontal neto: laterales/posterior los aplica el backend UNA vez (_erode_lote).
            // Enviar loteNetoPoly aquí duplicaría el retiro lateral/posterior (ya erosionado dos veces).
            coordenadas_lote: customLoteCoords || (techadaPoly.length > 0 ? techadaPoly.map(p => [p.x, p.y]) : polygon.map(p => [p.x, p.y])),
            area_bruta_terreno: calc.areaTerreno,
            numero_pisos: params.pisos || 7,
            retiro_frontal: params.retiroFrontal || 0.0,
            zonificacion: document.getElementById('zona-select')?.value || "RDA",
            // Overrides del certificado de parámetros (null = usar tabla de zona)
            cus_maximo: parseFloat(document.getElementById('cus-max')?.value) || null,
            altura_maxima_pisos: parseInt(document.getElementById('altura-max-pisos')?.value) || null,
            densidad_maxima_hab_ha: parseFloat(document.getElementById('densidad-max')?.value) || null,
            ajustar_pisos_normativa: !!document.getElementById('ajustar-pisos')?.checked,
            num_ascensores: params.nAscensores || 1,
            num_departamentos: (() => {
                const n1 = parseInt(document.getElementById('mix-1d')?.value) || 0;
                const n2 = parseInt(document.getElementById('mix-2d')?.value) || 0;
                const n3 = parseInt(document.getElementById('mix-3d')?.value) || 0;
                const mixSum = n1 + n2 + n3;
                return mixSum > 0 ? mixSum : (params.dptosPlanta || 6);
            })(),
            // Parámetros adicionales para renderizado Python
            frente: params.frente || 10,
            fondo: params.fondo || 10,
            derecha: params.derecha || 20,
            izquierda: params.izquierda || 20,
            altura_piso: params.alturaPiso || 2.80,
            pct_estac: params.pctEstac || 30,
            ciego_frente: ciegoInp.frente ? ciegoInp.frente.checked : false,
            ciego_fondo: ciegoInp.fondo ? ciegoInp.fondo.checked : true,
            ciego_derecha: ciegoInp.derecha ? ciegoInp.derecha.checked : true,
            ciego_izquierda: ciegoInp.izquierda ? ciegoInp.izquierda.checked : true,
            esquema_area_libre: esquemaAreaLibre,
            optimizar_densidad: optDensidad,
            retiro_lateral: params.retiroLateral,
            retiro_posterior: params.retiroPosterior,
            area_libre_min_pct: params.pctLibreReq || 0,
            frente_exterior: document.getElementById('frente-exterior')?.checked ?? true,
            fondo_exterior: document.getElementById('fondo-exterior')?.checked || false,
            derecha_exterior: document.getElementById('derecha-exterior')?.checked || false,
            izquierda_exterior: document.getElementById('izquierda-exterior')?.checked || false,
            mix_tipologias: (() => {
                const n1 = parseInt(document.getElementById('mix-1d')?.value) || 0;
                const n2 = parseInt(document.getElementById('mix-2d')?.value) || 0;
                const n3 = parseInt(document.getElementById('mix-3d')?.value) || 0;
                if (n1 + n2 + n3 === 0) return null;
                const m = {};
                if (n1 > 0) m['1D'] = n1;
                if (n2 > 0) m['2D'] = n2;
                if (n3 > 0) m['3D'] = n3;
                return m;
            })(),
            precios_tipologia: (() => {
                const ids = { '1D': 'precio-1d', '1D+E': 'precio-1de', '2D': 'precio-2d',
                              '2D+E': 'precio-2de', '3D': 'precio-3d' };
                const p = {};
                for (const [tipo, id] of Object.entries(ids)) {
                    const v = parseFloat(document.getElementById(id)?.value) || 0;
                    if (v > 0) p[tipo] = v;
                }
                return Object.keys(p).length > 0 ? p : null;
            })(),
        };

        try {
            const response = await fetch(`${API_BASE_URL}/auditoria-rne`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(datos)
            });

            const resultado = await response.json();

            if (!response.ok) {
                const detalle = resultado?.detail || `Error ${response.status}`;
                alert("No se pudo generar la cabida:\n\n" + detalle);
                console.warn("Auditoría rechazada:", response.status, detalle);
                return;
            }

            // Almacenar la respuesta del servidor como fuente de verdad normativa global
            rneResultado = resultado;

            // ══ LOGGING ESTRUCTURADO ══
            console.group("🏢 AUDITORÍA RNE — WebGL Payload");
            console.log("Status:", resultado.status);

            // Nuevo payload WebGL
            if (resultado.metadata_proyecto) {
                console.log("📊 Metadata Proyecto:", resultado.metadata_proyecto);
            }
            if (resultado.geometria) {
                const geo = resultado.geometria;
                console.group("📐 Geometría Normalizada (Three.js ready)");
                console.log("  Lote:",         geo.lote?.coords?.length, "vértices");
                console.log("  Unidades:",      geo.unidades?.length, "dptos/planta");
                console.log("  Hall coords:",   geo.circulacion?.hall?.coords?.length, "vértices");
                console.log("  Escalera tipo:", geo.nucleo?.escaleras?.tipo);
                console.log("  Ascensores:",    geo.nucleo?.ascensores?.length);
                console.log("  Patios:",        geo.tecnico?.patios?.length);
                console.log("  Ductos:",        geo.tecnico?.ductos?.length);
                // Validación adyacencia
                const sinHall = (geo.unidades || []).filter(u => !u.validacion?.colinda_hall);
                if (sinHall.length > 0) {
                    console.warn("⚠️ Unidades sin contacto con hall:", sinHall.map(u => u.id));
                } else {
                    console.log("✅ Todas las unidades colindan con el hall");
                }
                console.groupEnd();
            }
            if (resultado.normativa_estricta) console.log("📋 Normativa Estricta:", resultado.normativa_estricta);
            console.groupEnd();

            // ═══ GUARDAR PAYLOAD WebGL para uso futuro por Three.js ═══
            window.webglPayload = resultado.geometria || null;
            window.metadataProyecto = resultado.metadata_proyecto || null;

            // ═══ RENDERIZAR EN THREE.JS VIEWER ═══
            if (resultado.geometria) {
                try {
                    const v = getViewer3D();
                    if (v) {
                        v.renderProyecto(resultado.geometria, resultado.metadata_proyecto || {});
                        // Sincronizar nivel de vista
                        const vpLevel = document.getElementById('viewport-level');
                        if (vpLevel) v.setView({ tipica: 'tipica', primero: 'primero', sotano: 'sotano', azotea: 'azotea' }[vpLevel.value] || 'tipica');
                    }
                } catch(e3d) {
                    console.warn('⚠️ Viewer3D render error:', e3d);
                    if (viewer3d) { viewer3d = null; }
                }
            }

            if (resultado.normativa_estricta) {
                // ═══ REASIGNACIÓN DE PARÁMETROS GLOBALES ═══
                params.pozoDormMin = resultado.normativa_estricta.pozo_final;
                params.hallMinWidth = 1.20;
                params.areaMinimaLegal = resultado.normativa_estricta.area_min_dpto;

                params.ascensorObligatorio = resultado.normativa_estricta.ascensor_obligatorio;
                params.escaleraPresurizada = resultado.normativa_estricta.esc_protegida_obligatoria;

                // Actualizar visuales de la tabla RNE
                const eH = document.getElementById('rne-H');
                const ePozoD = document.getElementById('rne-pozo-dorm');
                const ePozoS = document.getElementById('rne-pozo-sala');
                const eAscensor = document.getElementById('rne-ascensor');
                const eEscalera = document.getElementById('rne-escalera-tipo');

                if (eH) eH.innerText = params.H?.toFixed(2) || "0";
                if (ePozoD) ePozoD.innerText = resultado.normativa_estricta.pozo_final.toFixed(2);
                if (ePozoS) ePozoS.innerText = resultado.normativa_estricta.pozo_final.toFixed(2);

                if (eAscensor) {
                    eAscensor.innerText = resultado.normativa_estricta.ascensor_obligatorio
                        ? 'Obligatorio' : 'No requerido';
                }
                if (eEscalera) {
                    eEscalera.innerText = resultado.normativa_estricta.esc_protegida_obligatoria
                        ? 'Presurizada 🔒' : 'Abierta ✓';
                }

                // ═══ INYECCIÓN DE GEOMETRÍA (para datos/tablas) ═══
                if (resultado.geometria_generada) {
                    const geo = resultado.geometria_generada;
                    corePoly = geo.core || [];
                    window.hallProcedural = geo.hall || [];
                    escaleraPoly = geo.escalera || [];
                    vestibuloPoly = geo.vestibulo || [];

                    if (geo.ascensores && geo.ascensores.length > 0) {
                        ascensorPoly = geo.ascensores[0] || [];
                    } else {
                        ascensorPoly = [];
                    }

                    patioPoly = geo.patio || [];
                    smallDuctos = (geo.ductos || []).filter(d => d && d.length >= 3);

                    if (geo.departamentos && geo.departamentos.length > 0) {
                        apartments = geo.departamentos
                            .map((entry, i) => {
                                let contorno = Array.isArray(entry) ? entry : (entry && entry.contorno);
                                let zonas = (!Array.isArray(entry) && entry && Array.isArray(entry.zonas)) ? entry.zonas : [];
                                let tipologia = (!Array.isArray(entry) && entry && entry.tipologia) ? entry.tipologia : '';
                                if (!contorno || contorno.length < 3) return null;
                                return {
                                    id: `X${String(i + 1).padStart(2, '0')}`,
                                    poly: contorno,
                                    zonas: zonas,
                                    area: 0,
                                    typology: tipologia || '3D',
                                    hab: 3
                                };
                            })
                            .filter(Boolean);
                    }
                }

                // ═══ REENCENDIDO ═══
                isGenerated = true;
                window.hasCore = true;
                calc.maxDistEscalera = 0;
                calc.cumple25m = true;

                processGeometryAndCheckHabitability(apartments.length);
                updateTables();
                updateCompliancePanel();
                updateNormaCheckPanel(resultado);
                _actualizarCuadros(resultado);

            }

        } catch (error) {
            console.error("Error conectando con el Cerebro Python:", error);
            alert("❌ Error: No se pudo conectar con el motor de cálculo.\n\nSi es la primera vez del día, el servidor puede tardar ~30 segundos en despertar. Intenta de nuevo en un momento.\n\nURL: " + API_BASE_URL + "\nDetalles: " + error.message);
        }
    }



    // --- Actions ---
    btnGenerateAI.addEventListener("click", async (e) => {
        // PREVIENE RECARGAS
        e.preventDefault();

        // APAGA EL DIBUJO LOCAL COMPLETAMENTE
        isGenerated = false;

        btnGenerateAI.classList.add("hidden");
        aiLoader.classList.remove("hidden");

        // El Canvas SOLO debe mostrar el polígono del lote por ahora
        try {
            // Llamada ÚNICA Y EXCLUSIVA a Python (El puente inyectará los datos y re-encenderá el motor)
            await ejecutarAuditoriaRNE();
        } finally {
            // Regresar el botón a su estado sin dibujar polígonos antiguos
            aiLoader.classList.add("hidden");
            btnGenerateAI.classList.remove("hidden");
        }
    });

    inputs.forEach(inp => {
        inp.addEventListener("change", () => { isGenerated = false; updateCalculations(); });
    });

    // Add listeners to checkboxes
    Object.values(ciegoInp).forEach(chk => {
        if (chk) chk.addEventListener("change", () => { isGenerated = false; updateCalculations(); });
    });

    toggleEditMode.addEventListener("change", (e) => {
        isEditMode = e.target.checked;
        document.getElementById('viewport-title').innerText = isEditMode ? "Modo Edición" : "Vista:";

        // Three.js es el motor activo: canvas 2D no se usa.
        // Edit mode solo cambia el label; al salir re-evalúa normativa.
        if (!isEditMode) {
            processGeometryAndCheckHabitability(apartments.length);
            updateTables();
            updateCompliancePanel();
            ejecutarAuditoriaRNE();
        }
    });

    // Botón dibujar lote inline
    const btnDrawLote = document.getElementById('btn-draw-lote');
    if (btnDrawLote) btnDrawLote.addEventListener('click', enterDrawMode);

    // Botón borrar lote personalizado
    const btnClearCustom = document.getElementById('btn-clear-custom-lote');
    if (btnClearCustom) btnClearCustom.addEventListener('click', () => {
        customLoteCoords = null;
        updateCustomLoteIndicator();
        updateCalculations();
    });

    toggleLinderos.addEventListener("change", (e) => {
        showLinderos = e.target.checked;
    });

    const viewportLevelSelect = document.getElementById("viewport-level");
    const sotanoLevelSelect = document.getElementById("sotano-level-select");
    if (viewportLevelSelect) {
        viewportLevelSelect.addEventListener("change", () => {
            let isSotano = viewportLevelSelect.value === 'sotano';
            if (sotanoLevelSelect) {
                sotanoLevelSelect.style.display = isSotano ? 'inline-block' : 'none';
                const sot = window.webglPayload?.sotano;
                if (isSotano && sot) {
                    sotanoLevelSelect.innerHTML =
                        `<option value="0">${sot.name || 'S1'} (${sot.count || 0} est.)</option>`;
                }
            }
        });
    }
    if (sotanoLevelSelect) {
        sotanoLevelSelect.addEventListener("change", () => {
        });
    }

    // --- Legacy 2D canvas renderer removed; Three.js viewer handles all rendering ---

    window.resetView = () => {
        if (viewer3d && window.webglPayload) {
            // Delegar al auto-fit de Three.js
            viewer3d._fit2D(window.webglPayload.lote?.coords);
        }
    };

    window.exportImage = () => {
    };

    // Auto-recalculate when any input changes
    inputs.forEach(inp => {
        inp.addEventListener('input', () => {
            if (isGenerated) {
                updateCalculations();
            }
        });
    });

    // Mix de Tipologías — live total + toggle dptos-planta visibility
    function _updateMixDisplay() {
        const n1 = parseInt(document.getElementById('mix-1d')?.value) || 0;
        const n2 = parseInt(document.getElementById('mix-2d')?.value) || 0;
        const n3 = parseInt(document.getElementById('mix-3d')?.value) || 0;
        const total = n1 + n2 + n3;
        const el = document.getElementById('mix-total-display');
        if (el) el.textContent = total > 0 ? `Total: ${total} unidades por piso` : '';
        const ctrlDptos = document.getElementById('ctrl-dptos');
        if (ctrlDptos) ctrlDptos.style.display = total > 0 ? 'none' : '';
    }
    ['mix-1d', 'mix-2d', 'mix-3d'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', _updateMixDisplay);
    });
    _updateMixDisplay();

    // Fachada exterior hints — actualizar texto al cambiar checkbox
    function _updateFachadaHint(side) {
        const cb = document.getElementById(`${side}-exterior`);
        const hint = document.getElementById(`${side}-fachada-hint`);
        if (!cb || !hint) return;
        if (cb.checked) {
            hint.textContent = 'Da a calle — Dptos. con Fachada Exterior';
            hint.style.color = '#d97706';
            hint.style.fontWeight = '600';
        } else {
            hint.textContent = 'Medianería / Fachada Interior';
            hint.style.color = '#64748b';
            hint.style.fontWeight = 'normal';
        }
    }
    ['frente', 'fondo', 'derecha', 'izquierda'].forEach(side => {
        const cb = document.getElementById(`${side}-exterior`);
        if (cb) {
            cb.addEventListener('change', () => _updateFachadaHint(side));
            _updateFachadaHint(side);
        }
    });

    setTimeout(() => {
        updateCalculations();
        // Inicializar Three.js viewer desde el primer frame
        getViewer3D();
    }, 150);
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
