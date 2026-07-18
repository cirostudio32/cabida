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
    const calcErrorBanner = document.getElementById("calc-error-banner");
    const toggleEditMode = document.getElementById("toggle-edit-mode");
    const toggleLinderos = document.getElementById("toggle-linderos");

    document.querySelectorAll('.accordion-header').forEach(header => {
        header.addEventListener('click', () => {
            header.closest('.accordion-panel').classList.toggle('expanded');
        });
    });

    function showCalcError(msg) {
        if (!calcErrorBanner) return;
        calcErrorBanner.textContent = msg;
        calcErrorBanner.style.display = 'block';
    }
    function hideCalcError() {
        if (!calcErrorBanner) return;
        calcErrorBanner.style.display = 'none';
    }

    // Validación previa al fetch (~30s round trip): evita enviar combinaciones geométricamente imposibles.
    function validarInputsPrevios(p) {
        if (p.retiroLateral * 2 >= p.frente) {
            return `El retiro lateral (${p.retiroLateral}m x2 lados) no puede ser ≥ al frente del lote (${p.frente}m). Reduce el retiro o aumenta el frente.`;
        }
        const fondoProm = (p.derecha + p.izquierda) / 2 || p.fondo;
        if (p.retiroPosterior >= fondoProm) {
            return `El retiro posterior (${p.retiroPosterior}m) no puede ser ≥ al fondo del lote (${fondoProm.toFixed(1)}m). Reduce el retiro o aumenta el fondo.`;
        }
        const n1 = parseInt(document.getElementById('mix-1d')?.value) || 0;
        const n2 = parseInt(document.getElementById('mix-2d')?.value) || 0;
        const n3 = parseInt(document.getElementById('mix-3d')?.value) || 0;
        const mixSum = n1 + n2 + n3;
        if (mixSum > 0 && mixSum !== p.dptosPlanta) {
            return `El mix de tipologías suma ${mixSum} dptos, pero "Dptos/Planta" indica ${p.dptosPlanta}. Ajusta uno de los dos valores para que coincidan.`;
        }
        return null;
    }

    // Leaflet Map State
    let leafletMap = null;
    let mapDrawLayer = null;
    let mapMeasureLayer = null;  // tooltips de longitud de segmento (dibujo en vivo)

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
            closeMobileSidebar();
        });
    });

    // Sidebar como drawer en móvil
    const sidebarEl = document.querySelector('.sidebar');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');
    const btnMobileMenu = document.getElementById('btn-mobile-menu');
    function openMobileSidebar() {
        sidebarEl?.classList.add('open');
        sidebarBackdrop?.classList.add('open');
    }
    function closeMobileSidebar() {
        sidebarEl?.classList.remove('open');
        sidebarBackdrop?.classList.remove('open');
    }
    btnMobileMenu?.addEventListener('click', () => {
        sidebarEl?.classList.contains('open') ? closeMobileSidebar() : openMobileSidebar();
    });
    sidebarBackdrop?.addEventListener('click', closeMobileSidebar);

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

        mapMeasureLayer = L.layerGroup().addTo(leafletMap);

        // Medidas en vivo mientras se colocan los vértices del polígono.
        leafletMap.on('pm:drawstart', ({ workingLayer }) => {
            const onVtx = () => _renderMapMeasures(_flatLatLngs(workingLayer.getLatLngs()));
            const onMove = (ev) => {
                const pts = _flatLatLngs(workingLayer.getLatLngs());
                _renderMapMeasures(pts.length ? [...pts, ev.latlng] : pts);
            };
            workingLayer.on('pm:vertexadded', onVtx);
            leafletMap.on('mousemove', onMove);
            leafletMap.once('pm:drawend', () => {
                workingLayer.off('pm:vertexadded', onVtx);
                leafletMap.off('mousemove', onMove);
            });
        });

        leafletMap.on('pm:create', function (e) {
            mapDrawLayer.clearLayers();
            mapDrawLayer.addLayer(e.layer);
            calculateMapMeasures(e.layer);
            _renderMapMeasures(_flatLatLngs(e.layer.getLatLngs()), true);  // medidas finales persistentes

            // Allow editing of the new layer
            e.layer.on('pm:edit', function (x) {
                calculateMapMeasures(x.layer);
                _renderMapMeasures(_flatLatLngs(x.layer.getLatLngs()), true);
            });
        });

        document.getElementById('btn-sync-map').addEventListener('click', () => {
            if (mapDrawLayer.getLayers().length > 0) {
                calculateMapMeasures(mapDrawLayer.getLayers()[0]);
                updateCalculations();
            }
            showPlanView();
            _renderHuellaPreview();  // huella dibujada + cotas en la vista planta (pre-recalcular)
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

    /** Aplana los latlngs de un layer (polyline plana o polígono anidado). */
    function _flatLatLngs(ll) {
        if (!Array.isArray(ll)) return [];
        return Array.isArray(ll[0]) ? ll[0] : ll;
    }

    /** Dibuja tooltips con la longitud (m) de cada segmento sobre el mapa.
     *  closed=true añade el segmento de cierre (último→primero). */
    function _renderMapMeasures(latlngs, closed = false) {
        if (!mapMeasureLayer) return;
        mapMeasureLayer.clearLayers();
        if (!latlngs || latlngs.length < 2) return;
        const seg = (a, b) => {
            const d = turf.distance(turf.point([a.lng, a.lat]), turf.point([b.lng, b.lat]), { units: 'meters' });
            L.marker([(a.lat + b.lat) / 2, (a.lng + b.lng) / 2], {
                interactive: false,
                icon: L.divIcon({ className: 'seg-measure', html: `${d.toFixed(1)} m`, iconSize: [46, 16] }),
            }).addTo(mapMeasureLayer);
        };
        for (let i = 0; i < latlngs.length - 1; i++) seg(latlngs[i], latlngs[i + 1]);
        if (closed && latlngs.length >= 3) seg(latlngs[latlngs.length - 1], latlngs[0]);
    }

    /** Render de la huella dibujada (customLoteCoords) en la vista planta con
     *  sus cotas, ANTES de recalcular. Reusa renderProyecto con solo el lote. */
    function _renderHuellaPreview() {
        if (!customLoteCoords || customLoteCoords.length < 3) return;
        const v = getViewer3D();
        if (!v) return;
        v.renderProyecto({ lote: { coords: customLoteCoords.map(p => [p[0], p[1]]) } },
                         { pisos: 1, altura_piso: 2.80 });
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
            showCalcError('Genera una distribución inicial primero (o usa el botón "Ver Mapa") para calibrar la escala del lote.');
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

    // ── Modo Edición: editar departamentos + hall de la planta típica ──
    // Arrastrar vértices/cuerpos, dibujar dptos a mano, eliminar, y botón
    // "Evaluar" que valida contra normativa SIN regenerar la distribución.
    let editSvgEl = null, editObjs = null, editSel = -1, editDragV = -1,
        editDragBody = null, editOrigJson = null, editDraw = null, editUI = null;
    let layoutFijado = false;  // diseño editado a mano; Recalcular pediría confirmación

    function _updateFijadoBadge() {
        let badge = document.getElementById('layout-fijado-badge');
        if (!layoutFijado) { badge?.remove(); return; }
        if (!badge) {
            badge = document.createElement('span');
            badge.id = 'layout-fijado-badge';
            Object.assign(badge.style, {
                display: 'inline-flex', alignItems: 'center', gap: '6px',
                marginLeft: '10px', padding: '3px 10px', borderRadius: '999px',
                background: 'rgba(79,70,229,0.12)', color: '#4f46e5',
                fontSize: '11px', fontWeight: '700', whiteSpace: 'nowrap',
            });
            badge.innerHTML = `Diseño fijado <button id="layout-fijado-liberar" title="Volver a la distribución generada" style="border:none;background:transparent;color:#4f46e5;cursor:pointer;font-weight:700;font-size:12px;line-height:1;">✕ liberar</button>`;
            document.getElementById('viewport-title')?.after(badge);
            badge.querySelector('#layout-fijado-liberar').onclick = () => {
                layoutFijado = false;
                _updateFijadoBadge();
            };
        }
    }

    function _editSnapshot() {
        return JSON.stringify(editObjs.map(o => ({ k: o.kind, v: o.verts })));
    }

    function enterEditMode() {
        const v = window.__viewer3d;
        const pl = window.webglPayload;
        if (!v || !pl?.unidades?.length) {
            showCalcError('Genera una distribución primero para poder editar los departamentos.');
            return false;
        }
        if (v.mode === '3d') {
            showCalcError('Cambia a vista Planta 2D para editar.');
            return false;
        }
        editObjs = pl.unidades.map(u => ({
            kind: 'unidad', ref: u, verts: u.coords.map(c => [c[0], c[1]]),
        }));
        const hallC = pl.circulacion?.hall?.coords;
        if (hallC?.length >= 3) {
            editObjs.push({ kind: 'hall', ref: pl.circulacion.hall, verts: hallC.map(c => [c[0], c[1]]) });
        }
        for (const d of (pl.tecnico?.ductos || [])) {
            if (d.coords?.length >= 3) editObjs.push({ kind: 'ducto', ref: d, verts: d.coords.map(c => [c[0], c[1]]) });
        }
        for (const p of (pl.tecnico?.pozos_luz || [])) {
            if (p.coords?.length >= 3) editObjs.push({ kind: 'pozo', ref: p, verts: p.coords.map(c => [c[0], c[1]]) });
        }
        editSel = -1; editDraw = null;
        editOrigJson = _editSnapshot();

        editSvgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        Object.assign(editSvgEl.style, {
            position: 'absolute', inset: '0', width: '100%', height: '100%',
            zIndex: '25', display: 'block', pointerEvents: 'none',
        });
        wrapper.appendChild(editSvgEl);

        editUI = document.createElement('div');
        editUI.id = 'edit-inst-banner';
        Object.assign(editUI.style, {
            position: 'absolute', top: '8px', left: '50%', transform: 'translateX(-50%)',
            zIndex: '26', background: '#232741', color: 'white',
            padding: '6px 10px', borderRadius: '6px', fontSize: '12px',
            fontWeight: '600', whiteSpace: 'nowrap',
            display: 'flex', gap: '8px', alignItems: 'center',
        });
        const btnStyle = 'padding:3px 10px;border:1px solid rgba(255,255,255,0.35);border-radius:5px;background:transparent;color:white;cursor:pointer;font-size:11px;font-weight:600;';
        editUI.innerHTML = `
            <button id="edit-btn-draw" style="${btnStyle}">＋ Dibujar dpto</button>
            <button id="edit-btn-del" style="${btnStyle}">Eliminar</button>
            <button id="edit-btn-eval" style="${btnStyle}background:#4f46e5;border-color:#4f46e5;">Evaluar</button>
            <span id="edit-hint" style="font-weight:500;opacity:0.85;"></span>`;
        wrapper.appendChild(editUI);
        editUI.querySelector('#edit-btn-draw').onclick = () => {
            editDraw = { verts: [], cursor: null }; editSel = -1;
            editSvgEl.style.pointerEvents = 'auto'; _renderEditSvg();
        };
        editUI.querySelector('#edit-btn-del').onclick = _editDeleteSel;
        editUI.querySelector('#edit-btn-eval').onclick = _editEvaluar;

        // Nube recordatorio: los cambios se aplican al salir del modo edición
        const cloud = document.createElement('div');
        cloud.id = 'edit-apply-cloud';
        Object.assign(cloud.style, {
            position: 'absolute', top: '8px', right: '8px',
            zIndex: '26', background: 'white', color: '#232741',
            padding: '7px 14px', borderRadius: '999px',
            boxShadow: '0 6px 20px rgba(20,24,48,0.16)', border: '1px solid #e5e7f0',
            fontSize: '12px', fontWeight: '600', whiteSpace: 'nowrap',
            display: 'flex', gap: '7px', alignItems: 'center', pointerEvents: 'none',
        });
        cloud.innerHTML = `<span style="display:inline-flex;width:16px;height:16px;border-radius:50%;background:#4f46e5;color:white;align-items:center;justify-content:center;font-size:11px;font-weight:700;">i</span>Sal del Modo Edición para aplicar los cambios`;
        wrapper.appendChild(cloud);

        editSvgEl.addEventListener('pointerdown', _onEditPointerDown);
        editSvgEl.addEventListener('pointermove', _onEditPointerMove);
        editSvgEl.addEventListener('pointerup',   _onEditPointerUp);
        editSvgEl.addEventListener('dblclick',    () => { if (editDraw) _editCloseDraw(); });
        // El overlay SVG tapa el canvas: reenviar wheel para que el zoom (OrbitControls,
        // escuchado directo sobre el canvas) siga funcionando en Modo Edición.
        editSvgEl.addEventListener('wheel', (e) => {
            e.preventDefault();
            v.renderer.domElement.dispatchEvent(new WheelEvent('wheel', {
                deltaY: e.deltaY, deltaX: e.deltaX, deltaMode: e.deltaMode,
                clientX: e.clientX, clientY: e.clientY, bubbles: true, cancelable: true,
            }));
        }, { passive: false });
        // El zoom/pan actualiza la cámara vía OrbitControls, no un evento de puntero
        // sobre el propio SVG: sin este listener el overlay queda "congelado" hasta
        // el próximo pointerdown/move (bug reportado: polígonos no siguen el zoom).
        v.controls.addEventListener('change', _renderEditSvg);
        document.addEventListener('keydown', _onEditKey);
        _renderEditSvg();
        return true;
    }

    function exitEditMode(save) {
        if (!editSvgEl) return false;
        editSvgEl.remove(); editSvgEl = null;
        editUI?.remove(); editUI = null;
        document.getElementById('edit-apply-cloud')?.remove();
        document.getElementById('edit-eval-panel')?.remove();
        document.removeEventListener('keydown', _onEditKey);
        window.__viewer3d?.controls?.removeEventListener('change', _renderEditSvg);
        editDragV = -1; editDragBody = null; editDraw = null;

        const changed = save && _editSnapshot() !== editOrigJson;
        if (changed) {
            const pl = window.webglPayload;
            // Ductos/pozos: reescribir coords y reconstruir arrays (por si se borró alguno)
            if (pl.tecnico) {
                pl.tecnico.ductos = editObjs.filter(o => o.kind === 'ducto')
                    .map(o => ({ ...o.ref, coords: o.verts.map(p => [p[0], p[1]]) }));
                pl.tecnico.pozos_luz = editObjs.filter(o => o.kind === 'pozo')
                    .map(o => ({ ...o.ref, coords: o.verts.map(p => [p[0], p[1]]) }));
            }
            const unidades = [];
            let nNew = 0;
            for (const o of editObjs) {
                if (o.kind === 'ducto' || o.kind === 'pozo') continue;  // ya tratados arriba
                if (o.kind === 'hall') {
                    o.ref.coords = o.verts.map(p => [p[0], p[1]]);
                    if (pl.circulacion?.halls?.length === 1) pl.circulacion.halls[0].coords = o.ref.coords;
                    if (pl.nucleo?.nucleos?.length === 1 && pl.nucleo.nucleos[0].hall) {
                        pl.nucleo.nucleos[0].hall.coords = o.ref.coords;
                    }
                    continue;
                }
                const area = calculatePolyArea(o.verts.map(p => ({ x: p[0], y: p[1] })));
                if (o.ref) {
                    const moved = JSON.stringify(o.ref.coords) !== JSON.stringify(o.verts);
                    o.ref.coords = o.verts.map(p => [p[0], p[1]]);
                    if (moved) {
                        // ponytail: área = polígono bruto (sin descuento de muros);
                        // las zonas interiores quedan obsoletas al deformar → fuera
                        o.ref.metadata.area = area;
                        o.ref.zonas = [];
                    }
                    unidades.push(o.ref);
                } else {
                    nNew++;
                    const dorm = o.dormitorios || 2;
                    unidades.push({
                        id: `M${String(nNew).padStart(2, '0')}`, type: 'apartment',
                        coords: o.verts.map(p => [p[0], p[1]]), zonas: [],
                        metadata: { area, area_gross: area, tipologia: `${dorm}D`, habitantes: dorm + 1, lado: 'fondo', es_reducida: false },
                        validacion: { colinda_hall: true, cumple_area_min: area >= 40 },
                    });
                }
            }
            pl.unidades = unidades;
            // Regenerar etiquetas de dptos (posición/área cambiaron)
            pl.anotaciones = (pl.anotaciones || []).filter(a => a.clase !== 'etiqueta');
            for (const u of unidades) {
                const cen = [
                    u.coords.reduce((s, p) => s + p[0], 0) / u.coords.length,
                    u.coords.reduce((s, p) => s + p[1], 0) / u.coords.length,
                ];
                const tip = u.metadata.tipologia ? `${u.metadata.tipologia} · ` : '';
                pl.anotaciones.push({ pos: cen, texto: `${tip}DPTO ${u.id} · ${u.metadata.area.toFixed(1)}m²`, clase: 'etiqueta' });
            }
            const v = window.__viewer3d;
            if (v) v.renderProyecto(pl, window.metadataProyecto || {});

            // Sincronizar cuadro de áreas multifamiliar con las unidades editadas
            apartments = unidades.map(u => {
                const t = getTypology(u.metadata.area);
                return {
                    id: u.id, poly: null,
                    area: u.metadata.area,
                    typology: u.metadata.tipologia || t.name,
                    hab: u.metadata.habitantes || t.hab,
                };
            });
            updateTables();

            layoutFijado = true;
            _updateFijadoBadge();
        }
        editObjs = null; editSel = -1;
        return changed;
    }

    function _onEditPointerDown(e) {
        const v = window.__viewer3d;
        if (!v) return;
        if (editDraw) {
            editDraw.verts.push(v.screenToWorld(e.clientX, e.clientY));
            _renderEditSvg();
            return;
        }
        const oi = e.target?.dataset?.oi;
        if (oi === undefined) {
            editSel = -1; _renderEditSvg(); return;
        }
        editSel = parseInt(oi, 10);
        const vi = e.target?.dataset?.vi;
        if (vi !== undefined) {
            editDragV = parseInt(vi, 10);
        } else {
            editDragBody = {
                start: v.screenToWorld(e.clientX, e.clientY),
                orig: editObjs[editSel].verts.map(p => [p[0], p[1]]),
            };
        }
        editSvgEl.setPointerCapture(e.pointerId);
        e.preventDefault();
        _renderEditSvg();
    }

    function _onEditPointerMove(e) {
        const v = window.__viewer3d;
        if (!v) return;
        if (editDragV >= 0 && editSel >= 0) {
            editObjs[editSel].verts[editDragV] = v.screenToWorld(e.clientX, e.clientY);
            _renderEditSvg();
        } else if (editDragBody && editSel >= 0) {
            const w = v.screenToWorld(e.clientX, e.clientY);
            const dx = w[0] - editDragBody.start[0], dy = w[1] - editDragBody.start[1];
            editObjs[editSel].verts = editDragBody.orig.map(p => [p[0] + dx, p[1] + dy]);
            _renderEditSvg();
        } else if (editDraw) {
            editDraw.cursor = v.screenToWorld(e.clientX, e.clientY);
            _renderEditSvg();
        }
    }

    function _onEditPointerUp() {
        editDragV = -1; editDragBody = null;
    }

    function _editCloseDraw() {
        if (editDraw && editDraw.verts.length >= 3) {
            editObjs.push({ kind: 'unidad', ref: null, verts: editDraw.verts });
            editSel = editObjs.length - 1;
            const newObj = editObjs[editSel];
            _askDormitorios(n => { newObj.dormitorios = n; _renderEditSvg(); });
        }
        editDraw = null;
        editSvgEl.style.pointerEvents = 'none';
        _renderEditSvg();
    }

    function _askDormitorios(onPick) {
        document.getElementById('edit-dorm-modal')?.remove();
        const m = document.createElement('div');
        m.id = 'edit-dorm-modal';
        Object.assign(m.style, {
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
            zIndex: '28', background: 'white', border: '1px solid #d7dbe8', borderRadius: '8px',
            boxShadow: '0 8px 24px rgba(20,24,48,0.18)', padding: '14px 16px', width: '240px',
            fontSize: '12px', color: '#232741',
        });
        m.innerHTML = `<div style="font-weight:700;margin-bottom:10px;">¿Cuántos dormitorios tiene el dpto nuevo?</div>
            <div style="display:flex;gap:6px;">
                ${[1, 2, 3, 4].map(n => `<button data-n="${n}" style="flex:1;padding:8px 0;border:1px solid #4f46e5;border-radius:5px;background:#4f46e5;color:white;cursor:pointer;font-weight:700;font-size:13px;">${n}</button>`).join('')}
            </div>`;
        wrapper.appendChild(m);
        m.querySelectorAll('button').forEach(b => {
            b.onclick = () => { const n = parseInt(b.dataset.n, 10); m.remove(); onPick(n); };
        });
    }

    function _showMobileEditWarning() {
        document.getElementById('mobile-edit-warning')?.remove();
        const backdrop = document.createElement('div');
        backdrop.id = 'mobile-edit-warning';
        Object.assign(backdrop.style, {
            position: 'fixed', inset: '0', zIndex: '999',
            background: 'rgba(15, 17, 32, 0.42)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        });
        const m = document.createElement('div');
        Object.assign(m.style, {
            background: 'white', border: '1px solid #d7dbe8', borderRadius: '10px',
            boxShadow: '0 8px 24px rgba(20,24,48,0.18)', padding: '20px', width: 'min(300px, 84vw)',
            fontSize: '13px', color: '#232741', textAlign: 'center',
        });
        m.innerHTML = `
            <div style="font-weight:700;margin-bottom:8px;font-size:14px;">Modo Edición no disponible en móvil</div>
            <div style="opacity:0.8;margin-bottom:16px;">Para editar la distribución de departamentos, usa la versión de escritorio.</div>
            <button id="mobile-edit-warning-ok" style="padding:8px 20px;border:none;border-radius:6px;background:#4f46e5;color:white;cursor:pointer;font-weight:700;font-size:13px;">Entendido</button>`;
        backdrop.appendChild(m);
        document.body.appendChild(backdrop);
        const close = () => backdrop.remove();
        backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
        m.querySelector('#mobile-edit-warning-ok').onclick = close;
    }

    function _showAreaLimitModal(area) {
        document.getElementById('area-limit-modal')?.remove();
        const backdrop = document.createElement('div');
        backdrop.id = 'area-limit-modal';
        Object.assign(backdrop.style, {
            position: 'fixed', inset: '0', zIndex: '999',
            background: 'rgba(15, 17, 32, 0.42)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        });
        const m = document.createElement('div');
        Object.assign(m.style, {
            background: 'white', border: '1px solid #d7dbe8', borderRadius: '10px',
            boxShadow: '0 8px 24px rgba(20,24,48,0.18)', padding: '22px', width: 'min(340px, 86vw)',
            fontSize: '13px', color: '#232741', textAlign: 'center',
        });
        m.innerHTML = `
            <div style="font-weight:700;margin-bottom:10px;font-size:15px;">Estamos en etapa de desarrollo</div>
            <div style="opacity:0.85;margin-bottom:8px;line-height:1.5;">
                Por ahora solo llegamos a lotes de <b>menos de 1400 m² · 1 torre</b>.
            </div>
            <div style="opacity:0.85;margin-bottom:18px;line-height:1.5;">
                Tu lote tiene <b>${area.toFixed(0)} m²</b>. Pronto, sorpresas ✨
            </div>
            <button id="area-limit-ok" style="padding:9px 24px;border:none;border-radius:6px;background:#4f46e5;color:white;cursor:pointer;font-weight:700;font-size:13px;">Entendido</button>`;
        backdrop.appendChild(m);
        document.body.appendChild(backdrop);
        const close = () => backdrop.remove();
        backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
        m.querySelector('#area-limit-ok').onclick = close;
    }

    function _editDeleteSel() {
        // Hall no es borrable (siempre debe existir); dptos, ductos y pozos sí
        if (editSel < 0 || !editObjs[editSel] || editObjs[editSel].kind === 'hall') return;
        editObjs.splice(editSel, 1);
        editSel = -1;
        _renderEditSvg();
    }

    function _onEditKey(e) {
        if (e.target.closest?.('input, select, textarea')) return;
        if (e.key === 'Escape') {
            if (editDraw) { editDraw = null; editSvgEl.style.pointerEvents = 'none'; _renderEditSvg(); return; }
            // Cancelar sin guardar y apagar el toggle
            exitEditMode(false);
            toggleEditMode.checked = false;
            toggleEditMode.dispatchEvent(new Event('change', { bubbles: true }));
        } else if (e.key === 'Enter' && editDraw) {
            _editCloseDraw();
        } else if ((e.key === 'Delete' || e.key === 'Backspace') && editSel >= 0 && !editDraw) {
            _editDeleteSel();
        }
    }

    async function _editEvaluar() {
        const pl = window.webglPayload;
        if (!pl?.lote?.coords || !editObjs) return;
        const body = {
            lote_coords: pl.lote.coords,
            unidades: editObjs.filter(o => o.kind === 'unidad').map((o, i) => ({
                id: o.ref?.id || `M${i + 1}`,
                tipologia: o.ref?.metadata?.tipologia || '',
                coords: o.verts,
            })),
            hall_coords: editObjs.find(o => o.kind === 'hall')?.verts || [],
            corridors: (pl.circulacion?.corridors || []).map(c => c.coords).filter(c => c?.length >= 3),
            pozos_luz: editObjs.filter(o => o.kind === 'pozo').map(o => o.verts).filter(c => c?.length >= 3),
            retiro_lateral: params.retiroLateral || 0,
            retiro_posterior: params.retiroPosterior || 0,
            numero_pisos: params.pisos || 7,
            altura_piso: params.alturaPiso || 2.80,
        };
        try {
            const r = await fetch(`${API_BASE_URL}/evaluar-layout`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const res = await r.json();
            if (!r.ok) throw new Error(res?.detail || `Error ${r.status}`);
            _editShowEval(res);
        } catch (err) {
            showCalcError('No se pudo evaluar el layout: ' + err.message);
        }
    }

    function _editShowEval(res) {
        document.getElementById('edit-eval-panel')?.remove();
        const p = document.createElement('div');
        p.id = 'edit-eval-panel';
        Object.assign(p.style, {
            position: 'absolute', top: '52px', right: '10px', zIndex: '27',
            background: 'white', border: '1px solid #d7dbe8', borderRadius: '8px',
            boxShadow: '0 8px 24px rgba(20,24,48,0.18)', padding: '12px 14px',
            width: '290px', maxHeight: '70%', overflowY: 'auto',
            fontSize: '12px', color: '#232741',
        });
        const ok = res.n_criticos === 0;
        let html = `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;">
            <strong style="font-size:13px;">Evaluación</strong>
            <span style="font-weight:700;color:${ok ? '#15803d' : '#b91c1c'};">${(res.status || '').toUpperCase()} · ${res.score}/100</span>
        </div>`;
        if (!res.defectos?.length) {
            html += `<div style="color:#15803d;">Sin defectos: cumple los parámetros evaluados.</div>`;
        } else {
            for (const d of res.defectos) {
                const crit = d.severidad === 'critico';
                html += `<div style="margin:4px 0;padding:6px 8px;border-radius:5px;background:${crit ? 'rgba(185,28,28,0.08)' : 'rgba(35,39,65,0.06)'};">
                    <span style="font-weight:700;color:${crit ? '#b91c1c' : '#232741'};">${crit ? 'CRÍTICO' : 'menor'}</span> · ${d.descripcion}</div>`;
            }
        }
        const m = res.metricas || {};
        html += `<div style="margin-top:8px;color:#5a5f7a;">Eficiencia ${m.eficiencia ?? '—'}% · Circulación ${m.pct_circ ?? '—'}% · Acceso ${m.acceso_ok ?? '—'}/${m.acceso_total ?? '—'} dptos</div>`;
        html += `<button id="edit-eval-close" style="margin-top:8px;width:100%;padding:5px;border:1px solid #d7dbe8;border-radius:5px;background:#f4f5fa;cursor:pointer;font-size:11px;">Cerrar</button>`;
        p.innerHTML = html;
        wrapper.appendChild(p);
        p.querySelector('#edit-eval-close').onclick = () => p.remove();
    }

    function _renderEditSvg() {
        const v = window.__viewer3d;
        if (!editSvgEl || !v || !editObjs) return;
        editSvgEl.innerHTML = '';
        const NS = 'http://www.w3.org/2000/svg';

        // Paleta por tipo de elemento
        const KIND = {
            unidad: { stroke: '#4f46e5', fill: 'rgba(79,70,229,0.08)', selFill: 'rgba(79,70,229,0.22)', ink: '#4f46e5' },
            hall:   { stroke: '#232741', fill: 'rgba(35,39,65,0.14)', selFill: 'rgba(35,39,65,0.22)', ink: '#232741' },
            ducto:  { stroke: '#6b7280', fill: 'rgba(107,114,128,0.18)', selFill: 'rgba(107,114,128,0.30)', ink: '#4b5563' },
            pozo:   { stroke: '#2563eb', fill: 'rgba(37,99,235,0.16)', selFill: 'rgba(37,99,235,0.28)', ink: '#1d4ed8' },
        };
        editObjs.forEach((o, oi) => {
            const pts = o.verts.map(p => v.worldToScreen(p[0], p[1]));
            const sel = oi === editSel;
            const st = KIND[o.kind] || KIND.unidad;
            const poly = document.createElementNS(NS, 'polygon');
            poly.setAttribute('points', pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '));
            poly.setAttribute('fill', sel ? st.selFill : st.fill);
            poly.setAttribute('stroke', st.stroke);
            poly.setAttribute('stroke-width', sel ? '2.5' : '1.5');
            if (sel) poly.setAttribute('stroke-dasharray', '6,4');
            poly.setAttribute('data-oi', String(oi));
            poly.style.cursor = 'move';
            poly.style.pointerEvents = 'auto';
            editSvgEl.appendChild(poly);

            // Etiqueta al centroide
            const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
            const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
            const area = calculatePolyArea(o.verts.map(p => ({ x: p[0], y: p[1] })));
            const txt = document.createElementNS(NS, 'text');
            txt.setAttribute('x', cx.toFixed(1));
            txt.setAttribute('y', cy.toFixed(1));
            txt.setAttribute('text-anchor', 'middle');
            txt.setAttribute('font-size', o.kind === 'ducto' || o.kind === 'pozo' ? '9' : '10');
            txt.setAttribute('font-weight', '700');
            txt.setAttribute('fill', st.ink);
            txt.setAttribute('stroke', 'white');
            txt.setAttribute('stroke-width', '3');
            txt.setAttribute('paint-order', 'stroke');
            txt.style.pointerEvents = 'none';
            const label = {
                hall: 'HALL', ducto: 'DUCTO', pozo: 'POZO',
            }[o.kind] || `${o.ref?.id || 'nuevo'} · ${area.toFixed(1)}m²`;
            txt.textContent = label;
            editSvgEl.appendChild(txt);
        });

        // Cotas + handles solo del objeto seleccionado
        if (editSel >= 0 && editObjs[editSel]) {
            const o = editObjs[editSel];
            const n = o.verts.length;
            for (let i = 0; i < n; i++) {
                const [wx1, wy1] = o.verts[i];
                const [wx2, wy2] = o.verts[(i + 1) % n];
                const len = Math.hypot(wx2 - wx1, wy2 - wy1);
                const mp = v.worldToScreen((wx1 + wx2) / 2, (wy1 + wy2) / 2);
                const txt = document.createElementNS(NS, 'text');
                txt.setAttribute('x', mp.x.toFixed(1));
                txt.setAttribute('y', (mp.y - 6).toFixed(1));
                txt.setAttribute('text-anchor', 'middle');
                txt.setAttribute('font-size', '11');
                txt.setAttribute('font-weight', '600');
                txt.setAttribute('fill', '#232741');
                txt.setAttribute('stroke', 'white');
                txt.setAttribute('stroke-width', '3');
                txt.setAttribute('paint-order', 'stroke');
                txt.style.pointerEvents = 'none';
                txt.textContent = `${len.toFixed(1)}m`;
                editSvgEl.appendChild(txt);
            }
            o.verts.forEach((p, i) => {
                const sp = v.worldToScreen(p[0], p[1]);
                const c = document.createElementNS(NS, 'circle');
                c.setAttribute('cx', sp.x.toFixed(1));
                c.setAttribute('cy', sp.y.toFixed(1));
                c.setAttribute('r', '7');
                c.setAttribute('fill', 'white');
                c.setAttribute('stroke', '#232741');
                c.setAttribute('stroke-width', '2.5');
                c.setAttribute('data-oi', String(editSel));
                c.setAttribute('data-vi', String(i));
                c.style.cursor = 'grab';
                c.style.pointerEvents = 'auto';
                editSvgEl.appendChild(c);
            });
        }

        // Dibujo en curso
        if (editDraw && editDraw.verts.length > 0) {
            const pts = editDraw.verts.map(p => v.worldToScreen(p[0], p[1]));
            const all = editDraw.cursor
                ? [...pts, v.worldToScreen(editDraw.cursor[0], editDraw.cursor[1])] : pts;
            const pline = document.createElementNS(NS, 'polyline');
            pline.setAttribute('points', all.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '));
            pline.setAttribute('fill', 'rgba(79,70,229,0.10)');
            pline.setAttribute('stroke', '#4f46e5');
            pline.setAttribute('stroke-width', '2');
            pline.setAttribute('stroke-dasharray', '6,4');
            pline.style.pointerEvents = 'none';
            editSvgEl.appendChild(pline);
            pts.forEach(p => {
                const c = document.createElementNS(NS, 'circle');
                c.setAttribute('cx', p.x.toFixed(1));
                c.setAttribute('cy', p.y.toFixed(1));
                c.setAttribute('r', '4');
                c.setAttribute('fill', '#4f46e5');
                c.style.pointerEvents = 'none';
                editSvgEl.appendChild(c);
            });
        }

        const hint = document.getElementById('edit-hint');
        if (hint) {
            hint.textContent = editDraw
                ? 'Click: vértice · doble click / Enter: cerrar · Esc: cancelar dibujo'
                : (editSel >= 0
                    ? 'Arrastra vértices o el cuerpo · Supr: eliminar'
                    : 'Click en un dpto o el hall para seleccionar · Esc: salir sin guardar');
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
            alturaPiso: parseFloat(document.getElementById("altura-piso")?.value) || 2.80,
            retiroLateral: (() => { const v = parseFloat(document.getElementById("retiro-lateral")?.value); return isNaN(v) ? 2.30 : v; })(),
            retiroPosterior: (() => { const v = parseFloat(document.getElementById("retiro-posterior")?.value); return isNaN(v) ? 2.30 : v; })(),
        };

        // === RNE A.010: Derived normative values ===
        let H = params.pisos * params.alturaPiso;
        params.H = H;
        params.pozoDormMin = Math.max(2.20, H / 3);  // RNE A.010: dormitorios d >= H/3, mín 2.20m
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
        let pozoMinReq = params.pozoDormMin; // H/3 min
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

            // Área/tipología ya vienen del backend (cuadro_unidades, ver ejecutarAuditoriaRNE);
            // finalArea (turf.difference local) solo sirve para el fallback sin backend.
            if (!ap.area) ap.area = Math.max(0, finalArea);
            let t = getTypology(ap.area);
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

            // CHEQUEO DE CONECTIVIDAD: Toca algún Hall o Core?
            // Multi-torre: window.nucleosConectividad agrega todos los halls+cores.
            // Fallback (sin backend, dibujo previo): hall/core únicos.
            let nucleos = window.nucleosConectividad;
            if (!nucleos || nucleos.length === 0) {
                nucleos = [];
                if (corePoly.length >= 3) nucleos.push(corePoly);
                if ((window.hallProcedural || []).length >= 3) nucleos.push(window.hallProcedural);
            }
            let connected = false;
            let turfUnit = polyToTurf(ap.poly);
            if (turfUnit) {
                for (const nuc of nucleos) {
                    if (connected) break;
                    let turfNuc = polyToTurf(nuc);
                    if (!turfNuc) continue;
                    try {
                        let buffered = turf.buffer(turfNuc, 0.30, { units: 'meters' });
                        if (turf.intersect(buffered, turfUnit)) connected = true;
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

        // Área común: usar cuadro_areas del backend (área comun real validada) si existe;
        // el fallback local (corePoly) solo aplica sin backend (dibujo previo a generar).
        let hallArea = rneResultado?.cuadro_areas?.area_comun_planta_m2;
        if (hallArea == null) {
            hallArea = calculatePolyArea(corePoly);
            if (hallArea < 5 && params.dptosPlanta > 1) hallArea = 15;
        }

        let actDuctArea = 0;
        smallDuctos.forEach(d => actDuctArea += calculatePolyArea(d));
        if (patioPoly.length) actDuctArea += calculatePolyArea(patioPoly);

        calc.realVendiblePlanta = totalVendible;
        calc.realComunPlanta = hallArea;
        calc.realTotalPlanta = totalVendible + hallArea;

        return true;
    }

    // Renderiza resultado.diseno (evaluación server-side de main.py _evaluar_diseno)
    // sin fusionar con la auditoría cliente-side existente (H4: mostrar ambos).
    function renderBackendEvaluacion(diseno) {
        let panel = document.getElementById('backend-eval-panel');
        if (!panel) return;
        if (!diseno || typeof diseno.score !== 'number') {
            panel.style.display = 'none';
            return;
        }
        let defectos = diseno.defectos || [];
        let sevIcon = s => s === 'critico' ? '🔴' : '🟡';
        let rows = defectos.map(d =>
            `<div style="margin-top:4px;">${sevIcon(d.severidad)} <strong>${d.tipo}</strong>: ${d.descripcion}</div>`
        ).join('') || '<div class="status-ok" style="margin-top:4px;">Sin defectos detectados.</div>';
        panel.innerHTML = `
            <div style="font-weight:700;margin-bottom:4px;">Evaluación motor (score ${diseno.score}/100)</div>
            ${rows}
        `;
        panel.style.display = 'block';
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
        let estLogrados = sot.count_total ?? (sot.count || 0);
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
            (sot.num_niveles || 1) > 1
                ? `${sot.num_niveles} sótanos (S1..S${sot.num_niveles})`
                : `Sótano ${sot.name || 'S1'}`);

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

        let mv = rneResultado?.normativa_estricta?.mivivienda_check;
        if (mv?.aplica) {
            html += row('Fondo Mivivienda — Checklist',
                `${mv.n_unidades - mv.n_bajo_area_min}/${mv.n_unidades} unidades ≥ ${mv.area_min_m2}m² · ${mv.n_sin_precio_definido} sin precio definido · ${mv.n_sobre_tope_precio} sobre tope`,
                mv.cumple_global,
                mv.nota);
        }

        html += '</div>';
        panel.innerHTML = html;
        panel.style.display = '';
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
                    ${fallas.length ? `<div class="status-fail">${fallas.join(' · ')}</div>` : ''}
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
                rneAsc.innerHTML = '<span class="status-fail">Obligatorio (&gt;4 pisos)</span>';
            } else {
                rneAsc.innerHTML = '<span class="status-ok">No requerido (≤4 pisos)</span>';
            }
        }
        if (rneEsc) {
            if (params.escaleraPresurizada) {
                rneEsc.innerHTML = '<span class="status-fail">Presurizada 🔒</span>';
            } else {
                rneEsc.innerHTML = '<span class="status-ok">Abierta ✓</span>';
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
                    : '<span class="text-muted">○ Interior</span>';
                const escOk = u.dist_esc_cumple
                    ? `<span class="status-ok">✓</span>`
                    : `<span class="status-fail">✗ >${u.dist_escalera_m?.toFixed(1)}m</span>`;
                const lado = u.lado === 'frente' ? 'Frente' : 'Fondo';
                let obs = [];
                if (u.es_reducida) obs.push('<span class="status-warn">Recortada</span>');
                if (!u.cumple_area_min) obs.push('<span class="status-fail">Sub-mín.</span>');
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
            acogido_mivivienda: !!document.getElementById('acogido-mivivienda')?.checked,
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
                showCalcError("No se pudo generar la cabida: " + detalle);
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

                renderBackendEvaluacion(resultado.diseno);

                // ═══ INYECCIÓN DE GEOMETRÍA (para datos/tablas) ═══
                if (resultado.geometria_generada) {
                    const geo = resultado.geometria_generada;
                    corePoly = geo.core || [];
                    window.hallProcedural = geo.hall || [];
                    // Conectividad: en topologías multi-torre (dos/N núcleos) cada
                    // torre tiene su propio hall+core. Agregarlos todos para no
                    // marcar como "isla" a las unidades que tocan otro núcleo.
                    {
                        const _halls = (geo.halls && geo.halls.length) ? geo.halls : (geo.hall ? [geo.hall] : []);
                        const _cores = (geo.cores && geo.cores.length) ? geo.cores : (geo.core ? [geo.core] : []);
                        window.nucleosConectividad = [..._halls, ..._cores].filter(p => Array.isArray(p) && p.length >= 3);
                    }
                    // Conectividad: en topologías multi-torre (dos_nucleos) cada torre
                    // tiene su propio hall+core. Agregarlos todos para no marcar como
                    // "isla" a las unidades que tocan el núcleo de la 2ª torre.
                    {
                        const _halls = (geo.halls && geo.halls.length) ? geo.halls : (geo.hall ? [geo.hall] : []);
                        const _cores = (geo.cores && geo.cores.length) ? geo.cores : (geo.core ? [geo.core] : []);
                        window.nucleosConectividad = [..._halls, ..._cores].filter(p => Array.isArray(p) && p.length >= 3);
                    }
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
                        // Áreas/tipología: usar SIEMPRE cuadro_unidades (motor Python) como
                        // fuente única de verdad -- el mismo id (X01, X02...) identifica la
                        // unidad ahí y en geo.departamentos. Evita que el cliente recalcule
                        // el área vía turf.difference y diverja del valor validado backend.
                        let unidadesById = {};
                        (resultado.cuadro_unidades || []).forEach(u => { unidadesById[u.id] = u; });

                        apartments = geo.departamentos
                            .map((entry, i) => {
                                let contorno = Array.isArray(entry) ? entry : (entry && entry.contorno);
                                let zonas = (!Array.isArray(entry) && entry && Array.isArray(entry.zonas)) ? entry.zonas : [];
                                let tipologia = (!Array.isArray(entry) && entry && entry.tipologia) ? entry.tipologia : '';
                                if (!contorno || contorno.length < 3) return null;
                                let id = `X${String(i + 1).padStart(2, '0')}`;
                                let u = unidadesById[id];
                                return {
                                    id,
                                    poly: contorno,
                                    zonas: zonas,
                                    area: u ? u.area_neta_m2 : 0,
                                    typology: (u && u.tipologia) || tipologia || '3D',
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
            showCalcError("No se pudo conectar con el motor de cálculo. Si es la primera vez del día, el servidor puede tardar ~30s en despertar — intenta de nuevo. (" + error.message + ")");
        }
    }



    // --- Actions ---
    btnGenerateAI.addEventListener("click", async (e) => {
        // PREVIENE RECARGAS
        e.preventDefault();

        // Guard: recalcular reemplaza el diseño editado a mano
        if (layoutFijado && !confirm('Tienes un diseño editado a mano (fijado). Recalcular la distribución lo reemplazará por completo. ¿Continuar?')) {
            return;
        }
        layoutFijado = false;
        _updateFijadoBadge();
        closeMobileSidebar();

        // Tope etapa de desarrollo: por ahora solo 1 torre hasta 1400 m².
        const areaLote = calc.areaTerreno || 0;
        if (areaLote > 1400) {
            _showAreaLimitModal(areaLote);
            return;
        }

        // Validación previa: evita el round trip de ~30s si la combinación es inválida
        const errorValidacion = validarInputsPrevios(params);
        if (errorValidacion) {
            showCalcError(errorValidacion);
            return;
        }

        // APAGA EL DIBUJO LOCAL COMPLETAMENTE
        isGenerated = false;

        btnGenerateAI.classList.add("hidden");
        aiLoader.classList.remove("hidden");
        hideCalcError();

        // El Canvas SOLO debe mostrar el polígono del lote por ahora
        try {
            // Llamada ÚNICA Y EXCLUSIVA a Python (El puente inyectará los datos y re-encenderá el motor)
            await ejecutarAuditoriaRNE();
            if (window.__tourOnRecalcDone) {
                const cb = window.__tourOnRecalcDone;
                window.__tourOnRecalcDone = null;
                cb();
            }
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
        if (e.target.checked && window.matchMedia('(max-width: 880px)').matches) {
            e.target.checked = false;
            _showMobileEditWarning();
            return;
        }
        isEditMode = e.target.checked;

        if (isEditMode) {
            if (!enterEditMode()) {
                // No hay geometría o está en 3D: revertir toggle
                isEditMode = false;
                e.target.checked = false;
                return;
            }
            document.getElementById('viewport-title').innerText = "Modo Edición";
        } else {
            document.getElementById('viewport-title').innerText = "Vista:";
            // Guarda ediciones en el payload y re-renderiza el viewer.
            // NO llama a ejecutarAuditoriaRNE: regenerar pisaría lo editado;
            // la validación se hace con el botón "Evaluar" dentro del modo.
            exitEditMode(true);
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
                    const niveles = (sot.niveles && sot.niveles.length)
                        ? sot.niveles
                        : [{ name: sot.name || 'S1', count: sot.count || 0 }];
                    sotanoLevelSelect.innerHTML = niveles
                        .map((nv, i) => `<option value="${i}">${nv.name} (${nv.count} est.)</option>`)
                        .join('');
                    getViewer3D()?.setSotanoLevel(0);
                }
            }
        });
    }
    if (sotanoLevelSelect) {
        sotanoLevelSelect.addEventListener("change", () => {
            getViewer3D()?.setSotanoLevel(parseInt(sotanoLevelSelect.value, 10) || 0);
        });
    }

    // Reemplaza el popup nativo del <select> (no skineable por CSS) por un listbox propio.
    // El <select> original queda oculto pero funcional: mismo .value, mismo evento 'change'.
    function initCustomSelect(selectEl) {
        if (!selectEl || selectEl.dataset.customSelectReady) return;
        selectEl.dataset.customSelectReady = '1';
        const isSm = selectEl.classList.contains('sm');

        const wrapper = document.createElement('div');
        wrapper.className = 'custom-select' + (isSm ? ' sm' : '');
        selectEl.parentNode.insertBefore(wrapper, selectEl);
        selectEl.classList.add('custom-select-native');
        wrapper.appendChild(selectEl);

        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'custom-select-trigger';
        trigger.setAttribute('aria-haspopup', 'listbox');
        trigger.setAttribute('aria-expanded', 'false');
        const triggerLabel = document.createElement('span');
        trigger.appendChild(triggerLabel);
        trigger.insertAdjacentHTML('beforeend',
            '<svg class="custom-select-chevron" viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><path d="M6 9l6 6 6-6"/></svg>');
        wrapper.appendChild(trigger);

        const listbox = document.createElement('div');
        listbox.className = 'custom-select-listbox';
        listbox.setAttribute('role', 'listbox');
        listbox.hidden = true;
        wrapper.appendChild(listbox);

        function renderOptions() {
            listbox.innerHTML = '';
            Array.from(selectEl.options).forEach(opt => {
                const item = document.createElement('div');
                item.className = 'custom-select-option' + (opt.value === selectEl.value ? ' active' : '');
                item.setAttribute('role', 'option');
                item.textContent = opt.textContent;
                item.addEventListener('click', () => {
                    if (selectEl.value !== opt.value) {
                        selectEl.value = opt.value;
                        selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    close();
                });
                listbox.appendChild(item);
            });
        }

        function syncTrigger() {
            const opt = selectEl.options[selectEl.selectedIndex];
            triggerLabel.textContent = opt ? opt.textContent : '';
        }

        function syncVisibility() {
            wrapper.style.display = selectEl.style.display === 'none' ? 'none' : '';
        }

        function position() {
            const rect = trigger.getBoundingClientRect();
            listbox.style.left = rect.left + 'px';
            listbox.style.top = (rect.bottom + 4) + 'px';
            listbox.style.minWidth = rect.width + 'px';
        }
        function open() {
            renderOptions();
            position();
            listbox.hidden = false;
            wrapper.classList.add('open');
            trigger.setAttribute('aria-expanded', 'true');
            document.addEventListener('mousedown', onOutsideClick, true);
            document.addEventListener('keydown', onKeydown, true);
            window.addEventListener('scroll', close, true);
            window.addEventListener('resize', close);
        }
        function close() {
            listbox.hidden = true;
            wrapper.classList.remove('open');
            trigger.setAttribute('aria-expanded', 'false');
            document.removeEventListener('mousedown', onOutsideClick, true);
            document.removeEventListener('keydown', onKeydown, true);
            window.removeEventListener('scroll', close, true);
            window.removeEventListener('resize', close);
        }
        function onOutsideClick(e) {
            if (!wrapper.contains(e.target)) close();
        }
        function onKeydown(e) {
            if (e.key === 'Escape') close();
        }

        trigger.addEventListener('click', () => {
            if (listbox.hidden) open(); else close();
        });

        // El resto del código sigue mutando el <select> original directamente
        // (innerHTML de opciones, style.display show/hide) — lo reflejamos aquí.
        new MutationObserver(() => { renderOptions(); syncTrigger(); syncVisibility(); })
            .observe(selectEl, { attributes: true, attributeFilter: ['style'], childList: true });

        syncTrigger();
        syncVisibility();
    }

    initCustomSelect(viewportLevelSelect);
    initCustomSelect(sotanoLevelSelect);

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
            hint.style.color = 'var(--brand-blue)';
            hint.style.fontWeight = '600';
        } else {
            hint.textContent = 'Medianería / Fachada Interior';
            hint.style.color = 'var(--text-secondary)';
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

    // ═══════════════════════════════════════════════════════
    //  VISITA GUIADA
    // ═══════════════════════════════════════════════════════
    const TOUR_KEY = 'cabidaTourCompletado';
    const WHATSAPP_URL = 'https://wa.me/51973903176';
    const isMobileViewport = () => window.matchMedia('(max-width: 880px)').matches;

    function _tourCleanupDOM() {
        document.getElementById('tour-overlay')?.remove();
        document.getElementById('tour-card')?.remove();
        document.querySelectorAll('.tour-highlight').forEach(el => el.classList.remove('tour-highlight'));
    }

    function _tourSkip() {
        window.__tourOnRecalcDone = null;
        _tourCleanupDOM();
    }

    function _tourHighlight(el) {
        document.querySelectorAll('.tour-highlight').forEach(e => e.classList.remove('tour-highlight'));
        if (el) el.classList.add('tour-highlight');
    }

    function _tourShowCard(target, title, body, onContinue) {
        document.getElementById('tour-card')?.remove();
        if (!document.getElementById('tour-overlay')) {
            const overlay = document.createElement('div');
            overlay.id = 'tour-overlay';
            document.body.appendChild(overlay);
        }
        const card = document.createElement('div');
        card.id = 'tour-card';
        card.innerHTML = `
            <h4>${title}</h4>
            <p>${body}</p>
            <div class="tour-actions">
                <button class="tour-btn-skip" id="tour-skip-btn">Omitir guía</button>
                <button class="tour-btn-primary" id="tour-continue-btn">Continuar</button>
            </div>`;
        document.body.appendChild(card);

        if (target) {
            const r = target.getBoundingClientRect();
            const cardW = 300;
            let left = Math.min(Math.max(r.left, 12), window.innerWidth - cardW - 12);
            let top = r.bottom + 12;
            if (top + 160 > window.innerHeight) top = Math.max(12, r.top - 172);
            card.style.left = `${left}px`;
            card.style.top = `${top}px`;
        } else {
            card.style.left = '50%';
            card.style.top = '50%';
            card.style.transform = 'translate(-50%, -50%)';
        }

        document.getElementById('tour-skip-btn').onclick = _tourSkip;
        document.getElementById('tour-continue-btn').onclick = onContinue;
        _tourHighlight(target);
    }

    function _showTourThanksModal() {
        document.getElementById('tour-thanks-modal')?.remove();
        const backdrop = document.createElement('div');
        backdrop.id = 'tour-thanks-modal';
        Object.assign(backdrop.style, {
            position: 'fixed', inset: '0', zIndex: '2010',
            background: 'rgba(15, 17, 32, 0.42)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        });
        const m = document.createElement('div');
        Object.assign(m.style, {
            background: 'white', border: '1px solid #d7dbe8', borderRadius: '10px',
            boxShadow: '0 8px 24px rgba(20,24,48,0.18)', padding: '22px', width: 'min(340px, 86vw)',
            fontSize: '13px', color: '#232741', textAlign: 'center',
        });
        m.innerHTML = `
            <div style="font-weight:700;margin-bottom:10px;font-size:15px;">¡Gracias!</div>
            <div style="opacity:0.85;margin-bottom:8px;line-height:1.5;">
                Puedes seguir probando la app. Recuerda que estamos en fase beta:
                nos ayudarías mucho con tu feedback.
            </div>
            <a class="whatsapp-link" href="${WHATSAPP_URL}" target="_blank" rel="noopener">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12.04 2C6.58 2 2.13 6.45 2.13 11.91c0 1.75.46 3.45 1.32 4.95L2 22l5.29-1.39a9.87 9.87 0 0 0 4.75 1.21h.01c5.46 0 9.9-4.45 9.9-9.91 0-2.65-1.03-5.14-2.9-7.01A9.87 9.87 0 0 0 12.04 2zm0 18.06h-.01a8.2 8.2 0 0 1-4.18-1.14l-.3-.18-3.12.82.84-3.04-.2-.31a8.2 8.2 0 0 1-1.26-4.4c0-4.54 3.7-8.24 8.25-8.24a8.2 8.2 0 0 1 5.83 2.42 8.18 8.18 0 0 1 2.41 5.83c0 4.55-3.7 8.24-8.26 8.24zm4.52-6.17c-.25-.12-1.47-.72-1.7-.81-.23-.08-.4-.12-.56.13-.17.25-.64.81-.79.97-.14.17-.29.19-.54.06-.25-.12-1.04-.38-1.98-1.22-.73-.65-1.23-1.46-1.37-1.7-.14-.25-.02-.38.11-.51.11-.11.25-.29.37-.43.12-.15.16-.25.25-.42.08-.16.04-.31-.02-.43-.06-.13-.56-1.35-.77-1.85-.2-.48-.41-.42-.56-.42h-.48c-.16 0-.43.06-.66.31-.22.25-.87.85-.87 2.07 0 1.22.89 2.4 1.02 2.57.12.16 1.75 2.67 4.24 3.75.59.25 1.05.4 1.41.52.59.19 1.13.16 1.56.1.47-.07 1.47-.6 1.68-1.18.21-.58.21-1.07.14-1.18-.06-.1-.23-.16-.48-.28z"/></svg>
                Danos tu feedback
            </a>`;
        backdrop.appendChild(m);
        document.body.appendChild(backdrop);
        backdrop.addEventListener('click', (e) => { if (e.target === backdrop) backdrop.remove(); });
    }

    function _tourFinish() {
        _tourCleanupDOM();
        localStorage.setItem(TOUR_KEY, '1');
        _showTourThanksModal();
    }

    function _tourStep4() {
        _tourShowCard(btnGenerateAI,
            'Calcula tu distribución',
            'Si ya elegiste tus variables, podrás ver tu distribución.',
            () => {
                _tourCleanupDOM();
                window.__tourOnRecalcDone = _tourFinish;
                btnGenerateAI.click();
            });
    }

    function _tourStep3() {
        const drawToolbar = document.querySelector('#leaflet-map-container .leaflet-pm-toolbar')
            || document.getElementById('leaflet-map-container');
        _tourShowCard(drawToolbar,
            'Dibuja el lote',
            'Usa las herramientas para dibujar el lote.',
            () => {
                showPlanView();
                _tourStep4();
            });
    }

    function _tourStep2() {
        _tourShowCard(btnToggleView,
            'Explora en el mapa',
            'También puedes dibujar la poligonal en algún terreno que viste en Lima!',
            () => {
                btnToggleView.click();
                setTimeout(_tourStep3, 200);
            });
    }

    function _tourStep1() {
        const target = document.getElementById('panel-dimensiones-terreno');
        _tourShowCard(target,
            '¡Inserta las dimensiones del terreno!',
            'También puedes llenar parámetros normativos y diseño inmobiliario.',
            _tourStep2);
    }

    function _startGuidedTour() {
        _tourStep1();
    }

    if (localStorage.getItem(TOUR_KEY) !== '1') {
        if (isMobileViewport()) {
            const toast = document.createElement('div');
            toast.id = 'tour-mobile-toast';
            toast.textContent = 'Para probar el tour entra desde la versión de escritorio!';
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 5000);
        } else {
            setTimeout(_startGuidedTour, 600);
        }
    }
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
