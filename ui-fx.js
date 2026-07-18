// Capa de animación (GSAP) — puramente presentacional, no toca lógica de main.js
(function () {
    if (typeof gsap === 'undefined') return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Tokens de easing compartidos (cubic-beziers "fuertes", ver AUDIT.md improve-animations)
    const EASE_OUT = 'power3.out';
    const EASE_INOUT = 'power2.inOut';

    gsap.defaults({ duration: 0.35, ease: EASE_OUT });

    // ── 1. Entrada inicial (se salta con reduced-motion: es puro adorno de carga) ──
    if (!reduced) {
        gsap.from('.top-nav', { y: -24, autoAlpha: 0, duration: 0.5, ease: EASE_OUT, clearProps: 'transform' });
        gsap.from('.accordion-panel', {
            y: 16, autoAlpha: 0, duration: 0.5, stagger: 0.06, ease: EASE_OUT, delay: 0.1, clearProps: 'transform',
        });
        gsap.from('.toolbar, .workspace-tabs', {
            y: -10, autoAlpha: 0, duration: 0.4, stagger: 0.06, delay: 0.15, clearProps: 'transform',
        });
    }

    // ── 2. Micro-interacciones de botones (delegado, cubre elementos dinámicos) ──
    // Solo el CTA primario lleva scale/lift (baja frecuencia); botones de toolbar/tabs
    // ya tienen su feedback de color en CSS — no se les suma movimiento (se hoverean
    // decenas de veces al día).
    const LIFT_SEL = '.btn-primary';
    const PRESS_SEL = '.btn-primary, .btn-tool';

    if (!reduced) {
        document.addEventListener('mouseover', (e) => {
            const el = e.target.closest(LIFT_SEL);
            if (el) gsap.to(el, { scale: 1.02, y: -2, overwrite: 'auto', duration: 0.2, ease: 'power2.out' });
        });
        document.addEventListener('mouseout', (e) => {
            const el = e.target.closest(LIFT_SEL);
            if (el && !el.contains(e.relatedTarget)) gsap.to(el, { scale: 1, y: 0, overwrite: 'auto', duration: 0.25, ease: 'power2.out' });
        });
    }
    document.addEventListener('mousedown', (e) => {
        const el = e.target.closest(PRESS_SEL);
        if (el) gsap.to(el, { scale: reduced ? 1 : 0.96, duration: 0.1, overwrite: 'auto', ease: 'power2.out' });
    });
    document.addEventListener('mouseup', (e) => {
        const el = e.target.closest(PRESS_SEL);
        if (!el) return;
        // Vuelve al mismo estado que dejaría el hover (evita pisarse con el listener de arriba)
        const restingScale = !reduced && el.matches(LIFT_SEL) && el.matches(':hover') ? 1.02 : 1;
        gsap.to(el, { scale: restingScale, duration: 0.18, ease: 'back.out(1.7)', overwrite: 'auto' });
    });

    // ── 3. Toggle switches: solo color de fondo (CSS) — sin rebote extra, se disparan seguido ──

    // ── 4. Transición al cambiar de tab ──
    function fadeIn(el) {
        gsap.fromTo(el, { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: 0.35, ease: 'power2.out', clearProps: 'transform' });
    }
    document.querySelectorAll('.tab-btn').forEach((btn) => {
        btn.addEventListener('click', () => requestAnimationFrame(() => {
            const active = document.querySelector('.tab-content.active');
            if (active) fadeIn(active);
        }));
    });
    document.querySelectorAll('.ws-tab').forEach((btn) => {
        btn.addEventListener('click', () => requestAnimationFrame(() => {
            document.querySelectorAll('.ws-tab-content').forEach((c) => {
                if (c.style.display !== 'none') fadeIn(c);
            });
        }));
    });

    // ── 5. Entrada de paneles flotantes / alertas al hacerse visibles ──
    const panelObserver = new MutationObserver((mutations) => {
        mutations.forEach((m) => {
            const el = m.target;
            const visible = getComputedStyle(el).display !== 'none' && !el.classList.contains('hidden');
            if (visible && !el.dataset.fxShown) {
                el.dataset.fxShown = '1';
                gsap.fromTo(el, { autoAlpha: 0, scale: 0.94, y: 6 }, { autoAlpha: 1, scale: 1, y: 0, duration: 0.3, ease: 'back.out(1.6)' });
            } else if (!visible) {
                el.dataset.fxShown = '';
            }
        });
    });
    document.querySelectorAll('.floating-panel, .ai-loader').forEach((el) => {
        panelObserver.observe(el, { attributes: true, attributeFilter: ['style', 'class'] });
    });

    // ── 6. Contadores KPI animados (count-up) ──
    // ponytail: parseo numérico simple; si el texto trae formato no numérico se ignora la animación.
    const KPI_IDS = [
        'kpi-terreno-bar', 'kpi-techada-bar', 'kpi-vendible-bar', 'kpi-eficiencia-bar', 'kpi-libre-bar', 'kpi-dptos-bar',
        'kpi-terreno', 'kpi-techada-total', 'kpi-vendible-total', 'kpi-eficiencia',
    ];
    KPI_IDS.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        let last = parseFloat(el.textContent) || 0;
        const obs = new MutationObserver(() => {
            if (el.dataset.fxAnimating) return;
            const target = parseFloat((el.textContent || '').replace(/,/g, ''));
            if (isNaN(target) || target === last) return;
            const decimals = target % 1 === 0 ? 0 : 2;
            const proxy = { val: last };
            el.dataset.fxAnimating = '1';
            gsap.to(proxy, {
                val: target,
                duration: 0.6,
                ease: EASE_INOUT,
                onUpdate: () => { el.textContent = proxy.val.toFixed(decimals); },
                onComplete: () => { el.textContent = target.toFixed(decimals); last = target; delete el.dataset.fxAnimating; },
            });
        });
        obs.observe(el, { childList: true, characterData: true, subtree: true });
    });
})();
