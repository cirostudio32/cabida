import ezdxf, re
from collections import defaultdict

for i in range(1, 7):
    doc = ezdxf.readfile(f'referencias/{i}.dxf')
    msp = doc.modelspace()

    lote_w = lote_h = 0
    lote_pts = None
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE' and e.dxf.layer == 'LINEA PERIMETRAL LEVANTAMIENTO':
            pts = list(e.get_points('xy'))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            lote_w = round(max(xs)-min(xs), 2)
            lote_h = round(max(ys)-min(ys), 2)

    at_ao = {}
    for e in msp:
        if e.dxftype() != 'TEXT':
            continue
        t = e.dxf.text.strip()
        nums = re.findall(r'[\d.]+', t)
        if t.startswith('AT') and nums:
            at_ao['AT'] = float(nums[-1])
        if t.startswith('AO') and nums:
            at_ao['AO'] = float(nums[-1])

    dpto_ids = set()
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.layer == 'MA_0.40':
            t = e.dxf.text.strip()
            if t.startswith('DPTO'):
                dpto_ids.add(t)

    m2_vals = []
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.layer == 'MA_TEXTO 1':
            t = e.dxf.text.strip()
            m = re.match(r'^([\d.]+)\s*m', t)
            if m:
                m2_vals.append(float(m.group(1)))

    print(f"DXF {i}: {lote_w}x{lote_h}m | nd/piso={len(dpto_ids)} | AT={at_ao.get('AT','?')} AO={at_ao.get('AO','?')} | areas={sorted(set(round(v,1) for v in m2_vals))}")
