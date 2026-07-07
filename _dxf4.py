import ezdxf, json
from shapely.geometry import Polygon, Point

def closed_polys(msp):
    out = []
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                try:
                    pg = Polygon(pts)
                    if pg.is_valid and pg.area > 0.5:
                        out.append((e.dxf.layer, pg, e.closed))
                except Exception:
                    pass
    return out

for n in range(1, 7):
    doc = ezdxf.readfile(f"referencias/{n}.dxf")
    msp = doc.modelspace()
    polys = closed_polys(msp)
    txts = []
    for e in msp:
        if e.dxftype() == "TEXT":
            p = e.dxf.insert
            txts.append((e.dxf.text.strip(), p.x, p.y))
    print(f"\n========= {n}.dxf  polys={len(polys)}")
    # lote
    for lay, pg, cl in polys:
        if "PERIMETRAL" in lay.upper():
            b = pg.bounds
            print(f"  LOTE: {b[2]-b[0]:.2f} x {b[3]-b[1]:.2f}  area={pg.area:.1f}")
    # dptos: polys con texto de área dentro
    for lay, pg, cl in polys:
        if "PERIMETRAL" in lay.upper():
            continue
        inside = [t for t in txts if pg.buffer(0.3).contains(Point(t[1], t[2]))]
        m2 = [t[0] for t in inside if "m²" in t[0] or "m2" in t[0]]
        tip = [t[0] for t in inside if t[0][:2] in ("1D","2D","3D") and "m" not in t[0]]
        nom = [t[0] for t in inside if t[0].startswith("DPTO")]
        b = pg.bounds
        tag = (tip[0] if tip else "") + " " + (m2[0] if m2 else "") + " " + (nom[0] if nom else "")
        print(f"  [{lay}] {b[2]-b[0]:5.2f}x{b[3]-b[1]:5.2f} area_geo={pg.area:7.1f} {tag.strip()}")
