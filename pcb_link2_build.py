"""
Two 2x2 via-fed PCB arrays (pcb_array2x2_build), separated for a TX->RX coupling
(S21) study. Each array is a full independent board (own ground, feed, port); they
are placed SEP mm apart along y. Port 1 = TX (excited), Port 2 = RX (matched load).

Interface: build_antenna(CSX, FDTD); add_feed_ports(FDTD) -> (port_tx, port_rx);
           f0, fc, feed_R, unit, z_top, SEP.

Run:  python pcb_link2_build.py   - AppCSXCAD with both feedpoints marked
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

import pcb_array2x2_build as a          # one-array geometry (imported values reused)

# ======================= LINK PARAMETERS (mm) =======================
SEP = 200.0                            # centre-to-centre separation (along y)
# Generous air so an all-PML boundary sits in the far field with a buffer beyond
# the ~28 mm PML_8 thickness (needed for accurate low-level S21).
air_xy, air_above, air_below = 55.0, 45.0, 30.0
mesh_res = (a.C0 / (a.f0 + a.fc)) / a.unit / 20.0
# ====================================================================

a._derive()                            # make sure the array geometry is current
f0, fc, feed_R, unit = a.f0, a.fc, a.feed_R, a.unit
z_feed, z_gnd, z_drv, z_air_top, z_par, z_top = a.z_feed, a.z_gnd, a.z_drv, a.z_air_top, a.z_par, a.z_top

oyA, oyB = -SEP/2, SEP/2               # array centres in y


def _box(prop, x0, y0_, x1, y1, z0, z1, prio=10):
    prop.AddBox([min(x0, x1), min(y0_, y1), min(z0, z1)],
                [max(x0, x1), max(y0_, y1), max(z0, z1)], priority=prio)


def _sheet(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([min(x0, x1), min(y0_, y1), z], [max(x0, x1), max(y0_, y1), z], priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    protect = protect or {}

    def protected(dirc, v):
        return any(abs(v - p) <= 1e-6 for p in protect.get(dirc, []))

    for dirc in 'xyz':
        lines = np.unique(np.asarray(mesh.GetLines(dirc), dtype=float))
        if lines.size < 3:
            continue
        kept = [lines[0]]
        for x in lines[1:-1]:
            if protected(dirc, x):
                while len(kept) > 1 and x - kept[-1] < floor and not protected(dirc, kept[-1]):
                    kept.pop()
                kept.append(x)
            elif x - kept[-1] >= floor:
                kept.append(x)
        last = lines[-1]
        if last - kept[-1] < floor and not protected(dirc, kept[-1]):
            kept[-1] = last
        else:
            kept.append(last)
        mesh.SetLines(dirc, kept)


def _draw_array(sub, gnd, drv, par, feed, oy):
    """Draw one complete array shifted by `oy` in y (all y-coords + oy)."""
    ap = a.antipad
    # dielectrics
    sub.AddBox([a.gnd_x0, a.gnd_y0+oy, z_feed],    [a.gnd_x1, a.gnd_y1+oy, z_drv], priority=0)
    sub.AddBox([a.gnd_x0, a.gnd_y0+oy, z_air_top], [a.gnd_x1, a.gnd_y1+oy, z_par], priority=0)
    # 4 driven patches (+ vias) and 4 parasitics
    for (xc, yc0) in a.centres:
        yc = yc0 + oy
        _sheet(drv, xc - a.L/2,  yc - a.W/2,  xc + a.L/2,  yc + a.W/2,  z_drv)
        _sheet(par, xc - a.Lp/2, yc - a.Wp/2, xc + a.Lp/2, yc + a.Wp/2, z_par)
        vx, vy = xc - a.fv, yc
        _box(drv, vx - a.a_via, vy - a.a_via, vx + a.a_via, vy + a.a_via, z_feed, z_drv, prio=15)
    # shared ground with 4 antipads (shifted)
    _sheet(gnd, a.gnd_x0, a.gnd_y0+oy, a.gnd_x1, -a.d/2 - ap + oy, z_gnd)
    _sheet(gnd, a.gnd_x0, -a.d/2 + ap + oy, a.gnd_x1, a.d/2 - ap + oy, z_gnd)
    _sheet(gnd, a.gnd_x0, a.d/2 + ap + oy, a.gnd_x1, a.gnd_y1+oy, z_gnd)
    for yc0 in (-a.d/2, a.d/2):
        yc = yc0 + oy
        _sheet(gnd, a.gnd_x0, yc - ap, a.xaL - ap, yc + ap, z_gnd)
        _sheet(gnd, a.xaL + ap, yc - ap, a.xaR - ap, yc + ap, z_gnd)
        _sheet(gnd, a.xaR + ap, yc - ap, a.gnd_x1, yc + ap, z_gnd)
    # L4 corporate H-tree feed (shifted)
    for ys in (-1.0, 1.0):
        yr = ys * a.d/2 + oy
        _sheet(feed, a.xaL, yr - a.h50, a.xfc, yr + a.h50, z_feed)
        _sheet(feed, a.xfc, yr - a.h50, a.xaR, yr + a.h50, z_feed)
        _sheet(feed, a.xfc - a.h35, yr, a.xfc + a.h35, yr - ys*a.Lq, z_feed)
        _sheet(feed, a.xfc - a.h50, yr - ys*a.Lq, a.xfc + a.h50, oy, z_feed)
    _sheet(feed, a.xfc - a.Lq, -a.h35 + oy, a.xfc, a.h35 + oy, z_feed)
    _sheet(feed, a.x_port, -a.h50 + oy, a.xfc - a.Lq, a.h50 + oy, z_feed)


def build_antenna(CSX, FDTD):
    kappa = 2 * np.pi * f0 * a.EPS0 * a.eps_r * a.tan_d
    sub  = CSX.AddMaterial('substrate', epsilon=a.eps_r, kappa=kappa)
    gnd  = CSX.AddMetal('gnd')
    drv  = CSX.AddMetal('driven')
    par  = CSX.AddMetal('parasitic')
    feed = CSX.AddMetal('feed')

    _draw_array(sub, gnd, drv, par, feed, oyA)
    _draw_array(sub, gnd, drv, par, feed, oyB)

    # ---------------- combined mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    y_lo = a.gnd_y0 + oyA - air_xy
    y_hi = a.gnd_y1 + oyB + air_xy
    mesh.AddLine('x', [a.gnd_x0 - air_xy, a.gnd_x1 + air_xy])
    mesh.AddLine('y', [y_lo, y_hi])
    mesh.AddLine('z', [z_feed - air_below, z_par + air_above])
    mesh.AddLine('z', np.linspace(z_feed, 0, 3))
    mesh.AddLine('z', np.linspace(0, z_drv, 7))
    mesh.AddLine('z', np.linspace(z_drv, z_air_top, 9))
    mesh.AddLine('z', np.linspace(z_air_top, z_par, 4))
    # fine x lines (same for both arrays)
    mesh.AddLine('x', [a.x_port, a.xfc - a.Lq, a.xfc - a.h35, a.xfc, a.xfc + a.h35,
                       a.xaL - a.a_via, a.xaL, a.xaL + a.a_via,
                       a.xaR - a.a_via, a.xaR, a.xaR + a.a_via])
    # fine y lines for each array (via rows + feed), shifted per instance
    for oy in (oyA, oyB):
        mesh.AddLine('y', [oy, -a.h50 + oy, a.h50 + oy, -a.h35 + oy, a.h35 + oy,
                           -a.d/2 - a.a_via + oy, -a.d/2 + oy, -a.d/2 + a.a_via + oy,
                            a.d/2 - a.a_via + oy,  a.d/2 + oy,  a.d/2 + a.a_via + oy])

    for prop in (gnd, drv, par, feed):
        FDTD.AddEdges2Grid(dirs='xy', properties=prop, metal_edge_res=a.edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, a.min_cell,
                      protect={'x': [a.x_port, a.xfc, a.xaL, a.xaR], 'y': [oyA, oyB]})
    return mesh


def add_feed_ports(FDTD):
    """Port 1 = TX (excited) on array A; Port 2 = RX (matched load) on array B."""
    p_tx = FDTD.AddLumpedPort(1, feed_R, [a.x_port, oyA, z_feed], [a.x_port, oyA, z_gnd],
                              'z', 1.0, priority=5, edges2grid='xy')
    p_rx = FDTD.AddLumpedPort(2, feed_R, [a.x_port, oyB, z_feed], [a.x_port, oyB, z_gnd],
                              'z', 0.0, priority=5, edges2grid='xy')
    return p_tx, p_rx


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    mk = CSX.AddMetal('feed_probe')
    for oy in (oyA, oyB):
        mk.AddCylinder([a.x_port, oy, z_feed], [a.x_port, oy, z_gnd], radius=0.4, priority=20)
    print('Two 2x2 arrays, %.0f mm apart (edge gap %.1f mm)' % (SEP, SEP - 2*a.gnd_y1))
    print('TX port at (x=%.1f, y=%.1f) | RX port at (x=%.1f, y=%.1f)'
          % (a.x_port, oyA, a.x_port, oyB))

    sim_path = os.path.join(os.getcwd(), 'pcb_link2_3p25GHz')
    os.makedirs(sim_path, exist_ok=True)
    xml = os.path.join(sim_path, 'antenna.xml')
    CSX.Write2XML(xml)
    nx, ny, nz = (len(CSX.GetGrid().GetLines(i)) for i in range(3))
    print('Mesh lines x:%d y:%d z:%d  (~%.2f M cells)' % (nx, ny, nz, nx*ny*nz/1e6))
    try:
        from CSXCAD import AppCSXCAD_BIN
        os.system(AppCSXCAD_BIN + ' "%s"' % xml)
    except Exception:
        os.system('AppCSXCAD "%s"' % xml)
