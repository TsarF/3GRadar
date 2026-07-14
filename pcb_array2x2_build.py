"""
2x2 array of the via-fed driven patch + air-gap parasitic (pcb_patch_build) on the
JLCPCB 4-layer stackup, with the corporate feed network BURIED on L4.

Because the feed sits on L4 below the L3 ground, the 1->4 corporate network is fully
shielded from the radiating (+z) hemisphere - this is the cure for the feed-radiation
that tilted/split the earlier same-layer array.

    parasitic patches (top of Sub2)        4x, at (+-d/2, +-d/2)
      Sub2 FR-4 + AIR GAP (h_air)
    -- main 4-layer PCB --
    L1  4 driven patches                    at (+-d/2, +-d/2)
    L3  shared ground (4 via antipads)
    L4  corporate H-tree feed (1 input)  -> 4 vias -> 4 driven patches

Every via is at the SAME -x offset (fv) from its patch centre, so all four elements
are fed in phase (broadside) and the H-tree is symmetric/equal-length about x = -fv.
Element dimensions are imported from pcb_patch_build (the optimized single element).

`d` = element centre-to-centre spacing (sweepable). Interface matches the others:
build_antenna(CSX, FDTD); add_feed_port(FDTD); f0, fc, feed_R, unit, z_top.

Run:  python pcb_array2x2_build.py   - AppCSXCAD with the input feedpoint marked
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

from pcb_patch_build import (
    C0, EPS0, unit, f0, fc, eps_r, tan_d,
    t_core, t_prepreg, h_sub2, L, W, Lp, Wp, h_air,
    fv, a_via, antipad, Wf, feed_R, edge_res, min_cell,
)

# ======================= ARRAY PARAMETERS (mm) =======================
d = 80.0                          # element spacing (~0.87 lambda0); 6-knob opt (Gr=13.0 dBi)

# L4 corporate-feed microstrip (50/35.4 ohm on the 0.21 mm feed substrate)
W50 = Wf                          # 0.38 mm (50 ohm)
W35 = 0.70                        # 35.4 ohm quarter-wave transformer
Lq  = 12.5                        # quarter-wave length at 3.25 GHz
Lin = 10.0                        # straight 50 ohm input lead before the port

margin    = 40.0
air_xy    = 30.0
air_above = 40.0
air_below = 20.0
mesh_res  = (C0 / (f0 + fc)) / unit / 20.0
# =====================================================================

h50, h35 = W50/2, W35/2

# fixed z-levels (z = 0 at L3 ground)
z_feed = -t_prepreg
z_gnd  = 0.0
z_drv  = t_core + t_prepreg


def _derive():
    """Recompute geometry that depends on the tunable knobs (d, L, fv, h_air)."""
    global z_air_top, z_par, z_top, centres, xfc, xaL, xaR
    global half_x, half_y, x_port, feed_x, feed_y, gnd_x0, gnd_x1, gnd_y0, gnd_y1
    z_air_top = z_drv + h_air
    z_par     = z_air_top + h_sub2
    z_top     = z_par
    centres = [(sx*d/2, sy*d/2) for sx in (-1, 1) for sy in (-1, 1)]
    xfc = -fv                                   # feed-tree centre x
    xaL, xaR = -d/2 - fv, d/2 - fv              # via x at left / right columns
    half_x = d/2 + max(L, Lp)/2
    half_y = d/2 + max(W, Wp)/2
    x_port = -(half_x + Lin)
    feed_x, feed_y = x_port, 0.0
    gnd_x0 = min(x_port, xaL) - margin
    gnd_x1 = half_x + margin
    gnd_y0 = -half_y - margin
    gnd_y1 =  half_y + margin


_derive()


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


def _ground_with_holes(gnd):
    """Shared L3 ground = full plane minus 4 square antipads on the 2x2 via grid."""
    ap = antipad
    # three full-width y-bands (clear of the antipad rows)
    _sheet(gnd, gnd_x0, gnd_y0, gnd_x1, -d/2 - ap, z_gnd)
    _sheet(gnd, gnd_x0, -d/2 + ap, gnd_x1, d/2 - ap, z_gnd)
    _sheet(gnd, gnd_x0, d/2 + ap, gnd_x1, gnd_y1, z_gnd)
    # the two antipad rows, each split into x-segments around the two via columns
    for yc in (-d/2, d/2):
        _sheet(gnd, gnd_x0, yc - ap, xaL - ap, yc + ap, z_gnd)
        _sheet(gnd, xaL + ap, yc - ap, xaR - ap, yc + ap, z_gnd)
        _sheet(gnd, xaR + ap, yc - ap, gnd_x1, yc + ap, z_gnd)


def build_antenna(CSX, FDTD):
    _derive()
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd  = CSX.AddMetal('gnd')
    drv  = CSX.AddMetal('driven')     # 4 patches + 4 vias
    par  = CSX.AddMetal('parasitic')  # 4 parasitic patches
    feed = CSX.AddMetal('feed')       # L4 corporate network

    # main board + parasitic carrier dielectrics
    substrate.AddBox([gnd_x0, gnd_y0, z_feed],    [gnd_x1, gnd_y1, z_drv], priority=0)
    substrate.AddBox([gnd_x0, gnd_y0, z_air_top], [gnd_x1, gnd_y1, z_par], priority=0)

    # 4 driven patches (+ vias) and 4 parasitics
    for (xc, yc) in centres:
        _sheet(drv, xc - L/2,  yc - W/2,  xc + L/2,  yc + W/2,  z_drv)
        _sheet(par, xc - Lp/2, yc - Wp/2, xc + Lp/2, yc + Wp/2, z_par)
        vx, vy = xc - fv, yc
        _box(drv, vx - a_via, vy - a_via, vx + a_via, vy + a_via, z_feed, z_drv, prio=15)

    # shared ground with 4 antipads
    _ground_with_holes(gnd)

    # ---------------- L4 corporate H-tree feed ----------------
    for ys in (-1.0, 1.0):                       # the two rows
        yr = ys * d/2
        _sheet(feed, xaL, yr - h50, xfc, yr + h50, z_feed)        # via->centre (left)
        _sheet(feed, xfc, yr - h50, xaR, yr + h50, z_feed)        # via->centre (right)
        _sheet(feed, xfc - h35, yr, xfc + h35, yr - ys*Lq, z_feed)  # row transformer
        _sheet(feed, xfc - h50, yr - ys*Lq, xfc + h50, 0.0, z_feed)  # row line to centre
    _sheet(feed, xfc - Lq, -h35, xfc, h35, z_feed)               # centre transformer (-x)
    _sheet(feed, x_port, -h50, xfc - Lq, h50, z_feed)            # 50 ohm input lead

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [gnd_x0 - air_xy, gnd_x1 + air_xy])
    mesh.AddLine('y', [gnd_y0 - air_xy, gnd_y1 + air_xy])
    mesh.AddLine('z', [z_feed - air_below, z_par + air_above])
    mesh.AddLine('z', np.linspace(z_feed, 0, 3))
    mesh.AddLine('z', np.linspace(0, z_drv, 7))
    mesh.AddLine('z', np.linspace(z_drv, z_air_top, 9))
    mesh.AddLine('z', np.linspace(z_air_top, z_par, 4))
    # fine lines at the feed tree, vias and antipads
    mesh.AddLine('x', [x_port, xfc - Lq, xfc - h35, xfc, xfc + h35,
                       xaL - a_via, xaL, xaL + a_via, xaR - a_via, xaR, xaR + a_via])
    mesh.AddLine('y', [-h50, 0.0, h50, -h35, h35,
                       -d/2 - a_via, -d/2, -d/2 + a_via, d/2 - a_via, d/2, d/2 + a_via])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=drv,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=par,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=feed, metal_edge_res=edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, min_cell,
                      protect={'x': [feed_x, xfc, xaL, xaR], 'y': [feed_y, d/2, -d/2]})
    return mesh


def add_feed_port(FDTD):
    """Single lumped port: L4 input lead end against the L3 ground (z: L4 -> L3)."""
    return FDTD.AddLumpedPort(1, feed_R,
                              [x_port, feed_y, z_feed], [x_port, feed_y, z_gnd],
                              'z', 1.0, priority=5, edges2grid='xy')


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    mark = CSX.AddMetal('feed_probe')
    mark.AddCylinder([x_port, 0, z_feed], [x_port, 0, z_gnd], radius=0.3, priority=20)
    print('2x2 via-fed array, spacing d = %.1f mm (%.2f lambda0)' % (d, d / (C0/f0/unit)))
    print('feed tree centred at x=%.2f | input port at x=%.2f' % (xfc, x_port))

    sim_path = os.path.join(os.getcwd(), 'pcb_array2x2_3p25GHz')
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
