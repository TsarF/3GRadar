"""
Fabry-Perot cavity antenna on the PTFE stack (JLCPCB ZYF300CA-C, Dk 2.94,
Df 0.0016).  A single inset-fed patch on the ground board excites a resonant cavity
formed with a Partially Reflecting Surface (PRS) - an N x N array of metal patches
on a thin superstrate suspended ~lambda/2 above the ground.  The cavity leaks over a
wide aperture, giving broadside gain far above the bare patch.

Stack (bottom -> top):
    z=0            ground plane
    0 .. h_sub     feed substrate (1.52 mm PTFE) + inset feed patch on top
    h_sub .. h_cav air cavity (nylon-spacer height)
    h_cav          PRS patch array (faces down into the cavity)
    h_cav .. +h_prs PRS carrier (0.762 mm PTFE)

Design order:  run fpc_prs_unitcell.py first to pick p, a and h_cav, then drop them
in here.  Directivity ~ (1+|Gamma|)/(1-|Gamma|); higher PRS reflectivity => more gain
but narrower bandwidth.

Interface matches the other builds: build_antenna(CSX, FDTD) + f0, fc, feed_x,
feed_y, feed_R, h_sub, z_stk_patch, unit.

Run:  python fpc_build.py     # AppCSXCAD with the feedpoint marked
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

C0   = 299792458.0
EPS0 = 8.854187812813e-12
unit = 1e-3

# ======================= DESIGN PARAMETERS (mm) =======================
f0 = 3.25e9
fc = 1.00e9

eps_r = 2.94                      # PTFE ZYF300CA-C (Dk 2.94, Df 0.0016)
tan_d = 0.0016
h_sub = 1.52                      # feed board
h_prs = 0.762                     # PRS carrier

# Feed: inset patch on the ground board
L  = 26.33
W  = 32.86
Wf = 3.89                         # 50 ohm feed width on 1.52 mm PTFE
y0 = 7.0                          # inset depth (match knob)
g  = 1.0                          # inset gap
Lf = 10.0                         # feed line out to the port

# PRS array (from fpc_prs_unitcell.py)
N_PRS = 7                         # patches per side (N x N)
p     = 24.0                      # unit-cell period
a     = 22.0                      # patch size (gap = p - a)

# Cavity (Trentini height from the unit-cell tool: p=24, a=22 -> |Gamma|=0.87, 49.9 mm)
h_cav = 49.9                      # ground -> PRS patch plane

board_margin = 4.0                # ground/board beyond the PRS aperture
air_xy, air_above, air_below = 30.0, 40.0, 20.0
feed_R = 50.0

edge_res = 0.4
mesh_res = (C0 / (f0 + fc)) / unit / 20.0
min_cell = 0.05
# ======================================================================


def _recompute():
    global hf, x_in, x_port, feed_x, feed_y, ap_half, Bhalf
    global z_gnd, z_feed_patch, z_prs, z_prs_top, z_stk_patch
    hf = Wf / 2.0
    x_in   = -L/2 + y0
    x_port = -L/2 - Lf
    feed_x, feed_y = x_port, 0.0
    ap_half = N_PRS * p / 2.0
    Bhalf   = ap_half + board_margin
    z_gnd = 0.0
    z_feed_patch = h_sub
    z_prs   = h_cav
    z_prs_top = h_cav + h_prs
    z_stk_patch = h_cav            # NF2FF center reference lands mid-cavity


_recompute()


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    protect = protect or {}

    def protected(d, v):
        return any(abs(v - q) <= 1e-6 for q in protect.get(d, []))

    for d in 'xyz':
        lines = np.unique(np.asarray(mesh.GetLines(d), dtype=float))
        if lines.size < 3:
            continue
        kept = [lines[0]]
        for x in lines[1:-1]:
            if protected(d, x):
                while len(kept) > 1 and x - kept[-1] < floor and not protected(d, kept[-1]):
                    kept.pop()
                kept.append(x)
            elif x - kept[-1] >= floor:
                kept.append(x)
        last = lines[-1]
        if last - kept[-1] < floor and not protected(d, kept[-1]):
            kept[-1] = last
        else:
            kept.append(last)
        mesh.SetLines(d, kept)


def build_antenna(CSX, FDTD):
    _recompute()
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    feed_sub = CSX.AddMaterial('feed_sub', epsilon=eps_r, kappa=kappa)
    prs_sub  = CSX.AddMaterial('prs_sub',  epsilon=eps_r, kappa=kappa)
    gnd   = CSX.AddMetal('gnd')
    patch = CSX.AddMetal('feed_patch')
    prs   = CSX.AddMetal('prs_patch')

    feed_sub.AddBox([-Bhalf, -Bhalf, 0],     [Bhalf, Bhalf, h_sub],     priority=0)
    prs_sub.AddBox([-Bhalf, -Bhalf, z_prs],  [Bhalf, Bhalf, z_prs_top], priority=0)
    _box(gnd, -Bhalf, -Bhalf, Bhalf, Bhalf, z_gnd)

    # ---- inset feed patch on the ground board ----
    z = z_feed_patch
    _box(patch, x_in,   -W/2,    L/2,   W/2,       z)   # bulk beyond inset
    _box(patch, -L/2,    hf + g, x_in,  W/2,       z)   # upper arm
    _box(patch, -L/2,   -W/2,    x_in, -(hf + g),  z)   # lower arm
    _box(patch, x_port, -hf,     x_in,  hf,        z)   # feed strip -> port

    # ---- PRS patch array (faces down into the cavity) ----
    off = (N_PRS - 1) / 2.0
    for i in range(N_PRS):
        cx = (i - off) * p
        for j in range(N_PRS):
            cy = (j - off) * p
            _box(prs, cx - a/2, cy - a/2, cx + a/2, cy + a/2, z_prs)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [-Bhalf - air_xy, Bhalf + air_xy])
    mesh.AddLine('y', [-Bhalf - air_xy, Bhalf + air_xy])
    mesh.AddLine('z', [-air_below, z_prs_top + air_above])
    mesh.AddLine('z', np.linspace(0, h_sub, 4))
    mesh.AddLine('z', np.linspace(h_sub, z_prs, 12))          # cavity fill
    mesh.AddLine('z', [z_prs, z_prs_top])
    mesh.AddLine('y', [-hf, 0, hf])
    mesh.AddLine('x', [x_in, feed_x])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd,   metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=prs,   metal_edge_res=edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x], 'y': [feed_y]})
    return mesh


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    fp = CSX.AddMetal('feed_probe')
    fp.AddCylinder([feed_x, feed_y, z_gnd], [feed_x, feed_y, z_feed_patch], radius=0.4, priority=20)
    lam0 = C0 / f0 / unit
    print('FPC antenna | PRS %dx%d, period %.1f, patch %.1f (gap %.1f) | aperture %.0f mm (%.2f lam0)'
          % (N_PRS, N_PRS, p, a, p - a, 2 * ap_half, 2 * ap_half / lam0))
    print('cavity h=%.1f mm (%.3f lam0) | feed at (x=%.2f, y=%.2f)'
          % (h_cav, h_cav / lam0, feed_x, feed_y))

    sim_path = os.path.join(os.getcwd(), 'fpc_3p25GHz')
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
