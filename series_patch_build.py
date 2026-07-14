"""
Series-fed linear patch array on 1.52 mm PTFE (JLCPCB ZYF300CA-C, Dk 2.94,
Df 0.0016).  N rectangular patches strung along +x, each joined to the next by a
high-impedance microstrip connecting line; the string is fed at one end through an
inset feed on the first patch (the inset depth is the input-match knob).

The resonant-length dimension L runs along the array axis (x), so each patch's two
radiating edges face +/-x and the array factor builds a broadside beam along x.
For an in-phase (broadside) resonant array the patch-edge-to-edge connecting line is
about a half guided-wavelength; Lc is left as a tunable knob so an optimizer can set
the phasing.  The last patch is open (resonant standing-wave array, not terminated).

Interface matches the other builds: build_antenna(CSX, FDTD) + f0, fc, feed_x,
feed_y, feed_R, h_sub, z_stk_patch, unit.

Run:  python series_patch_build.py     # AppCSXCAD with the feedpoint marked
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

# Substrate: JLCPCB ZYF300CA-C PTFE (Dk 2.94, Df 0.0016)
eps_r = 2.94
tan_d = 0.0016
h_sub = 1.52

N_PATCH = 4                       # patches in the series string

# Patch (L along array/feed axis x, W across y)
L = 26.33                         # resonant length (~lambda_g/2 at 3.25 GHz)
W = 32.86                         # patch width

# Inset feed on the FIRST patch (matching)
Wf = 3.89                         # 50 ohm feed width on 1.52 mm PTFE
y0 = 7.0                          # inset depth into patch 0 (match knob)
g  = 1.0                          # inset gap
Lf = 10.0                         # feed-line length out to the port

# Connecting line between adjacent patches (phasing)
Wc = 1.0                          # connecting-line width (high impedance)
Lc = 31.1                         # patch-edge to patch-edge length (~lambda_g/2)

margin = 12.0
air_xy, air_above, air_below = 25.0, 40.0, 20.0
feed_R = 50.0

edge_res = 0.4
mesh_res = (C0 / (f0 + fc)) / unit / 20.0
min_cell = 0.05
# ======================================================================


def _recompute():
    """Derive the array layout from the current parameters (patch centers,
    feed point, ground/air extents, z-levels).  Called at import and at the top
    of build_antenna so optimizers can just set globals and rebuild."""
    global hf, S, xc, x_l0, x_in, x_port, feed_x, feed_y
    global gnd_x0, gnd_x1, gnd_y0, gnd_y1, z_gnd, z_patch, z_stk_patch
    hf = Wf / 2.0
    S  = L + Lc                                    # patch center-to-center pitch
    xc = [(i - (N_PATCH - 1) / 2.0) * S for i in range(N_PATCH)]
    x_l0   = xc[0] - L / 2.0                        # left edge of patch 0
    x_in   = x_l0 + y0                              # inset depth reaches here
    x_port = x_l0 - Lf                              # feed line start (port)
    feed_x, feed_y = x_port, 0.0
    x_last = xc[-1] + L / 2.0
    gnd_x0 = x_port - margin
    gnd_x1 = x_last + margin
    gnd_y0 = -W / 2.0 - margin
    gnd_y1 =  W / 2.0 + margin
    z_gnd = 0.0
    z_patch = h_sub
    z_stk_patch = h_sub                            # single-layer: patch plane = top


_recompute()


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    protect = protect or {}

    def protected(d, v):
        return any(abs(v - p) <= 1e-6 for p in protect.get(d, []))

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
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd_metal = CSX.AddMetal('gnd')
    patch     = CSX.AddMetal('patch')

    substrate.AddBox([gnd_x0, gnd_y0, 0], [gnd_x1, gnd_y1, h_sub], priority=0)
    _box(gnd_metal, gnd_x0, gnd_y0, gnd_x1, gnd_y1, z_gnd)

    z = z_patch

    # ---- patch 0 with inset feed (U-cut around the feed strip) ----
    _box(patch, x_in,   -W/2,      xc[0] + L/2, W/2,        z)   # bulk beyond inset
    _box(patch, x_l0,    hf + g,   x_in,        W/2,        z)   # upper arm
    _box(patch, x_l0,   -W/2,      x_in,       -(hf + g),   z)   # lower arm
    _box(patch, x_port, -hf,       x_in,        hf,         z)   # feed strip -> port

    # ---- patches 1..N-1 (plain rectangles) ----
    for i in range(1, N_PATCH):
        _box(patch, xc[i] - L/2, -W/2, xc[i] + L/2, W/2, z)

    # ---- connecting lines between adjacent patches ----
    for i in range(N_PATCH - 1):
        _box(patch, xc[i] + L/2, -Wc/2, xc[i+1] - L/2, Wc/2, z)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [gnd_x0 - air_xy, gnd_x1 + air_xy])
    y_lim = max(abs(gnd_y0), abs(gnd_y1)) + air_xy
    mesh.AddLine('y', [-y_lim, y_lim])
    mesh.AddLine('z', [-air_below, z_patch + air_above])
    mesh.AddLine('z', np.linspace(0, h_sub, 5))
    mesh.AddLine('y', [-hf, 0, hf, -Wc/2, Wc/2])
    mesh.AddLine('x', [x_in, feed_x])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd_metal, metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch,     metal_edge_res=edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x], 'y': [feed_y]})
    return mesh


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    fp = CSX.AddMetal('feed_probe')
    fp.AddCylinder([feed_x, feed_y, z_gnd], [feed_x, feed_y, z_patch], radius=0.4, priority=20)
    print('Series-fed array: %d patches | pitch S=%.1f mm (%.2f lambda0) | span %.0f mm'
          % (N_PATCH, S, S / (C0 / f0 / unit), xc[-1] - xc[0] + L))
    print('Feedpoint at (x=%.2f, y=%.2f), inset y0=%.1f mm' % (feed_x, feed_y, y0))

    sim_path = os.path.join(os.getcwd(), 'series_patch_3p25GHz')
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
