"""
Inset-fed patch with a U-SLOT in the driven patch + air-gap stacked parasitic.
Same stack as inset_patch_build (PTFE ZYF300CA-C, Dk 2.94, Df 0.0016; driven 0.762 mm,
parasitic carrier 1.6 mm), but the driven patch carries a U-slot to add a third
resonance for bandwidth (an experiment - the slot can be optimized down to nothing
if it doesn't help).

The U-slot opens toward the -x (feed) edge so it cups the inset feed - the canonical
U-slot arrangement, where the slot interacts with the feed to broaden the match.
Tunable slot knobs: slot_len (arm length, x), slot_w (arm separation/tongue, y),
slot_x (base position). The slot channel width sw_slot is fixed.

Interface matches inset_patch_build: build_antenna(CSX, FDTD) + f0, fc, feed_x,
feed_y, feed_R, h_sub, z_stk_patch, unit.

Run:  python inset_slot_build.py     # AppCSXCAD with the feedpoint marked
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
h_sub  = 0.762                    # driven substrate Sub1 (feed + driven patch)
h_sub2 = 1.6                      # parasitic carrier Sub2

# Driven patch (from the plain-inset optimizer: worst-in-band S11 = -8.32 dB)
L  = 26.5                          # DE optimized: worst-in-band S11 = -12.41 dB
W  = 26.4

# Inset feed
Wf = 1.94                         # 50 ohm feed width on 0.762 mm PTFE
y0 = 2.1                          # inset depth (match)
g  = 1.0                          # inset gap
Lf = 12.0                         # feed-line length out to the port

# U-slot in the driven patch (bandwidth experiment)
slot_len = 2.0                    # U arm length (x)  [DE drove this to min: slot ~off]
slot_w   = 3.6                    # U arm separation / tongue (y) [tunable]
slot_x   = 7.2                    # U base x-position             [tunable]
sw_slot  = 1.0                    # slot channel width (fixed)

# Air gap + parasitic
h_air = 5.6
Lp = 29.3
Wp = 29.2

margin = 15.0
air_xy, air_above, air_below = 30.0, 40.0, 20.0
feed_R = 50.0

edge_res = 0.4
mesh_res = (C0 / (f0 + fc)) / unit / 20.0
min_cell = 0.05
# ======================================================================

# ---- derived geometry ----
hf       = Wf / 2.0
x_in     = -L/2 + y0
x_port   = -L/2 - Lf
feed_x, feed_y = x_port, 0.0
gnd_x0 = x_port - margin
gnd_x1 =  L/2   + margin
gnd_y0 = -W/2   - margin
gnd_y1 =  W/2   + margin

z_gnd       = 0.0
z_drv_patch = h_sub
z_stk_bot   = h_sub + h_air
z_stk_patch = h_sub + h_air + h_sub2


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _slot_geom():
    """U-slot rectangles, clamped to stay valid inside the patch bulk.
    Returns (fit, xb, xo, wi, sw)."""
    sw = sw_slot
    wi = max(min(slot_w, W - 2*sw - 2.0), 1.0)
    xb = max(x_in + 1.0, min(slot_x, L/2 - 2.0 - slot_len))
    xo = xb + slot_len
    fit = (slot_len >= 2.0) and (xo <= L/2 - 1.0) and (xb >= x_in + 0.5)
    return fit, xb, xo, wi, sw


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
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd_metal = CSX.AddMetal('gnd')
    drv = CSX.AddMetal('driven_patch')
    par = CSX.AddMetal('parasitic_patch')

    substrate.AddBox([gnd_x0, gnd_y0, 0],         [gnd_x1, gnd_y1, h_sub],       priority=0)
    substrate.AddBox([gnd_x0, gnd_y0, z_stk_bot], [gnd_x1, gnd_y1, z_stk_patch], priority=0)
    _box(gnd_metal, gnd_x0, gnd_y0, gnd_x1, gnd_y1, z_gnd)

    z = z_drv_patch
    fit, xb, xo, wi, sw = _slot_geom()

    # driven patch BULK (x >= x_in), with the U-slot tiled out if it fits.
    # The U opens toward -x (feed): base slot at the +x end, tongue joins the patch
    # on the -x side so the slot cups the inset feed.
    if fit:
        _box(drv, x_in, -W/2,        xb,      W/2,         z)   # left (tongue joins here, feed side)
        _box(drv, xo,   -W/2,        L/2,     W/2,         z)   # right of U base
        _box(drv, xb,   -W/2,        xo,     -(wi/2 + sw), z)   # bottom solid
        _box(drv, xb,   -wi/2,       xo - sw, wi/2,        z)   # tongue (inside U, opens -x)
        _box(drv, xb,    wi/2 + sw,  xo,      W/2,         z)   # top solid
    else:
        _box(drv, x_in, -W/2, L/2, W/2, z)                     # plain bulk

    # inset arms + feed strip (unchanged)
    _box(drv, -L/2,   hf + g,   x_in,  W/2,        z)   # upper arm
    _box(drv, -L/2,  -W/2,      x_in, -(hf + g),   z)   # lower arm
    _box(drv, x_port, -hf,      x_in,  hf,         z)   # feed strip

    # parasitic patch
    _box(par, -Lp/2, -Wp/2, Lp/2, Wp/2, z_stk_patch)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    x_lim_lo = gnd_x0 - air_xy
    x_lim_hi = gnd_x1 + air_xy
    y_lim    = max(abs(gnd_y0), abs(gnd_y1)) + air_xy
    mesh.AddLine('x', [x_lim_lo, x_lim_hi])
    mesh.AddLine('y', [-y_lim, y_lim])
    mesh.AddLine('z', [-air_below, z_stk_patch + air_above])
    mesh.AddLine('z', np.linspace(0, h_sub, 5))
    mesh.AddLine('z', np.linspace(z_stk_bot, z_stk_patch, 5))
    mesh.AddLine('z', np.linspace(z_drv_patch, z_stk_bot, 9))
    mesh.AddLine('y', [-hf, 0, hf, feed_y])
    mesh.AddLine('x', [x_in, feed_x])
    if fit:
        mesh.AddLine('x', [xb, xo - sw, xo])
        mesh.AddLine('y', [-(wi/2 + sw), -wi/2, wi/2, wi/2 + sw])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd_metal, metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=drv,       metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=par,       metal_edge_res=edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x], 'y': [feed_y]})
    return mesh


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    fp = CSX.AddMetal('feed_probe')
    fp.AddCylinder([feed_x, feed_y, z_gnd], [feed_x, feed_y, z_drv_patch], radius=0.4, priority=20)
    fit, xb, xo, wi, sw = _slot_geom()
    print('Slotted inset patch | U-slot fit=%s: base x=%.1f arms->%.1f tongue=%.1f mm'
          % (fit, xb, xo, wi))
    print('Feedpoint at (x=%.2f, y=%.2f)' % (feed_x, feed_y))

    sim_path = os.path.join(os.getcwd(), 'inset_slot_3p25GHz')
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
