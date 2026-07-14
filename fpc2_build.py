"""
High-gain Fabry-Perot cavity antenna for 3.1-3.4 GHz, replicating the wideband
dual-layer PRS of Ding et al. (Results in Engineering 27 (2025) 106647) but
trimmed for the job at hand: we only need ~9% bandwidth, so the paper's RCM
(bandwidth) and CBM (RCS) layers are dropped and the PRS is pushed to high
reflectivity for MAXIMUM gain.  This is the paper's "Antenna I" topology on the
JLCPCB PTFE stack.

Stack (bottom -> top):
    z=0              ground plane
    0 .. h_sub       feed board (1.52 mm PTFE) + inset feed patch on top
    h_sub .. h_cav   air cavity
    h_cav            PRS bottom face: slotted square (loop) array  [faces cavity]
    h_cav .. +h_prs  PRS board (1.52 mm PTFE)
    h_cav+h_prs      PRS top face: circular patch array

PRS geometry comes from fpc2_prs_unitcell.py (|Gamma|~0.98, h_cav~47.5 mm).  Board
is 9x9 units = 193.5 mm (< 200 mm) ~ 2.1 lambda0, which sets the ~17 dBi aperture
ceiling; that aperture is the main gain lever.

Interface matches the other builds: build_antenna(CSX, FDTD) + f0, fc, feed_x,
feed_y, feed_R, h_sub, z_stk_patch, unit.

Run:  python fpc2_build.py     # AppCSXCAD with the feedpoint marked
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
h_prs = 1.52                      # PRS board

# Feed: inset patch on the ground board (Stage-A tuned: -18 dB match at 3.25 GHz)
L  = 26.5
W  = 29.4
Wf = 3.89
y0 = 9.5
g  = 1.0
Lf = 10.0

# Dual-layer PRS (tuned by fpc2_prs_unitcell.py: |Gamma|~0.96 flat over 3.1-3.4)
N_PRS = 9                         # units per side (9x9 -> 193.5 mm < 200 mm)
P     = 21.5                      # unit-cell period
r1    = 6.5                       # top circular patch radius
pb    = 20.7                      # bottom square outer size
sb    = 16.0                      # bottom square slot opening (loop)

# Cavity height (raised from the ideal 44.6 to pull resonance 3.35 -> 3.25 GHz)
h_cav = 46.0                      # ground -> PRS bottom (slotted) face

air_xy, air_above, air_below = 35.0, 45.0, 20.0
feed_R = 50.0

FEED_ONLY = False                 # True -> bare patch (no PRS/cavity) for Stage-A feed tuning

edge_res = 0.4
mesh_res = (C0 / (f0 + fc)) / unit / 26.0     # finer: recover PRS effective reflectivity
min_cell = 0.05
# ======================================================================


def _recompute():
    global hf, x_in, x_port, feed_x, feed_y, ap_half
    global z_gnd, z_feed_patch, z_prs_bot, z_prs_top, z_stk_patch
    hf = Wf / 2.0
    x_in   = -L/2 + y0
    x_port = -L/2 - Lf
    feed_x, feed_y = x_port, 0.0
    ap_half = N_PRS * P / 2.0
    z_gnd = 0.0
    z_feed_patch = h_sub
    z_prs_bot = h_cav
    z_prs_top = h_cav + h_prs
    z_stk_patch = z_prs_top        # NF2FF center reference


_recompute()


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _square_loop(prop, cx, cy, size, opening, z):
    o, i = size / 2.0, opening / 2.0
    prop.AddBox([cx - o, cy - o, z], [cx + o, cy - i, z], priority=10)   # bottom
    prop.AddBox([cx - o, cy + i, z], [cx + o, cy + o, z], priority=10)   # top
    prop.AddBox([cx - o, cy - i, z], [cx - i, cy + i, z], priority=10)   # left
    prop.AddBox([cx + i, cy - i, z], [cx + o, cy + i, z], priority=10)   # right


def _disc(prop, cx, cy, rad, z, nseg=24):
    """Flat circular patch as an n-gon polygon (a zero-height cylinder is a
    degenerate primitive openEMS drops as 'unused'; a polygon rasterizes)."""
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    prop.AddPolygon(np.array([cx + rad * np.cos(a), cy + rad * np.sin(a)]), 'z', z, priority=10)


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
    prs_b = CSX.AddMetal('prs_bottom')
    prs_t = CSX.AddMetal('prs_top')

    # feed-only (Stage A) uses a compact ground; full antenna uses the PRS footprint
    B = (max(L, W) / 2.0 + 20.0) if FEED_ONLY else ap_half
    z_top_air = (z_feed_patch if FEED_ONLY else z_prs_top) + air_above

    feed_sub.AddBox([-B, -B, 0], [B, B, h_sub], priority=0)
    _box(gnd, -B, -B, B, B, z_gnd)

    # ---- inset feed patch ----
    z = z_feed_patch
    _box(patch, x_in,   -W/2,    L/2,   W/2,       z)
    _box(patch, -L/2,    hf + g, x_in,  W/2,       z)
    _box(patch, -L/2,   -W/2,    x_in, -(hf + g),  z)
    _box(patch, x_port, -hf,     x_in,  hf,        z)

    if not FEED_ONLY:
        prs_sub.AddBox([-B, -B, z_prs_bot], [B, B, z_prs_top], priority=0)
        off = (N_PRS - 1) / 2.0
        for i in range(N_PRS):
            cx = (i - off) * P
            for j in range(N_PRS):
                cy = (j - off) * P
                _square_loop(prs_b, cx, cy, pb, sb, z_prs_bot)      # slotted square (cavity side)
                _disc(prs_t, cx, cy, r1, z_prs_top)                 # circular patch

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [-B - air_xy, B + air_xy])
    mesh.AddLine('y', [-B - air_xy, B + air_xy])
    mesh.AddLine('z', [-air_below, z_top_air])
    mesh.AddLine('z', np.linspace(0, h_sub, 4))
    mesh.AddLine('y', [-hf, 0, hf])
    mesh.AddLine('x', [x_in, feed_x])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd,   metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=edge_res)
    if not FEED_ONLY:
        mesh.AddLine('z', np.linspace(h_sub, z_prs_bot, 13))       # cavity fill
        mesh.AddLine('z', [z_prs_bot, z_prs_top])
        cc = [(k - (N_PRS - 1) / 2.0) * P for k in range(N_PRS)]   # resolve disc arrays
        disc_lines = sorted(set([c + d for c in cc for d in (-r1, -r1/2, 0.0, r1/2, r1)]))
        mesh.AddLine('x', disc_lines); mesh.AddLine('y', disc_lines)
        FDTD.AddEdges2Grid(dirs='xy', properties=prs_b, metal_edge_res=edge_res)
        FDTD.AddEdges2Grid(dirs='xy', properties=prs_t, metal_edge_res=edge_res)

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
    print('FPC-II (patch + dual-layer PRS) | %dx%d PRS, period %.1f | board %.0f mm (%.2f lam0)'
          % (N_PRS, N_PRS, P, 2 * ap_half, 2 * ap_half / lam0))
    print('cavity h=%.1f mm (%.3f lam0) | top circle r1=%.1f | bottom loop %.1f/%.1f | feed (%.2f,%.2f)'
          % (h_cav, h_cav / lam0, r1, pb, sb, feed_x, feed_y))

    sim_path = os.path.join(os.getcwd(), 'fpc2_3p25GHz')
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
