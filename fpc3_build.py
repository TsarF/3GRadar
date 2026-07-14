"""
Three-layer Fabry-Perot antenna for 3.1-3.4 GHz: feed patch + dual-layer PRS + RCM
(Resonant Complementary Metasurface).  This is the paper's "Antenna II" - the RCM is
a second partially-reflecting superstrate ~lambda/2 above the PRS that raises the
cavity's aperture efficiency, lifting broadside gain from the single-PRS ~11 dBi
toward the ~15 dBi aperture ceiling of the 194 mm board.

Stack (bottom -> top):
    z=0                     ground plane
    0 .. h_sub              feed board (1.52 mm PTFE) + inset feed patch
    h_sub .. h_cav          air cavity 1
    h_cav .. +h_prs         PRS board (loops on bottom, circles on top)
    +h3                     air cavity 2
    z_rcm .. +h_rcm         RCM board (square-patch array on bottom face)

The paper's RCM uses three varied square sizes for BANDWIDTH; we need gain over 9%,
so this uses a uniform square-patch array (size rcm_s, gap h3 tunable) - the second
reflector that does the gain lifting.  h3 from the analytic design (~38.8 mm).

Interface matches the other builds.  Run:  python fpc3_build.py
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

eps_r = 2.94
tan_d = 0.0016
cu_sigma = 5.8e7                  # copper conductivity [S/m] -- finite (not PEC): realistic
cu_t     = 35e-6                  # copper thickness [m] (1 oz). Loss lowers the runaway cavity
                                  # Q (PEC = infinite Q -> pathological ring-down) and de-biases
                                  # the realized-gain estimate downward to a realistic value.
h_sub = 1.52
h_prs = 1.52
h_rcm = 1.52                      # RCM board

# Feed -- current best from the 6-knob feed+cavity realized-gain DE (co-tuned WITH the
# cavity: feed-only matching does NOT transfer, since the cavity loads the feed to R~15-30
# ohm, X->+77j). Best so far: worst-in-band realized gain +9.40 dBi across 3.1-3.4 GHz.
# These defaults double as the optimizer's warm-start seed (used when no de_state/JSON exists,
# e.g. a fresh EC2 clone), so the search resumes from this design without any file copy.
L  = 24.5
W  = 30.8
Wf = 3.89
y0 = 5.0
g  = 1.0
Lf = 10.0

# U-slot in the feed patch (adds a 2nd resonance -> broadband match; paper's slotted feed).
# The U opens toward -x (feed side). All tunable for the optimizer.
# U-slot DISABLED: feed-only DE drove it to a vestigial 4x4 (no real 2nd resonance), and the
# match it found did not survive cavity loading. The feed is matched via in-cavity co-tuning.
SLOT_ON  = False
slot_len = 8.0                    # U arm length (x)
slot_w   = 10.0                   # U arm separation / tongue width (y)
slot_x   = 3.0                    # U base x-position
slot_sw  = 1.0                    # slot channel width

FEED_ONLY = False                 # True -> bare feed patch only (no PRS/RCM), for fast slot tuning

# Narrowband feed match (design freq 3.25 GHz).  The bare feed presents Z=38+j80 ohm at
# 3.25 (high-Q cavity; broadband match is Bode-Fano impossible).  We PREPEND, OUTWARD from
# the old port, a 50-ohm offset section (reference-plane shift to a real Z~260 ohm) + a
# quarter-wave transformer (Zt=sqrt(50*260)=114 ohm) to bring it to 50 ohm.  The
# patch/inset/cavity/RCM are left UNTOUCHED (the load seen inward is preserved).
MATCH_ON = False                  # off: measure the cavity's intrinsic match (multi-size RCM test)
Loff = 4.68                       # 50-ohm offset extension beyond old port (ref-plane shift)
Wt   = 0.75                       # quarter-wave transformer width  (Zt~114 ohm on PTFE)
Lt   = 15.68                      # quarter-wave transformer length (lambda_g/4 @3.25 GHz)

# Dual-layer PRS
N_PRS = 8                         # 8x8 units (per the paper)
P     = 21.5
r1    = 10
pb    = 20.7
sb    = 16.0
h_cav = 52.0                      # ground -> PRS bottom face (Trentini height; DE co-opt best)

# PRS bottom = fine INDUCTIVE wire mesh (paper's "meshed patch"), not isolated loops.
# Gives ~4x flatter reflection-phase slope + rising |Gamma| -> wideband. (unit-cell tuned)
PRS_MESH = True                   # True -> wire mesh; False -> square loop (old)
mesh_N   = 1                      # holes per cell side (4 -> ~8M cells, tractable; slope -9)
mesh_wt  = 0.6                    # mesh trace width (mm)

# RCM (second reflector). Per the paper: a COARSE grid at ~2x the PRS period, so each
# RCM cell spans 2x2 PRS cells and the largest tile ~= two PRS cells wide.
RCM_ON = True
N_RCM  = 4                        # 4x4 RCM cells over the 8x8 PRS (2x2 PRS cells each)
P_RCM  = 2 * P                    # RCM period = 2x PRS period = 43 mm
rcm_s  = 41.0                     # RCM tile size (~2 PRS cells; large square)
h3     = 51.0                     # PRS top -> RCM plane (DE co-opt best)

# Multi-size RCM (paper's bandwidth mechanism): each RCM super-cell is SUBDIVIDED into an
# NxN array of squares. Interleaving tiles of 1x1 / 3x3 / 4x4 staggers their resonances
# (big square = low freq, fine grid of small squares = high freq) -> wideband.
RCM_MULTI = True                  # True -> subdivided tiles per RCM_MAP; False -> all 1x1
# Per-tile subdivision map (paper's arrangement), row-major over the 4x4 grid:
#   3 4 4 3   (corners 3x3, center 2x2 = 1x1 big squares, edges 4x4)
#   4 1 1 4
#   4 1 1 4
#   3 4 4 3
RCM_MAP   = [3, 4, 4, 3, 4, 1, 1, 4, 4, 1, 1, 4, 3, 4, 4, 3]
# every tile has the SAME footprint (rcm_s), so inter-tile gaps are all identical
# (P_RCM - rcm_s). Sub-squares fill that footprint edge-to-edge, split by rcm_gap.
rcm_gap   = 5.5                   # spacing between sub-squares within a tile (DE co-opt best)

air_xy, air_above, air_below = 35.0, 45.0, 20.0
feed_R = 50.0

edge_res = 0.4
mesh_res = (C0 / (f0 + fc)) / unit / 24.0
min_cell = 0.05
# ======================================================================


def _recompute():
    global hf, x_in, x_port, x_tr_out, feed_x, feed_y, ap_half
    global z_gnd, z_feed_patch, z_prs_bot, z_prs_top, z_rcm, z_rcm_top, z_stk_patch
    hf = Wf / 2.0
    x_in   = -L/2 + y0
    # x_port = outer end of the 50-ohm feed line (extended by Loff when matching)
    x_port = -L/2 - Lf - (Loff if MATCH_ON else 0.0)
    # x_tr_out = outer end of the quarter-wave transformer = the (new) port when matching
    x_tr_out = x_port - (Lt if MATCH_ON else 0.0)
    feed_x = x_tr_out if MATCH_ON else x_port
    feed_y = 0.0
    ap_half = N_PRS * P / 2.0
    z_gnd = 0.0
    z_feed_patch = h_sub
    z_prs_bot = h_cav
    z_prs_top = h_cav + h_prs
    z_rcm     = z_prs_top + h3            # RCM patch plane (faces cavity 2)
    z_rcm_top = z_rcm + h_rcm
    z_stk_patch = z_rcm_top if RCM_ON else z_prs_top


_recompute()


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _slot_geom():
    """U-slot rectangles, clamped to stay valid inside the patch bulk.
    Returns (fit, xb, xo, wi, sw); fit=False -> plain bulk (no slot)."""
    sw = slot_sw
    wi = max(min(slot_w, W - 2*sw - 2.0), 1.0)
    xb = max(x_in + 1.0, min(slot_x, L/2 - 2.0 - slot_len))
    xo = xb + slot_len
    fit = SLOT_ON and (slot_len >= 2.0) and (xo <= L/2 - 1.0) and (xb >= x_in + 0.5)
    return fit, xb, xo, wi, sw


def _square_loop(prop, cx, cy, size, opening, z):
    o, i = size / 2.0, opening / 2.0
    prop.AddBox([cx - o, cy - o, z], [cx + o, cy - i, z], priority=10)
    prop.AddBox([cx - o, cy + i, z], [cx + o, cy + o, z], priority=10)
    prop.AddBox([cx - o, cy - i, z], [cx - i, cy + i, z], priority=10)
    prop.AddBox([cx + i, cy - i, z], [cx + o, cy + i, z], priority=10)


def _disc(prop, cx, cy, rad, z, nseg=24):
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    prop.AddPolygon(np.array([cx + rad * np.cos(a), cy + rad * np.sin(a)]), 'z', z, priority=10)


def _wire_mesh_global(prop, z):
    """Full-aperture inductive wire grid, built ONCE (each trace added a single time ->
    no duplicate boxes at shared cell boundaries). Traces at pitch P/mesh_N."""
    pos = np.linspace(-ap_half, ap_half, N_PRS * mesh_N + 1)
    for x in pos:
        prop.AddBox([x - mesh_wt/2, -ap_half, z], [x + mesh_wt/2, ap_half, z], priority=10)
    for y in pos:
        prop.AddBox([-ap_half, y - mesh_wt/2, z], [ap_half, y + mesh_wt/2, z], priority=10)


def _rcm_tile(prop, cx, cy, Ndiv, z):
    """One RCM super-cell: fixed footprint rcm_s subdivided into Ndiv x Ndiv squares that
    FILL the footprint edge-to-edge, separated by rcm_gap (1x1 -> low freq; 4x4 -> high)."""
    F = rcm_s
    s = (F - (Ndiv - 1) * rcm_gap) / Ndiv
    x0 = cx - F/2 + s/2
    y0 = cy - F/2 + s/2
    for a in range(Ndiv):
        sx = x0 + a * (s + rcm_gap)
        for b in range(Ndiv):
            sy = y0 + b * (s + rcm_gap)
            prop.AddBox([sx - s/2, sy - s/2, z], [sx + s/2, sy + s/2, z], priority=10)


def _rcm_lines(cx, cy, Ndiv):
    """Sub-square edge x/y positions for one RCM tile (for meshing)."""
    F = rcm_s
    s = (F - (Ndiv - 1) * rcm_gap) / Ndiv
    x0 = cx - F/2 + s/2
    xs = []
    for a in range(Ndiv):
        sc = x0 + a * (s + rcm_gap)
        xs += [sc - s/2, sc + s/2]
    return xs


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
    rcm_sub  = CSX.AddMaterial('rcm_sub',  epsilon=eps_r, kappa=kappa)
    # finite-conductivity copper (not PEC): realistic loss + tames the cavity Q
    def _cu(name):
        return CSX.AddConductingSheet(name, conductivity=cu_sigma, thickness=cu_t)
    gnd   = _cu('gnd')
    patch = _cu('feed_patch')
    prs_b = _cu('prs_bottom')
    prs_t = _cu('prs_top')
    rcm   = _cu('rcm')

    B = (max(L, W) / 2.0 + 20.0) if FEED_ONLY else ap_half   # compact ground for feed-only
    off = (N_PRS - 1) / 2.0

    feed_sub.AddBox([-B, -B, 0], [B, B, h_sub], priority=0)
    _box(gnd, -B, -B, B, B, z_gnd)

    # feed patch (bulk carries an optional U-slot that opens toward the feed edge)
    z = z_feed_patch
    fit, xb, xo, wi, sw = _slot_geom()
    if fit:
        _box(patch, x_in, -W/2,       xb,      W/2,         z)   # left of U base (tongue joins)
        _box(patch, xo,   -W/2,       L/2,     W/2,         z)   # right of U base
        _box(patch, xb,   -W/2,       xo,     -(wi/2 + sw), z)   # bottom solid
        _box(patch, xb,   -wi/2,      xo - sw, wi/2,        z)   # tongue (inside the U)
        _box(patch, xb,    wi/2 + sw, xo,      W/2,         z)   # top solid
    else:
        _box(patch, x_in, -W/2, L/2, W/2, z)                    # plain bulk
    _box(patch, -L/2,    hf + g, x_in,  W/2,       z)
    _box(patch, -L/2,   -W/2,    x_in, -(hf + g),  z)
    _box(patch, x_port, -hf,     x_in,  hf,        z)   # 50-ohm feed line (extended by Loff)
    if MATCH_ON:
        _box(patch, x_tr_out, -Wt/2, x_port, Wt/2, z)   # quarter-wave transformer (Zt~114 ohm)

    if not FEED_ONLY:
        # dual-layer PRS
        prs_sub.AddBox([-B, -B, z_prs_bot], [B, B, z_prs_top], priority=0)
        if PRS_MESH:
            _wire_mesh_global(prs_b, z_prs_bot)      # full grid, once (no duplicate traces)
        for i in range(N_PRS):
            cx = (i - off) * P
            for j in range(N_PRS):
                cy = (j - off) * P
                if not PRS_MESH:
                    _square_loop(prs_b, cx, cy, pb, sb, z_prs_bot)
                _disc(prs_t, cx, cy, r1, z_prs_top)

        # RCM square-patch superstrate
        if RCM_ON:
            rcm_sub.AddBox([-B, -B, z_rcm], [B, B, z_rcm_top], priority=0)
            off_r = (N_RCM - 1) / 2.0
            for i in range(N_RCM):
                cx = (i - off_r) * P_RCM
                for j in range(N_RCM):
                    cy = (j - off_r) * P_RCM
                    Ndiv = RCM_MAP[i * N_RCM + j] if RCM_MULTI else 1
                    _rcm_tile(rcm, cx, cy, Ndiv, z_rcm)

    # ---------------- mesh ----------------
    if FEED_ONLY:
        z_end = z_feed_patch + air_above
    else:
        z_end = (z_rcm_top if RCM_ON else z_prs_top) + air_above
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [-B - air_xy, B + air_xy])
    mesh.AddLine('y', [-B - air_xy, B + air_xy])
    mesh.AddLine('z', [-air_below, z_end])
    mesh.AddLine('z', np.linspace(0, h_sub, 4))
    mesh.AddLine('y', [-hf, 0, hf])
    mesh.AddLine('x', [x_in, x_port, feed_x])
    _fit, _xb, _xo, _wi, _sw = _slot_geom()
    if _fit:
        mesh.AddLine('x', [_xb, _xo - _sw, _xo])
        mesh.AddLine('y', [-(_wi/2 + _sw), -_wi/2, _wi/2, _wi/2 + _sw])
    if MATCH_ON:
        mesh.AddLine('x', [x_tr_out])
        mesh.AddLine('y', [-Wt/2, Wt/2])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd,   metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=edge_res)

    if not FEED_ONLY:
        mesh.AddLine('z', np.linspace(h_sub, z_prs_bot, 13))      # cavity 1
        mesh.AddLine('z', [z_prs_bot, z_prs_top])
        cc = [(kk - off) * P for kk in range(N_PRS)]
        disc_lines = sorted(set([c + d for c in cc for d in (-r1, -r1/2, 0.0, r1/2, r1)]))
        mesh.AddLine('x', disc_lines); mesh.AddLine('y', disc_lines)
        if PRS_MESH:
            # explicit lines at each wire-mesh trace edge so thin traces rasterize
            wc = sorted(set([(kk - off) * P + p for kk in range(N_PRS)
                             for p in np.linspace(-P/2, P/2, mesh_N + 1)]))
            wlines = sorted(set([w + d for w in wc for d in (-mesh_wt/2, mesh_wt/2)]))
            mesh.AddLine('x', wlines); mesh.AddLine('y', wlines)
        FDTD.AddEdges2Grid(dirs='xy', properties=prs_b, metal_edge_res=edge_res)
        FDTD.AddEdges2Grid(dirs='xy', properties=prs_t, metal_edge_res=edge_res)
        if RCM_ON:
            mesh.AddLine('z', np.linspace(z_prs_top, z_rcm, 11))  # cavity 2
            mesh.AddLine('z', [z_rcm, z_rcm_top])
            off_r = (N_RCM - 1) / 2.0
            rcm_lines = []
            for i in range(N_RCM):
                for j in range(N_RCM):
                    Ndiv = RCM_MAP[i * N_RCM + j] if RCM_MULTI else 1
                    rcm_lines += _rcm_lines((i - off_r) * P_RCM, (j - off_r) * P_RCM, Ndiv)
            rcm_lines = sorted(set(rcm_lines))
            mesh.AddLine('x', rcm_lines); mesh.AddLine('y', rcm_lines)
            FDTD.AddEdges2Grid(dirs='xy', properties=rcm, metal_edge_res=edge_res)

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
    print('FPC-III (feed + PRS + RCM=%s) | %dx%d | board %.0f mm (%.2f lam0)'
          % (RCM_ON, N_PRS, N_PRS, 2 * ap_half, 2 * ap_half / lam0))
    print('PRS @ %.1f mm | RCM @ %.1f mm (gap h3=%.1f, patch %.1f) | profile %.0f mm'
          % (h_cav, z_rcm, h3, rcm_s, z_rcm_top))

    sim_path = os.path.join(os.getcwd(), 'fpc3_3p25GHz')
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
