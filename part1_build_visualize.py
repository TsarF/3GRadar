"""
Part 1 - Build and visualize the stacked, corner-truncated, slotted CP patch
element in openEMS.

Single element, linearly scaled (k = 1.5) from Shekhawat et al. (IEEE AWPL 2010)
to a 3.25 GHz centre, but kept on STANDARD 1.6 mm FR-4 (instead of the ideal
scaled 2.37 mm). Because h is not scaled, these dimensions are a STARTING POINT
- the simulation in Part 2 tells you how far off 3.25 GHz you are so you can
re-tune the patch lengths and feed position.

Stack-up (z, mm), ground at z = 0:
    0.00            ground plane
    0.00 -> 1.60    driven FR-4 substrate
    1.60            driven patch (corner-truncated + slot)
    1.60 -> 8.35    air gap (6.75 mm)
    8.35 -> 9.95    stacked FR-4 substrate
    9.95            stacked (parasitic) patch (slot, no truncation)

Run:  python part1_build_visualize.py      # opens AppCSXCAD
Needs: openEMS with Python bindings (openEMS, CSXCAD) + AppCSXCAD for viewing.
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

# ---------- constants / drawing unit ----------
C0   = 299792458.0
EPS0 = 8.854187812813e-12
unit = 1e-3                       # 1 drawing unit = 1 mm

# ======================= DESIGN PARAMETERS (mm) =======================
# Excitation band (also sets max-frequency mesh resolution)
f0 = 3.25e9                       # centre frequency
fc = 1.50e9                       # 20 dB Gaussian corner -> covers ~1.75-4.75 GHz

# Substrate (kept at standard 1.6 mm FR-4)
eps_r = 4.37
tan_d = 0.025
h_sub = 1.6

# Air gap between driven and stacked layers (scaled value)
h_air = 7.4

# Driven patch (corner-truncated, slotted) - scaled starting point
L1, W1 = 23.75, 21.0              # resonant length (x), width (y); L1/W1 = 1.143
tc     = 4.88                    # corner truncation, applied to TR and BL corners
l1, w1 = 9.0, 1.5               # driven slot: length (x), width (y), centred

# Stacked / parasitic patch (slotted, NOT truncated)
L2, W2 = 29.25, 20.0
l2, w2 = 12.0, 1.5

# Probe (SMA) feed on the driven patch  <-- KEY CP + MATCH TUNING PARAMETER
feed_x, feed_y = 6.275, 0.0
feed_R = 50.0

# Ground / substrate lateral size (covers largest patch + margin)
gnd_x = L2 + 24.0
gnd_y = W2 + 24.0

# Air box margins around the structure (free-space buffer to the boundaries)
air_xy    = 30.0                 # beyond the ground edges in x and y
air_above = 40.0                 # above the top patch
air_below = 20.0                 # below the ground plane

# Mesh control
edge_res = 0.4                                   # fine cells at metal edges/slots
mesh_res = (C0 / (f0 + fc)) / unit / 20.0        # coarse cells in the air (~3 mm)
min_cell = 0.05                                  # floor on cell size (mm): mesh
                                                 #   lines closer than this are
                                                 #   merged so two near-coincident
                                                 #   edges can't form a degenerate
                                                 #   (timestep-killing) tiny cell
# ======================================================================

# ---- derived z-levels (mm) ----
z_gnd       = 0.0
z_drv_patch = h_sub
z_stk_bot   = h_sub + h_air
z_stk_patch = h_sub + h_air + h_sub


def _box(prop, x0, y0, x1, y1, z, prio=10):
    """Add a zero-thickness metal sheet box at height z."""
    prop.AddBox([x0, y0, z], [x1, y1, z], priority=prio)


def _poly(prop, pts_xy, z, prio=10):
    """Add a flat polygon (list of (x, y)) in the z = const plane."""
    pts = np.array(pts_xy, dtype=float).T        # shape (2, N)
    prop.AddPolygon(pts, 'z', z, priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    """Drop mesh lines closer than `floor` (mm) to their kept neighbour, so the
    grid can never hold a degenerate near-zero-width cell. The FDTD timestep is
    set by the smallest cell (CFL limit), so one sliver tanks the whole run;
    near-coincident metal edges collapse to a single line. The two domain
    boundaries per axis are always preserved so the simulation extent is exact.

    `protect` is an optional {axis: [coords]} of lines that must survive (e.g. the
    feed location, which the lumped port must land on to excite). When a kept line
    and a protected line are too close, the protected one wins; the other is the
    one dropped."""
    protect = protect or {}

    def protected(d, v):
        return any(abs(v - p) <= 1e-6 for p in protect.get(d, []))

    for d in 'xyz':
        lines = np.unique(np.asarray(mesh.GetLines(d), dtype=float))
        if lines.size < 3:
            continue
        kept = [lines[0]]
        for x in lines[1:-1]:                        # interior lines
            if protected(d, x):
                # never drop a protected line; instead drop the kept neighbours
                # that crowd it (but never the domain boundary at kept[0])
                while len(kept) > 1 and x - kept[-1] < floor and not protected(d, kept[-1]):
                    kept.pop()
                kept.append(x)
            elif x - kept[-1] >= floor:
                kept.append(x)
        last = lines[-1]                             # keep the domain edge exact
        if last - kept[-1] < floor and not protected(d, kept[-1]):
            kept[-1] = last                          # too close: replace, don't add
        else:
            kept.append(last)
        mesh.SetLines(d, kept)


def build_antenna(CSX, FDTD):
    """Populate CSX with materials + primitives and build the mesh on FDTD's grid.
    Excitation, ports and NF2FF are added separately (Part 2)."""

    # ---------------- materials ----------------
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d        # dielectric loss as conductivity
    substrate   = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd_metal   = CSX.AddMetal('gnd')
    drv_patch   = CSX.AddMetal('driven_patch')
    stk_patch   = CSX.AddMetal('stacked_patch')

    # ---------------- substrates (two FR-4 slabs) ----------------
    substrate.AddBox([-gnd_x/2, -gnd_y/2, 0],         [gnd_x/2, gnd_y/2, h_sub],       priority=0)
    substrate.AddBox([-gnd_x/2, -gnd_y/2, z_stk_bot], [gnd_x/2, gnd_y/2, z_stk_patch], priority=0)

    # ---------------- ground plane ----------------
    _box(gnd_metal, -gnd_x/2, -gnd_y/2, gnd_x/2, gnd_y/2, z_gnd, prio=10)

    # ---------------- driven patch: 2 truncated polygons + 2 bridge boxes ----------------
    # Left block (x: -L1/2 .. -l1/2), bottom-left corner truncated
    _poly(drv_patch, [(-l1/2, -W1/2), (-l1/2,  W1/2), (-L1/2,  W1/2),
                      (-L1/2, -W1/2 + tc), (-L1/2 + tc, -W1/2)], z_drv_patch)
    # Right block (x: l1/2 .. L1/2), top-right corner truncated
    _poly(drv_patch, [( l1/2, -W1/2), ( L1/2, -W1/2), ( L1/2,  W1/2 - tc),
                      ( L1/2 - tc,  W1/2), ( l1/2,  W1/2)], z_drv_patch)
    # Bridges above / below the centred slot
    _box(drv_patch, -l1/2,  w1/2, l1/2,  W1/2, z_drv_patch)
    _box(drv_patch, -l1/2, -W1/2, l1/2, -w1/2, z_drv_patch)

    # ---------------- stacked patch: 4 boxes around the centred slot ----------------
    _box(stk_patch, -L2/2, -W2/2, -l2/2, W2/2, z_stk_patch)
    _box(stk_patch,  l2/2, -W2/2,  L2/2, W2/2, z_stk_patch)
    _box(stk_patch, -l2/2,  w2/2,  l2/2, W2/2, z_stk_patch)
    _box(stk_patch, -l2/2, -W2/2,  l2/2, -w2/2, z_stk_patch)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # air box extent
    x_lim = gnd_x/2 + air_xy
    y_lim = gnd_y/2 + air_xy
    mesh.AddLine('x', [-x_lim, x_lim])
    mesh.AddLine('y', [-y_lim, y_lim])
    mesh.AddLine('z', [-air_below, z_stk_patch + air_above])

    # discretize the two substrates and the air gap in z
    mesh.AddLine('z', np.linspace(0, h_sub, 5))                       # 4 cells in driven sub
    mesh.AddLine('z', np.linspace(z_stk_bot, z_stk_patch, 5))         # 4 cells in stacked sub
    mesh.AddLine('z', np.linspace(z_drv_patch, z_stk_bot, 9))         # 8 cells across air gap

    # help the slots and feed get resolved before smoothing
    mesh.AddLine('y', [-w1/2, 0, w1/2, -w2/2, w2/2, feed_y])
    mesh.AddLine('x', [-l1/2, l1/2, -l2/2, l2/2, feed_x])

    # thirds-rule edge refinement on the metals
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd_metal,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=drv_patch,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=stk_patch,  metal_edge_res=edge_res)

    # fill the rest of the grid up to the coarse resolution
    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)

    # remove any sliver cells left by near-coincident edges (keeps the timestep
    # sane regardless of the exact patch dimensions). Protect the feed lines: the
    # lumped port must land exactly on them to excite, so they outrank the floor.
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x], 'y': [feed_y]})

    return mesh


if __name__ == '__main__':
    # Build geometry only and open it in AppCSXCAD for a visual check.
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)

    build_antenna(CSX, FDTD)

    # Mark the probe feedpoint so it is obvious in AppCSXCAD: a short vertical post
    # at (feed_x, feed_y) from the ground plane up to the driven patch - exactly
    # where Part 2 places the lumped port. This is its own property so AppCSXCAD
    # colours it distinctly. VISUALIZATION ONLY: it is added here, not inside
    # build_antenna, so it never enters Part 2's simulation (a real metal post
    # there would short the patch to ground).
    feed_probe = CSX.AddMetal('feed_probe')
    feed_probe.AddCylinder([feed_x, feed_y, z_gnd],
                           [feed_x, feed_y, z_drv_patch], radius=0.5, priority=20)
    print('Feedpoint marked at (x=%.3f, y=%.3f) mm, z: %.2f -> %.2f mm'
          % (feed_x, feed_y, z_gnd, z_drv_patch))

    sim_path = os.path.join(os.getcwd(), 'cp_patch_3p25GHz')
    os.makedirs(sim_path, exist_ok=True)
    xml_file = os.path.join(sim_path, 'antenna.xml')
    CSX.Write2XML(xml_file)
    print('Geometry written to:', xml_file)

    nx, ny, nz = (len(CSX.GetGrid().GetLines(d)) for d in range(3))
    print('Mesh lines  x:%d  y:%d  z:%d  (~%.2f M cells)'
          % (nx, ny, nz, nx * ny * nz / 1e6))

    try:
        from CSXCAD import AppCSXCAD_BIN
        os.system(AppCSXCAD_BIN + ' "%s"' % xml_file)
    except Exception:
        os.system('AppCSXCAD "%s"' % xml_file)
