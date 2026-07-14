"""
Inset-fed rectangular patch with a stacked parasitic patch, 3.25 GHz.
Substrate: JLCPCB ZYF300CA-C PTFE (Dk 2.94, Df 0.0016), 1.6 mm.
NOTE: the patch dimensions below were optimized for FR-4 - they are DETUNED on this
lower-Dk laminate (they now resonate ~20% high). Re-run the optimizer to retune.

A linearly-polarized alternative to the corner-truncated CP element in
part1_build_visualize.py. Topology after Khalily et al. (IEEE T-AP 2018): an
inset-fed driven patch broadbanded by an equal-size parasitic patch stacked above
an air gap. The inset feed sets the match (no probe/via needed); the air gap and
parasitic size set the bandwidth.

Dimensions are a transmission-line-model starting point for 3.25 GHz on standard
1.6 mm FR-4 - run part2 to see the S11 and let part3 retune inset depth + air gap
for S11 < -10 dB across 3.1-3.4 GHz.

Stack-up (z, mm), ground at z = 0:
    0.00            ground plane
    0.00 -> 0.762   driven PTFE substrate (Sub1, 0.762 mm)
    0.762           inset-fed driven patch + 50 ohm microstrip feed line
    0.762 -> +ga    air gap
    .. -> ..+1.60   parasitic PTFE substrate (Sub2, 1.60 mm)
    top             parasitic patch (equal size, no feed)

Public interface matches part1_build_visualize so part2/part3 can use it by
changing their import: build_antenna(CSX, FDTD) plus the module-level names
f0, fc, feed_x, feed_y, feed_R, h_sub, z_stk_patch, unit.

Run:  python inset_patch_build.py      # opens AppCSXCAD with the feedpoint marked
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
fc = 1.00e9                       # Gaussian corner -> covers ~2.25-4.25 GHz

# Substrate: JLCPCB ZYF300CA-C PTFE/Teflon (Dk 2.94, Df 0.0016)
eps_r = 2.94
tan_d = 0.0016
h_sub  = 0.762                     # driven substrate Sub1 thickness (feed + driven patch)
h_sub2 = 1.6                       # parasitic carrier Sub2 thickness

# Driven patch (PTFE-scaled starting estimate; run part3 to optimize on this stack)
L  = 26.5                          # resonant length (x): radiating edges at +-L/2
W  = 32.0                          # width (y)

# Inset feed  <-- KEY MATCH TUNING PARAMETER (y0)
Wf = 1.94                         # 50 ohm microstrip feed width on 0.762 mm PTFE (Dk 2.94)
y0 = 3.1                          # inset depth into the patch (sets the match)
g  = 1.0                          # inset gap (clearance each side of the feed)
Lf = 12.0                         # feed-line length from patch edge out to the port

# Air gap between driven and parasitic layers  <-- KEY BANDWIDTH PARAMETER
h_air = 7.2

# Parasitic patch (equal size by default; Lp/Wp are independent tuning knobs)
Lp = 31.0
Wp = 42.0

# Lateral margin of ground / substrate beyond the metal extent
margin = 15

# Air box margins around the structure (free-space buffer to the boundaries)
air_xy    = 30.0
air_above = 40.0
air_below = 20.0

# Probe/feed
feed_R = 50.0

# Mesh control
edge_res = 0.4                                   # fine cells at metal edges/slots
mesh_res = (C0 / (f0 + fc)) / unit / 20.0        # coarse cells in the air (~3 mm)
min_cell = 0.05                                  # floor on cell size (mm): merges
                                                 #   near-coincident mesh lines so no
                                                 #   degenerate (tiny) cell forms
# ======================================================================

# ---- derived geometry (mm) ----
hf       = Wf / 2.0                       # half feed width
x_in     = -L/2 + y0                      # x of the inset bottom (feed-to-patch join)
x_port   = -L/2 - Lf                      # x of the feed-line outer end = port

# feed/port location exposed for part2's lumped port (vertical, z=0 -> h_sub)
feed_x, feed_y = x_port, 0.0

# ground / substrate footprint (snug around the metal extent)
gnd_x0 = x_port - margin
gnd_x1 =  L/2   + margin
gnd_y0 = -W/2   - margin
gnd_y1 =  W/2   + margin

# ---- derived z-levels (mm) ----
z_gnd       = 0.0
z_drv_patch = h_sub
z_stk_bot   = h_sub + h_air
z_stk_patch = h_sub + h_air + h_sub2


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    """Add a zero-thickness metal sheet box at height z."""
    prop.AddBox([x0, y0_, z], [x1, y1, z], priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    """Drop mesh lines closer than `floor` (mm) to their kept neighbour so the grid
    can never hold a degenerate tiny cell (the FDTD timestep follows the smallest
    cell). `protect` is {axis: [coords]} of lines that must survive (the feed must
    land on its line to excite); a protected line outranks a crowding neighbour.
    Domain boundaries per axis are always kept so the extent stays exact."""
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
    """Populate CSX with the inset-fed + parasitic patch and build the mesh.
    Excitation/ports/NF2FF are added separately (part2)."""

    # ---------------- materials ----------------
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d        # dielectric loss
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd_metal = CSX.AddMetal('gnd')
    drv_patch = CSX.AddMetal('driven_patch')             # patch + inset feed line
    par_patch = CSX.AddMetal('parasitic_patch')

    # ---------------- substrates (two FR-4 slabs) ----------------
    substrate.AddBox([gnd_x0, gnd_y0, 0],         [gnd_x1, gnd_y1, h_sub],       priority=0)
    substrate.AddBox([gnd_x0, gnd_y0, z_stk_bot], [gnd_x1, gnd_y1, z_stk_patch], priority=0)

    # ---------------- ground plane ----------------
    _box(gnd_metal, gnd_x0, gnd_y0, gnd_x1, gnd_y1, z_gnd, prio=10)

    # ---------------- driven patch with inset feed ----------------
    # Decomposed into rectangles so the two inset notches stay bare metal-free:
    #   A: patch bulk beyond the inset depth
    #   B/C: the upper/lower patch arms alongside the inset
    #   D: the central feed strip + external 50 ohm line (one continuous trace)
    _box(drv_patch, x_in,      -W/2,      L/2,   W/2,        z_drv_patch)   # A bulk
    _box(drv_patch, -L/2,       hf + g,   x_in,  W/2,        z_drv_patch)   # B upper arm
    _box(drv_patch, -L/2,      -W/2,      x_in, -(hf + g),   z_drv_patch)   # C lower arm
    _box(drv_patch, x_port,    -hf,       x_in,  hf,         z_drv_patch)   # D feed strip
    # -> the bare regions x in [-L/2, x_in], |y| in [hf, hf+g] are the inset gaps

    # ---------------- parasitic patch (plain rectangle on Sub2) ----------------
    _box(par_patch, -Lp/2, -Wp/2, Lp/2, Wp/2, z_stk_patch)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # air box extent
    x_lim_lo = gnd_x0 - air_xy
    x_lim_hi = gnd_x1 + air_xy
    y_lim    = max(abs(gnd_y0), abs(gnd_y1)) + air_xy
    mesh.AddLine('x', [x_lim_lo, x_lim_hi])
    mesh.AddLine('y', [-y_lim, y_lim])
    mesh.AddLine('z', [-air_below, z_stk_patch + air_above])

    # discretize the two substrates and the air gap in z
    mesh.AddLine('z', np.linspace(0, h_sub, 5))                       # driven sub
    mesh.AddLine('z', np.linspace(z_stk_bot, z_stk_patch, 5))         # parasitic sub
    mesh.AddLine('z', np.linspace(z_drv_patch, z_stk_bot, 9))         # air gap

    # resolve the feed, inset gaps and patch edges before smoothing
    mesh.AddLine('x', [x_port, -L/2, x_in, L/2, feed_x])
    mesh.AddLine('y', [-W/2, -(hf + g), -hf, feed_y, hf, hf + g, W/2])

    # thirds-rule edge refinement on the metals
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd_metal, metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=drv_patch, metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=par_patch, metal_edge_res=edge_res)

    # fill the rest of the grid up to the coarse resolution
    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)

    # remove sliver cells; protect the feed lines so the lumped port lands on them
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x], 'y': [feed_y]})

    return mesh


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)

    build_antenna(CSX, FDTD)

    # Mark the feedpoint (where part2 places the lumped port): a short vertical post
    # at the feed-line end, ground -> trace. VISUALIZATION ONLY (added here, not in
    # build_antenna), so it never enters part2's simulation.
    feed_probe = CSX.AddMetal('feed_probe')
    feed_probe.AddCylinder([feed_x, feed_y, z_gnd],
                           [feed_x, feed_y, z_drv_patch], radius=0.5, priority=20)
    print('Feedpoint marked at (x=%.3f, y=%.3f) mm, z: %.2f -> %.2f mm'
          % (feed_x, feed_y, z_gnd, z_drv_patch))

    sim_path = os.path.join(os.getcwd(), 'inset_patch_3p25GHz')
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
