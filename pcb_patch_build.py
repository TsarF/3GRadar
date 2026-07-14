"""
Via-fed microstrip patch on a JLCPCB 4-layer impedance-controlled stackup
(JLC04161H-7628, 1.6 mm), with an AIR-GAP stacked parasitic above it. 3.25 GHz.

The 4-layer board carries the driven element and feed; a second board (Sub2) holds
the parasitic patch, suspended above the main board by an air gap (standoffs):

    parasitic patch (top of Sub2)             z = z_par   (= z_drv + h_air + h_sub2)
      Sub2 FR-4 (h_sub2)
      AIR GAP (h_air)  <-- tunable: driven<->parasitic coupling / bandwidth
    --- main 4-layer PCB ---
    L1 (top)    driven patch (via-fed)        z = +1.275 mm
      prepreg 0.21 + core 1.065
    L2          empty
    L3          ground plane (+ via antipad)  z = 0
      prepreg 0.21
    L4 (bottom) 50 ohm microstrip feed        z = -0.21 mm

A 0.3 mm via (square-modeled) carries the feed from the L4 microstrip up through a
square antipad in the L3 ground (and the empty L2) to the L1 driven patch. The
air-gap parasitic broadbands the element (the strong dual resonance the thin
prepreg gap can't give). Feed is below the ground -> no feed radiation in the beam.

Tunable knobs: L, W (driven), Lp, Wp (parasitic), fv (via offset = match),
antipad (via reactance), h_air (gap = bandwidth). h_sub2 is a fixed board choice.

Interface: build_antenna(CSX, FDTD); add_feed_port(FDTD); f0, fc, feed_R, unit, z_top.

Run:  python pcb_patch_build.py   - AppCSXCAD with the feed via marked
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

# materials: main board (JLC04161H-7628 FR-4) and the parasitic carrier Sub2 (FR-4)
eps_r  = 4.3
tan_d  = 0.02
t_core    = 1.065                 # L2 <-> L3 core
t_prepreg = 0.21                  # L1<->L2 and L3<->L4 prepreg (7628)
h_sub2    = 1.60                  # parasitic carrier board thickness (fixed)

# driven + parasitic dimensions (optimized: worst-in-band |S11| = -12.6 dB)
L  = 22.2                         # driven length (x)   [2x2 6-knob opt, Gr=13.0 dBi]
W  = 20.2                         # driven width  (y)
Lp = 26.7                         # parasitic length (x)
Wp = 29.4                         # parasitic width  (y)

# air gap (driven L1 -> parasitic Sub2 bottom)  <-- KEY BANDWIDTH KNOB
h_air = 7.0

# via feed
fv      = 9.8                     # via offset from patch centre toward -x  (match)
a_via   = 0.15                    # half-side of the 0.3 mm square via
antipad = 0.60                    # half-side of the square ground clearance

# L4 microstrip feed line (50 ohm on 0.21 mm FR-4)
Wf = 0.38
Lf = 10.0

margin    = 40.0
air_xy    = 30.0
air_above = 40.0
air_below = 20.0
feed_R = 50.0

edge_res = 0.3
mesh_res = (C0 / (f0 + fc)) / unit / 20.0
min_cell = 0.08
# =====================================================================

# fixed z-levels in the main board (z = 0 at L3 ground)
z_feed = -t_prepreg               # L4
z_gnd  = 0.0                      # L3
z_drv  = t_core + t_prepreg       # L1 driven (top of main board)


def _derive():
    """Recompute geometry that depends on the tunable knobs."""
    global xv, x_port, feed_x, feed_y, gnd_x0, gnd_x1, gnd_y0, gnd_y1
    global z_air_top, z_par, z_top
    xv = -fv
    x_port = -L/2 - Lf
    feed_x, feed_y = x_port, 0.0
    half_x = max(L, Lp) / 2.0
    half_y = max(W, Wp) / 2.0
    gnd_x0 = x_port - margin
    gnd_x1 = half_x + margin
    gnd_y0 = -half_y - margin
    gnd_y1 =  half_y + margin
    z_air_top = z_drv + h_air            # bottom of Sub2
    z_par     = z_air_top + h_sub2       # parasitic patch (top of Sub2)
    z_top     = z_par                    # nf2ff-centre reference


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


def build_antenna(CSX, FDTD):
    """Via-fed driven patch (4-layer board) + air-gap parasitic + mesh."""
    _derive()
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd  = CSX.AddMetal('gnd')        # L3
    drv  = CSX.AddMetal('driven')     # L1 + via
    par  = CSX.AddMetal('parasitic')  # top of Sub2
    feed = CSX.AddMetal('feed')       # L4

    # main 4-layer board dielectric (L4 -> L1) and the parasitic carrier Sub2
    substrate.AddBox([gnd_x0, gnd_y0, z_feed],    [gnd_x1, gnd_y1, z_drv], priority=0)
    substrate.AddBox([gnd_x0, gnd_y0, z_air_top], [gnd_x1, gnd_y1, z_par], priority=0)

    # L1 driven patch + parasitic patch on top of Sub2
    _sheet(drv, -L/2,  -W/2,  L/2,  W/2,  z_drv)
    _sheet(par, -Lp/2, -Wp/2, Lp/2, Wp/2, z_par)

    # L3 ground = full plane minus a square antipad around the via
    _sheet(gnd, gnd_x0, gnd_y0, xv - antipad, gnd_y1, z_gnd)
    _sheet(gnd, xv + antipad, gnd_y0, gnd_x1, gnd_y1, z_gnd)
    _sheet(gnd, xv - antipad, antipad,  xv + antipad, gnd_y1, z_gnd)
    _sheet(gnd, xv - antipad, gnd_y0, xv + antipad, -antipad, z_gnd)

    # L4 feed line + via (L4 -> L1 driven; clears the L3 antipad, passes empty L2)
    _sheet(feed, x_port, -Wf/2, xv, Wf/2, z_feed)
    _box(drv, xv - a_via, -a_via, xv + a_via, a_via, z_feed, z_drv, prio=15)

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [gnd_x0 - air_xy, gnd_x1 + air_xy])
    mesh.AddLine('y', [gnd_y0 - air_xy, gnd_y1 + air_xy])
    mesh.AddLine('z', [z_feed - air_below, z_par + air_above])
    mesh.AddLine('z', np.linspace(z_feed, 0, 3))            # feed substrate
    mesh.AddLine('z', np.linspace(0, z_drv, 7))             # main board above ground
    mesh.AddLine('z', np.linspace(z_drv, z_air_top, 9))     # air gap
    mesh.AddLine('z', np.linspace(z_air_top, z_par, 4))     # Sub2
    mesh.AddLine('x', [x_port, xv - antipad, xv - a_via, xv, xv + a_via, xv + antipad])
    mesh.AddLine('y', [-antipad, -Wf/2, -a_via, 0.0, a_via, Wf/2, antipad])

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=drv,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=par,  metal_edge_res=edge_res)
    FDTD.AddEdges2Grid(dirs='xy', properties=feed, metal_edge_res=edge_res)

    mesh.SmoothMeshLines('all', mesh_res, ratio=1.4)
    _enforce_min_cell(mesh, min_cell, protect={'x': [feed_x, xv], 'y': [feed_y]})
    return mesh


def add_feed_port(FDTD):
    """Lumped port: L4 trace end against the L3 ground (z: L4 -> L3)."""
    return FDTD.AddLumpedPort(1, feed_R,
                              [x_port, feed_y, z_feed], [x_port, feed_y, z_gnd],
                              'z', 1.0, priority=5, edges2grid='xy')


if __name__ == '__main__':
    FDTD = openEMS()
    CSX  = ContinuousStructure()
    FDTD.SetCSX(CSX)
    build_antenna(CSX, FDTD)
    mark = CSX.AddMetal('via_marker')
    mark.AddCylinder([xv, 0, z_feed], [xv, 0, z_drv], radius=0.18, priority=20)
    print('Via-fed driven patch (L1) + air-gap parasitic')
    print('z: feed=%.3f gnd=0 driven=%.3f | air gap %.2f | parasitic=%.3f mm'
          % (z_feed, z_drv, h_air, z_par))
    print('via at x=%.2f | port at x=%.2f' % (xv, x_port))

    sim_path = os.path.join(os.getcwd(), 'pcb_patch_3p25GHz')
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
