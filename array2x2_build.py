"""
2x2 array of the optimized inset-fed parasitic patch, fed by a corporate
(1 -> 4) microstrip network with quarter-wave transformers, on a shared ground.

Topology (symmetric H-tree, all paths equal length -> elements in phase ->
broadside beam):

      TL ---.        .--- TR        4 patches at (+-d/2, +-d/2), feeds pointing
            T (row)  T              inward to the central channel (x=0).
            |        |              Each row T combines two 50 ohm feeds (-> 25 ohm)
            '--[Q]---'              and a lambda/4 35.4 ohm transformer [Q] brings
                 |                  it back to 50 ohm.
      BL ---.    |    .--- BR
            T----+----T             The two 50 ohm row lines meet at the centre T,
               [Q]                  a final lambda/4 35.4 ohm transformer -> 50 ohm,
                |                    then a 50 ohm line out to the input port.
              50ohm in

Element dimensions are imported from inset_patch_build (your optimized design), so
the array always tracks the latest single element. `d` (centre-to-centre spacing)
is the main knob - sweep it to trade directivity against sidelobes/grating lobes.

NOTE: a real feed network almost always needs simulation tuning (transformer
impedance/length, T-junction compensation, and a small element retune for mutual
coupling). This is a correct FIRST-CUT geometry; run part2 to see S11 / pattern
and expect to nudge TRANS_Z / lengths.

Public interface matches inset_patch_build (build_antenna + f0, fc, feed_x,
feed_y, feed_R, h_sub, z_stk_patch, unit) so part2 can simulate it unchanged.

Run:  python array2x2_build.py      # AppCSXCAD with the input feedpoint marked
"""

import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

# reuse the optimized single-element dimensions + material/mesh settings
from inset_patch_build import (
    C0, EPS0, unit, f0, fc, eps_r, tan_d, h_sub, h_air,
    L, W, y0, g, Wf, Lp, Wp, feed_R, edge_res, min_cell,
)

# ======================= ARRAY PARAMETERS (mm) =======================
d = 69.2                       # element centre-to-centre spacing (0.75 lambda0).
                               #   sweep: 64.6 (0.7), 69.2 (0.75), 73.8 (0.8) lambda

# corporate-feed microstrip (synthesized for FR-4, 1.6 mm)
W50   = 3.08                   # 50 ohm line width
W35   = 5.25                   # 35.4 ohm quarter-wave transformer width
Lq    = 12.40                  # quarter-wave length at 3.25 GHz
Lin   = 14.0                   # straight 50 ohm input lead before the port

arr_margin = 12.0              # ground/substrate margin beyond the metal extent
air_xy, air_above, air_below = 30.0, 40.0, 20.0

mesh_res = (C0 / (f0 + fc)) / unit / 20.0
# =====================================================================

hf  = Wf / 2.0                 # half 50 ohm feed width
h35 = W35 / 2.0                # half transformer width

# z-levels (same stack-up as the single element)
z_gnd, z_drv_patch = 0.0, h_sub
z_stk_bot   = h_sub + h_air
z_stk_patch = h_sub + h_air + h_sub

# array extents (parasitic patch is the widest metal)
half_x = d/2 + Lp/2
half_y = d/2 + Wp/2
x_port = -(half_x + Lin)                       # input port at the -x board edge
feed_x, feed_y = x_port, 0.0

gnd_x0 = x_port - arr_margin
gnd_x1 = half_x + arr_margin
gnd_y0 = -half_y - arr_margin
gnd_y1 =  half_y + arr_margin


def _box(prop, x0, y0_, x1, y1, z, prio=10):
    prop.AddBox([min(x0, x1), min(y0_, y1), z], [max(x0, x1), max(y0_, y1), z], priority=prio)


def _enforce_min_cell(mesh, floor, protect=None):
    """Merge mesh lines closer than `floor` (mm); protect must-keep lines (feed)."""
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


def _draw_element(drv, par, xc, yc):
    """Draw one inset-fed driven patch (feed pointing toward x=0) + its parasitic.
    Returns nothing; the 50 ohm feed strip runs from the inset out to x=0 at y=yc."""
    d_sign = -1.0 if xc > 0 else 1.0          # +x for left column, -x for right column
    xf   = xc + d_sign * (L/2)                 # inner (feed) edge, facing centre
    x_in = xf - d_sign * y0                    # inset bottom (into the patch)
    x_out = xc - d_sign * (L/2)                # outer radiating edge

    # driven patch (bulk + two arms beside the inset), z = driven patch
    _box(drv, x_in, yc - W/2, x_out, yc + W/2, z_drv_patch)               # bulk
    _box(drv, xf,   yc + hf + g, x_in, yc + W/2, z_drv_patch)             # upper arm
    _box(drv, xf,   yc - W/2,    x_in, yc - hf - g, z_drv_patch)          # lower arm
    # 50 ohm feed strip: inset bottom -> through edge -> central channel x=0
    _box(drv, 0.0, yc - hf, x_in, yc + hf, z_drv_patch)
    # (the two regions x in [xf,x_in], |y-yc| in [hf, hf+g] stay bare = inset gaps)

    # parasitic patch on Sub2
    _box(par, xc - Lp/2, yc - Wp/2, xc + Lp/2, yc + Wp/2, z_stk_patch)


def build_antenna(CSX, FDTD):
    """Build the corporate-fed 2x2 array geometry + mesh."""
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    substrate = CSX.AddMaterial('substrate', epsilon=eps_r, kappa=kappa)
    gnd_metal = CSX.AddMetal('gnd')
    drv = CSX.AddMetal('driven')          # 4 patches + entire corporate feed
    par = CSX.AddMetal('parasitic')       # 4 parasitic patches

    # substrates + ground (shared)
    substrate.AddBox([gnd_x0, gnd_y0, 0],         [gnd_x1, gnd_y1, h_sub],       priority=0)
    substrate.AddBox([gnd_x0, gnd_y0, z_stk_bot], [gnd_x1, gnd_y1, z_stk_patch], priority=0)
    _box(gnd_metal, gnd_x0, gnd_y0, gnd_x1, gnd_y1, z_gnd)

    # four elements at the grid corners
    for xc in (-d/2, d/2):
        for yc in (-d/2, d/2):
            _draw_element(drv, par, xc, yc)

    # ---------------- corporate feed network (z = driven patch) ----------------
    # row combiners at (0, +-d/2): the two patch strips meet here; a lambda/4
    # transformer drops toward the centre, then a 50 ohm line to (0,0)
    for ys in (-1.0, 1.0):
        yT = ys * d/2
        _box(drv, -h35, yT, h35, yT - ys*Lq, z_drv_patch)          # row transformer
        _box(drv, -hf,  yT - ys*Lq, hf, 0.0, z_drv_patch)          # 50 ohm row line to centre

    # centre combiner at (0,0): two row lines meet; final lambda/4 transformer
    # runs in -x, then a straight 50 ohm lead to the input port
    _box(drv, -Lq, -h35, 0.0, h35, z_drv_patch)                    # centre transformer
    _box(drv, x_port, -hf, -Lq, hf, z_drv_patch)                   # 50 ohm input lead

    # ---------------- mesh ----------------
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [gnd_x0 - air_xy, gnd_x1 + air_xy])
    mesh.AddLine('y', [gnd_y0 - air_xy, gnd_y1 + air_xy])
    mesh.AddLine('z', [-air_below, z_stk_patch + air_above])
    mesh.AddLine('z', np.linspace(0, h_sub, 5))
    mesh.AddLine('z', np.linspace(z_stk_bot, z_stk_patch, 5))
    mesh.AddLine('z', np.linspace(z_drv_patch, z_stk_bot, 9))
    # resolve the central feed channel and the port
    mesh.AddLine('x', [x_port, -Lq, -h35, -hf, 0.0, hf, h35])
    mesh.AddLine('y', [-hf, 0.0, hf, -h35, h35])

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

    # mark the single input feedpoint (where part2 places the lumped port)
    fp = CSX.AddMetal('feed_probe')
    fp.AddCylinder([feed_x, feed_y, z_gnd], [feed_x, feed_y, z_drv_patch],
                   radius=0.5, priority=20)
    print('2x2 array, spacing d = %.1f mm (%.2f lambda0)' % (d, d / (C0/f0/unit)))
    print('Input feedpoint at (x=%.2f, y=%.2f) mm' % (feed_x, feed_y))

    sim_path = os.path.join(os.getcwd(), 'array2x2_3p25GHz')
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
