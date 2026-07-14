"""
Fabry-Perot cavity DIRECTIVITY via Method-of-Moments (bempp-cl), PEC-in-air.
A horizontal Hertzian dipole (patch-feed proxy) under a finite PEC ground excites
the dual-layer PRS array; the cavity redirects the radiation broadside.  We solve
the EFIE for the induced surface currents and integrate the far field to get the
broadside directivity - the quantity that decides 11 vs 15 vs 17 dBi.

Frequency-domain, so NO high-Q ringdown (the thing that made FDTD slow).  PEC-in-air
ignores the thin PTFE (an approximation) and the feed is a dipole (another).  This is
a VALIDATION run: compare the directivity to the FDTD 9x9 single-PRS (~11 dBi).

Run:  python fpc_bempp_directivity.py
"""

import sys
import time
import numpy as np
import bempp_cl.api as bempp
from math import sqrt, cos, sin

bempp.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"          # GPU gives no speedup at this size

# ---------------- config (mm, matches fpc2_build) ----------------
f0 = 3.25e9
c0 = 299792458.0
lam = c0 / f0 * 1e3
k = 2 * np.pi / (lam * 1e-3) * 1e-3                   # 1/mm  (work in mm)

N_PRS = 9
P = 21.5
r1 = 6.5
pb = 20.7
sb = 16.0
h_cav = 45.2                                         # PRS bottom (loops)
h_prs = 1.52
GND = N_PRS * P                                      # 194 mm ground

MESH_GND = 10.0                                      # ground triangle size
MESH_PRS = 3.0                                       # PRS feature triangle size

# dipole feed proxy: x-oriented, at patch center just above the board
R0 = (0.0, 0.0, 1.52)
P0 = (1.0, 0.0, 0.0)
R0X, R0Y, R0Z = R0
PX, PY, PZ = P0
KK = k
# -----------------------------------------------------------------


def rect(x0, y0, x1, y1, z, hmax):
    nx = max(1, int(round(abs(x1 - x0) / hmax)))
    ny = max(1, int(round(abs(y1 - y0) / hmax)))
    xs = np.linspace(x0, x1, nx + 1); ys = np.linspace(y0, y1, ny + 1)
    V = np.array([(x, y, z) for y in ys for x in xs], float)
    idx = lambda i, j: j * (nx + 1) + i
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, cc, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [(a, b, cc), (a, cc, d)]
    return V, np.array(T, int)


def disc(cx, cy, rad, z, hmax):
    nseg = max(8, int(round(2 * np.pi * rad / hmax)))
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ring = np.array([(cx + rad * np.cos(a), cy + rad * np.sin(a), z) for a in ang])
    V = np.vstack([[cx, cy, z], ring])
    T = [(0, 1 + i, 1 + (i + 1) % nseg) for i in range(nseg)]
    return V, np.array(T, int)


def square_loop(cx, cy, outer, inner, z, hmax):
    o, ii = outer / 2, inner / 2
    parts = [rect(cx - o, cy - o, cx + o, cy - ii, z, hmax),
             rect(cx - o, cy + ii, cx + o, cy + o, z, hmax),
             rect(cx - o, cy - ii, cx - ii, cy + ii, z, hmax),
             rect(cx + ii, cy - ii, cx + o, cy + ii, z, hmax)]
    return merge(parts)


def merge(parts):
    Vs, Ts, off = [], [], 0
    for V, T in parts:
        Vs.append(V); Ts.append(T + off); off += len(V)
    return np.vstack(Vs), np.vstack(Ts)


def build_grid(hc, rcm_s=0.0, h3=0.0):
    parts = [rect(-GND/2, -GND/2, GND/2, GND/2, 0.0, MESH_GND)]      # ground
    off = (N_PRS - 1) / 2.0
    z_rcm = hc + h_prs + h3
    for i in range(N_PRS):
        cx = (i - off) * P
        for j in range(N_PRS):
            cy = (j - off) * P
            parts.append(square_loop(cx, cy, pb, sb, hc, MESH_PRS))          # PRS bottom
            parts.append(disc(cx, cy, r1, hc + h_prs, MESH_PRS))             # PRS top
            if rcm_s > 0:
                parts.append(rect(cx - rcm_s/2, cy - rcm_s/2,
                                  cx + rcm_s/2, cy + rcm_s/2, z_rcm, MESH_PRS))   # RCM patch
    V, T = merge(parts)
    return bempp.Grid(V.T.copy(), T.T.astype(np.uint32).copy())


@bempp.complex_callable
def rhs(x, nrm, d, res):
    Rx = x[0]-R0X; Ry = x[1]-R0Y; Rz = x[2]-R0Z
    rr = sqrt(Rx*Rx+Ry*Ry+Rz*Rz)
    nx = Rx/rr; ny = Ry/rr; nz = Rz/rr
    ndp = nx*PX+ny*PY+nz*PZ
    eikr = complex(cos(KK*rr), sin(KK*rr))
    t1 = KK*KK/rr
    c2 = complex(1.0/rr**3, -KK/rr**2)
    Ex = eikr*(t1*(PX-nx*ndp) + c2*(3*nx*ndp-PX))
    Ey = eikr*(t1*(PY-ny*ndp) + c2*(3*ny*ndp-PY))
    Ez = eikr*(t1*(PZ-nz*ndp) + c2*(3*nz*ndp-PZ))
    res[0] = Ey*nrm[2]-Ez*nrm[1]
    res[1] = Ez*nrm[0]-Ex*nrm[2]
    res[2] = Ex*nrm[1]-Ey*nrm[0]


def directivity(hc, rcm_s=0.0, h3=0.0):
    grid = build_grid(hc, rcm_s, h3)
    rwg = bempp.function_space(grid, "RWG", 0)
    snc = bempp.function_space(grid, "SNC", 0)
    efie = bempp.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k)
    b = bempp.GridFunction(rwg, fun=rhs, dual_space=snc)
    sol = bempp.linalg.lu(efie, b)                    # direct: robust at cavity resonance
    th = np.linspace(0, np.pi/2, 46)
    ph = np.linspace(0, 2*np.pi, 73)
    TH, PH = np.meshgrid(th, ph)
    u = np.vstack([(np.sin(TH)*np.cos(PH)).ravel(),
                   (np.sin(TH)*np.sin(PH)).ravel(), np.cos(TH).ravel()])
    ffo = bempp.operators.far_field.maxwell.electric_field(rwg, u, k)
    Esc = ffo * sol
    p = np.array(P0); r0 = np.array(R0)
    udp = (u*p[:, None]).sum(0)
    Edip = k**2*(p[:, None]-u*udp)*np.exp(-1j*k*(u*r0[:, None]).sum(0))
    Pw = (np.abs(Esc+Edip)**2).sum(0)
    dth = th[1]-th[0]; dph = ph[1]-ph[0]
    tot = np.sum(Pw*np.sin(TH).ravel())*dth*dph
    ib = int(np.argmax(Pw))
    D_peak = 10*np.log10(4*np.pi*Pw.max()/tot)
    D_bs = 10*np.log10(4*np.pi*Pw[np.argmin(TH.ravel())]/tot)
    return D_peak, np.degrees(TH.ravel()[ib]), D_bs, rwg.global_dof_count


def main():
    HC = 54.0                                          # bempp single-PRS resonance
    print("=== bempp RCM screen (PEC-in-air, h_cav=%.0f) ===" % HC)
    print("  rcm_s  h3  | peak dBi @theta | broadside dBi | s")
    t0 = time.time()
    dp, ang, dbs, nd = directivity(HC)
    print("  BASELINE (no RCM) | %6.2f @%3.0f deg | %6.2f | %.0f" % (dp, ang, dbs, time.time() - t0))
    for rcm_s in (16.0, 19.0):
        for h3 in (42.0, 46.0, 50.0):
            t0 = time.time()
            dp, ang, dbs, nd = directivity(HC, rcm_s, h3)
            print("  %5.1f %4.1f | %6.2f @%3.0f deg | %6.2f | %.0f"
                  % (rcm_s, h3, dp, ang, dbs, time.time() - t0))
    print("baseline single-PRS bempp ~8 dBi; RCM helps if broadside climbs above that")


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
