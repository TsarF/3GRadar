"""
bempp (MoM, PEC-in-air) SCREEN for the multi-size RCM bandwidth idea.

Question this answers (QUALITATIVELY - direction only, per HANDOFF gotcha #7):
  A UNIFORM square-patch RCM is a single resonance -> tall narrow broadside-gain peak
  that sags at 3.10 / 3.40 GHz.  Does a MULTI-SIZE RCM (several square sizes per
  super-cell, like the paper's 3 sizes) add a 2nd/3rd resonance that FLATTENS the
  broadside directivity across 3.10-3.40 GHz?

Method (ONE process, compile amortized): loop FREQS (outer) x variants (inner).  The
numba dipole-RHS freezes k at compile time, so we build a FRESH rhs per frequency via a
closure factory make_rhs(kval).  First frequency pays the one-time numba/OpenCL compile;
later frequencies reuse the warm bempp context.  At each freq we print broadside
directivity for: no-RCM | uniform RCM | multi-size RCM, then a trend summary.

PEC-in-air => absolute dBi undershoots FDTD by ~3 dB and optimum spacings differ
(~0.83x); we ONLY read the TREND (does multi-size sag less at band edges?).  Any number
reported to the user must come from FDTD.

Run (thread-limited so it does not starve the DE):
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 python fpc_bempp_rcm_screen.py
"""

import sys
import time
import numpy as np
import bempp_cl.api as bempp
from math import sqrt, cos, sin

bempp.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"

# ---------------- config (mm) ----------------
c0 = 299792458.0
FREQS = (3.10e9, 3.25e9, 3.40e9)

N_PRS = 9
P = 21.5
r1 = 6.5
pb = 20.7
sb = 16.0
HC = 54.0                                            # bempp single-PRS resonance (PEC-in-air)
h_prs = 1.52
GND = N_PRS * P

MESH_GND = 10.0
MESH_PRS = 3.0

# RCM screen points (bempp geometry space; not the FDTD numbers)
H3 = 46.0
RCM_UNIFORM = 17.0
RCM_MULTI = (14.5, 17.0, 19.5)                       # 3 interleaved sizes -> 3 resonances

# dipole feed proxy (x-oriented, above patch center)
R0 = (0.0, 0.0, 1.52)
P0 = (1.0, 0.0, 0.0)
R0X, R0Y, R0Z = R0
PX, PY, PZ = P0
# ---------------------------------------------


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


def build_grid(hc, rcm_sizes=None, h3=0.0):
    """rcm_sizes: None -> no RCM; float -> uniform; tuple -> interleaved multi-size."""
    parts = [rect(-GND/2, -GND/2, GND/2, GND/2, 0.0, MESH_GND)]
    off = (N_PRS - 1) / 2.0
    z_rcm = hc + h_prs + h3
    for i in range(N_PRS):
        cx = (i - off) * P
        for j in range(N_PRS):
            cy = (j - off) * P
            parts.append(square_loop(cx, cy, pb, sb, hc, MESH_PRS))
            parts.append(disc(cx, cy, r1, hc + h_prs, MESH_PRS))
            if rcm_sizes is not None:
                if isinstance(rcm_sizes, (tuple, list)):
                    s = rcm_sizes[(i + j) % len(rcm_sizes)]   # interleave sizes
                else:
                    s = float(rcm_sizes)
                if s > 0:
                    parts.append(rect(cx - s/2, cy - s/2, cx + s/2, cy + s/2, z_rcm, MESH_PRS))
    V, T = merge(parts)
    return bempp.Grid(V.T.copy(), T.T.astype(np.uint32).copy())


def make_rhs(kval):
    """Fresh numba-compiled dipole-illumination tangential-E RHS with k frozen = kval."""
    KK = kval

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
    return rhs


def directivity(kval, rhs_fun, hc, rcm_sizes=None, h3=0.0):
    grid = build_grid(hc, rcm_sizes, h3)
    rwg = bempp.function_space(grid, "RWG", 0)
    snc = bempp.function_space(grid, "SNC", 0)
    efie = bempp.operators.boundary.maxwell.electric_field(rwg, rwg, snc, kval)
    b = bempp.GridFunction(rwg, fun=rhs_fun, dual_space=snc)
    sol = bempp.linalg.lu(efie, b)
    th = np.linspace(0, np.pi/2, 46)
    ph = np.linspace(0, 2*np.pi, 73)
    TH, PH = np.meshgrid(th, ph)
    u = np.vstack([(np.sin(TH)*np.cos(PH)).ravel(),
                   (np.sin(TH)*np.sin(PH)).ravel(), np.cos(TH).ravel()])
    ffo = bempp.operators.far_field.maxwell.electric_field(rwg, u, kval)
    Esc = ffo * sol
    p = np.array(P0); r0 = np.array(R0)
    udp = (u*p[:, None]).sum(0)
    Edip = kval**2*(p[:, None]-u*udp)*np.exp(-1j*kval*(u*r0[:, None]).sum(0))
    Pw = (np.abs(Esc+Edip)**2).sum(0)
    dth = th[1]-th[0]; dph = ph[1]-ph[0]
    tot = np.sum(Pw*np.sin(TH).ravel())*dth*dph
    ib = int(np.argmax(Pw))
    D_peak = 10*np.log10(4*np.pi*Pw.max()/tot)
    D_bs = 10*np.log10(4*np.pi*Pw[np.argmin(TH.ravel())]/tot)
    return D_peak, np.degrees(TH.ravel()[ib]), D_bs, rwg.global_dof_count


def main():
    variants = (("no-RCM", None),
                ("uniform s=%.1f" % RCM_UNIFORM, RCM_UNIFORM),
                ("multi %s" % (RCM_MULTI,), RCM_MULTI))
    print("=== bempp multi-size RCM bandwidth screen (PEC-in-air) | HC=%.0f h3=%.0f ===" % (HC, H3))
    print("  reading TREND only: does broadside D sag LESS across band for multi vs uniform?\n")
    bs_table = {name: {} for name, _ in variants}
    for f0 in FREQS:
        lam = c0 / f0 * 1e3
        kval = 2 * np.pi / (lam * 1e-3) * 1e-3            # 1/mm
        rhs_fun = make_rhs(kval)
        print("  f=%.3f GHz  (k=%.5f /mm)" % (f0/1e9, kval))
        print("    variant                 | peak dBi @theta | BROADSIDE dBi | DOF |   s")
        for name, sizes in variants:
            t0 = time.time()
            dp, ang, dbs, nd = directivity(kval, rhs_fun, HC, sizes, H3)
            bs_table[name][f0] = dbs
            print("    %-22s | %6.2f @%3.0f deg | %6.2f | %d | %.0f"
                  % (name, dp, ang, dbs, nd, time.time() - t0))
            sys.stdout.flush()
        print()

    print("=== BROADSIDE directivity trend (dBi) - the screen result ===")
    hdr = "  %-24s" % "variant" + "".join("  %7.3fGHz" % (f/1e9) for f in FREQS) + "   sag(max-min)"
    print(hdr)
    for name, _ in variants:
        row = [bs_table[name][f] for f in FREQS]
        sag = max(row) - min(row)
        print("  %-24s" % name + "".join("  %10.2f" % v for v in row) + "   %6.2f" % sag)
    print("\n  Interpretation: SMALLER sag = flatter across band. If multi-size sag < uniform")
    print("  sag AND its band-edge D >= uniform's, the multi-size RCM is worth an FDTD check.")


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
