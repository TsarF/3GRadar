"""
Dual-layer PRS unit cell for the wideband FP antenna (replicating the PRS of
Ding et al., Results in Engineering 27 (2025) 106647, scaled from 7 GHz to
3.25 GHz).  The paper's PRS is what makes the cavity WIDEBAND instead of
narrowband: a solid CIRCULAR patch on the top face and a SLOTTED square patch
(square loop) on the bottom face of one thin board.  The two coupled layers give
a reflection phase that rises with frequency, so the Fabry-Perot resonance
condition stays satisfied across a wide band (a flat/positive phase slope keeps
the required cavity height nearly constant vs. frequency).

Extraction is the same TEM "waveguide simulator" as fpc_prs_unitcell.py: a PxP
air guide with PEC (x) / PMC (y) walls carries a normal-incidence plane wave; the
reflection, de-embedded to the PRS plane, gives |Gamma| and phi.

Run:  python fpc2_prs_unitcell.py
Tune P, r1, pb, sb here for high |Gamma| (~0.8-0.9) AND a gently rising phase
across 3.1-3.4 GHz, then carry them into fpc2_build.py.
"""

import os
import sys
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')

from CSXCAD import ContinuousStructure
from openEMS import openEMS

C0   = 299792458.0
EPS0 = 8.854187812813e-12
unit = 1e-3

# ============================ dual-layer PRS unit (mm) ============================
f0 = 3.25e9
fc = 1.00e9
BAND_LO, BAND_HI = 3.10e9, 3.40e9

eps_r = 2.94                      # PTFE ZYF300CA-C (paper used F4B 2.65)
tan_d = 0.0016
h_prs = 1.52                      # PRS carrier thickness

P  = 21.5                         # unit-cell period       (paper 10 mm x2.15)
r1 = 6.5                          # TOP circular patch radius (tuned: resonance above band)
pb = 20.7                         # BOTTOM square outer size  (paper p5 9.6 mm)
sb = 16.0                         # BOTTOM square slot opening (tuned: |Gamma|~0.96 flat)

d_probe  = 30.0
d_top    = 40.0
NFREQ    = 401

sim_path = os.path.join(os.getcwd(), 'fpc2_prs_unitcell')
os.makedirs(sim_path, exist_ok=True)
# ================================================================================

z_port   = 5.0
port_len = 2.0
z_prs    = z_port + d_probe        # bottom (slotted) face -> faces the cavity
z_top    = z_prs + h_prs           # top (circular) face
z_end    = z_top + d_top


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def _square_loop(prop, size, opening, z):
    """Square loop (frame) of outer `size` with a central square hole `opening`."""
    o, i = size / 2.0, opening / 2.0
    prop.AddBox([-o, -o, z], [o, -i, z], priority=10)   # bottom bar
    prop.AddBox([-o,  i, z], [o,  o, z], priority=10)    # top bar
    prop.AddBox([-o, -i, z], [-i, i, z], priority=10)    # left bar
    prop.AddBox([ i, -i, z], [o,  i, z], priority=10)    # right bar


def _disc(prop, rad, z, nseg=24):
    """Flat circular patch as a polygon (zero-height cylinders are dropped as
    'unused' by openEMS)."""
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    prop.AddPolygon(np.array([rad * np.cos(a), rad * np.sin(a)]), 'z', z, priority=10)


def build_and_run(run_dir):
    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    FDTD = openEMS(NrTS=40000, EndCriteria=1e-5)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PMC', 'PMC', 'MUR', 'MUR'])   # E along x
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)

    sub    = CSX.AddMaterial('prs_sub', epsilon=eps_r, kappa=kappa)
    m_bot  = CSX.AddMetal('prs_bottom')     # slotted square, faces cavity
    m_top  = CSX.AddMetal('prs_top')        # circular patch

    sub.AddBox([-P/2, -P/2, z_prs], [P/2, P/2, z_top], priority=0)
    _square_loop(m_bot, pb, sb, z_prs)
    _disc(m_top, r1, z_top)

    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [-P/2, P/2]); mesh.AddLine('y', [-P/2, P/2])
    mesh.AddLine('z', [0, z_port, z_port + port_len, z_prs, z_top, z_end])
    # resolve the circular patch
    mesh.AddLine('x', [-r1, -r1/2, 0, r1/2, r1]); mesh.AddLine('y', [-r1, -r1/2, 0, r1/2, r1])
    res = (C0 / (f0 + fc)) / unit / 25.0
    FDTD.AddEdges2Grid(dirs='xy', properties=m_bot, metal_edge_res=0.3)
    mesh.AddLine('z', np.linspace(z_prs - 4, z_top + 4, 9))
    mesh.SmoothMeshLines('all', res, ratio=1.4)

    port = FDTD.AddWaveGuidePort(0, [-P/2, -P/2, z_port], [P/2, P/2, z_port + port_len], 'z',
                                 ['1', '0', '0'], ['0', '1', '0'], 0, excite=1)
    os.makedirs(run_dir, exist_ok=True)
    CSX.Write2XML(os.path.join(run_dir, 'prs.xml'))
    FDTD.Run(run_dir, verbose=2, cleanup=False, numThreads=8)
    return port


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    for bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
        try:
            matplotlib.use(bk, force=True); break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    run_dir = os.path.join(sim_path, 'run')
    print('Dual-layer PRS: period P=%.1f | top circle r1=%.1f | bottom loop %.1f/%.1f mm'
          % (P, r1, pb, sb))
    t0 = time.time()
    port = build_and_run(run_dir)

    f = np.linspace(f0 - fc, f0 + fc, NFREQ)
    port.CalcPort(run_dir, f)
    _rmtree_retry(run_dir)
    print('solve done in %.0f s' % (time.time() - t0))

    # manual de-embed to the PRS bottom face (openEMS ref_plane_shift segfaults post-run)
    Zr    = port.Z_ref
    shift = (z_prs - (z_port + port_len)) * unit
    ph    = np.real(port.beta) * shift
    uf = port.uf_tot * np.cos(-ph) + 1j * port.if_tot * Zr * np.sin(-ph)
    it = port.if_tot * np.cos(-ph) + 1j * port.uf_tot / Zr * np.sin(-ph)
    uinc  = 0.5 * (uf + it * Zr)
    gamma = (uf - uinc) / uinc
    mag   = np.abs(gamma)
    phi   = np.unwrap(np.angle(gamma))
    lam   = C0 / f

    base  = lam / (4 * np.pi) * (phi + np.pi) * 1e3
    half  = lam / 2 * 1e3
    h_cav = base + np.round((half - base) / half) * half
    D0_dBi = 10 * np.log10(np.clip((1 + mag) / (1 - mag), 1e-6, None))

    lo = int(np.argmin(np.abs(f - BAND_LO)))
    i0 = int(np.argmin(np.abs(f - f0)))
    hi = int(np.argmin(np.abs(f - BAND_HI)))
    slope = (np.degrees(phi[hi]) - np.degrees(phi[lo])) / ((f[hi] - f[lo]) / 1e9)
    print('\n================ dual-layer PRS across 3.10-3.40 GHz ================')
    for tag, k in (('3.10', lo), ('3.25', i0), ('3.40', hi)):
        print('  %s GHz | |Gamma|=%.3f | phi=%+7.1f deg | h_cav=%.1f mm | D0=%.1f dBi'
              % (tag, mag[k], np.degrees(phi[k]), h_cav[k], D0_dBi[k]))
    print('  reflection-phase slope: %+.1f deg/GHz  (positive => wideband-friendly)' % slope)
    print('  cavity-height spread in band: %.1f -> %.1f mm' % (h_cav[lo], h_cav[hi]))
    print('====================================================================\n')
    print('  -> use P=%.1f, r1=%.1f, pb=%.1f, sb=%.1f and h_cav~%.1f in fpc2_build.py'
          % (P, r1, pb, sb, h_cav[i0]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(f/1e9, mag); ax[0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
    ax[0].set(title='PRS reflection |Gamma|', xlabel='GHz', ylabel='|Gamma|', ylim=(0, 1))
    ax[1].plot(f/1e9, np.degrees(phi)); ax[1].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
    ax[1].set(title='PRS reflection phase', xlabel='GHz', ylabel='phi (deg)')
    ax[2].plot(f/1e9, h_cav, label='h_cav (mm)')
    ax2 = ax[2].twinx(); ax2.plot(f/1e9, D0_dBi, 'C1'); ax2.set_ylabel('D0 (dBi)')
    ax[2].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
    ax[2].set(title='Cavity height & ideal D0', xlabel='GHz', ylabel='h (mm)')
    for a_ in ax:
        a_.grid(True)
    fig.tight_layout()
    out = os.path.join(sim_path, 'prs_reflection.png')
    fig.savefig(out, dpi=130); print('Saved', out)
    plt.show()


if __name__ == '__main__':
    main()
