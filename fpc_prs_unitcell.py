"""
PRS (Partially Reflecting Surface) unit-cell tool for the Fabry-Perot cavity
antenna.  Simulates ONE unit cell of a periodic metal-patch PRS at normal
incidence and extracts the reflection magnitude |Gamma| and phase phi_PRS, then
turns those into the two numbers the FPC design needs:

    * cavity directivity potential   D0 = (1+|Gamma|)/(1-|Gamma|)   [dBi]
    * resonant cavity height (Trentini):
          h = (lambda/4pi)*(phi_PRS + phi_ground) + N*lambda/2 ,  phi_ground = pi

Method (the "waveguide simulator"): a p x p air guide with PEC walls on x and PMC
walls on y supports a TEM wave (E along x) that is exactly a normal-incidence plane
wave on the infinite periodic array imaged by the walls.  A TEM waveguide port
(kc = 0) launches it, and its reflection - de-embedded to the PRS plane - is Gamma.

Run:  python fpc_prs_unitcell.py
Tune p (period) and a (patch size) here so |Gamma| is high (~0.85-0.95) at 3.25 GHz;
those same p/a go into fpc_build.py, and the printed height is the cavity height.
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

# ============================ PRS unit cell (mm) ============================
f0 = 3.25e9
fc = 1.00e9

eps_r = 2.94                      # PRS superstrate: PTFE ZYF300CA-C
tan_d = 0.0016
h_prs = 0.762                     # PRS carrier thickness

p = 24.0                          # unit-cell period (lattice pitch)
a = 22.0                          # square patch size  (gap g = p - a)

d_probe = 30.0                    # air below patch (port -> PRS de-embed length)
d_top   = 40.0                    # air above PRS to the open boundary
NFREQ   = 401

sim_path = os.path.join(os.getcwd(), 'fpc_prs_unitcell')
os.makedirs(sim_path, exist_ok=True)
# ===========================================================================

z_port    = 5.0
port_len  = 2.0                    # port box thickness (excite at start, probe at stop)
z_prs     = z_port + d_probe
z_top     = z_prs + h_prs + d_top


def _rmtree_retry(path, tries=5):
    for _ in range(tries):
        try:
            shutil.rmtree(path); return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.3)


def build_and_run(run_dir):
    FDTD = openEMS(NrTS=40000, EndCriteria=1e-5)
    FDTD.SetGaussExcite(f0, fc)
    # E along x -> PEC on x-walls, PMC on y-walls; open (MUR) top & bottom
    FDTD.SetBoundaryCond(['PEC', 'PEC', 'PMC', 'PMC', 'MUR', 'MUR'])
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)

    kappa = 2 * np.pi * f0 * EPS0 * eps_r * tan_d
    sub  = CSX.AddMaterial('prs_sub', epsilon=eps_r, kappa=kappa)
    prs  = CSX.AddMetal('prs_patch')

    sub.AddBox([-p/2, -p/2, z_prs], [p/2, p/2, z_prs + h_prs], priority=0)
    prs.AddBox([-a/2, -a/2, z_prs], [a/2, a/2, z_prs], priority=10)   # patch faces cavity

    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh.AddLine('x', [-p/2, p/2]); mesh.AddLine('y', [-p/2, p/2])
    mesh.AddLine('z', [0, z_port, z_port + port_len, z_prs, z_prs + h_prs, z_top])
    res = (C0 / (f0 + fc)) / unit / 25.0
    FDTD.AddEdges2Grid(dirs='xy', properties=prs, metal_edge_res=0.3)
    mesh.AddLine('z', np.linspace(z_prs - 6, z_prs + 6, 9))
    mesh.SmoothMeshLines('all', res, ratio=1.4)

    # TEM waveguide port (kc=0): E=x_hat, H=y_hat -> Poynting +z toward the PRS.
    # Excitation at the start plane, U/I probes at the stop plane.
    port = FDTD.AddWaveGuidePort(0, [-p/2, -p/2, z_port], [p/2, p/2, z_port + port_len], 'z',
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
    print('PRS unit cell: period p=%.1f mm, patch a=%.1f mm (gap %.1f mm) on %.3f mm PTFE'
          % (p, a, p - a, h_prs))
    t0 = time.time()
    port = build_and_run(run_dir)

    f = np.linspace(f0 - fc, f0 + fc, NFREQ)
    port.CalcPort(run_dir, f)                      # raw waves at the probe plane
    _rmtree_retry(run_dir)
    print('solve done in %.0f s' % (time.time() - t0))

    # De-embed the reference plane from the probe up to the PRS surface.  openEMS's
    # own ref_plane_shift path calls back into the CSX grid post-run and segfaults on
    # this build, so we apply its exact phase-shift formula in numpy instead.
    Zr    = port.Z_ref
    shift = (z_prs - (z_port + port_len)) * unit   # probe plane -> PRS, metres
    ph    = np.real(port.beta) * shift
    uf = port.uf_tot * np.cos(-ph) + 1j * port.if_tot * Zr * np.sin(-ph)
    it = port.if_tot * np.cos(-ph) + 1j * port.uf_tot / Zr * np.sin(-ph)
    uinc  = 0.5 * (uf + it * Zr)
    gamma = (uf - uinc) / uinc
    mag   = np.abs(gamma)
    phi   = np.angle(gamma)                       # rad, referenced at PRS
    lam   = C0 / f

    # Trentini cavity height: h = lam/(4pi)*(phi + pi) + N*lam/2.  The N=0 root is a
    # sub-wavelength gap (not a real cavity); pick the order N closest to lam/2, which
    # is the fundamental Fabry-Perot cavity mode.
    base  = lam / (4 * np.pi) * (phi + np.pi) * 1e3   # mm
    half  = lam / 2 * 1e3
    h_cav = base + np.round((half - base) / half) * half
    D0_dBi = 10 * np.log10(np.clip((1 + mag) / (1 - mag), 1e-6, None))

    i0 = int(np.argmin(np.abs(f - f0)))
    print('\n================ PRS @ %.3f GHz ================' % (f0 / 1e9))
    print('  |Gamma|        = %.3f' % mag[i0])
    print('  reflection phi = %+.1f deg' % np.degrees(phi[i0]))
    print('  cavity height  = %.2f mm  (%.3f lambda0)' % (h_cav[i0], h_cav[i0] / (lam[i0] * 1e3)))
    print('  directivity D0 = %.1f dBi  (aperture-limited, ideal)' % D0_dBi[i0])
    print('===============================================\n')
    print('  -> put p=%.1f, a=%.1f and h_cav=%.1f into fpc_build.py' % (p, a, h_cav[i0]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(f / 1e9, mag); ax[0].axvline(f0 / 1e9, color='r', ls='--', lw=0.8)
    ax[0].set(title='PRS reflection |Gamma|', xlabel='Frequency (GHz)', ylabel='|Gamma|', ylim=(0, 1))
    ax[1].plot(f / 1e9, np.degrees(phi)); ax[1].axvline(f0 / 1e9, color='r', ls='--', lw=0.8)
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set(title='PRS reflection phase', xlabel='Frequency (GHz)', ylabel='phi_PRS (deg)')
    ax[2].plot(f / 1e9, h_cav, label='cavity height (mm)')
    ax2 = ax[2].twinx(); ax2.plot(f / 1e9, D0_dBi, 'C1', label='D0 (dBi)')
    ax[2].axvline(f0 / 1e9, color='r', ls='--', lw=0.8)
    ax[2].set(title='Cavity height & ideal directivity', xlabel='Frequency (GHz)',
              ylabel='h (mm)'); ax2.set_ylabel('D0 (dBi)')
    for a_ in ax:
        a_.grid(True)
    fig.tight_layout()
    out = os.path.join(sim_path, 'prs_reflection.png')
    fig.savefig(out, dpi=130); print('Saved', out)
    plt.show()


if __name__ == '__main__':
    main()
