"""
Part 2 - Simulate the inset-fed parasitic patch element and post-process results.

Imports the geometry builder from inset_patch_build (so the scripts can never
disagree), adds the microstrip feed (lumped port) and an NF2FF box, runs the FDTD
engine, then reports / plots (this is a LINEARLY-POLARIZED antenna):
    * |S11| and input impedance        -> resonances + impedance bandwidth
    * broadside directivity vs frequency
    * E-/H-plane co-pol patterns at the operating frequency
    * a rotatable 3-D radiation (directivity) pattern
    * directivity, radiation efficiency, mismatch efficiency, realized gain
      evaluated at the in-band best-match frequency

Run:  python part2_simulate.py
Note: a real 3-D FDTD run; uses all CPU cores. The final plt.show() opens an
interactive window - drag the 3-D pattern to rotate it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from CSXCAD import ContinuousStructure
from openEMS import openEMS

from array2x2_build import (
    build_antenna, f0, fc, feed_x, feed_y, feed_R,
    h_sub, z_stk_patch, unit,
)

BAND_LO, BAND_HI = 3.10e9, 3.40e9                 # band of interest
sim_path = os.path.join(os.getcwd(), 'inset_patch_3p25GHz')
os.makedirs(sim_path, exist_ok=True)


def norm_pattern_db(res):
    """Normalized total-field pattern (dB) for a single-freq NF2FF cut result."""
    E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
    return 20 * np.log10(E / np.max(E) + 1e-12)


# ============================ build + excite ============================
FDTD = openEMS(NrTS=60000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'PML_8'])   # PML on top

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

build_antenna(CSX, FDTD)                                             # geometry + mesh

# microstrip feed: vertical lumped port from ground (z=0) up to the trace
port = FDTD.AddLumpedPort(1, feed_R,
                          [feed_x, feed_y, 0], [feed_x, feed_y, h_sub],
                          'z', 1.0, priority=5, edges2grid='xy')

# far-field recording box
nf2ff = FDTD.CreateNF2FFBox()

# ================================ run ================================
CSX.Write2XML(os.path.join(sim_path, 'antenna.xml'))
FDTD.Run(sim_path, verbose=3, cleanup=True, numThreads=8)

# =============================== S11 / Z ===============================
f = np.linspace(f0 - fc, f0 + fc, 601)
port.CalcPort(sim_path, f, ref_impedance=feed_R)

s11    = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))
Zin    = port.uf_tot / port.if_tot

# resonances (local minima of |S11|) and -10 dB impedance bandwidth
below = s11_dB < -10.0
print('\n================ RESULTS ================')
if np.any(below):
    band = f[below]
    f_lo, f_hi = band.min(), band.max()
    fc_imp = 0.5 * (f_lo + f_hi)
    print('Impedance band (S11<-10 dB): %.3f - %.3f GHz  (%.1f%% @ %.3f GHz)'
          % (f_lo/1e9, f_hi/1e9, 100*(f_hi - f_lo)/fc_imp, fc_imp/1e9))
else:
    print('No point below -10 dB - retune L / inset depth y0.')
res_idx = [i for i in range(1, len(f)-1)
           if s11_dB[i] < s11_dB[i-1] and s11_dB[i] < s11_dB[i+1] and s11_dB[i] < -6]
print('Resonances (|S11| minima): ' +
      ', '.join('%.3f GHz' % (f[i]/1e9) for i in res_idx))

# operating frequency = best match (|S11| minimum) WITHIN the target band
band_mask = (f >= BAND_LO) & (f <= BAND_HI)
f_op = float(f[band_mask][np.argmin(np.abs(s11)[band_mask])])
print('Operating frequency (in-band best match): %.3f GHz' % (f_op/1e9))

# ===================== far field: directivity + pattern =====================
center = [0, 0, (z_stk_patch / 2) * unit]                # SI metres, mid-structure

# broadside directivity across the band
f_dir  = np.linspace(BAND_LO - 0.15e9, BAND_HI + 0.15e9, 15)
d_res  = nf2ff.CalcNF2FF(sim_path, f_dir, np.array([0.0]), np.array([0.0]), center=center)
Dbroad = 10 * np.log10(np.array([d_res.Dmax[i] for i in range(len(f_dir))]))

# full-sphere directivity at the operating frequency (for the 3-D plot + gain)
theta = np.linspace(0, 180, 91)                        # DEGREES (CalcNF2FF wants degrees)
phi   = np.linspace(0, 360, 181)                       # DEGREES
sph   = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta, phi, center=center)
U     = np.abs(sph.E_theta[0])**2 + np.abs(sph.E_phi[0])**2     # radiation intensity
D_lin = sph.Dmax[0] * U / np.max(U)                      # linear directivity, peak = Dmax
D_dBi = 10 * np.log10(D_lin + 1e-12)

# ================ directivity / realized gain @ f_op ================
Dmax = sph.Dmax[0]
try:
    Prad = sph.Prad[0]
    Pacc = 0.5 * np.real(np.interp(f_op, f, port.uf_tot) *
                         np.conj(np.interp(f_op, f, port.if_tot)))
    s11_op    = np.interp(f_op, f, np.abs(s11))
    eff_rad   = float(np.clip(Prad / Pacc, 0, 1))        # dielectric/conductor loss
    eff_match = float(1 - s11_op**2)                      # mismatch efficiency, 1-|S11|^2
    realized_g = 10 * np.log10(Dmax * eff_rad * eff_match)
    print('@ %.3f GHz | Directivity: %.2f dBi | rad.eff: %.0f%% | match: %.0f%% | '
          'realized gain: %.2f dBi'
          % (f_op/1e9, 10*np.log10(Dmax), 100*eff_rad, 100*eff_match, realized_g))
except Exception as e:
    print('Directivity: %.2f dBi  (gain breakdown skipped: %s)' % (10*np.log10(Dmax), e))
print('=========================================\n')

# principal-plane co-pol cuts at f_op (theta -180..180, all DEGREES for CalcNF2FF)
theta_cut = np.arange(-180, 180.5, 1.0)
e_cut = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta_cut,
                        np.array([0.0]), center=center)        # E-plane (phi=0)
h_cut = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta_cut,
                        np.array([90.0]), center=center)       # H-plane (phi=90)
Eplane, Hplane = norm_pattern_db(e_cut), norm_pattern_db(h_cut)

# ============================ 2x2 summary plot ============================
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

ax[0, 0].plot(f/1e9, s11_dB)
ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
ax[0, 0].set(title='Reflection coefficient', xlabel='Frequency (GHz)',
             ylabel='|S11| (dB)'); ax[0, 0].legend(); ax[0, 0].grid(True)

ax[0, 1].plot(f/1e9, np.real(Zin), label='Re')
ax[0, 1].plot(f/1e9, np.imag(Zin), label='Im')
ax[0, 1].axhline(50, color='k', ls=':', lw=0.8)
ax[0, 1].set(title='Input impedance', xlabel='Frequency (GHz)',
             ylabel='Z (ohm)'); ax[0, 1].legend(); ax[0, 1].grid(True)

ax[1, 0].plot(f_dir/1e9, Dbroad, 'o-')
ax[1, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
ax[1, 0].set(title='Broadside directivity', xlabel='Frequency (GHz)',
             ylabel='D (dBi)'); ax[1, 0].grid(True)

ax[1, 1].plot(theta_cut, Eplane, label='E-plane (phi=0)')
ax[1, 1].plot(theta_cut, Hplane, '--', label='H-plane (phi=90)')
ax[1, 1].set(title='Co-pol pattern @ %.3f GHz' % (f_op/1e9), xlabel='Theta (deg)',
             ylabel='Normalized (dB)', ylim=(-40, 2), xlim=(-90, 90))
ax[1, 1].legend(); ax[1, 1].grid(True)

fig.tight_layout()
fig.savefig(os.path.join(sim_path, 'results.png'), dpi=130)
print('Saved summary plots to', os.path.join(sim_path, 'results.png'))

# ===================== rotatable 3-D radiation pattern =====================
# Radius = LINEAR directivity normalized to the peak (broadside -> 1, nulls -> 0),
# so the beam SHAPE is visible. A dB radius with a fixed floor adds a big constant
# offset that swamps the angular variation and renders the dome as a near-sphere.
Dmax_dBi = 10 * np.log10(Dmax)
rad = D_lin / np.max(D_lin)                             # 0..1 pattern shape
TH, PH = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing='ij')   # deg -> rad for geometry
X = rad * np.sin(TH) * np.cos(PH)
Y = rad * np.sin(TH) * np.sin(PH)
Z = rad * np.cos(TH)

# colour by directivity in dBi over a 30 dB window below the peak
vmin, vmax = Dmax_dBi - 30.0, Dmax_dBi
cnorm = np.clip((D_dBi - vmin) / (vmax - vmin), 0, 1)

fig3d = plt.figure(figsize=(8, 7))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(X, Y, Z, facecolors=cm.jet(cnorm),
                  rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax3d.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
         xlabel='x', ylabel='y', zlabel='z (broadside)')
ax3d.set_box_aspect((1, 1, 1))
ax3d.set_title('3-D directivity pattern @ %.3f GHz  (peak %.2f dBi)\n'
               'radius = normalized linear directivity - drag to rotate'
               % (f_op/1e9, Dmax_dBi))
mappable = cm.ScalarMappable(cmap=cm.jet, norm=Normalize(vmin, vmax))
mappable.set_array(D_dBi)
fig3d.colorbar(mappable, ax=ax3d, shrink=0.6, label='Directivity (dBi)')
fig3d.savefig(os.path.join(sim_path, 'pattern_3d.png'), dpi=130)
print('Saved 3-D pattern snapshot to', os.path.join(sim_path, 'pattern_3d.png'))

plt.show()
