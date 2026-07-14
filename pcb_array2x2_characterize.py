"""
Characterize the 2x2 via-fed PCB array (pcb_array2x2_build): S11, input impedance,
broadside directivity vs frequency, E/H-plane co-pol cuts, realized-gain breakdown,
and a rotatable 3-D directivity pattern (radius + colour in dBi).

Usage:
    python pcb_array2x2_characterize.py            # run the FDTD solve, save data, plot
    python pcb_array2x2_characterize.py --replot   # skip the solve, replot from saved data

The full solve (a few M cells) takes minutes; all post-processed data is saved to
pcb_array2x2_3p25GHz/char_data.npz so you can re-plot / re-style instantly.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import pcb_array2x2_build as a

BAND_LO, BAND_HI = 3.10e9, 3.40e9
BOUNDARY = ['MUR', 'MUR', 'MUR', 'MUR', 'PML_8', 'PML_8']   # patch up, feed down
DR_3D    = 30.0                                             # 3-D plot dynamic range (dB below peak)
sim_path = os.path.join(os.getcwd(), 'pcb_array2x2_3p25GHz')
os.makedirs(sim_path, exist_ok=True)
data_file = os.path.join(sim_path, 'char_data.npz')

REPLOT = ('--replot' in sys.argv) or ('replot' in sys.argv[1:])


def norm_pattern_db(res):
    E = np.sqrt(np.abs(res.E_theta[0][:, 0])**2 + np.abs(res.E_phi[0][:, 0])**2)
    return 20 * np.log10(E / np.max(E) + 1e-12)


# =============================== compute or load ===============================
if not REPLOT:
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    FDTD = openEMS(NrTS=200000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(a.f0, a.fc)
    FDTD.SetBoundaryCond(BOUNDARY)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    a.build_antenna(CSX, FDTD)
    port = a.add_feed_port(FDTD)
    nf2ff = FDTD.CreateNF2FFBox()
    CSX.Write2XML(os.path.join(sim_path, 'antenna.xml'))
    FDTD.Run(sim_path, verbose=3, cleanup=True, numThreads=8)

    # S11 / Z
    f = np.linspace(a.f0 - a.fc, a.f0 + a.fc, 601)
    port.CalcPort(sim_path, f, ref_impedance=a.feed_R)
    s11    = port.uf_ref / port.uf_inc
    s11_dB = 20 * np.log10(np.abs(s11))
    Zin    = port.uf_tot / port.if_tot

    band_mask = (f >= BAND_LO) & (f <= BAND_HI)
    f_op = float(f[band_mask][np.argmin(np.abs(s11)[band_mask])])

    # far field
    center = [0, 0, (a.z_top / 2) * a.unit]
    f_dir  = np.linspace(BAND_LO - 0.15e9, BAND_HI + 0.15e9, 15)
    d_res  = nf2ff.CalcNF2FF(sim_path, f_dir, np.array([0.0]), np.array([0.0]), center=center)
    Dbroad = 10 * np.log10(np.array([d_res.Dmax[i] for i in range(len(f_dir))]))

    theta = np.linspace(0, 180, 91)                # DEGREES
    phi   = np.linspace(0, 360, 181)               # DEGREES
    sph   = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta, phi, center=center)
    U     = np.abs(sph.E_theta[0])**2 + np.abs(sph.E_phi[0])**2
    Dmax  = float(sph.Dmax[0])
    D_dBi = 10 * np.log10(Dmax * U / np.max(U) + 1e-12)

    # realized-gain breakdown @ f_op
    try:
        Prad = sph.Prad[0]
        Pacc = 0.5 * np.real(np.interp(f_op, f, port.uf_tot) * np.conj(np.interp(f_op, f, port.if_tot)))
        s11_op    = np.interp(f_op, f, np.abs(s11))
        eff_rad   = float(np.clip(Prad / Pacc, 0, 1))
        eff_match = float(1 - s11_op**2)
        realized_g = float(10 * np.log10(Dmax * eff_rad * eff_match))
    except Exception:
        eff_rad = eff_match = realized_g = float('nan')

    theta_cut = np.arange(-180, 180.5, 1.0)        # DEGREES
    e_cut = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta_cut, np.array([0.0]),  center=center)
    h_cut = nf2ff.CalcNF2FF(sim_path, np.array([f_op]), theta_cut, np.array([90.0]), center=center)
    Eplane, Hplane = norm_pattern_db(e_cut), norm_pattern_db(h_cut)

    np.savez(data_file,
             f=f, s11=s11, s11_dB=s11_dB, Zin=Zin, f_op=f_op,
             f_dir=f_dir, Dbroad=Dbroad,
             theta=theta, phi=phi, D_dBi=D_dBi, Dmax=Dmax,
             theta_cut=theta_cut, Eplane=Eplane, Hplane=Hplane,
             eff_rad=eff_rad, eff_match=eff_match, realized_g=realized_g)
    print('Saved characterization data to', data_file)

else:
    if not os.path.exists(data_file):
        sys.exit('No saved data at %s - run without --replot first.' % data_file)
    Z = np.load(data_file)
    f, s11, s11_dB, Zin = Z['f'], Z['s11'], Z['s11_dB'], Z['Zin']
    f_op = float(Z['f_op'])
    f_dir, Dbroad = Z['f_dir'], Z['Dbroad']
    theta, phi, D_dBi, Dmax = Z['theta'], Z['phi'], Z['D_dBi'], float(Z['Dmax'])
    theta_cut, Eplane, Hplane = Z['theta_cut'], Z['Eplane'], Z['Hplane']
    eff_rad, eff_match, realized_g = float(Z['eff_rad']), float(Z['eff_match']), float(Z['realized_g'])
    print('Replotting from', data_file)


# ================================ report ================================
print('\n================ ARRAY RESULTS ================')
below = s11_dB < -10.0
if np.any(below):
    band = f[below]
    print('Impedance band (S11<-10 dB): %.3f - %.3f GHz' % (band.min()/1e9, band.max()/1e9))
else:
    print('No point below -10 dB.')
print('Operating frequency (in-band best match): %.3f GHz' % (f_op/1e9))
if np.isfinite(realized_g):
    print('@ %.3f GHz | Directivity: %.2f dBi | rad.eff: %.0f%% | match: %.0f%% | '
          'realized gain: %.2f dBi'
          % (f_op/1e9, 10*np.log10(Dmax), 100*eff_rad, 100*eff_match, realized_g))
else:
    print('Directivity: %.2f dBi' % (10*np.log10(Dmax)))
print('===============================================\n')


# ============================ 2x2 summary plot ============================
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].plot(f/1e9, s11_dB); ax[0, 0].axhline(-10, color='r', ls='--', lw=0.8)
ax[0, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target')
ax[0, 0].set(title='Array reflection coefficient', xlabel='Frequency (GHz)',
             ylabel='|S11| (dB)'); ax[0, 0].legend(); ax[0, 0].grid(True)
ax[0, 1].plot(f/1e9, np.real(Zin), label='Re'); ax[0, 1].plot(f/1e9, np.imag(Zin), label='Im')
ax[0, 1].axhline(50, color='k', ls=':', lw=0.8)
ax[0, 1].set(title='Input impedance', xlabel='Frequency (GHz)',
             ylabel='Z (ohm)'); ax[0, 1].legend(); ax[0, 1].grid(True)
ax[1, 0].plot(f_dir/1e9, Dbroad, 'o-'); ax[1, 0].axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12)
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

# ===================== rotatable 3-D directivity pattern (dBi radius) =====================
Dmax_dBi = 10 * np.log10(Dmax)
floor = Dmax_dBi - DR_3D
r = np.clip(D_dBi - floor, 0.0, None)              # radius = directivity in dB above floor
TH, PH = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing='ij')
X = r * np.sin(TH) * np.cos(PH); Y = r * np.sin(TH) * np.sin(PH); Z = r * np.cos(TH)
cnorm = np.clip((D_dBi - floor) / DR_3D, 0, 1)
fig3d = plt.figure(figsize=(8, 7))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(X, Y, Z, facecolors=cm.jet(cnorm), rstride=1, cstride=1,
                  linewidth=0, antialiased=False, shade=False)
ax3d.set(xlim=(-DR_3D, DR_3D), ylim=(-DR_3D, DR_3D), zlim=(-DR_3D, DR_3D),
         xlabel='x', ylabel='y', zlabel='z (broadside)')
ax3d.set_box_aspect((1, 1, 1))
ax3d.set_title('3-D directivity @ %.3f GHz  (peak %.2f dBi)\n'
               'radius & colour = dBi (%.0f dB range) - drag to rotate'
               % (f_op/1e9, Dmax_dBi, DR_3D))
mapp = cm.ScalarMappable(cmap=cm.jet, norm=Normalize(floor, Dmax_dBi)); mapp.set_array(D_dBi)
fig3d.colorbar(mapp, ax=ax3d, shrink=0.6, label='Directivity (dBi)')
fig3d.savefig(os.path.join(sim_path, 'pattern_3d.png'), dpi=130)
print('Saved 3-D pattern to', os.path.join(sim_path, 'pattern_3d.png'))
plt.show()
