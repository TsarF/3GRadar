"""
Two-port coupling characterization of two 2x2 arrays 200 mm apart (pcb_link2_build):
    S11 = TX reflection   (port 1 match)
    S21 = TX -> RX coupling (isolation between the two antennas)

Excite port 1 (TX), port 2 is a matched 50 ohm load (RX). S21 = the coupled wave
emerging at port 2 normalized to the incident wave at port 1.

Usage:
    python pcb_link2_characterize.py            # run the FDTD solve, save data, plot
    python pcb_link2_characterize.py --replot   # skip the solve, replot from saved data
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import pcb_link2_build as lk

BAND_LO, BAND_HI = 3.10e9, 3.40e9
sim_path = os.path.join(os.getcwd(), 'pcb_link2_3p25GHz')
os.makedirs(sim_path, exist_ok=True)
data_file = os.path.join(sim_path, 'link_data.npz')

REPLOT = ('--replot' in sys.argv) or ('replot' in sys.argv[1:])

if not REPLOT:
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    # Accurate S21 down to ~-45 dB needs (1) PML on ALL sides - MUR reflects at
    # ~-30..-40 dB, the same level as the coupling we're measuring - and (2) a
    # tight energy criterion so the numerical noise floor sits well below S21.
    # EndCriteria=1e-5 (-50 dB energy decay) -> trustworthy to roughly -40..-45 dB.
    # High-Q + tight decay = many timesteps, so NrTS is raised; the run stops early
    # once EndCriteria is met (watch the log to confirm it's reached, not capped).
    FDTD = openEMS(NrTS=400000, EndCriteria=1e-5)
    FDTD.SetGaussExcite(lk.f0, lk.fc)
    FDTD.SetBoundaryCond(['PML_8'] * 6)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    lk.build_antenna(CSX, FDTD)
    p_tx, p_rx = lk.add_feed_ports(FDTD)     # port 1 excited, port 2 matched load
    CSX.Write2XML(os.path.join(sim_path, 'antenna.xml'))
    FDTD.Run(sim_path, verbose=3, cleanup=True, numThreads=8)

    f = np.linspace(lk.f0 - lk.fc, lk.f0 + lk.fc, 801)
    p_tx.CalcPort(sim_path, f, ref_impedance=lk.feed_R)
    p_rx.CalcPort(sim_path, f, ref_impedance=lk.feed_R)
    s11 = p_tx.uf_ref / p_tx.uf_inc          # TX reflection
    s21 = p_rx.uf_ref / p_tx.uf_inc          # TX -> RX coupling

    np.savez(data_file, f=f, s11=s11, s21=s21, sep=lk.SEP)
    print('Saved link data to', data_file)
else:
    if not os.path.exists(data_file):
        sys.exit('No saved data at %s - run without --replot first.' % data_file)
    Z = np.load(data_file)
    f, s11, s21, = Z['f'], Z['s11'], Z['s21']
    print('Replotting from', data_file)

s11_dB = 20 * np.log10(np.abs(s11))
s21_dB = 20 * np.log10(np.abs(s21))

# ================================ report ================================
band = (f >= BAND_LO) & (f <= BAND_HI)
i_c  = int(np.argmin(np.abs(f - 3.25e9)))
worst_s21 = float(np.max(s21_dB[band]))       # highest coupling in band
print('\n================ TWO-ANTENNA LINK (%.0f mm apart) ================' % lk.SEP)
print('@ 3.250 GHz : S11 = %.1f dB | S21 (coupling) = %.1f dB' % (s11_dB[i_c], s21_dB[i_c]))
print('In-band 3.1-3.4 GHz : worst-case coupling S21 = %.1f dB (isolation %.1f dB)'
      % (worst_s21, -worst_s21))
print('==================================================================\n')

# ================================ plot ================================
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(f/1e9, s11_dB, label='S11 (TX match)')
ax.plot(f/1e9, s21_dB, label='S21 (TX->RX coupling)', lw=2)
ax.axhline(-10, color='r', ls='--', lw=0.8)
ax.axvspan(BAND_LO/1e9, BAND_HI/1e9, color='g', alpha=0.12, label='target band')
ax.set(title='Two 2x2 arrays %.0f mm apart: reflection & coupling' % lk.SEP,
       xlabel='Frequency (GHz)', ylabel='dB')
ax.legend(); ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(sim_path, 'link_s21.png'), dpi=130)
print('Saved plot to', os.path.join(sim_path, 'link_s21.png'))
plt.show()
