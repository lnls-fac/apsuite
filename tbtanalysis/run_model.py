#!/usr/bin/env python-sirius

"""."""

import numpy as np
import matplotlib.pyplot as plt
from pymodels import si
import pyaccel
from pyaccel.naff import naff_general as _naff_general


def plot_dtune(x0_max, nrpts=5):
    tr = si.create_accelerator()
    tr.cavity_on = False
    tr.radiation_on = False
    twiss, *_ = pyaccel.optics.calc_twiss(tr)
    # fam_data = si.get_family_data(tr)
    betax = twiss.betax[0]
    # plt.plot(twiss.etax[bpms_idx], 'o-')
    # plt.show()

    x0_max /= 1e6  # [m]
    x0_tiny = 1/1e6  # [m]

    x0 = np.linspace(0, x0_max, nrpts) + x0_tiny
    dtunex = np.zeros(len(x0))

    nrturns = 6*333
    for i in range(len(x0)):
        print(i)
        traj, *_ = pyaccel.tracking.ring_pass(tr, [x0[i],0,0,0,0,0], nrturns, True)
        dtunex[i], _ = _naff_general(signal=traj[0,:], is_real=True, nr_ff=1, window=1)
    dtunex -= dtunex[0]

    # plt.plot(traj[0,:])

    # --- (tune shift com amplitude linear com ação)
    # dtunex ~ kxx x² 
    # dtunex ~ kxx (betax * J)
    p = np.polyfit(x0*1e6, dtunex, 2)
    x0_fit = np.linspace(0, max(x0), 40)
    dtunex_fit = np.polyval(p, x0_fit*1e6)

    kxx = p[0]  # 1/um²
    kxx_norm = kxx*(betax*1e6)  # 1/um
    print('betax     : {} m'.format(betax))
    print('kxx       : {} 1/um²'.format(kxx))
    print('kxx*betax : {} 1/um'.format(kxx_norm))
    plt.plot(x0*1e6, dtunex, 'o', label='dnux = ({}/um²) x²'.format(kxx))
    plt.plot(x0_fit*1e6, dtunex_fit, '-', label='fit')
    plt.xlabel('x0 [um]')
    plt.ylabel('dnux')
    plt.legend()
    plt.show()


plot_dtune(x0_max=328, nrpts=10)
