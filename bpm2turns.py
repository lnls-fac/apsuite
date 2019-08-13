#!/usr/bin/env python-sirius
"""."""

import epics
import numpy as np
import matplotlib.pyplot as plt


cA02 = epics.PV('BO-02U:DI-BPM:SP_AArrayData')
cB02 = epics.PV('BO-02U:DI-BPM:SP_BArrayData')
cC02 = epics.PV('BO-02U:DI-BPM:SP_CArrayData')
cD02 = epics.PV('BO-02U:DI-BPM:SP_DArrayData')

cA03 = epics.PV('BO-03U:DI-BPM:SP_AArrayData')
cB03 = epics.PV('BO-03U:DI-BPM:SP_BArrayData')
cC03 = epics.PV('BO-03U:DI-BPM:SP_CArrayData')
cD03 = epics.PV('BO-03U:DI-BPM:SP_DArrayData')

PosKxSP02 = epics.PV('BO-02U:DI-BPM:PosKx-SP')
PosKySP02 = epics.PV('BO-02U:DI-BPM:PosKy-SP')
PosKxSP03 = epics.PV('BO-03U:DI-BPM:PosKx-SP')
PosKySP03 = epics.PV('BO-03U:DI-BPM:PosKy-SP')

nrpts_pre02 = epics.PV('BO-02U:DI-BPM:ACQSamplesPre-SP')
nrpts_pst02 = epics.PV('BO-02U:DI-BPM:ACQSamplesPost-SP')
nrpts_pre03 = epics.PV('BO-03U:DI-BPM:ACQSamplesPre-SP')
nrpts_pst03 = epics.PV('BO-03U:DI-BPM:ACQSamplesPost-SP')


def calc_pos(A, B, C, D, Kx, Ky):
    """."""
    # A: outter top
    # B: inner bottom
    # C: inner top
    # D: outter bottom
    x = ((A-B)/(A+B) + (D-C)/(D+C))*Kx/2
    y = ((A-B)/(A+B) - (D-C)/(D+C))*Ky/2
    return x, y


data = cA02.value
plt.plot(data)
plt.show()


def process():
    """."""
    A02 = np.array(cA02.value)
    B02 = np.array(cB02.value)
    C02 = np.array(cC02.value)
    D02 = np.array(cD02.value)
    Kx = PosKxSP02.value
    Ky = PosKySP02.value
    nrpts = int((nrpts_pre02.value + nrpts_pst02.value)/2)
    nrpts = nrpts_pre02.value + nrpts_pst02.value - 3
    x02_1, y02_1 = calc_pos(np.std(A02[:nrpts]), np.std(B02[:nrpts]),
                            np.std(C02[:nrpts]), np.std(D02[:nrpts]),
                            Kx, Ky)
    x02_2, y02_2 = calc_pos(np.std(A02[nrpts:]), np.std(B02[nrpts:]),
                            np.std(C02[nrpts:]), np.std(D02[nrpts:]),
                            Kx, Ky)
    A03 = np.array(cA03.value)
    B03 = np.array(cB03.value)
    C03 = np.array(cC03.value)
    D03 = np.array(cD03.value)
    Kx = PosKxSP03.value
    Ky = PosKySP03.value
    nrpts = int((nrpts_pre03.value + nrpts_pst03.value)/2)
    nrpts = nrpts_pre03.value + nrpts_pst03.value - 3
    x03_1, y03_1 = calc_pos(np.std(A03[:nrpts]), np.std(B03[:nrpts]),
                            np.std(C03[:nrpts]), np.std(D03[:nrpts]),
                            Kx, Ky)
    x03_2, y03_2 = calc_pos(np.std(A03[nrpts:]), np.std(B03[nrpts:]),
                            np.std(C03[nrpts:]), np.std(D03[nrpts:]),
                            Kx, Ky)
    bpm02x = [x02_1, x02_2]
    bpm02y = [y02_1, y02_2]
    bpm03x = [x03_1, x03_2]
    bpm03y = [y03_1, y03_2]
    return bpm02x, bpm02y, bpm03x, bpm03y
