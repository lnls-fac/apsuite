#!/usr/bin/env python-sirius
"""."""

import sys
import epics
import numpy as np
import matplotlib.pyplot as plt
import time


parms = {
    # dt, ampl, nrpts_period, nr_periods, tau_period
    'BO-Fam:PS-QF': [0.5, 20.0, 50, 4, 1, False],
    'BO-Fam:PS-QD': [0.5, 30.0, 50, 4, 1, False],
    'BO-Fam:PS-SF': [0.5, 149.0, 50, 4, 1, False],
    'BO-Fam:PS-SD': [0.5, 149.0, 50, 4, 1, False],
    'BO-Fam:PS-B-2': [0.5, 60.0, 100, 4, 1, False],
    'BO-02D:PS-QS': [0.5, 10.0, 50, 4, 1, False],
    'TB-Fam:PS-B': [0.5, 250.0, 50, 4, 1, True],
    'TB-01:PS-QD1': [0.5, 10.0, 50, 4, 1, False],
    'TB-01:PS-QF1': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-QD2A': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-QF2A': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-QD2B': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-QF2B': [0.5, 10.0, 50, 4, 1, False],
    'TB-03:PS-QD3': [0.5, 10.0, 50, 4, 1, False],
    'TB-03:PS-QF3': [0.5, 10.0, 50, 4, 1, False],
    'TB-04:PS-QD4': [0.5, 10.0, 50, 4, 1, False],
    'TB-04:PS-QF4': [0.5, 10.0, 50, 4, 1, False],
    'TB-01:PS-CH-1': [0.5, 10.0, 50, 4, 1, False],
    'TB-01:PS-CV-1': [0.5, 10.0, 50, 4, 1, False],
    'TB-01:PS-CH-2': [0.5, 10.0, 50, 4, 1, False],
    'TB-01:PS-CV-2': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-CH-1': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-CV-1': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-CH-2': [0.5, 10.0, 50, 4, 1, False],
    'TB-02:PS-CV-2': [0.5, 10.0, 50, 4, 1, False],
    'TB-04:PS-CH': [0.5, 10.0, 50, 4, 1, False],
    'TB-04:PS-CV-1': [0.5, 10.0, 50, 4, 1, False],
    'TB-04:PS-CV-2': [0.5, 10.0, 50, 4, 1, False],
}


def gen_waveform(dt, ampl, nrpts_period, nr_periods, tau_period, square):
    """."""
    period = dt * nrpts_period
    tau = tau_period*period
    tot_nrpts = nrpts_period * nr_periods
    v = list(range(0, tot_nrpts))
    t = dt * np.array(v)
    s = np.sin(2*np.pi*t/period) * np.exp(-t/tau)
    if square:
        w = ampl * s**2
    else:
        w = ampl * s
    t = np.append(t, t[-1] + dt)
    w = np.append(w, 0.0)

    return t, w


def ps_cycle(psname, plot=True):
    """."""
    t, w = gen_waveform(*parms[psname])
    if plot:
        plt.plot(t, w, 'o')
        plt.xlabel('time [s]')
        plt.ylabel('Current [A]')
        plt.show()
    else:
        pv_sp = epics.PV(psname + ':Current-SP')
        for i in range(0, len(t)-1):
            print('{}/{} : {} A'.format(i+1, len(w), w[i]))
            pv_sp.value = w[i]
            time.sleep(t[i+1]-t[i])
        print('{}/{} : {} A'.format(len(w), len(w), w[-1]))
        pv_sp.value = w[-1]


def run():
    """."""
    if len(sys.argv) > 1:
        psname = sys.argv[1]
        if len(sys.argv) > 2:
            plot_flag = False if sys.argv[2] == 'true' else True
        else:
            plot_flag = False
        # print(psname, plot_flag)
        ps_cycle(psname, plot=plot_flag)


run()
