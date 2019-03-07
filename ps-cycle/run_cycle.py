#!/usr/bin/env python-sirius
"""."""

import sys
import epics
import numpy as np
import matplotlib.pyplot as plt
import time


parms = {
    # dt, ampl, nrpts_period, nr_periods, tau_period
    'BO-Fam:PS-QF': [0.5, 120.0, 50, 4, 1],
    'BO-Fam:PS-QD': [0.5, 30.0, 50, 4, 1],
    'BO-Fam:PS-SF': [0.5, 149.0, 50, 4, 1],
    'BO-Fam:PS-SD': [0.5, 149.0, 50, 4, 1],
    'BO-Fam:PS-B-2': [0.5, 60.0, 50, 4, 1],
    'TB-Fam:PS-B': [0.5, 250.0, 50, 4, 1],
    'TB-01:PS-QD1': [0.5, 10.0, 50, 4, 1],
    'TB-01:PS-QF1': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-QD2A': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-QF2A': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-QD2B': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-QF2B': [0.5, 10.0, 50, 4, 1],
    'TB-03:PS-QD3': [0.5, 10.0, 50, 4, 1],
    'TB-03:PS-QF3': [0.5, 10.0, 50, 4, 1],
    'TB-04:PS-QD4': [0.5, 10.0, 50, 4, 1],
    'TB-04:PS-QF4': [0.5, 10.0, 50, 4, 1],
    'TB-01:PS-CH-1': [0.5, 10.0, 50, 4, 1],
    'TB-01:PS-CV-1': [0.5, 10.0, 50, 4, 1],
    'TB-01:PS-CH-2': [0.5, 10.0, 50, 4, 1],
    'TB-01:PS-CV-2': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-CH-1': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-CV-1': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-CH-2': [0.5, 10.0, 50, 4, 1],
    'TB-02:PS-CV-2': [0.5, 10.0, 50, 4, 1],
    'TB-04:PS-CH': [0.5, 10.0, 50, 4, 1],
    'TB-04:PS-CV-1': [0.5, 10.0, 50, 4, 1],
    'TB-04:PS-CV-2': [0.5, 10.0, 50, 4, 1],
}


def gen_waveform(dt, ampl, nrpts_period, nr_periods, tau_period):
    """."""
    period = dt * nrpts_period
    tau = tau_period*period
    tot_nrpts = nrpts_period * nr_periods
    v = list(range(0, tot_nrpts))
    t = dt * np.array(v)
    w = ampl * np.sin(2*np.pi*t/period) * np.exp(-t/tau)
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
            print('{}/{} : {} A'.format(i, len(w), w[i]))
            pv_sp.value = w[i]
            time.sleep(t[i+1]-t[i])
        print('{}/{} : {} A'.format(len(w)-1, len(w), w[-1]))
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
