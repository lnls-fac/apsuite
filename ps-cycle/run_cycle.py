#!/usr/bin/env python-sirius
"""."""

import sys
import epics
import numpy as np
import matplotlib.pyplot as plt
import time


parms = {
    # dt, ampl, nrpts_period, nr_periods, tau_period
    'BO-Fam:PS-QF': [0.5, 5.0, 50, 4, 1],
    'BO-Fam:PS-QD': [0.5, 5.0, 50, 4, 1],
    'BO-Fam:PS-SF': [0.5, 5.0, 50, 4, 1],
    'BO-Fam:PS-SD': [0.5, 5.0, 50, 4, 1],
    'BO-Fam:PS-B-2': [0.5, 50.0, 50, 4, 1],
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
