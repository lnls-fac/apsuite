#!/usr/bin/env python-sirius

"""."""

import numpy as np
import matplotlib.pyplot as plt
from pymodels import si
import pyaccel
from pyaccel.naff import naff_general as _naff_general


def plot_model_dtunex(u_max, plane, nrpts=5, plot_flag=True):

    plane = plane.lower()
    tr = si.create_accelerator()
    tr.cavity_on = False
    tr.radiation_on = False
    twiss, *_ = pyaccel.optics.calc_twiss(tr)
    # print(twiss.mux[-1]/2/np.pi, twiss.muy[-1]/2/np.pi)
    
    if plane == 'xx':
        beta = twiss.betax[0] # [m]
        u_init = 0 # [m]
        tune0_nom = twiss.mux[-1]/2/np.pi - 49
        xlabel, ylabel = r'$J_x$ [um]', r'$\delta\nu_x \; x \; 10^4$'
        label = r'fit of tracking: $k_{{xx}} = {:.1f} \; mm^{{-1}}$'
    elif plane == 'yy':
        beta = twiss.betay[0] # [m]
        u_init = 50e-6 # [m]
        tune0_nom = twiss.muy[-1]/2/np.pi - 14
        xlabel, ylabel = r'$J_y$ [um]', r'$\delta\nu_y \; x \; 10^4$'
        label = r'fit of tracking: $k_{{yy}} = {:.1f} \; mm^{{-1}}$'
    elif plane == 'xy':
        beta = twiss.betay[0] # [m]
        u_init = 50e-6 # [m]
        tune0_nom = 0.09618826
        xlabel, ylabel = r'$J_y$ [um]', r'$\delta\nu_x \; x \; 10^4$'
        label = r'fit of tracking: $k_{{xy}} = {:.1f} \; mm^{{-1}}$'
    if plane == 'yx':
        beta = twiss.betax[0] # [m]
        u_init = 0 # [m]
        tune0_nom = 0.19237653
        xlabel, ylabel = r'$J_x$ [um]', r'$\delta\nu_y \; x \; 10^4$'
        label = r'fit of tracking: $k_{{yx}} = {:.1f} \; mm^{{-1}}$'

    tiny = 1e-6  # [m]
    u0 = np.linspace(u_init, u_max, nrpts) + tiny
    tune = np.zeros(len(u0))

    nrturns = 6*333
    for i in range(len(u0)):
        if plane in ('xx', 'yx'):
            traj, *_ = pyaccel.tracking.ring_pass(tr, [u0[i],0,tiny,0,0,0], nrturns, True)
        elif plane in ('yy', 'xy'):
            traj, *_ = pyaccel.tracking.ring_pass(tr, [tiny,0,u0[i],0,0,0], nrturns, True)
        if plane in ('xx', 'xy'):
            ffqs, _ = _naff_general(signal=traj[0,:], is_real=True, nr_ff=3, window=1)
        elif plane in ('yx', 'yy'):
            ffqs, _ = _naff_general(signal=traj[2,:], is_real=True, nr_ff=3, window=1)
        ffqs = abs(ffqs)
        # print(ffqs)
        idx = (np.abs(ffqs - tune0_nom)).argmin()
        tune[i] = ffqs[idx]
        tune0_nom = tune[i]
        print('point: {:03d}, pos: {:.3f} mm, tune: {:7f}'.format(i, u0[i]*1e3, tune[i]))

    # --- (tune shift com amplitude linear com ação)
    # tune ~ tune0 + k u² 
    # tune ~ tune0 + k (beta * J)

    J = u0**2/beta  # [m]

    p = np.polyfit(J, tune, min(5,len(J)-1))
    J_fit = np.linspace(min(J), max(J), 40)
    tune_fit = np.polyval(p, J_fit)

    tune0 = p[-1]
    k_norm = p[-2]  # 1/m

    if plot_flag:
        
        print('tune0  : {:.6f}'.format(tune0))
        print('coefJ² : {:.5e} 1/m²'.format(p[0]))
        print('k_norm : {:0f} 1/m'.format(k_norm))
        print('k_norm : {:2f} 1/mm'.format(k_norm/1e3))
        print('k_norm : {:4f} 1/um'.format(k_norm/1e6))
        
        plt.plot(J_fit*1e6, 1e4*(tune_fit - tune0), '--', color='C0', label=label.format(k_norm/1e3))
        plt.plot(J*1e6, 1e4*(tune - tune0), 'o', color='C0')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.title('Model Tune-Shift with Amplitude')
        plt.grid()
        plt.show()

    return k_norm, np.array(tune), np.array(J), tune0, np.array(J_fit), np.array(tune_fit), u0


def calc_kxx(u_max=3000e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'xx', nrpts, plot_flag)


def calc_kxy(u_max=1200e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'xy', nrpts, plot_flag)


def calc_kyy(u_max=1200e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'yy', nrpts, plot_flag)


def calc_kyx(u_max=3000e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'yx', nrpts, plot_flag)


def call_all(fnametype='svg'):

    kxx, tunexx, Jxx, tune0xx, J_fitxx, tune_fitxx, u0 = calc_kxx(plot_flag=False)
    kxy, tunexy, Jxy, tune0xy, J_fitxy, tune_fitxy, u0 = calc_kxy(plot_flag=False)
    kyx, tuneyx, Jyx, tune0yx, J_fityx, tune_fityx, u0 = calc_kyx(plot_flag=False)
    kyy, tuneyy, Jyy, tune0yy, J_fityy, tune_fityy, u0 = calc_kyy(plot_flag=False)

    ylim = (-80, 160)

    graph, (ax1, ax2) = plt.subplots(1, 2)
    graph.subplots_adjust(wspace=0)
    graph.suptitle(
        'Tune-Shifts with Amplitudes\n' +
        r'Nominal Model $\xi_{{x,y}} \approx 2.5$')

    ax1.plot(J_fitxy*1e6, +1e4*(tune_fitxy - tune0xy), '--', color='C0', label=r'$\partial \nu_x / \partial J_y : k_{{xy}} = {:.1f} \; mm^{{-1}} $'.format(kxy/1e3))
    ax1.plot(J_fityy*1e6, +1e4*(tune_fityy - tune0yy), '--', color='C1', label=r'$\partial \nu_y / \partial J_y : k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy/1e3))
    ax1.set(xlabel=r'$J_y \, [\mu m]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax1.set_xlim([0.5, 0])
    ax1.set_ylim(ylim)
    ax1.grid()
    ax1.legend()
    
    ax2.plot(J_fitxx*1e6, +1e4*(tune_fitxx - tune0xx), '--', color='C0', label=r'$\partial \nu_x / \partial J_x : k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx/1e3))
    ax2.plot(J_fityx*1e6, +1e4*(tune_fityx - tune0yx), '--', color='C1', label=r'$\partial \nu_y / \partial J_x : k_{{yx}} = {:.1f} \; mm^{{-1}} $'.format(kyx/1e3))
    ax2.set(xlabel=r'$J_x \, [\mu m]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim(ylim)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.grid()
    ax2.legend()

    plt.savefig('model-tuneshifts.' + fnametype)
    plt.show()
    

if __name__ == "__main__":
    # calc_kxx(plot_flag=True)
    # calc_kxy(plot_flag=True)
    # calc_kyy(plot_flag=True)
    # calc_kyx(plot_flag=True)
    call_all()

