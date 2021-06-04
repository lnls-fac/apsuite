#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt

from mathphys.functions import save_pickle as _save_pickle

from pymodels import si
import pyaccel
from pyaccel.naff import naff_general as _naff_general



def kicker_effective_emit():

    ex = 0.25e-3
    ey = 0.03*0.25e-3
    J = _np.linspace(0, 0.5, 1000)
    kickspreadx = 0.02/100
    kickspready = 0.03/100
    
    Jx = _np.array([
        4.36075486e-04, 7.91695818e-04, 3.38817660e-03, 1.41073453e-02, 6.10738276e-02,
        1.46164620e-01, 2.71883147e-01, 4.36631526e-01])
    Jy = _np.array([0.00117827, 0.00499604, 0.0122923,  0.02344385, 0.03874838, 0.05795771, 0.08192801, 0.11014405, 0.1428096,  0.18038581, 0.22284173, 0.26999985])
    exeffM = _np.sqrt(ex**2+(kickspreadx*Jx)**2)
    eyeffM = _np.sqrt(ey**2+(kickspready*Jy)**2)
    
    print(100*kickspreadx*J[-1]/ex)
    print(100*kickspready*J[-1]/ey)

    exeff = _np.sqrt(ex**2+(kickspreadx*J)**2)
    eyeff = _np.sqrt(ey**2+(kickspready*J)**2)
    _plt.loglog(J, exeff/ex, '--', color=(0.4,0.4,1), alpha=1.0)
    _plt.loglog(Jx, exeffM/ex, 'o', color=(0.4,0.4,1.0), alpha=1.0, label='Applied Horiz. Kicks')
    _plt.loglog(J, eyeff/ey, '--', color=(1,0.4,0.4), alpha=1.0)
    _plt.loglog(Jy, eyeffM/ey, 'o', color=(1,0.4,0.4), alpha=1.0, label='Applied Verti. Kicks')
    # _plt.set_yscale('log')
    _plt.title('Effective Emittances from Kick-spreaded\nMultibunch Beam', fontsize=16)
    _plt.grid(which='both')
    _plt.xlim([1e-4,1])
    _plt.ylim([0.1,100])
    _plt.xlabel(r'$J_{x/y} \, [ \, \mu m^{-1} \, ]$', fontsize=14)
    _plt.ylabel(r'$\epsilon_{eff} \, / \, \epsilon_0$', fontsize=14)
    _plt.legend()
    _plt.show()


def multibunch_kick_spread():
    """."""
    # rev = 1.7e-6
    # kickx_width = 2 * rev
    # kicky_width = 3 * rev
    kickx_width = 2.19e-6
    kicky_width = 1.63e-6
    
    bunch_half_duration = 50e-9 / 2
    time = bunch_half_duration

    print('kickx/bunch_duration: {}'.format(kickx_width/(2*bunch_half_duration)))
    percentx = 100*(_np.cos((_np.pi/2)*time/(kickx_width/2)) - 1)
    print('optim kickx max.diff.: {} %'.format(percentx))
    ax = _np.pi*bunch_half_duration*2/2/kickx_width
    spreadx = 100*_np.sqrt((0.5*(1+_np.sin(2*ax)/(2*ax))-(_np.sin(ax)/ax)**2))
    print('optim kickx dist.spread: {} %'.format(spreadx))
    percentx = 100*(_np.sin((_np.pi/2)*time/(kickx_width/2)))
    print('worst kickx max.diff: {} %'.format(percentx))
    
    print()
    
    print('kicky/bunch_duration: {}'.format(kicky_width/(2*bunch_half_duration)))
    percenty = 100*(_np.cos((_np.pi/2)*time/(kicky_width/2)) - 1)
    print('optim kicky max.diff.: {} %'.format(percenty))
    ay = _np.pi*bunch_half_duration*2/2/kicky_width
    spready = 100*_np.sqrt((0.5*(1+_np.sin(2*ay)/(2*ay))-(_np.sin(ay)/ay)**2))
    print('optim kicky dist.spread: {} %'.format(spready))
    percenty = 100*(_np.sin((_np.pi/2)*time/(kicky_width/2)))
    print('worst kicky max.diff: {} %'.format(percenty))


def track_tune(tr, posi, posf, tune0, _npts=3, nrturns=6*333):
    tiny = _np.array([1e-6, 0, 1e-6, 0, 0, 0])
    pos_vec = [posi + f*(posf - posi) + tiny for f in _np.linspace(0,1.0, _npts)]
    tune = [tune0]
    print(tune[-1])
    for pos in pos_vec[1:]:
        traj, *_ = pyaccel.tracking.ring_pass(tr, pos, nrturns, True)
        x = traj[0,:] - _np.mean(traj[0,:])
        y = traj[2,:] - _np.mean(traj[2,:])
        ffqs1, _ = _naff_general(signal=x, is_real=True, nr_ff=3, window=1)
        ffqs2, _ = _naff_general(signal=y, is_real=True, nr_ff=3, window=1)
        ffqs1 = abs(ffqs1)
        ffqs2 = abs(ffqs2)
        idx1 = (_np.abs(ffqs1 - tune[-1][0])).argmin()
        idx2 = (_np.abs(ffqs2 - tune[-1][0])).argmin()
        tune.append([ffqs1[idx1], ffqs2[idx2]])
        print(tune[-1])
    return _np.array(tune)


def plot_model_dtunex(u_max, plane, nrpts=5, plot_flag=True):

    plane = plane.lower()
    tr = si.create_accelerator()
    tr.cavity_on = False
    tr.radiation_on = False
    twiss, *_ = pyaccel.optics.calc_twiss(tr)
    # print(twiss.mux[-1]/2/_np.pi, twiss.muy[-1]/2/_np.pi)
    
    if plane == 'xx':
        beta = twiss.betax[0] # [m]
        u_init = 0 # [m]
        tune0_nom = twiss.mux[-1]/2/_np.pi - 49
        xlabel, ylabel = r'$J_x$ [um]', r'$\delta\nu_x \; x \; 10^4$'
        label = r'fit of tracking: $k_{{xx}} = {:.1f} \; mm^{{-1}}$'
    elif plane == 'yy':
        beta = twiss.betay[0] # [m]
        u_init = 50e-6 # [m]
        tune0_nom = twiss.muy[-1]/2/_np.pi - 14
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
    u0 = _np.linspace(u_init, u_max, nrpts) + tiny
    tune = _np.zeros(len(u0))

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
        idx = (_np.abs(ffqs - tune0_nom)).argmin()
        tune[i] = ffqs[idx]
        tune0_nom = tune[i]
        print('point: {:03d}, pos: {:.3f} mm, tune: {:7f}'.format(i, u0[i]*1e3, tune[i]))

    # --- (tune shift com amplitude linear com ação)
    # tune ~ tune0 + k u² 
    # tune ~ tune0 + k (beta * J)

    J = u0**2/beta  # [m]

    p = _np.polyfit(J, tune, min(5,len(J)-1))
    J_fit = _np.linspace(min(J), max(J), 40)
    tune_fit = _np.polyval(p, J_fit)

    tune0 = p[-1]
    k_norm = p[-2]  # 1/m

    if plot_flag:
        
        print('tune0  : {:.6f}'.format(tune0))
        print('coefJ² : {:.5e} 1/m²'.format(p[0]))
        print('k_norm : {:0f} 1/m'.format(k_norm))
        print('k_norm : {:2f} 1/mm'.format(k_norm/1e3))
        print('k_norm : {:4f} 1/um'.format(k_norm/1e6))
        
        _plt.plot(J_fit*1e6, 1e4*(tune_fit - tune0), '--', color='C0', label=label.format(k_norm/1e3))
        _plt.plot(J*1e6, 1e4*(tune - tune0), 'o', color='C0')
        _plt.xlabel(xlabel)
        _plt.ylabel(ylabel)
        _plt.legend()
        _plt.title('Model Tune-Shift with Amplitude')
        _plt.grid()
        _plt.show()

    return k_norm, _np.array(tune), _np.array(J), tune0, _np.array(J_fit), _np.array(tune_fit), u0


def calc_tuneshift_JxJydE(x0, y0):

    tr = si.create_accelerator()
    tr.cavity_on = False
    tr.radiation_on = False
    twiss, *_ = pyaccel.optics.calc_twiss(tr)
    twiss, *_ = pyaccel.optics.calc_twiss(tr)
    tunex_n = twiss.mux[-1]/2/_np.pi % 1
    tuney_n = twiss.muy[-1]/2/_np.pi % 1

    _npts = 5
    de = _np.linspace(-0.1/100,+0.1/100, _npts)

    # move to initial scan point
    posi = _np.array([0, 0, 0, 0, 0, 0])
    posf = _np.array([x0, 0, y0, 0, de[0], 0])
    tune0 = _np.array([tunex_n, tuney_n])
    tunes = track_tune(tr, posi, posf, tune0, 3)
    print()

    # scan energy
    posi = posf
    posf = _np.array([x0, 0, y0, 0, de[-1], 0])
    tunes = track_tune(tr, posi, posf, tunes[-1], _npts)

    tunex, tuney = tunes.T
    px = _np.polyfit(de, tunex, 2)
    py = _np.polyfit(de, tuney, 2)
    
    print('Jx: {} um.rad'.format(1e6*x0**2/twiss.betax[0]))
    print('Jy: {} um.rad'.format(1e6*y0**2/twiss.betay[0]))
    print('chromx: ', px[1])
    print('chromy: ', py[1])
    

def calc_kxx(u_max=3000e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'xx', nrpts, plot_flag)


def calc_kxy(u_max=1200e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'xy', nrpts, plot_flag)


def calc_kyy(u_max=1200e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'yy', nrpts, plot_flag)


def calc_kyx(u_max=3000e-6, nrpts=10, plot_flag=False):
    return plot_model_dtunex(u_max, 'yx', nrpts, plot_flag)


def calc_tuneshift_JxJy(fnametype='svg'):

    kxx, tunexx, Jxx, tune0xx, J_fitxx, tune_fitxx, u0 = calc_kxx(plot_flag=False)
    kxy, tunexy, Jxy, tune0xy, J_fitxy, tune_fitxy, u0 = calc_kxy(plot_flag=False)
    kyx, tuneyx, Jyx, tune0yx, J_fityx, tune_fityx, u0 = calc_kyx(plot_flag=False)
    kyy, tuneyy, Jyy, tune0yy, J_fityy, tune_fityy, u0 = calc_kyy(plot_flag=False)

    ylim = (-80, 160)

    graph, (ax1, ax2) = _plt.subplots(1, 2)
    graph.subplots_adjust(wspace=0)
    graph.suptitle(
        'Tune-Shifts with Amplitudes\n' +
        r'Nominal Model $\xi_{{x,y}} \approx 2.5$')

    ax1.plot(J_fitxy*1e6, +1e4*(tune_fitxy - tune0xy), '--', color='C0', label=r'$\partial \nu_x / \partial J_y : k_{{xy}} = {:.1f} \; mm^{{-1}} $'.format(kxy/1e3))
    ax1.plot(J_fityy*1e6, +1e4*(tune_fityy - tune0yy), '--', color='C1', label=r'$\partial \nu_y / \partial J_y : k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy/1e3))
    ax1.set(xlabel=r'$J_y \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax1.set_xlim([0.5, 0])
    ax1.set_ylim(ylim)
    ax1.grid()
    ax1.legend()
    
    ax2.plot(J_fitxx*1e6, +1e4*(tune_fitxx - tune0xx), '--', color='C0', label=r'$\partial \nu_x / \partial J_x : k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx/1e3))
    ax2.plot(J_fityx*1e6, +1e4*(tune_fityx - tune0yx), '--', color='C1', label=r'$\partial \nu_y / \partial J_x : k_{{yx}} = {:.1f} \; mm^{{-1}} $'.format(kyx/1e3))
    ax2.set(xlabel=r'$J_x \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim(ylim)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.grid()
    ax2.legend()

    _plt.savefig('results/model/model-tuneshifts.' + fnametype)
    _plt.show()

    data = dict()
      
    data['kxx'] = kxx
    data['tunexx'] = tunexx
    data['Jxx'] = Jxx
    data['tune0xx'] = tune0xx
    data['J_fitxx'] = J_fitxx
    data['tune_fitxx'] = tune_fitxx
    data['kxy'] = kxy
    data['tunexy'] = tunexy
    data['Jxy'] = Jxx
    data['tune0xy'] = tune0xy
    data['J_fitxy'] = J_fitxy
    data['tune_fitxy'] = tune_fitxy
    data['kyx'] = kyx
    data['tuneyx'] = tuneyx
    data['Jyx'] = Jxx
    data['tune0yx'] = tune0yx
    data['J_fityx'] = J_fityx
    data['tune_fityx'] = tune_fityx
    data['kyy'] = kyy
    data['tuneyy'] = tuneyy
    data['Jyy'] = Jxx
    data['tune0yy'] = tune0yy
    data['J_fityy'] = J_fityy
    data['tune_fityy'] = tune_fityy
    _save_pickle(data, 'results/model/model-tuneshifts.pickle', True)


if __name__ == "__main__":
    # calc_kxx(plot_flag=True)
    # calc_kxy(plot_flag=True)
    # calc_kyy(plot_flag=True)
    # calc_kyx(plot_flag=True)
    # calc_tuneshift_JxJy()
    # calc_tuneshift_JxJydE(0.000, 0.000); calc_chrom(0.002, 0.000); calc_chrom(0.000, 0.001)
    # multibunch_kick_spread()
    kicker_effective_emit()
