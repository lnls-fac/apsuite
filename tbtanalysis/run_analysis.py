#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt
from datetime import datetime

from mathphys.functions import load_pickle as _load_pickle

from lib import create_newtbt as _create_newtbt


# --- get data files ---

def read_data(fnames, datatype):
    data = list()
    for fname in fnames:
        fname = 'results/' + fname.replace('/','-').replace('.pickle', '-' + datatype + '.pickle')
        data_ = _load_pickle(fname)
        data.append(data_)
    return data


def get_data_multibunch_horizontal(datatype):
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fnames = [
        folder + 'tbt_data_horizontal_m005urad_chrom=2p5.pickle',
        folder + 'tbt_data_horizontal_m010urad_chrom=2p5.pickle',
        folder + 'tbt_data_horizontal_m025urad_chrom=2p5.pickle',
        folder + 'tbt_data_horizontal_m050urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m100urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m150urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m200urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m250urad_after_cycle.pickle',
        ]
    return read_data(fnames, datatype)


def get_data_multibunch_vertical(datatype):
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fnames = [
        folder + 'tbt_data_vertical_100volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_150volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_200volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_250volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_300volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_350volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_400volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_450volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_500volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_550volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_600volts_after_cycle.pickle',
        folder + 'tbt_data_vertical_650volts_after_cycle.pickle',
        # folder + 'tbt_data_vertical_700volts_after_cycle.pickle',
        ]
    return read_data(fnames, datatype)


# --- data analysis ---


def data_fft():


    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fnames = [
        'tbt_data_horizontal_m010urad_chrom=2p5.pickle',
        'tbt_data_horizontal_m025urad_chrom=2p5.pickle',
        'tbt_data_horizontal_m050urad_after_cycle.pickle',
        'tbt_data_horizontal_m100urad_after_cycle.pickle',
        'tbt_data_horizontal_m150urad_after_cycle.pickle',
        'tbt_data_horizontal_m200urad_after_cycle.pickle',
        'tbt_data_horizontal_m200urad_after_cycle.pickle',
        'tbt_data_horizontal_m250urad_after_cycle.pickle',
        ]

    fftvec = []
    for i, fname in enumerate(fnames):
        kicktype = 'CHROMX' if 'orizontal' in fname else 'CHROMY'
        tbt = _create_newtbt(folder + fname, kicktype=kicktype)
        traj = tbt.data_traj[0]
        data = traj[:, 89]
        data = data - _np.mean(data)
        fft = _np.abs(_np.fft.rfft(data))
        idx = _np.argmax(fft)
        fftvec.append(fft[idx-60:idx+60])

    for fft in fftvec:
        _plt.plot(fft)
    _plt.show()
    
    
    # tunex = idx / len(data)


# --- optics fitting analysis ---


def optics_tuneshift(figtype='svg'):

    datax = get_data_multibunch_horizontal('optics')
    datay = get_data_multibunch_vertical('optics')

    # two first horiz kicks taken with different optics
    datax = datax[2:]

    # model
    model = _load_pickle('results/model/model-tuneshifts.pickle')
    kxx = model['kxx'] # [1/m]
    tunexx = model['tunexx']
    Jxx = model['Jxx']
    tune0xx = model['tune0xx']
    J_fitxx = model['J_fitxx']
    tune_fitxx = model['tune_fitxx']
    kxy = model['kxy']
    tunexy = model['tunexy']
    Jxx = model['Jxy']
    tune0xy = model['tune0xy']
    J_fitxy = model['J_fitxy']
    tune_fitxy = model['tune_fitxy']
    kyx = model['kyx']
    tuneyx = model['tuneyx']
    Jxx = model['Jyx']
    tune0yx = model['tune0yx']
    J_fityx = model['J_fityx']
    tune_fityx = model['tune_fityx']
    kyy = model['kyy']
    tuneyy = model['tuneyy']
    Jxx = model['Jyy']
    tune0yy = model['tune0yy']
    J_fityy = model['J_fityy']
    tune_fityy = model['tune_fityy']

    # data
    if datax:
        Jx = _np.array([d.beta.J for d in datax])
        tune = _np.array([d.tune for d in datax])
        tune_err = _np.array([d.tune_err for d in datax])
        p = _np.polyfit(Jx, tune, 1)
        dtunex = tune - p[-1]
        kxx_data = p[-2] # [1/um]
    if datay:
        Jy = _np.array([d.beta.J for d in datay])
        tune = _np.array([d.tune for d in datay])
        tune_err = _np.array([d.tune_err for d in datay])
        p = _np.polyfit(Jy, tune, 1)
        dtuney = tune - p[-1]
        kyy_data = p[-2] # [1/um]


    # plots
    ylim = (-80, 160)

    graph, (ax1, ax2) = _plt.subplots(1, 2)
    graph.subplots_adjust(wspace=0)
    # graph.suptitle('Tune-Shifts with Amplitudes\n' + 'TbT x Model')

    # ax1.plot(J_fitxy*1e6, +1e4*(tune_fitxy - tune0xy), '--', color='C0', label=r'$\partial \nu_x$ model : $k_{{xy}} = {:.1f} \; mm^{{-1}} $'.format(kxy/1e3))
    # ax1.plot(J_fityy*1e6, +1e4*(tune_fityy - tune0yy), '--', color='C1', label=r'$\partial \nu_y$ model : $k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy/1e3))
    # ax1.plot(Jy, +1e4*dtuney, 'x', color='C1', label=r'$\partial \nu_y$ TbT : $k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy_data*1e3))
    # ax1.set(xlabel=r'$J_y \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax1.plot(J_fitxy*1e6, +1e4*(tune_fitxy - tune0xy), '--', color='C0', label=r'$\partial \nu_x$ M : $k_{{xy}} = {:.1f}$'.format(kxy/1e3))
    ax1.plot(J_fityy*1e6, +1e4*(tune_fityy - tune0yy), '--', color='C1', label=r'$\partial \nu_y$ M : $k_{{yy}} = {:.1f}$'.format(kyy/1e3))
    ax1.plot(Jy, +1e4*dtuney, 'x', color='C1', label=r'$\partial \nu_y$ T : $k_{{yy}} = {:.1f}$'.format(kyy_data*1e3))
    ax1.set(xlabel=r'$J_y \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax1.xaxis.label.set_fontsize(16)
    ax1.yaxis.label.set_fontsize(16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax1.set_xlim([0.5, 0])
    ax1.set_ylim(ylim)
    ax1.grid()
    ax1.legend(fontsize=12)
    
    # ax2.plot(J_fitxx*1e6, +1e4*(tune_fitxx - tune0xx), '--', color='C0', label=r'$\partial \nu_x$ model : $k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx/1e3))
    # ax2.plot(Jx, +1e4*dtunex, 'x', color='C0', label=r'$\partial \nu_x$ TbT : $k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx_data*1e3))
    # ax2.plot(J_fityx*1e6, +1e4*(tune_fityx - tune0yx), '--', color='C1', label=r'$\partial \nu_y$ model : $k_{{yx}} = {:.1f} \; mm^{{-1}} $'.format(kyx/1e3))
    # ax2.set(xlabel=r'$J_x \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax2.plot(J_fitxx*1e6, +1e4*(tune_fitxx - tune0xx), '--', color='C0', label=r'$\partial \nu_x$ M : $k_{{xx}} = {:.1f}$'.format(kxx/1e3))
    ax2.plot(Jx, +1e4*dtunex, 'x', color='C0', label=r'$\partial \nu_x$ T : $k_{{xx}} = {:.1f}$'.format(kxx_data*1e3))
    ax2.plot(J_fityx*1e6, +1e4*(tune_fityx - tune0yx), '--', color='C1', label=r'$\partial \nu_y$ M : $k_{{yx}} = {:.1f}$'.format(kyx/1e3))
    ax2.set(xlabel=r'$J_x \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax2.xaxis.label.set_fontsize(16)
    ax2.yaxis.label.set_fontsize(16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim(ylim)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.grid()
    ax2.legend(fontsize=12)

    _plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.95, wspace=0, hspace=0)
    _plt.savefig('results/data-tuneshifts.' + figtype)
    _plt.show()    


def optics_betabeat(idxdatax=0, idxdatay=0, comparison_type='meas', figtype='svg'):

    datax = get_data_multibunch_horizontal('optics')
    datay = get_data_multibunch_vertical('optics')

    bpmind = datax[0].beta.bmpind
    spos = datax[0].beta.spos

    datax = datax[idxdatax]
    datay = datay[idxdatay]
    if hasattr(datax.beta, 'betabeta'):
        betabeatx_tbt = datax.beta.betabeta
        betabeatx_tbt_err = datax.beta.betabeta_err
    else:
        betabeatx_tbt = datax.beta.betabeat
        betabeatx_tbt_err = datax.beta.betabeat_err
    if hasattr(datay.beta, 'betabeta'):
        betabeaty_tbt = datay.beta.betabeta
        betabeaty_tbt_err = datay.beta.betabeta_err
    else:
        betabeaty_tbt = datay.beta.betabeat
        betabeaty_tbt_err = datay.beta.betabeat_err
    Jx, Jx_err = datax.beta.J, datax.beta.J_err
    Jy, Jy_err = datay.beta.J, datay.beta.J_err

    data = _np.loadtxt('results/model/betabeating_LOCO_2021-05-11.txt')
    betabeatx_loc = 100 * data[bpmind, 0]
    betabeaty_loc = 100 * data[bpmind, 1]
    
    data = _np.loadtxt('results/model/betabeating_TuneShift_2021-03-16.txt')
    sposmea, bxn, bxm, byn, bym = data.T
    betabeatx_mea = 100 * (bxm - bxn) / bxn
    betabeaty_mea = 100 * (bym - byn) / byn


    alpha1, alpha2 = 0.8, 0.4


    # betabeaty 
    
    color = (1.0,0.0,0.0)
    J, J_err = Jy, Jy_err
    betabeat_tbt = betabeaty_tbt
    betabeat_tbt_err = betabeaty_tbt_err

    if comparison_type == 'meas':
        # Meas comparison
        spos_loc = sposmea
        betabeat_loc = betabeaty_mea
        tbt_std = _np.std(betabeat_tbt)
        loc_std = _np.std(betabeat_loc)
        # title = 'Vertical Beta Beat: TbT x Meas.\n' + \
        #     r'$Jy = {:.3f} \; \mu m.rad$'.format(J)
        title = r'$Jy = {:.3f} \; \mu m.rad$'.format(J)
        savename = 'results/betabeaty-vs-meas.' + figtype
        label_loc = 'M - {:.1f}% std'.format(loc_std)
    elif comparison_type == 'loco':
        # LOCO comparison
        spos_loc = spos[bpmind]
        betabeat_loc = betabeaty_loc
        tbt_std, tbt_max = _np.std(betabeat_tbt), _np.max(_np.abs(betabeat_tbt))
        loc_std, loc_max = _np.std(betabeat_loc), _np.max(_np.abs(betabeat_loc))
        title = 'Vertical Beta Beat: TbT x LOCO \n' + \
            r'$Jy = {:.3f} \; \mu m.rad$'.format(J)
        savename = 'results/betabeaty-vs-loco.' + figtype
        label_loc = 'LOCO - {:.1f}% std'.format(loc_std)

    graph, ax1 = _plt.subplots(1, 1)

    ylabel_beat = r'$\delta\beta_y/\beta_{{y,nom}} \; $ [%]'    
    label_tbt = 'T - {:.1f}% std'.format(tbt_std)
    _plt.plot(spos_loc, betabeat_loc, 'o', color=color, alpha=alpha2, label=label_loc)
    _plt.errorbar(spos[bpmind], betabeat_tbt, betabeat_tbt_err, fmt='.', color=color, alpha=alpha1, label=label_tbt)
    
    ax1.xaxis.label.set_fontsize(16)
    ax1.yaxis.label.set_fontsize(16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    
    _plt.xlabel('Pos [m]')
    _plt.ylabel(ylabel_beat)
    _plt.legend(fontsize=12)
    _plt.title(title, fontsize=16)
    _plt.grid()
    _plt.savefig(savename)
    _plt.show()   

    # betabeatx 

    color = (0.0,0.0,1.0)
    J, J_err = Jx, Jx_err
    betabeat_tbt = betabeatx_tbt
    betabeat_tbt_err = betabeatx_tbt_err

    if comparison_type == 'meas':
        # Meas comparison
        spos_loc = sposmea
        betabeat_loc = betabeatx_mea
        tbt_std = _np.std(betabeat_tbt)
        loc_std = _np.std(betabeat_loc)
        # title = 'Horizontal Beta Beat: TbT x Meas.\n' + \
        #     r'$Jx = {:.3f} \; \mu m.rad$'.format(J)
        title = r'$Jx = {:.3f} \; \mu m.rad$'.format(J)
        savename = 'results/betabeatx-vs-meas.' + figtype
        label_loc = 'M - {:.1f}% std'.format(loc_std)
    elif comparison_type == 'loco':
        # LOCO comparison
        spos_loc = spos[bpmind]
        betabeat_loc = betabeatx_loc
        tbt_std = _np.std(betabeat_tbt)
        loc_std = _np.std(betabeat_loc)
        title = 'Horizontal Beta Beat: TbT x LOCO \n' + \
            r'$Jx = {:.3f} \; \mu m.rad$'.format(J)
        savename = 'results/betabeatx-vs-loco.' + figtype
        label_loc = 'LOCO - std:{:.1f}%'.format(loc_std)

    graph, ax1 = _plt.subplots(1, 1)

    ylabel_beat = r'$\delta\beta_x/\beta_{{x,nom}} \; $ [%]'    
    label_tbt = 'T - {:.1f}% std'.format(tbt_std)
    _plt.plot(spos_loc, betabeat_loc, 'o', color=color, alpha=alpha2, label=label_loc)
    _plt.errorbar(spos[bpmind], betabeat_tbt, betabeat_tbt_err, fmt='.', color=color, alpha=alpha1, label=label_tbt)

    ax1.xaxis.label.set_fontsize(16)
    ax1.yaxis.label.set_fontsize(16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    _plt.xlabel('Pos [m]')
    _plt.ylabel(ylabel_beat)
    _plt.legend(fontsize=12)
    _plt.title(title, fontsize=16)
    _plt.grid()
    _plt.savefig(savename)
    _plt.show()    


# --- espread fitting analysis ---


def plot_tbt_data(data, ix, bpmidx):

    J = data.optics.beta.J
    espread = 100*data.espread
    
    xcolor = (0,0,1)
    ycolor = (1,0,0)
    print(data.tbt.select_kicktype)
    if data.tbt.select_plane_x:
        title1 = 'TbT Fit of Energy Spread - Horizontal - BPM Index {:03d}\n'.format(bpmidx)
        color = xcolor
        fname = 'tbt-data-horizontal'
    else:
        title1 = 'TbT Fit of Energy Spread - Vertical - BPM Index {:03d}\n'.format(bpmidx)
        color = ycolor
        fname = 'tbt-data-vertical'
    nrturns = data.args[2]
    title = title1 + 'J = {:.3f} '.format(J) + r'$\mu m.rad$'
    label = '$\sigma_\epsilon$ = {:.3f} %'.format(100*data.espread)
    traj1 = data.tbt.data_traj[0][:, bpmidx]
    traj2 = data.tbt.data_traj[0][:nrturns, bpmidx]
    _plt.plot(traj1, color=color, alpha=0.5)
    _plt.plot(traj2, color=color, alpha=1.0, label=label)
    _plt.xlabel('Turn number')
    _plt.ylabel(r'$Pos \; [\,\mu m\,]$')
    _plt.grid()
    _plt.legend()
    _plt.title(title)
    _plt.savefig(fname + '-J{:02d}-bpm{:03d}.svg'.format(ix, bpmidx))
    _plt.show()

    
def plot_espread(fittype, datax, datay, figtype='svg'):
    
    # datax = get_data_multibunch_horizontal('espread1')

    # print(datax[0].__dict__.keys())


    Jx = _np.array([d.optics.beta.J for d in datax])
    tunesx = _np.array([d.tunes for d in datax])
    tunesx_err = _np.array([d.tunes_err for d in datax])
    espreadx = 100*_np.array([d.espread for d in datax])
    espreadx_err = 100*_np.array([d.espread_err for d in datax])
    chrom_decohx = _np.array([d.chrom_decoh for d in datax])
    chrom_decohx_err = _np.array([d.chrom_decoh_err for d in datax])

    Jy = _np.array([d.optics.beta.J for d in datay])
    tunesy = _np.array([d.tunes for d in datay])
    tunesy_err = _np.array([d.tunes_err for d in datay])
    espready = 100*_np.array([d.espread for d in datay])
    espready_err = 100*_np.array([d.espread_err for d in datay])
    chrom_decohy = _np.array([d.chrom_decoh for d in datay])
    chrom_decohy_err = _np.array([d.chrom_decoh_err for d in datay])

    # res = _np.array([d.optics.residue for d in datax])
    # amp = _np.array([d.optics.r0 for d in datax])
    # for j,r,a in zip(Jx, res, amp):
    #     print(j, 100*r/_np.sqrt(_np.sum(a**2)/a.size))

    # res = _np.array([d.optics.residue for d in datay])
    # amp = _np.array([d.optics.r0 for d in datay])
    # for j,r,a in zip(Jy, res, amp):
    #     print(j, 100*r/_np.sqrt(_np.sum(a**2)/a.size))
    
    # return

    print('Jx: ', Jx)
    print('Jy: ', Jy)
    print('Jx: ', Jx)
    print('Jy: ', Jy)
    print('EspreadX: ', espreadx)
    print('EspreadY: ', espready)
    print('EspreadX_err: ', espreadx_err)
    print('EspreadY_err: ', espready_err)

    ycolor, yalpha = (1,0.4,0.4), 1.0

    graph, ax1 = _plt.subplots(1, 1)
    graph.subplots_adjust(bottom=0.15, left=0.15, top=0.95)

    # --- espread ---
    J, param, param_err = Jx, espreadx, espreadx_err
    _plt.errorbar(J, param, param_err, fmt='o--', label='Horiz')
    J, param, param_err = Jy, espready, espready_err
    _plt.errorbar(J, param, param_err, fmt='o--', color=ycolor, alpha=yalpha, label='Verti.')
    
    ax1.xaxis.label.set_fontsize(16)
    ax1.yaxis.label.set_fontsize(16)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    
    _plt.xlim([0,0.5])
    _plt.ylim([0.06,0.10])
    # _plt.title('Energy Spread')
    _plt.xlabel(r'J $[\, \mu m.rad \,]$')
    _plt.ylabel(r'$\sigma_\epsilon$ [%]')
    _plt.grid()
    _plt.legend(fontsize=12, loc='lower right')
    _plt.savefig('espread-'+fittype+'.'+figtype)
    _plt.show()

    return

    # --- tunes  ---
    J, param, param_err = Jx, tunesx, tunesx_err
    _plt.errorbar(J, 1000*param, 1000*param_err, fmt='o--', label='Fit Horiz. TbT')
    J, param, param_err = Jy, tunesy, tunesy_err
    _plt.errorbar(J, 1000*param, 1000*param_err, fmt='o--', color=ycolor, alpha=yalpha, label='Fit Verti. TbT')
    _plt.xlim([0,0.5])
    _plt.ylim([2.5,7.0])
    _plt.title('Longitudinal Tune')
    _plt.xlabel(r'J $[\, \mu m.rad\,]$')
    _plt.ylabel(r'$\nu_s\, \times \, 10^3$')
    _plt.grid()
    _plt.legend()
    _plt.savefig('tunes-'+fittype+'.'+figtype)
    _plt.show()
    
    # --- chrom_decoh ---
    J, param, param_err = Jx, chrom_decohx, chrom_decohx_err
    _plt.errorbar(J, param, param_err, fmt='o--', label='Fit Horiz. TbT')
    J, param, param_err = Jy, chrom_decohy, chrom_decohy_err
    _plt.errorbar(J, param, param_err, fmt='o--', color=ycolor, alpha=yalpha, label='Fit Verti. TbT')
    _plt.xlim([0,0.5])
    _plt.ylim([0.4,1.6])
    _plt.title('Chromaticity Decoherence Coefficient')
    _plt.xlabel(r'J $[\, \mu m.rad\,]$')
    _plt.ylabel(r'2 $\xi \, \sigma_\delta \, \nu_s^{-1}$')
    _plt.grid()
    _plt.legend(loc='center')
    _plt.savefig('decoh-'+fittype+'.'+figtype)
    _plt.show()

    


def espread_multibunch(figfmt='png'):
    fittype = 'espread1'
    datax = get_data_multibunch_horizontal(fittype)
    datay = get_data_multibunch_vertical(fittype)
    plot_espread(fittype, datax, datay)


def timestamps():
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'

    fnames = [
        'tbt_data_horizontal_m050urad_after_cycle.pickle',
        'tbt_data_horizontal_m100urad_after_cycle.pickle',
        'tbt_data_horizontal_m150urad_after_cycle.pickle',
        'tbt_data_horizontal_m200urad_after_cycle.pickle',
        'tbt_data_horizontal_m250urad_after_cycle.pickle',
    ]
    for fname in fnames:
        # print(fname)
        tbt = _create_newtbt(folder+fname, print_flag=False)
        data = tbt.data
        tstamp = datetime.fromtimestamp(data['time'])
        curr = data['stored_current']
        cx = data['chromx'] - 864*1.645e-4
        cy = data['chromy']- 864*1.645e-4
        cxe = data['chromx_err']
        cye = data['chromy_err']
        # print('{} I:{:.2f} mA,  ChromX:{:.2f}, ChromY:{:.2f}, ChromXE:{:.2f}, ChromYE:{:.2f}'.format(tstamp, curr, cx, cxe, cy, cye))
        print(data['buffer_count'])
    print()


    fnames = [
        'tbt_data_vertical_100volts_after_cycle.pickle',
        'tbt_data_vertical_150volts_after_cycle.pickle',
        'tbt_data_vertical_200volts_after_cycle.pickle',
        'tbt_data_vertical_250volts_after_cycle.pickle',
        'tbt_data_vertical_300volts_after_cycle.pickle',
        'tbt_data_vertical_350volts_after_cycle.pickle',
        'tbt_data_vertical_400volts_after_cycle.pickle',
        'tbt_data_vertical_450volts_after_cycle.pickle',
        'tbt_data_vertical_500volts_after_cycle.pickle',
        'tbt_data_vertical_550volts_after_cycle.pickle',
        'tbt_data_vertical_600volts_after_cycle.pickle',
        'tbt_data_vertical_650volts_after_cycle.pickle',
        'tbt_data_vertical_700volts_after_cycle.pickle',
    ]
    for fname in fnames:
        tbt = _create_newtbt(folder+fname, print_flag=False)
        data = tbt.data
        tstamp = datetime.fromtimestamp(data['time'])
        curr = data['stored_current']
        cx = data['chromx'] - 864*1.645e-4
        cy = data['chromy']- 864*1.645e-4
        cxe = data['chromx_err']
        cye = data['chromy_err']
        print('{} I:{:.2f} mA,  ChromX:{:.2f}, ChromY:{:.2f}, ChromXE:{:.2f}, ChromYE:{:.2f}'.format(tstamp, curr, cx, cxe, cy, cye))
    print()

    fnames = [
        'tbt_data_horizontal_m025urad_chrom=2p5.pickle',
        'tbt_data_horizontal_m010urad_chrom=2p5.pickle',
        'tbt_data_horizontal_m005urad_chrom=2p5.pickle',
    ]
    for fname in fnames:
        tbt = _create_newtbt(folder+fname, print_flag=False)
        data = tbt.data
        tstamp = datetime.fromtimestamp(data['time'])
        curr = data['stored_current']
        cx = data['chromx'] - 864*1.645e-4
        cy = data['chromy']- 864*1.645e-4
        cxe = data['chromx_err']
        cye = data['chromy_err']
        print('{} I:{:.2f} mA,  ChromX:{:.2f}, ChromY:{:.2f}, ChromXE:{:.2f}, ChromYE:{:.2f}'.format(tstamp, curr, cx, cxe, cy, cye))
    print()

    



# --- main ---

if __name__ == "__main__":
    
    # data_fft()
    # optics_tuneshift('png')
    # optics_betabeat(idxdatax=2+2, idxdatay=4, comparison_type='meas', figtype='png')
    espread_multibunch()
    # timestamps()
