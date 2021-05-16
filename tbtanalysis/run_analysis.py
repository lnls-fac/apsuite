#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt

from mathphys.functions import load_pickle as _load_pickle


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
        # folder + 'tbt_data_horizontal_m005urad_chrom=2p5.pickle',
        # folder + 'tbt_data_horizontal_m010urad_chrom=2p5.pickle' ,
        # folder + 'tbt_data_horizontal_m025urad_chrom=2p5.pickle',
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
        folder + 'tbt_data_vertical_700volts_after_cycle.pickle',
        ]
    return read_data(fnames, datatype)


def plot_tuneshift(datax=None, datay=None, figtype='svg'):

    # model
    model = _load_pickle('model-tuneshifts.pickle')
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
    graph.suptitle('TbT Data x Model Tune-Shifts with Amplitudes')

    ax1.plot(J_fitxy*1e6, +1e4*(tune_fitxy - tune0xy), '--', color='C0', label=r'$\partial \nu_x$ model : $k_{{xy}} = {:.1f} \; mm^{{-1}} $'.format(kxy/1e3))
    ax1.plot(J_fityy*1e6, +1e4*(tune_fityy - tune0yy), '--', color='C1', label=r'$\partial \nu_y$ model : $k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy/1e3))
    ax1.plot(Jy, +1e4*dtuney, 'x', color='C1', label=r'$\partial \nu_y$ TbT : $k_{{yy}} = {:.1f} \; mm^{{-1}} $'.format(kyy_data*1e3))
    ax1.set(xlabel=r'$J_y \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax1.set_xlim([0.5, 0])
    ax1.set_ylim(ylim)
    ax1.grid()
    ax1.legend()
    
    ax2.plot(J_fitxx*1e6, +1e4*(tune_fitxx - tune0xx), '--', color='C0', label=r'$\partial \nu_x$ model : $k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx/1e3))
    ax2.plot(Jx, +1e4*dtunex, 'x', color='C0', label=r'$\partial \nu_x$ TbT : $k_{{xx}} = {:.1f} \; mm^{{-1}} $'.format(kxx_data*1e3))
    ax2.plot(J_fityx*1e6, +1e4*(tune_fityx - tune0yx), '--', color='C1', label=r'$\partial \nu_y$ model : $k_{{yx}} = {:.1f} \; mm^{{-1}} $'.format(kyx/1e3))
    ax2.set(xlabel=r'$J_x \, [\mu m . rad]$', ylabel=r'$\partial \nu \; x \; 10^{{4}}$')
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim(ylim)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.grid()
    ax2.legend()

    _plt.savefig('data-tuneshifts.' + figtype)
    _plt.show()    


def plot_betabeta(idxdatax=0, idxdatay=0, figtype='svg'):

    datax = get_data_multibunch_horizontal('optics')
    datay = get_data_multibunch_vertical('optics')

    bpmind = datax[0].beta.bmpind
    spos = datax[0].beta.spos

    datax = datax[idxdatax]
    datay = datay[idxdatay]
    betabeatx_tbt = datax.beta.betabeta
    betabeaty_tbt = datay.beta.betabeta
    betabeatx_tbt_err = datax.beta.betabeta_err
    betabeaty_tbt_err = datay.beta.betabeta_err
    Jx, Jx_err = datax.beta.J, datax.beta.J_err
    Jy, Jy_err = datay.beta.J, datay.beta.J_err

    data = _np.loadtxt('results/model/betabeating_LOCO_2021-05-11.txt')
    betabeatx_loc = 100 * data[bpmind, 0]
    betabeaty_loc = 100 * data[bpmind, 1]
    
    alpha1, alpha2 = 0.4, 0.8

    # betabeatx 

    color = (0.0,0.0,1.0)
    J, J_err = Jx, Jx_err
    betabeat_tbt = betabeatx_tbt
    betabeat_loc = betabeatx_loc
    betabeat_tbt_err = betabeatx_tbt_err
    ylabel_beat = r'$\delta\beta_x/\beta_{{x,nom}} \; $ [%]'
    title = 'TbT x LOCO Horizontal Beta Beating\n' + \
        r'$Jx = {:.3f} \; \mu m.rad$'.format(J)

    tbt_std, tbt_max = _np.std(betabeat_tbt), _np.max(_np.abs(betabeat_tbt))
    loc_std, loc_max = _np.std(betabeat_loc), _np.max(_np.abs(betabeat_loc))
    label_tbt = 'TbT  - std:{:.1f}%, max:{:.1f}%'.format(tbt_std, tbt_max)
    label_loc = 'LOCO - std:{:.1f}%, max:{:.1f}%'.format(loc_max, loc_max)
    _plt.errorbar(spos[bpmind], betabeat_tbt, betabeat_tbt_err, fmt='o', mfc='w', color=color, alpha=alpha1, label=label_tbt)
    _plt.plot(spos[bpmind], betabeat_loc, '.', color=color, alpha=alpha2, label=label_loc)
    _plt.xlabel('Pos [m]')
    _plt.ylabel(ylabel_beat)
    _plt.legend()
    _plt.title(title)
    _plt.grid()
    _plt.savefig('betabeatx.' + figtype)
    _plt.show()    

    # betabeaty

    color = (1.0,0.0,0.0)
    J, J_err = Jy, Jy_err
    betabeat_tbt = betabeaty_tbt
    betabeat_loc = betabeaty_loc
    betabeat_tbt_err = betabeaty_tbt_err
    ylabel_beat = r'$\delta\beta_y/\beta_{{y,nom}} \; $ [%]'
    title = 'TbT x LOCO Vertical Beta Beating\n' + \
        r'$Jy = {:.3f} \; \mu m.rad$'.format(J)

    tbt_std, tbt_max = _np.std(betabeat_tbt), _np.max(_np.abs(betabeat_tbt))
    loc_std, loc_max = _np.std(betabeat_loc), _np.max(_np.abs(betabeat_loc))
    label_tbt = 'TbT  - std:{:.1f}%, max:{:.1f}%'.format(tbt_std, tbt_max)
    label_loc = 'LOCO - std:{:.1f}%, max:{:.1f}%'.format(loc_max, loc_max)
    _plt.errorbar(spos[bpmind], betabeat_tbt, betabeat_tbt_err, fmt='o', mfc='w', color=color, alpha=alpha1, label=label_tbt)
    _plt.plot(spos[bpmind], betabeat_loc, '.', color=color, alpha=alpha2, label=label_loc)
    _plt.xlabel('Pos [m]')
    _plt.ylabel(ylabel_beat)
    _plt.legend()
    _plt.title(title)
    _plt.grid()
    _plt.savefig('betabeaty.' + figtype)
    _plt.show()   


def tuneshift_multibunch(figfmt='svg'):
    datax = get_data_multibunch_horizontal('optics')
    datay = get_data_multibunch_vertical('optics')
    plot_tuneshift(datax, datay)


if __name__ == "__main__":
    
    # tuneshift_multibunch('svg')
    plot_betabeta(idxdatax=2, idxdatay=4, figtype='svg')

