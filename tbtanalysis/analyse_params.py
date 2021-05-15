#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as plt
import scipy.optimize as _opt

from mathphys.functions import load_pickle as _load_pickle

from apsuite.tbt_analysis import TbTAnalysis

from lib import create_tbt, calc_param_stats


def read_data(fnames):
    data = dict()
    for fname in fnames:
        fname = 'results/' + fname.replace('/','-').replace('.pickle', '-results.pickle')
        data_ = _load_pickle(fname)
        for k,v in data_.items():
            if k not in data:
                data[k] = [v]
            else:
                try:
                    data[k].append(v)
                except KeyError:
                    print(fname, ' missing ', k)
    for k,v in data.items():
        data[k] = _np.array(v)
    return data


def get_data_multibunch_horizontal():
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fnames = [
        folder + 'tbt_data_horizontal_m005urad_chrom=2p5.pickle',
        folder + 'tbt_data_horizontal_m010urad_chrom=2p5.pickle' ,
        folder + 'tbt_data_horizontal_m025urad_chrom=2p5.pickle',
        folder + 'tbt_data_horizontal_m050urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m100urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m150urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m200urad_after_cycle.pickle',
        folder + 'tbt_data_horizontal_m250urad_after_cycle.pickle',
    ]
    return read_data(fnames)


def get_data_multibunch_vertical():
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
    ]
    return read_data(fnames)


def get_data_singlebunch_horizontal():
    folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    fnames = [
        folder + 'tbt_data_horizontal_m025urad_single_bunch.pickle',
        folder + 'tbt_data_horizontal_m050urad_single_bunch.pickle',
        folder + 'tbt_data_horizontal_m100urad_single_bunch.pickle',
        folder + 'tbt_data_horizontal_m150urad_single_bunch.pickle',
        folder + 'tbt_data_horizontal_m200urad_single_bunch.pickle',
        folder + 'tbt_data_horizontal_m250urad_single_bunch.pickle',
    ]
    return read_data(fnames)


def plot_tuneshift(plane, data, title, fname):
    # J and dtune
    J = _np.array(data['J'])
    J_avg = _np.mean(J, axis=1)
    tune = data['tune']
    # J and dtune errors
    J_err = _np.array(data['J_err'])
    J_std = _np.std(J, axis=1)
    J_avg_err = _np.sqrt(_np.sum(J_err**2, axis=1))/J_err.shape[1]    
    J_avg_err = _np.array([max(J_std[i], J_avg_err[i]) for i in range(len(J_std))])
    tune_err = data['tune_err']
    # polyfit
    pfit, pcov = _np.polyfit(J_avg, tune, deg=1, cov=True)
    pfit_err = _np.sqrt(_np.diagonal(pcov))
    J_fit = _np.linspace(0, 1.1*_np.max(J_avg), 2)
    tune_fit = _np.polyval(pfit, J_fit)
    # conversion to emittance
    k, tune_incoh = pfit
    # conversion to emittance error
    k_err, tune_incoh_err = pfit_err
    # nominal
    if plane == 'x':
        # k_nom = TbTAnalysis.NOM_KXX_DECOH_NORM
        txt_equ = r'$\nu_{x} = \nu^{incoh}_{x} + k_{xx} \; J_x$  '
        txt_k = r'$k_{xx} = '
        xlabel = r'$J_{{x}} \; [\,\mu m]$'
        ylabel = r'$\left( \nu_{{x}} - \nu^{{incoh}}_{{x}}\right) \; x \; 10^{{4}}$'
        titlepref = 'Horizontal'
    else:
        # k_nom = TbTAnalysis.NOM_KYY_DECOH_NORM
        txt_equ = r'$\nu_{y} = \nu^{incoh}_{y} + k_{yy} \; J_y$  '
        txt_k = r'$k_{yy} = '
        xlabel = r'$J_{{y}} \; [\,\mu m]$'
        ylabel = r'$\left( \nu_{{y}} - \nu^{{incoh}}_{{y}} \right) \; x \; 10^{{4}}$'
        titlepref = 'Vertical'

    txt_fit = 'fitting: ' + \
        txt_k + r'\left(' + '{:.1f}'.format(1e3*k) + r' \pm ' + '{:.1f}'.format(1e3*k_err) + r'\right) \; mm^{-1}$'
    plt.plot(J_fit, 1e4*(tune_fit - tune_incoh), '--', color='C0', label=txt_fit)
    plt.plot(J_avg, 1e4*(tune - tune_incoh), 'X', color='C0', label='TbT Data ({} points)'.format(len(J_avg)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titlepref + ' Tune Shift with Amplitude - ' + title + '\n' + txt_equ)
    plt.xlim([0, max(J_fit)])
    plt.legend()
    plt.grid()
    plt.savefig(fname)
    plt.show()


def plot_espread(plane, data, title, fname):
    
    chrom = _np.array(data['chrom'])
    chrom_err = _np.array(data['chrom_err'])
    chrom_avg = _np.mean(chrom)
    chrom_std = max(_np.std(chrom), max(chrom_err))

    if plane == 'x':    
        titlel = 'Energy Spread from TbT - Horizontal Kick\n' + \
            r'$\xi_x = {:+.2f} \; ± \; {:.2f}$'.format(chrom_avg, chrom_std)
        xlabel = r'$J_x \; [\, \mu m . rad \,]$'
    else:
        titlel = 'Energy Spread from TbT - Horizontal Kick\n' + \
            r'$\xi_y = {:+.2f} \; ± \; {:.2f}$'.format(chrom_avg, chrom_std)
        xlabel = r'$J_y \; [\, \mu m . rad \,]$'
    J = data['J']
    
    # kicks = - _np.array(data['kick'])
    kicks = _np.mean(_np.array(J), 1)/1e3

    espread = data['espread']
    espread_err = data['espread_err']
    espread_avg = _np.mean(espread)
    espread_std = _np.std(espread)
    dkicks = max(kicks) - min(kicks)
    x = _np.linspace(min(kicks) - 0.1*dkicks, max(kicks) + 0.1*dkicks, 2)
    plt.plot(1e3*x, 100*espread_avg*_np.ones(x.shape), '-', color='C0', label='avg: {:.3f} %'.format(100*espread_avg))
    plt.plot(1e3*x, 100*(espread_avg+espread_std)*_np.ones(x.shape), '--', color='C0', label='std: {:.3f} %'.format(100*espread_std))
    plt.plot(1e3*x, 100*(espread_avg-espread_std)*_np.ones(x.shape), '--', color='C0')
    plt.plot(1e3*kicks, 100*espread, 'o', color='C0')
    # plt.errorbar(1e3*kicks, 100*espread, 100*espread_err, fmt='o', color='C0')
    
    # plt.xlabel('kick [urad]')
    plt.xlabel(xlabel)
    plt.ylabel(r'$\sigma_E$ [%]')
    plt.legend()
    plt.grid()
    plt.title(titlel)
    plt.savefig(fname)
    plt.show()


def plot_tunes(plane, data, title, fname):
    if plane == 'x':
        titlel = 'Longitudinal Tune - Horizontal Kick - ' + title
    else:
        titlel = 'Longitudinal Tune - Vertical Kick - ' + title
    kicks = - _np.array(data['kick'])
    tunes = data['tunes']
    tunes_err = data['tunes_err']
    tunes_avg = _np.mean(tunes)
    tunes_std = _np.std(tunes)
    dkicks = max(kicks) - min(kicks)
    x = _np.linspace(min(kicks) - 0.1*dkicks, max(kicks) + 0.1*dkicks, 2)
    plt.plot(1e3*x, 100*tunes_avg*_np.ones(x.shape), '-', color='C0', label='avg: {:.3f} %'.format(100*tunes_avg))
    plt.plot(1e3*x, 100*(tunes_avg+tunes_std)*_np.ones(x.shape), '--', color='C0', label='std: {:.3f} %'.format(100*tunes_std))
    plt.plot(1e3*x, 100*(tunes_avg-tunes_std)*_np.ones(x.shape), '--', color='C0')
    # plt.plot(1e3*x, 100*(tunes_avg+3*tunes_std)*_np.ones(x.shape), ':', color='C0', label='3 std')
    # plt.plot(1e3*x, 100*(tunes_avg-3*tunes_std)*_np.ones(x.shape), ':', color='C0')
    plt.errorbar(1e3*kicks, 100*tunes, 100*tunes_err, fmt='o', color='C0')
    plt.xlabel('kick [urad]')
    plt.ylabel(r'$\nu_s$')
    plt.legend()
    plt.grid()
    plt.title(titlel)
    plt.savefig(fname)
    plt.show()


# ---

def tunes_multibunch_horizontal(figfmt='svg'):
    data = get_data_multibunch_horizontal()
    plot_tunes(
        'x', data, 
        r'Multi-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-03-SI_commissioning-equilibrium_parameters_tbt-horizontal-tunes-multibunch.' + figfmt
        )


def espread_multibunch_horizontal(figfmt='svg'):
    data = get_data_multibunch_horizontal()
    plot_espread(
        'x', data, 
        r'Multi-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-03-SI_commissioning-equilibrium_parameters_tbt-horizontal-espread-multibunch.' + figfmt
        )


def tuneshift_multibunch_horizontal(figfmt='svg'):
    data = get_data_multibunch_horizontal()
    plot_tuneshift(
        'x', data, 
        r'Multi-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-03-SI_commissioning-equilibrium_parameters_tbt-horizontal-tuneshift-multibunch.' + figfmt
        )


def espread_multibunch_vertical(figfmt='svg'):
    data = get_data_multibunch_vertical()
    plot_espread(
        'y', data, 
        r'Multi-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-03-SI_commissioning-equilibrium_parameters_tbt-vertical-espread-multibunch.' + figfmt
        )


def tuneshift_multibunch_vertical(figfmt='svg'):
    data = get_data_multibunch_vertical()
    plot_tuneshift(
        'y', data, 
        r'Multi-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-03-SI_commissioning-equilibrium_parameters_tbt-vertical-tuneshift-multibunch.' + figfmt
        )


def espread_singlebunch_horizontal(figfmt='svg'):
    data = get_data_singlebunch_horizontal()
    plot_espread(
        'x', data, 
        r'Single-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-04-SI_commissioning-equilibrium_parameters_tbt-horizontal-espread-singlebunch.' + figfmt
        )


def tuneshift_singlebunch_horizontal(figfmt='svg'):
    data = get_data_singlebunch_horizontal()
    plot_tuneshift(
        'x', data, 
        r'Single-bunch - $\xi_{x,y} \approx 2.4$',
        'results/2021-05-04-SI_commissioning-equilibrium_parameters_tbt-horizontal-tuneshift-singlebunch.' + figfmt
        )


if __name__ == "__main__":
    
    # tunes_multibunch_horizontal()
    espread_multibunch_horizontal('svg')
    # tuneshift_multibunch_horizontal()
    
    # espread_multibunch_vertical()
    # tuneshift_multibunch_vertical()

    # espread_singlebunch_horizontal()
    # tuneshift_singlebunch_horizontal()
