#!/usr/bin/env python-sirius

"""."""

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import load_pickle

from siriuspy.search import BPMSearch
import pyaccel
import pymodels
import apsuite.optics_analysis
from apsuite.tbt_analysis import TbTAnalysis
import scipy.signal as _scysig
import numpy as _np

def convert_data(fname):
    data = load_pickle(fname)
    if 'trajx' in data:
        return data
    ndata = dict()
    ndata['trajx'] = data['sofb_tbt']['x'].reshape((1, -1, 160))
    ndata['trajy'] = data['sofb_tbt']['y'].reshape((1, -1, 160))
    ndata['trajsum'] = data['sofb_tbt']['sum'].reshape((1, -1, 160))
    ndata['kicks'] = [data['kick']]
    ndata['tunex'] = data['tune']['x']
    ndata['tuney'] = data['tune']['y']
    return ndata

bpms = BPMSearch.get_names({'sec':'SI', 'dev':'BPM'})


def get_model_twiss(goal_tunes=None):

    # model
    si = pymodels.si.create_accelerator()

    if goal_tunes is not None:
        print('correcting model tunes...')
        tc = apsuite.optics_analysis.TuneCorr(si, 'SI')
        print('init tunes: ', tc.get_tunes())
        print('goal tunes: ', goal_tunes)
        tc.correct_parameters(goal_tunes)
        print('corr tunes: ', tc.get_tunes())
        print()

    twiss, *_ = pyaccel.optics.calc_twiss(si)
    fam_data = pymodels.si.get_family_data(si)
    bpms_idx = [v[0] for v in fam_data['BPM']['index']]

    return twiss, bpms_idx


def create_tbt(kicktype=None):
    newdata = convert_data(folder+fname)
    tbt = TbTAnalysis(data=newdata, kicktype=kicktype)
    print('Meas. ChromX: {:+.4f} +/- {:.4f}'.format(tbt.chromx, 0))
    print('Meas. ChromY: {:+.4f} +/- {:.4f}'.format(tbt.chromy, 0))
    print()
    return tbt


def calc_param_stats(param, cutoff):

    param = np.array(param)
    stdval = np.std(param)
    meanval = np.median(param)
    filtered = (abs(param - meanval) <= cutoff*stdval)
    filtered_out = (abs(param - meanval) > cutoff*stdval)
    param_mean = np.mean(param[filtered])
    param_std = np.std(param[filtered])

    return filtered, filtered_out, param_mean, param_std


def calc_param_stats_print(param, cutoff, ylabel):
    filtered, filtered_out, param_mean, param_std = calc_param_stats(param, cutoff)
    print('Outlier BPMs ({}):'.format(ylabel))
    idx = np.arange(param.size)
    for i in idx[filtered_out]:
        print('{:03d}: {}'.format(i, bpms[i]))
    print()
    return idx, filtered, filtered_out, param_mean, param_std


def plot_fitted_global_param(param, cutoff, ylabel, rtitle, factor):
    idx, filtered, filtered_out, param_mean, param_std = calc_param_stats_print(param, cutoff, ylabel)
    plt.plot(idx[filtered_out], param[filtered_out].ravel() * factor, 'rx', label='outliers')
    plt.plot(idx[filtered], param[filtered].ravel() * factor, 'o', color='C0', label='data')
    plt.plot(idx, np.ones(idx.size) * param_mean * factor, '-', color='C1', label='avg')
    plt.plot(idx, np.ones(idx.size) * (param_mean + param_std) * factor, '--', color='C1', label='std')
    plt.plot(idx, np.ones(idx.size) * (param_mean - param_std) * factor, '--', color='C1')
    title = rtitle.format(factor * param_mean, factor * param_std, cutoff)
    plt.title(title)
    plt.xlabel('bpm index')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    return param_mean, param_std


def plot_fitted_params(bpm_indices=None, cutoff=3):

    tbt = create_tbt()

    # --- tbt fitting ---
    bpm_indices, rx0, mux, tunex_frac, tunes_frac, espread, residue = \
        tbt.analysis_run(0, bpm_indices)

    # --- plot residue ---
    maxtraj = np.max(np.abs(tbt.data_trajx[tbt.select_idx_kick]), axis=0)
    param = 100*residue/maxtraj
    ylabel = r'$R / X_{max}$ [%]'
    rtitle = ('Fitted residue: '
        r'({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, cutoff, ylabel, rtitle, 1)

    # --- plot espread ---
    param = espread
    ylabel = 'energy spread [%]'
    rtitle = ('Fitted energy spread: '
        r'$\sigma_{{\delta}}$ = ({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, cutoff, ylabel, rtitle, 100)

    # --- plot tunex_frac ---
    param = tunex_frac
    ylabel = 'tunex_frac [%]'
    rtitle = ('Fitted TuneX: '
        r'$\nu_{{x}}$ = ({:.4f} $\pm$ {:.4f}) [%]'
        '\n{:.1f} x sigma cutoff')
    tunex_frac_avg, _ = plot_fitted_global_param(param, cutoff, ylabel, rtitle, 100)

    # --- plot tunes_frac ---
    param = tunes_frac
    ylabel = 'tunes_frac [%]'
    rtitle = ('Fitted TuneS: '
        r'$\nu_{{s}}$ = ({:.4f} $\pm$ {:.4f}) [%]'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, cutoff, ylabel, rtitle, 100)

    # model
    tunex = 49 + tunex_frac_avg
    # twiss, bpms_idx = get_model_twiss([tunex, 14.15194311])
    twiss, bpms_idx = get_model_twiss([tunex, None])

    # --- plot betax ---
    betax = rx0**2
    scaling = np.sum(betax * twiss.betax[bpms_idx]) / np.sum(betax * betax)
    plt.plot(twiss.betax[bpms_idx], 'o-', label='model')
    plt.plot(betax * scaling, 'o-', label='tbt (scale matched)')
    plt.xlabel('bpm index')
    plt.ylabel('betax [m]')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX')
    plt.show()

    # --- plot betax beating ---
    beta_beat = 100 * (betax * scaling - twiss.betax[bpms_idx]) / twiss.betax[bpms_idx]
    calc_param_stats_print(beta_beat, 2.0, 'betax_beat')
    plt.plot(beta_beat, 'o-', label='w.r.t. nominal')
    plt.xlabel('bpm index')
    plt.ylabel('betax beating [%]')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX Beating')
    plt.show()

    # --- plot mux ---
    shift = (np.sum(twiss.mux[bpms_idx]) - np.sum(mux)) / len(mux)
    mux2 = (mux + shift)
    muxt = twiss.mux[bpms_idx]
    # for i in range(len(mux2)):
    #     # n = int((muxt[i] - mux2[i]) // np.pi)
    #     delta = np.pi * ((muxt[i] - mux2[i]) // np.pi)
    #     if abs(mux2[i] + delta - muxt[i]) < abs(mux2[i] - muxt[i]):
    #         mux2[i:] += delta
    #     elif abs(mux2[i] - delta - muxt[i]) < abs(mux2[i] - muxt[i]):
    #         mux2[i:] -= delta
    plt.plot(muxt / 2 / np.pi, 'o-', label='model')
    plt.plot(mux2 / 2 / np.pi, 'o-', label='tbt (shift matched)')
    plt.legend()
    plt.xlabel('bpm index')
    plt.ylabel('mux / 2 / pi')
    plt.title('TbT BetaX Phase')
    plt.show()

    # --- plot mux error ---
    mux_error = (mux2 - muxt) / 2 / np.pi
    calc_param_stats_print(mux_error, 2.0, 'mux_error')
    plt.plot(mux_error, 'o-', label='w.r.t. nominal')
    plt.xlabel('bpm index')
    plt.ylabel('mux error / (2*pi)')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX Phase Error')
    plt.show()

    return tbt


def calc_sigmax(bpm_index):
    """."""
    sigmas = {0: 67, }  # [um]
    if bpm_index in sigmas:
        sigmax = sigmas[bpm_index]
    else:
        twiss, bpms_idx = get_model_twiss()
        emitx = 0.25e-9  # [m.rad] nominal emittance
        sigmax = 1e6 * np.sqrt(emitx * twiss.betax[bpms_idx[bpm_index]])
        print('bpm_index {} -> sigmax -> {}'.format(bpm_index, sigmax))
    return sigmax


def study_nonlinear():

    idx_kick = 0

    # select bpm
    idx_bpm = 0

    # get TbT data and fit chrom decoherence parameters
    tbt = create_tbt()
    tbt.select_idx_kick = idx_kick
    tbt.select_idx_bpm = idx_bpm
    tbt.fit_run()
    print(tbt)

    trajx_mea, trajx_fit = tbt.fit_trajs()
    # trajx_mea = trajx_mea - np.mean(trajx_mea)
    plt.plot(trajx_mea, label='mea')
    plt.plot(trajx_fit, label='fit')
    plt.legend()
    plt.show()

    return

    sigmax = calc_sigmax(idx_bpm)


    # --- hilbert ---
    data_anal = _np.array(_scysig.hilbert(trajx))
    # calculate DFT:
    data_dft = _np.fft.fft(data_anal)
    freq = _np.fft.fftfreq(data_anal.shape[0])

    center_freq = 0.0688
    sigma_freq = 0.01
    # Apply Gaussian filter to get only the synchrotron frequency
    H = _np.exp(-(freq - center_freq)**2/2/sigma_freq**2)
    H += _np.exp(-(freq + center_freq)**2/2/sigma_freq**2)
    H /= H.max()
    data_dft *= H
    # get the processed data by inverse DFT
    data_anal = _np.fft.ifft(data_dft)

    phase_hil = _np.unwrap(_np.angle(data_anal))
    instant_freq = _np.gradient(phase_hil)/(2*_np.pi)
    amplitude = np.abs(data_anal)


    # plt.plot(trajx, label='trajx')
    # plt.plot(trajy, label='trajy')
    # plt.xlabel('pos [um]')
    # plt.ylabel('bpm index')
    # plt.legend()
    # plt.show()

    rx0 = abs(max(trajx))

    n = np.arange(tbt.data_nr_turns)

    # tune shift of 0.5 in 5 mm
    kxx = 0.125 * 0.5 / (5000)**2  # [1/um**2]

    # tune = tune0 + kxx * x0**2

    a = 2*kxx*sigmax**2
    b = rx0**2/2/sigmax**2

    theta = 2*np.pi*a*n

    f0 = 2*np.pi*tbt.tunex_frac*n
    f1 = 1/(1+theta**2)
    f2 = b*theta*f1
    f3 = np.exp(-f2*theta)
    f4 = 2*np.arctan(theta)

    f2 -= f2[1]*n
    f4 -= f4[1]*n

    phase = f0+f2+f4
    dphase = (phase[1:] - phase[:-1])/2/np.pi

    # plt.plot(rx0*f1, label='f1')
    # plt.plot(rx0*f1*f3, label='f1*f3')
    # plt.plot(f0+f2+f4), label='all')
    plt.plot(-rx0*f1*f3*np.sin(f0+f2+f4), label='fit')
    plt.plot(trajx, label='trajx')

    # plt.plot(np.sin(f0), label='f0')
    # plt.plot(np.sin(f0+f2+f4), label='f0+f2+f4')

    # plt.plot(instant_freq, label='hilbert')
    # plt.plot(dphase, label='theory')

    # plt.plot(phase_hil, label='hilbert')
    # plt.plot(phase, label='theory')

    # plt.plot(trajx, label='trajx')
    # plt.plot(amplitude, label='hilbert')

    # plt.plot(phase - phase_hil, label='theory')
    plt.legend()
    plt.show()


# folder = '2021-03-23-SI_commissioning-dynap_meas/'
# fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
# study_nonlinear()

folder = '2021-03-23-SI_commissioning-dynap_meas/'
fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
tbt = plot_fitted_params()

# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# fname = 'tbt_data_horizontal_m050urad.pickle'
# plot_fitted_params()

# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# fname = 'tbt_data_horizontal_m050urad_chrom=1p25'
# plot_fitted_params()

# folder = '2021-04-26-SI_commissioning_pinger_vertical/'
# fname = 'dynap_data_pingerv_150volts_injsys_off.pickle'
# tbt = create_tbt(kicktype='Y')
# plot_fitted_params()

# nrpts = 6*300+1
# signal = tbt.data_trajy[0][:nrpts, 0]
# signal = signal - np.mean(signal)
# freqs, fourier = pyaccel.naff.naff_general(signal=signal, is_real=True, nr_ff=10, window=1)
# print(freqs)
# print(abs(freqs[0] - freqs[1]))






