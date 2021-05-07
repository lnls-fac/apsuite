#!/usr/bin/env python-sirius

"""."""

import apsuite.optics_analysis
import matplotlib.pyplot as plt
import numpy as np
import numpy as _np
import pyaccel
import pymodels
import scipy.signal as _scysig
from apsuite.tbt_analysis import TbTAnalysis
from mathphys.functions import load_pickle
from siriuspy.search import BPMSearch


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
    sigmas = {
        # sigmax, betax
        0: (66.92446418691087, 17915535.626820464)
    }
    if bpm_index in sigmas:
        sigmax, betax = sigmas[bpm_index]
    else:
        twiss, bpms_idx = get_model_twiss()
        emitx = 0.25e-9  # [m.rad] nominal emittance
        betax = 1e6 * twiss.betax[bpms_idx[bpm_index]]
        sigmax = 1e3 * np.sqrt(emitx * betax)
        print('bpm_index: {} -> sigmax: {}, betax: {}'.format(bpm_index, sigmax, betax))
    return sigmax, betax


def study_nonlinear():

    # get TbT data and fit chrom decoherence parameters
    tbt = create_tbt()
    tbt.select_idx_kick = 0
    tbt.select_idx_bpm = 0
    tbt.fit_run()

    # Chrom decoh fitting
    #
    # print(tbt)
    # trajx_mea, trajx_fit = tbt.fit_trajs()
    # plt.plot(trajx_mea, label='mea')
    # plt.plot(trajx_fit, label='fit')
    # plt.legend()
    # plt.show()

    # Sum signal has freq components?
    #
    # data = tbt.data_trajsum[0][:, 0]
    # fft = np.abs(np.fft.rfft(data))
    # plt.plot(fft, 'o')
    # plt.show()


    # take BPM raw data
    trajx_mea = tbt.data_trajx[tbt.select_idx_kick][:, tbt.select_idx_bpm]
    trajx_mea -= np.mean(trajx_mea)
    # plt.plot(trajx_mea)

    # --- hilbert ---
    tbt.select_idx_turn_stop = tbt.data_nr_turns
    params, offset, traj_mea = tbt._get_fit_inputs()
    traj_mea = traj_mea - offset
    args = [tbt.select_idx_turn_start, tbt.select_idx_turn_stop, offset]
    _, _, _, _, exp, *_ = tbt.calc_traj_chrom(params, *args)

    # plt.plot(traj_mea/exp)
    # plt.show()

    # signal = trajx_mea / exp
    # data_anal = _np.array(_scysig.hilbert(signal))
    # # calculate DFT:
    # data_dft = _np.fft.fft(data_anal)
    # freq = _np.fft.fftfreq(data_anal.shape[0])

    # center_freq = tbt.tunex_frac  # 0.0688
    # sigma_freq = 0.01
    # # Apply Gaussian filter to get only the synchrotron frequency
    # H = _np.exp(-(freq - center_freq)**2/2/sigma_freq**2)
    # H += _np.exp(-(freq + center_freq)**2/2/sigma_freq**2)
    # H /= H.max()
    # data_dft *= H
    # # get the processed data by inverse DFT
    # data_anal = _np.fft.ifft(data_dft)

    # phase_hil = _np.unwrap(_np.angle(data_anal))
    # instant_freq = _np.gradient(phase_hil)/(2*_np.pi)
    # amplitude = np.abs(data_anal)

    # plt.plot(amplitude, 'o')
    # plt.show()

    xk = abs(max(trajx_mea))
    phi0 = 0*np.pi/2
    # xk = -tbt.rx0
    # phi = tbt.mux + np.pi/2

    sigmax = calc_sigmax(tbt.select_idx_bpm)
    kxx = 0.125 * 0.5 / (5000)**2  # [1/um**2] - tune shift of 0.5 in 5 mm
    tunex0_frac = tbt.tunex_frac - kxx*(4*sigmax**2+xk**2)

    a = 2*kxx*sigmax**2
    b = xk**2/sigmax**2/2

    n = np.arange(tbt.data_nr_turns)
    theta = 2*np.pi*a*n
    f0 = 2*np.pi*tunex0_frac*n + phi0
    f1 = 1/(1+theta**2)
    f2 = b*theta*f1
    f3 = np.exp(-f2*theta)
    f4 = 2*np.arctan(theta)
    # f2 -= f2[1]*n
    # f4 -= f4[1]*n

    # plt.plot(f0, label='f0')
    # plt.plot(f2, label='f2')
    # plt.plot(f4, label='f4')

    phase = f0+f2+f4
    fa = -xk*f1*f3*np.sin(phase)

    print(tunex0_frac)
    print(sigmax)
    print(kxx)
    print(xk)

    # dphase0 = (f0[1:] - f0[:-1])/2/np.pi
    # dphase = (phase[1:] - phase[:-1])/2/np.pi
    # plt.plot(dphase0, label='dphase0')
    # plt.plot(dphase, label='dphase')

    # plt.plot(rx0*f1, label='f1')
    # plt.plot(rx0*f1*f3, label='f1*f3')
    # plt.plot(f0+f2+f4), label='all')
    plt.plot(trajx_mea, label='trajx')
    plt.plot(fa, label='fit')

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


def test_kxx():

    tbt = create_tbt()
    tbt.select_kicktype = tbt.ATYPE_CHROMX
    tbt.select_idx_kick = 0
    tbt.select_idx_bpm = 0
    tbt.fit_run()

    # sigmax, betax = calc_sigmax(tbt.select_idx_bpm)
    # # print(sigmax, betax)
    # k_decoh_norm = 0.04658039262973321  # dtune/action [1/(rad.um)]
    # k_decoh = k_decoh_norm / betax
    # # print(k_decoh)
    # # k_decoh = 2.6e-9
    # r0 = tbt.rx0
    # # tune0_frac = 0.06819424070644278
    # tune0_frac = tbt.tunex_frac - k_decoh*(4*sigmax**2+r0**2)
    # mu0 = tbt.mux

    # params = [
    #     tbt.tunes_frac*1.04, tune0_frac,
    #     tbt.chromx_decoh, r0, mu0,
    #     sigmax, k_decoh]

    # tbt.select_idx_turn_stop = tbt.data_nr_turns
    # args = [tbt.select_idx_turn_start, tbt.select_idx_turn_stop, tbt.rx_offset]

    # traj_mea = tbt.select_get_traj()
    # traj_fit = tbt.calc_traj_tuneshift(params, *args)
    # plt.plot(traj_mea)
    # plt.plot(traj_fit)
    # plt.show()

    tbt.select_kicktype = tbt.ATYPE_KXX
    tbt.select_idx_turn_stop = tbt.data_nr_turns
    tbt.sigmax, betax = calc_sigmax(tbt.select_idx_bpm)
    k_decoh_norm = 0.04658039262973321  # dtune/action [1/(rad.um)]
    tbt.kxx_decoh = k_decoh_norm / betax
    tbt.tunex0_frac = tbt.tunex_frac - tbt.kxx_decoh*(4*tbt.sigmax**2+tbt.rx0**2)

    fm, f1 = tbt.fit_trajs()
    r1 = tbt.fit_residue()
    # plt.plot(fm)
    # plt.plot(f1)
    # plt.show()

    tbt.fit_leastsqr()
    _, f2 = tbt.fit_trajs()
    r2 = tbt.fit_residue()

    plt.plot(fm, label='mea')
    # plt.plot(f1, label='fit1 ({} um)'.format(r1))
    plt.plot(f2 - fm, label='fit2 ({} um)'.format(r2))
    plt.legend()
    plt.show()

# folder = '2021-03-23-SI_commissioning-dynap_meas/'
# fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
# # study_nonlinear()
# test_kxx()

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






