#!/usr/bin/env python-sirius

"""."""

import apsuite.optics_analysis
import matplotlib.pyplot as plt
import numpy as np
import numpy as _np
import pyaccel
import scipy.signal as _scysig
from apsuite.tbt_analysis import TbTAnalysis
from mathphys.functions import load_pickle
from siriuspy.search import BPMSearch


def convert_data(fname):
    data = load_pickle(fname)
    if 'trajx' in data:
        if isinstance(data['chromx_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromx_err'])
        if isinstance(data['chromy_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromy_err'])
        data['kicktype'] = 'CHROMX' if data['kicktype'] == 'X' else data['kicktype']
        data['kicktype'] = 'CHROMY' if data['kicktype'] == 'Y' else data['kicktype']
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


def plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, factor):
    
    idx, filtered, filtered_out, param_mean, param_std = calc_param_stats_print(param, cutoff, ylabel)
    if param_err is not None:
        param_err_fin = param_err[filtered].ravel() * factor
        param_err_fout = param_err[filtered_out].ravel() * factor
    else:
        param_err_fin = None
        param_err_fout = None
    plt.errorbar(idx[filtered_out], param[filtered_out].ravel() * factor, param_err_fout, fmt='x', color='red', label='outliers')
    plt.errorbar(idx[filtered], param[filtered].ravel() * factor, param_err_fin, fmt='o', color='C0', label='data')
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


def plot_fitted_params_chrom(tbt, bpm_indices=None, cutoff=3):

    # --- tbt fitting ---
    bpm_indices, residue, params, params_err = \
        tbt.analysis_run_chrom(0, bpm_indices)
    rx0, mux, tunex_frac, tunes_frac, espread = params
    rx0_err, mux_err, tunex_frac_err, tunes_frac_err, espread_err = params_err

    # --- plot residue ---
    maxtraj = np.max(np.abs(tbt.data_trajx[tbt.select_idx_kick]), axis=0)
    param, param_err = 100*residue/maxtraj, None
    ylabel = r'$R / X_{max}$ [%]'
    rtitle = ('Fitted residue: '
        r'({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 1)

    # --- plot espread ---
    param, param_err = espread, espread_err
    
    ylabel = 'energy spread [%]'
    rtitle = ('Fitted energy spread: '
        r'$\sigma_{{\delta}}$ = ({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 100)

    # --- plot tunex_frac ---
    param, param_err = tunex_frac, tunex_frac_err
    ylabel = 'tunex_frac [%]'
    rtitle = ('Fitted TuneX: '
        r'$\nu_{{x}}$ = ({:.4f} $\pm$ {:.4f}) [%]'
        '\n{:.1f} x sigma cutoff')
    tunex_frac_avg, _ = plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 100)

    # --- plot tunes_frac ---
    param, param_err = tunes_frac, tunes_frac_err
    ylabel = 'tunes_frac [%]'
    rtitle = ('Fitted TuneS: '
        r'$\nu_{{s}}$ = ({:.4f} $\pm$ {:.4f}) [%]'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 100)

    # model
    tunex = 49 + tunex_frac_avg
    twiss, bpms_idx = tbt.calc_model_twiss(goal_tunes=[tunex, None])

    # --- plot betax ---
    betax = rx0**2 + rx0_err**2
    betax_err = rx0_err*np.sqrt(2*(rx0**2+rx0_err*2))
    scaling = np.sum(betax * twiss.betax[bpms_idx]) / np.sum(betax * betax)
    plt.plot(twiss.spos[bpms_idx], twiss.betax[bpms_idx], 'o-', label='model')
    # plt.plot(betax * scaling, 'o-', label='tbt (scale matched)')
    # plt.plot(list(np.arange(bpms_idx), betax_err * scaling, 'o-', label='tbt (scale matched)')
    plt.errorbar(twiss.spos[bpms_idx], betax * scaling, betax_err * scaling, fmt='o', label='tbt (fitted scale)')
    plt.xlabel('pos [m]')
    plt.ylabel('betax [m]')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX')
    plt.show()

    # --- plot betax beating ---
    beta_beat = 100 * (betax * scaling - twiss.betax[bpms_idx]) / twiss.betax[bpms_idx]
    beta_beat_err = 100 * scaling * betax_err / twiss.betax[bpms_idx]
    # beta_beat_err = 100 * betax_err * scaling
    calc_param_stats_print(beta_beat, 2.0, 'betax_beat')
    plt.errorbar(twiss.spos[bpms_idx], beta_beat, beta_beat_err, fmt='o', label='w.r.t. nominal')
    plt.xlabel('pos [m]')
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
    plt.plot(twiss.spos[bpms_idx], muxt / 2 / np.pi, 'o-', label='model')
    plt.errorbar(twiss.spos[bpms_idx], mux2 / 2 / np.pi, mux_err / 2 / np.pi, fmt='o', label='tbt (shift matched)')
    plt.legend()
    plt.xlabel('pos [m]')
    plt.ylabel('mux / 2 / pi')
    plt.title('TbT BetaX Phase')
    plt.show()

    # --- plot mux error ---
    mux_error = (mux2 - muxt) / 2 / np.pi
    calc_param_stats_print(mux_error, 2.0, 'mux_error')
    plt.errorbar(twiss.spos[bpms_idx], mux_error, mux_err / 2 / np.pi, fmt='o', label='w.r.t. nominal')
    plt.xlabel('bpm index')
    plt.ylabel('mux error / (2*pi)')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX Phase Error')
    plt.show()

    return tbt


def plot_fitted_params_tuneshift(tbt, bpm_indices=None, cutoff=3):

    # --- tbt fitting ---
    
    bpm_indices, residue, params, params_err = \
        tbt.analysis_run_tuneshift(0, bpm_indices)

    rx0, mux, dtunex_frac, tunes_frac, espread, kxx_decoh, sigmax = params
    rx0_err, mux_err, dtunex_frac_err, tunes_frac_err, espread_err, kxx_decoh_err, sigmax_err = params_err

    # --- plot residue ---
    maxtraj = np.max(np.abs(tbt.data_trajx[tbt.select_idx_kick]), axis=0)
    param, param_err = 100*residue/maxtraj, None
    ylabel = r'$R / X_{max}$ [%]'
    rtitle = ('Fitted residue: '
        r'({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 1)

    # --- plot kxx_decoh ---
    param, param_err = kxx_decoh, kxx_decoh_err
    ylabel = 'kxx_decoh'
    rtitle = ('Fitted kxx_decoh: '
        r'$k_{{xx}}$ = ({:.6f} $\pm$ {:.6f})'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 1)
    
    # --- plot espread ---
    param, param_err = espread, espread_err    
    ylabel = 'energy spread [%]'
    rtitle = ('Fitted energy spread: '
        r'$\sigma_{{\delta}}$ = ({:.3f} $\pm$ {:.3f}) %'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 100)

    # --- plot tunes_frac ---
    param, param_err = tunes_frac, tunes_frac_err
    ylabel = 'tunes_frac [%]'
    rtitle = ('Fitted TuneS: '
        r'$\nu_{{s}}$ = ({:.4f} $\pm$ {:.4f}) [%]'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 100)

    # --- plot dtunex_frac ---
    param, param_err = dtunex_frac, dtunex_frac_err
    ylabel = 'dtunes_frac'
    rtitle = ('Fitted dtuneX: '
        r'$\delta\nu_{{x}}$ = ({:.6f} $\pm$ {:.6f})'
        '\n{:.1f} x sigma cutoff')
    plot_fitted_global_param(param, param_err, cutoff, ylabel, rtitle, 1)
    
    # --- sigmax ---
    twiss, bpms_idx = tbt.calc_model_twiss()
    k = tbt.NOM_COUPLING
    emit0 = tbt.NOM_EMITTANCE
    emitx = emit0 * 1 / (1 + k)
    betax = 1e6 * twiss.betax[bpms_idx[bpm_indices]]
    etax = 1e6 * twiss.etax[bpms_idx[bpm_indices]]
    sigmax_nom = _np.sqrt(emitx * betax + (etax * tbt.NOM_ESPREAD)**2)
    plt.plot(twiss.spos[bpms_idx[bpm_indices]], sigmax_nom, 'o-', label='model')
    plt.errorbar(twiss.spos[bpms_idx[bpm_indices]], sigmax, sigmax_err, fmt='o', label='fitted')
    plt.xlabel('pos [m]')
    plt.ylabel('sigmax [um]')
    plt.grid()
    plt.legend()
    plt.title('TbT SigmaX')
    plt.show()
    return

    # --- plot betax beating ---
    beta_beat = 100 * (betax * scaling - twiss.betax[bpms_idx]) / twiss.betax[bpms_idx]
    beta_beat_err = 100 * scaling * betax_err / twiss.betax[bpms_idx]
    # beta_beat_err = 100 * betax_err * scaling
    calc_param_stats_print(beta_beat, 2.0, 'betax_beat')
    plt.errorbar(twiss.spos[bpms_idx], beta_beat, beta_beat_err, fmt='o', label='w.r.t. nominal')
    plt.xlabel('pos [m]')
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
    plt.plot(twiss.spos[bpms_idx], muxt / 2 / np.pi, 'o-', label='model')
    plt.errorbar(twiss.spos[bpms_idx], mux2 / 2 / np.pi, mux_err / 2 / np.pi, fmt='o', label='tbt (shift matched)')
    plt.legend()
    plt.xlabel('pos [m]')
    plt.ylabel('mux / 2 / pi')
    plt.title('TbT BetaX Phase')
    plt.show()

    # --- plot mux error ---
    mux_error = (mux2 - muxt) / 2 / np.pi
    calc_param_stats_print(mux_error, 2.0, 'mux_error')
    plt.errorbar(twiss.spos[bpms_idx], mux_error, mux_err / 2 / np.pi, fmt='o', label='w.r.t. nominal')
    plt.xlabel('bpm index')
    plt.ylabel('mux error / (2*pi)')
    plt.grid()
    plt.legend()
    plt.title('TbT BetaX Phase Error')
    plt.show()

    return tbt


def calc_sigmax(tbt, bpm_index):
    """."""
    sigmas = {
        # sigmax, betax
        # 0: (66.92446418692471, 17915535.626820464)
    }
    if bpm_index in sigmas:
        sigmax, betax = sigmas[bpm_index]
    else:
        twiss, bpms_idx = tbt.calc_model_twiss()
        emitx = 0.25e-9 * 1e6  # [um.rad] nominal emittance
        betax = 1e6 * twiss.betax[bpms_idx[bpm_index]]
        etax = 1e6 * twiss.etax[bpms_idx[bpm_index]]
        sigmax = np.sqrt(emitx * betax + (etax * tbt.espread)**2)
        print('bpm_index: {} -> sigmax: {}, betax: {}'.format(bpm_index, sigmax, betax))
    return sigmax, betax


def study_nonlinear():

    # get TbT data and fit chrom decoherence parameters
    tbt = create_tbt()
    tbt.select_idx_kick = 0
    tbt.select_idx_bpm = 0
    tbt.fit_run_chrom()

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

    sigmax = calc_sigmax(tbt, tbt.select_idx_bpm)
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


def test_kxx(tbt):

    # fit chrom decoh
    tbt.select_kicktype = tbt.ATYPE_CHROMX
    tbt.select_idx_kick = 0
    tbt.select_idx_bpm = 1
    tbt.fit_run_chrom()
    tbt.fit_run_tuneshift()

    fm, f2 = tbt.fit_trajs()
    fr2 = f2 - fm
    r2 = tbt.fit_residue()
    plt.plot(fm, label='mea')
    plt.plot(fr2, label='fit2 - mea ({} um)'.format(r2))
    plt.legend()
    plt.show()


def multibunch_kick_spread():
    """."""
    rev = 1.7e-6
    kickx_width = 2 * rev
    kicky_width = 3 * rev
    bunch_half_duration = 50e-9 / 2
    percentx = 100*(np.cos((np.pi/2)*(bunch_half_duration)/kickx_width) - 1)
    percenty = 100*(np.cos((np.pi/2)*(bunch_half_duration)/kicky_width) - 1)
    print('kickx reduction: {} %'.format(percentx))
    print('kicky reduction: {} %'.format(percenty))


folder = '2021-03-23-SI_commissioning-dynap_meas/'
fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
# fname = 'tbt_data_horizontal_m050urad_single_bunch.pickle'
# fname = 'tbt_data_horizontal_m200urad_single_bunch.pickle'
tbt = create_tbt()
plot_fitted_params_tuneshift(tbt)
# plot_fitted_params_chrom(tbt)
# test_kxx(tbt)
# print(tbt)

# tbt.select_idx_kick = 0
# tbt.select_kicktype = TbTAnalysis.ATYPE_CHROMX
# tbt.select_idx_bpm = 100
# tbt.fit_run_chrom()
# print(tbt)
# tbt.select_idx_turn_stop = tbt.data_nr_turns
# tbt.select_kicktype = TbTAnalysis.ATYPE_KXX
# tbt.init_twiss_from_model()
# tbt.init_k_decoh()
# tbt.fit_run_tuneshift()
# tm, tf = tbt.fit_trajs()
# print(tbt)
# plt.plot(tf, label='fit')
# plt.plot(tm, label='mea')
# plt.legend()
# plt.show()

# multibunch_kick_spread()
# raise

# folder = '2021-03-23-SI_commissioning-dynap_meas/'
# fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
# # # study_nonlinear()
# test_kxx2()


# tbt.select_kicktype = TbTAnalysis.ATYPE_KXX
# tbt.fit_run_tuneshift()


# plot_fitted_params_tuneshift(tbt)
# tm, tf = tbt.fit_trajs()
# plt.plot(tm, label='mea')
# plt.plot(tf, label='fit')
# plt.legend()
# plt.show()
# print(tbt)


# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# fname = 'tbt_data_horizontal_m050urad.pickle'
# plot_fitted_params_chrom()

# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# fname = 'tbt_data_horizontal_m005urad_chrom=2p5.pickle'
# # plot_fitted_params_chrom()
# test_kxx2()

# folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
# fname = 'tbt_data_horizontal_m050urad_chrom=1p25'
# plot_fitted_params_chrom()

# folder = '2021-04-26-SI_commissioning_pinger_vertical/'
# fname = 'dynap_data_pingerv_150volts_injsys_off.pickle'
# tbt = create_tbt(kicktype='Y')
# plot_fitted_params_chrom()

# nrpts = 6*300+1
# signal = tbt.data_trajy[0][:nrpts, 0]
# signal = signal - np.mean(signal)
# freqs, fourier = pyaccel.naff.naff_general(signal=signal, is_real=True, nr_ff=10, window=1)
# print(freqs)
# print(abs(freqs[0] - freqs[1]))






