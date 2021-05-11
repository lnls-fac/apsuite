#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as plt
import scipy.optimize as _opt


from lib import create_tbt, calc_param_stats


def calc_traj_chrom(params, *args):
        """BPM averaging due to longitudinal dynamics decoherence.

        nu ~ nu0 + chrom * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        tunes_frac = params[0]
        tune_frac = params[1]
        chrom_decoh = params[2]
        r0 = params[3]
        mu = params[4]

        nrturns, data, bpmidx = args
        offset = _np.mean(data[:nrturns, bpmidx])
        n = _np.arange(0, nrturns)
        cn = _np.cos(2 * _np.pi * tune_frac * n)
        sn = _np.sin(2 * _np.pi * tune_frac * n)
        cos = cn * _np.cos(mu) - sn * _np.sin(mu)
        alp = chrom_decoh * _np.sin(_np.pi * tunes_frac * n)
        exp = _np.exp(-alp**2/2.0)
        traj = r0 * exp * cos + offset
        return traj, cn, sn, alp, exp, cos


def calc_chrom_residue(params, *args):
    """."""

    nrturns, data = args
    tunes, tune, chromx_decoh = params[:3]
    nrbpms = data.shape[1]
    r0 = params[3+0*nrbpms:3+1*nrbpms]
    mu = params[3+1*nrbpms:3+2*nrbpms]
    # tunes, tune, chromx_decoh, r0, mu = params
    fit = _np.zeros((nrturns, nrbpms))
    res = _np.zeros((nrturns, nrbpms))
    for bpmidx in range(nrbpms):
        bpm_args = (nrturns, data, bpmidx)
        bpm_params = (tunes, tune, chromx_decoh, r0[bpmidx], mu[bpmidx])
        fit[:, bpmidx], *_ = calc_traj_chrom(bpm_params, *bpm_args)
    mea = data[:nrturns,:]
    res = fit - mea
    return res, mea, fit


def calc_chrom_residue_vector(params, *args):
    # print('h')
    # print(params[0])
    res, _, _ = calc_chrom_residue(params, *args)
    return _np.reshape(res, res.size)  # all bpms, every turn, in this order


def calc_chrom_residue_norm(params, *args):
    vec = calc_chrom_residue_vector(params, *args)
    res = _np.sqrt(_np.sum(vec**2)/len(vec))
    return res, vec


def conv_chrom_fit_data(tbt, params_fit, params_fit_err):
    tunes, tunex, chrom_decoh = params_fit[:3]
    nrbpms = tbt.data_nr_bpms
    rx0 = _np.array(params_fit[3+0*nrbpms:3+1*nrbpms])
    mux = _np.array(params_fit[3+1*nrbpms:3+2*nrbpms])
    tunes_err, tunex_err, chrom_decoh_err = params_fit_err[:3]
    rx0_err = _np.array(params_fit_err[3+0*nrbpms:3+1*nrbpms])
    mux_err = _np.array(params_fit_err[3+1*nrbpms:3+2*nrbpms])
    # espread from params
    espread = tunes * chrom_decoh / tbt.chromx / 2
    er1, er2, er3 = tunes_err/tunes, chrom_decoh_err/chrom_decoh, tbt.chromx_err/tbt.chromx
    espread_err = espread*_np.sqrt(er1**2 + er2**2 + er3**2)
    # unscaled betax
    betax = rx0**2
    rel_err = rx0_err / rx0
    betax_err = 2*betax*rel_err*_np.sqrt(1 + 0.5*rel_err**2)
    # params and errors
    params = dict()
    params['tunes'] = tunes
    params['tunex'] = tunex
    params['espread'] = espread
    params['betax'] = betax
    params['rx0']  = rx0
    params['mux'] = mux
    params['tunes_err'] = tunes_err
    params['tunex_err'] = tunex_err
    params['espread_err'] = espread_err
    params['betax_err'] = betax_err
    params['rx0_err']  = rx0_err
    params['mux_err'] = mux_err
    return params


def fit_chrom_params_init(tbt, kickidx):
    
    # do bpm-by-bpm analysis and use average values for global parameters
    _, _, params, params_err = \
            tbt.analysis_run_chrom(0, None)
    rx0, mux, tunex, tunes, espread, chromx_decoh = params
    rx0_err, mux_err, *_ = params_err
    _, _, tunes_mean, tunes_std = calc_param_stats(tunes, 3.0)
    _, _, tunex_mean, tunex_std = calc_param_stats(tunex, 3.0)
    _, _, chromx_decoh_mean, chromx_decoh_std = calc_param_stats(chromx_decoh, 3.0)
    _ = espread
    nrturns = int(1/tunes_mean)
    params_global = [tunes_mean, tunex_mean, chromx_decoh_mean]
    params_local = list(rx0) + list(mux)
    params_ini = params_global + params_local
    params_ini_err = [tunes_std, tunex_std, chromx_decoh_std] + list(rx0_err) + list(mux_err)

    args = (nrturns, tbt.data_trajx[kickidx])

    return args, params_ini, params_ini_err


def fit_chrom_params_leastsqr(tbt, params, *args):
    fit_data = _opt.least_squares(
        fun=calc_chrom_residue_vector,
        x0=params,
        args=args,
        method='lm')
    params_fit = fit_data['x']
    params_fit_err = tbt.calc_leastsqr_fitting_error(fit_data)
    return params_fit, params_fit_err


def print_parameters(tbt, params, params_err, *args):
    parms = conv_chrom_fit_data(tbt, params, params_err)
    print('tunes   : {:.8f} ± {:.8f}'.format(parms['tunes'], parms['tunes_err']))
    print('tunex   : {:.8f} ± {:.8f}'.format(parms['tunex'], parms['tunex_err']))
    print('espread : {:.4f} ± {:.4f} %'.format(100*parms['espread'], 100*parms['espread_err']))
    res, vec0 = calc_chrom_residue_norm(params, *args)
    print('residue : {:2f} um'.format(res))
    return parms


def plot_beta(tbt, beta, beta_err):

    twiss, bpmind = tbt.calc_model_twiss()
    beta_model = twiss.betax[bpmind]
    scaling = _np.sum(beta_model * beta) / _np.sum(beta * beta)

    plt.plot(twiss.spos, twiss.betax, '-', color='C0')
    plt.plot(twiss.spos[bpmind], beta_model, 'o', color='C0', label='model')
    plt.errorbar(twiss.spos[bpmind], scaling * beta, scaling * beta_err, fmt='o', color='C1', label='fit')
    plt.xlabel('pos [m]')
    plt.ylabel('betax [m]')
    plt.legend()
    plt.show()

    betabeat = 100*(beta*scaling - beta_model)/beta_model
    betabeat_err = 100*beta_err*scaling/beta_model
    plt.errorbar(_np.arange(tbt.data_nr_bpms), betabeat, betabeat_err, fmt='o')
    plt.xlabel('BPM index')
    plt.ylabel('betabeat [%]')
    plt.grid()
    plt.show()


def plot_fit_residual(tbt, params, *args):
    res, mea, _ = calc_chrom_residue(params, *args)
    nturn = _np.arange(mea.size) / tbt.data_nr_bpms
    mea_vec = _np.reshape(mea, mea.size)
    mea_rms = _np.sqrt(_np.sum(mea_vec*mea_vec)/mea_vec.size)
    res_vec = _np.reshape(res, res.size)
    res_rms = _np.sqrt(_np.sum(res_vec*res_vec)/res_vec.size)
    plt.plot(nturn, mea_vec, label='TbT data (rms: {:.1f} um)'.format(mea_rms))
    plt.plot(nturn, res_vec, label='Fit residue (rms: {:.1f} um)'.format(res_rms))
    plt.xlabel('turn')
    plt.ylabel('posx [um]')
    plt.legend()
    plt.show()


def analysis_chrom(folder, fname, kickidx=0, print_flag=True, plot_flag=True):

    # create tbt object and do separate bpm fitting
    tbt = create_tbt(folder+fname)
    tbt.select_idx_kick = kickidx
    args, params_ini, params_ini_err = fit_chrom_params_init(tbt, 0)

    # print initial fitting results
    if print_flag:
        print('--- global parameters averaged over BPMs ---')
        print_parameters(tbt, params_ini, params_ini_err, *args)
    
    # do least square fit with all BPM data
    params_fit, params_fit_err = fit_chrom_params_leastsqr(tbt, params_ini, *args)

    # print final fitting results
    if print_flag:
        print('--- global parameters fitted with all BPMs ---')
        parms = print_parameters(tbt, params_fit, params_fit_err, *args)
    else:
        parms = conv_chrom_fit_data(tbt, params_fit, params_fit_err)

    # plot local parameters
    if plot_flag:
        plot_fit_residual(tbt, params_fit, *args)
        plot_beta(tbt, parms['betax'], parms['betax_err'])

    return parms


if __name__ == "__main__":
    
    # folder = '2021-03-23-SI_commissioning-dynap_meas/'
    # fname = 'dynap_data_kick_m050urad_chromcorr_coupiter3_loco_corr.pickle'
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m050urad_chrom=1p25'
    analysis_chrom(folder, fname, kickidx=0)
