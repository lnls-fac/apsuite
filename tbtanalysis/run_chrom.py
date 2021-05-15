#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.optimize as _opt
from mathphys.functions import save_pickle as _save_pickle
from mathphys.functions import load_pickle as _load_pickle

from lib import create_tbt, calc_param_stats


def calc_traj_chrom(params, *args):
    """BPM averaging due to longitudinal dynamics decoherence.

    nu ~ nu0 + chrom * delta_energy
    See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
    """
    nrturns, data, bpmidx = args

    tunes_frac = params[0]
    tune_frac = params[1]
    chrom_decoh = params[2]
    r0 = params[3]
    mu = params[4]

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
    tunes, tune, chrom_decoh = params[:3]
    nrbpms = data.shape[1]
    r0 = params[3+0*nrbpms:3+1*nrbpms]
    mu = params[3+1*nrbpms:3+2*nrbpms]
    fit = _np.zeros((nrturns, nrbpms))
    res = _np.zeros((nrturns, nrbpms))
    for bpmidx in range(nrbpms):
        bpm_args = (nrturns, data, bpmidx)
        bpm_params = (tunes, tune, chrom_decoh, r0[bpmidx], mu[bpmidx])
        fit[:, bpmidx], *_ = calc_traj_chrom(bpm_params, *bpm_args)
    mea = data[:nrturns,:]
    res = fit - mea
    return res, mea, fit


def calc_chrom_residue_vector(params, *args):
    # print('h')
    # print(params[0])
    res, _, _ = calc_chrom_residue(params, *args)
    # res = res[:,0]
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

    twiss, bpmind = tbt.calc_model_twiss()
    if tbt.select_kicktype in (tbt.ATYPE_CHROMX, tbt.ATYPE_KXX):
        chrom, chrom_err = tbt.chromx, tbt.chromx_err
        beta_model = twiss.betax[bpmind] * 1e6
    else:
        chrom, chrom_err = tbt.chromy, tbt.chromy_err
        beta_model = twiss.betay[bpmind] * 1e6

    # espread from params
    espread = tunes * chrom_decoh / chrom / 2
    er1, er2, er3 = tunes_err/tunes, chrom_decoh_err/chrom_decoh, chrom_err/chrom
    espread_err = espread*_np.sqrt(er1**2 + er2**2 + er3**2)
    # scaled betax
    betax = rx0**2
    rel_err = rx0_err / rx0
    betax_err = 2*betax*rel_err*_np.sqrt(1 + 0.5*rel_err**2)
    scaling = _np.sum(beta_model * betax) / _np.sum(betax * betax)
    betax *= scaling
    betax_err *= scaling
    # action J
    Jx = rx0**2/betax
    Jx_err = Jx * _np.sqrt((rx0_err/rx0)**2 + (betax_err/betax)**2)

    # params and errors
    params = dict()
    if 'tunex' in tbt.data:
        params['tunex_mea'] = tbt.data['tunex']
        params['tuney_mea'] = tbt.data['tuney']
        params['tunex_mea_err'] = tbt.NOM_TUNE_ERR
        params['tuney_mea_err'] = tbt.NOM_TUNE_ERR

    params['chrom'] = chrom
    params['chrom_err'] = chrom_err
    params['kick'] = tbt.data_kicks[tbt.select_idx_kick]
    params['tunes'] = tunes
    params['tune'] = tunex
    params['chrom_decoh'] = chrom_decoh
    params['espread'] = espread
    params['beta'] = betax
    params['J'] = Jx
    params['r0']  = rx0
    params['mu'] = mux
    params['tunes_err'] = tunes_err
    params['tune_err'] = tunex_err
    params['chrom_decoh_err'] = chrom_decoh_err
    params['espread_err'] = espread_err
    params['beta_err'] = betax_err
    params['J_err'] = Jx_err
    params['r0_err']  = rx0_err
    params['mu_err'] = mux_err
    return params


def init_chrom_params(tbt, kickidx):
    
    # do bpm-by-bpm analysis and use average values for global parameters
    _, residues, params, params_err = \
        tbt.analysis_run_chrom(0, None, unwrap=False)
    # _plt.plot(residues); _plt.title('residue x BPM'); _plt.show()

    rx0, mux, tunex, tunes, espread, chromx_decoh = params
    # _plt.plot(tunex); _plt.title('tuneX x BPM'); _plt.show()
    # _plt.plot(tunes); _plt.title('tuneS x BPM'); _plt.show()
    # _plt.plot(espread); _plt.title('espread x BPM'); _plt.show()

    rx0_err, mux_err, *_ = params_err
    _, _, tunes_mean, tunes_std = calc_param_stats(tunes, 3.0)
    _, _, tunex_mean, tunex_std = calc_param_stats(tunex, 3.0)
    _, _, chromx_decoh_mean, chromx_decoh_std = calc_param_stats(chromx_decoh, 3.0)
    _ = espread

    # leastsqr of fit_chrom moves tunes, which gives invalid nrturns based on it
    # using initial value of nrturns from firt BPM
    # tbt.select_idx_turn_stop = tbt.data_nr_turns
    # tbt.search_tunes(peak_frac = 0.999, plot_flag=False)
    nrturns = tbt.select_idx_turn_stop
    # nrturns = int(1/tunes_mean)
    
    # print(espread)
    # print(chromx_decoh)
    # print(tbt.chromx)

    params_global = [tunes_mean, tunex_mean, chromx_decoh_mean]
    params_local = list(rx0) + list(mux)
    params_ini = params_global + params_local
    params_ini_err = [tunes_std, tunex_std, chromx_decoh_std] + list(rx0_err) + list(mux_err)

    if tbt.select_kicktype in (tbt.ATYPE_CHROMX, tbt.ATYPE_KXX):
        args = (nrturns, tbt.data_trajx[kickidx])
    else:
        args = (nrturns, tbt.data_trajy[kickidx])

    # res, mea, fit = calc_chrom_residue(params_ini, *args)
    # res_bpm = _np.sqrt(_np.sum(res**2, axis=0)/res.shape[1])
    # _plt.plot(res_bpm); _plt.xlabel('bpm index'); _plt.ylabel('BPM residue rms [um]'); _plt.show()
    # bpms = range(res.shape[1]); bpms = [0, 1, 100]
    # for bpm in bpms:
    #     _plt.plot(mea[:, bpm])
    #     _plt.plot(fit[:, bpm])
    #     # _plt.plot(res[:, bpm]); 
    #     _plt.title('BPM {:03d}'.format(bpm)); _plt.show()

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
    print('tune    : {:.8f} ± {:.8f}'.format(parms['tune'], parms['tune_err']))
    print('espread : {:.4f} ± {:.4f} %'.format(100*parms['espread'], 100*parms['espread_err']))
    res, vec0 = calc_chrom_residue_norm(params, *args)
    print('residue : {:2f} um'.format(res))
    return parms


def plot_beta(tbt, beta, beta_err, save_flag=True, plot_flag=True):

    if not plot_flag and not save_flag:
        return

    twiss, bpmind = tbt.calc_model_twiss()
    if tbt.select_kicktype in (tbt.ATYPE_CHROMX, tbt.ATYPE_KXX):
        beta_model = twiss.betax
        label = 'betax [m]'
    else:
        beta_model = twiss.betay
        label = 'betay [m]'
    scaling = 1e-6 # um -> m

    _plt.plot(twiss.spos, beta_model, '-', color='C0')
    _plt.plot(twiss.spos[bpmind], beta_model[bpmind], 'o', color='C0', label='model')
    _plt.errorbar(twiss.spos[bpmind], scaling * beta, scaling * beta_err, fmt='o', color='C1', label='fit')
    _plt.xlabel('pos [m]')
    _plt.ylabel(label)
    _plt.legend()
    if save_flag:
        fname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle','-beta.svg')
        _plt.savefig(fname)
        if not plot_flag:
            _plt.clf()
    if plot_flag:
        _plt.show()

    betabeat = 100*(beta*scaling - beta_model[bpmind])/beta_model[bpmind]
    betabeat_err = 100*beta_err*scaling/beta_model[bpmind]
    _plt.errorbar(_np.arange(tbt.data_nr_bpms), betabeat, betabeat_err, fmt='o')
    _plt.xlabel('BPM index')
    _plt.ylabel('betabeat [%]')
    _plt.grid()
    if save_flag:
        fname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle','-betabeat.svg')
        _plt.savefig(fname)
        if not plot_flag:
            _plt.clf()
    if plot_flag:
        _plt.show()


def plot_fit_residue(tbt, params_args, save_flag=True, plot_flag=True):
    if not plot_flag and not save_flag:
        return

    params, args = params_args
    res, mea, _ = calc_chrom_residue(params, *args)
    nturn = _np.arange(mea.size) / tbt.data_nr_bpms
    mea_vec = _np.reshape(mea, mea.size)
    mea_rms = _np.sqrt(_np.sum(mea_vec*mea_vec)/mea_vec.size)
    res_vec = _np.reshape(res, res.size)
    res_rms = _np.sqrt(_np.sum(res_vec*res_vec)/res_vec.size)
    _plt.plot(nturn, mea_vec, label='TbT data (rms: {:.1f} um)'.format(mea_rms))
    _plt.plot(nturn, res_vec, label='Fit residue (rms: {:.1f} um)'.format(res_rms))
    _plt.xlabel('turn')
    _plt.ylabel('posx [um]')
    _plt.legend()
    if save_flag:
        fname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle','-residue.svg')
        _plt.savefig(fname)
        if not plot_flag:
            _plt.clf()
    if plot_flag:
        _plt.show()


def plot_fit_residue_bpm(tbt, bpm, parms, params_args):

    params, args = params_args

    res, mea, fit = calc_chrom_residue(params, *args)
    nrturns = tbt.select_idx_turn_stop
    residue = _np.sqrt(_np.sum(res[:nrturns, bpm]**2)/nrturns)
    J = _np.mean(_np.array(parms['J']))
    figname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle','-residue-BPM{:03d}.svg'.format(bpm))
    _plt.plot(mea[:nrturns, bpm], label='Data for BPM index {:03d}'.format(bpm))
    _plt.plot(fit[:nrturns, bpm], label=r'Fit, ($J_x = {:.03f} \; \mu m.rad$)'.format(J))
    _plt.plot(res[:nrturns, bpm], label=r'Residue ({:.1f} $\mu m$ rms)'.format(residue))
    _plt.xlabel('Turn number')
    if tbt.select_plane_x:
        _plt.title('Fitting of TbT Data - Horizontal Kick')
        _plt.ylabel(r'X [$\mu m$]')
    else:
        _plt.title('Fitting of TbT Data - Vertical Kick')
        _plt.ylabel(r'Y [$\mu m$]')
    _plt.legend()
    _plt.grid()
    _plt.savefig(figname)
    _plt.savefig(figname.replace('.svg', '.png'))
    _plt.show()


def analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=True, print_flag=True, plot_flag=True):

    # create tbt object and do separate bpm fitting
    if print_flag:
        print('data: ', folder+fname)
    tbt = create_tbt(folder+fname, kicktype)
    

    fname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle', '-metadata.txt')
    with open(fname, 'w') as fp:
        for k,v in tbt.data.items():
            if k not in ('trajx', 'trajy', 'trajsum'):
                fp.write('{:<30s}: {}\n'.format(k, v))
    return

    # trajx = tbt.data['trajx'][0]
    # trajy = tbt.data['trajy'][0]
    # # _plt.plot(trajy[:,0], label='Y')
    # _plt.plot(trajx[:,0], label='X')
    # _plt.legend()
    # _plt.show()
    # return

    tbt.select_idx_kick = kickidx
    args, params_ini, params_ini_err = init_chrom_params(tbt, 0)
    # print(params_ini[:3])
    # parms = conv_chrom_fit_data(tbt, params_ini, params_ini_err)
    # plot_fit_residue_bpm(tbt, 0, parms, (params_ini, args))
    
    # print initial fitting results
    if print_flag:
        print('--- global parameters (averaged over fitted BPMs) ---')
        print_parameters(tbt, params_ini, params_ini_err, *args)
    

    # do least square fit with all BPM data
    params_fit, params_fit_err = fit_chrom_params_leastsqr(tbt, params_ini, *args)
    # print(params_fit[:3])

    # print final fitting results
    if print_flag:
        print('--- global parameters (fitted with all BPMs) ---')
        parms = print_parameters(tbt, params_fit, params_fit_err, *args)
    else:
        parms = conv_chrom_fit_data(tbt, params_fit, params_fit_err)

    # plot residue for BPM
    parms = conv_chrom_fit_data(tbt, params_fit, params_fit_err)
    plot_fit_residue_bpm(tbt, 0, parms, (params_fit, args))

    # plot local parameters
    plot_fit_residue(tbt, (params_fit, args), save_flag, plot_flag)
    plot_beta(tbt, parms['beta'], parms['beta_err'], save_flag, plot_flag)

    if print_flag:
        print()

    if save_flag:
        fname = 'results/' + tbt.data_fname.replace('/','-').replace('.pickle','-results.pickle')
        _save_pickle(parms, fname, True)

    return parms




if __name__ == "__main__":
    
    save_flag = True
    print_flag = True
    plot_flag = False

    # --- multibunch horizontal - after cycling ---

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m150urad_after_cycle.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m200urad_after_cycle.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m250urad_after_cycle.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- multibunch small amplitude kicks - before cycling ---
    
    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m005urad_chrom=2p5.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m010urad_chrom=2p5.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m025urad_chrom=2p5.pickle'
    parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- multibunch vertical - after cycling ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_100volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_150volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_200volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_250volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_300volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_350volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_400volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_450volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- single-bunch horizontal ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m025urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m050urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m100urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m150urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m200urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m250urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)


    # --- single-bunch vertical ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # !!! halted! fname = 'tbt_data_vertical_050volts_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)
    