#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt

from mathphys.functions import save_pickle as _save_pickle
from pyaccel.naff import naff_general as _naff_general

from lib import create_newtbt as _create_newtbt
from lib import calc_stats as _calc_stats
from lib import fit_leastsqr as _fit_leastsqr
from lib import Analysis as _Analysis


def calc_traj(params, *args):
    """."""
    
    data, nrturns, bpmidx = args
    tune, r0, mu = params
    
    offset = _np.mean(data[:nrturns, bpmidx])
    n = _np.arange(0, nrturns)
    cn = _np.cos(2 * _np.pi * tune * n)
    sn = _np.sin(2 * _np.pi * tune * n)
    cos = cn * _np.cos(mu) - sn * _np.sin(mu)
    traj = r0 * cos + offset
    return traj, cn, sn, cos


def calc_residue(params, *args):
    """."""
    bpmidx, data, nrturns = args
    tune, r0, mu = get_params(params, *args)
    # tune = params[0]
    if bpmidx is None:
        # all bpms    
        # r0 = params[1+0*nrbpms:1+1*nrbpms]
        # mu = params[1+1*nrbpms:1+2*nrbpms]
        nrbpms = data.shape[1]
        mea = data[:nrturns,:]
        fit = _np.zeros((nrturns, nrbpms))
        for idx in range(nrbpms):
            bpm_args = (data, nrturns, idx)
            bpm_params = (tune, r0[idx], mu[idx])
            fit[:,idx], *_ = calc_traj(bpm_params, *bpm_args)
    else:
        # r0, mu = params[1:]
        mea = data[:nrturns, bpmidx]
        bpm_args = (data, nrturns, bpmidx)
        bpm_params = (tune, r0, mu)
        fit, *_ = calc_traj(bpm_params, *bpm_args)

    res = fit - mea
    return res, mea, fit


def calc_residue_vector(params, *args):
    res, _, _ = calc_residue(params, *args)
    return _np.reshape(res, res.size)  # all bpms, every turn, in this order


def calc_residue_norm(params, *args):
    vec = calc_residue_vector(params, *args)
    res = _np.sqrt(_np.sum(vec**2)/len(vec))
    return res, vec


def init_params(optics, bpmidx=None):

    tbt = optics.tbt
    r0mu = optics.r0mu
    tune_best, tune = r0mu.tune_avg, r0mu.tune
    r0, mu, nrturns = r0mu.r0, r0mu.mu, r0mu.nrturns

    data = tbt.data_traj[tbt.select_idx_kick]
    args = (bpmidx, data, nrturns)
    if bpmidx is not None:
        params = (tune[bpmidx], r0[bpmidx], mu[bpmidx])
    else:
        params = (tune_best, ) + tuple(r0) + tuple(mu)
    return params, args


def get_params(params, *args):
    bpmidx, data, nrturns = args
    tune = params[0]
    if bpmidx is None:
        nrbpms = data.shape[1]
        r0 = params[1+0*nrbpms:1+1*nrbpms]
        mu = params[1+1*nrbpms:1+2*nrbpms]
    else:
        r0, mu = params[1:]
    return tune, _np.array(r0), _np.array(mu)


def optics_naff(optics, save_flag, print_flag, plot_flag=False):
    
    # NOTE: using NAFF sometimes a) leads to wrong tune.
    # also, b) initial rms is usually worse than FFT
    # and yields c) more outliers. why?!

    use_naff = False

    tbt = optics.tbt
    tune = _np.zeros(tbt.data_nr_bpms)
    tbt.select_idx_turn_stop = tbt.data_nr_turns
    for bpm in range(tbt.data_nr_bpms):
        tbt.select_idx_bpm = bpm
        data = tbt.select_get_traj()
        data -= _np.mean(data)
        if use_naff:
            nrpts_naff = 1 + 6 * ((len(data)-1) // 6)
            ffqs, _ = _naff_general(signal=data[:nrpts_naff], is_real=True, nr_ff=3, window=1)
            tune[bpm] = ffqs[0]
        else:
            _, tune[bpm], _ = tbt.calc_fft(data, peak_frac=0.99)

    tune_avg, tune_std, outliers, insiders = _calc_stats(tune)
    
    if print_flag:
        print('outliers : {:03d} / {:03d}'.format(len(outliers), len(data)))
        print('tune_avg : {:.6f}'.format(tune_avg))
        print('tune_std : {:.6f}'.format(tune_std))

    if plot_flag:
        inds, onev = _np.arange(len(tune)), _np.ones(tune.shape)
        _plt.plot(inds, onev * (tune_avg + 1*tune_std), '-.', color='C0', label='std')
        _plt.plot(inds, onev * (tune_avg + 0*tune_std), '--', color='C0', label='avg')
        _plt.plot(inds, onev * (tune_avg - 1*tune_std), '-.', color='C0')
        _plt.plot(inds[insiders], tune[insiders], 'o', color='C0')
        _plt.plot(inds[outliers], tune[outliers], 'x', color='red')
        _plt.xlabel('BPM index')
        _plt.ylabel(r'$\nu$')
        _plt.title('NAFF Tunes for BPMs')
        _plt.grid()
        _plt.show()

    naff = _Analysis()
    naff.tune = tune
    naff.tune_avg = tune_avg
    naff.tune_std = tune_std
    optics.naff = naff


def optics_search_r0_mu(optics, save_flag, print_flag, plot_flag):

    tbt = optics.tbt
    naff = optics.naff
    
    tune_avg, tune = naff.tune_avg, naff.tune
    r0, mu = _np.zeros(tune.shape), _np.zeros(tune.shape)
    res = _np.zeros(tune.shape)
    tune_best = _np.zeros(tune.shape)
    nrturns = int(3/tune_avg)

    for bpmidx in range(len(tune)):
        # print(bpmidx)
        tbt.select_idx_bpm = bpmidx
        tbt.tune_frac = tune[bpmidx]
        tbt.select_idx_turn_stop = nrturns
        # search best solution varying tune around naff/fft and calculate r0 mu
        tbest, tune_delta = tune[bpmidx], 0.01
        while tune_delta > 1e-5:
            tunes = tbest + _np.linspace(-tune_delta, +tune_delta, 5)
            res_min, i_min = float('Inf'), 0
            for i, ttune in enumerate(tunes):
                tbt.tune_frac = ttune
                tbt.search_r0_mu()
                mea, fit = tbt.fit_trajs()
                tres = _np.sqrt(_np.sum((fit - mea)**2)/len(mea))
                if tres < res_min:
                    res_min, i_min = tres, i
            tbest = tunes[i_min]
            # print(tune_best, res_min)
            tune_delta /= 4
        tbt.search_r0_mu()
        mea, fit = tbt.fit_trajs()
        tres = _np.sqrt(_np.sum((fit - mea)**2)/len(mea))

        # _plt.plot(mea)
        # _plt.plot(fit)
        # _plt.show()
        tune_best[bpmidx] = tbest
        r0[bpmidx] = tbt.r0
        mu[bpmidx] = tbt.mu
        res[bpmidx] = tres
        # if print_flag:
        #     print('{:03d}, tune:{:6f}, nrturns:{:03d}'.format(
        #         bpmidx, tune[bpmidx], nrturns))
    
    tune_avg, tune_std, outliers, insiders = _calc_stats(tune_best)

    r0mu = _Analysis()
    r0mu.r0 = r0
    r0mu.mu = mu
    r0mu.tune = tune_best
    r0mu.tune_avg = tune_avg
    r0mu.tune_std = tune_std
    r0mu.res = res
    r0mu.nrturns = nrturns
    optics.r0mu = r0mu


def optics_beta(optics):
    tbt = optics.tbt
    r0, r0_err = optics.r0, optics.r0_err
    tune = optics.tune
    twiss, bmpind = tbt.calc_model_twiss()
    spos = twiss.spos
    # NOTE: should we correct model tunes to measured tune-shifted tunes?! Not sure...
    if tbt.select_plane_x:
        goal_tunes = [int(twiss.mux[-1]/2/_np.pi) + tune, None]
        twiss, bpmind = tbt.calc_model_twiss(goal_tunes)
        beta_model = twiss.betax * 1e6
    else:
        goal_tunes = [None, int(twiss.muy[-1]/2/_np.pi) + tune]
        twiss, bpmind = tbt.calc_model_twiss(goal_tunes)
        beta_model = twiss.betay * 1e6
    
    # J/beta
    r2beta = _np.sum(r0**2 * beta_model[bpmind])
    r4 = _np.sum(r0**4)
    J = r4 / r2beta
    beta_ = r0**2/J
    # J/beta errors
    r2beta_err = _np.sqrt(_np.sum(((2*r0*beta_model[bpmind]) * r0_err)**2))
    r4_err = _np.sqrt(_np.sum(((4*r0**3) * r0_err)**2))
    J_err = _np.sqrt(((1/r2beta)*r4_err)**2 + ((r4/r2beta**2)*r2beta_err)**2)
    beta_err = _np.sqrt(((2*r0/J)*r0_err)**2 + ((r0**2/J**2)*J_err)**2)

    # beta beating
    betabeat = 100 * (beta_ - beta_model[bpmind])/beta_model[bpmind]
    betabeat_err = betabeat * _np.sqrt(_np.sum((beta_err/beta_)**2))

    beta = _Analysis()
    beta.spos = _np.array([v for v in spos])
    beta.bmpind = bpmind
    beta.beta_model = beta_model
    beta.J = J
    beta.J_err = J_err
    beta.beta = beta_
    beta.beta_err = beta_err
    beta.betabeta = betabeat
    beta.betabeta_err = betabeat_err

    optics.beta = beta

    if plot_flag or save_flag:

        alpha1, alpha2, alpha3 = 0.4, 0.6, 0.8
        if tbt.select_plane_x:
            color = (0.0,0.0,1.0)
            title = r'TbT x Nominal $\beta_x$'
            ylabel = r'$\beta_x \; [m]$'
            ylabel_beat = r'$\delta\beta_x/\beta_{{x,model}} \; $ [%]'
        else:
            title = r'TbT x Nominal $\beta_y$'
            color = (1.0,0.0,0.0)
            ylabel = r'$\beta_y \; [m]$'
            ylabel_beat = r'$\delta\beta_y/\beta_{{y,model}} \;$ [%]'

        # beta - beta_model
        _plt.plot(spos, beta_model/1e6, color=color, alpha=alpha1, label='model')
        _plt.plot(spos[bpmind], beta_model[bpmind]/1e6, 'o', mfc='w', color=color, alpha=alpha2)
        _plt.errorbar(spos[bpmind], beta_/1e6, beta_err/1e6, fmt='.', color=color, alpha=alpha3, label='fit')
        _plt.xlabel('pos [m]')
        _plt.ylabel(ylabel)
        _plt.legend()
        _plt.title(title)
        if save_flag:
            fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-beta.' + save_flag)
            _plt.savefig(fname)
            if not plot_flag:
                _plt.clf()
        if plot_flag:
            _plt.show()

        # betabeat
        _plt.errorbar(_np.arange(len(betabeat)), betabeat, betabeat_err, fmt='o', mfc='w', color=color, alpha=alpha3)
        _plt.xlabel('pos [m]')
        _plt.ylabel(ylabel_beat)
        _plt.title(title)
        _plt.grid()
        if save_flag:
            fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-betabeat.' + save_flag)
            _plt.savefig(fname)
            if not plot_flag:
                _plt.clf()
        if plot_flag:
            _plt.show()


def optics_fit(folder, fname, kicktype, kickidx, save_flag, print_flag, plot_flag):

    optics = _Analysis()

    tbt = _create_newtbt(folder+fname, kicktype)
    optics.tbt = tbt

    # residue from separate BPMs
    optics_naff(optics, save_flag=False, print_flag=False, plot_flag=False)
    optics_search_r0_mu(optics, save_flag=False, print_flag=False, plot_flag=False)
    res, nrturns = optics.r0mu.res, optics.r0mu.nrturns
    res = _np.sqrt(_np.sum(nrturns * res**2)/(nrturns * tbt.data_nr_bpms))
    optics.residue_r0mu = res
    if print_flag:
        print('initial residue individual tune : {:.2f}'.format(res))

    # residue average of best tunes, all BPMs
    params, args = init_params(optics, None)
    tune = optics.r0mu.tune_avg
    res, _ = calc_residue_norm(params, *args)
    optics.params_ini = params
    optics.residue_r0mu_tune_avg = res
    if print_flag:
        print('initial residue average tune    : {:.2f} tune:{:.6f}'.format(res, tune))
    
    # fit all bpms
    params_fit, params_fit_err = _fit_leastsqr(
        tbt, args, optics.params_ini, calc_residue_vector)
    tune, r0, mu = get_params(params, *args)
    tune_err, r0_err, mu_err = get_params(params_fit_err, *args)
    res, vec = calc_residue_norm(params, *args)
    if print_flag:
        print('fit all bpms with the same tune : {:.2f} tune:{:.6f}'.format(res, tune))

    optics.args = args
    optics.params_fit = params_fit
    optics.params_fit_err = params_fit_err
    optics.tune = tune
    optics.r0 = r0
    optics.mu = mu
    optics.tune_err = tune_err
    optics.r0_err = r0_err
    optics.mu_err = mu_err
    optics.residue = res

    # _plt.plot(r0)
    # _plt.plot(r0_err)
    # _plt.show()

    if save_flag or plot_flag:
        res, mea, fit = calc_residue(params, *args)
        _plt.plot(_np.reshape(mea, mea.size, 'F'), label='mea')
        _plt.plot(_np.reshape(fit, fit.size, 'F'), label='fit')
        _plt.plot(_np.reshape(res, res.size, 'F'), label='res')
        _plt.xlabel('Turn number')
        _plt.ylabel('Pos [um]')
        if save_flag:
            fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-residue.' + save_flag)
            _plt.savefig(fname)
            if not plot_flag:
                _plt.clf()
        if plot_flag:
            _plt.show()

    optics_beta(optics)

    # save analysis
    if save_flag:
        fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-optics.pickle')
        optics.tbt = None
        _save_pickle(optics, fname, True)


if __name__ == "__main__":
    
    save_flag = 'svg'
    print_flag = True
    plot_flag = False
    allbpms = True

    # --- multibunch horizontal - after cycling ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m150urad_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m200urad_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m250urad_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- multibunch small amplitude kicks - before cycling ---
    
    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m005urad_chrom=2p5.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m010urad_chrom=2p5.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m025urad_chrom=2p5.pickle'
    # optics_fit(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- multibunch vertical - after cycling ---

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_vertical_100volts_after_cycle.pickle'
    optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_150volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_200volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_250volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_300volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_350volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_400volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_450volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_500volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_550volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_600volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_650volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_700volts_after_cycle.pickle'
    # optics_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)



    # --- single-bunch horizontal ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m025urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m050urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m100urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m150urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m200urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m250urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)
    

    # --- single-bunch vertical ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # !!! halted! fname = 'tbt_data_vertical_050volts_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=save_flag, print_flag, plot_flag=plot_flag)
    



