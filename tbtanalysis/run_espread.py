#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt

from mathphys.functions import load_pickle as _load_pickle
from mathphys.functions import save_pickle as _save_pickle
from pyaccel.naff import naff_general as _naff_general

from lib import create_newtbt as _create_newtbt
from lib import calc_stats as _calc_stats
from lib import fit_leastsqr as _fit_leastsqr
from lib import Analysis as _Analysis


def calc_traj(params, *args):
    """BPM averaging due to longitudinal dynamics decoherence.

    nu ~ nu0 + chrom * delta_energy
    See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
    """
    
    data, nrturns, bpmidx  = args
    tune, tunes, chrom_decoh, r0n2, r0, mu = params
        
    # if bpmidx == 0:
    #     print(params)
        
    offset = _np.mean(data[:nrturns, bpmidx])

    n = _np.arange(0, nrturns)
    cn = _np.cos(2 * _np.pi * tune * n)
    sn = _np.sin(2 * _np.pi * tune * n)
    cos = cn * _np.cos(mu) - sn * _np.sin(mu)
    alp = chrom_decoh * _np.sin(_np.pi * tunes * n)
    exp = _np.exp(-alp**2/2)
    # rn = r0 * (1 - (r0n2 * n)**2)
    rn = r0*_np.exp(-r0n2*n**2)


    traj = rn * exp * cos + offset

    return traj, alp, exp, cn, sn, cos, rn, offset


def calc_residue(params, *args):
    bpmidx, data, nrturns = args
    tune, tunes, chrom_decoh, r0n2, r0, mu = get_params(params, *args)
    if bpmidx is None:
        # all bpms    
        nrbpms = data.shape[1]
        mea = data[:nrturns,:]
        fit = _np.zeros((nrturns, nrbpms))
        for idx in range(nrbpms):
            bpm_args = (data, nrturns, idx)
            bpm_params = (tune, tunes, chrom_decoh, r0n2[idx], r0[idx], mu[idx])
            fit[:,idx], *_ = calc_traj(bpm_params, *bpm_args)
    else:
        raise
        # r0, mu = params[1:]
        # mea = data[:nrturns, bpmidx]
        # bpm_args = (data, nrturns, bpmidx)
        # bpm_params = (tune, r0, mu)
        # fit, *_ = calc_traj(bpm_params, *bpm_args)

    res = fit - mea
    return res, mea, fit


def get_params(params, *args):
    """."""
    bpmidx, data = args[:2]
    tune, tunes, chrom_decoh = params[:3]
    if bpmidx is None:
        nrbpms = data.shape[1]
        r0n2 = params[3+0*nrbpms:3+1*nrbpms]
        r0 = params[3+1*nrbpms:3+2*nrbpms]
        mu = params[3+2*nrbpms:3+3*nrbpms]
    else:
        raise
    return tune, tunes, chrom_decoh, _np.array(r0n2), _np.array(r0), _np.array(mu)


def calc_residue_vector(params, *args):
    res, _, _ = calc_residue(params, *args)
    return _np.reshape(res, res.size)  # all bpms, every turn, in this order


def calc_residue_norm(params, *args):
    vec = calc_residue_vector(params, *args)
    res = _np.sqrt(_np.sum(vec**2)/len(vec))
    return res, vec


def init_params(espread, nrperiods, bpmidx=None):
    tbt = espread.tbt
    optics = espread.optics
    # data, nrturns, bpmidx, tune, r0, mu = args
    # tunes, chrom_decoh, r0n2 = params

    # initial chrom_decoh is defined by nominal espread and NAFF tunes
    tune = optics.tune
    tunes = optics.naff.tunes_avg
    tbt.tunes_frac = tunes
    tbt.espread = 1*tbt.NOM_ESPREAD
    chrom_decoh = tbt.chrom_decoh
    
    # number of turns - one synchrotron period
    nrturns = int(nrperiods * 1/tunes)
    data = tbt.data_traj[tbt.select_idx_kick]
    args = [bpmidx, data, nrturns]

    nrbpms = tbt.data_nr_bpms
    
    # r0n2 =  1*(_np.sqrt(0.5)/350) * _np.ones(nrbpms)
    data2 = data.copy()
    for bpm in range(data2.shape[1]):
        data2[:,bpm] -= _np.mean(data2[:,bpm])
    nsyn = int(1/tunes)
    max1 = _np.max(data2[0:nsyn,:], axis=0)
    max2 = _np.max(data2[nsyn:2*nsyn,:], axis=0)
    r0n2 = -_np.log(max2/max1)/nsyn**2
    # r0n2 = _np.sqrt(1-max2/max1)/(nsyn)
    
    r0 = _np.array(optics.r0)
    mu = _np.array(optics.mu)


    if bpmidx is not None:
        raise
        # params = (tunes, chrom_decoh, r0n2[bpmidx])
    else:
        params = (tune, tunes, chrom_decoh, ) + tuple(r0n2) + tuple(r0) + tuple(mu)
    return params, args


def espread_fit(folder, fname, kicktype, kickidx, nrperiods, save_flag, print_flag, plot_flag):

    fitlabel = 'espread'

    espread = _Analysis()
    
    # read optics data
    fname_ = (folder + fname).replace('/', '-').replace('.pickle', '-optics.pickle')
    espread.optics = _load_pickle('results/' + fname_)
    
    # create TbT
    tbt = _create_newtbt(folder+fname, kicktype)

    espread.tbt = tbt
    tbt.select_idx_kick = kickidx

    params_ini, args = init_params(espread, nrperiods)
    tune, tunes, chrom_decoh, r0n2, r0, mu = get_params(params_ini, *args)
    tbt.tune_frac = tune
    tbt.tunes_frac = tunes
    tbt.chrom_decoh = chrom_decoh
    espread_ = tbt.espread

    res, vec = calc_residue_norm(params_ini, *args)
    if print_flag:
        print('ini residue : {:.2f} um,  params:{} {} {} {}'.format(res, tune, tunes, chrom_decoh, espread_))

    if plot_flag:
        bpmidx = 0
        if tbt.select_plane_x:
            plane = 'Horizontal'
            color = (0,0,1)
            color_fit = 'C1'
            Jtxt = r'$J_x = {:.3f} \; \mu m.rad$'.format(espread.optics.beta.J)
        else:
            plane = 'Vertical'
            color = (1,0,0)
            color_fit = 'y'
            Jtxt = r'$J_x = {:.3f} \; \mu m.rad$'.format(espread.optics.beta.J)
        orig = args[2]
        args[2] = tbt.data_nr_turns
        res, mea, fit = calc_residue(params_ini, *args)

        graph, ax1 = _plt.subplots(1, 1)

        _plt.plot(mea[:, bpmidx], color=color, alpha=0.5, label='All turns')
        args[2] = orig
        res, mea, fit = calc_residue(params_ini, *args)
        _plt.plot(mea[:, bpmidx], color=color, alpha=1.0, label='Selected turns')

        args_ = [mea, orig, bpmidx]
        params_ = [tune, tunes, chrom_decoh, r0n2[bpmidx], r0[bpmidx], mu[bpmidx]]
        _, _, exp, _, _, _, rn, offset = calc_traj(params_, *args_)
        _plt.plot(rn*exp + offset, color=color_fit, label='Initial Envelope')

        ax1.xaxis.label.set_fontsize(16)
        ax1.yaxis.label.set_fontsize(16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        # _plt.suptitle(plane + ' TbT Data - BPM Index {:03d}'.format(bpmidx))
        _plt.title(Jtxt, fontsize=16)

        _plt.xlabel('Turn number')
        _plt.ylabel(r'Pos [$\mu m$]')
        _plt.legend(fontsize=12)
        _plt.grid()
        _plt.show()

    # fit all bpms
    params_fit, params_fit_err = _fit_leastsqr(
        tbt, args, params_ini, calc_residue_vector)
    
    tune, tunes, chrom_decoh, r0n2, r0, mu = get_params(params_fit, *args)
    tune_err, tunes_err, chrom_decoh_err, r0n2_err, r0_err, mu_err = get_params(params_fit_err, *args)
    tbt.tune_frac = tune
    tbt.tunes_frac = tunes
    tbt.chrom_decoh = chrom_decoh
    espread_ = tbt.espread
    tbt.tune_frac_err = tune_err
    tbt.tunes_frac_err = tunes_err
    tbt.chrom_decoh_err = chrom_decoh_err
    espread_err_ = tbt.espread_err

    res, vec = calc_residue_norm(params_fit, *args)
    if print_flag:
        print('fit residue : {:.2f} um,  params:{} {} {} {}'.format(res, tune, tunes, chrom_decoh, espread_))

    if plot_flag or save_flag:
        bpmidx = 0
        if tbt.select_plane_x:
            plane = 'Horizontal'
            color = (0,0,1)
            color_fit = 'C1'
            Jtxt = r'$J_x = {:.3f} \; \mu m.rad$'.format(espread.optics.beta.J)
        else:
            plane = 'Vertical'
            color = (1,0,0)
            color_fit = (0,1,0)
            Jtxt = r'$J_y = {:.3f} \; \mu m.rad$'.format(espread.optics.beta.J)
        orig = args[2]
        args[2] = tbt.data_nr_turns
        res, mea, fit = calc_residue(params_ini, *args)

        graph, ax1 = _plt.subplots(1, 1)
        graph.subplots_adjust(bottom=0.15, left=0.15)

        _plt.plot(mea[:, bpmidx], color=color, alpha=0.5, label='All turns')
        args[2] = orig
        res, mea, fit = calc_residue(params_ini, *args)
        _plt.plot(mea[:, bpmidx], color=color, alpha=1.0, label='Selected turns')

        args_ = [mea, orig, bpmidx]
        params_ = [tune, tunes, chrom_decoh, r0n2[bpmidx], r0[bpmidx], mu[bpmidx]]
        _, _, exp, _, _, _, rn, offset = calc_traj(params_, *args_)
        _plt.plot(rn*exp + offset, color=color_fit, linewidth=3, label='Fit Envelope')

        ax1.xaxis.label.set_fontsize(16)
        ax1.yaxis.label.set_fontsize(16)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

        # _plt.suptitle(plane + ' TbT Data - BPM Index {:03d}'.format(bpmidx))
        _plt.title(Jtxt, fontsize=16)

        _plt.xlabel('Turn number')
        _plt.ylabel(r'Pos [$\mu m$]')
        _plt.legend(fontsize=12)
        _plt.grid()
        if save_flag:
            fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-' + fitlabel + '-beta.' + save_flag)
            _plt.savefig(fname)
            if not plot_flag:
                _plt.clf()
        if plot_flag:
            _plt.show()


    espread.args = args
    espread.params_ini = params_ini
    espread.params_fit = params_fit
    espread.params_fit_err = params_fit_err
    espread.tune = tune
    espread.r0 = r0
    espread.mu = mu
    espread.tunes = tunes
    espread.chrom_decoh = chrom_decoh
    espread.espread = espread_
    espread.r0n2 = r0n2
    espread.tune_err = tune_err
    espread.r0_err = r0_err
    espread.mu_err = mu_err
    espread.tunes_err = tunes_err
    espread.chrom_decoh_err = chrom_decoh_err
    espread.espread_err = espread_err_
    espread.r0n2_err = r0n2_err
    espread.residue = res

    print()

    if save_flag or plot_flag:

        optics = espread.optics
        tbt = espread.tbt

        if tbt.select_plane_x:
            plane = 'Horizontal' 
            Jtxt = r'$J_x = {:.3f} \; \mu m.rad$'.format(optics.beta.J)
            color= (0,0,1)
        else:
            plane = 'Vertical'
            Jtxt = r'$J_y = {:.3f} \; \mu m.rad$'.format(optics.beta.J)
            color= (1,0,0)

        res, mea, fit = calc_residue(espread.params_fit, *(espread.args))
        resn, _ = calc_residue_norm(espread.params_fit, *(espread.args))

        bpmidx = 0

        # title = plane + ' TbT Chromaticity Decoherence Fit\n' + Jtxt + ' - BPM Index {:03d}'.format(bpmidx)
        
        _plt.plot(mea[:, bpmidx], color='C2', label='TbT')
        _plt.plot(fit[:, bpmidx], color=color, alpha=0.8, label='Fit')
        _plt.plot(res[:, bpmidx], color='C1', label='Res {:.1f} '.format(resn) + r'$\mu m$')
        _plt.ylabel('Pos [um]')
        _plt.xlabel('Turn number')
        # _plt.title(title)
        _plt.title(Jtxt, fontsize=20)

        _plt.legend()

        if save_flag:
            fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-' + fitlabel + '-residue.' + save_flag)
            _plt.savefig(fname)
            if not plot_flag:
                _plt.clf()
            if plot_flag:
                _plt.show()

    # save analysis
    if save_flag:
        espread.tbt._model_twiss = None
        fname = 'results/' + tbt.data_fname.replace('/', '-').replace('.pickle', '-' + fitlabel + '.pickle')
        _save_pickle(espread, fname, True)

    
if __name__ == "__main__":
    
    save_flag = 'png'
    print_flag = True
    plot_flag = False

    # --- multibunch small amplitude kicks - before cycling ---
    
    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m005urad_chrom=2p5.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)
    
    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m010urad_chrom=2p5.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m025urad_chrom=2p5.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)
      
    # # --- multibunch horizontal - after cycling ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m150urad_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m200urad_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m250urad_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMX', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)


    # --- multibunch vertical - after cycling ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_100volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_150volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_200volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_250volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_300volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_350volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_400volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, nrperiods=1.2, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_450volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_500volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_550volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_600volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_650volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_700volts_after_cycle.pickle'
    # espread_fit(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)
