#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.optimize as _opt

from mathphys.functions import load_pickle as _load_pickle

from lib import create_tbt, calc_param_stats
from lib import fit_leastsqr as _fit_leastsqr
from lib import Analysis as _Analysis

from run_optics import optics_beta as _optics_beta


def calc_traj(params, *args):
    """BPM averaging due to tune-shift decoherence.

    nu ~ nu0 + k_decoh * a**2
    See Laurent Nadolski Thesis, Chapter 4, pg. 123, Eq. 4.28
    """
    nrturns, data, bpmidx, plane, \
        tune, tunes, chrom_decoh, J, \
        kxx, kxy, kyy, kyx, mu = args
    e0, coup = params

    offset = _np.mean(data[:nrturns, bpmidx])
    ex = e0 * 1 / (1 + coup)
    ey = e0 * coup / (1 + coup)

    if not plane:
        ex, ey = ey, ex
        kxx, kyy, kxy, kyx = kyy, kxx, kyx, kxy

    n = _np.arange(0, nrturns)
    pi = _np.pi

    tune0 = tune - kxx * (4 * ex + J)
    thetaXX = 4 * pi * (kxx * ex) * n
    thetaXY = 4 * pi * (kxy * ey) * n
    
    thetaXI = 2 * chrom_decoh * _np.sin(pi * tunes * n)
    
    F1 = 1 / (1 + thetaXX*2)

    FXI = _np.exp(-thetaXI**2/8)
    FXX = F1 * _np.exp(-(J/ex/2) * thetaXX**2 * F1)
    FXY = 1 / (1 + thetaXY**2)

    A0 = 2 * pi * tune0 * n + (mu - pi/2)
    A1 = 2 * _np.arctan(thetaXX)
    A2 = 2 * _np.arctan(thetaXY)
    A3 = (J/ex/2) * thetaXX * F1
    
    traj = -_np.sqrt(J) * _np.sin(A0 + A1 + A2 + A3) * FXI * FXX * FXY + offset

    if bpmidx == -1:
        print(dmp[-1])
        # print('bpmidx          :', bpmidx)
        # print('tunes           :', tunes)
        # print('chrom_decoh     :', chrom_decoh)
        # print('tune            :', tune)
        # print('J               :', J)
        # print('e0              :', e0)
        # print()
        
    return traj


def calc_residue(params, *args):
    """."""    

    nrturns, data, bpmidx, plane, \
        tune, tunes, chrom_decoh, J, \
        kxx, kxy, kyy, kyx, mu = args
    e0, coup = params
    nrbpms = data.shape[1]

    # tunes, chrom_decoh, J, ex, ey = params
    fit = _np.zeros((nrturns, nrbpms))
    res = _np.zeros((nrturns, nrbpms))
    mea = data[:nrturns,:]
    for bpmidx in range(nrbpms):
        bpm_args = (
            nrturns, mea, bpmidx, plane, 
            tune, tunes, chrom_decoh, J, 
            kxx, kxy, kyy, kyx, mu[bpmidx])
        bpm_params = params
        fit[:, bpmidx] = calc_traj(bpm_params, *bpm_args)
    res = fit - mea
    return res, mea, fit


def get_params(params, *args):
    """."""
    nrturns, data, bpmidx, plane, \
        tune, tunes, chrom_decoh, J, \
        kxx, kxy, kyy, kyx, mu = args
    e0, coup = params
    # return tune, tunes, chrom_decoh, J, kxx, kxy, kyy, kyx, _np.array(r0n2), _np.array(r0), _np.array(mu)
    return e0, coup


def calc_residue_vector(params, *args):
    # print('h')
    # print(params[0])
    res, _, _ = calc_residue(params, *args)
    # res = res[:,0]
    return _np.reshape(res, res.size)  # all bpms, every turn, in this order


def calc_residue_norm(params, *args):
    vec = calc_residue_vector(params, *args)
    res = _np.sqrt(_np.sum(vec**2)/len(vec))
    return res, vec


def init_params(tuneshift):
    
    espread = tuneshift.espread
    tune = espread.tune
    tunes = espread.tunes
    chrom_decoh = espread.chrom_decoh
    mu = espread.mu
    J = espread.beta.J
    beta = espread.beta.beta

    tbt = espread.tbt

    # calc sigma
    twiss, bpmsidx = tbt.calc_model_twiss()
    coup = tbt.NOM_COUPLING
    e0 = tbt.NOM_EMITTANCE
    ex = e0 * 1    / (1 + coup)
    ey = e0 * coup / (1 + coup)
    
    if tbt.select_plane_x:
        # eta = twiss.etax[bpmsidx] * 1e6
        # beta = twiss.betax[bpmsidx] * 1e6
        # sigma = _np.sqrt(ex * beta + 0*(espread * eta)**2)
        data_traj = tbt.data_trajx[tbt.select_idx_kick].copy()
    else:
        # eta = twiss.etay[bpmsidx] * 1e6
        # beta = twiss.betay[bpmsidx] * 1e6
        # sigma = _np.sqrt(ey * beta + 0*(espread * eta)**2)
        data_traj = tbt.data_trajy[kickidx].copy()
    # print(ex)
    # print(ey)
    # print(espread)
    # print(beta)
    # print(sigma)
    
    for bpmidx in range(data_traj.shape[1]):
        data_traj[:,bpmidx] /= _np.sqrt(beta[bpmidx])

    factor = 1
    # kxx = factor * tbt.NOM_KXX_DECOH_NORM
    # kxy = factor * tbt.NOM_KXY_DECOH_NORM
    # kyx = factor * tbt.NOM_KYX_DECOH_NORM
    # kyy = factor * tbt.NOM_KYY_DECOH_NORM

    kxx = -9.6/1000
    kxy = factor * tbt.NOM_KXY_DECOH_NORM
    kyx = factor * tbt.NOM_KYX_DECOH_NORM
    kyy = 44.7/1000

    # args
    nrturns = tbt.data_nr_turns
    args = (
        nrturns, data_traj, None, tbt.select_plane_x, 
        tune, tunes, chrom_decoh, J, 
        kxx, kxy, kyy, kyx, mu)
    
    # params
    params_ini = (e0, coup)
    
    # params_ini_err
    params_ini_err = None

    return args, params_ini, params_ini_err


def print_params(tbt, params, *args):
    # tunes, chrom_decoh, tune, J, e0 = params
    nrturns, data_traj, _, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, _, mu, beta = args
    tune, e0 = params

    tbt.tunes_frac = tunes
    tbt.chromx_decoh = chrom_decoh
    residue, _ = calc_residue_norm(params, *args)
    print('residue [um]    :', residue)
    print('tunex           :', tune)
    print('tunes           :', tunes)
    print('espread [%]     :', tbt.espread*100)
    print('kxx     [1/um]  :', kxx)
    print('J       [um]    :', J)
    print('e0      [nm.rad]:', e0*1e3)


def analysis_tuneshift(folder, fname, kicktype, kickidx=0, save_flag=True, print_flag=True, plot_flag=False):

    tuneshift = _Analysis()

    # read optics data
    fname_ = (folder + fname).replace('/', '-').replace('.pickle', '-espread.pickle')
    tuneshift.espread = _load_pickle('results/' + fname_)

    tbt = create_tbt(folder+fname, kicktype)
    tbt.select_idx_kick = kickidx

    tuneshift.espread.tbt = tbt
    _optics_beta(tuneshift.espread, save_flag=False, print_flag=False, plot_flag=False)
    
    args, params_ini, params_ini_err = init_params(tuneshift)

    res, mea, fit = calc_residue(params_ini, *args)
    _plt.plot(fit[:, 0])
    _plt.plot(mea[:, 0])
    _plt.show()

    res, vec = calc_residue_norm(params_ini, *args)
    if print_flag:
        print('ini residue : {:.2f} um,  params:{}'.format(res, params_ini))

    params_fit, params_fit_err = _fit_leastsqr(
        tbt, args, params_ini, calc_residue_vector)
    
    res, mea, fit = calc_residue(params_fit, *args)
    _plt.plot(fit[:, 0])
    _plt.plot(mea[:, 0])
    _plt.show()

    # e0, coup = get_params(params_fit, *args)
    res, vec = calc_residue_norm(params_fit, *args)
    if print_flag:
        print('fit residue : {:.2f} um,  params:{}'.format(res, params_fit))

    
if __name__ == "__main__":
    # multibunch_kick_spread(50)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    # fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'

    analysis_tuneshift(folder, fname, 'KXX', kickidx=0, save_flag=True, print_flag=True, plot_flag=False)



