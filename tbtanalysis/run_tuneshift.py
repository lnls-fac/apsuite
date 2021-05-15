#!/usr/bin/env python-sirius

"""."""

import numpy as _np
import matplotlib.pyplot as _plt
import scipy.optimize as _opt

from mathphys.functions import load_pickle as _load_pickle
from lib import create_tbt, calc_param_stats, multibunch_kick_spread

def calc_traj_tuneshift(params, *args):
    """BPM averaging due to tune-shift decoherence.

    nu ~ nu0 + k_decoh * a**2
    See Laurent Nadolski Thesis, Chapter 4, pg. 123, Eq. 4.28
    """

    # nrturns, data, bpmidx, plane, kxx, kxy, kyx, kyy, coup, mu, beta = args
    nrturns, data, bpmidx, plane, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, coup, mu, beta = args

    # tunes, chrom_decoh, tune, J, e0 = params
    tune, e0 = params

    ex = e0 * 1 / (1 + coup)
    ey = e0 * coup / (1 + coup)
    alphax = 1.73/ 17000
    alphay = 1.73/ 22000
    alphas = 1.73/ 13000

    offset = _np.mean(data[:nrturns, bpmidx])
    
    n = _np.arange(0, nrturns)
    pi = _np.pi

    if plane:
        dmp = _np.exp(-n*alphax)
        tune0 = tune - kxx * (4 * ex + J)
        J *= dmp
        thetaD = (4* pi * kxx * ex) * n
        thetaC = (4* pi * kxy * ey) * n
        thetaJ = (4* pi * kxx * J * dmp * dmp) * n
    else:
        dmp = _np.exp(-n*alphay)
        tune0 = tune - kyy * (4 * ey + J)
        J *= dmp
        thetaD = (4* pi * kyy * ey) * n
        thetaC = (4* pi * kyx * ex) * n
        thetaJ = (4* pi * kyy * J * dmp * dmp) * n
        nu = ny
    
    Fxx1 = 1 / (1 + thetaD**2)
    Fxx2 = _np.exp(-0.5 * thetaD * thetaJ * Fxx1)
    Fxy = 1 / ( 1 + thetaC**2)
    A0 = 2*pi * (tune0) * n + (mu - _np.pi/2)
    A1 = 2*_np.arctan(thetaD)
    A2 = 0.5 * thetaJ * Fxx1
    A3 = 2*_np.arctan(thetaC)

    alp = chrom_decoh * _np.sin(_np.pi * tunes * n)
    exp = _np.exp(-alp**2/2.0)
    # dmp = 1
    traj = -_np.sqrt(J*beta) * Fxx1 * Fxx2 * Fxy * exp * _np.sin(A0 + A1 + A2 + A3) + offset
    # return traj, alp, exp
    # traj = -_np.sqrt(J) * Fxx1 * Fxx2 + offset

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


def calc_tuneshift_residue(params, *args):
    """."""
    # args = (nrturns, data_traj, kxx, kxy, kyx, kyy)
    nrturns, data, plane, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, coup, mu, beta = args
    # nrturns, data = args[:2]
    nrbpms = data.shape[1]
    # mu = args[7+0*nrbpms:7+1*nrbpms]
    
    # tunes, chrom_decoh, J, ex, ey = params
    fit = _np.zeros((nrturns, nrbpms))
    res = _np.zeros((nrturns, nrbpms))
    mea = data[:nrturns,:]
    for bpmidx in range(nrbpms):
        bpm_args = (nrturns, mea, bpmidx, plane, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, coup, mu[bpmidx], beta[bpmidx])
        bpm_params = params
        fit[:, bpmidx] = calc_traj_tuneshift(bpm_params, *bpm_args)
    res = fit - mea
    return res, mea, fit


def calc_tuneshift_residue_vector(params, *args):
    # print('h')
    # print(params[0])
    res, _, _ = calc_tuneshift_residue(params, *args)
    # res = res[:,0]
    return _np.reshape(res, res.size)  # all bpms, every turn, in this order


def calc_tuneshift_residue_norm(params, *args):
    vec = calc_tuneshift_residue_vector(params, *args)
    res = _np.sqrt(_np.sum(vec**2)/len(vec))
    return res, vec


def init_tuneshift_params(tbt, kickidx=0):
    # load chrom params from file
    fname = tbt.data_fname.replace('/', '-').replace('.pickle', '-results.pickle')
    chrom_params = _load_pickle('results/' + fname)
    tunes = chrom_params['tunes'] 
    tune = chrom_params['tune']
    beta = chrom_params['beta']
    espread = chrom_params['espread']
    chrom_decoh = chrom_params['chrom_decoh']
    r0 = chrom_params['r0']
    mu = chrom_params['mu']
    J = _np.mean(chrom_params['J'])

    # calc sigma
    twiss, bpmsidx = tbt.calc_model_twiss()
    k = tbt.NOM_COUPLING
    e0 = tbt.NOM_EMITTANCE
    ex = e0 * 1 / (1 + k)
    ey = e0 * k / (1 + k)
    
    if tbt.select_plane_x:
        # eta = twiss.etax[bpmsidx] * 1e6
        # beta = twiss.betax[bpmsidx] * 1e6
        # sigma = _np.sqrt(ex * beta + 0*(espread * eta)**2)
        data_traj = tbt.data_trajx[kickidx].copy()
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
    
    # for bpmidx in range(data_traj.shape[1]):
    #     data_traj[:,bpmidx] /= _np.sqrt(beta[bpmidx])

    factor = 1
    kxx = factor * tbt.NOM_KXX_DECOH_NORM
    kxy = factor * tbt.NOM_KXY_DECOH_NORM
    kyx = factor * tbt.NOM_KYX_DECOH_NORM
    kyy = factor * tbt.NOM_KYY_DECOH_NORM

    kxx = -9.6/1000
    kxy = factor * tbt.NOM_KXY_DECOH_NORM
    kyx = factor * tbt.NOM_KYX_DECOH_NORM
    kyy = 44.7/1000

    # args
    nrturns = tbt.data_nr_turns
    # nrturns = 300 * 2
    # args = (nrturns, data_traj, tbt.select_plane_x, kxx, kxy, kyx, kyy, tbt.NOM_COUPLING, mu, beta)
    args = (nrturns, data_traj, tbt.select_plane_x, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, tbt.NOM_COUPLING, mu, beta)
    # args += tuple(mu)

    # params
    # params_ini = [tunes, chrom_decoh, tune, J, e0]
    params_ini = [tune, e0]
    # print('params_ini: ', params_ini)
    # params_ini += list(r0) + list(mu)

    
    # params_ini_err
    params_ini_err = None

    return args, params_ini, params_ini_err


def print_params(tbt, params, *args):
    # tunes, chrom_decoh, tune, J, e0 = params
    nrturns, data_traj, _, tunes, chrom_decoh, J, kxx, kxy, kyx, kyy, _, mu, beta = args
    tune, e0 = params

    tbt.tunes_frac = tunes
    tbt.chromx_decoh = chrom_decoh
    residue, _ = calc_tuneshift_residue_norm(params, *args)
    print('residue [um]    :', residue)
    print('tunex           :', tune)
    print('tunes           :', tunes)
    print('espread [%]     :', tbt.espread*100)
    print('kxx     [1/um]  :', kxx)
    print('J       [um]    :', J)
    print('e0      [nm.rad]:', e0*1e3)

def analysis_tuneshift(folder, fname, kicktype, kickidx=0):
    tbt = create_tbt(folder+fname, kicktype)
    tbt.select_idx_kick = kickidx
    args, params_ini, params_ini_err = init_tuneshift_params(tbt, 0)

    # _, _, tune, beta, sigma = args


    print_params(tbt, params_ini, *args)
    res, mea, fit = calc_tuneshift_residue(params_ini, *args)
    bpmidx = 0
    _plt.plot(fit[:,bpmidx], label='fit')
    _plt.plot(mea[:,bpmidx], label='mea')
    _plt.title('BPM {:03d} TbT data - Fit x Meas.'.format(bpmidx))
    _plt.legend()
    _plt.xlabel('turn')
    _plt.ylabel('posx [um]')
    _plt.savefig('fit-before.svg')
    _plt.show()

    # print(params_ini)

    fit_data = _opt.least_squares(
        fun=calc_tuneshift_residue_vector,
        x0=params_ini,
        args=args,
        method='lm')
    params_fit = fit_data['x']

    print_params(tbt, params_fit, *args)
    res, mea, fit = calc_tuneshift_residue(params_fit, *args)
    for bpmidx in range(160):
        rms_res = _np.sqrt(_np.sum(res[:,bpmidx]**2)/res.shape[0])
        rms_mea = _np.sqrt(_np.sum((mea[:,bpmidx]-_np.mean(mea[:,bpmidx]))**2)/mea.shape[0])
        _plt.plot(fit[:,bpmidx], label='fit')
        _plt.plot(mea[:,bpmidx], label='mea') # (rms: {:.1f} um)'.format(rms_mea))
        # _plt.plot(res[:,bpmidx], label='res') # (rms: {:.1f} um)'.format(rms_res))
        _plt.title('BPM {:03d} TbT data - Fit x Meas.'.format(bpmidx))
        _plt.legend()
        _plt.xlabel('turn')
        _plt.ylabel('pos [um]')
        _plt.savefig('fit-after.svg')
        _plt.show()
        break


if __name__ == "__main__":
    # multibunch_kick_spread(50)

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    # fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    fname = 'tbt_data_horizontal_m250urad_after_cycle.pickle'

    analysis_tuneshift(folder, fname, 'KXX', kickidx=0)



