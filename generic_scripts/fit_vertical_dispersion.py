"""Script to fit vertical dispersion with skew-quadrupoles.""""

import numpy as np
from mathphys.functions import load_pickle

import pyaccel as pa
from pymodels import si

import siriuspy.clientconfigdb as servconf
from apsuite.orbcorr import OrbRespmat
from apsuite.optics_analysis.tune_correction import TuneCorr


def get_dispersion_from_orm(orm, alpha, rf_freq):
    """."""
    return - alpha * rf_freq * orm[:, -1]


def calc_model_dispersion(mod, bpmidx):
    """."""
    twi, _ = pa.optics.calc_twiss(mod)
    return np.hstack((twi.etax[bpmidx], twi.etay[bpmidx]))


def calc_rms(vec):
    """."""
    return np.sqrt(np.mean(vec*vec))


def get_orm_data(name):
    setup = load_pickle(name)
    if 'data' in setup:
        setup = setup['data']
    return setup


def proces_data(setup, find_best_alpha=True):
    # Get nominal model
    simod = si.create_accelerator()
    simod.cavity_on = True
    simod.radiation_on = False

    simod = adjust_tunes(simod, setup)
    # Get nominal orbit matrix and dispersion
    matrix_nominal = OrbRespmat(simod, 'SI', '6d').get_respm()
    alpha0 = pa.optics.get_mcf(simod)
    print('momentum compaction: {:e}'.format(alpha0))
    idx = pa.lattice.find_indices(simod, 'pass_method', 'cavity_pass')[0]
    rf_freq = simod[idx].frequency
    disp_nominal = get_dispersion_from_orm(matrix_nominal, alpha0, rf_freq)

    # Get measured orbit matrix from configuration server
    client = servconf.ConfigDBClient(config_type='si_orbcorr_respm')
    orbmat_name = setup['orbmat_name']

    orbmat_meas = np.array(client.get_config_value(name=orbmat_name))
    orbmat_meas = np.reshape(orbmat_meas, (320, 281))
    orbmat_meas[:, -1] *= 1e-6

    rf_freq_meas = setup['rf_frequency']
    alpha_meas = alpha0
    if find_best_alpha:
        alphas = (1 + np.linspace(-10, 10, 10001)/100)*alpha0
        errs = []
        for al in alphas:
            disp_meas = get_dispersion_from_orm(orbmat_meas, al, rf_freq_meas)
            err = disp_meas - disp_nominal
            err = np.sqrt(np.mean(err*err))
            errs.append(err)
        alpha_meas = alphas[np.argmin(errs)]
        print('Factor needed for momentum compaction:')
        print(alphas[np.argmin(errs)]/alpha0)

    disp_meas = get_dispersion_from_orm(orbmat_meas, alpha_meas, rf_freq_meas)


def adjust_tunes(simod, setup):
    """."""
    # Adjust tunes to match measured ones
    tunex_goal = 49 + setup['tunex']
    tuney_goal = 14 + setup['tuney']

    print('--- correcting si tunes...')
    tunecorr = TuneCorr(
        simod, 'SI', method='Proportional', grouping='TwoKnobs')
    tunecorr.get_tunes(simod)
    print('    tunes init  : ', tunecorr.get_tunes(simod))
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
        model=simod,
        goal_parameters=np.array([tunex_goal, tuney_goal]),
        jacobian_matrix=tunemat)
    print('    tunes final : ', tunecorr.get_tunes(simod))
    return simod


def calc_dispmat(simod, dksl=1e-6):
    fam = si.get_family_data(simod)
    qsidx = np.array(fam['QS']['index']).ravel()
    # get only chromatic skew quads
    chrom = []
    for qs in qsidx:
        if '0' not in simod[qs].fam_name:
            chrom.append(qs)
    qsidx = np.array(chrom).ravel()
    bpmidx = np.array(fam['BPM']['index']).ravel()

    # eta0 = calc_model_dispersion(simod, bpmidx)
    eta_mat = np.zeros((2*bpmidx.size, qsidx.size))
    for idx, qs in enumerate(qsidx):
        mod = simod[:]
        mod[qs].KsL += dksl/2
        etap = calc_model_dispersion(mod, bpmidx)
        mod[qs].KsL -= dksl
        etan = calc_model_dispersion(mod, bpmidx)
        eta_mat[:, idx] = (etap-etan)/dksl
        mod[qs].KsL += dksl
    return eta_mat


def fit_dispersion(
        simod, eta_mat, disp_meas, bpmidx, qsidx, svals=35, niter=10):
    umat, smat, vhmat = np.linalg.svd(eta_mat, full_matrices=False)
    ismat = 1/smat
    svals = 35
    ismat[svals:] = 0
    imat = vhmat.T @ np.diag(ismat) @ umat.T
    modcorr = simod[:]
    for _ in range(niter):
        eta = calc_model_dispersion(modcorr, bpmidx)
        diff = disp_meas - eta
        # minimize error of vertical dispersion
        diff[:160] *= 0
        print(calc_rms(diff)*1e3)
        stren = imat @ diff
        for idx, qs in enumerate(qsidx):
            modcorr[qs].KsL += stren[idx]
