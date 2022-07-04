"""Script to fit vertical dispersion with skew-quadrupoles."""

import numpy as np
from mathphys.functions import load_pickle

import pyaccel as pa
from pymodels import si

import siriuspy.clientconfigdb as servconf
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat
from apsuite.optics_analysis.tune_correction import TuneCorr

import matplotlib.pyplot as plt
import matplotlib.gridspec as plt_gs


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


def get_orm_setup(name):
    """."""
    print('--- loading orm setup')
    setup = load_pickle(name)
    if 'data' in setup:
        setup = setup['data']
    return setup


def get_data_servconf_adjust_model(setup, find_best_alpha=True):
    """."""
    # Get nominal model
    simod = si.create_accelerator()
    simod.cavity_on = True
    simod.radiation_on = False

    simod = adjust_tunes(simod, setup)
    # Get nominal orbit matrix and dispersion
    matrix_nominal = OrbRespmat(simod, 'SI', '6d').get_respm()
    alpha0 = pa.optics.get_mcf(simod)

    idx = pa.lattice.find_indices(simod, 'pass_method', 'cavity_pass')[0]
    rf_freq = simod[idx].frequency
    disp_nominal = get_dispersion_from_orm(matrix_nominal, alpha0, rf_freq)

    # Get measured orbit matrix from configuration server
    print('--- loading orm from servconf')
    client = servconf.ConfigDBClient(config_type='si_orbcorr_respm')
    orm_name = setup['orbmat_name']
    orm_meas = np.array(client.get_config_value(name=orm_name))
    orm_meas = np.reshape(orm_meas, (320, 281))
    orm_meas[:, -1] *= 1e-6

    rf_freq_meas = setup['rf_frequency']
    alpha_meas = alpha0
    if find_best_alpha:
        print('     search for best mcf to fit dispersion')
        alphas = (1 + np.linspace(-10, 10, 1001)/100)*alpha0
        errs = []
        for alp in alphas:
            disp_meas = get_dispersion_from_orm(orm_meas, alp, rf_freq_meas)
            err = disp_meas - disp_nominal
            err = np.sqrt(np.mean(err*err))
            errs.append(err)
        alpha_meas = alphas[np.argmin(errs)]
        print('     nominal mcf: {:e}'.format(alpha0))
        print('     best mcf: {:e}'.format(alpha_meas))
        print('     ratio best/nominal: {:f}'.format(alpha_meas/alpha0))
    disp_meas = get_dispersion_from_orm(orm_meas, alpha_meas, rf_freq_meas)
    return simod, disp_meas, orm_meas


def adjust_tunes(simod, setup):
    """."""
    # Adjust tunes to match measured ones
    tunex_goal = 49 + setup['tunex']
    tuney_goal = 14 + setup['tuney']

    print('--- moving tunes to match measured values')
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
    """."""
    print('--- calculating dispersion/KsL matrix')
    fam = si.get_family_data(simod)
    qsidx = np.array(fam['QS']['index']).ravel()
    # get only chromatic skew quads
    chrom = []
    for qs in qsidx:
        if '0' not in simod[qs].fam_name:
            chrom.append(qs)
    qsidx = np.array(chrom).ravel()
    bpmidx = np.array(fam['BPM']['index']).ravel()
    disp_mat = np.zeros((2*bpmidx.size, qsidx.size))
    for idx, qs in enumerate(qsidx):
        mod = simod[:]
        mod[qs].KsL += dksl/2
        dispp = calc_model_dispersion(mod, bpmidx)
        mod[qs].KsL -= dksl
        dispn = calc_model_dispersion(mod, bpmidx)
        disp_mat[:, idx] = (dispp-dispn)/dksl
        mod[qs].KsL += dksl
    return disp_mat, bpmidx, qsidx


def fit_dispersion(
        simod, disp_mat, disp_meas, bpmidx, qsidx, setup, svals=35, niter=5):
    """."""
    print('--- fitting vertical dispersion')
    umat, smat, vhmat = np.linalg.svd(disp_mat, full_matrices=False)
    ismat = 1/smat
    ismat[svals:] = 0
    imat = vhmat.T @ np.diag(ismat) @ umat.T
    modfit = simod[:]
    stren_ksl = np.zeros(qsidx.size)
    print('     {:d}/{:d} singular values used'.format(svals, smat.size))
    for _ in range(niter):
        disp = calc_model_dispersion(modfit, bpmidx)
        diff = disp_meas - disp
        # minimize error of vertical dispersion
        # ignore horizontal dispersion
        diff[:160] *= 0
        stren = imat @ diff
        stren_ksl += stren
        print('     residue: {:.2f}, ksl: {:.2e} 1/m (rms)'.format(
            calc_rms(diff)*1e3, calc_rms(stren_ksl)))
        for idx, qs in enumerate(qsidx):
            modfit[qs].KsL += stren[idx]
    modfit = adjust_tunes(modfit, setup)
    return modfit, diff, stren_ksl


def fit_dispersion_scan_singular_values(
        simod, disp_mat, disp_meas, bpmidx, qsidx, setup, niter=5, svals=None):
    """."""
    print('--- scanning singular values used to fit')
    if svals is None:
        svals = list(range(1, 61))
    twi_nom, _ = pa.optics.calc_twiss(simod)
    eta_errors, qsksl = [], []
    betabeatx, betabeaty = [], []
    emitx, emity = [], []
    emitx_ri, emity_ri = [], []
    tunesep = []
    for nsv in svals:
        modfit, diff, kslfit = fit_dispersion(
                    simod, disp_mat, disp_meas, bpmidx, qsidx, setup,
                    svals=nsv, niter=niter)
        twi_fit, _ = pa.optics.calc_twiss(modfit)
        bbx = (twi_fit.betax - twi_nom.betax)/twi_nom.betax*100
        bby = (twi_fit.betay - twi_nom.betay)/twi_nom.betay*100
        eq_fit = pa.optics.EqParamsFromBeamEnvelope(modfit)
        eq_rad = pa.optics.EqParamsFromRadIntegrals(modfit)
        edteng, *_ = pa.optics.calc_edwards_teng(modfit)
        numin, _ = pa.optics.estimate_coupling_parameters(edteng)

        eta_errors.append(calc_rms(diff))
        betabeatx.append(calc_rms(bbx))
        betabeaty.append(calc_rms(bby))
        emitx.append(eq_fit.emit1)
        emity.append(eq_fit.emit2)
        emitx_ri.append(eq_rad.emitx)
        emity_ri.append(eq_rad.emity)
        tunesep.append(numin)
        qsksl.append(kslfit)

    return dict(
        nsvals=np.array(svals),
        eta_errors=np.array(eta_errors),
        betabeatx=np.array(betabeatx),
        betabeaty=np.array(betabeaty),
        emitx=np.array(emitx),
        emity=np.array(emity),
        emitx_ri=np.array(emitx_ri),
        emity_ri=np.array(emity_ri),
        tunesep=np.array(tunesep),
        qsksl=np.array(qsksl))


def plot_dispersion_fit(disp_meas, modfit, bpmidx, svals, svalsmax):
    """."""
    fig = plt.figure(figsize=(10, 4))
    gs = plt_gs.GridSpec(2, 1)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0])

    dispfit = calc_model_dispersion(modfit, bpmidx)
    spos = pa.lattice.find_spos(modfit)[bpmidx]

    ax.plot(spos, disp_meas[:160]*100, '.-', label='meas', linewidth=1)
    ax.plot(spos, dispfit[:160]*100, '-', label='fit', linewidth=1)

    ay.plot(spos, disp_meas[160:]*100, '.-', label='meas', linewidth=1)
    ay.plot(spos, dispfit[160:]*100, '-', label='fit', linewidth=1)

    ay.set_xlabel('bpm idx')
    ax.set_ylabel(r'$\eta_x$ [cm]')
    ay.set_ylabel(r'$\eta_y$ [cm]')
    title = 'Measured and fitted dispersion with skew quadrupoles'
    title += '\n'
    title += f'singular values used: {svals:d}/{svalsmax:d}'
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.5)
    ay.grid(True, ls='--', alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig('vertical_dispersion_fit.png', dpi=300, format='png')
    plt.show()


def plot_dispersion_fit_scan_singular_values(
        nsvals, eta_errors, betabeatx, betabeaty,
        emitx, emity, emitx_ri, emity_ri, tunesep, qsksl):
    """."""
    fig = plt.figure(figsize=(8, 6))
    gs = plt_gs.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0, 0])
    ay1 = ax1.twinx()
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])
    rms_ksl = np.std(qsksl, axis=1)

    ax1.plot(nsvals, eta_errors*1000, 'o-', color='C0')
    ay1.plot(nsvals, rms_ksl*1000, 'o-', color='C1')
    ax1.set_ylabel(r'$\eta_y$ fitting residue [mm]', color='C0')
    ax1.set_yscale('log')
    ay1.set_ylabel(r'KsL $\times 10^3$ [1/m]', color='C1')

    ax2.set_ylabel('Beta-Beatings [%]')
    ax2.plot(
        nsvals, betabeatx, 'o-', color='C0',
        label=r'$\Delta \beta_x/\beta_x$')
    ax2.plot(
        nsvals, betabeaty, 'o-', color='C1',
        label=r'$\Delta \beta_y/\beta_y$')

    ax3.plot(
        nsvals, emity/emitx*100, 'o-', color='C0',
        label=r'$\epsilon_y/\epsilon_x$ Beam Env.')
    ax3.plot(
        nsvals, emity_ri/emitx_ri*100, 'o-', color='C1',
        label=r'$\epsilon_y/\epsilon_x$ Rad. Int.')
    ax3.plot(
        nsvals, tunesep*100, 'o-', color='tab:gray',
        label=r'min $\Delta \nu$')
    ax3.set_xlabel('Number of singular values')
    ax3.set_ylabel('Emittance ratio and \n Min. tune separation [%]')

    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.5)
    ay1.grid(True, ls='--', alpha=0.5)
    ax2.grid(True, ls='--', alpha=0.5)
    ax3.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        'vertical_dispersion_fit_vs_singular_values.png',
        dpi=300, format='png')
    plt.show()


def print_strengths_fitted_model(modfit):
    """."""
    fam = si.get_family_data(modfit)
    qnidx = np.array(fam['QN']['index']).ravel()
    qsidx = np.array(fam['QS']['index']).ravel()
    stren_qn = pa.lattice.get_attribute(modfit, 'KL', qnidx)
    stren_qs = pa.lattice.get_attribute(modfit, 'KsL', qsidx)
    print('--- strengths QN and QS to be used in pymodels.fitted_models')
    print('stren_qn = np.array(')
    print(stren_qn, ')')
    print('stren_qs = np.array(')
    print(stren_qs, ')')


def compare_orms(modfit, orm_meas, chidx=0, cvidx=0):
    """."""
    ormc = OrbRespmat(modfit, acc='SI', dim='4d')
    orm_fit = ormc.get_respm()
    fam = si.get_family_data(modfit)
    namex = np.array(fam['CH']['devnames'])
    namey = np.array(fam['CV']['devnames'])

    correx = []
    correy = []
    for ax in range(0, 120):
        v1_, v2_ = orm_meas[160:, ax], orm_fit[160:, ax]
        correx.append(np.abs(np.corrcoef(v1_, v2_)[0, 1]))
    for ay in range(120, 280):
        v1_, v2_ = orm_meas[:160, ay], orm_fit[:160, ay]
        correy.append(np.abs(np.corrcoef(v1_, v2_)[0, 1]))

    correx = np.array(correx)*100
    correy = np.array(correy)*100

    print('Good correlation CH')
    print(namex[correx > 90])
    print('Good correlation CV')
    print(namey[correy > 95])
    print('Bad correlation CH')
    print(namex[correx < 5])
    print('Bad correlation CV')
    print(namey[correy < 5])

    fig = plt.figure(figsize=(10, 4))
    gs = plt_gs.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    ax.plot(correx, 'o-', label=r'$M_{yx}$')
    ax.plot(correy, 'o-', label=r'$M_{xy}$')
    ax.legend()
    ax.set_xlabel('corrector idx')
    ax.set_ylabel('correlation between meas. and fit ORM [%]')
    ax.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        'compare_off_diagonal_orm_allcorrs_correlation.png',
        dpi=300, format='png')
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    gs = plt_gs.GridSpec(2, 1)
    ax = plt.subplot(gs[0, 0])
    ay = plt.subplot(gs[1, 0])
    ax.plot(orm_meas[160:, chidx], label='meas')
    ax.plot(orm_fit[160:, chidx], label='fit')
    ay.plot(orm_meas[:160, 120+cvidx])
    ay.plot(orm_fit[:160, 120+cvidx])
    ax.set_title(namex[chidx] + f', correlation: {correx[chidx]:.2f} %')
    ax.legend()
    ax.set_xlabel('bpm idx')
    ax.set_ylabel(r'$M_{yx}$ [m/rad]')
    ay.set_title(namey[cvidx] + f', correlation: {correy[cvidx]:.2f} %')
    ay.set_xlabel('bpm idx')
    ay.set_ylabel(r'$M_{xy}$ [m/rad]')
    ax.grid(True, ls='--', alpha=0.5)
    ay.grid(True, ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        'compare_off_diagonal_orm_bad_corresp.png',
        dpi=300, format='png')
    plt.show()


def plot_singular_values_dispmat(dispmat):
    """."""
    _, smat, _ = np.linalg.svd(dispmat, full_matrices=False)
    plt.figure()
    plt.plot(smat/smat[0], 'o-')
    plt.grid(True, ls='--', alpha=0.5)
    plt.yscale('log')
    plt.ylabel('sv/sv_max (log)')
    plt.xlabel('sv index')
    plt.savefig('singular_values_dispmat.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    your_dir = ''
    orm_name = your_dir
    orm_name += '/shared/screens-iocs/data_by_day/2022-05-24-SI_LOCO/'
    orm_name += 'respmat_endofmay22_bpms_03sector_switching_issue.pickle'
    setup = get_orm_setup(orm_name)
    simod, disp_meas, orm_meas = get_data_servconf_adjust_model(
        setup, find_best_alpha=True)
    disp_mat, bpmidx, qsidx = calc_dispmat(simod)
    # plot_singular_values_dispmat(dispmat)

    # # scan singular values
    # out = fit_dispersion_scan_singular_values(
    #     simod, disp_mat, disp_meas, bpmidx, qsidx, setup, niter=10)
    # save_pickle(out, 'etay_fitting_svals_scan')
    # out = load_pickle('etay_fitting_svals_scan')
    # plot_dispersion_fit_scan_singular_values(**out)

    # fit vertical dispersion with fixed singular value
    svals = 41
    modfit, *_ = fit_dispersion(
        simod, disp_mat, disp_meas, bpmidx, qsidx, setup,
        svals=svals, niter=10)
    plot_dispersion_fit(
        disp_meas, modfit, bpmidx, svals=svals, svalsmax=qsidx.size)
    # print_strengths_fitted_model(modfit)
    # good correlation idx: chidx=22, cvidx=35
    # bad correlation idx: chidx=34, cvidx=23
    compare_orms(modfit, orm_meas, chidx=34, cvidx=23)
