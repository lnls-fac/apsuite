"""."""

import numpy as np
import matplotlib.pyplot as mplt

from pymodels import si
import pyaccel
from . import OrbitCorr

MARKER_NAMES = {
    'B1': 'B1_SRC',
    'B2': 'B2_SRC',
    'BC': 'mc',
}

BENDING_BPM_SEC_IDCS = {
    'B1': [0, 1],
    'B2': [2, 3],
    'BC': [3, 4],
}


def _get_sec_bpm_indices(bending_type='B1'):
    bdict = BENDING_BPM_SEC_IDCS
    return bdict[bending_type][0], bdict[bending_type][1]


def _get_closest_bpms_indices(bending_type='B1', bidx=0, n_bpms_out=2):
    bpm1_sec_index, bpm2_sec_index = _get_sec_bpm_indices(bending_type)
    idlist = np.arange(0, 160, 1)
    idcs_ignore = list()
    for i in np.arange(n_bpms_out):
        idcs_ignore.append(idlist[bpm1_sec_index + 8*bidx - (i+1)])
        idcs_ignore.append(idlist[bpm2_sec_index + 8*bidx + (i+1)])
    idcs_ignore = np.array(idcs_ignore)
    idcs_ignore = np.tile(idcs_ignore, 2)
    idcs_ignore[n_bpms_out*2:] += 160
    return idcs_ignore


def _get_removed_corrs_indices(bending_type='B1', bidx=0):
    ch_idcs = dict()
    cv_idcs = dict()
    ch_idcs['B1'] = [bidx*6 + 0, bidx*6 + 1]
    cv_idcs['B1'] = [bidx*8 + 0, bidx*8 + 1]

    ch_idcs['B2'] = [bidx*6 + 2]
    cv_idcs['B2'] = [bidx*8 + 2, bidx*8 + 3]

    ch_idcs['BC'] = []
    cv_idcs['BC'] = []

    return ch_idcs[bending_type], cv_idcs[bending_type]


def calc_matrices(bending_type='B1', minsingval=0.2, bidx=0,
                  deltax=10e-6, n_bpms_out=3):
    """."""
    # NOTE: valid only for bendings in subsectors C1, C2 or BC
    bpm1_sec_index, bpm2_sec_index = _get_sec_bpm_indices(bending_type)

    # Create accelerator and orbcorr
    mod = si.create_accelerator()
    mod.cavity_on = True
    orbcorr = OrbitCorr(mod, 'SI', use6dorb=True)
    orbcorr.params.enblrf = True
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = minsingval

    # Get source marker idx
    bidx = max(min(bidx, 19), 0)
    nr = 1 if bending_type == 'BC' else 2
    mbend = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', MARKER_NAMES[bending_type])[nr*bidx]

    orb0 = orbcorr.get_orbit()
    kicks0 = orbcorr.get_kicks()

    # Get BPM indices
    idcs = np.array(
        [bpm1_sec_index, bpm2_sec_index,
         160+bpm1_sec_index, 160+bpm2_sec_index])
    idcs += 8*bidx

    # remove corrs between BPMs
    ch_idcs, cv_idcs = _get_removed_corrs_indices(bending_type, bidx)
    if len(ch_idcs) != 0:
        orbcorr.params.enbllistch[ch_idcs] = False
    if len(cv_idcs) != 0:
        orbcorr.params.enbllistcv[cv_idcs] = False

    # remove closest BPMS
    idcs_ignore = _get_closest_bpms_indices(bending_type, bidx, n_bpms_out)
    if idcs_ignore.size != 0:
        orbcorr.params.enbllistbpm[idcs_ignore] = False

    mat_i2s = np.zeros((4, 4), dtype=float)
    mat_i2r = np.zeros((4, 4), dtype=float)
    for i, idx in enumerate(idcs):
        gorb = orb0.copy()
        orbcorr.set_kicks(kicks0)

        gorb[idx] += deltax/2
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbp = orbcorr.get_orbit()[idcs]
        b2p = pyaccel.tracking.find_orbit(
            orbcorr.respm.model, indices='open'
        )
        b2p = b2p[0:4, mbend]

        gorb[idx] -= deltax
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbn = orbcorr.get_orbit()[idcs]
        b2n = pyaccel.tracking.find_orbit(
            orbcorr.respm.model, indices='open')
        b2n = b2n[0:4, mbend]

        mat_i2s[:, i] = (b2p - b2n) / deltax
        mat_i2r[:, i] = (orbp - orbn) / deltax

    mat_s2r = np.linalg.solve(mat_i2s.T, mat_i2r.T).T
    return mat_i2s, mat_i2r, mat_s2r


def test_matrices(flag_n_bpms=True, flag_singvals=True):
    """."""
    ms_i2s = []
    ms_i2r = []
    ms_s2r = []
    cases = []
    svals = 0.2
    if flag_n_bpms:
        for n_bpms in [0, 1, 2]:
            cases.append((n_bpms, svals))
            m_i2s, m_i2r, m_s2r = calc_matrices(
                minsingval=svals, bidx=0, n_bpms_out=n_bpms
            )
            ms_i2s.append(m_i2s)
            ms_i2r.append(m_i2r)
            ms_s2r.append(m_s2r)

    n_bpms = 0
    if flag_singvals:
        for svals in [0.2, 2, 20]:
            cases.append((n_bpms, svals))
            m_i2s, m_i2r, m_s2r = calc_matrices(
                minsingval=svals, bidx=0, n_bpms_out=n_bpms
            )
            ms_i2s.append(m_i2s)
            ms_i2r.append(m_i2r)
            ms_s2r.append(m_s2r)

    fig, (a_i2s, a_i2r, a_s2r) = mplt.subplots(3, 1, figsize=(6, 9))

    for m_i2s, m_i2r, m_s2r, case in zip(ms_i2s, ms_i2r, ms_s2r, cases):
        lab = f'nbpm={case[0]:d} svals={case[1]:.2f}'
        a_i2s.plot(m_i2s.ravel(), '-o', label=lab)
        a_i2r.plot(m_i2r.ravel(), '-o', label=lab)
        a_s2r.plot(m_s2r.ravel(), '-o', label=lab)

    a_i2r.legend(loc='lower center', bbox_to_anchor=(0.5, 1))
    fig.tight_layout()
    return fig, (a_i2s, a_i2r, a_s2r)


def test_bumps(
    bending_type='B1',
    bidx=0,
    angx=50e-6,
    angy=50e-6,
    posx=100e-6,
    posy=100e-6,
    n_bpms_out=3,
    m_s2r=None
):
    """."""
    bpm1_sec_index, bpm2_sec_index = _get_sec_bpm_indices(bending_type)

    # Get bump matrices
    if m_s2r is None:
        _, _, m_s2r = calc_matrices(
            bending_type=bending_type, minsingval=0.2,
            bidx=bidx, n_bpms_out=n_bpms_out
        )

    vec = np.array([posx, angx, posy, angy])

    # Create accelerator and orbcorr
    mod = si.create_accelerator()
    mod.cavity_on = True
    orbcorr = OrbitCorr(mod, 'SI', use6dorb=True)
    orbcorr.params.enblrf = True
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = 0.2

    # Get source marker idx
    nr = 1 if bending_type == 'BC' else 2
    mbend = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', MARKER_NAMES[bending_type])[nr*bidx]

    # Get BPM indices
    idcs = np.array(
        [bpm1_sec_index, bpm2_sec_index,
         160+bpm1_sec_index, 160+bpm2_sec_index])
    idcs += 8*bidx

    # remove corrs between BPMs
    ch_idcs, cv_idcs = _get_removed_corrs_indices(bending_type, bidx)
    if len(ch_idcs) != 0:
        orbcorr.params.enbllistch[ch_idcs] = False
    if len(cv_idcs) != 0:
        orbcorr.params.enbllistcv[cv_idcs] = False

    # remove closest BPMS
    idcs_ignore = _get_closest_bpms_indices(bending_type, bidx, n_bpms_out)
    if idcs_ignore.size != 0:
        orbcorr.params.enbllistbpm[idcs_ignore] = False

    gorb = orbcorr.get_orbit()

    x = np.dot(m_s2r, vec)
    gorb[idcs] = x
    orbcorr.correct_orbit(goal_orbit=gorb)
    xres = pyaccel.tracking.find_orbit(
        orbcorr.respm.model, indices='open')[0:4, mbend]

    fig, (ax, ay, az) = mplt.subplots(3, 1, figsize=(6, 9))

    ax.plot(1e6*vec, '-o', label='Input bump (posx, angx, posy, angy)')
    ax.plot(1e6*xres, '-o', label='Resultant bump ')
    ax.legend()

    ay.plot(orbcorr.get_kicks()[:-1]*1e6)
    ay.set_ylabel('Corr. kicks [urad]')
    ay.set_xlabel('Corr idx')

    az.plot(orbcorr.get_orbit()*1e6)
    az.set_ylabel('Orbit [urad]')
    az.set_xlabel('BPMS idx')

    fig.tight_layout()
    return fig, (ax, ay, az)
