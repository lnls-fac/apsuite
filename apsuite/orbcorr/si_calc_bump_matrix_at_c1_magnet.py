"""."""

import numpy as np
import matplotlib.pyplot as mplt

from pymodels import si
import pyaccel
from . import OrbitCorr


def calc_matrices(minsingval=0.2, mb1idx=0, deltax=10e-6, n_bpms_out=3):
    """."""
    # NOTE: valid only for first B1 in sector, the one in subsec C1!

    bpm1_sec_index = 0
    bpm2_sec_index = 1

    mod = si.create_accelerator()
    mod.cavity_on = True
    orbcorr = OrbitCorr(mod, 'SI', use6dorb=True)
    orbcorr.params.enblrf = True
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = minsingval

    mb1idx = max(min(mb1idx, 19), 0)
    mb1i = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'B1_SRC')[2*mb1idx]

    orb0 = orbcorr.get_orbit()
    kicks0 = orbcorr.get_kicks()

    idcs = np.array(
        [bpm1_sec_index, bpm2_sec_index,
         160+bpm1_sec_index, 160+bpm2_sec_index])
    idcs += 8*mb1idx

    # remove corrs between BPMs
    orbcorr.params.enbllistch[mb1idx*6 + 0] = False
    orbcorr.params.enbllistch[mb1idx*6 + 1] = False
    orbcorr.params.enbllistcv[mb1idx*8 + 0] = False
    orbcorr.params.enbllistcv[mb1idx*8 + 1] = False

    # remove closest BPMS
    idlist = np.arange(0, 160, 1)
    idcs_ignore = list()
    for i in np.arange(n_bpms_out):
        idcs_ignore.append(idlist[bpm1_sec_index + 8*mb1idx - (i+1)])
        idcs_ignore.append(idlist[bpm2_sec_index + 8*mb1idx + (i+1)])
    idcs_ignore = np.array(idcs_ignore)
    idcs_ignore = np.tile(idcs_ignore, 2)
    idcs_ignore[n_bpms_out*2:] += 160
    if idcs_ignore.size != 0:
        orbcorr.params.enbllistbpm[idcs_ignore] = False

    matb2 = np.zeros((4, 4), dtype=float)
    mator = np.zeros((4, 4), dtype=float)
    for i, idx in enumerate(idcs):
        gorb = orb0.copy()
        orbcorr.set_kicks(kicks0)

        gorb[idx] += deltax/2
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbp = orbcorr.get_orbit()[idcs]
        b2p = pyaccel.tracking.find_orbit(
            orbcorr.respm.model, indices='open'
        )
        b2p = b2p[0:4, mb1i]

        gorb[idx] -= deltax
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbn = orbcorr.get_orbit()[idcs]
        b2n = pyaccel.tracking.find_orbit(
            orbcorr.respm.model, indices='open')
        b2n = b2n[0:4, mb1i]

        matb2[:, i] = (b2p - b2n) / deltax
        mator[:, i] = (orbp - orbn) / deltax

    matful = np.linalg.solve(matb2.T, mator.T).T
    return matb2, mator, matful


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
                minsingval=svals, mb1idx=0, n_bpms_out=n_bpms
            )
            ms_i2s.append(m_i2s)
            ms_i2r.append(m_i2r)
            ms_s2r.append(m_s2r)

    n_bpms = 0
    if flag_singvals:
        for svals in [0.2, 2, 20]:
            cases.append((n_bpms, svals))
            m_i2s, m_i2r, m_s2r = calc_matrices(
                minsingval=svals, mb1idx=0, n_bpms_out=n_bpms
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
    angx=50e-6,
    angy=50e-6,
    posx=100e-6,
    posy=100e-6,
    mb1idx=0,
    n_bpms_out=3,
    m_s2r=None
):
    """."""
    bpm1_sec_index = 0
    bpm2_sec_index = 1

    if m_s2r is None:
        _, _, m_s2r = calc_matrices(
            minsingval=0.2, mb1idx=mb1idx, n_bpms_out=n_bpms_out
        )

    vec = np.array([posx, angx, posy, angy])

    mod = si.create_accelerator()
    mod.cavity_on = True
    orbcorr = OrbitCorr(mod, 'SI', use6dorb=True)
    orbcorr.params.enblrf = True
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = 0.2

    mci = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'B1_SRC')[2*mb1idx]

    idcs = np.array(
        [bpm1_sec_index, bpm2_sec_index,
         160+bpm1_sec_index, 160+bpm2_sec_index])
    idcs += 8*mb1idx

    # remove closest BPMS
    idlist = np.arange(0, 160, 1)
    idcs_ignore = list()
    for i in np.arange(n_bpms_out):
        idcs_ignore.append(idlist[bpm1_sec_index + 8*mb1idx - (i+1)])
        idcs_ignore.append(idlist[bpm2_sec_index + 8*mb1idx + (i+1)])
    idcs_ignore = np.array(idcs_ignore)
    idcs_ignore = np.tile(idcs_ignore, 2)
    idcs_ignore[n_bpms_out*2:] += 160
    if idcs_ignore.size != 0:
        orbcorr.params.enbllistbpm[idcs_ignore] = False

    # remove corrs between BPMs
    orbcorr.params.enbllistch[mb1idx*6 + 0] = False
    orbcorr.params.enbllistch[mb1idx*6 + 1] = False
    orbcorr.params.enbllistcv[mb1idx*8 + 0] = False
    orbcorr.params.enbllistcv[mb1idx*8 + 1] = False

    gorb = orbcorr.get_orbit()

    x = np.dot(m_s2r, vec)
    gorb[idcs] = x
    orbcorr.correct_orbit(goal_orbit=gorb)
    xres = pyaccel.tracking.find_orbit(
        orbcorr.respm.model, indices='open')[0:4, mci]

    fig, (ax, ay, az) = mplt.subplots(3, 1, figsize=(6, 9))

    ax.plot(1e6*vec, '-o')
    ax.plot(1e6*xres, '-o')

    ay.plot(orbcorr.get_kicks()[:-1]*1e6)
    az.plot(orbcorr.get_orbit()*1e6)

    fig.tight_layout()
    return fig, (ax, ay, az)
