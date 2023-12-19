"""."""

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs

from pymodels import si
import pyaccel
from . import OrbitCorr


def calc_matrices(minsingval=0.2, mb2idx=0, deltax=10e-6):
    """."""
    # NOTE: valid only for first B2 in sector, the one in subsec C2!

    bpm1_sec_index = 2
    bpm2_sec_index = 3
    srcpnt_seg_offset_from_cmarker = 10

    mod = si.create_accelerator()
    orbcorr = OrbitCorr(mod, 'SI')
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = minsingval

    mb2idx = max(min(mb2idx, 19), 0)
    mb2i = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'mb2')[2*mb2idx]

    orb0 = orbcorr.get_orbit()
    kicks0 = orbcorr.get_kicks()

    idcs = np.array(
        [bpm1_sec_index, bpm2_sec_index,
         160+bpm1_sec_index, 160+bpm2_sec_index])
    idcs += 8*mb2idx

    # remove corrs between BPMs
    orbcorr.params.enbllistch[mb2idx*6 + 2] = False
    orbcorr.params.enbllistcv[mb2idx*8 + 2] = False
    orbcorr.params.enbllistcv[mb2idx*8 + 3] = False

    matb2 = np.zeros((4, 4), dtype=float)
    mator = np.zeros((4, 4), dtype=float)
    for i, idx in enumerate(idcs):
        gorb = orb0.copy()
        orbcorr.set_kicks(kicks0)

        gorb[idx] += deltax/2
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbp = orbcorr.get_orbit()[idcs]
        b2p = pyaccel.tracking.find_orbit6(
            orbcorr.respm.model, indices='open')
        b2p = b2p[0:4, mb2i - srcpnt_seg_offset_from_cmarker]

        gorb[idx] -= deltax
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbn = orbcorr.get_orbit()[idcs]
        b2n = pyaccel.tracking.find_orbit6(
            orbcorr.respm.model, indices='open')
        b2n = b2n[0:4, mb2i - srcpnt_seg_offset_from_cmarker]

        matb2[:, i] = (b2p - b2n) / deltax
        mator[:, i] = (orbp - orbn) / deltax

    matful = np.linalg.solve(matb2.T, mator.T).T
    return matb2, mator, matful


def test_matrices():
    """."""
    mbc0, mor0, mful0 = calc_matrices(minsingval=0.2, mb2idx=0)
    mbc1, mor1, mful1 = calc_matrices(minsingval=2, mb2idx=0)
    mbc2, mor2, mful2 = calc_matrices(minsingval=20, mb2idx=0)

    fig = mplt.figure(figsize=(6, 9))
    gs = mgs.GridSpec(3, 1)

    abc = fig.add_subplot(gs[0, 0])
    aor = fig.add_subplot(gs[1, 0])
    apr = fig.add_subplot(gs[2, 0])

    abc.plot(mbc0.ravel(), '-o')
    abc.plot(mbc1.ravel(), '-o')
    abc.plot(mbc2.ravel(), '-o')

    aor.plot(mor0.ravel(), '-o')
    aor.plot(mor1.ravel(), '-o')
    aor.plot(mor2.ravel(), '-o')

    apr.plot(mful0.ravel(), '-o')
    apr.plot(mful1.ravel(), '-o')
    apr.plot(mful2.ravel(), '-o')

    mplt.show()


def test_bumps(angx=50e-6, angy=50e-6, posx=100e-6, posy=100e-6, mb2idx=0):
    """."""
    _, _, mful = calc_matrices(minsingval=0.2, mb2idx=0)

    vec = np.array([posx, angx, posy, angy])

    mod = si.create_accelerator()
    orbcorr = OrbitCorr(mod, 'SI')
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = 0.2

    mci = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'mb2')[2*mb2idx]

    idcs = np.array([2, 3, 160+2, 160+3])
    idcs += 8*mb2idx

    idcs_ignore = np.array([0, 1, 4, 5])
    idcs_ignore = np.r_[idcs_ignore, 160 + idcs_ignore]
    idcs_ignore += 8*mb2idx
    orbcorr.params.enbllistbpm[idcs_ignore] = False
    # remove corrs between BPMs
    orbcorr.params.enbllistch[mb2idx*6 + 2] = False
    orbcorr.params.enbllistcv[mb2idx*8 + 2] = False
    orbcorr.params.enbllistcv[mb2idx*8 + 3] = False

    gorb = orbcorr.get_orbit()

    x = np.dot(mful, vec)
    gorb[idcs] = x
    orbcorr.correct_orbit(goal_orbit=gorb)
    xres = pyaccel.tracking.find_orbit6(
        orbcorr.respm.model, indices='open')[0:4, mci - 10]

    fig = mplt.figure(figsize=(6, 9))
    gs = mgs.GridSpec(3, 1)

    ax = fig.add_subplot(gs[0, 0])
    ay = fig.add_subplot(gs[1, 0])
    az = fig.add_subplot(gs[2, 0])

    ax.plot(1e6*vec, '-o')
    ax.plot(1e6*xres, '-o')

    ay.plot(orbcorr.get_kicks()[:-1]*1e6)
    az.plot(orbcorr.get_orbit()*1e6)

    mplt.show()
