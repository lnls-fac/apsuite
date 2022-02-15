"""."""

import numpy as np
import matplotlib.pyplot as mplt
import matplotlib.gridspec as mgs

from pymodels import si
import pyaccel
from . import OrbitCorr


def calc_matrices(minsingval=0.2, mcidx=0, deltax=10e-6):
    """."""
    mod = si.create_accelerator()
    orbcorr = OrbitCorr(mod, 'SI')
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = minsingval

    mcidx = max(min(mcidx, 19), 0)
    mci = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'mc')[mcidx]

    orb0 = orbcorr.get_orbit()
    kicks0 = orbcorr.get_kicks()

    idcs = np.array([3, 4, 160+3, 160+4])
    idcs += 8*mcidx

    matbc = np.zeros((4, 4), dtype=float)
    mator = np.zeros((4, 4), dtype=float)
    for i, idx in enumerate(idcs):
        gorb = orb0.copy()
        orbcorr.set_kicks(kicks0)
        gorb[idx] += deltax/2
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbp = orbcorr.get_orbit()[idcs]
        bcp = pyaccel.tracking.find_orbit6(
            orbcorr.respm.model, indices='open')[0:4, mci]

        gorb[idx] -= deltax
        orbcorr.correct_orbit(goal_orbit=gorb)
        orbn = orbcorr.get_orbit()[idcs]
        bcn = pyaccel.tracking.find_orbit6(
            orbcorr.respm.model, indices='open')[0:4, mci]
        matbc[:, i] = (bcp - bcn) / deltax
        mator[:, i] = (orbp - orbn) / deltax

    matful = np.linalg.solve(matbc.T, mator.T).T
    return matbc, mator, matful


def test_matrices():
    """."""
    mbc0, mor0, mful0 = calc_matrices(minsingval=0.2, mcidx=0)
    mbc1, mor1, mful1 = calc_matrices(minsingval=2, mcidx=0)
    mbc2, mor2, mful2 = calc_matrices(minsingval=20, mcidx=0)

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


def test_bumps(angx=50e-6, angy=50e-6, posx=100e-6, posy=100e-6, bcidx=0):
    """."""
    _, _, mful = calc_matrices(minsingval=0.2, mcidx=0)

    vec = np.array([posx, angx, posy, angy])

    mod = si.create_accelerator()
    orbcorr = OrbitCorr(mod, 'SI')
    orbcorr.params.tolerance = 1e-9
    orbcorr.params.minsingval = 0.2

    mci = pyaccel.lattice.find_indices(
        orbcorr.respm.model, 'fam_name', 'mc')[bcidx]

    idcs = np.array([3, 4, 160+3, 160+4])
    idcs += 8*bcidx

    idcs_ignore = np.array([1, 2, 5, 6])
    idcs_ignore = np.r_[idcs_ignore, 160 + idcs_ignore]
    idcs_ignore += 8*bcidx
    orbcorr.params.enbllistbpm[idcs_ignore] = False

    gorb = orbcorr.get_orbit()

    x = np.dot(mful, vec)
    gorb[idcs] = x
    orbcorr.correct_orbit(goal_orbit=gorb)
    xres = pyaccel.tracking.find_orbit6(
        orbcorr.respm.model, indices='open')[0:4, mci]

    fig = mplt.figure(figsize=(6, 9))
    gs = mgs.GridSpec(3, 1)

    ax = fig.add_subplot(gs[0, 0])
    ay = fig.add_subplot(gs[1, 0])
    az = fig.add_subplot(gs[2, 0])

    ax.plot(vec, '-o')
    ax.plot(xres, '-o')

    ay.plot(orbcorr.get_kicks()[:-1]*1e6)
    az.plot(orbcorr.get_orbit()*1e6)

    mplt.show()
