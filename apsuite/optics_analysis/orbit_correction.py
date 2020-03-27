"""."""

from copy import deepcopy as _dcopy
from collections import namedtuple as _namedtuple
import numpy as np

import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat


class OrbitCorr():
    """."""

    MAX_HKICK = 300e-6  # urad
    MAX_VKICK = 300e-6  # urad
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model, acc, dim='4d'):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self.bpm_idx = self.respm.fam_data['BPM']['index']
        self.ch_idx = self.respm.fam_data['CH']['index']
        self.cv_idx = self.respm.fam_data['CV']['index']
        self.corr_idx = np.vstack((self.ch_idx, self.cv_idx))

    @property
    def _nbpm(self):
        return len(self.bpm_idx)

    @property
    def _nch(self):
        return len(self.ch_idx)

    @property
    def _ncv(self):
        return len(self.cv_idx)

    @property
    def _ncorr(self):
        return self._nch + self._ncv

    def calc_jacobian_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model
        self.respm.model = model
        return self.respm.get_respm()

    def _get_orbit_residue(self, model):
        if self.dim == '4d':
            cod = pyaccel.tracking.find_orbit4(model, indices='open')
        elif self.dim == '6d':
            cod = pyaccel.tracking.find_orbit6(model, indices='open')
        codx = cod[0, self.bpm_idx].flatten()
        cody = cod[2, self.bpm_idx].flatten()
        res = np.hstack((codx, cody))
        return res

    def get_kicks(self, model=None, chidx=None, cvidx=None):
        """Return skew quadrupoles strengths."""
        if model is None:
            model = self.model
        if chidx is None:
            chidx = self.ch_idx
        if cvidx is None:
            cvidx = self.cv_idx
        hkick = []
        for mag in chidx:
            hkick_seg = []
            for seg in mag:
                hkick_seg.append(model[seg].hkick_polynom)
            hkick.append(hkick_seg)
        vkick = []
        for mag in cvidx:
            vkick_seg = []
            for seg in mag:
                vkick_seg.append(model[seg].vkick_polynom)
            vkick.append(vkick_seg)
        hkick = np.array(hkick)
        vkick = np.array(vkick)
        return np.vstack((hkick, vkick))

    def set_kicks(self, model=None, corridx=None, kicks=None):
        """Set skew quadrupoles strengths in the model."""
        if model is None:
            model = self.model
        if corridx is None:
            corridx = self.corr_idx
        if kicks is None:
            raise Exception('Missing Kicks values')
        newmod = _dcopy(model)
        for idx_mag, mag in enumerate(corridx):
            for idx_seg, seg in enumerate(mag):
                kick = kicks[idx_mag][idx_seg]
                if idx_mag < self._nch:
                    if np.abs(kick) > OrbitCorr.MAX_HKICK:
                        kick = np.sign(kick) * OrbitCorr.MAX_HKICK
                        print('Max CH reach')
                    newmod[seg].hkick_polynom = kick
                else:
                    if np.abs(kick) > OrbitCorr.MAX_VKICK:
                        kick = np.sign(kick) * OrbitCorr.MAX_VKICK
                        print('Max CV reach')
                    newmod[seg].vkick_polynom = kick
        return newmod

    @staticmethod
    def get_fm(res):
        """Calculate figure of merit from residue vector."""
        return np.sqrt(np.sum(np.abs(res)**2)/res.size)

    def orbit_corr(self,
                   model,
                   jacobian_matrix=None,
                   goal_orbit=None,
                   nsv=None, nr_max=10, tol=1e-6):
        """Orbit correction.

        Calculates the pseudo-inverse of orbit correction matrix via SVD
        and minimizes the residue vector [CODx@BPM, CODy@BPM].
        """
        if goal_orbit is None:
            goal_orbit = np.zeros(2 * self._nbpm)

        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        else:
            jmat = _dcopy(jacobian_matrix)

        if jmat.shape[1] > self._ncorr:
            jmat = jmat[:, :-1]

        umat, smat, vmat = np.linalg.svd(jmat, full_matrices=False)
        ismat = 1/smat
        ismat[np.isnan(ismat)] = 0
        ismat[np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ismat = np.diag(ismat)
        ismat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)
        orb = self._get_orbit_residue(model)
        dorb = orb - goal_orbit
        bestfigm = OrbitCorr.get_fm(dorb)
        if bestfigm < tol:
            return OrbitCorr.CORR_STATUS.Sucess

        kicks = self.get_kicks(model)

        for _ in range(nr_max):
            dkicks = np.dot(ismat, dorb)
            kicks += np.reshape(dkicks, (-1, 1))
            model = self.set_kicks(model=model, kicks=kicks)
            orb = self._get_orbit_residue(model)
            dorb = orb - goal_orbit
            figm = OrbitCorr.get_fm(dorb)
            diff_figm = np.abs(bestfigm - figm)
            if figm < bestfigm:
                bestfigm = figm
            if diff_figm < tol:
                break
        else:
            return OrbitCorr.CORR_STATUS.Fail
        return OrbitCorr.CORR_STATUS.Sucess
