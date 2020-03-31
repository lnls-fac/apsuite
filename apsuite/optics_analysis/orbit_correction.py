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
        """Return corrector kicks."""
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
            hkick.append(sum(hkick_seg))
        vkick = []
        for mag in cvidx:
            vkick_seg = []
            for seg in mag:
                vkick_seg.append(model[seg].vkick_polynom)
            vkick.append(sum(vkick_seg))
        hkick = np.array(hkick)
        vkick = np.array(vkick)
        return np.hstack((hkick, vkick))

    def _set_delta_kicks(self, model=None, corridx=None, delta_kicks=None):
        if model is None:
            model = self.model
        if corridx is None:
            corridx = self.corr_idx
        if delta_kicks is None:
            raise Exception('Missing Delta Kicks values')
        for idx_mag, mag in enumerate(corridx):
            delta = delta_kicks[idx_mag]/len(mag)
            for _, seg in enumerate(mag):
                if idx_mag < self._nch:
                    kick = model[seg].hkick_polynom + delta
                    model[seg].hkick_polynom = kick
                else:
                    kick = model[seg].vkick_polynom + delta
                    model[seg].vkick_polynom = kick

    def _check_kicks(self, model=None, chidx=None, cvidx=None):
        if model is None:
            model = self.model
        if chidx is None:
            chidx = self.ch_idx
        if cvidx is None:
            cvidx = self.cv_idx

        kicks = self.get_kicks(model, chidx, cvidx)
        kicksx, kicksy = kicks[:self._nch], kicks[self._nch:]
        kicksx_above = kicksx > OrbitCorr.MAX_HKICK
        kicksy_above = kicksy > OrbitCorr.MAX_VKICK

        if sum(kicksx_above) or sum(kicksy_above):
            str_above = '\n'
            str_above += 'HKick > MaxHKick at CHs: {0:s} \n'
            str_above += 'VKick > MaxVKick at CVs: {1:s}'
            xlist = str(np.argwhere(kicksx_above).flatten())
            ylist = str(np.argwhere(kicksy_above).flatten())
            raise ValueError(str_above.format(xlist, ylist))

    @staticmethod
    def get_figm(res):
        """Calculate figure of merit from residue vector."""
        return np.sqrt(np.sum(res*res)/res.size)

    def orbit_corr(self,
                   model=None,
                   jacobian_matrix=None,
                   goal_orbit=None,
                   nsv=None, nr_max=10, tol=1e-6):
        """Orbit correction.

        Calculates the pseudo-inverse of orbit correction matrix via SVD
        and minimizes the residue vector [CODx@BPM, CODy@BPM].
        """
        if model is None:
            model = self.model
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
        bestfigm = OrbitCorr.get_figm(dorb)
        if bestfigm < tol:
            return OrbitCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dkicks = np.dot(ismat, dorb)
            self._set_delta_kicks(model=model, delta_kicks=dkicks)
            self._check_kicks(model=model)
            orb = self._get_orbit_residue(model)
            dorb = orb - goal_orbit
            figm = OrbitCorr.get_figm(dorb)
            diff_figm = np.abs(bestfigm - figm)
            if figm < bestfigm:
                bestfigm = figm
            if diff_figm < tol:
                break
        else:
            return OrbitCorr.CORR_STATUS.Fail
        return OrbitCorr.CORR_STATUS.Sucess
