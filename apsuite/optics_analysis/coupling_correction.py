"""."""

from copy import deepcopy as _dcopy
import numpy as np
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
import pyaccel


class CouplingCorr():
    """."""

    def __init__(self, model, acc, dim='4d'):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
        self.coup_matrix = []
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self.skew_idx = self.respm.fam_data['QS']['index']
        self.bpm_idx = self.respm.fam_data['BPM']['index']
        self.ch_idx = self.respm.fam_data['CH']['index']
        self.cv_idx = self.respm.fam_data['CV']['index']
        self._nbpm = len(self.bpm_idx)
        self._nch = len(self.ch_idx)
        self._ncv = len(self.cv_idx)
        self._nskew = len(self.skew_idx)

    def calc_coupling_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        nvec = self._nbpm * (self._nch + self._ncv + 1)
        self.coup_matrix = np.zeros((nvec, len(self.skew_idx)))
        delta = 1e-6

        for idx, nmag in enumerate(self.skew_idx):
            modcopy = _dcopy(model)
            for seg in nmag:
                modcopy[seg].KsL += delta/len(nmag)
                elem = self.get_coupling_residue(modcopy) / (delta/len(nmag))
            self.coup_matrix[:, idx] = elem
        return self.coup_matrix

    def get_coupling_residue(self, model):
        """."""
        self.respm.model = model
        orbmat = self.respm.get_respm()
        twi, *_ = pyaccel.optics.calc_twiss(model)
        dispy = twi.etay[self.bpm_idx]
        w_dispy = (self._nch + self._ncv)*10
        dispy *= w_dispy
        mxy = orbmat[:self._nbpm, self._nch:-1]
        myx = orbmat[self._nbpm:, :self._nch]
        res = mxy.flatten()
        res = np.hstack((res, myx.flatten()))
        res = np.hstack((res, dispy.flatten()))
        return res

    def get_ksl(self, model=None, skewidx=None):
        """."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        ksl = []
        for mag in skewidx:
            ksl_seg = []
            for seg in mag:
                ksl_seg.append(model[seg].KsL)
            ksl.append(ksl_seg)
        return np.array(ksl)

    def set_ksl(self, model=None, skewidx=None, ksl=None):
        """."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        if ksl is None:
            raise Exception('Missing KsL values')
        newmod = _dcopy(model)
        for idx_mag, mag in enumerate(skewidx):
            for idx_seg, seg in enumerate(mag):
                newmod[seg].KsL = ksl[idx_mag][idx_seg]
        return newmod

    def correct_coupling(self,
                         model,
                         matrix=None,
                         nsv=None,
                         niter=10,
                         tol=1e-6,
                         res0=None):
        """."""
        if matrix is None:
            matrix = self.calc_coupling_matrix(model)
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_matrix = -np.dot(np.dot(v.T, inv_s), u.T)
        if res0 is None:
            res = self.get_coupling_residue(model)
        else:
            res = res0
        bestfm = np.sqrt(np.sum(np.abs(res)**2)/res.size)
        ksl0 = self.get_ksl(model)
        ksl = ksl0

        for i in range(niter):
            dksl = np.dot(inv_matrix, res)
            ksl += np.reshape(dksl, (-1, 1))
            model = self.set_ksl(model=model, ksl=ksl)
            res = self.get_coupling_residue(model)
            fm = np.sqrt(np.sum(np.abs(res)**2)/res.size)
            diff_fm = np.abs(bestfm - fm)
            print(i, bestfm)
            if fm < bestfm:
                bestfm = fm
            if diff_fm < tol:
                break
        print('done!')
        return model
