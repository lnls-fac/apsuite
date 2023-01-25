"""."""

from copy import deepcopy as _dcopy
from mathphys.functions import get_namedtuple as _get_namedtuple
import numpy as _np
import multiprocessing as _mp

import pyaccel

from ..orbcorr import OrbRespmat


class CouplingCorr():
    """."""

    CORR_STATUS = _get_namedtuple('CorrStatus', ['Fail', 'Sucess'])
    CORR_METHODS = _get_namedtuple('CorrMethods', ['Orbrespm'])

    def __init__(self, model, acc, dim='4d',
                 skew_list=None, correction_method=None):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
        self._corr_method = CouplingCorr.CORR_METHODS.Orbrespm
        self.coup_matrix = []
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self.bpm_idx = self.respm.fam_data['BPM']['index']
        if skew_list is None:
            self.skew_idx = self.respm.fam_data['QS']['index']
        else:
            self.skew_idx = skew_list
        self._corr_method = correction_method
        self._freq = None
        self._alpha = None

    @property
    def corr_method(self):
        """."""
        return self._corr_method

    @corr_method.setter
    def corr_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._corr_method = int(
                value not in CouplingCorr.CORR_METHODS._fields[0])
        elif int(value) in CouplingCorr.CORR_METHODS:
            self._corr_method = int(value)

    @property
    def corr_method_str(self):
        """."""
        return CouplingCorr.CORR_METHODS._fields[self._corr_method]

    @property
    def _nbpm(self):
        return len(self.bpm_idx)

    @property
    def _nch(self):
        return len(self.respm.fam_data['CH']['index'])

    @property
    def _ncv(self):
        return len(self.respm.fam_data['CV']['index'])

    @property
    def _nskew(self):
        return len(self.skew_idx)

    def _calc_jacobian_matrix_idx(
            self, model=None, indices=None, weight_dispy=1):
        """."""
        if model is None:
            model = self.model

        if indices is None:
            indices = self.skew_idx

        nvec = self._nbpm * (self._nch + self._ncv + 1)
        coup_matrix = _np.zeros((nvec, len(indices)))
        delta = 1e-6

        modcopy = _dcopy(model)
        for idx, nmag in enumerate(indices):
            dlt = delta/len(nmag)
            for seg in nmag:
                modcopy[seg].KsL += dlt
                elem = self._get_coupling_residue(modcopy, weight_dispy) / dlt
                modcopy[seg].KsL -= dlt
            coup_matrix[:, idx] = elem
        return coup_matrix

    def calc_jacobian_matrix(self, model=None, weight_dispy=1):
        """Coupling correction response matrix.

        Calculates the variation of off-diagonal elements of orbit response
        matrix and vertical dispersion wrt variations of KsL.
        """
        if model is None:
            model = self.model
        self.coup_matrix = self._parallel_base(
            model, weight_dispy=weight_dispy)
        return self.coup_matrix

    def _get_coupling_residue(self, model, weight_dispy=1):
        if None in {self._freq, self._alpha}:
            self._freq = model[self.respm.rf_idx[0]].frequency
            self._alpha = pyaccel.optics.get_mcf(model)
        self.respm.model = model
        orbmat = self.respm.get_respm()
        mxy = orbmat[:self._nbpm, self._nch:-1].ravel()
        myx = orbmat[self._nbpm:, :self._nch].ravel()
        dispy = orbmat[self._nbpm:, -1]*weight_dispy
        dispy *= -self._freq*self._alpha
        res = _np.r_[mxy, myx, dispy]
        return res

    def _parallel_base(self, model, weight_dispy=1):
        slcs = self._get_slices_multiprocessing(True, len(self.skew_idx))
        with _mp.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(
                    self._calc_jacobian_matrix_idx,
                    (model, self.skew_idx[slc], weight_dispy)))
            mat = [re.get() for re in res]
        mat = _np.concatenate(mat, axis=1)
        return mat

    def _get_slices_multiprocessing(self, npart):
        nrproc = _mp.cpu_count() - 3
        nrproc = max(nrproc, 1)
        nrproc = min(nrproc, npart)

        np_proc = (npart // nrproc)*_np.ones(nrproc, dtype=int)
        np_proc[:(npart % nrproc)] += 1
        parts_proc = _np.r_[0, _np.cumsum(np_proc)]
        return [slice(parts_proc[i], parts_proc[i+1]) for i in range(nrproc)]

    def get_ksl(self, model=None, skewidx=None):
        """Return skew quadrupoles strengths."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        ksl_mag = []
        for mag in skewidx:
            ksl_seg = []
            for seg in mag:
                ksl_seg.append(model[seg].KsL)
            ksl_mag.append(sum(ksl_seg))
        return _np.array(ksl_mag)

    def _set_delta_ksl(self, model=None, skewidx=None, delta_ksl=None):
        """Set skew quadrupoles strengths in the model."""
        if model is None:
            model = self.model
        if skewidx is None:
            skewidx = self.skew_idx
        if delta_ksl is None:
            raise Exception('Missing Delta KsL values')
        for idx_mag, mag in enumerate(skewidx):
            delta = delta_ksl[idx_mag]/len(mag)
            for seg in mag:
                model[seg].KsL += delta

    @staticmethod
    def get_figm(res):
        """Calculate figure of merit from residue vector."""
        return _np.sqrt(_np.sum(res*res)/res.size)

    def coupling_corr_orbrespm_dispy(self,
                                     model,
                                     jacobian_matrix=None,
                                     nsv=None, nr_max=10, tol=1e-6,
                                     res0=None, weight_dispy=1):
        """Coupling correction with orbrespm.

        Calculates the pseudo-inverse of coupling correction matrix via SVD
        and minimizes the residue vector [Mxy, Myx, Etay].
        """
        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        umat, smat, vmat = _np.linalg.svd(jmat, full_matrices=False)
        ismat = 1/smat
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ismat = _np.diag(ismat)
        ijmat = -_np.dot(_np.dot(vmat.T, ismat), umat.T)
        if res0 is None:
            res = self._get_coupling_residue(model, weight_dispy=weight_dispy)
        else:
            res = res0
        bestfigm = CouplingCorr.get_figm(res)
        if bestfigm < tol:
            return CouplingCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dksl = _np.dot(ijmat, res)
            self._set_delta_ksl(model=model, delta_ksl=dksl)
            res = self._get_coupling_residue(
                model, weight_dispy=weight_dispy)
            figm = CouplingCorr.get_figm(res)
            diff_figm = _np.abs(bestfigm - figm)
            if figm < bestfigm:
                bestfigm = figm
            if diff_figm < tol:
                break
        else:
            return CouplingCorr.CORR_STATUS.Fail
        return CouplingCorr.CORR_STATUS.Sucess

    def coupling_correction(self,
                            model,
                            jacobian_matrix=None,
                            nsv=None, nr_max=10, tol=1e-6,
                            res0=None, weight_dispy=1):
        """Coupling correction method selection.

        Methods available:
        - Minimization of off-diagonal elements of orbit response matrix and
        vertical dispersion.
        """
        if self.corr_method == CouplingCorr.CORR_METHODS.Orbrespm:
            result = self.coupling_corr_orbrespm_dispy(
                model=model, jacobian_matrix=jacobian_matrix,
                nsv=nsv, nr_max=nr_max, tol=tol, res0=res0,
                weight_dispy=weight_dispy)
        else:
            raise Exception('Chosen method is not implemented!')
        return result
