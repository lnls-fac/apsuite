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
        self._corr_method = None
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self.bpm_idx = self.respm.fam_data['BPM']['index']
        if skew_list is None:
            self.skew_idx = self.respm.fam_data['QS']['index']
        else:
            self.skew_idx = skew_list
        self.corr_method = correction_method or \
            CouplingCorr.CORR_METHODS.Orbrespm
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

    @staticmethod
    def _calc_jacobian_matrix_idx(indices, *args):
        """."""
        coup_matrix = []
        delta = 1e-6
        model, *_ = args
        for nmag in indices:
            dlt = delta/len(nmag)
            for seg in nmag:
                model[seg].KsL += dlt
                elem = CouplingCorr._get_coupling_residue(*args)
                elem /= dlt
                model[seg].KsL -= dlt
            coup_matrix.append(elem)
        return _np.array(coup_matrix).T

    def calc_jacobian_matrix(self, model=None, weight_dispy=1):
        """Coupling correction response matrix.

        Calculates the variation of off-diagonal elements of orbit response
        matrix and vertical dispersion wrt variations of KsL.
        """
        if model is None:
            model = self.model
        if None in {self._freq, self._alpha}:
            self._freq = model[self.respm.rf_idx[0]].frequency
            self._alpha = pyaccel.optics.get_mcf(model)
        coup_matrix = CouplingCorr._parallel_base(
            self.skew_idx, model, self.respm, self._freq, self._alpha,
            self._nbpm, self._nch, weight_dispy)
        return coup_matrix

    @staticmethod
    def _get_coupling_residue(*args):
        model, obj_respm, freq, alpha, nbpm, nch, weight_dispy = args
        obj_respm.model = model
        orbmat = obj_respm.get_respm()
        mxy = orbmat[:nbpm, nch:-1].ravel()
        myx = orbmat[nbpm:, :nch].ravel()
        dispy = orbmat[nbpm:, -1]*weight_dispy
        dispy *= -freq*alpha
        res = _np.r_[mxy, myx, dispy]
        return res

    @staticmethod
    def _parallel_base(*args):
        idcs, model, obj_respm, freq, alpha, nbpm, nch, weight_dispy = args
        slcs = CouplingCorr._get_slices_multiprocessing(len(idcs))
        with _mp.Pool(processes=len(slcs)) as pool:
            res = []
            for slc in slcs:
                res.append(pool.apply_async(
                    CouplingCorr._calc_jacobian_matrix_idx,
                    (idcs[slc], model, obj_respm, freq,
                        alpha, nbpm, nch, weight_dispy)))
            mat = [re.get() for re in res]
        mat = _np.concatenate(mat, axis=1)
        return mat

    @staticmethod
    def _get_slices_multiprocessing(npart):
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
                                     model=None,
                                     jacobian_matrix=None,
                                     nsv=None, nr_max=10, tol=1e-6,
                                     res0=None, weight_dispy=1):
        """Coupling correction with orbrespm.

        Calculates the pseudo-inverse of coupling correction matrix via SVD
        and minimizes the residue vector [Mxy, Myx, weight*Etay].
        """
        self.model = model or self.model
        jac = jacobian_matrix
        jac = jac if jac is not None else self.calc_jacobian_matrix()
        umat, smat, vmat = _np.linalg.svd(jac, full_matrices=False)
        ismat = 1/smat
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ijmat = vmat.T @ _np.diag(ismat) @ umat.T
        if res0 is None:
            res = CouplingCorr._get_coupling_residue(
                model, self.respm, self._freq,
                self._alpha, self._nbpm, self._nch, weight_dispy)
        else:
            res = res0
        bestfigm = CouplingCorr.get_figm(res)
        if bestfigm < tol:
            return CouplingCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dksl = -ijmat @ res
            self._set_delta_ksl(model=model, delta_ksl=dksl)
            res = CouplingCorr._get_coupling_residue(
                model, self.respm, self._freq,
                self._alpha, self._nbpm, self._nch, weight_dispy)
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
                            model=None,
                            jacobian_matrix=None,
                            nsv=None, nr_max=10, tol=1e-6,
                            res0=None, weight_dispy=1):
        """Coupling correction method selection.

        Methods available:
        - Minimization of off-diagonal elements of orbit response matrix and
        vertical dispersion.
        """
        if self.corr_method == CouplingCorr.CORR_METHODS.Orbrespm:
            model = model or self.model
            result = self.coupling_corr_orbrespm_dispy(
                model=model, jacobian_matrix=jacobian_matrix,
                nsv=nsv, nr_max=nr_max, tol=tol, res0=res0,
                weight_dispy=weight_dispy)
        else:
            raise Exception('Chosen method is not implemented!')
        return result
