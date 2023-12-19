"""."""

from copy import deepcopy as _dcopy
from mathphys.functions import get_namedtuple as _get_namedtuple
import numpy as np

import pyaccel
from pymodels import bo, si


class OpticsCorr():
    """."""

    METHODS = _get_namedtuple('Methods', ['Additional', 'Proportional'])
    CORR_STATUS = _get_namedtuple('CorrStatus', ['Fail', 'Sucess'])
    CORR_METHODS = _get_namedtuple('CorrMethods', ['LOCO'])

    def __init__(self, model, acc, dim='4d', knobs_list=None,
                 method=None, correction_method=None):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
        self._corr_method = OpticsCorr.CORR_METHODS.LOCO
        self._method = OpticsCorr.METHODS.Proportional
        self.jacobian_matrix = []
        if self.acc == 'BO':
            self.fam_data = bo.families.get_family_data(self.model)
        elif self.acc == 'SI':
            self.fam_data = si.families.get_family_data(self.model)
        else:
            raise Exception('Set models: BO or SI')
        if knobs_list is None:
            self.knobs_idx = self.fam_data['QN']['index']
        else:
            self.knobs_idx = knobs_list
        self.bpm_idx = self._get_idx(self.fam_data['BPM']['index'])
        self.ch_idx = self._get_idx(self.fam_data['CH']['index'])
        self.cv_idx = self._get_idx(self.fam_data['CV']['index'])
        self.method = method
        self.corr_method = correction_method

    @property
    def method(self):
        """."""
        return self._method

    @method.setter
    def method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._method = int(value in OpticsCorr.METHODS._fields[1])
        elif int(value) in OpticsCorr.METHODS:
            self._method = int(value)

    @property
    def method_str(self):
        """."""
        return OpticsCorr.METHODS._fields[self._method]

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
                value not in OpticsCorr.CORR_METHODS._fields[0])
        elif int(value) in OpticsCorr.CORR_METHODS:
            self._corr_method = int(value)

    @property
    def corr_method_str(self):
        """."""
        return OpticsCorr.CORR_METHODS._fields[self._corr_method]

    @staticmethod
    def _get_idx(indcs):
        return np.array([idx[0] for idx in indcs])

    def calc_jacobian_matrix(self, model=None):
        """Optics correction response matrix.

        Calculates the variation of Twiss functions and tune given the
        variation of integrated quadrupoles strength.
        """
        if model is None:
            model = self.model

        res0 = self._get_optics_vector(model)
        self.jacobian_matrix = np.zeros((res0.size, len(self.knobs_idx)))
        delta = 1e-6

        modcopy = model[:]
        for idx, nmag in enumerate(self.knobs_idx):
            dlt = delta/len(nmag)
            for seg in nmag:
                modcopy[seg].KL += dlt
            res = self._get_optics_vector(modcopy)
            self.jacobian_matrix[:, idx] = (res-res0)/delta
            for seg in nmag:
                modcopy[seg].KL -= dlt
        return self.jacobian_matrix

    def _get_optics_vector(self, model):
        twi, *_ = pyaccel.optics.calc_twiss(model)
        w_beta = 1e2  # order of centimeters
        w_disp = 1e4  # order of tenth of milimeters
        w_tune = 1e4  # order of 10^-2

        betax_bpm = twi.betax[self.bpm_idx]
        betax_ch = twi.betax[self.ch_idx]
        betay_bpm = twi.betay[self.bpm_idx]
        betay_cv = twi.betay[self.cv_idx]
        etax_bpm = twi.etax[self.bpm_idx]
        tunex = twi.mux[-1]/2/np.pi
        tuney = twi.muy[-1]/2/np.pi

        vec_bx = np.hstack((betax_bpm, betax_ch))
        vec_by = np.hstack((betay_bpm, betay_cv))
        vec_dx = etax_bpm
        vec_tune = np.hstack((tunex, tuney))

        vec = vec_bx * w_beta / vec_bx.size
        vec = np.hstack((vec, vec_by * w_beta / vec_by.size))
        vec = np.hstack((vec, vec_dx * w_disp / vec_dx.size))
        vec = np.hstack((vec, vec_tune * w_tune / vec_tune.size))
        return vec

    def _get_kl(self, model=None, knobsidx=None):
        """Return integrated quadrupoles strengths."""
        if model is None:
            model = self.model
        if knobsidx is None:
            knobsidx = self.knobs_idx
        kl_mag = []
        for mag in knobsidx:
            kl_seg = []
            for seg in mag:
                kl_seg.append(model[seg].KL)
            kl_mag.append(sum(kl_seg))
        return np.array(kl_mag)

    def _set_delta_kl(self, model=None, knobsidx=None, deltas_kl=None):
        """Set integrated quadrupoles strengths in the model."""
        if model is None:
            model = self.model
        if knobsidx is None:
            knobsidx = self.knobs_idx
        if deltas_kl is None:
            raise Exception('Missing Delta KL values')
        for idx_mag, mag in enumerate(knobsidx):
            delta = deltas_kl[idx_mag]/len(mag)
            for seg in mag:
                stren = model[seg].KL
                if self.method == OpticsCorr.METHODS.Proportional:
                    stren *= (1 + delta)
                else:
                    stren += delta
                model[seg].KL = stren

    @staticmethod
    def get_figm(res):
        """Calculate figure of merit from residue vector."""
        return np.sqrt(np.sum(res*res)/res.size)

    def optics_corr_loco(self,
                         model=None,
                         goal_model=None,
                         jacobian_matrix=None,
                         nsv=None, nr_max=10, tol=1e-6):
        """Optics correction LOCO-like.

        Calculates the pseudo-inverse of optics correction matrix via SVD
        and minimizes the residue vector:
        Delta [Bx@BPM, Bx@CH, By@BPM, By@CV, Dx@BPM, Tunex, Tuney].
        """
        if model is None:
            model = self.model
        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        else:
            jmat = _dcopy(jacobian_matrix)

        nominal_stren = self._get_kl(model)
        if self.method == OpticsCorr.METHODS.Proportional:
            jmat *= nominal_stren.ravel()

        umat, smat, vmat = np.linalg.svd(jmat, full_matrices=False)
        ismat = 1/smat
        ismat[np.isnan(ismat)] = 0
        ismat[np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ismat = np.diag(ismat)
        ijmat = np.dot(np.dot(vmat.T, ismat), umat.T)
        goal_vec = self._get_optics_vector(goal_model)
        vec = self._get_optics_vector(model)
        dvec = goal_vec - vec
        bestfigm = OpticsCorr.get_figm(dvec)
        if bestfigm < tol:
            return OpticsCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dkl = np.dot(ijmat, dvec)
            self._set_delta_kl(model=model, deltas_kl=dkl)
            vec = self._get_optics_vector(model)
            dvec = goal_vec - vec
            figm = OpticsCorr.get_figm(dvec)
            if figm < bestfigm:
                bestfigm = figm
            if bestfigm < tol:
                break
        else:
            return OpticsCorr.CORR_STATUS.Fail
        return OpticsCorr.CORR_STATUS.Sucess

    def optics_correction(self,
                          model=None,
                          goal_model=None,
                          jacobian_matrix=None,
                          nsv=None, nr_max=10, tol=1e-6):
        """Optics correction method selection.

        Methods available:
        - LOCO-like correction
        """
        if model is None:
            model = self.model
        if self.corr_method == OpticsCorr.CORR_METHODS.LOCO:
            result = self.optics_corr_loco(
                model=model, goal_model=goal_model,
                jacobian_matrix=jacobian_matrix, nsv=nsv, nr_max=nr_max,
                tol=tol)
        else:
            raise Exception('Chosen method is not implemented!')
        return result
