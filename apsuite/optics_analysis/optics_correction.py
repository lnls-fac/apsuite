"""."""

from copy import deepcopy as _dcopy
from collections import namedtuple as _namedtuple
import numpy as np

import pyaccel
from pymodels import bo, si


class OpticsCorr():
    """."""

    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model, acc, dim='4d', knobs_list=None, method=None):
        """."""
        self.model = model
        self.acc = acc
        self.dim = dim
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

    def get_kl(self, model=None, knobsidx=None):
        """Return integrated quadrupoles strengths."""
        if model is None:
            model = self.model
        if knobsidx is None:
            knobsidx = self.knobs_idx
        kl = []
        for mag in knobsidx:
            kl_seg = []
            for seg in mag:
                kl_seg.append(model[seg].KL)
            kl.append(kl_seg)
        return np.array(kl)

    def set_kl(self, model=None, knobsidx=None, kl=None):
        """Set integrated quadrupoles strengths in the model."""
        if model is None:
            model = self.model
        if knobsidx is None:
            knobsidx = self.knobs_idx
        if kl is None:
            raise Exception('Missing KL values')
        newmod = _dcopy(model)
        for idx_mag, mag in enumerate(knobsidx):
            for idx_seg, seg in enumerate(mag):
                newmod[seg].KL = kl[idx_mag][idx_seg]
        return newmod

    @staticmethod
    def get_figm(res):
        """Calculate figure of merit from residue vector."""
        return np.sqrt(np.sum(np.abs(res)**2)/res.size)

    def optics_corr(self,
                    model,
                    goal_model=None,
                    jacobian_matrix=None,
                    nsv=None, nr_max=10, tol=1e-6):
        """Optics correction LOCO-like.

        Calculates the pseudo-inverse of optics correction matrix via SVD
        and minimizes the residue vector:
        Delta [Bx@BPM, Bx@CH, By@BPM, By@CV, Dx@BPM, Tunex, Tuney].
        """
        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        else:
            jmat = _dcopy(jacobian_matrix)

        nominal_stren = self.get_kl(model)
        if self._method == OpticsCorr.METHODS.Proportional:
            jmat *= nominal_stren.flatten()

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
        klcorr = nominal_stren
        dvec = goal_vec - vec
        bestfigm = OpticsCorr.get_figm(dvec)
        if bestfigm < tol:
            return OpticsCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dkl = np.dot(ijmat, dvec)
            dkl = np.reshape(dkl, (-1, 1))
            if self._method == OpticsCorr.METHODS.Proportional:
                klcorr *= (1 + dkl)
            else:
                klcorr += dkl
            model = self.set_kl(model=model, kl=klcorr)
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
