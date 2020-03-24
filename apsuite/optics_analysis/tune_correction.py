"""."""

from collections import namedtuple as _namedtuple
from copy import deepcopy as _dcopy
import numpy as np
from pymodels import bo, si
import pyaccel

__KNOBTYPES = _namedtuple('KnobTypes', ['QFs', 'QDs'])


class KnobTypes(__KNOBTYPES):
    """."""

    @property
    def ALL(self):
        """."""
        return self.QFs + self.QDs


class TuneCorr():
    """."""

    SI_DEF_KNOBS = KnobTypes(
        ['QFA', 'QFB', 'QFP'],
        ['QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2'])
    BO_DEF_KNOBS = KnobTypes(['QF', ], ['QD', ])

    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    GROUPING = _namedtuple('Grouping', ['Individual', 'TwoKnobs'])(0, 1)
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model, acc, qf_knobs=None, qd_knobs=None,
                 method=None, grouping=None):
        """."""
        self.model = model
        self.acc = acc
        self._method = TuneCorr.METHODS.Proportional
        self._grouping = TuneCorr.GROUPING.TwoKnobs
        if acc == 'BO':
            qf_knobs = qf_knobs or TuneCorr.BO_DEF_KNOBS.QFs
            qd_knobs = qd_knobs or TuneCorr.BO_DEF_KNOBS.QDs
            self.knobs = KnobTypes(qf_knobs, qd_knobs)
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            qf_knobs = qf_knobs or TuneCorr.SI_DEF_KNOBS.QFs
            qd_knobs = qd_knobs or TuneCorr.SI_DEF_KNOBS.QDs
            self.knobs = KnobTypes(qf_knobs, qd_knobs)
            self.fam = si.get_family_data(model)
        self.method = method
        self.grouping = grouping

    def __str__(self):
        """."""
        strg = '{0:25s}= {1:s}\n'.format(
            'focusing quadrupoles', str(self.knobs.QFs))
        strg += '{0:25s}= {1:s}\n'.format(
            'defocusing quadrupoles', str(self.knobs.QDs))
        strg += '{0:25s}= {1:30s}\n'.format(
            'correction method', self.method_str)
        strg += '{0:25s}= {1:30s}\n'.format(
            'grouping', self.grouping_str)
        return strg

    @property
    def grouping(self):
        """."""
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        if value is None:
            return
        elif isinstance(value, str):
            self._grouping = int(value in TuneCorr.GROUPING._fields[1])
        elif int(value) in TuneCorr.GROUPING:
            self._grouping = int(value)

    @property
    def grouping_str(self):
        """."""
        return TuneCorr.GROUPING._fields[self._grouping]

    @property
    def method(self):
        """."""
        return self._method

    @method.setter
    def method(self, value):
        if value is None:
            return
        elif isinstance(value, str):
            self._method = int(value in TuneCorr.METHODS._fields[1])
        elif int(value) in TuneCorr.METHODS:
            self._method = int(value)

    @property
    def method_str(self):
        """."""
        return TuneCorr.METHODS._fields[self._method]

    def get_tunes(self, model=None):
        """."""
        if model is None:
            model = self.model
        twinom, *_ = pyaccel.optics.calc_twiss(
            accelerator=model, indices='open')
        nux = twinom.mux[-1]/2/np.pi
        nuy = twinom.muy[-1]/2/np.pi
        return nux, nuy

    def calc_tune_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        tune_matrix = np.zeros((2, len(self.knobs.ALL)))
        nux0, nuy0 = self.get_tunes(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.ALL):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    modcopy[seg].KL += delta/len(nmag)
            nux, nuy = self.get_tunes(model=modcopy)
            tune_matrix[:, idx] = [
                (nux - nux0)/delta, (nuy - nuy0)/delta]
        return tune_matrix

    def get_kl(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs.ALL
        kl = []
        for knb in knobs:
            kl_mag = []
            for mag in self.fam[knb]['index']:
                kl_seg = []
                for seg in mag:
                    kl_seg.append(model[seg].KL)
                kl_mag.append(sum(kl_seg))
            kl.append(np.mean(kl_mag))
        return np.array(kl)

    def correct_tunes(self,
                      nux, nuy,
                      model=None,
                      tune_matrix=None,
                      tol=1e-6,
                      nr_max=10,
                      nsv=None):
        """."""
        if model is None:
            model = self.model
        if self.method not in TuneCorr.METHODS:
            raise Exception('Invalid correction method!')

        if tune_matrix is None:
            tunemat = self.calc_tune_matrix(model)
        else:
            tunemat = _dcopy(tune_matrix)

        nominal_kl = self.get_kl(model)
        if self._method == TuneCorr.METHODS.Proportional:
            tunemat *= nominal_kl
        if self._grouping == TuneCorr.GROUPING.TwoKnobs:
            dkl = np.zeros(2)
            tunemat = self._group_2knobs_matrix(tunemat)
        else:
            dkl = np.zeros(nominal_kl.shape)

        U, S, V = np.linalg.svd(tunemat, full_matrices=False)
        iS = 1/S
        iS[np.isnan(iS)] = 0
        iS[np.isinf(iS)] = 0
        if nsv is not None:
            iS[nsv:] = 0
        iS = np.diag(iS)
        invmat = -1 * np.dot(np.dot(V.T, iS), U.T)
        nux0, nuy0 = self.get_tunes(model)
        nux_new, nuy_new = nux0, nuy0

        for _ in range(nr_max):
            dtune = [nux_new-nux, nuy_new-nuy]
            dkl = np.dot(invmat, dtune)
            self._add_deltakl(dkl, model=model)
            nux_new, nuy_new = self.get_tunes(model)
            if abs(nux_new - nux) < tol and abs(nuy_new - nuy) < tol:
                break
        else:
            return TuneCorr.CORR_STATUS.Fail
        return TuneCorr.CORR_STATUS.Sucess

    def _add_deltakl(self, deltakl, model=None):
        """."""
        if model is None:
            model = self.model

        for idx_knb, knb in enumerate(self.knobs.ALL):
            if self._grouping == TuneCorr.GROUPING.TwoKnobs:
                if knb in self.knobs.QFs:
                    delta = deltakl[0]
                elif knb in self.knobs.QDs:
                    delta = deltakl[1]
            else:
                delta = deltakl[idx_knb]
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    if self._method == TuneCorr.METHODS.Proportional:
                        model[seg].KL *= (1 + delta/len(mag))
                    else:
                        model[seg].KL += delta/len(mag)

    def _group_2knobs_matrix(self, tune_matrix=None):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(self.model)

        tune_2knobs_matrix = np.zeros((2, 2))
        nfocus = len(self.knobs.QFs)

        for nf, _ in enumerate(self.knobs.QFs):
            tune_2knobs_matrix[:, 0] += tune_matrix[:, nf]
        for ndf, _ in enumerate(self.knobs.QDs):
            tune_2knobs_matrix[:, 1] += tune_matrix[:, ndf+nfocus]
        return tune_2knobs_matrix
