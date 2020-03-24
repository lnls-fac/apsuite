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

    SIDEFKNOBS = KnobTypes(
        ['QFA', 'QFB', 'QFP'],
        ['QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2'])
    BODEFKNOBS = KnobTypes(['QF', ], ['QD', ])

    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    GROUPING = _namedtuple('Grouping', ['Individual', 'TwoKnobs'])(0, 1)

    def __init__(self, model, acc, qf_knobs=None, qd_knobs=None,
                 method=None, grouping=None):
        """."""
        self.model = model
        self.acc = acc
        self._method = TuneCorr.METHODS.Proportional
        self._grouping = TuneCorr.GROUPING.TwoKnobs
        if acc == 'BO':
            qf_knobs = qf_knobs or TuneCorr.BODEFKNOBS.QFs
            qd_knobs = qd_knobs or TuneCorr.BODEFKNOBS.QDs
            self.knobs = KnobTypes(qf_knobs, qd_knobs)
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            qf_knobs = qf_knobs or TuneCorr.SIDEFKNOBS.QFs
            qd_knobs = qd_knobs or TuneCorr.SIDEFKNOBS.QDs
            self.knobs = KnobTypes(qf_knobs, qd_knobs)
            self.fam = si.get_family_data(model)
        self.method = method
        self.grouping = grouping

    @property
    def grouping(self):
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
        return TuneCorr.GROUPING._fields[self._grouping]

    @property
    def method(self):
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
        return TuneCorr.METHODS._fields[self._method]

    def get_tunes(self, model=None):
        """."""
        if model is None:
            model = self.model
        twinom, *_ = pyaccel.optics.calc_twiss(
            accelerator=model, indices='open')
        tunex = twinom.mux[-1]/2/np.pi
        tuney = twinom.muy[-1]/2/np.pi
        return tunex, tuney

    def calc_tune_matrix(self, model=None, acc=None):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        tune_matrix = np.zeros((2, len(self.knobs.ALL)))
        nux0, nuy0 = self.get_tunes(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.ALL):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    if self._method == TuneCorr.METHODS.Additional:
                        modcopy[seg].KL += delta/len(nmag)
                    else:
                        modcopy[seg].KL *= 1 + delta
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

    def set_deltakl(self,
                    model=None,
                    deltakl=None):
        """."""
        if model is None:
            model = self.model
        if deltakl is None:
            raise Exception('Missing KL values')
        if group and deltakl.size > 2:
            raise Exception(
                'Grouping option requires only 2 delta KL values')
        if method not in ['proportional', 'additional']:
            raise Exception('Invalid correction method!')
        mod = model[:]

        if group:
            kl_qf = self.get_kl(mod, knobs=self.knobs.QFs)
            kl_qd = self.get_kl(mod, knobs=self.knobs.QDs)
            kl_qfsum = np.sum(kl_qf)
            kl_qdsum = np.sum(kl_qd)
        factor = 1
        for idx_knb, knb in enumerate(self.knobs.ALL):
            if group:
                if knb in self.knobs.QFs:
                    idx = self.knobs.QFs.index(knb)
                    delta = deltakl[0]
                    factor = kl_qf[idx]/kl_qfsum
                elif knb in self.knobs.QDs:
                    idx = self.knobs.QDs.index(knb)
                    delta = deltakl[1]
                    factor = kl_qd[idx]/kl_qdsum
            else:
                delta = deltakl[idx_knb]
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    if method == 'proportional':
                        mod[seg].KL *= (1 + delta/len(mag))
                    elif method == 'additional':
                        mod[seg].KL += delta/len(mag) * factor
        return mod

    def correct_tunes(self,
                      model,
                      tunex, tuney,
                      tune_matrix=None,
                      tol=1e-6,
                      nr_max=10,
                      nsv=None):
        """."""
        if method not in ['proportional', 'additional']:
            raise Exception('Invalid correction method!')

        if tune_matrix is None:
            tunemat = self.calc_tune_matrix(model)
        else:
            tunemat = _dcopy(tune_matrix)

        nominal_kl = self.get_kl(model)
        if method == 'proportional':
            tunemat *= nominal_kl
        if group:
            dkl = np.zeros(2)
            tunemat = self._group_2knobs_matrix(tunemat)
            if method == 'additional':
                tunemat[:, 0] /= len(self.knobs.QFs)
                tunemat[:, 1] /= len(self.knobs.QDs)
        else:
            dkl = np.zeros(nominal_kl.shape)

        mod = model[:]
        U, S, V = np.linalg.svd(tunemat, full_matrices=False)
        iS = 1/S
        iS[np.isnan(iS)] = 0
        iS[np.isinf(iS)] = 0
        if nsv is not None:
            iS[nsv:] = 0
        iS = np.diag(iS)
        invmat = -1 * np.dot(np.dot(V.T, iS), U.T)
        tunex0, tuney0 = self.get_tunes(mod)
        print(tunex0, tuney0)
        tunex_new, tuney_new = tunex0, tuney0

        for _ in range(nr_max):
            mod = model[:]
            dtune = [tunex_new-tunex, tuney_new-tuney]
            dkl += np.dot(invmat, dtune)
            mod = self.set_deltakl(
                model=mod, deltakl=dkl, method=method, group=group)
            tunex_new, tuney_new = self.get_tunes(mod)
            print(tunex_new, tuney_new)
            if abs(tunex_new - tunex) < tol and abs(tuney_new - tuney) < tol:
                break
        print('done!')
        return mod

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
