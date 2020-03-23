"""."""

from copy import deepcopy as _dcopy
import numpy as np
from pymodels import bo, si
import pyaccel


class TuneCorr():
    """."""

    SIKNOBS = ['QFA', 'QFB', 'QFP', 'QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2']
    BOKNOBS = ['QF', 'QD']

    def __init__(self, model, acc, knobs_names=None):
        """."""
        self.model = model
        self.acc = acc
        if acc == 'BO':
            if knobs_names is None:
                self.knobs = self.BOKNOBS
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            if knobs_names is None:
                self.knobs = self.SIKNOBS
            self.fam = si.get_family_data(model)
        self.tune_matrix = []
        self.focusing_knobs = []
        self.defocusing_knobs = []
        self._group_knobs()
        self.tunex, self.tuney = self.get_tunes()

    def get_tunes(self, model=None):
        """."""
        if model is None:
            model = self.model
        twinom, *_ = pyaccel.optics.calc_twiss(
            accelerator=model, indices='open')
        tunex = twinom.mux[-1]/2/np.pi
        tuney = twinom.muy[-1]/2/np.pi
        return tunex, tuney

    def _group_knobs(self):
        for knb in self.knobs:
            if 'QF' in knb:
                self.focusing_knobs.append(knb)
            else:
                self.defocusing_knobs.append(knb)

    def calc_tune_matrix(self, model=None, acc=None):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        self.tune_matrix = np.zeros((2, len(self.knobs)))

        delta = 1e-3
        for idx, knb in enumerate(self.knobs):
            modcopy = _dcopy(model)
            for nmag in self.fam[knb]['index']:
                dlt = delta/len(nmag)
                for seg in nmag:
                    modcopy[seg].KL += dlt
            tunex, tuney = self.get_tunes(model=modcopy)
            self.tune_matrix[:, idx] = [
                (tunex - self.tunex)/dlt, (tuney - self.tuney)/dlt]
        return self.tune_matrix

    def calc_group_matrix(self, tune_matrix=None):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(self.model)

        tune_group_matrix = np.zeros(tune_matrix.shape)
        nfocus = len(self.focusing_knobs)

        for fnum, knbf in enumerate(self.focusing_knobs):
            idxf = self.knobs.index(knbf)
            tune_group_matrix[:, fnum] += tune_matrix[:, idxf]
        for dfnum, knbdf in enumerate(self.defocusing_knobs):
            idxdf = self.knobs.index(knbdf)
            tune_group_matrix[:, dfnum + nfocus] += tune_matrix[:, idxdf]
        return tune_group_matrix

    def calc_group_2knobs_matrix(self, tune_matrix=None):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(self.model)

        tune_group_2knobs_matrix = np.zeros((2, 2))

        for knbf in self.focusing_knobs:
            idxf = self.knobs.index(knbf)
            tune_group_2knobs_matrix[:, 0] += tune_matrix[:, idxf]
        for knbdf in self.defocusing_knobs:
            idxdf = self.knobs.index(knbdf)
            tune_group_2knobs_matrix[:, 1] += tune_matrix[:, idxdf]
        return tune_group_2knobs_matrix

    def get_kl(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs
        kl = []
        for knb in knobs:
            kl_mag = []
            for mag in self.fam[knb]['index']:
                kl_seg = []
                for seg in mag:
                    kl_seg.append(model[seg].KL)
                kl_mag.append(kl_seg)
            kl.append(np.mean(kl_mag))
        return np.array(kl)

    def get_kl_group(self, kl):
        """."""
        kl_group = np.zeros(kl.shape)
        nfocus = len(self.focusing_knobs)

        for fnum, knbf in enumerate(self.focusing_knobs):
            idxf = self.knobs.index(knbf)
            kl_group[fnum] = kl[idxf]
        for dfnum, knbdf in enumerate(self.defocusing_knobs):
            idxdf = self.knobs.index(knbdf)
            kl_group[dfnum + nfocus] = kl[idxdf]
        return kl_group

    def set_kl(self, model=None, knobs=None, kl=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs
        if kl is None:
            raise Exception('Missing KL values')
        mod = model[:]
        for idx_knb, knb in enumerate(knobs):
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    mod[seg].KL = kl[idx_knb]/len(mag)
        return mod

    def set_kl_group(self, model=None, kl=None):
        """."""
        if model is None:
            model = self.model
        if kl is None:
            raise Exception('Missing KL values')
        mod = model[:]
        nfocus = len(self.focusing_knobs)
        for idx_knbf, knbf in enumerate(self.focusing_knobs):
            for mag in self.fam[knbf]['index']:
                for seg in mag:
                    mod[seg].KL = kl[idx_knbf]/len(mag)
        for idx_knbdf, knbdf in enumerate(self.defocusing_knobs):
            for mag in self.fam[knbdf]['index']:
                for seg in mag:
                    mod[seg].KL = kl[idx_knbdf + nfocus]/len(mag)
        return mod

    def correct_tunes(self,
                      model,
                      tunex, tuney,
                      tune_matrix=None,
                      tol=1e-6,
                      nr_max=10,
                      nsv=None,
                      method='proportional'):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(model)

        if method == 'proportional':
            nominal_kl = self.get_kl_group(model)
            tune_matrix *= nominal_kl
        elif method == 'additional':
            nominal_kl = self.get_kl(model)

        mod = model[:]
        u, s, v = np.linalg.svd(tune_matrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        if nsv is not None:
            inv_s[nsv:] = 0
        inv_s = np.diag(inv_s)
        inv_matrix = np.dot(np.dot(v.T, inv_s), u.T)
        tunex0, tuney0 = self.get_tunes(mod)
        print(tunex0, tuney0)
        tunex_new, tuney_new = tunex0, tuney0
        dkl = np.zeros(nominal_kl.shape)

        for _ in range(nr_max):
            dtune = [tunex-tunex_new, tuney-tuney_new]
            dkl += np.dot(inv_matrix, dtune)
            if method == 'proportional':
                newkl = nominal_kl * (1 + dkl)
                mod = self.set_kl_group(model=mod, kl=newkl)
            elif method == 'additional':
                newkl = nominal_kl + dkl
                mod = self.set_kl(model=mod, kl=newkl)
            tunex_new, tuney_new = self.get_tunes(mod)
            print(tunex_new, tuney_new)
            if abs(tunex_new - tunex) < tol and abs(tuney_new - tuney) < tol:
                break
        print('done!')
        return mod
