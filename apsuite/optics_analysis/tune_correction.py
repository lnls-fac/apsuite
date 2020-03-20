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

    def calc_tune_matrix(self, model=None, acc=None):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        self.tune_matrix = np.zeros((2, len(self.knobs)))

        delta = 1e-2
        for idx, q in enumerate(self.knobs):
            modcopy = _dcopy(model)
            for nmag in self.fam[q]['index']:
                dlt = delta/len(nmag)
                for seg in nmag:
                    modcopy[seg].KL += dlt
            tunex, tuney = self.get_tunes(model=modcopy)
            self.tune_matrix[:, idx] = [
                (tunex - self.tunex)/dlt, (tuney - self.tuney)/dlt]
        return self.tune_matrix

    def get_kl(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs
        kl = []
        for knb in knobs:
            kl.append(model[self.fam[knb]['index'][0][0]].KL)
        return kl

    def set_kl(self, model=None, knobs=None, kl=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs
        if kl is None:
            raise Exception('Missing KL values')
        newmod = _dcopy(model)
        for idx, knb in enumerate(knobs):
            newmod[self.fam[knb]['index'][0][0]].KL = kl[idx]
        return newmod

    def change_tunes(self, model, tunex, tuney, tune_matrix=None):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(model)
        u, s, v = np.linalg.svd(tune_matrix, full_matrices=False)
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        inv_s = np.diag(inv_s)
        inv_matrix = np.dot(np.dot(v.T, inv_s), u.T)
        tunex0, tuney0 = self.get_tunes(model)
        print(tunex0, tuney0)
        tunex_new, tuney_new = tunex0, tuney0
        kl = self.get_kl(model)
        tol = 1e-6

        while abs(tunex_new - tunex) > tol or abs(tuney_new - tuney) > tol:
            dtune = [tunex-tunex_new, tuney-tuney_new]
            dkl = np.dot(inv_matrix, dtune)
            kl += dkl
            model = self.set_kl(model=model, kl=kl)
            tunex_new, tuney_new = self.get_tunes(model)
            print(tunex_new, tuney_new)
        print('done!')
        return model
