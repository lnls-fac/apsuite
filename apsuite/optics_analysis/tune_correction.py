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
                self.knobs_names = self.BOKNOBS
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            if knobs_names is None:
                self.knobs_names = self.SIKNOBS
            self.fam = si.get_family_data(model)
        self.tune_matrix = []
        self.focusing_knobs = []
        self.defocusing_knobs = []
        self.all_knobs = []
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
        for knb in self.knobs_names:
            if 'QF' in knb:
                self.focusing_knobs.append(knb)
            else:
                self.defocusing_knobs.append(knb)
        self.all_knobs = self.focusing_knobs + self.defocusing_knobs

    def calc_tune_matrix(self, model=None, acc=None):
        """."""
        if model is None:
            model = self.model
        if acc is None:
            acc = self.acc

        self.tune_matrix = np.zeros((2, len(self.all_knobs)))

        delta = 1e-6
        for idx, knb in enumerate(self.all_knobs):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    modcopy[seg].KL += delta/len(nmag)
            tunex, tuney = self.get_tunes(model=modcopy)
            self.tune_matrix[:, idx] = [
                (tunex - self.tunex)/delta, (tuney - self.tuney)/delta]
        return self.tune_matrix

    def calc_tune_2knobs_matrix(self, tune_matrix=None):
        """."""
        if tune_matrix is None:
            tune_matrix = self.calc_tune_matrix(self.model)

        tune_2knobs_matrix = np.zeros((2, 2))
        nfocus = len(self.focusing_knobs)

        for nf, _ in enumerate(self.focusing_knobs):
            tune_2knobs_matrix[:, 0] += tune_matrix[:, nf]
        for ndf, _ in enumerate(self.defocusing_knobs):
            tune_2knobs_matrix[:, 1] += tune_matrix[:, ndf+nfocus]
        return tune_2knobs_matrix

    def get_kl(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.all_knobs
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
                    deltakl=None,
                    method='proportional',
                    group=False):
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
            kl_f = self.get_kl(mod, knobs=self.focusing_knobs)
            kl_df = self.get_kl(mod, knobs=self.defocusing_knobs)
            kl_fsum = np.sum(kl_f)
            kl_dfsum = np.sum(kl_df)
        factor = 1
        for idx_knb, knb in enumerate(self.all_knobs):
            if group:
                if knb in self.focusing_knobs:
                    idx = self.focusing_knobs.index(knb)
                    delta = deltakl[0]
                    factor = kl_f[idx]/kl_fsum
                elif knb in self.defocusing_knobs:
                    idx = self.defocusing_knobs.index(knb)
                    delta = deltakl[1]
                    factor = kl_df[idx]/kl_dfsum
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
                      nsv=None,
                      group=False,
                      method='proportional'):
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
            tunemat = self.calc_tune_2knobs_matrix(tunemat)
            if method == 'additional':
                tunemat[:, 0] /= len(self.focusing_knobs)
                tunemat[:, 1] /= len(self.defocusing_knobs)
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
        invmat = -np.dot(np.dot(V.T, iS), U.T)
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
