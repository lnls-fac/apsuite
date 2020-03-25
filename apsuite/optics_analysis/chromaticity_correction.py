"""."""

from collections import namedtuple as _namedtuple
from copy import deepcopy as _dcopy
import numpy as np
from pymodels import bo, si
import pyaccel

__KNOBTYPES = _namedtuple('KnobTypes', ['SFs', 'SDs'])


class KnobTypes(__KNOBTYPES):
    """."""

    @property
    def ALL(self):
        """."""
        return self.SFs + self.SDs


class ChromCorr():
    """."""

    SI_DEF_KNOBS = KnobTypes(
        ['SFA1', 'SFA2', 'SFB1', 'SFB2', 'SFP1', 'SFP2'],
        ['SDA1', 'SDA2', 'SDA3',
         'SDB1', 'SDB2', 'SDB3',
         'SDP1', 'SDP2', 'SDP3'])
    BO_DEF_KNOBS = KnobTypes(['SF', ], ['SD', ])

    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    GROUPING = _namedtuple('Grouping', ['Individual', 'TwoKnobs'])(0, 1)
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model, acc, sf_knobs=None, sd_knobs=None,
                 method=None, grouping=None):
        """."""
        self.model = model
        self.acc = acc
        self._method = ChromCorr.METHODS.Proportional
        self._grouping = ChromCorr.GROUPING.TwoKnobs
        if acc == 'BO':
            sf_knobs = sf_knobs or ChromCorr.BO_DEF_KNOBS.SFs
            sd_knobs = sd_knobs or ChromCorr.BO_DEF_KNOBS.SDs
            self.knobs = KnobTypes(sf_knobs, sd_knobs)
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            sf_knobs = sf_knobs or ChromCorr.SI_DEF_KNOBS.SFs
            sd_knobs = sd_knobs or ChromCorr.SI_DEF_KNOBS.SDs
            self.knobs = KnobTypes(sf_knobs, sd_knobs)
            self.fam = si.get_family_data(model)
        self.method = method
        self.grouping = grouping

    def __str__(self):
        """."""
        strg = '{0:25s}= {1:s}\n'.format(
            'focusing sextupoles', str(self.knobs.SFs))
        strg += '{0:25s}= {1:s}\n'.format(
            'defocusing sextupoles', str(self.knobs.SDs))
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
            self._grouping = int(value in ChromCorr.GROUPING._fields[1])
        elif int(value) in ChromCorr.GROUPING:
            self._grouping = int(value)

    @property
    def grouping_str(self):
        """."""
        return ChromCorr.GROUPING._fields[self._grouping]

    @property
    def method(self):
        """."""
        return self._method

    @method.setter
    def method(self, value):
        if value is None:
            return
        elif isinstance(value, str):
            self._method = int(value in ChromCorr.METHODS._fields[1])
        elif int(value) in ChromCorr.METHODS:
            self._method = int(value)

    @property
    def method_str(self):
        """."""
        return ChromCorr.METHODS._fields[self._method]

    def calc_chrom_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        chrom_matrix = np.zeros((2, len(self.knobs.ALL)))
        chromx0, chromy0 = pyaccel.optics.get_chromaticities(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.ALL):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    modcopy[seg].SL += delta/len(nmag)
            chromx, chromy = pyaccel.optics.get_chromaticities(modcopy)
            chrom_matrix[:, idx] = [
                (chromx - chromx0)/delta, (chromy - chromy0)/delta]
        return chrom_matrix

    def get_sl(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs.ALL
        sl = []
        for knb in knobs:
            sl_mag = []
            for mag in self.fam[knb]['index']:
                sl_seg = []
                for seg in mag:
                    sl_seg.append(model[seg].SL)
                sl_mag.append(sum(sl_seg))
            sl.append(np.mean(sl_mag))
        return np.array(sl)

    def correct_chrom(self,
                      chromx, chromy,
                      model=None,
                      chrom_matrix=None,
                      tol=1e-6,
                      nr_max=10,
                      nsv=None):
        """."""
        if model is None:
            model = self.model
        if self.method not in ChromCorr.METHODS:
            raise Exception('Invalid correction method!')

        if chrom_matrix is None:
            chrommat = self.calc_chrom_matrix(model)
        else:
            chrommat = _dcopy(chrom_matrix)

        nominal_sl = self.get_sl(model)
        if self._method == ChromCorr.METHODS.Proportional:
            chrommat *= nominal_sl
        if self._grouping == ChromCorr.GROUPING.TwoKnobs:
            chrommat = self._group_2knobs_matrix(chrommat)

        U, S, V = np.linalg.svd(chrommat, full_matrices=False)
        iS = 1/S
        iS[np.isnan(iS)] = 0
        iS[np.isinf(iS)] = 0
        if nsv is not None:
            iS[nsv:] = 0
        iS = np.diag(iS)
        invmat = -1 * np.dot(np.dot(V.T, iS), U.T)
        chromx_new, chromy_new = pyaccel.optics.get_chromaticities(model)
        dchrom = np.array([chromx_new-chromx, chromy_new-chromy])
        if np.sum(dchrom*dchrom) < tol:
            return ChromCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dsl = np.dot(invmat, dchrom)
            self._add_deltasl(dsl, model=model)
            chromx_new, chromy_new = pyaccel.optics.get_chromaticities(model)
            dchrom = np.array([chromx_new-chromx, chromy_new-chromy])
            if np.sum(dchrom*dchrom) < tol:
                break
        else:
            return ChromCorr.CORR_STATUS.Fail
        return ChromCorr.CORR_STATUS.Sucess

    def _add_deltasl(self, deltasl, model=None):
        """."""
        if model is None:
            model = self.model

        for idx_knb, knb in enumerate(self.knobs.ALL):
            if self._grouping == ChromCorr.GROUPING.TwoKnobs:
                if knb in self.knobs.SFs:
                    delta = deltasl[0]
                elif knb in self.knobs.SDs:
                    delta = deltasl[1]
            else:
                delta = deltasl[idx_knb]
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    if self._method == ChromCorr.METHODS.Proportional:
                        model[seg].SL *= (1 + delta/len(mag))
                    else:
                        model[seg].SL += delta/len(mag)

    def _group_2knobs_matrix(self, chrom_matrix=None):
        """."""
        if chrom_matrix is None:
            chrom_matrix = self.calc_chrom_matrix(self.model)

        chrom_2knobs_matrix = np.zeros((2, 2))
        nfocus = len(self.knobs.SFs)

        for nf, _ in enumerate(self.knobs.SFs):
            chrom_2knobs_matrix[:, 0] += chrom_matrix[:, nf]
        for ndf, _ in enumerate(self.knobs.SDs):
            chrom_2knobs_matrix[:, 1] += chrom_matrix[:, ndf+nfocus]
        return chrom_2knobs_matrix
