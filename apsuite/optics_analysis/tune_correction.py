"""."""

import numpy as np

import pyaccel
from pymodels import bo, si

from .base_correction import BaseCorr
from mathphys.functions import get_namedtuple as _get_namedtuple


class TuneCorr(BaseCorr):
    """."""

    _STRENGTH_TYPE = 'KL'
    SI_QF = ['QFA', 'QFB', 'QFP']
    SI_QD = ['QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2']
    BO_QF = ['QF']
    BO_QD = ['QD']
    OPTICS = _get_namedtuple('Optics', ['EdwardsTeng', 'Twiss'])

    def __init__(self, model, acc, qf_knobs=None, qd_knobs=None,
                 method=None, grouping=None, type_optics=None, idcs_out=None):
        """."""
        super().__init__(
            model=model, acc=acc, method=method, grouping=grouping,
            idcs_out=idcs_out)
        self._type_optics = TuneCorr.OPTICS.EdwardsTeng
        self.type_optics = type_optics
        self.idcs_out = idcs_out
        if self.type_optics == self.OPTICS.EdwardsTeng:
            self._optics_func = pyaccel.optics.calc_edwards_teng
        elif self._type_optics == self.OPTICS.Twiss:
            self._optics_func = pyaccel.optics.calc_twiss
        else:
            raise TypeError('Optics type not supported.')

        if self.acc == 'BO':
            self.knobs.focusing = qf_knobs or TuneCorr.BO_QF
            self.knobs.defocusing = qd_knobs or TuneCorr.BO_QD
            self.fam = bo.get_family_data(model)
        elif self.acc == 'SI':
            self.knobs.focusing = qf_knobs or TuneCorr.SI_QF
            self.knobs.defocusing = qd_knobs or TuneCorr.SI_QD
            self.fam = si.get_family_data(model)
        else:
            raise TypeError('Accelerator not supported.')

    def __str__(self):
        """."""
        strg = '{0:25s}= {1:s}\n'.format(
            'focusing quadrupoles', str(self.knobs.focusing))
        strg += '{0:25s}= {1:s}\n'.format(
            'defocusing quadrupoles', str(self.knobs.defocusing))
        strg += '{0:25s}= {1:30s}\n'.format(
            'correction method', self.method_str)
        strg += '{0:25s}= {1:30s}\n'.format(
            'grouping', self.grouping_str)
        strg += '{0:25s}= {1:30s}\n'.format(
            'type_optics', self.type_optics_str)
        return strg

    @property
    def type_optics_str(self):
        """."""
        return TuneCorr.OPTICS._fields[self._type_optics]

    @property
    def type_optics(self):
        """."""
        return self._type_optics

    @type_optics.setter
    def type_optics(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._type_optics = int(value in TuneCorr.OPTICS._fields[1])
        elif int(value) in TuneCorr.OPTICS:
            self._type_optics = int(value)

    def get_tunes(self, model=None):
        """."""
        if model is None:
            model = self.model
        return self._get_parameter(model)

    def get_kl(self, model):
        """."""
        return self._get_strength(model)

    def calc_jacobian_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        tune_matrix = np.zeros((2, len(self.knobs.all)))
        nu0 = self.get_tunes(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.all):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    if self.idcs_out is None or seg not in self.idcs_out:
                        modcopy[seg].KL += delta/len(nmag)
            nu = self.get_tunes(model=modcopy)
            tune_matrix[:, idx] = (nu-nu0)/delta
        return tune_matrix

    def _get_parameter(self, model=None):
        """."""
        if model is None:
            model = self.model
        linopt, *_ = self._optics_func(
            accelerator=model, indices='open')
        if self.type_optics == self.OPTICS.EdwardsTeng:
            nu1 = linopt.mu1[-1]
            nu2 = linopt.mu2[-1]
        else:
            nu1 = linopt.mux[-1]
            nu2 = linopt.muy[-1]
        return np.array([nu1, nu2])/2/np.pi
