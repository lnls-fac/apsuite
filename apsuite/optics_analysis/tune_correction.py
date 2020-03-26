"""."""

import numpy as np
from pymodels import bo, si
from apsuite.optics_analysis.base_correction import BaseCorr, KnobTypes
import pyaccel


class TuneCorr(BaseCorr):
    """."""

    __STRENGTH_TYPE = 'KL'
    SI_QF = ['QFA', 'QFB', 'QFP']
    SI_QD = ['QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2']
    BO_QF = ['QF']
    BO_QD = ['QD']

    def __init__(self, model, acc, qf_knobs=None, qd_knobs=None,
                 method=None, grouping=None):
        """."""
            self.knobs.focusing = qf_knobs or TuneCorr.BO_QF
            self.knobs.defocusing = qd_knobs or TuneCorr.BO_QD
            self.fam = bo.get_family_data(model)
            self.knobs.focusing = qf_knobs or TuneCorr.SI_QF
            self.knobs.defocusing = qd_knobs or TuneCorr.SI_QD
            self.fam = si.get_family_data(model)
        self.knobs = KnobTypes(Focusing=qf_knobs, Defocusing=qd_knobs)
        self.strength_type = 'KL'
        self.method = method
        self.grouping = grouping

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
        return strg

    def get_tunes(self, model):
        """."""
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
                    modcopy[seg].KL += delta/len(nmag)
            nu = self.get_tunes(model=modcopy)
            tune_matrix[:, idx] = (nu-nu0)/delta
        return tune_matrix

    def _get_parameter(self, model=None):
        """."""
        if model is None:
            model = self.model
        twinom, *_ = pyaccel.optics.calc_twiss(
            accelerator=model, indices='open')
        nux = twinom.mux[-1]/2/np.pi
        nuy = twinom.muy[-1]/2/np.pi
        return np.array([nux, nuy])
