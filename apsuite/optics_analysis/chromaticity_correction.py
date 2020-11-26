"""."""
import numpy as np

import pyaccel
from pymodels import bo, si

from .base_correction import BaseCorr


class ChromCorr(BaseCorr):
    """."""

    _STRENGTH_TYPE = 'SL'
    SI_SF = ['SFA1', 'SFA2', 'SFB1', 'SFB2', 'SFP1', 'SFP2']
    SI_SD = ['SDA1', 'SDA2', 'SDA3',
             'SDB1', 'SDB2', 'SDB3',
             'SDP1', 'SDP2', 'SDP3']
    BO_SF = ['SF']
    BO_SD = ['SD']

    def __init__(self, model, acc, sf_knobs=None, sd_knobs=None,
                 method=None, grouping=None):
        """."""
        super().__init__(
            model=model, acc=acc, method=method, grouping=grouping)
        if self.acc == 'BO':
            self.knobs.focusing = sf_knobs or ChromCorr.BO_SF
            self.knobs.defocusing = sd_knobs or ChromCorr.BO_SD
            self.fam = bo.get_family_data(model)
        elif self.acc == 'SI':
            self.knobs.focusing = sf_knobs or ChromCorr.SI_SF
            self.knobs.defocusing = sd_knobs or ChromCorr.SI_SD
            self.fam = si.get_family_data(model)
        else:
            raise TypeError('Accelerator not supported.')

    def __str__(self):
        """."""
        strg = '{0:25s}= {1:s}\n'.format(
            'focusing sextupoles', str(self.knobs.focusing))
        strg += '{0:25s}= {1:s}\n'.format(
            'defocusing sextupoles', str(self.knobs.defocusing))
        strg += '{0:25s}= {1:30s}\n'.format(
            'correction method', self.method_str)
        strg += '{0:25s}= {1:30s}\n'.format(
            'grouping', self.grouping_str)
        return strg

    def get_chromaticities(self, model=None):
        """."""
        if model is None:
            model = self.model
        return self._get_parameter(model)

    def get_sl(self, model):
        """."""
        return self._get_strength(model)

    def calc_jacobian_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        chrom_matrix = np.zeros((2, len(self.knobs.all)))
        chrom0 = self.get_chromaticities(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.all):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    modcopy[seg].SL += delta/len(nmag)
            chrom = self.get_chromaticities(model=modcopy)
            chrom_matrix[:, idx] = (chrom - chrom0)/delta
        return chrom_matrix

    def _get_parameter(self, model=None):
        """."""
        if model is None:
            model = self.model
        chromx, chromy = pyaccel.optics.get_chromaticities(model)
        return np.array([chromx, chromy])
