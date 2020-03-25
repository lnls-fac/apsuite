"""."""
import numpy as np
from pymodels import bo, si
import pyaccel
from apsuite.optics_analysis.base_correction import BaseCorr


class ChromCorr(BaseCorr):
    """."""

    SI_SF = ['SFA1', 'SFA2', 'SFB1', 'SFB2', 'SFP1', 'SFP2']
    SI_SD = ['SDA1', 'SDA2', 'SDA3',
             'SDB1', 'SDB2', 'SDB3',
             'SDP1', 'SDP2', 'SDP3']
    BO_SF = ['SF']
    BO_SD = ['SD']

    def __init__(self, model, acc, sf_knobs=None, sd_knobs=None,
                 method=None, grouping=None):
        """."""
        super().__init__()
        self.model = model
        self.acc = acc
        self._method = ChromCorr.METHODS.Proportional
        self._grouping = ChromCorr.GROUPING.TwoKnobs
        if acc == 'BO':
            sf_knobs = sf_knobs or ChromCorr.BO_SF
            sd_knobs = sd_knobs or ChromCorr.BO_SD
            self.fam = bo.get_family_data(model)
        elif acc == 'SI':
            sf_knobs = sf_knobs or ChromCorr.SI_SF
            sd_knobs = sd_knobs or ChromCorr.SI_SD
            self.fam = si.get_family_data(model)
        self.define_knobs(sf_knobs, sd_knobs, strength_type='SL')
        self.method = method
        self.grouping = grouping

    def __str__(self):
        """."""
        strg = '{0:25s}= {1:s}\n'.format(
            'focusing sextupoles', str(self.knobs.Focusing))
        strg += '{0:25s}= {1:s}\n'.format(
            'defocusing sextupoles', str(self.knobs.Defocusing))
        strg += '{0:25s}= {1:30s}\n'.format(
            'correction method', self.method_str)
        strg += '{0:25s}= {1:30s}\n'.format(
            'grouping', self.grouping_str)
        return strg

    def get_parameter(self, model=None):
        """."""
        chromx, chromy = pyaccel.optics.get_chromaticities(model)
        return np.array([chromx, chromy])

    def calc_jacobian_matrix(self, model=None):
        """."""
        if model is None:
            model = self.model

        chrom_matrix = np.zeros((2, len(self.knobs.ALL)))
        chrom0 = self.get_parameter(model)

        delta = 1e-6
        for idx, knb in enumerate(self.knobs.ALL):
            modcopy = model[:]
            for nmag in self.fam[knb]['index']:
                for seg in nmag:
                    modcopy[seg].SL += delta/len(nmag)
            chrom = self.get_parameter(model=modcopy)
            chrom_matrix[:, idx] = (chrom - chrom0)/delta
        return chrom_matrix
