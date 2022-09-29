"""."""
import numpy as np

from mathphys.functions import save_pickle as _save_pickle, \
    load_pickle as _load_pickle
from siriuspy.devices import SOFB, RFGen

from .. import asparams as _asparams

class DispMeas:
    """."""

    MOM_COMPACT = _asparams.BO_MOM_COMPACT

    def __init__(self):
        """."""
        self.sofb = SOFB(SOFB.DEVICES.BO)
        self.rfgen = RFGen()
        self.orb0 = np.array([])
        self.orbp = np.array([])
        self.orbm = np.array([])
        self.dispersion = np.array([])
        self.freq0 = 0
        self.deltaf = 0
        self.freqp = 0
        self.freqm = 0

    @property
    def deltaen(self):
        """."""
        return - self.deltaf/self.freq0/self.MOM_COMPACT

    def get_orb(self):
        """."""
        self.sofb.cmd_reset()
        self.sofb.wait_buffer()
        return np.array([self.sofb.trajx, self.sofb.trajy]).transpose()

    def calc_disp(self):
        """."""
        if self.deltaen:
            self.dispersion = (self.orbp - self.orbm)/2/self.deltaen
            self.dispersion *= 1e-4  # conversion from um to cm
        else:
            raise Exception('Delta Energy is Zero!')

    def run_meas(self, deltaf=0):
        """."""
        self.deltaf = deltaf
        print('Getting initial orbit...')
        self.orb0 = self.get_orb()
        self.freq0 = self.rfgen.frequency
        self.deltaen = - self.deltaf/self.freq0/self.MOM_COMPACT
        print('Initial frequency is {:.6f} MHz'.format(self.freq0*1e-6))
        print('Applying Delta + {:.6f} kHz'.format(deltaf*1e-3))
        self.rfgen.frequency(self.freq0+deltaf)
        print('Getting orbit ...')
        self.orbp = self.get_orb()
        self.freqp = self.rfgen.frequency
        print('Applying Delta - {:.6f} kHz'.format(deltaf*1e-3))
        self.rfgen.frequency(self.freq0-deltaf)
        print('Getting orbit ...')
        self.orbm = self.get_orb()
        self.freqm = self.rfgen.frequency
        print('Setting initial frequency ...')
        self.rfgen.frequency(self.freq0)
        self.calc_disp()
        print('Finished!')

    def save_data(self, fname, overwrite=True):
        """."""
        data = dict(
            freq0=self.freq0,
            orb0=self.orb0,
            delta_freq=self.deltaf,
            delta_energy=self.deltaen,
            freqp=self.freqp,
            orbp=self.orbp,
            freqm=self.freqm,
            orbm=self.orbm,
            dispersion=self.dispersion,
            )
        _save_pickle(data, fname, overwrite=overwrite)

    @staticmethod
    def load_data(fname):
        """."""
        return _load_pickle(fname)
