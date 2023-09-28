"""."""

import numpy as _np
from siriuspy.devices import RFGen, SOFB
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from pymodels import si
import pyaccel
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat


class CorrectVerticalDispersionParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.delta_rf_freq = 100  # [Hz]
        self.nrpoints_sofb = 20  # [Hz]

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        # stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('delta_rf_freq', self.delta_rf_freq, '[Hz]')
        stg += dtmp('nrpoints_sofb', self.nrpoints_sofb)
        return stg


class CorrectVerticalDispersion(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)

        self.alpha = 1.6357564091403232e-4
        if self.isonline:
            self.devices['rfgen'] = RFGen(
                props2init=['GeneralFreq-SP', 'GeneralFreq-RB'])
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

    def get_orb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        return _np.r_[sofb.orbx, sofb.orby]

    def measure_dispersion(self):
        """."""
        rfgen, sofb = self.devices['rfgen'], self.devices['sofb']
        prms = self.params
        rf_freq0 = rfgen.frequency
        sofb.nr_points = prms.nrpoints_sofb
        rfgen.frequency = rf_freq0 + prms.delta_rf_freq/2
        orbplus = self.get_orb()
        rfgen.frequency = rf_freq0 - prms.delta_rf_freq/2
        orbminus = self.get_orb()
        dorb_dfreq = (orbplus - orbminus)/prms.delta_rf_freq
        disp = - self.alpha * rf_freq0 * dorb_dfreq
        return disp, rf_freq0

    @staticmethod
    def calc_model_dispersion(model):
        """."""
        orm = OrbRespmat(model=model, acc='SI', dim='6d')
        respmat = orm.get_respm()
        rf_freq = orm.model[orm.rf_idx[0]].frequency
        alpha = pyaccel.optics.get_mcf(model)
        return - alpha * rf_freq * respmat[:, -1] * 1e-6

    @classmethod
    def calc_dispmat(cls, mod, qsidx=None, dksl=1e-6):
        """."""
        print('--- calculating dispersion/KsL matrix')
        if qsidx is None:
            fam = si.get_family_data(mod)
            qsidx = _np.array(fam['QS']['index']).ravel()

            # get only chromatic skew quads
            chrom = []
            for qs in qsidx:
                if '0' not in mod[qs].fam_name:
                    chrom.append(qs)
            qsidx = _np.array(chrom).ravel()
        disp_mat = []
        for qs in qsidx:
            modt = mod[:]
            modt[qs].KsL += dksl/2
            dispp = cls.calc_model_dispersion(modt)
            modt[qs].KsL -= dksl
            dispn = cls.calc_model_dispersion(modt)
            disp_mat.append((dispp-dispn)/dksl)
            modt[qs].KsL += dksl
        disp_mat = _np.array(disp_mat).T
        return disp_mat, qsidx

    def calc_correction(self, dispy):
        """."""
        return

    def _meas_func(self):
        return None
