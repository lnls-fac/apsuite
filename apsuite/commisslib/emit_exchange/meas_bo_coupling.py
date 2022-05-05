"""Booster Coupling Measurement from Minimal Tune Separation.
(Work in progress)
"""
import numpy as _np
from scipy.optimize import least_squares
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _mpl_gs

from siriuspy.ramp.ramp import BoosterRamp
# from siriuspy.ramp.conn import ConnPS
# from siriuspy.devices import PowerSupply, EVG, Screen
from siriuspy.epics import PV

from . import BOTunebyBPM

from ...utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BOCouplingParams(_ParamsBaseClass):
    """."""

    QUADS = ('QF, QD')

    def __init__(self):
        """."""
        super().__init__()
        # I suggest to use a config without emittance exchange
        self.ramp_base_config = 'bo_ramp_flop'
        self.ramp_meas_time = 294  # [ms]
        self._quadfam_name = 'QD'
        self.nr_points = 21
        self.time_wait = 5  # [s]
        self.neg_percent = 0.1/100
        self.pos_percent = 0.1/100
        self.coupling_resolution = 0.02/100

    @property
    def quadfam_name(self):
        """."""
        return self._quadfam_name

    @quadfam_name.setter
    def quadfam_name(self, val):
        """."""
        if isinstance(val, str) and val.upper() in self.QUADS:
            self._quadfam_name = val.upper()

    def __str__(self):
        """."""
        stmp = '{0:22s} = {1:4s}  {2:s}\n'.format
        ftmp = '{0:22s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:22s} = {1:9d}  {2:s}\n'.format
        stg = stmp('quadfam_name', self.quadfam_name, '')
        stg += ftmp('ramp_meas_time', self.ramp_meas_time, '[ms]')
        stg += dtmp('nr_points', self.nr_points, '')
        stg += ftmp('time_wait', self.time_wait, '[s]')
        stg += ftmp('neg_percent', self.neg_percent, '')
        stg += ftmp('pos_percent', self.pos_percent, '')
        stg += ftmp('coupling_resolution', self.coupling_resolution, '')
        return stg


class BOMeasCoupling(_BaseClass):
    """BO Coupling measurement and fitting.

    tunex = coeff1 * quad_parameter + offset1
    tuney = coeff2 * quad_parameter + offset2

    tune1, tune2 = Eigenvalues([[tunex, coupling/2], [coupling/2, tuney]])

    fit parameters: coeff1, offset1, coeff2, offset2, coupling

    NOTE: It maybe necessary to add a quadratic quadrupole strength
          dependency for tunes!
    """

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=BOCouplingParams(), target=self._do_meas, isonline=isonline)
        if self.isonline:
            self.devices['bo_tune'] = BOTunebyBPM(isonline=True)
            self.devices['bo_ramp'] = BoosterRamp(self.params.ramp_base_config)
            self.devices['bo_ramp'].load()
            self.pvs['bo-qf-wfm'] = PV('BO-Fam:PS-QF:Wfm-RB')
            self.pvs['bo-qd-wfm'] = PV('BO-Fam:PS-QD:Wfm-RB')

    def _do_meas(self):
        """Extract tunes and dtunes based on orbit measurements"""
        pass

    def plot_fitting(self):
        "NOTE: Use MeasCoupling class to make this step"
        pass
