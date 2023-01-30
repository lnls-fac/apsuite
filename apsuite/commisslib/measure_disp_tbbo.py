"""."""
import time as _time

import numpy as np

import pyaccel
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import SOFB, DevLILLRF

from ..optimization import SimulAnneal
from ..utils import MeasBaseClass as _BaseClass, \
    ThreadedMeasBaseClass as _TBaseClass, ParamsBaseClass as _ParamsBaseClass


class ParamsDisp(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.klystron_delta = -2
        self.wait_time = 40
        self.timeout_orb = 10
        self.num_points = 10
        # self.klystron_excit_coefs = [1.098, 66.669]  # old
        # self.klystron_excit_coefs = [1.01026423, 71.90322743]  # > 2.5nC
        self.klystron_excit_coefs = [0.80518365, 87.56545895]  # < 2.5nC


class MeasureDispTBBO(_BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(ParamsDisp())
        self.devices = {
            'bo_sofb': SOFB(SOFB.DEVICES.BO),
            'tb_sofb': SOFB(SOFB.DEVICES.TB),
            'kly2': DevLILLRF(DevLILLRF.DEVICES.LI_KLY2),
            }

    @property
    def energy(self):
        """."""
        return np.polyval(
            self.params.klystron_excit_coefs, self.devices['kly2'].amplitude)

    @property
    def trajx(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajx, self.devices['bo_sofb'].trajx])

    @property
    def trajy(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajy, self.devices['bo_sofb'].trajy])

    @property
    def nr_points(self):
        """."""
        return min(
            self.devices['tb_sofb'].nr_points,
            self.devices['bo_sofb'].nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.devices['tb_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['tb_sofb'].wait_buffer(timeout=timeout)
        self.devices['bo_sofb'].wait_buffer(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['tb_sofb'].cmd_reset()
        self.devices['bo_sofb'].cmd_reset()
        _time.sleep(1)

    def measure_dispersion(self):
        """."""
        self.nr_points = self.params.num_points
        delta = self.params.klystron_delta

        self.reset(3)
        self.wait(self.params.timeout_orb)
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        ene0 = self.energy

        origamp = self.devices['kly2'].amplitude
        self.devices['kly2'].amplitude = origamp + delta

        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)
        orb.append(np.hstack([self.trajx, self.trajy]))
        ene1 = self.energy

        self.devices['kly2'].amplitude = origamp

        d_ene = ene1/ene0 - 1
        return np.array(orb).sum(axis=0) / d_ene


def calc_model_dispersionTBBO(model, bpms):
    """."""
    dene = 0.0001
    rin = np.array([
        [0, 0, 0, 0, dene/2, 0],
        [0, 0, 0, 0, -dene/2, 0]]).T
    rout, *_ = pyaccel.tracking.line_pass(
        model, rin, bpms)
    dispx = (rout[0, 0, :] - rout[0, 1, :]) / dene
    dispy = (rout[2, 0, :] - rout[2, 1, :]) / dene
    return np.hstack([dispx, dispy])
