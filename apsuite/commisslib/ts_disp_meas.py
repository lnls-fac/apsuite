"""."""

import time as _time

import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

import pyaccel
from siriuspy.devices import SOFB as _SOFB, RFGen as _RFGen, EVG as _EVG
from siriuspy.search import BPMSearch as _BPMSearch
from apsuite.utils import MeasBaseClass as _MeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from pymodels import ts as _pyts, si as _pysi, bo as _pybo


class Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.deltarf = 100  # Hz
        self.bo_mom_compact = 7.2e-4
        self.ts_optics_mode = 'M1'


class MeasDispTS(_MeasBaseClass):
    """
    Class to measure the joined dispersion of BO, TS and SI.

    This class assumes BO, TS and SI SOFB are properly configured.
    All of them must be with the SyncWithInjection selected.
    BO must be in Monit1 Rate with the index at the end of the ramp.
    """

    def __init__(self, isonline=True):
        """."""
        super().__init__(params=Params(), isonline=isonline)
        if self.isonline:
            self.devices['si_sofb'] = _SOFB(_SOFB.DEVICES.SI)
            self.devices['bo_sofb'] = _SOFB(_SOFB.DEVICES.BO)
            self.devices['ts_sofb'] = _SOFB(_SOFB.DEVICES.TS)
            self.devices['evg'] = _EVG()
            self.devices['rfgen'] = _RFGen()

        bomod = _pybo.create_accelerator()
        idx = pyaccel.lattice.find_indices(bomod, 'fam_name', 'EjeSeptF')[0]
        bomod = pyaccel.lattice.shift(bomod, idx)
        self.bo_model = bomod

        famdata = _pybo.get_family_data(bomod)
        bpm0 = famdata['BPM']['devnames'][0]
        _bpm_names = _BPMSearch.get_names({'sec': 'BO', 'dev': 'BPM'})
        idx = _bpm_names.index(bpm0)
        self._idx_shift_bodata = idx

        simod = _pysi.create_accelerator()
        idx = pyaccel.lattice.find_indices(simod, 'fam_name', 'InjSeptF')[0]
        simod = pyaccel.lattice.shift(simod, idx)
        self.si_model = simod

    def make_measurement(self):
        """."""
        sisofb = self.devices['si_sofb']
        bosofb = self.devices['bo_sofb']
        tssofb = self.devices['ts_sofb']
        evg = self.devices['evg']
        rfgen = self.devices['rfgen']

        self.data['rffreq_ini'] = rfgen.frequency

        rfgen.frequency += self.params.deltarf / 2
        _time.sleep(2)

        bosofb.cmd_reset()
        tssofb.cmd_reset()
        sisofb.cmd_reset()
        evg.cmd_turn_on_pulses()
        evg.wait()
        _time.sleep(2)

        self.data['rffreq_pos'] = rfgen.frequency
        self.data['botrajx_pos'] = bosofb.trajx
        self.data['tstrajx_pos'] = tssofb.trajx
        self.data['sitrajx_pos'] = sisofb.trajx
        self.data['botrajy_pos'] = bosofb.trajy
        self.data['tstrajy_pos'] = tssofb.trajy
        self.data['sitrajy_pos'] = sisofb.trajy

        rfgen.frequency -= self.params.deltarf

        bosofb.cmd_reset()
        tssofb.cmd_reset()
        sisofb.cmd_reset()
        evg.cmd_turn_on_pulses()
        evg.wait()
        _time.sleep(2)

        self.data['rffreq_neg'] = rfgen.frequency
        self.data['botrajx_neg'] = bosofb.trajx
        self.data['tstrajx_neg'] = tssofb.trajx
        self.data['sitrajx_neg'] = sisofb.trajx
        self.data['botrajy_neg'] = bosofb.trajy
        self.data['tstrajy_neg'] = tssofb.trajy
        self.data['sitrajy_neg'] = sisofb.trajy

        rfgen.frequency += self.params.deltarf / 2

    def process_data(self):
        """."""
        dta = self.data
        deltarf = dta['rffreq_pos'] - dta['rffreq_neg']
        dener = -deltarf/dta['rffreq_ini'] / self.params.bo_mom_compact
        anl = dict()
        anl['delta_energy'] = dener
        anl['bodispx'] = (dta['botrajx_pos'] - dta['botrajx_neg'])/dener * 1e-6
        anl['bodispy'] = (dta['botrajy_pos'] - dta['botrajy_neg'])/dener * 1e-6
        anl['tsdispx'] = (dta['tstrajx_pos'] - dta['tstrajx_neg'])/dener * 1e-6
        anl['tsdispy'] = (dta['tstrajy_pos'] - dta['tstrajy_neg'])/dener * 1e-6
        anl['sidispx'] = (dta['sitrajx_pos'] - dta['sitrajx_neg'])/dener * 1e-6
        anl['sidispy'] = (dta['sitrajy_pos'] - dta['sitrajy_neg'])/dener * 1e-6

        self.analysis = anl

    def plot_data(self):
        """."""
        anl = self.analysis
        bodispx = _np.roll(anl['bodispx'], -self._idx_shift_bodata)
        bodispy = _np.roll(anl['bodispy'], -self._idx_shift_bodata)

        dispx = _np.r_[bodispx, anl['tsdispx'], anl['sidispx']]
        dispy = _np.r_[bodispy, anl['tsdispy'], anl['sidispy']]

        fig = _mplt.figure(figsize=(18, 6))
        gs = _mgs.GridSpec(1, 1)
        gs.update(
            left=0.07, right=0.98, top=0.92, bottom=0.1, hspace=0.25,
            wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])

        botwi, _ = pyaccel.optics.calc_twiss(self.bo_model)
        twi0 = botwi[0]

        tsmod, _ = _pyts.create_accelerator(
            optics_mode=self.params.ts_optics_mode)
        line = self.bo_model + tsmod + self.si_model

        twi, _ = pyaccel.optics.calc_twiss(line, init_twiss=twi0)
        bpmidx = pyaccel.lattice.find_indices(line, 'fam_name', 'BPM')

        spos = twi.spos[bpmidx]
        ax.plot(spos, twi.etax[bpmidx] * 100, '-o', label='Model')
        ax.plot(spos, dispx * 100, '-o', label='Measured X')
        ax.plot(spos, dispy * 100, '-o', label='Measured Y')

        bol = self.bo_model.length
        tsl = tsmod.length
        ax.axvline(bol, linestyle='--', color='k')
        ax.axvline(bol+tsl, linestyle='--', color='k')

        ax.set_ylabel('Dispersion [cm]')
        ax.set_xlabel('Position [m]')
        ax.legend(loc='best')

        return fig
