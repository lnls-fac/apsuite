"""Control injection for optimizing BO injection during top-up.

Once fired, controls the injection system to pulse in between top-up
injections. The optimization pulses are configured so that the injected
beam is not transported to BO extraction. The beam is dumped before the
end of the ramp. Pulses are triggered separated by
`self.params.wait_between_injections` seconds, as long as the next top-up
pulse time is larger then `self.params.stop_pulsning_time` seconds away in
the future.
"""

import time as _time
import logging as _log

from siriuspy.devices import EGBias, EVG, InjCtrl, Trigger

from ..utils import ParamsBaseClass as _ParamsBase
from ..utils import ThreadedMeasBaseClass as _BaseClass
from .. import asparams as _asp


class InjCtrlPulseInjBOParams(_ParamsBase):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.wait_between_injections = 3  # [s]
        self.stop_pulsing_time = 14  # [s]
        self.bias_voltage = -35.0  # [V]
        self.ejekckr_delay = 80  # [ms]

    def __str__(self):
        """."""
        TMPF = '{:30s}: {:10.3f} {:s}\n'.format
        stg = TMPF(
            'wait_between_injections', self.wait_between_injections, '[s]'
        )
        stg += TMPF('stop_pulsing_time', self.stop_pulsing_time, '[s]')
        stg += TMPF('bias_voltage', self.bias_voltage, '[V]')
        stg += TMPF(
            'ejekckr_delay', self.ejekckr_delay, '[ms] (approximately)'
        )
        return stg


class InjCtrlPulseInjBO(_BaseClass):
    """Control injection for optimizing BO injection during top-up.

    Once fired, controls the injection system to pulse in between top-up
    injections. The optimization pulses are configured so that the injected
    beam is not transported to BO extraction. The beam is dumped before the
    end of the ramp. Pulses are triggered separated by
    `self.params.wait_between_injections` seconds, as long as the next top-up
    pulse time is larger then `self.params.stop_pulsning_time` seconds away in
    the future.
    """

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=InjCtrlPulseInjBOParams(),
            target=self.ctrl_injection,
            isonline=isonline,
        )
        if self.isonline:
            self._create_devices()
        self.allow_injection = True

    def ctrl_injection(self):
        """."""
        if not self.wait_for_connection():
            _log.warning('Failed to connect devices.')
            return
        injc = self.devices['injctrl']

        delay_inj = self.devices['trig_ejekckr'].delay / 1000  # us -> ms
        delay_raw_inj = self.devices['trig_ejekckr'].delay_raw

        delta_dly = abs(self.params.ejekckr_delay - delay_inj) / 1000 # ms -> s
        delta_dly_raw = int(delta_dly / _asp.BO_REV_TIME)
        delta_dly_raw *= _asp.BO_HARM_NUM // _asp.TIMING_RF_DIVISOR

        delay_raw_opt = delay_raw_inj - delta_dly_raw

        try:

            self._prepare_injector_state(start_opt=True)

            while not self._stopevt.is_set():
                t0 = _time.time()

                is_topup = injc.injmode == injc.InjMode.TopUp and injc.topup_state
                do_inj = (not is_topup) or (
                    injc.topup_nextinj_timestamp - _time.time() >=
                    self.params.stop_pulsing_time
                )

                if do_inj:
                    if self.allow_injection:
                        self.prepare_for_opt(delay_raw_opt)
                        self.devices['evg'].cmd_turn_on_injection()
                        self.devices['evg'].wait_injection_finish()
                        _log.info('Injecting for optimization...')
                    else:
                        _log.info(
                            'self.allow_injection is False '
                            + 'Not injecting while False.'
                        )
                else:
                    _log.info(
                        'Not injecting for optimization. Preparing for top-up.'
                    )
                    self.prepare_for_inj(delay_raw_inj)
                _time.sleep(
                    max(
                        0,
                        self.params.wait_between_injections
                        - (_time.time() - t0),
                    )
                )
        finally:
            _log.info('Exiting. Setting up injection for top-up')
            self.prepare_for_inj(delay_raw_inj)
            self._prepare_injector_state(start_opt=False)

    def prepare_for_opt(self, dly):
        """."""
        self.devices['injctrl'].biasfb_model_updatedata = 0
        self.set_trigger_delay_raw(self.devices['trig_ejekckr'], dly)
        self.devices['egunbias'].voltage = self.params.bias_voltage
        self._set_triggers_state(state=0)

    def prepare_for_inj(self, dly):
        """."""
        self.devices['injctrl'].biasfb_model_updatedata = 1
        self.set_trigger_delay_raw(self.devices['trig_ejekckr'], dly)
        self.devices['egunbias'].voltage = self.devices[
            'injctrl'
        ].bias_volt_multbun
        self._set_triggers_state(state=1)

    def set_trigger_delay_raw(self, trig, val):
        """."""
        if trig.delay_raw != val:
            trig.delay_raw = val

    def _create_devices(self):
        """."""
        self.devices['injctrl'] = InjCtrl()
        self.devices['evg'] = EVG()
        self.devices['egunbias'] = EGBias()
        self.devices['trig_ejekckr'] = Trigger('BO-48D:TI-EjeKckr')
        self.devices['trig_ejeseptf'] = Trigger('TS-01:TI-EjeSeptF')
        self.devices['trig_ejeseptg'] = Trigger('TS-01:TI-EjeSeptG')
        self.devices['trig_injsetpg1'] = Trigger('TS-04:TI-InjSeptG-1')
        self.devices['trig_injsetpg2'] = Trigger('TS-04:TI-InjSeptG-2')
        self.devices['trig_injsetpf'] = Trigger('TS-04:TI-InjSeptF')
        self.devices['trig_nlk'] = Trigger('SI-01SA:TI-InjNLKckr')
        self.devices['trig_ffcors'] = Trigger('SI-01:TI-Mags-FFCorrs')
        self.devices['trig_osc_ejebo'] = Trigger('AS-Glob:TI-Osc-EjeBO')
        self.devices['trig_osc_injsi1'] = Trigger('AS-Glob:TI-Osc-InjSI')
        self.devices['trig_osc_injsi2'] = Trigger('AS-Glob:TI-Osc-InjSI2')

    def _set_triggers_state(self, state):
        """."""
        self.devices['trig_ejeseptf'].state = state
        self.devices['trig_ejeseptg'].state = state
        self.devices['trig_injsetpg1'].state = state
        self.devices['trig_injsetpg2'].state = state
        self.devices['trig_injsetpf'].state = state
        self.devices['trig_nlk'].state = state
        self.devices['trig_ffcors'].state = state
        self.devices['trig_osc_ejebo'].state = state
        self.devices['trig_osc_injsi1'].state = state
        self.devices['trig_osc_injsi2'].state = state

    def _prepare_injector_state(self, start_opt=True):
        injc = self.devices['injctrl']

        stdby_stt = not start_opt
        injc.topup_warmup_li_rf = stdby_stt
        injc.topup_standby_bo_rf = stdby_stt
        injc.topup_standby_bo_injkckr = stdby_stt
        injc.topup_standby_bo_ejekckr = stdby_stt
        injc.topup_standby_tb_injsept = stdby_stt

        # After turning off the warmup of the LI LLRF we have to turn the
        # injection system on to make sure the LI LLRF is pulsing at 2Hz.
        # We can't turn the system off at the end to not compromise topup.
        if start_opt:
            injc.cmd_injsys_turn_on()
