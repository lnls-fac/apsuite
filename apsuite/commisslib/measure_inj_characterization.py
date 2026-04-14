"""Measurement class to characterize the injector system.

Intended to characterize the injector system baseline behavior as
well as its response to typical changes in its main knobs. This
characterization was thought to be performed before and after the
interventions in the TB transport line during the machine shutdown of
march/april 2026, related to rearrangements in the vacuum chamber ceramic
transition.

"""

import time as _time

from mathphys.functions import get_namedtuple as _get_namedtuple
from siriuspy.devices import (
    CurrInfoAS,
    EGBias,
    EVG,
    InjCtrl,
    LILLRF,
    PosAng,
    PowerSupply,
    PowerSupplyPU,
    RFGen,
    SOFB,
    Screen,
    Trigger
)
from siriuspy.epics import PV as _PV, CAThread as _Thread

from apsuite.utils import ParamsBaseClass, ThreadedMeasBaseClass


class SeptCharacterizationParams(ParamsBaseClass):
    """."""

    AcqType = _get_namedtuple('AcqType', ['Asynchronous', 'Synchronous'])
    ScreenOptions = _get_namedtuple(
        'ScreenOptions',
        ['TB_4', 'TB_5', 'TB_6', 'BO_1', 'BO_2', 'BO_3', 'none'],
    )

    def __init__(self):
        """."""
        super().__init__()
        self._acq_type = self.AcqType.Asynchronous
        self._which_screen = self.ScreenOptions.BO_1
        self.acq_interval = 1.0
        self.asyn_wait_after_screen = 0.1

    @property
    def acq_type_str(self):
        """."""
        return self.AcqType._fields[self._acq_type]

    @property
    def acq_type(self):
        """."""
        return self._acq_type

    @acq_type.setter
    def acq_type(self, value):
        """."""
        if isinstance(value, str):
            value = self.AcqType._fields.index(value)
        else:
            value = int(value)
        self._acq_type = value

    @property
    def which_screen_str(self):
        """."""
        return self.ScreenOptions._fields[self._which_screen]

    @property
    def which_screen(self):
        """."""
        return self._which_screen

    @which_screen.setter
    def which_screen(self, value):
        """."""
        if isinstance(value, str):
            value = self.ScreenOptions._fields.index(value)
        else:
            value = int(value)
        self._which_screen = value

    def __str__(self):
        """."""
        stg = ''
        stg += f'\nacq_type: {self.acq_type} ({self.acq_type_str})'
        stg += f'\nwhich_screen: {self.which_screen} ({self.which_screen_str})'
        stg += f'\nacq_interval: {self.acq_interval} [s] (used with sync acq.)'
        stg += f'\nasyn_wait_after_screen: {self.asyn_wait_after_screen} [s]'
        return stg


class SeptCharacterization(ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=SeptCharacterizationParams(),
            target=self.do_measurement,
            isonline=isonline,
        )
        if self.isonline:
            self.create_devices()

        self._thread_cb = None

    def create_devices(self):
        """."""
        self.devices['scrn_tb_4'] = Screen(Screen.DEVICES.TB_4)
        self.devices['scrn_tb_5'] = Screen(Screen.DEVICES.TB_5)
        self.devices['scrn_tb_6'] = Screen(Screen.DEVICES.TB_6)
        self.devices['scrn_bo_1'] = Screen(Screen.DEVICES.BO_1)
        self.devices['scrn_bo_2'] = Screen(Screen.DEVICES.BO_2)
        self.devices['scrn_bo_3'] = Screen(Screen.DEVICES.BO_3)

        self.devices['currinfo'] = CurrInfoAS()

        self.devices['rfgen'] = RFGen(
            props2init=['GeneralFreq-SP', 'GeneralFreq-RB']
        )
        self.devices['evg'] = EVG()
        self.devices['injctrl'] = InjCtrl()
        self.devices['egun_bias'] = EGBias()

        self.devices['li_llrf'] = LILLRF()

        self.devices['sofb_tb'] = SOFB(SOFB.DEVICES.TB)
        self.devices['sofb_bo'] = SOFB(SOFB.DEVICES.BO)

        self.devices['trig_kckr'] = Trigger('BO-01D:TI-InjKckr')
        self.devices['trig_sept'] = Trigger('TB-04:TI-InjSept')
        self.devices['trig_scrn'] = Trigger('AS-Fam:TI-Scrn-TBBO')
        self.devices['trig_sofb_tb'] = Trigger('TB-Fam:TI-BPM')
        self.devices['trig_sofb_bo'] = Trigger('BO-Fam:TI-BPM')

        self.devices['posang'] = PosAng(PosAng.DEVICES.TB)

        self.devices['pwrsply_kckr'] = PowerSupplyPU('BO-01D:PU-InjKckr')
        self.devices['pwrsply_sept'] = PowerSupplyPU('TB-04:PU-InjSept')
        self.devices['pwrsply_tb_ch1'] = PowerSupply('TB-04:PS-CH-1')
        self.devices['pwrsply_tb_ch2'] = PowerSupply('TB-04:PS-CH-2')
        self.devices['pwrsply_tb_cv1'] = PowerSupply('TB-04:PS-CV-1')
        self.devices['pwrsply_tb_cv2'] = PowerSupply('TB-04:PS-CV-2')

        self.pvs['temp_tb_cham_1'] = _PV('TB-04:VA-PT100-ED1:Temp-Mon')
        self.pvs['temp_tb_cham_2'] = _PV('TB-04:VA-PT100-ED2:Temp-Mon')
        self.pvs['temp_tb_sept_bg'] = _PV('TB-04:PU-InjSept-BG:Temp-Mon')
        self.pvs['temp_tb_sept_ed'] = _PV('TB-04:PU-InjSept-ED:Temp-Mon')

    def get_data(self):
        """."""
        data = {}
        data['timestamp'] = _time.time()

        if self.params.which_screen == self.params.ScreenOptions.none:
            data['sofb_bo_trajx'] = self.devices['sofb_bo'].mt_trajx
            data['sofb_bo_trajy'] = self.devices['sofb_bo'].mt_trajy
            data['sofb_bo_sum'] = self.devices['sofb_bo'].mt_sum
        else:
            scrn = self.devices[f'scrn_{self.params.which_screen_str.lower()}']
            data['scrn_enabled'] = scrn.enabled
            data['scrn_position'] = scrn.screen_position
            data['scrn_gain'] = scrn.gain
            data['scrn_exposure_time'] = scrn.exposure_time
            data['scrn_image_raw'] = scrn.image
            data['scrn_centerx'] = scrn.centerx
            data['scrn_centery'] = scrn.centery
            data['scrn_sigmax'] = scrn.sigmax
            data['scrn_sigmay'] = scrn.sigmay
            data['scrn_theta'] = scrn.angle
            data['scrn_scalex'] = scrn.scale_factor_x
            data['scrn_scaley'] = scrn.scale_factor_y

        data['sofb_tb_trajx'] = self.devices['sofb_tb'].sp_trajx
        data['sofb_tb_trajy'] = self.devices['sofb_tb'].sp_trajy
        data['sofb_tb_sum'] = self.devices['sofb_tb'].sp_sum

        currinfo = self.devices['currinfo']
        data['currinfo_li_charge1'] = currinfo.li.charge_ict1
        data['currinfo_li_charge2'] = currinfo.li.charge_ict2
        data['currinfo_tb_charge1'] = currinfo.tb.charge_ict1
        data['currinfo_tb_charge2'] = currinfo.tb.charge_ict2

        data['rf_freq'] = self.devices['rfgen'].frequency
        data['evg_injcount'] = self.devices['evg'].injection_count
        data['injctrl_injmode_str'] = self.devices['injctrl'].injmode_str
        data['egun_bias_voltage'] = self.devices['egun_bias'].voltage

        lillrf = self.devices['li_llrf']
        data['lillrf_shb_phase'] = lillrf.dev_shb.phase
        data['lillrf_shb_amplitude'] = lillrf.dev_shb.amplitude
        data['lillrf_kly1_phase'] = lillrf.dev_klystron1.phase
        data['lillrf_kly1_amplitude'] = lillrf.dev_klystron1.amplitude
        data['lillrf_kly2_phase'] = lillrf.dev_klystron2.phase
        data['lillrf_kly2_amplitude'] = lillrf.dev_klystron2.amplitude

        data['trig_kckr_state'] = self.devices['trig_kckr'].state
        data['trig_kckr_delay_raw'] = self.devices['trig_kckr'].delay_raw
        data['trig_kckr_source_str'] = self.devices['trig_kckr'].source_str
        data['trig_sept_state'] = self.devices['trig_sept'].state
        data['trig_sept_delay_raw'] = self.devices['trig_sept'].delay_raw
        data['trig_sept_source_str'] = self.devices['trig_sept'].source_str
        data['trig_scrn_state'] = self.devices['trig_scrn'].state
        data['trig_scrn_delay_raw'] = self.devices['trig_scrn'].delay_raw
        data['trig_scrn_source_str'] = self.devices['trig_scrn'].source_str
        data['trig_sofb_tb_state'] = self.devices['trig_sofb_tb'].state
        data['trig_sofb_tb_delay_raw'] = self.devices['trig_sofb_tb'].delay_raw
        data['trig_sofb_tb_source_str'] = self.devices['trig_sofb_tb'].source_str
        data['trig_sofb_bo_state'] = self.devices['trig_sofb_bo'].state
        data['trig_sofb_bo_delay_raw'] = self.devices['trig_sofb_bo'].delay_raw
        data['trig_sofb_bo_source_str'] = self.devices['trig_sofb_bo'].source_str

        data['posang_delta_posx'] = self.devices['posang'].delta_posx
        data['posang_delta_angx'] = self.devices['posang'].delta_angx
        data['posang_delta_posy'] = self.devices['posang'].delta_posy
        data['posang_delta_angy'] = self.devices['posang'].delta_angy

        data['pwrsply_kckr_voltage'] = self.devices['pwrsply_kckr'].voltage
        data['pwrsply_kckr_pwrstate'] = self.devices['pwrsply_kckr'].pwrstate
        data['pwrsply_kckr_pulsestate'] = self.devices['pwrsply_kckr'].pulse
        data['pwrsply_sept_voltage'] = self.devices['pwrsply_sept'].voltage
        data['pwrsply_sept_pwrstate'] = self.devices['pwrsply_sept'].pwrstate
        data['pwrsply_sept_pulsestate'] = self.devices['pwrsply_sept'].pulse

        data['pwrsply_tb_ch1_current'] = self.devices['pwrsply_tb_ch1'].current
        data['pwrsply_tb_ch2_current'] = self.devices['pwrsply_tb_ch2'].current
        data['pwrsply_tb_cv1_current'] = self.devices['pwrsply_tb_cv1'].current
        data['pwrsply_tb_cv2_current'] = self.devices['pwrsply_tb_cv2'].current

        data['temp_tb_cham_1'] = self.pvs['temp_tb_cham_1'].value
        data['temp_tb_cham_2'] = self.pvs['temp_tb_cham_2'].value
        data['temp_tb_sept_bg'] = self.pvs['temp_tb_sept_bg'].value
        data['temp_tb_sept_ed'] = self.pvs['temp_tb_sept_ed'].value
        return data

    def do_measurement(self):
        """."""
        if self.params.which_screen == self.params.ScreenOptions.none:
            pvo = self.devices['sofb_bo'].pv_object('MTurnOrbX-Mon')
        else:
            scrn = self.devices[f'scrn_{self.params.which_screen_str.lower()}']
            if not scrn.enabled:
                raise RuntimeError(
                    'Selected screen is not enabled. '
                    + 'Please enable it or select another one.'
                )
            elif scrn.screen_position != scrn.ScrnPosition.Fluorescent:
                raise RuntimeError(
                    'Selected screen is not in the fluorescent position. '
                    + 'Please move it to the correct position.'
                )
            pvo = scrn.screen.pv_object('ImgData-Mon')

        print('Starting measurement!')
        self.data = {}
        if self.params.acq_type == self.params.AcqType.Asynchronous:
            pvo.auto_monitor = True
            pvo.add_callback(self._launch_thread_cb)

        while not self._stopevt.is_set():
            if self.params.acq_type == self.params.AcqType.Synchronous:
                self._get_data_cb()
                _time.sleep(self.params.acq_interval)
            else:
                _time.sleep(0.5)

        if self.params.acq_type == self.params.AcqType.Asynchronous:
            pvo.auto_monitor = False
            pvo.clear_callbacks()

        print('Done!')

    def _launch_thread_cb(self, **kwargs):
        """."""
        _ = kwargs  # not used
        if self._thread_cb is None or not self._thread_cb.is_alive():
            self._thread_cb = _Thread(target=self._get_data_cb, daemon=True)
            self._thread_cb.start()
        else:
            print('Acquistion thread is taking longer than expected')

    def _get_data_cb(self):
        """."""
        if self.params.acq_type == self.params.AcqType.Asynchronous:
            _time.sleep(self.params.asyn_wait_after_screen)
        self._update_data_dict(self.get_data())

    def _update_data_dict(self, data_dict):
        """Update self.data with the values in the provided dictionary.

        Args:
            data_dict (dict): Dictionary containing parameter names and their
                corresponding values.
        """
        for k, v in data_dict.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)
