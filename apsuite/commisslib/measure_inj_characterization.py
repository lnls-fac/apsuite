"""Measurement class to characterize the injector system.

Intended to characterize the injector system baseline behavior as
well as its response to typical changes in its main knobs. This
characterization was thought to be performed before and after the
interventions in the TB transport line during the machine shutdown of
march/april 2026, related to rearrangements in the vacuum chamber ceramic
transition.

"""

import time

from siriuspy.devices import EVG, Screen
from siriuspy.epics import PV
from siriuspy.oscilloscope import Keysight, ScopeSignals

from apsuite.utils import ParamsBaseClass, ThreadedMeasBaseClass

KNOBS_PVNAMES = [
    'BO-01D:PU-InjKckr:Voltage-SP',
    'TB-04:PU-InjSept:Voltage-SP',
    'TB-04:PS-CH-1:Current-SP',
    'TB-04:PS-CV-1:Current-SP',
    'TB-04:PS-CV-2:Current-SP',
    'AS-Glob:AP-InjCtrl:MultBunBiasVolt-SP',
    # 'LA-RF:LLRF:BUN1:SET_AMP',  # SHB Amp
    # 'LA-RF:LLRF:KLY1:SET_AMP',  # Kly1 Amp
    'LA-RF:LLRF:KLY2:SET_AMP',  # KLY2 Amp
    # 'LA-RF:LLRF:KLY1:SET_PHASE',  # Kly1 Phase
    # 'LA-RF:LLRF:BUN1:SET_PHASE',  # SHB Amp
    # 'LA-RF:LLRF:KLY2:SET_PHASE',  # KLY2 Phase
]

OBSERVABLES_PVNAMES = [
    'TB-04:VA-PT100-ED1:Temp-Mon',  # Temp. Câmaras de vácuo do Septa da TB
    'TB-04:VA-PT100-ED2:Temp-Mon',  # Temp. Câmaras de vácuo do Septa da TB
    'TB-04:PU-InjSept-BG:Temp-Mon',  # Temp. do corpo do Septa
    'TB-04:PU-InjSept-ED:Temp-Mon',  # Temp. do corpo do Septa
]

KNOBS_NUDGES = {
    'BO-01D:PU-InjKckr:Voltage-SP': 0.5,  # [V]
    'TB-04:PU-InjSept:Voltage-SP': 0.5,  # [V]
    'TB-04:PS-CH-1:Current-SP': 0.25,  # [A]
    'TB-04:PS-CV-1:Current-SP': 0.25,  # [A]
    'TB-04:PS-CV-2:Current-SP': 0.25,  # [A]
    'AS-Glob:AP-InjCtrl:MultBunBiasVolt-SP': 1,  # [V]
    # 'LA-RF:LLRF:BUN1:SET_AMP': 0.5,  # [%]
    # 'LA-RF:LLRF:KLY1:SET_AMP': 1,  # [%]
    'LA-RF:LLRF:KLY2:SET_AMP': 0.5,  # [%]
    # 'LA-RF:LLRF:BUN1:SET_PHASE': 1,  # [deg]
    # 'LA-RF:LLRF:KLY1:SET_PHASE': 1,  # [deg]
    # 'LA-RF:LLRF:KLY2:SET_PHASE': 1,  # [deg]s
}


class InjCharacterizationParams(ParamsBaseClass):
    """."""

    def __init__(
        self,
        meas_type='acquire',
        knobs_pvnames=KNOBS_PVNAMES,
        observables_pvnames=OBSERVABLES_PVNAMES,
        knobs_nudges=KNOBS_NUDGES,
        smoke_test=True,
    ):
        """."""
        super().__init__()
        self.meas_type = meas_type
        self.bo_screen = Screen.DEVICES.BO_1  # TODO: screen enum
        # self.bo_screen = Screen.DEVICES.BO_2
        # self.bo_screen = Screen.DEVICES.BO_3
        self.knobs_pvnames = knobs_pvnames
        self.observables_pvnames = observables_pvnames
        self.knobs_nudges = knobs_nudges
        self.pvnames = self.knobs_pvnames + self.observables_pvnames
        self.smoke_test = smoke_test


class InjCharacterizationMeas(ThreadedMeasBaseClass):
    """."""

    def __init__(self, params=None, isonline=True):
        """."""
        super().__init__(
            params=params or InjCharacterizationParams(),
            target=self.do_measurement,
            isonline=isonline,
        )
        self.scopes = dict()  # scopes not in devices
        # because they fail the "connected" and related checks
        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """."""
        self.devices['evg'] = EVG()
        self.devices['bo_screen'] = Screen(self.params.bo_screen)
        self.scopes['tb_ict2_scope'] = Keysight(
            scopesignal=ScopeSignals.TB_ICT2
        )

        for pvname in self.params.pvnames:
            self.pvs[pvname] = PV(pvname)

    def get_delta_knob(self, knob_pvname):
        """."""
        val = self.params.knobs_nudges[knob_pvname]
        return val

    def set_delta_knob_and_get_data(self, knob_pvname, tag):
        """."""
        delta = self.get_delta_knob(knob_pvname)

        if tag.tolower() != 'pos':
            delta = -delta

        pv = self.pvs[knob_pvname]
        val0 = pv.value
        print(f'Setting {knob_pvname} from {val0} to {val0 + delta}.')
        if not self.params.smoke_test:
            pv.value = val0 + delta
            time.sleep(1)
        self.inject_and_get_data(knob_pvname, tag=tag)
        if not self.params.smoke_test:
            pv.value = val0
        time.sleep(3)
        print(f'Restored {knob_pvname} to {val0}.\n')

    def get_screen_data(self):
        """."""
        screen = self.devices['bo_screen']
        data = dict()
        data['devname'] = screen.devname
        data['image_raw'] = screen.image
        data['centerx'] = screen.centerx
        data['centery'] = screen.centery
        data['sigmax'] = screen.sigmax
        data['sigmay'] = screen.sigmay
        data['theta'] = screen.angle
        data['scalex'] = screen.scale_factor_x
        data['scaley'] = screen.scale_factor_y
        return data

    def get_scope_data(self):
        """."""
        scope = self.scopes['tb_ict2_scope']
        data = dict()
        data['t'], data['w'] = scope.wfm_get_data()
        return data

    def inject_and_get_data(self, knob_pvname=None, tag='baseline'):
        """."""
        if not self.params.smoke_test:
            print('Turning on injection...')
            self.devices['evg'].cmd_turn_on_injection()
        else:
            print('Smoke test: skipping injection.')
        data = self.data
        dt = data.setdefault(knob_pvname, {})
        # key = 'pos' if positive else 'neg'

        dt[tag] = {
            'timestamp': time.time(),
            'bo_screen': self.get_screen_data(),
            'tb_ict2_scope': self.get_scope_data(),
        }
        for pvname in self.params.pvnames:
            dt[tag][pvname] = self.pvs[pvname].value

    def do_measurement(self):
        """."""
        if self.params.meas_type == 'nudge':
            self.do_measurement_nudges()
        else:
            self.do_measurement_acquire()

    def do_measurement_nudges(self):
        """."""
        for knob_pvname in self.params.knobs_pvnames:
            print(f'Measuring {knob_pvname}')
            self.inject_and_get_data(knob_pvname=knob_pvname, tag='baseline')
            if self._stopevt.is_set():
                print('Measurement stopped!')
                break
            self.set_delta_knob_and_get_data(knob_pvname, tag='pos')
            if self._stopevt.is_set():
                print('Measurement stopped!')
                break
            self.set_delta_knob_and_get_data(knob_pvname, tag='neg')
            if self._stopevt.is_set():
                print('Measurement stopped!')
                break
            print(f'Finished {knob_pvname}.\n')
            print('#####################################################')
            print('\n')

    def do_measurement_acquire(self, dtime):
        """."""
        if not self.params.smoke_test:
            print('Turning on injection...')
            self.devices['evg'].cmd_turn_on_injection()
        else:
            print('Smoke test: skipping injection.')
        dt = list()
        while not self._stopevt.is_set():
            dic = dict()
            dic['timestamp'] = time.time()
            dic['bo_screen'] = self.get_screen_data()
            dic['tb_ict2_scope'] = self.get_scope_data()
            for pvname in self.params.pvnames:
                dic[pvname] = self.pvs[pvname].value
            dt.append(dic)
            time.sleep(dtime)
        return dt
