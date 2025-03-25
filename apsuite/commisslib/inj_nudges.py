import threading as _threading
import time as _time

import numpy as _np
from siriuspy.devices import EVG
from siriuspy.epics import PV

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _BaseClass


class InjNudgesParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.num_inj_samples = 3
        self.topup_current = 200  # [mA]
        self.pv_connection_timeout = 3
        self.knobs_lims = {
            #        knob                   [lim_low, lim_high]  unit
            "BO-01D:PU-InjKckr:Voltage-SP"   : [-1, 1],        # [V]
            "TB-04:PS-CH-1:Current-SP"       : [-0.25, 0.25],  # [A]
            "TB-04:PU-InjSept:Voltage-SP"    : [-1, 1],        # [V]
            "TB-04:PS-CV-1:Current-SP"       : [-0.25, 0.25],  # [A]
            "TB-04:PS-CV-2:Current-SP"       : [-0.25, 0.25],  # [A]
            "LA-RF:LLRF:BUN1:SET_AMP"        : [-0.5, 0.5],    # [%]
            "LA-RF:LLRF:KLY1:SET_AMP"        : [-1, 1],        # [%]
            "LA-RF:LLRF:KLY2:SET_AMP"        : [-1, 1],        # [%]
            "RA-RaBO01:RF-LLRF:RmpPhsBot-SP" : [-5, 5],        # [deg]
            "BO-Fam:PS-QD:WfmOffset-SP"      : [-0.02, 0.02],  # [A]
            "BO-Fam:PS-QF:WfmOffset-SP"      : [-0.02, 0.02],  # [A]
            "BO-Fam:PS-SD:WfmOffset-SP"      : [-0.05, 0.05],  # [A]
            "BO-Fam:PS-SF:WfmOffset-SP"      : [-0.05, 0.05],  # [A]
            "BO-Fam:PS-B-1:WfmOffset-SP"     : [-0.05, 0.05],  # [A]
            "BO-Fam:PS-B-2:WfmOffset-SP"     : [-0.05, 0.05],  # [A]
        }
        self.knobs_pvs_names = sorted(self.knobs.keys())
        self.injeff_pv_name = None
        self.si_curr_pv_name = "SI-Glob:AP-CurrInfo:Current-Mon"
        self.low_lims = [
            self.knobs_lims[kn][0] for kn in self.knobs_pvs_names
        ]
        self.upper_lims = [
            self.knobs_lims[kn][1] for kn in self.knobs_pvs_names
        ]


class InjNudges(_BaseClass):
    """."""
    def __init__(self, isonline=True):
        """."""
        self.params = InjNudgesParams
        super().__init__(
            self, params=self.params,
            target=self.do_measure, isonline=isonline
        )
        self.ref_injeff_mean = None
        self.ref_injeff_std = None

        if self.isonline:
            self.connect_pvs()

    def connect_pvs(self):
        """."""
        print("Is online. Connecting to PVs.")
        pvs_names = [self.params.injeff_pv_name]
        pvs_names += [self.params.si_curr_pv_name]
        pvs_names += self.params.knobs_pvs_names

        pvs = {}
        for pv_name in pvs_names:
            pv = PV(pv_name)
            pv.wait_for_connection(timeout=self.params.pv_connection_timeout)
            print(f"PV {pv_name:<35s} connected: {pv.connected}")
            pvs[pv_name] = pv

        self.pvs = pvs
        self.devices = {"evg": EVG()}

    def get_pos(self):
        """."""
        pos = list()
        for pv_name in self.params.knobs_pvs_names:
            pos.append(self.pvs[pv_name].value)
        return pos

    def set_pos(self, pos):
        """."""
        for i, pv_name in enumerate(self.params.knobs_pvs_names):
            self.pvs[pv_name].value = pos[i]

    def get_knob_nudge(self):
        """."""
        nudges = _np.random.uniform(
            low=self.params.low_lims, high=self.params.upper_lims
        )
        return nudges

    def acquire_efficiencies(self, ref_mean, ref_sigma):
        """."""
        print("\tacquiring injeffs...")
        injeff_pv = self.pvs[self.params.injeff_pv_name]
        tim, vals = [], []
        event = _threading.Event()
        injeff_pv.add_callback(
            self, self.on_change, tim=tim,
            vals=vals, event=event,
            ref_mean=ref_mean, ref_sigma=ref_sigma)
        event.wait()
        injeff_pv.clear_callbacks()
        mean, sigma = _np.mean(vals), _np.std(vals)

        print(f"\t\tmean injeff {mean:3.2f} +- {sigma:.2f} %")
        print("\tacquisitions finished.")
        return tim, vals, mean, sigma

    def on_change(
            self, timestamp, value, tim, vals, event,
            ref_mean, ref_sigma, num_samples, **kwargs
    ):
        """Callback function to be added to the injeff PV."""
        print(f"\t\tinjeff now at {value:3.2f} %")
        tim.append(timestamp)
        vals.append(value)
        if len(vals) >= num_samples:
            event.set()
        if None in (ref_mean, ref_sigma):
            return
        if len(vals) > 1 and (_np.mean(vals) < ref_mean - ref_sigma):
            event.set()

    def recover_topup(self, pos_good, sleep_time=30):
        """."""
        print("")
        print(
            f"SI current below {self.params.topup_curent}. Recovering top-up."
        )
        self.set_pos(pos_good)
        si_curr = self.pvs[self.params.si_curr_pv_name]
        curr = si_curr.value
        while curr < 180:
            _time.sleep(sleep_time)
            curr = si_curr.value
        else:
            print(f"\tcurrent at {curr:3.2f} mA.")
            print("\tcurrent restored. Continuing nudges.")

    def do_measure(self):
        """."""
        self.data["timestamps"] = list()
        self.data["values"] = list()
        self.data["positions"] = list()

        print("Starting injection nudging loop.\n")
        print("Efficiency acquisitions for initial reference")

        tim, vals, ref_mean, ref_sigma = self.acquire_efficiencies()

        pos = self.get_pos()

        while True:

            delta_pos = self.get_knob_nudge()

            i = _np.random.randint(len(delta_pos))
            dposi = delta_pos[i]
            knobi_name = self.params.knobs_pvs_names[i]
            print("")
            print(f"Nudging {knobi_name} in {dposi:2.2f}")
            posi0 = pos[i]
            pos[i] += dposi

            evg = self.devices["evg"]
            if evg.injection_state:
                print("\tinjection is on. Waiting for it to finish...")
                evg.wait_injection_finish()

            if self._stopevt.is_set():
                print("Stop event was set.")
                self.finish_meas()

            print("\tsetting new position")
            self.set_pos(pos)

            tim, vals, mean_injeff, sigma_injeff = self.acquire_efficiencies(
                ref_mean=ref_mean, ref_sigma=ref_sigma
            )

            self.data["timestamps"].append(tim)
            self.data["values"].append(vals)
            self.data["positions"].append(pos)

            if mean_injeff >= ref_mean:
                print(f"\tchanges in {knobi_name} improved injeff!")
                ref_mean, ref_sigma = mean_injeff, sigma_injeff
                print(f"\tnew reference injeff: {ref_mean:3.3f} %")
            else:
                print(f"\treverting changes in {knobi_name}...")
                pos[i] = posi0
                self.set_pos(pos)

            si_curr = self.pvs[self.params.si_curr_pv_name].value
            if si_curr < self.params.topup_current:
                self.recover_topup(pos, sleep_time=10)

            self.save_data(fname="inj_nudges", overwrite=True)

    def finish_meas(self):
        """."""
        print("Finishing measurement.")
        self.save_data(fname="inj_nudges", overwrite=True)
        self.pvs[self.params.injeff_pv_name].clear_callbacks()
        return
