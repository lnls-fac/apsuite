"""Module for online nudging of injection efficiency knobs."""
import threading as _threading
import time as _time

import numpy as _np
from siriuspy.clientarch import Time as _Time
from siriuspy.devices import EVG as _EVG
from siriuspy.epics import PV as _PV

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _BaseClass

BOINJEFF_PVNAME = "BO-Glob:AP-CurrInfo:RampEff-Mon"
BOINJ_KNOBS = {
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

BOINJ_OBSERVABLES = [
    "LINAC:Temperatura-Mon",  # LINAC env temp
    "LA-CN:H1MPS-1:A1Temp1",  # Accelerating strurctures
    "LA-CN:H1MPS-1:K1Temp1",  # (K1 body)
    "LA-CN:H1MPS-1:K1Temp2",  # (K1 Waveguide)
    "LA-CN:H1MPS-1:K2Temp1",  # (K2 body)
    "LA-CN:H1MPS-1:K2Temp2",  # (K2 Waveguide)
    "TB-04:PU-InjSept-BG:Temp-Mon",   # septa
    "TB-04:PU-InjSept-ED:Temp-Mon",   # septa
]

SIINJEFF_PVNAME = "SI-Glob:AP-CurrInfo:InjEff-Mon"
SIINJ_KNOBS = {
    #        knob                   [lim_low, lim_high]  unit
    "BO-48D:PU-EjeKckr:Voltage-SP"    : [-1, 1],        # [V]
    "TS-01:PU-EjeSeptF:Voltage-SP"    : [-1, 1],        # [V]
    "TS-01:PU-EjeSeptG:Voltage-SP"    : [-1, 1],        # [V]
    "TS-04:PU-InjSeptF:Voltage-SP"    : [-1, 1],        # [V]
    "TS-04:PU-InjSeptG-1:Voltage-SP"  : [-1, 1],        # [V]
    "TS-04:PU-InjSeptG-2:Voltage-SP"  : [-1, 1],        # [V]
    "SI-01SA:PU-InjNLKckr:Voltage-SP" : [-1, 1],        # [V]
    "TS-04:PS-CV-0:Current-SP"        : [-0.25, 0.25],  # [A] (check)
    "TS-04:PS-CV-1:Current-SP"        : [-0.25, 0.25],  # [A]
    "TS-04:PS-CV-1E2:Current-SP"      : [-0.25, 0.25],  # [A]
    "TS-04:PS-CV-2:Current-SP"        : [-0.25, 0.25],  # [A]
    "RA-RaBO01:RF-LLRF:RmpPhsTop-SP"  : [-5, 5],        # [deg]
}

SIINJ_OBSERVABLES = [
    # to be implemented
]


class InjNudgesBaseParams(_ParamsBaseClass):
    """."""
    def __init__(self):
        """."""
        super().__init__()
        self.pv_connection_timeout = 3
        self.num_inj_samples = 3
        self.num_attempts_per_ref = 5
        self.min_topup_current = 198.5  # [mA]
        self.filename2use = "inj_nudges"

        self._knobs_lims = None
        self.knobs_pvsnames = None
        self.low_lims = None
        self.upper_lims = None
        self.si_curr_pvname = "SI-Glob:AP-CurrInfo:Current-Mon"
        self.observables_pvs_names = ""

    @property
    def knobs_lims(self):
        """."""
        return self._knobs_lims

    @knobs_lims.setter
    def knobs_lims(self, lims):
        """."""
        if not isinstance(lims, dict):
            raise ValueError("knobs_lims must be a dictionary")
        self._knobs_lims = lims
        self.knobs_pvsnames = sorted(self._knobs_lims.keys())
        self.low_lims = [
            self._knobs_lims[kn][0] for kn in self.knobs_pvsnames
        ]
        self.upper_lims = [
            self._knobs_lims[kn][1] for kn in self.knobs_pvsnames
        ]

    def __str__(self):
        """."""
        ftmp = "{0:24s} = {1:9.3f}  {2:s}\n".format
        dtmp = "{0:24s} = {1:9d}  {2:s}\n".format
        stmp = "{0:24s} = {1:9s}  {2:s}\n".format

        stg = ""
        stg += ftmp(
            "pv_connection_timeout", self.pv_connection_timeout, "[s]"
        )
        stg += dtmp("num_inj_samples", self.num_inj_samples, "")
        stg += dtmp("num_attempts_per_ref", self.num_attempts_per_ref, "")
        stg += ftmp("min_topup_current", self.min_topup_current, "mA")
        stg += stmp("filename2use", self.filename2use, "")
        stg += stmp("injeff_pvname", self.injeff_pvname, "")
        stg += "\n"
        stg += "        knob                   [lim_low, lim_high]\n"
        for kn in self.knobs_pvsnames:
            stg += f"{kn:<35s} : {self.knobs_lims[kn]}\n"

        stg += "\n"
        stg += "other observables\n"
        for pvname in self.observables_pvs_names:
            stg += f"{pvname:<30s}"
        return stg


class BOInjNudgesParams(InjNudgesBaseParams):
    """."""
    def __init__(self):
        """."""
        super().__init__()
        self.injeff_pvname = BOINJEFF_PVNAME
        self.knobs_lims = BOINJ_KNOBS
        self.observables_pvs_names = BOINJ_OBSERVABLES


class SIInjNudgesParams(InjNudgesBaseParams):
    """."""
    def __init__(self):
        """."""
        super().__init__()
        self.injeff_pvname = SIINJEFF_PVNAME
        self.knobs_lims = SIINJ_KNOBS
        self.observables_pvs_names = SIINJ_OBSERVABLES


class InjNudges(_BaseClass):
    """."""
    def __init__(self, isonline=True, inj_system="BO"):
        """."""
        super().__init__(
            self, target=self.do_measure, isonline=isonline
        )
        if inj_system.lower() == "bo":
            self.params = BOInjNudgesParams()
        elif inj_system.lower() == "si":
            self.params = SIInjNudgesParams()
        else:
            raise ValueError("Injection system must be SI or BO.")

        self.ref_injeff_mean = None
        self.ref_injeff_std = None

        if self.isonline:
            self.connect_pvs_and_evg()

    def connect_pvs_and_evg(self):
        """."""
        print("Is online. Connecting to PVs.")
        pvs_names = [self.params.injeff_pvname]
        pvs_names += [self.params.si_curr_pvname]
        pvs_names += self.params.knobs_pvsnames
        pvs_names += self.params.observables_pvs_names

        pvs = {}
        for pv_name in pvs_names:
            pv = _PV(pv_name)
            pv.wait_for_connection(timeout=self.params.pv_connection_timeout)
            print(f"PV {pv_name:<35s} connected: {pv.connected}")
            pvs[pv_name] = pv

        self.pvs = pvs
        self.devices = {"evg": _EVG()}

    def get_pos(self):
        """Returns knobs positions as a dict of (timestamp, value)."""
        vals = self.get_pvs_tmtsp_and_vals()
        pos = {
            pvname: vals[pvname] for pvname in self.params.knobs_pvsnames
        }
        return pos

    def get_pvs_tmtsp_and_vals(self):
        """."""
        vals = dict()
        for pvname, pv in self.pvs.items():
            vals[pvname] = [pv.timestamp, pv.value]
        return vals

    def set_pos(self, pos):
        """."""
        for pv_name in self.params.knobs_pvsnames:
            _, value = pos[pv_name]
            pv = self.pvs[pv_name]
            if value != pv.value:
                pv.value = value

    def get_knobs_nudges(self):
        """."""
        nudges = _np.random.uniform(
            low=self.params.low_lims, high=self.params.upper_lims
        )
        return nudges

    def acquire_efficiencies(self, ref_mean=None, ref_sigma=None):
        """."""
        print("\tacquiring inj. effs. ...")

        injeff_pv = self.pvs[self.params.injeff_pvname]
        tim, effs, pvvals = [], [], []
        event = _threading.Event()  # signals acqs. are finished

        injeff_pv.add_callback(
            self, self.on_change, tim=tim, effs=effs, pvvals=pvvals,
            event=event, ref_mean=ref_mean, ref_sigma=ref_sigma
        )
        event.wait(timeout=(60 + 2) * self.params.num_inj_samples)
        injeff_pv.clear_callbacks()

        mean, sigma = _np.mean(effs), _np.std(effs)

        print(f"\t\tmean injeff {mean:3.2f} +- {sigma:.2f} %")
        print("\tacquisitions finished.")
        return tim, effs, mean, sigma, pvvals

    def on_change(
            self, timestamp, value, tim, effs, pvvals, event,
            ref_mean, ref_sigma, **kwargs
    ):
        """Callback function to be added to the injeff PV."""
        tmstp = _Time.fromtimestamp(_time.time())
        datetime = tmstp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\t\t{datetime} inj. eff.: {value:3.2f}%, SI curr.:")

        pvvals.append(self.get_pvs_tmtsp_and_vals())
        tim.append(timestamp)
        effs.append(value)

        if len(effs) >= self.params.num_inj_samples:
            event.set()
        if None in (ref_mean, ref_sigma):
            return  # What to do?
        if len(effs) > 1 and (_np.mean(effs) < ref_mean - ref_sigma):
            event.set()

    def recover_topup(self, pos_good, sleep_time=30):
        """."""
        print("")
        print(
            f"SI current below {self.params.topup_curent}. Recovering top-up."
        )
        self.set_pos(pos_good)
        si_curr = self.pvs[self.params.si_curr_pvname]
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
        self.data["effs"] = list()
        self.data["pvvals"] = list()
        self.data["positions"] = list()

        print("Starting injection nudging loop.\n")
        print("Efficiency acquisitions for initial reference")

        ret = self.acquire_efficiencies()
        if ret is None:
            print("No samples for initial inj. eff. reference.")
            raise ValueError
        tim, effs, ref_mean, ref_sigma, pvvals = ret

        pos = self.get_pos()
        fails_w_same_ref = 0
        evg = self.devices["evg"]

        while True:

            delta_pos = self.get_knobs_nudges()

            i = _np.random.randint(len(delta_pos))
            dposi = float(delta_pos[i])
            knobi_name = self.params.knobs_pvsnames[i]
            print("")
            print(f"Nudging {knobi_name} in {dposi:2.2f}")
            posi0 = pos[knobi_name].copy()
            pos[knobi_name][1] += dposi

            if evg.injection_state:
                print("\tinjection is on. Waiting for it to finish...")
                evg.wait_injection_finish()

            if self._stopevt.is_set():
                print("Stop event was set.")
                # TODO: handle stops during self.acquire_efficiencies
                # & self.on_change
                self.finish_meas()

            print("\tsetting new position")
            self.set_pos(pos)

            ret = self.acquire_efficiencies(
                ref_mean=ref_mean, ref_sigma=ref_sigma
            )
            if ret is None:
                print("None returned. Starting over again.")
                self.set_pos(pos)
                continue

            tim, effs, mean_injeff, sigma_injeff, pvvals = ret
            self.data["timestamps"].append(tim)
            self.data["effs"].append(effs)
            self.data["positions"].append(pos)
            self.data["pvvals"] = pvvals

            if mean_injeff - sigma_injeff >= ref_mean - ref_sigma:
                print(f"\tchanges in {knobi_name} improved injeff!")
                ref_mean, ref_sigma = mean_injeff, sigma_injeff
                fails_w_same_ref = 0
                print(f"\tnew reference injeff: {ref_mean:3.3f} %")
            else:
                print(f"\treverting changes in {knobi_name}...")
                fails_w_same_ref += 1
                pos[i] = posi0
                self.set_pos(pos)

                if fails_w_same_ref > self.params.num_attempts_per_ref:
                    t = "\nSame references used for too long."
                    t += "Updating references"
                    print(t)

                    ret = self.acquire_efficiencies()
                    if ret is not None:
                        _, _, ref_mean, ref_sigma, _ = ret
                        fails_w_same_ref = 0
            si_curr = self.pvs[self.params.si_curr_pvname].value
            if si_curr < self.params.topup_current:
                self.recover_topup(pos, sleep_time=10)

            self.save_data(fname=self.params.filename2use, overwrite=True)

    def finish_meas(self):
        """."""
        print("Finishing measurement.")
        self.save_data(fname=self.params.filename2use, overwrite=True)
        # TODO: automatic naming convention if filename2use is None
        self.pvs[self.params.injeff_pvname].clear_callbacks()
        return
