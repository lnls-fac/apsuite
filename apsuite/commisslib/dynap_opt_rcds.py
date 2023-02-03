"""."""
import time as _time
import numpy as _np

from pymodels import si as _si
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, EGTriggerPS, RFGen, InjSysPUModeHandler

from ..optimization.rcds import RCDS as _RCDS
from ..optics_analysis import ChromCorr


class OptimizeDA(_RCDS):
    """."""

    SEXT_FAMS = (
        'SDA0', 'SDB0', 'SDP0',
        'SFA0', 'SFB0', 'SFP0',
        'SDA1', 'SDB1', 'SDP1',
        'SDA2', 'SDB2', 'SDP2',
        'SDA3', 'SDB3', 'SDP3',
        'SFA1', 'SFB1', 'SFP1',
        'SFA2', 'SFB2', 'SFP2')
    SEXT_FAMS_ACHROM = SEXT_FAMS[:6]
    SEXT_FAMS_CHROM = SEXT_FAMS[6:]

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _RCDS.__init__(self, isonline=isonline, use_thread=use_thread)
        self.sextupoles = []
        if self.isonline:
            self._create_devices()
        self.chrom_corr = ChromCorr(_si.create_accelerator(), acc='SI')

        self.full_chrom_mat = _np.zeros((2, len(self.SEXT_FAMS)), dtype=float)

        idcs = [self.SEXT_FAMS.index(sx) for sx in self.chrom_corr.knobs.all]
        self.full_chrom_mat[:, idcs] = self.chrom_corr.calc_jacobian_matrix()

        self.names_sexts2corr = [
            'SDA2', 'SDB2', 'SDP2', 'SFA2', 'SFB2', 'SFP2']
        self.names_sexts2use = []
        for sext in self.SEXT_FAMS:
            if sext in self.names_sexts2corr:
                continue
            self.names_sexts2use.append(sext)
        self.phase_offset = 0 #  to be determined during the experiment
        self.data['phase_offset'] = self.phase_offset

    def initialization(self, init_obj_func=-50):
        """."""
        if self.isonline:
            self.data['timestamp'] = _time.time()
            self.data['strengths'] = [self.get_strengths_from_machine(), ]
            self.data['obj_funcs'] = [init_obj_func, ]
            self._prepare_evg

    def objective_function(self, pos, offaxis_weigth=1, onaxis_weigth=1):
        """."""
        if not offaxis_weigth and not onaxis_weigth:
            raise ValueError('At least one weigth must be nonzero')

        evg = self.devices['evg']
        currinfo = self.devices['currinfo']
        nlk = self.devices['nlk']
        rfgen = self.devices['rfgen']
        injsys = self.devices['injsys']

        strengths = self.get_isochrom_strengths(pos)
        self.set_strengths_to_machine(strengths)
        self.data['strengths'].append(strengths)
        _time.sleep(1)

        objective = 0
        if not offaxis_weigth:
            injsys.cmd_switch_to_optim()
            # nlk_ref = nlk.strength
            # nlk.strength = - 2.25 * 1e-3
            inj0 = currinfo.injeff
            evg.cmd_turn_on_injection()
            for _ in range(50): # what about evg.wait_injection_finish(timeout=5)?
                if inj0 != currinfo.injeff:
                    break
                _time.sleep(0.1)
            objective += offaxis_weigth * currinfo.injeff
            # nlk.strength = nlk_ref
            # then kill the beam?

        if not onaxis_weigth:
            injsys.cmd_switch_to_onaxis()
            # need to config?
            phase_ref = rfgen.phase
            rfgen.phase = self.phase_offset #  to be determined
            inj0 = currinfo.injeff
            evg.cmd_turn_on_injection()
            for _ in range(50):
                if inj0 != currinfo.injeff:
                    break
                _time.sleep(0.1)
            rfgen.phase = phase_ref
            objective += onaxis_weigth * currinfo.injeff

        objective /= offaxis_weigth + onaxis_weigth
        self.data['obj_funcs'].append(objective)

        return objective

    def _prepare_evg(self):
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)

    def objective_function_(self, pos):
        """."""
        evg = self.devices['evg']

        if not self.inject_beam(min_curr=self.params.min_stored_current):
            raise ValueError(
                'It was not possible to inject beam in the machine.')

        strengths = self.get_isochrom_strengths(pos)
        self.set_strengths_to_machine(strengths)
        self.data['strengths'].append(strengths)

        _time.sleep(1)
        bpms = self.devices['bpms']
        bpms.mturn_reset_flags()

        evg.cmd_turn_on_injection()
        bpms.mturn_wait_update_flags()

        psum = bpms.get_mturn_orbit(return_sum=True)[2].mean(axis=1)
        sum0 = _np.mean(psum[0:10])
        sumf = _np.mean(psum[-10:])

        loss = 1 - sumf/sum0
        loss = max(min(loss, 1), 0)

        return loss*100

    def inject_beam(self, min_curr=2.0, timeout=10):
        """Inject current in the storage ring.

        Args:
            min_curr (float, optional): Desired current in [mA].
                Defaults to 2.0 mA.
            timeout (int, optional): Maximum time to wait injection in [s].
                Defaults to 10 s.

        """
        evg, currinfo = self.devices['evg'], self.devices['currinfo']
        egun = self.devices['egun']
        pingh, pingv = self.devices['pingh'], self.devices['pingv']
        nlk = self.devices['nlk']
        if currinfo.current >= min_curr:
            return True

        evg.nrpulses = 0
        nlk.pulse = True
        egun.enable = True
        pingh.pulse = False
        pingv.pulse = False
        currp = self.get_strengths_from_machine()
        self.set_strengths_to_machine(self.data['strengths'][0])
        _time.sleep(0.1)
        evg.cmd_turn_on_injection()
        niter = int(timeout/0.5)
        for _ in range(niter):
            _time.sleep(0.5)
            if currinfo.current >= min_curr:
                break
        evg.cmd_turn_off_injection()
        _time.sleep(0.5)

        self.set_strengths_to_machine(currp)
        evg.nrpulses = 1
        nlk.pulse = False
        egun.enable = False
        pingh.pulse = True
        pingv.pulse = True
        _time.sleep(0.1)
        return currinfo.current >= min_curr

    def get_isochrom_strengths(self, pos):
        """."""
        idcs_corr = [self.SEXT_FAMS.index(sx) for sx in self.names_sexts2corr]
        idcs_2use = [self.SEXT_FAMS.index(sx) for sx in self.names_sexts2use]

        mat2corr = self.full_chrom_mat[:, idcs_corr]
        imat2corr = _np.linalg.pinv(mat2corr, rcond=1e-15)

        mat2use = self.full_chrom_mat[:, idcs_2use]

        str0 = self.data['strengths'][0].copy()
        pos0 = str0[idcs_2use]
        dpos = pos - pos0
        dchrom = mat2use @ dpos
        dcorr = - imat2corr @ dchrom

        strengths = _np.full(len(self.SEXT_FAMS), _np.nan)
        strengths[idcs_2use] = pos
        strengths[idcs_corr] = dcorr + str0[idcs_corr]
        return strengths

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        if pos is None:
            pos = self.params.initial_position
        obj = []
        for i in range(nr_evals):
            obj.append(self.objective_function(pos))
            print(f'{i+1:02d}/{nr_evals:02d}  --> obj. = {obj[-1]:.3f}')
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_current_position(self, return_limits=False):
        """Return current strengths of sextupoles used by RCDS.

        Returns:
            numpy.ndarray (N, 1): strengths of the N sextupoles used in
                optimization.

        """
        stren, (lower0, upper0) = self.get_strengths_from_machine(
            return_limits=True)
        pos, lower, upper = [], [], []
        for sxt, stg, low, upp in zip(self.SEXT_FAMS, stren, lower0, upper0):
            if sxt in self.names_sexts2use:
                pos.append(stg)
                lower.append(low)
                upper.append(upp)
        if not return_limits:
            return _np.array(pos)
        return _np.array(pos), (_np.array(lower), _np.array(upper))

    def get_strengths_from_machine(self, return_limits=False):
        """."""
        val, lower, upper = [], [], []
        for sxt in self.sextupoles:
            val.append(sxt.strengthref_mon)
            if not return_limits:
                continue
            lims = sxt.pv_object('SL-Mon').get_ctrlvars()
            upper.append(lims['upper_disp_limit'])
            lower.append(lims['lower_disp_limit'])
        if not return_limits:
            return _np.array(val)
        return _np.array(val), (_np.array(lower), _np.array(upper))

    def set_strengths_to_machine(self, strengths):
        """."""
        if len(strengths) != len(self.sextupoles):
            raise ValueError(
                'Length of strengths must match number of sextupole families.')

        for i, stg in enumerate(strengths):
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = stg
        _time.sleep(2)

    def _create_devices(self):
        for fam in self.SEXT_FAMS:
            sext = PowerSupply('SI-Fam:PS-'+fam)
            self.devices[fam] = sext
            self.sextupoles.append(sext)

        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['nlk'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
        self.devices['pingv'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['evt_study'] = Event('Study')
        self.devices['egun'] = EGTriggerPS()
        self.devices['rfgen'] = RFGen()
        self.devices['injsys'] = InjSysPUModeHandler()
