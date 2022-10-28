"""."""
import time as _time
import numpy as _np

from pymodels import si as _si
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, FamBPMs

from ..utils import ThreadedMeasBaseClass as _BaseClass
from ..optimization.rcds import RCDS as _RCDS
from ..optics_analysis import ChromCorr


class OptimizeDA(_RCDS, _BaseClass):
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
        _BaseClass.__init__(self, isonline=isonline)
        _RCDS.__init__(self, use_thread=use_thread)
        self.sextupoles = []
        if self.isonline:
            self._create_devices()
        self.chrom_corr = ChromCorr(_si.create_accelerator(), acc='SI')

        self.full_chrom_mat = _np.zeros((len(self.SEXT_FAMS), 2), dtype=float)

        idcs = [self.SEXT_FAMS.index(sx) for sx in self.chrom_corr.knobs]
        self.full_chrom_mat[:, idcs] = self.chrom_corr.calc_jacobian_matrix()

        self.names_sexts2corr = [
            'SDA2', 'SDB2', 'SDP2', 'SFA2', 'SFB2', 'SFP2']
        self.names_sexts2use = []
        for sext in self.SEXT_FAMS:
            if sext in self.names_sexts2corr:
                continue
            self.names_sexts2use.append(sext)

    def initialization(self):
        """."""
        if self.isonline:
            self.data['timestamp'] = _time.time()
            self.data['strengths'] = [self.get_strengths_from_machine(), ]

    def objective_function(self, pos):
        """."""
        evt_study = self.devices['evt_study']

        if not self.inject_beam(min_curr=self.params.min_stored_current):
            raise ValueError(
                'It was not possible to inject beam in the machine.')

        strengths = self.get_isochrom_strengths(pos)
        self.set_strengths_to_machine(strengths)
        self.data['strengths'].append(strengths)

        _time.sleep(1)
        bpms = self.devices['bpms']
        bpms.mturn_reset_flags()

        evt_study.cmd_external_trigger()
        bpms.mturn_wait_update_flags()

        psum = bpms.get_mturn_orbit(return_sum=True)[2].mean(axis=1)
        sum0 = _np.mean(psum[0:10])
        sumf = _np.mean(psum[-10:])

        loss = 1 - sum0/sumf
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
        if currinfo.current >= min_curr:
            return True

        evg.cmd_turn_on_injection()
        niter = int(timeout/0.5)
        for _ in range(niter):
            _time.sleep(0.5)
            if currinfo.current >= min_curr:
                break
        evg.cmd_turn_off_injection()
        _time.sleep(0.5)

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
        for _ in range(nr_evals):
            obj.append(self.objective_function(pos))
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_current_position(self):
        """Return current strengths of sextupoles used by RCDS.

        Returns:
            numpy.ndarray (N, 1): strengths of the N sextupoles used in
                optimization.

        """
        strengths0 = self.get_strengths_from_machine()  # machine knobs
        pos0 = []
        for sxt, stg in zip(self.SEXT_FAMS, strengths0):
            if sxt in self.names_sexts2use:
                pos0.append(stg)
        return _np.array(pos0)

    def get_strengths_from_machine(self):
        """."""
        return _np.array([sx.strengthref_mon for sx in self.sextupoles])

    def set_strengths_to_machine(self, strengths):
        """."""
        if len(strengths) != len(self.sextupoles):
            raise ValueError(
                'Length of strengths must match number of sextupole families.')

        for i, stg in strengths:
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = stg

    def _create_devices(self):
        for fam in self.SEXT_FAMS:
            sext = PowerSupply('SI-Fam:PS-'+fam)
            self.devices[fam] = sext
            self.sextupoles.append(sext)

        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['pingv'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['bpms'] = FamBPMs()
        self.devices['evt_study'] = Event('Study')

    def save_optimization_data(self, fname):
        """."""
        self.data['best_positions'] = self.best_positions
        self.data['best_objfuncs'] = self.best_objfuncs
        self.data['final_search_directions'] = self.final_search_directions
        self.save_data(fname)
