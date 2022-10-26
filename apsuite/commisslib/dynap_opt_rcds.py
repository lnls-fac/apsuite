"""."""
import time as _time
from kiwisolver import strength
import numpy as _np

from pymodels import si as _si
import pyaccel as _pa

from ..utils import ThreadedMeasBaseClass as _BaseClass
from siriuspy.devices import PowerSupply, PowerSupplyPU, Tune, CurrInfoSI, \
     EVG, SOFB
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
        self.chrom_corr = ChromCorr(
            _si.create_accelerator(), acc='SI',
            sf_knobs=['SFA2', 'SFB2', 'SFP2'],
            sd_knobs=['SDA2', 'SDB2', 'SDP2'],
            method=ChromCorr.METHODS.Proportional,
            grouping=ChromCorr.GROUPING.TwoKnobs)

        self.full_chrom_mat = None
        self.corr_chrom_mat = self.chrom_corr.calc_jacobian_matrix()
        U, s, Vt = _np.linalg.svd(self.corr_chrom_mat, full_matrices=False)
        self.corr_chrom_pseudoinv = Vt.T / s @ U.T

        self.corr_sexts = set(self.chrom_corr.knobs.all)
        self.names_sexts2use = []
        for sext in self.SEXT_FAMS:
            if sext in self.corr_sexts:
                continue
            self.names_sexts2use.append(sext)

    def objective_function(self, pos):
        """."""
        strengths = self.get_isochromatic_strengths(pos)
        self.set_strengths_to_machine(strengths)

        evg, currinfo = self.devices['evg'], self.devices['currinfo']
        evg.cmd_turn_on_injection()
        _time.sleep(1)
        evg.wait_injection_finish()
        _time.sleep(0.5)
        return -currinfo.injeff

    def get_isochrom_strengths(self, pos):
        """."""
        if self.full_chrom_mat is None:
            chrom_corr = ChromCorr(
                self.chrom_corr._acc, acc='SI',
                sf_knobs=self.SEXT_FAMS[6:15],
                sd_knobs=self.SEXT_FAMS[15:],
                method=ChromCorr.METHODS.Proportional,
                grouping=ChromCorr.GROUPING.TwoKnobs)
            self.full_chrom_mat = chrom_corr.calc_jacobian_matrix()

        stg0 = self.get_strengths_from_machine()[6:]  # achroms not included
        pos_vec = _np.array([
            pos[6:9].ravel(), _np.zeros(3).ravel(), pos[12:18].ravel(),
            _np.zeros(3).ravel()])  # opt knobs w/ blank space for corr knobs
        opt_deltaSL = pos_vec - stg0
        deltaChrom = self.full_chrom_mat @ opt_deltaSL
        corr_deltaSL = - self.corr_chrom_pseudoinv @ deltaChrom
        strengths = _np.array([pos[:6].ravel(), pos_vec.ravel()])
        strengths[9:12] += corr_deltaSL[:3]
        strengths[18:21] += corr_deltaSL[3:]
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

    def get_strengths_from_machine(self):
        """."""
        return _np.array([sext.strengthref_mon for sext in self.sextupoles])

    def set_strengths_to_machine(self, strengths):
        """."""
        for i, stg in strengths:
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
        self.devices['toca'] = SOFB(SOFB.DEVICES.SI)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)

    def _prepare_evg(self):
        # injection scheme?
        evg = self.devices['evg']
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)
