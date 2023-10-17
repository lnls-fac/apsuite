"""."""
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, Tune
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from pymodels import si
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat

class Shift_SOFB_nr_svals_Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points_sofb = 20  # [Hz]
        self.nr_iters = 10
        self.target_nr_svals = 200 # best (from previous studies)
        self.ch_reduce_factor = 0.70 # multiply
        self.cv_reduce_factor = 0.70 # multiply
        self.rf_reduce_factor = -200 # [Hz] sum
        self.ref_orbit = None

    def __str__(self):
        """."""
        # dtmp = '{0:25s} = {1:9d}\n'.format
        # ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        # stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = f'target nr svals: {self.target_nr_svals:3d}'
        return stg
    
class Shift_SOFB_nr_svals(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)
        self.model = si.create_accelerator()
        self.fam_data = si.get_family_data(self.model)
        self.orm = OrbRespmat(model=self.model, acc='SI', dim='6d')
        self.inverse_orm = None
        if self.isonline:
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)

        self.params.ref_orbit = _np.r_[self.devices['sofb'].refx, self.devices['sofb'].refy]

    def get_orbit(self):
        """."""
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        return _np.r_[sofb.orbx, sofb.orby]
    
    def get_kicks(self): # unused
        """."""
        sofb = self.devices['sofb']
        return _np.hstack((
            sofb.kickch, sofb.kickcv, self.kickrf
        ))

    def _meas_func(self):
        tune = self.devices['tune']
        self.inverse_orm = self.invert_dispmat(self.orm, self.params.target_nr_svals)

        data = {}
        data['timestamp'] = _time.time()
        data['tunex_history'] = []
        data['tuney_history'] = []
        data['xkicks_history'] = []
        data['ykicks_history'] = []
        data['rf_freq_history'] = []
        data['orbx_history'] = []
        data['orby_history'] = []

        # do nr_svals shift

        for i in range(self.params.nr_iters):
            orb = self.get_orbit()
            kicks = self.get_kicks()
            data['orbx_history'].append(orb[:160])
            data['orby_history'].append(orb[160:])
            data['tunex_history'].append(tune.tunex)
            data['tuney_history'].append(tune.tuney)
            data['rf_freq_history'].append(kicks[-1])
            data['xkicks_history'].append(kicks[:120])
            data['ykicks_history'].append(kicks[120:-1])
            self.svals_shift_single_iter()

        orb = self.get_orbit()
        kicks = self.get_kicks()
        data['orbx_history'].append(orb[:160])
        data['orby_history'].append(orb[160:])
        data['tunex_history'].append(tune.tunex)
        data['tuney_history'].append(tune.tuney)
        data['rf_freq_history'].append(kicks[-1])
        data['xkicks_history'].append(kicks[:120])
        data['ykicks_history'].append(kicks[120:-1])

        self.data = data

    @staticmethod
    def invert_dispmat(dispmat, nr_svals=None):
        """."""
        mat = dispmat
        mat[:, -1] *= 1e10 # force rf sval to be present
        umat, smat, vhmat = _np.linalg.svd(mat, full_matrices=False)
        ismat = 1/smat
        if nr_svals is not None:
            ismat[nr_svals:] = 0
        inv_dispmat = vhmat.T @ _np.diag(ismat) @ umat.T
        inv_dispmat[-1, :] *= 1e10
        return inv_dispmat

    def svals_shift_single_iter(self):
        """Procedure for shifting machine-SOFB-state to other nr_svals"""
        # reduce strengs
        self.reduce_strens()
        # calc correction
        deltach, deltacv, deltarf = self.calc_correction()
        # apply kicks    
        self.apply_delta_kicks(deltach, deltacv, deltarf)

    def reduce_strens(self):
        """ Reduce correctors strenghts to force an orbit distotion"""
        sofb = self.devices['sofb']
        kickch = sofb.kickch
        kickcv = sofb.kickcv
        # set machine new kicks to distort orbit
        self.apply_delta_kicks(kickch*(1 - self.params.ch_reduce_factor), 
                               kickcv*(1 - self.params.cv_reduce_factor),
                               self.params.rf_reduce_factor)

    def calc_correction(self):
        orb = self.get_orb()
        dorb = orb - self.params.ref_orbit
        dkicks = (-1) * _np.dot(self.inverse_orm, dorb)
        return dkicks[:120], dkicks[120:-1], dkicks[-1]
    
    def apply_delta_kicks(self, dkch, dkcv, dkrf):
        """Apply kicks"""
        sofb = self.devices['sofb']

        # for CH
        nr_steps = int(_np.max(_np.abs(dkch))/sofb.maxdeltakickch) + 1
        toapk = dkch/nr_steps
        for _ in range(nr_steps):
            sofb.deltakickch = toapk
            sofb.wait_apply_delta_kick()

        # for CV
        nr_steps = int(_np.max(_np.abs(dkcv))/sofb.maxdeltakickcv) + 1
        toapk = dkcv/nr_steps
        for _ in range(nr_steps):
            sofb.deltakickcv = toapk
            sofb.wait_apply_delta_kick()

        # for RF
        nr_steps = int(_np.abs(dkrf)/sofb.maxdeltakickrf) + 1
        toapk = dkrf/nr_steps
        for _ in range(nr_steps):
            sofb.deltakickrf = toapk
            sofb.wait_apply_delta_kick()



