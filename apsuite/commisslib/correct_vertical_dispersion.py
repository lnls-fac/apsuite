"""."""
import time as _time
import numpy as _np
from siriuspy.devices import RFGen, SOFB, PowerSupply, Tune
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from pymodels import si
import pyaccel
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat


class CorrectVerticalDispersionParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.delta_rf_freq = 100  # [Hz]
        self.nr_points_sofb = 20  # [Hz]
        self.nr_iters_corr = 1
        self.nr_svals = 100  # all qs
        self.factor2apply = 1
        self.qs2use_idx = _np.arange(0, 100)

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        # stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('delta_rf_freq', self.delta_rf_freq, '[Hz]')
        stg += dtmp('nr_points_sofb', self.nr_points_sofb)
        stg += dtmp('nr_iters_corr', self.nr_iters_corr)
        stg += dtmp('nr_svals', self.nr_svals)
        stg += ftmp('factor2apply', self.factor2apply, '')
        stg += '{0:25s} = ['.format('qs2use_idx')
        stg += ','.join(f'{idx:3d}' for idx in self.qs2use_idx) + ']'
        return stg


class CorrectVerticalDispersion(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)

        self.model = si.create_accelerator()
        self.fam_data = si.get_family_data(self.model)
        self.model_alpha = pyaccel.optics.get_mcf(self.model)
        self.model_dispmat = None
        self.meas_dispmat = None
        self.orm = OrbRespmat(model=self.model, acc='SI', dim='6d')
        if self.isonline:
            self.devices['rfgen'] = RFGen(
                props2init=['GeneralFreq-SP', 'GeneralFreq-RB'])
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.create_skews()

    def create_skews(self):
        """."""
        qs_names = _np.array(self.fam_data['QS']['devnames'], dtype=object)
        self.qs_names = qs_names
        for name in self.qs_names:
            self.devices[name] = PowerSupply(name)

    def get_skew_strengths(self):
        """."""
        names = self.qs_names[self.params.qs2use_idx]
        stren = [self.devices[name].strength for name in names]
        return _np.array(stren)

    def apply_skew_strengths(self, strengths, factor=1):
        """."""
        names = self.qs_names[self.params.qs2use_idx]
        for idx, name in enumerate(names):
            self.devices[name].strength = factor * strengths[idx]

    def apply_skew_delta_strengths(self, delta_strengths, factor=1):
        """."""
        names = self.qs_names[self.params.qs2use_idx]
        for idx, name in enumerate(names):
            self.devices[name].strength += factor * delta_strengths[idx]

    def get_orb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.wait_buffer()
        return _np.r_[sofb.orbx, sofb.orby]

    def measure_dispersion(self):
        """."""
        rfgen, sofb = self.devices['rfgen'], self.devices['sofb']
        prms = self.params
        rf_freq0 = rfgen.frequency
        sofb.nr_points = prms.nr_points_sofb
        dfreq = prms.delta_rf_freq
        rfgen.set_frequency(rf_freq0 + dfreq/2)
        orbplus = self.get_orb()
        rfgen.set_frequency(rf_freq0 - dfreq/2)
        orbminus = self.get_orb()
        dorb_dfreq = (orbplus - orbminus)/dfreq
        disp = - self.model_alpha * rf_freq0 * dorb_dfreq * 1e-6
        rfgen.set_frequency(rf_freq0)
        return disp, rf_freq0

    def _meas_func(self):
        data = {}
        data['timestamp'] = _time.time()
        data['dispersion_history'] = []
        data['dksl_history'] = []
        data['ksl_history'] = []
        data['initial_ksl'] = self.get_skew_strengths()
        data['rf_frequency'] = []
        data['tunex'] = []
        data['tuney'] = []

        if self.model_dispmat is None:
            all_qsidx_model = _np.array(self.fam_data['QS']['index']).ravel()
            subset_qsidx = all_qsidx_model[self.params.qs2use_idx]
            self.model_dispmat = self.calc_dispmat(
                self.model, qsidx=subset_qsidx)[160:, :]
        idmat = self.invert_dispmat(self.model_dispmat, self.params.nr_svals)
        for idx in range(self.params.nr_iters_corr):
            if self._stopevt.is_set():
                print('Stop!')
                break
            disp, rffreq = self.measure_dispersion()
            dispy = disp[160:]
            data['dispersion_history'].append(disp)
            data['rf_frequency'].append(rffreq)
            data['tunex'].append(self.devices['tune'].tunex)
            data['tuney'].append(self.devices['tune'].tuney)
            dstrens = self.calc_correction(dispy, idmat)
            stg = f'Iter. {idx+1:02d}/{self.params.nr_iters_corr:02d} | '
            stg += f'dispy rms: {_np.std(dispy)*1e6:.2f} [um], '
            stg += f'dKsL rms: {_np.std(dstrens)*1e3:.4f} [1/km] \n'
            print(stg)
            print('-'*50)
            data['dksl_history'].append(dstrens)
            data['ksl_history'].append(self.get_skew_strengths())
            self.apply_skew_delta_strengths(
                dstrens, factor=self.params.factor2apply)
            self.devices['sofb'].correct_orbit_manually(nr_iters=10, residue=5)

        dispf, rffreq = self.measure_dispersion()
        data['final_dispersion'] = dispf
        data['final_rf_frequency'] = rffreq
        data['final_ksl'] = self.get_skew_strengths()

        print('='*50)
        print('Correction result:')
        dispy0, dispyf = data['dispersion_history'][0][160:], dispf[160:]
        ksl0, kslf = data['initial_ksl'], data['final_ksl']
        stg = f'dispy rms {_np.std(dispy0)*1e6:.2f} [um] '
        stg += f'--> {_np.std(dispyf)*1e6:.2f} [um] \n'
        stg += f"KsL rms: {_np.std(ksl0)*1e3:.4f} [1/km] "
        stg += f'--> {_np.std(kslf)*1e3:.4f} [1/km] \n'
        print(stg)
        print('='*50)
        print('Finished!')
        self.data = data

    def calc_model_dispersion(self, model=None):
        """."""
        if model is None:
            model = self.model
        self.orm.model = model
        respmat = self.orm.get_respm()
        rf_freq = self.orm.model[self.orm.rf_idx[0]].frequency
        alpha = pyaccel.optics.get_mcf(model)
        return - alpha * rf_freq * respmat[:, -1]

    def calc_dispmat(self, qsidx=None, dksl=1e-6):
        """."""
        print('--- calculating dispersion/KsL matrix')
        if qsidx is None:
            qsidx = _np.array(self.fam_data['QS']['index']).ravel()
        dispmat = []
        for idx, qs in enumerate(qsidx):
            print(f'{idx+1:02d}/{qsidx.size:02d} ')
            modt = self.model[:]
            modt[qs].KsL += dksl/2
            dispp = self.calc_model_dispersion(modt)
            modt[qs].KsL -= dksl
            dispn = self.calc_model_dispersion(modt)
            dispmat.append((dispp-dispn)/dksl)
            modt[qs].KsL += dksl
        dispmat = _np.array(dispmat).T
        self.model_dispmat = dispmat
        return dispmat

    @staticmethod
    def invert_dispmat(dispmat, nr_svals=None):
        """."""
        umat, smat, vhmat = _np.linalg.svd(dispmat, full_matrices=False)
        ismat = 1/smat
        if nr_svals is not None:
            ismat[nr_svals:] = 0
        inv_dispmat = vhmat.T @ _np.diag(ismat) @ umat.T
        return inv_dispmat

    def calc_correction(self, dispy, inv_dispmat=None, nr_svals=None):
        """."""
        if inv_dispmat is None:
            dispmat = self.calc_dispmat(self.model)
            inv_dispmat = self.invert_dispmat(dispmat, nr_svals)
        dstren = - inv_dispmat @ dispy
        return dstren

    def measure_dispmat(self, dksl=1e-4):
        """."""
        meas_dispmat = []
        names = self.qs_names[self.params.qs2use_idx]
        print('Measuring respmat: dispersion/KsL skew')
        for name in names:
            print(f'{name:s}')
            qsmag = self.devices[name]
            stren0 = qsmag.strength
            qsmag.set_strength(stren0 + dksl/2)
            dispp, _ = self.measure_dispersion()
            qsmag.set_strength(stren0 - dksl/2)
            dispm, _ = self.measure_dispersion()
            meas_dispmat.append((dispp-dispm)/dksl)
            qsmag.set_strength(stren0)
        meas_dispmat = _np.array(meas_dispmat).T
        print('Finished!')
        self.meas_dispmat = meas_dispmat
        return meas_dispmat
