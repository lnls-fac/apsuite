"""."""
import time as _time
import numpy as _np
from siriuspy.devices import RFGen, SOFB, PowerSupply, Tune
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from apsuite.commisslib.meas_bpms_signals import AcqBPMsSignals
from apsuite.commisslib.measure_orbit_stability import OrbitAnalysis
from pymodels import si
import pyaccel
from apsuite.orbcorr.calc_orbcorr_mat import OrbRespmat
from mathphys.functions import get_namedtuple as _get_namedtuple


class CorrectVerticalDispersionParams(_ParamsBaseClass):
    """."""

    CORR_METHOD = _get_namedtuple('METHODS', ['QS', 'CV'])

    def __init__(self):
        """."""
        super().__init__()
        self.delta_rf_freq = 100  # [Hz]
        self.nr_points_sofb = 20  # [Hz]
        self.nr_iters_corr = 1
        self.nr_svals = 100  # all qs
        self.amp_factor2apply = 1
        self.qs2use_idx = _np.arange(0, 100)
        self.cv2use_idx = _np.arange(0, 160)
        self._corr_method = CorrectVerticalDispersionParams.CORR_METHOD.CV

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        stg = ftmp('delta_rf_freq', self.delta_rf_freq, '[Hz]')
        stg += dtmp('nr_points_sofb', self.nr_points_sofb)
        stg += dtmp('nr_iters_corr', self.nr_iters_corr)
        stg += dtmp('nr_svals', self.nr_svals)
        stg += ftmp('amp_factor2apply', self.amp_factor2apply, '')
        if self.corr_method == self.CORR_METHOD.QS:
            stg += '{0:25s} = ['.format('qs2use_idx')
            stg += ','.join(f'{idx:3d}' for idx in self.qs2use_idx) + ']'
        else:
            pass
        return stg


    @property
    def corr_method(self):
        """."""
        """."""
        return self._corr_method

    @corr_method.setter
    def corr_method(self, value):
        if value is None:
            return
        _prms_mthd = CorrectVerticalDispersionParams.CORR_METHOD
        if isinstance(value, str):
            self._corr_method = int(value in _prms_mthd._fields[1])
        elif int(value) in _prms_mthd:
            self._corr_method = int(value)


class CorrectVerticalDispersion(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, params, isonline=True):
        """."""
        super().__init__(
            target=self._meas_func, isonline=isonline, params=params)
        self.model = si.create_accelerator()
        self.fam_data = si.get_family_data(self.model)
        self.model_alpha = pyaccel.optics.get_mcf(self.model)
        self.dispmat = None
        self.orm = OrbRespmat(model=self.model, acc='SI', dim='6d')
        self.qs_names = None  # to avoid errors
        self.orbanly = OrbitAnalysis(isonline=False)
        if self.isonline:
            self.devices['rfgen'] = RFGen(
                props2init=['GeneralFreq-SP', 'GeneralFreq-RB'])
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.orbacq = AcqBPMsSignals(isonline=True)
            if self.params.corr_method == \
                    CorrectVerticalDispersionParams.CORR_METHOD.QS:
                self._create_skews()

    def _configure_acq_params(self):
        """."""
        params = self.orbacq.params
        params.signals2acq = 'XY'

        params.acq_rate = 'FOFB'
        params.timeout = 200

        params.nrpoints_before = 0
        params.nrpoints_after = 100_000
        params.acq_repeat = 0
        params.trigbpm_delay = 0
        params.trigbpm_nrpulses = 1
        params.event_mode = 'External'
        params.do_pulse_evg = True
        params.timing_event = 'Study'

    def _get_trig_data(self):
        init_state_orbacq = self.orbacq.get_timing_state()
        self.orbacq.prepare_timing()
        self.orbacq.acquire_data()
        self.orbacq.recover_timing_state(init_state_orbacq)

    def _process_dispersion(self):
        self.orbanly.data = self.orbacq.data
        self.orbanly.get_appropriate_orm_data('ref_respmat')
        self.orbanly._get_sampling_freq()
        self.orbanly._get_switching_freq()
        self.orbanly.process_data_energy(
            central_freq=32*64, window=15, use_eta_meas=False)
        return self.orbanly.analysis['measured_dispersion']*1e-6

    def _create_skews(self):
        """."""
        qs_names = _np.array(self.fam_data['QS']['devnames'], dtype=object)
        self.qs_names = qs_names
        for name in self.qs_names:
            self.devices[name] = PowerSupply(name)

    def get_strengths(self):
        """."""
        _corr_mthd = CorrectVerticalDispersionParams.CORR_METHOD
        _prms = self.params
        if _prms.corr_method == _corr_mthd.QS:
            names = self.qs_names[self.params.qs2use_idx]
            stren = [self.devices[name].strength for name in names]
            return _np.array(stren)
        elif _prms.corr_method == _corr_mthd.CV:
            return self.devices['sofb'].kickcv
        return None

    def apply_strengths(self, strengths, factor=1):
        """."""
        if self.params.corr_method == \
                CorrectVerticalDispersionParams.CORR_METHOD.CV:
            sofb = self.devices['sofb']
            kick0 = sofb.kickcv
            sofb.deltakickcv = factor*(strengths - kick0)
            sofb.cmd_applycorr_cv()
        else:
            names = self.qs_names[self.params.qs2use_idx]
            for idx, name in enumerate(names):
                self.devices[name].strength = factor * strengths[idx]

    def apply_delta_strengths(self, delta_strengths, factor=1):
        """."""
        if self.params.corr_method == \
                CorrectVerticalDispersionParams.CORR_METHOD.CV:
            sofb = self.devices['sofb']
            sofb.deltakickcv = factor*delta_strengths
            sofb.cmd_applycorr_cv()
        else:
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

    def measure_dispersion_trigacq(self):
        """."""
        self._get_trig_data()
        disp = self._process_dispersion()
        rf_freq0 = self.devices['rfgen'].frequency
        return disp, rf_freq0

    def correct_orbit_with_ch_rf(self, nr_iters=5):
        """."""
        sofb = self.devices['sofb']
        for _ in range(nr_iters):
            sofb.wait_buffer()
            sofb.cmd_calccorr()
            sofb.cmd_applycorr_ch()
            sofb.cmd_applycorr_rf()
            sofb.wait_apply_delta_kick()
            sofb.cmd_reset()

    def _meas_func(self):
        prms = self.params
        qs_method = CorrectVerticalDispersionParams.CORR_METHOD.QS
        cv_method = CorrectVerticalDispersionParams.CORR_METHOD.CV

        self._configure_acq_params()

        data = {}
        data['timestamp'] = _time.time()
        data['dispersion_history'] = []

        # CORR WITH QS
        if prms.corr_method == qs_method:
            data['dksl_history'] = []
            data['ksl_history'] = []
            data['initial_ksl'] = self.get_strengths()
        elif prms.corr_method == cv_method:
            # CORR WITH CV
            data['dkickcv_history'] = []
            data['kickcv_history'] = []
            data['initial_kickcv'] = self.get_strengths()
            data['initial_orbit'] = self.get_orb()
            data['orbit_history'] = []

        data['rf_frequency'] = []
        data['tunex'] = []
        data['tuney'] = []

        if self.dispmat is None:
            raise Exception('self.dispmat is None')

        idmat = self.invert_dispmat(
            self.dispmat, prms.nr_svals)

        for idx in range(prms.nr_iters_corr):
            if self._stopevt.is_set():
                print('Stop!')
                break
            disp, rffreq = self.measure_dispersion()
            # disp, rffreq = self.measure_dispersion_trigacq()
            dispy = disp[160:]
            data['dispersion_history'].append(disp)
            data['rf_frequency'].append(rffreq)
            data['tunex'].append(self.devices['tune'].tunex)
            data['tuney'].append(self.devices['tune'].tuney)
            dstrens = - idmat @ dispy
            stg = f'Iter. {idx+1:02d}/{prms.nr_iters_corr:02d} | '
            stg += f'dispy rms: {_np.std(dispy)*1e6:.2f} [um], '
            if prms.corr_method == qs_method:
                stg += f'dKsL rms: {_np.std(dstrens)*1e3:.4f} [1/km] \n'
                data['dksl_history'].append(dstrens)
                data['ksl_history'].append(self.get_strengths())
            elif prms.corr_method == cv_method:
                stg += f'dkickV rms: {_np.std(dstrens)*1e6:.4f} [urad] \n'
                data['dkickcv_history'].append(dstrens)
                data['kickcv_history'].append(self.get_strengths())
                data['orbit_history'].append(self.get_orb())
                dstrens *= 1e6  # [rad] -> [urad]
            print(stg)
            print('-'*50)
            self.apply_delta_strengths(
                dstrens, factor=prms.amp_factor2apply)
            if prms.corr_method == qs_method:
                self.devices['sofb'].correct_orbit_manually(
                    nr_iters=10, residue=5)
            else:
                self.correct_orbit_with_ch_rf()

        dispf, rffreq = self.measure_dispersion()
        # dispf, rffreq = self.measure_dispersion_trigacq()
        data['final_dispersion'] = dispf
        data['final_rf_frequency'] = rffreq
        if prms.corr_method == qs_method:
            data['final_ksl'] = self.get_strengths()
        elif prms.corr_method == cv_method:
            data['final_kickcv'] = self.get_strengths()
            data['final_orbit'] = self.get_orb()

        print('='*50)
        print('Correction result:')
        dispy0, dispyf = data['dispersion_history'][0][160:], dispf[160:]
        stg = f'dispy rms {_np.std(dispy0)*1e6:.2f} [um] '
        stg += f'--> {_np.std(dispyf)*1e6:.2f} [um] \n'
        if prms.corr_method == qs_method:
            ksl0, kslf = data['initial_ksl'], data['final_ksl']
            stg += f"KsL rms: {_np.std(ksl0)*1e3:.4f} [1/km] "
            stg += f'--> {_np.std(kslf)*1e3:.4f} [1/km] \n'
        elif prms.corr_method == cv_method:
            kick0, kickf = data['initial_kickcv'], data['final_kickcv']
            stg += f"kickCV rms: {_np.std(kick0):.4f} [urad] "
            stg += f'--> {_np.std(kickf):.4f} [urad] \n'

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

    def calc_ks_dispmat(self, qsidx=None, dksl=1e-6):
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
        self.model_ks_dispmat = dispmat
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

    def measure_ks_dispmat(self, dksl=1e-4):
        """."""
        meas_ks_dispmat = []
        names = self.qs_names[self.params.qs2use_idx]
        print('Measuring respmat: dispersion/KsL skew')
        sofb = self.devices['sofb']
        for name in names:
            print(f'{name:s}')
            qsmag = self.devices[name]
            stren0 = qsmag.strength
            qsmag.set_strength(stren0 + dksl/2)
            sofb.correct_orbit_manually(nr_iters=10, residue=5)
            dispp, _ = self.measure_dispersion()
            qsmag.set_strength(stren0 - dksl/2)
            sofb.correct_orbit_manually(nr_iters=10, residue=5)
            dispm, _ = self.measure_dispersion()
            meas_ks_dispmat.append((dispp-dispm)/dksl)
            qsmag.set_strength(stren0)
            sofb.correct_orbit_manually(nr_iters=10, residue=5)
        meas_ks_dispmat = _np.array(meas_ks_dispmat).T
        print('Finished!')
        self.meas_ks_dispmat = meas_ks_dispmat
        return meas_ks_dispmat

    def measure_kick_dispmat(self, dkick=20):
        """."""
        meas_dispmat = []
        print('Measuring respmat: dispersion/KickCV')
        dkicks = _np.zeros(160)
        for idx in self.params.cv2use_idx:
            dkicks[idx] = +dkick/2
            self.apply_delta_strengths(dkicks)
            self.correct_orbit_with_ch_rf(nr_iters=2)
            dispp, _ = self.measure_dispersion()
            dkicks[idx] = -dkick
            self.apply_delta_strengths(dkicks)
            self.correct_orbit_with_ch_rf(nr_iters=2)
            dispm, _ = self.measure_dispersion()
            meas_dispmat.append((dispp-dispm)/dkick)
            dkicks[idx] = 0
            self.apply_delta_strengths(dkicks)
            self.correct_orbit_with_ch_rf(nr_iters=2)
        meas_dispmat = _np.array(meas_dispmat).T
        print('Finished!')
        self.meas_dispmat = meas_dispmat
        return meas_dispmat
