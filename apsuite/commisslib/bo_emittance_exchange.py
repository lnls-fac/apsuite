"""Scripts to study and simulate the emittance exchange in the Sirius Booster
"""
import numpy as _np
import time as _time
import pyaccel as _pa
import pymodels as _pm
from numpy.random import default_rng as _default_rng
from numpy.fft import rfft as _rfft,  rfftfreq as _rfftfreq
from scipy.signal import spectrogram as _spectrogram
import matplotlib.pyplot as _plt

from siriuspy.epics import PV
from siriuspy.devices import CurrInfoBO, \
    Trigger, Event, EVG, RFGen, FamBPMs

from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class BOTunebyBPMParams(_ParamsBaseClass):
    """Methods for measure tunes by bpm data"""

    def __init__(self):
        """"""
        super().__init__()
        self.event = 'DigBO'
        self.trigger_source = 'DigBO'
        self.trigger_source_mode = 'Injection'
        self.extra_delay = 0
        self.nr_pulses = 1
        self.nr_points_after = 10000
        self.nr_points_before = 0
        self.bpms_timeout = 30  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += stmp('event', self.event, '')
        stg += stmp('trigger_source', self.trigger_source, '')
        stg += stmp('trigger_source_mode', self.trigger_source_mode, '')
        stg += ftmp('extra_delay', self.extra_delay, '[us]')
        stg += dtmp('nr_pulses', self.nr_pulses, '')
        stg += dtmp('nr_points_after', self.nr_points_after, '')
        stg += dtmp('nr_points_before', self.nr_points_before, '')
        stg += ftmp('bpms_timeout', self.bpms_timeout, '[s]')

        return stg


class BOTunebyBPM(_ThreadBaseClass):
    """."""
    def __init__(self, params=None, isonline=True):
        """."""
        params = BOTunebyBPMParams() if params is None else params
        super().__init__(params=params, isonline=isonline)
        if self.isonline:
            self._create_devices()
            self._create_pvs()

    def _create_pvs(self):
        self.pvs['bo-qf-wfm'] = PV('BO-Fam:PS-QF:Wfm-RB')
        self.pvs['bo-qd-wfm'] = PV('BO-Fam:PS-QD:Wfm-RB')

    def _create_devices(self):
        """."""
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['bobpms'] = FamBPMs(FamBPMs.DEVICES.BO)
        self.devices['event'] = Event(self.params.event)
        self.devices['evg'] = EVG()
        self.devices['trigbpm'] = Trigger('BO-Fam:TI-BPM')
        self.devices['rfgen'] = RFGen()

    def configure_bpms(self):
        """."""
        prms = self.params
        bobpms = self.devices['bobpms']
        trigbpm = self.devices['trigbpm']

        bobpms.mturn_config_acquisition(
            nr_points_after=prms.nr_points_after,
            nr_points_before=prms.nr_points_before,
            acq_rate='TbT', repeat=False, external=True)
        bobpms.mturn_reset_flags()
        trigbpm.source = prms.trigger_source
        trigbpm.nr_pulses = prms.nr_pulses

    def get_orbit(self, injection=False, external_trigger=False):
        """Get orbit data from BPMs in TbT acquisition rate..
        If injection is True, then injection is turned on before the measure.
        If external_trigger is True, the event will listen a external trigger.
        """
        prms = self.params
        bobpms = self.devices['bobpms']
        trigbpm = self.devices['trigbpm']

        delay0 = trigbpm.delay
        trigbpm.delay = delay0 + prms.extra_delay
        self.devices['event'].mode = prms.trigger_source_mode

        # Inject and start acquisition
        bobpms.mturn_reset_flags()
        if external_trigger:
            self.devices['event'].cmd_external_trigger()
        if injection:
            self.devices['evg'].cmd_turn_on_injection()
        ret = bobpms.mturn_wait_update_flags(timeout=prms.bpms_timeout)
        if ret:
            trigbpm.delay = delay0
            self.data = dict()
            raise AssertionError(
                f'Problem waiting BPMs update. Error code: {ret:d}')
        orbx, orby = bobpms.get_mturn_orbit()
        bobpms.cmd_mturn_acq_abort()
        trigbpm.delay = delay0

        self.data['orbx'], self.data['orby'] = orbx, orby
        self.data['timestamp'] = _time.time()

    def get_data(
            self, delta='', injection=False, external_trigger=False,
            orbit=True):
        """."""
        # Store orbit
        if orbit:
            self.get_orbit(
                injection=injection, external_trigger=external_trigger)

        # Store auxiliar data
        bobpms = self.devices['bobpms']
        trigbpm = self.devices['trigbpm']
        bpm0 = bobpms[0]
        csbpm = bpm0.csdata
        data = dict()
        data['delta'] = delta
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['current_150mev'] = self.devices['currinfo'].current150mev
        data['current_1gev'] = self.devices['currinfo'].current1gev
        data['current_2gev'] = self.devices['currinfo'].current2gev
        data['current_3gev'] = self.devices['currinfo'].current3gev
        data['mt_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = trigbpm.delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
                                        bpm0.switching_mode]
        data['qf_wfm'] = self.pvs['bo-qf-wfm'].get()
        data['qd_wfm'] = self.pvs['bo-qd-wfm'].get()

        self.data.update(data)

    def load_orbit(self, data=None, orbx=None, orby=None):
        """Load orbit data into the object. You can pass the
        intire data dictionary or just the orbits. If data argument
        is provided, orbx and orby become optional"""

        if data is not None:
            self.data = data
        if orbx is not None:
            self.data['orbx'] = orbx
        if orby is not None:
            self.data['orby'] = orby

    def dft(self, bpm_indices=None):
        """Apply a dft at BPMs data.

        Args:
        - bpm_indices (int, list or np.array): BPM indices whose dft will
        be applied. Default is return the dft of all BPMs.

        Returns:
         - spectrumx, spectrumy (np.arrays): Two matrices of dimension #freqs x
         #bpm_indices containing the spectra of each BPM for the horizontal and
         vertical, respectively.

         - freqs (np.array):  frequency domain values.
        """
        if bpm_indices is not None:
            orbx = self.data['orbx'][:, bpm_indices]
            orby = self.data['orby'][:, bpm_indices]
        else:
            orbx = self.data['orbx']
            orby = self.data['orby']

        x_beta = orbx - orbx.mean(axis=0)
        y_beta = orby - orby.mean(axis=0)

        N = x_beta.shape[0]
        freqs = _rfftfreq(N)

        spectrumx = _np.abs(_rfft(x_beta, axis=0))
        spectrumy = _np.abs(_rfft(y_beta, axis=0))

        return spectrumx, spectrumy, freqs

    def naff_tunes(
            self, dn=None, window_param=1, bpm_indices=None, interval=None):
        """Computes the tune evolution from the BPMs matrix with a moving
            window of length dn.
           If dn is not passed, the tunes are computed using all points."""

        if bpm_indices is not None:
            x = self.data['orbx'][:, bpm_indices]
            y = self.data['orby'][:, bpm_indices]
        elif (bpm_indices is not None) and (interval is not None):
            x = self.data['orbx'][interval, bpm_indices]
            y = self.data['orby'][interval, bpm_indices]
        elif interval is not None:
            x = self.data['orbx'][interval, :]
            y = self.data['orby'][interval, :]
        else:
            x = self.data['orbx']
            y = self.data['orby']
        N = x.shape[0]

        if dn is None:
            return self.tune_by_naff(x, y)
        else:
            tune1_list = []
            tune2_list = []
            slices = _np.arange(0, N, dn)
            for idx in range(len(slices)-1):
                idx1, idx2 = slices[idx], slices[idx+1]
                sub_x = x[idx1:idx2, :]
                sub_y = y[idx1:idx2, :]
                tune1, tune2 = self.tune_by_naff(
                    sub_x, sub_y, window_param=1, decimal_only=True)
                tune1_list.append(tune1)
                tune2_list.append(tune2)

        return _np.array(tune1_list), _np.array(tune2_list)

    def spectrogram(
            self, bpm_indices=None, dn=None, interval=None, overlap=None,
            window=('tukey', 0.25)
            ):
        """Compute a spectrogram with consecutive Fourier transforms in segments
        of length dn. You also can set a window and control the overlap between
        the segments. If more than one BPMs is used, the outputted spectrogram
        is a normalized mean of the individual BPMs spectrograms. This method
        is a adaptation of scipy.signal.spectrogram function, see the below
        link for more information:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

        Args:
            - bpm_indices (list, optional) = Indices of the BPMs that will be
            used for compute the spectogram

            - dn (int, optional): Length of each segment. Defaults is 256 if
            window is str or tuple and len(window) if window is an array.

            - overlap (int, optional): Number of points to overlap between
            segments. Default is dn//8.

            - window (str, tuple or array): Desired window to use. If window
            is a string or tuple, it is passed to scipy.signal.get_window to
            generate the desidered window. See the below link to consult the
            avaliable windows:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
            You also can pass an array describing the desirable window. Default
            is a Tukey window with shape parameter equal to 0.25.

        Returns:
            - tune1_matrix, tune2_matrix: Spectrograms of the tunes x and y,
            respectively.

            - freqs: Array of sample frequencies.

            - revs : Array of segmented number of revolutions.
        """

        if bpm_indices is not None:
            x = self.data['orbx'][:, bpm_indices]
            y = self.data['orby'][:, bpm_indices]
        elif (bpm_indices is not None) and (interval is not None):
            x = self.data['orbx'][interval, bpm_indices]
            y = self.data['orby'][interval, bpm_indices]
        elif interval is not None:
            x = self.data['orbx'][interval, :]
            y = self.data['orby'][interval, :]
        else:
            x = self.data['orbx']
            y = self.data['orby']

        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)

        freqs, revs, Sx = _spectrogram(
            x.T, nperseg=dn, noverlap=overlap, window=window)
        _, _, Sy = _spectrogram(
            y.T, nperseg=dn, noverlap=overlap, window=window)

        tune1_matrix = Sx.mean(axis=0)
        tune2_matrix = Sy.mean(axis=0)
        tune_matrix = tune1_matrix + tune2_matrix

        # normalizing this matrix to get a better heatmap plot:
        tune_matrix -= tune_matrix.min()
        tune_matrix /= tune_matrix.max()

        # plots spectogram
        _plot_heatmap(freqs, revs, tune_matrix)

        return tune1_matrix, tune2_matrix, freqs, revs

    @staticmethod
    def tune_by_naff(x, y, window_param=1, decimal_only=True):
        """."""
        M = x.shape[1]
        beta_osc_x = x - _np.mean(x, axis=0)
        beta_osc_y = y - _np.mean(y, axis=0)

        Ax = beta_osc_x.ravel()
        Ay = beta_osc_y.ravel()

        freqx, _ = _pa.naff.naff_general(
            Ax, is_real=True, nr_ff=1, window=window_param)
        freqy, _ = _pa.naff.naff_general(
            Ay, is_real=True, nr_ff=1, window=window_param)
        tune1, tune2 = M*freqx, M*freqy

        if decimal_only is False:
            return _np.abs(tune1), _np.abs(tune2)
        else:
            tune1, tune2 = _np.abs(tune1 % 1), _np.abs(tune2 % 1)
            if tune1 > 0.5:
                tune1 = _np.abs(1-tune1)
            if tune2 > 0.5:
                tune2 = _np.abs(1-tune2)
            return tune1 % 1, tune2 % 1


def _plot_heatmap(freqs, revs, tune_matrix):
    _plt.figure()
    _plt.pcolormesh(revs, freqs, tune_matrix,
                    shading='gouraud', cmap='hot')
    _plt.colorbar().set_label(label="Relative Amplitude")
    _plt.ylabel('Frac. Frequency')
    _plt.xlabel('Turns')
    _plt.tight_layout()
    _plt.show()


class EmittanceExchangeSimul:

    REV_PERIOD = 1.657e-3  # [m s]

    def __init__(self, init_delta=None, c=0.01, radiation=True, s=1):

        self._model = _pm.bo.create_accelerator(energy=3e9)
        self._model.vchamber_on = True
        self._model.cavity_on = True

        if radiation:
            self._model.radiation_on = True

        self._radiation = radiation
        self._tune_crossing_vel = s
        self._coupling_coeff = c
        self._init_delta = init_delta
        self._tunes = None
        self._emittances = None
        self._exchange_quality = None

        self._qf_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value='QF')
        self._KL_default = _pa.lattice.get_attribute(
            lattice=self.model, attribute_name='KL',
            indices=self._qf_idxs[0])[0]

        self._qs_idxs = _pa.lattice.find_indices(
            lattice=self.model, attribute_name='fam_name', value='QS')
        self._KS_default = _pa.lattice.get_attribute(
            lattice=self.model, attribute_name='KsL',
            indices=self._qs_idxs[0])[0]

        if init_delta is not None:
            self._set_initial_delta(init_delta)
        else:
            self._init_delta = self._calc_delta()

        if c:
            self._set_coupling(c=c)
        self._bunch = None

    @property
    def qf_idxs(self):
        return self._qf_idxs

    @property
    def qs_idxs(self):
        return self._qs_idxs

    @property
    def model(self):
        return self._model

    @property
    def bunch(self):
        return self._bunch

    @property
    def coupling_coeff(self):
        return self._coupling_coeff

    @property
    def tune_crossing_vel(self):
        return self._tune_crossing_vel

    @property
    def tunes(self):
        return self._tunes

    @property
    def deltas(self):
        tunes_diff = self.tunes[0] - self.tunes[1]
        c = _np.min(_np.abs(tunes_diff))
        return _np.sign(tunes_diff) * _np.sqrt(tunes_diff**2 - c**2)

    @property
    def emittances(self):
        return self._emittances

    @property
    def exchange_quality(self):
        if self.emittances is not None:
            self._exchange_quality = self.calc_exchange_quality()
        else:
            raise AttributeError("Emittance exchange wasn't simulated yet.")
        return self._exchange_quality

    def experimental_params(self):
        """."""
        c = self.coupling_coeff
        s = self.tune_crossing_vel
        Tr = self.REV_PERIOD
        dtune_dt = s*c**2/Tr
        l_quad = 2*_pa.lattice.get_attribute(
            self.model, 'length', indices=self._qf_idxs[0])[0]
        dkl_dt = self._calc_dk(dtune_dt) * l_quad
        tt = (_np.abs(self._init_delta) + _np.abs(2.8*c))/dtune_dt
        dc_dksL = self._ksl_to_c(1)

        ftmp = '{0:24s} = {1:9.5f}  {2:s}\n'.format

        stg = ''
        stg += ftmp('d(tune)/dt', dtune_dt, '[1/ms]')
        stg += ftmp('dKL/dt', dkl_dt, '[1/(m . ms)]')
        stg += ftmp('Exchange time', tt, '[ms]')
        stg += ftmp('dC/dKsL', dc_dksL, '[m]')
        print(stg)

    def generate_bunch(self, n_part):
        """."""
        init_env = _pa.optics.calc_beamenvelope(
            accelerator=self.model, indices=[0], full=False)
        self._bunch = _pa.tracking.generate_bunch(
            n_part=n_part, envelope=init_env[0])

    def dynamic_emit_exchange(self, verbose=True):
        s = self.tune_crossing_vel
        c = self.coupling_coeff
        delta = self._calc_delta()
        n_turns = int(_np.abs(delta)/(s * c**2))  # Turns until the exchange
        rng = _default_rng()

        print("---------------------Tracking particles----------------------\n"
              "Initial delta = {:.3f} [C] \n N = {}".format(delta/c, 2*n_turns)
              )

        qf_idx = self.qf_idxs
        K_default = self.model[qf_idx[0]].K
        dK = self._calc_dk(-delta)
        K_list = _np.linspace(K_default, K_default + 2*dK, 2*n_turns)

        emittances = _np.zeros([2, K_list.size])
        tunes = emittances.copy()

        bunch0 = self.bunch
        npart = bunch0.shape[1]
        env0 = _np.cov(self.bunch).reshape([1, 6, 6])

        for i, K in enumerate(K_list):

            # Changing quadrupole forces
            _pa.lattice.set_attribute(
                lattice=self._model, attribute_name='K', values=K,
                indices=self.qf_idxs)

            if self._radiation:
                # Tracking with quantum excitation
                env0, cum_mat, bdiff, _ = _pa.optics.calc_beamenvelope(
                    accelerator=self._model, init_env=env0[-1],
                    indices='closed', full=True)
                bunch0 = _np.dot(cum_mat[-1], bunch0)

                bunch_excit = rng.multivariate_normal(
                    mean=_np.zeros(6), cov=bdiff[-1],
                    method='cholesky', size=npart).T
                bunch0 += bunch_excit
            else:
                m = _pa.tracking.find_m66(self.model)
                bunch0 = _np.dot(m, bunch0)
            self._bunch = bunch0

            # Computing the RMS emittance
            emittances[:, i] = self._calc_emittances()

            # Computing Tunes
            tunes[:, i] = self._calc_tunes()

            if verbose:
                if i % 100 == 0:
                    print(f"step {i}", end='\t')
                if i % 500 == 0:
                    print('\n')
                if i == K_list.size-1:
                    print('Done!')

        self._emittances = emittances
        self._tunes = tunes

    def plot_exchange(self, save=False, fname=None):
        """."""
        r, best_r_idx = self._calc_exchange_quality()
        fig, ax = _plt.subplots(1, 1, figsize=(6, 4.5))
        emitx, emity = self.emittances[0], self.emittances[1]
        deltas = self.deltas

        ax.plot(deltas, emitx*1e9, lw=3, label=r'$\epsilon_x \;[nm]$')
        ax.plot(deltas, emity*1e9, lw=3, label=r'$\epsilon_y \;[nm]$')
        ax.axvline(
            deltas[best_r_idx],
            label=f'max(R) = {r[best_r_idx]*100:.2f} [\\%]', ls='--',
            c='k', lw='3')
        ax.set_xlabel(r'$\Delta$')
        ax.set_ylabel(r'$\epsilon \; [nm]$')
        secax = ax.secondary_xaxis(
            'top', functions=(self._delta2time, self._time2delta))
        secax.set_xlabel('Time [ms]')
        ax.legend(loc='best')
        fig.tight_layout()
        if save:
            if fname is None:
                date_string = _time.strftime("%Y-%m-%d-%H:%M")
                fname = 'emit_exch_simul_{}.png'.format(date_string)
            fig.savefig(fname, format='png', dpi=300)
        _plt.show()
        return fig, ax

    def _calc_delta(self):
        """."""
        twi, *_ = _pa.optics.calc_twiss(self.model)
        tunex = twi.mux[-1]/(2*_np.pi)
        tuney = twi.muy[-1]/(2*_np.pi)
        tunex %= 1
        tuney %= 1
        if tunex > 0.5:
            tunex = _np.abs(1-tunex)
        if tuney > 0.5:
            tuney = _np.abs(1-tuney)
        return tunex - tuney

    def _calc_dk(self, dtune, plane='x'):
        if plane == 'x':
            sign = 1
        elif plane == 'y':
            sign = -1
        else:
            raise ValueError("Plane must be 'x' or 'y'")
        sum_beta_l, _ = self._calc_sum_beta_l()
        deltaK = sign*4*_np.pi*dtune/(sum_beta_l)
        return deltaK

    def _calc_tunes(self):
        ed_teng, _ = _pa.optics.calc_edwards_teng(self.model)
        tune1 = ed_teng.mu1[-1]/(2*_np.pi)
        tune2 = ed_teng.mu2[-1]/(2*_np.pi)
        tune1 %= 1
        tune2 %= 1
        if tune1 > 0.5:
            tune1 = _np.abs(1-tune1)
        if tune2 > 0.5:
            tune2 = _np.abs(1-tune2)
        return _np.array([tune1, tune2])

    def _calc_emittances(self):
        """."""
        twi, *_ = _pa.optics.calc_twiss(self.model)
        bunch0 = self.bunch
        etax, etapx = twi.etax[0], twi.etapx[0]
        etay, etapy = twi.etay[0], twi.etapy[0]
        disp = _np.array([[etax], [etapx], [etay], [etapy], [0], [0]])
        bunch_nodisp = \
            bunch0 - bunch0[4, :]*disp - _np.mean(bunch0, axis=1)[:, None]
        emitx = _np.sqrt(_np.linalg.det(_np.cov(bunch_nodisp[:2, :])))
        emity = _np.sqrt(_np.linalg.det(_np.cov(bunch_nodisp[2:4, :])))
        emittances_array = _np.array([emitx, emity])
        return emittances_array

    def _calc_exchange_quality(self):
        """."""
        emit1_0 = self.emittances[0, 0]
        emit2_0 = self.emittances[1, 0]
        emit1 = self.emittances[0]
        r = 1 - (emit1 - emit2_0)/(emit1_0 - emit2_0)
        best_r_idx = _np.argmax(r)
        return r, best_r_idx

    def _set_coupling(self, c):
        """."""
        ksl = self._c_to_ksl(c)
        _pa.lattice.set_attribute(
            lattice=self._model, attribute_name='KsL', values=ksl,
            indices=self.qs_idxs)

    def _set_initial_delta(self, init_delta):
        "Sets delta to delta_f"
        delta = self._calc_delta()
        qf_idx = self.qf_idxs
        sum_beta_l, _ = self._calc_sum_beta_l()
        dv_x = init_delta - delta
        dk_x = 4*_np.pi*dv_x/(sum_beta_l)
        k_x = self.model[qf_idx[0]].K + dk_x
        _pa.lattice.set_attribute(
            lattice=self._model, attribute_name='K', indices=qf_idx,
            values=k_x)

    def _c_to_ksl(self, C):
        """."""
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0]]
        beta2 = ed_tang.beta2[qs_idx[0]]
        KsL = -2 * _np.pi * C / _np.sqrt(beta1 * beta2)

        return KsL[0]

    def _ksl_to_c(self, KsL):
        """."""
        fam_data = _pm.bo.get_family_data(self.model)
        qs_idx = fam_data['QS']['index']
        ed_tang, *_ = _pa.optics.calc_edwards_teng(accelerator=self.model)
        beta1 = ed_tang.beta1[qs_idx[0][0]]
        beta2 = ed_tang.beta2[qs_idx[0][0]]
        C = _np.abs(KsL * _np.sqrt(beta1 * beta2)/(2 * _np.pi))

        return C

    def _calc_sum_beta_l(self):
        """."""
        ed_teng, _ = _pa.optics.calc_edwards_teng(self.model)
        qf_idx = self.qf_idxs
        betax = ed_teng.beta1
        betay = ed_teng.beta2
        length = self.model[qf_idx[0]].length
        betasx = _np.zeros(len(qf_idx))
        betasy = betasx.copy()

        for i in range(0, len(qf_idx), 2):
            idx1, idx2 = qf_idx[i], qf_idx[i+1]
            betay_values = betay[[idx1, idx2, idx2+1]]
            betax_values = betax[[idx1, idx2, idx2+1]]
            betasx[i] = _np.mean(betax_values)
            betasy[i] = _np.mean(betay_values)
        sum_beta_l = 2*length*(_np.sum(betasx) + _np.sum(betasy))

        return sum_beta_l, length

    def _delta2time(self, delta):
        s = self._tune_crossing_vel
        c = self.coupling_coeff
        tr = self.REV_PERIOD
        time = tr*delta/(s*c**2)
        time = time - _np.min(time)
        return time

    def _time2delta(self, time):
        s = self._tune_crossing_vel
        c = self.coupling_coeff
        tr = self.REV_PERIOD
        return time*s*c**2/tr
