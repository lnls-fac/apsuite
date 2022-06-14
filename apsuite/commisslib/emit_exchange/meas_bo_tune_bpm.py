"Measures booster tunes using bpms"
import numpy as _np
import matplotlib.pyplot as _plt
import pyaccel as _pa
import time as _time

from numpy.fft import rfft as _rfft,  rfftfreq as _rfftfreq
from scipy.signal import spectrogram as _spectrogram

from siriuspy.epics import PV
from siriuspy.devices import CurrInfoBO, \
    Trigger, Event, EVG, RFGen, FamBPMs

from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
    MeasBaseClass as _BaseClass


class BOTunebyBPMParams(_ParamsBaseClass):

    def __init__(self):
        """"""
        super().__init__()
        self.event = 'DigBO'
        self.event_mode = 'External'
        self.event_delay = 0
        self.trigger_source = 'DigBO'
        # self.trigger_source_mode = 'Injection'

        self.nr_pulses = 1
        self.nr_points_after = 4000
        self.nr_points_before = 0
        self.bpms_timeout = 30  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += stmp('event', self.event, '')
        stg += stmp('event_mode', self.event_mode, '')
        stg += ftmp('event_delay', self.event_delay, '[us]')
        stg += stmp('trigger_source', self.trigger_source, '')
        # stg += stmp('trigger_source_mode', self.trigger_source_mode, '')
        stg += dtmp('nr_pulses', self.nr_pulses, '')
        stg += dtmp('nr_points_after', self.nr_points_after, '')
        stg += dtmp('nr_points_before', self.nr_points_before, '')
        stg += ftmp('bpms_timeout', self.bpms_timeout, '[s]')

        return stg


class BOTunebyBPM(_BaseClass):
    """Methods for measuring tunes by bpm data.

    Before starting the experiment:
    You need to configure the extraction kicker to act at the point of the ramp
    where you want to know the tunes. The tunes will be measured based on the
    Fourier transform of the betatron oscillations induced by the kick.
    NOTE:Maybe its necessary to include this step in this class.

    The BPMs need to be configured to acquire data immediately after the kicker
    pulse, a few thousands of revolutions acquired data are necessary to
    compute the tunes with precision.
    """
    BPM_TRIGGER = 'BO-Fam:TI-BPM'

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
        self.devices['trigbpm'] = Trigger(BOTunebyBPM.BPM_TRIGGER)
        self.devices['rfgen'] = RFGen()

    def get_initial_state(self):
        """."""
        trigbpm = self.devices['trigbpm']
        event = self.devices['event']
        state = dict()
        state['trigbpm_source'] = trigbpm.source
        state['trigbpm_nrpulses'] = trigbpm.nr_pulses
        state['trigbpm_delay'] = trigbpm.delay
        state['event_delay'] = event.delay
        state['event_mode'] = event.mode
        return state

    def recover_initial_state(self, state):
        """."""
        trigbpm = self.devices['trigbpm']
        event = self.devices['event']

        trigbpm.source = state['trigbpm_source']
        trigbpm.nr_pulses = state['trigbpm_nrpulses']
        trigbpm.delay = state['trigbpm_delay']
        event.delay = state['event_delay']
        event.mode = state['event_mode']

    def prepare_timing(self):
        """."""
        trigbpm = self.devices['trigbpm']
        evt_study = self.devices['evt_study']

        trigbpm.delay = self.params.trigbpm_delay
        trigbpm.nr_pulses = self.params.trigbpm_nrpulses
        trigbpm.source = self.trigger_source

        evt_study.delay = self.params.event_delay
        evt_study.mode = self.params.event_mode

        # Update event configurations in EVG
        self.devices['evg'].cmd_update_events()

    def configure_bpms(self):
        """."""
        prms = self.params
        bobpms = self.devices['bobpms']
        bobpms.cmd_mturn_acq_abort()
        bobpms.mturn_config_acquisition(
            nr_points_after=prms.nr_points_after,
            nr_points_before=prms.nr_points_before,
            acq_rate='TbT', repeat=False)

    def get_orbit(self, injection=False, ext_trigger=False):
        """Get orbit data from BPMs in TbT acquisition rate..
        If injection is True, then injection is turned on before the measure.
        If external_trigger is True, the event will listen a external trigger.
        """
        prms = self.params
        bobpms = self.devices['bobpms']
        event = self.devices['event']

        # Inject and start acquisition
        if injection:
            self.devices['evg'].cmd_turn_on_injection()
        if ext_trigger:
            event.cmd_external_trigger()

        ret = bobpms.mturn_wait_update_flags(timeout=prms.bpms_timeout)
        if ret:
            self.data = dict()
            raise AssertionError(
                f'Problem waiting BPMs update. Error code: {ret:d}')
        orbx, orby = bobpms.get_mturn_orbit()

        self.data['orbx'], self.data['orby'] = orbx, orby
        self.data['timestamp'] = _time.time()

    def load_orbit(self, data=None, orbx=None, orby=None):
        """Load orbit data into the object. You can pass the
        entire data dictionary or just the orbits. If data argument
        is provided, orbx and orby become optional"""

        if data is not None:
            self.data = data
        if orbx is not None:
            self.data['orbx'] = orbx
        if orby is not None:
            self.data['orby'] = orby

    def get_data(
            self, delta='', injection=False, orbit=True, ext_trigger=False):
        """."""
        # Store orbit
        if orbit:
            self.get_orbit(injection, ext_trigger)

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

    def dft_tunes(self, init_cut=50):
        """."""
        sx, sy, freqs = self.dft()
        win = _np.arange(init_cut, freqs.size, dtype=int)
        freqs, sx, sy = freqs[win], sx[win], sy[win]
        peaksx, peaksy = _np.argmax(sx, axis=0), _np.argmax(sy, axis=0)

        meas_tunesx, meas_tunesy = freqs[peaksx], freqs[peaksy]

        tunes = _np.array([meas_tunesx.mean(), meas_tunesy.mean()])
        dtunes = _np.array([meas_tunesx.std(), meas_tunesy.std()])
        return tunes, dtunes

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
