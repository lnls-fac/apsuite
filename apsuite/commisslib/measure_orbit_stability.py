"""."""
import numpy as _np
import matplotlib.pyplot as _plt
import datetime as _datetime
import time as _time

from mathphys.functions import load_pickle
import siriuspy.clientconfigdb as _sconf
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from siriuspy.devices import Tune, CurrInfoSI, \
    Trigger, Event, EVG, RFGen, FamBPMs


class OrbitAcquisitionParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        self.trigbpm_delay = 0.0
        self.trigbpm_nrpulses = 1
        self.event_delay = 0.0
        self.event_mode = 'External'
        self.orbit_timeout = 40
        self.orbit_nrpoints_after = 20000
        self.orbit_acq_rate = 'Monit1'
        self.orbit_acq_repeat = False

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:15s} = {1:9}  {2:s}\n'.format
        stg = ''
        stg += ftmp('trigbpm_delay', self.trigbpm_delay, '[us]')
        stg += dtmp('trigbpm_nrpulses', self.trigbpm_nrpulses, '')
        stg += ftmp('event_delay', self.event_delay, '[us]')
        stg += stmp('event_mode', self.event_mode, '')
        stg += ftmp('orbit_timeout', self.orbit_timeout, '[s]')
        stg += dtmp('orbit_nrpoints_after', self.orbit_nrpoints_after, '')
        stg += stmp('orbit_acq_rate', self.orbit_acq_rate, '')
        stg += dtmp('orbit_acq_repeat', self.orbit_acq_repeat, '')
        return stg


class OrbitAcquisition(_BaseClass):
    """."""

    BPM_TRIGGER = 'SI-Fam:TI-BPM'

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=OrbitAcquisitionParams(), isonline=isonline)

        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """."""
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['fambpms'] = FamBPMs(FamBPMs.DEVICES.SI)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['trigbpm'] = Trigger(OrbitAcquisition.BPM_TRIGGER)
        self.devices['evt_study'] = Event('Study')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()

    def get_initial_state(self):
        """."""
        trigbpm = self.devices['trigbpm']
        evt_study = self.devices['evt_study']
        state = dict()
        state['trigbpm_source'] = trigbpm.source
        state['trigbpm_nrpulses'] = trigbpm.nr_pulses
        state['trigbpm_delay'] = trigbpm.delay
        state['evt_study_delay'] = evt_study.delay
        state['evt_study_mode'] = evt_study.mode
        return state

    def recover_initial_state(self, state):
        """."""
        trigbpm = self.devices['trigbpm']
        evt_study = self.devices['evt_study']

        trigbpm.source = state['trigbpm_source']
        trigbpm.nr_pulses = state['trigbpm_nrpulses']
        trigbpm.delay = state['trigbpm_delay']
        evt_study.delay = state['evt_study_delay']
        evt_study.mode = state['evt_study_mode']

    def prepare_timing(self):
        """."""
        trigbpm = self.devices['trigbpm']
        evt_study = self.devices['evt_study']

        trigbpm.delay = self.params.trigbpm_delay
        trigbpm.nr_pulses = self.params.trigbpm_nrpulses
        trigbpm.source = 'Study'

        evt_study.delay = self.params.event_delay
        evt_study.mode = self.params.event_mode

        # Update event configurations in EVG
        self.devices['evg'].cmd_update_events()

    def prepare_bpms_acquisition(self):
        """."""
        fambpms = self.devices['fambpms']
        prms = self.params
        fambpms.cmd_mturn_acq_abort()
        fambpms.mturn_config_acquisition(
            nr_points_after=prms.orbit_nrpoints_after,
            acq_rate=prms.orbit_acq_rate,
            repeat=prms.orbit_acq_repeat)

    def acquire_data(self):
        """."""
        fambpms = self.devices['fambpms']
        evt_study = self.devices['evt_study']
        evt_study.cmd_external_trigger()
        fambpms.mturn_wait_update_flags(timeout=self.params.orbit_timeout)
        orbx, orby = fambpms.get_mturn_orbit()

        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'], data['orby'] = orbx, orby
        tune = self.devices['tune']
        data['tunex'], data['tuney'] = tune.tunex, tune.tuney
        bpm0 = self.devices['fambpms'].devices[0]
        csbpm = bpm0.csdata
        data['bpms_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpm'].delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
            bpm0.switching_mode]
        data['tunex_enable'] = tune.enablex
        data['tuney_enable'] = tune.enabley
        self.data = data


class OrbitAnalysis:
    """."""

    MOM_COMPACT = 1.636e-04
    NR_BPMS_SI = 160
    HARM_NR = 864
    ENERGY_SPREAD = 0.085  # [%]
    BPM_SWITCHING_FREQ = 12.5e3  # Hz
    FOFB_DOWNSAMPLING = 23
    MONIT1_DOWNSAMPLING = 25

    def __init__(self, filename=''):
        """."""
        self.fname = filename
        self.orm_client = _sconf.ConfigDBClient(config_type='si_orbcorr_respm')
        self._data = None
        self._etax, self._etay = None, None
        self._orbx, self._orby = None, None
        self._orm_meas = None
        self._sampling_freq = None
        self.load_orb()
        self.get_closest_orm()
        self.sampling_freq = self._get_sampling_freq(self.data)

    @property
    def fname(self):
        """."""
        return self._fname

    @fname.setter
    def fname(self, val):
        self._fname = val

    @property
    def data(self):
        """."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def orbx(self):
        """."""
        return self._orbx

    @orbx.setter
    def orbx(self, val):
        self._orbx = val

    @property
    def orby(self):
        """."""
        return self._orby

    @orby.setter
    def orby(self, val):
        self._orby = val

    @property
    def etax(self):
        """."""
        return self._etax

    @etax.setter
    def etax(self, val):
        self._etax = val

    @property
    def etay(self):
        """."""
        return self._etay

    @etay.setter
    def etay(self, val):
        self._etay = val

    @property
    def orm(self):
        """."""
        return self._orm

    @orm.setter
    def orm(self, val):
        self._orm = val

    @property
    def sampling_freq(self):
        """."""
        return self._sampling_freq

    @sampling_freq.setter
    def sampling_freq(self, val):
        self._sampling_freq = val

    def load_orb(self):
        """."""
        data = load_pickle(self.fname)
        timestamp = _datetime.datetime.fromtimestamp(data['timestamp'])
        gtmp = '{0:<20s} = {1:}  {2:}\n'.format
        ftmp = '{0:<20s} = {1:9.5f}  {2:}\n'.format
        stg = gtmp('filename', self.fname, '')
        stg += gtmp('timestamp', timestamp, '')
        stg += ftmp('stored_current', data['stored_current'], 'mA')
        stg += ftmp('tunex', data['tunex'], '')
        stg += ftmp('tuney', data['tuney'], '')
        stg += gtmp('tunex_enable', bool(data['tunex_enable']), '')
        stg += gtmp('tuney_enable', bool(data['tuney_enable']), '')
        stg += gtmp('bpms_acq_rate', data['bpms_acq_rate'], '')
        stg += gtmp('bpms_switching_mode', data['bpms_switching_mode'], '')
        stg += gtmp('bpms_nrsamples_pre', data['bpms_nrsamples_pre'], '')
        stg += gtmp('bpms_nrsamples_post', data['bpms_nrsamples_post'], '')
        # print(stg)
        orbx, orby = data['orbx'], data['orby']
        # zero mean in samples dimension
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.data = data
        self.orbx, self.orby = orbx, orby

    def get_closest_orm(self):
        """Find Orbit Response Matrix measured close to data acquisition."""
        configs = self.orm_client.find_configs()
        delays = []
        for cfg in configs:
            dtime = abs(self.data['timestamp']-cfg['created'])
            delays.append(dtime)
        orm_name = configs[_np.argmin(delays)]['name']
        orm_meas = _np.array(
            self.orm_client.get_config_value(name=orm_name))
        orm_meas = _np.reshape(orm_meas, (2*self.NR_BPMS_SI, -1))
        rf_freq = self.data['rf_frequency']
        etaxy = orm_meas[:, -1] * (-self.MOM_COMPACT*rf_freq)  # units of [um]
        self.etax, self.etay = etaxy[:self.NR_BPMS_SI], etaxy[self.NR_BPMS_SI:]
        self.orm = orm_meas

    def calc_integrated_spectrum(self, data, inverse=False):
        """."""
        spec, freq = self.calc_spectrum(data, fs=self.sampling_freq)
        spec *= spec
        if inverse:
            intpsd = _np.sqrt(2*_np.cumsum(spec[::-1], axis=0))[::-1]
        else:
            intpsd = _np.sqrt(2*_np.cumsum(spec, axis=0))
        return intpsd, freq

    def remove_switching_freq(self):
        """."""
        fs = self.sampling_freq
        fil_orbx, freq = self.filter_matrix(
            self.orbx, fmin=0, fmax=self.BPM_SWITCHING_FREQ*0.9, fs=fs)
        fil_orby, _ = self.filter_matrix(
            self.orby, fmin=0, fmax=self.BPM_SWITCHING_FREQ*0.9, fs=fs)
        return fil_orbx, fil_orby, freq

    def filter_around_sync_freq(self, central_freq=24*64, window=5):
        """."""
        fil_orbx, fil_orby, _ = self.remove_switching_freq()
        fmin = central_freq - window/2
        fmax = central_freq + window/2
        fs = self.sampling_freq
        fil_orbx, _ = self.filter_matrix(
            self.orbx, fmin=fmin, fmax=fmax, fs=fs)
        fil_orby, _ = self.filter_matrix(
            self.orby, fmin=fmin, fmax=fmax, fs=fs)
        return fil_orbx, fil_orby

    def energy_stability_analysis(self, central_freq=24*64, window=5):
        """."""
        orbx, orby = self.filter_around_sync_freq(
            central_freq=central_freq, window=window)
        orbxy_fil = _np.hstack((orbx, orby))
        etaxy = _np.hstack((self.etax, self.etay))
        _, _, vhmat = self._calc_pca(orbxy_fil)
        corrs = []
        etaxy -= _np.mean(etaxy)
        for mode in range(vhmat.shape[0]):
            vech_nomean = vhmat[mode] - _np.mean(vhmat[mode])
            corrs.append(abs(self._calc_correlation(vech_nomean, etaxy)))
        maxcorr_idx = _np.argmax(corrs)
        vheta = vhmat[maxcorr_idx]
        vheta -= _np.mean(vheta)
        gamma = _np.dot(etaxy, vheta)/_np.dot(etaxy, etaxy)
        eta_meas = vheta/gamma
        orbx_ns, orby_ns, _ = self.remove_switching_freq()
        orbxy = _np.hstack((orbx_ns, orby_ns))
        orbxy -= _np.mean(orbxy, axis=1)[:, None]
        denergy = _np.dot(orbxy, vheta) * gamma
        return eta_meas, denergy

    def plot_energy_spectrum(self, denergy, label='', figname=''):
        """."""
        energy_spec, freq = self.calc_spectrum(denergy, fs=self.sampling_freq)

        fig, ax = _plt.subplots(1, 1, figsize=(18, 6))
        ax.plot(freq, energy_spec, label=label)
        self._plot_ripple_rfjitter_harmonics(freq, ax)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Amplitude for DFT of $\delta(t)$')
        ax.legend(
            loc='upper right', bbox_to_anchor=(1.12, 1.02), prop={'size': 14})

        ax.set_xlim([0, 5000])
        ax.set_ylim([1e-10, 1e-2])
        ax.set_yscale('log')
        ax.grid(False)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        fig.show()
        return fig

    def plot_energy_integrated_psd(
            self, denergy, inverse=True, title='', label='', figname=''):
        """."""
        fig, ax = _plt.subplots(1, 1, figsize=(12, 6))
        intpsd, freq = self.calc_integrated_spectrum(
            denergy, inverse=inverse)
        freq = freq/1e3

        ax.plot(freq, intpsd, label=label)
        self._plot_ripple_rfjitter_harmonics(freq, ax)
        ax.axhline(
            OrbitAnalysis.ENERGY_SPREAD*0.1,
            ls='--', label=r'10$\%$ of $\sigma_{\delta}$', color='k')
        ax.legend(
            loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})

        ax.set_title(title)
        ax.set_xlabel('Frequency [kHz]')
        ax.set_ylabel(r'Sqrt of Int. Spec. [$\%$]')
        ax.set_xlim([0.1, 3])
        ax.grid(False)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        fig.show()
        return fig, ax

    # static methods
    @staticmethod
    def filter_matrix(matrix, fmin=0, fmax=None, fs=1):
        """."""
        if fmax is None:
            fmax = fs/2
        dft = _np.fft.rfft(matrix, axis=0)
        freq = _np.fft.rfftfreq(matrix.shape[0], d=1/fs)
        idcs = (freq < fmin) | (freq > fmax)
        dft[idcs] = 0
        return _np.fft.irfft(dft, axis=0), freq

    @staticmethod
    def calc_spectrum(data, fs=1):
        """."""
        dft = _np.fft.rfft(data, axis=0)
        freq = _np.fft.rfftfreq(data.shape[0], d=1/fs)
        spec = _np.abs(dft)/data.shape[0]
        return spec, freq

    @staticmethod
    def _get_sampling_freq(data):
        """."""
        fs = data['rf_frequency'] / OrbitAnalysis.HARM_NR
        if data['bpms_acq_rate'] == 'FOFB':
            return fs / OrbitAnalysis.FOFB_DOWNSAMPLING
        elif data['bpms_acq_rate'] == 'Monit1':
            return fs / OrbitAnalysis.MONIT1_DOWNSAMPLING

    @staticmethod
    def _calc_pca(data):
        """."""
        umat, svals, vhmat = _np.linalg.svd(data, full_matrices=False)
        return umat, svals, vhmat

    @staticmethod
    def _calc_correlation(vec1, vec2):
        """."""
        return _np.corrcoef(vec1, vec2)[0, 1]

    @staticmethod
    def _calc_ripple_rfjitter_harmonics(freq):
        rfreq = round(_np.max(freq)/60)
        ripple = _np.arange(0, rfreq) * 60
        jfreq = round(_np.max(freq)/64)
        rfjitt = _np.arange(0, jfreq) * 64
        return ripple, rfjitt

    @staticmethod
    def _plot_ripple_rfjitter_harmonics(freq, ax):
        ripple, rfjitt = OrbitAnalysis._calc_ripple_rfjitter_harmonics(freq)
        for idx, rip in enumerate(ripple):
            lab = r'n $\times$ 60Hz' if not idx else ''
            ax.axvline(
                x=rip, ls='--', lw=1, label=lab, color='k')
        for idx, jit in enumerate(rfjitt):
            lab = r'n $\times$ 64Hz' if not idx else ''
            ax.axvline(
                x=jit, ls='--', lw=2, label=lab, color='tab:red')
