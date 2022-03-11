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
        self._data = None
        self._etax, self._etay = None, None
        self._orbx, self._orby = None, None
        self._orm_meas = None
        self._sampling_freq = None
        self.analysis = dict()
        self.orm_client = _sconf.ConfigDBClient(config_type='si_orbcorr_respm')
        if self.fname:
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
        """Load files in old format."""
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
        orbx, orby = data['orbx'], data['orby']
        # zero mean in samples dimension
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.data = data
        self.orbx, self.orby = orbx, orby
        return stg

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

    def calc_integrated_spectrum(self, spec, inverse=False):
        """."""
        spec2 = spec*spec
        if inverse:
            intpsd = _np.sqrt(2*_np.cumsum(spec2[::-1], axis=0))[::-1]
        else:
            intpsd = _np.sqrt(2*_np.cumsum(spec2, axis=0))
        return intpsd

    def remove_switching_freq(self, orbx=None, orby=None):
        """."""
        orbx = orbx or self.orbx
        orby = orby or self.orby
        fs = self.sampling_freq
        fil_orbx, freq = self.filter_matrix(
            orbx, fmin=0, fmax=self.BPM_SWITCHING_FREQ*0.9, fs=fs)
        fil_orby, _ = self.filter_matrix(
            orby, fmin=0, fmax=self.BPM_SWITCHING_FREQ*0.9, fs=fs)
        return fil_orbx, fil_orby, freq

    def filter_around_freq(
            self, orbx=None, orby=None, central_freq=24*64, window=5):
        """."""
        fmin = central_freq - window/2
        fmax = central_freq + window/2
        fs = self.sampling_freq
        fil_orbx, _ = self.filter_matrix(orbx, fmin=fmin, fmax=fmax, fs=fs)
        fil_orby, _ = self.filter_matrix(orby, fmin=fmin, fmax=fmax, fs=fs)
        return fil_orbx, fil_orby

    def energy_stability_analysis(
            self, central_freq=24*64, window=5, inverse=True):
        """."""
        orbx_ns, orby_ns, _ = self.remove_switching_freq()
        orbx, orby = self.filter_around_freq(
            orbx=orbx_ns, orby=orby_ns,
            central_freq=central_freq, window=window)

        orbxy_fil = _np.hstack((orbx, orby))
        _, _, vhmat = self._calc_pca(orbxy_fil)
        etaxy = _np.hstack((self.etax, self.etay))
        etaxy -= _np.mean(etaxy)

        correls = []
        for mode in range(vhmat.shape[0]):
            vech_nomean = vhmat[mode] - _np.mean(vhmat[mode])
            correls.append(abs(self._calc_correlation(vech_nomean, etaxy)))

        maxcorr_idx = _np.argmax(correls)
        vheta = vhmat[maxcorr_idx]
        vheta -= _np.mean(vheta)

        gamma = _np.dot(etaxy, vheta)/_np.dot(etaxy, etaxy)
        orbxy = _np.hstack((orbx_ns, orby_ns))
        orbxy -= _np.mean(orbxy, axis=1)[:, None]
        eta_meas = vheta/gamma
        denergy = _np.dot(orbxy, vheta) * gamma

        energy_spec, freq = self.calc_spectrum(denergy, fs=self.sampling_freq)
        intpsd = self.calc_integrated_spectrum(energy_spec, inverse=inverse)

        self.analysis['measured_dispersion'] = eta_meas
        self.analysis['energy_freqmax'] = central_freq + window/2
        self.analysis['energy_freqmin'] = central_freq - window/2
        self.analysis['energy_deviation'] = denergy
        self.analysis['energy_spectrum'] = energy_spec
        self.analysis['energy_freq'] = freq
        self.analysis['energy_ipsd'] = intpsd
        return eta_meas, denergy

    def orbit_stability_analysis(
            self, central_freq=60, window=10, inverse=False):
        """."""
        orbx, orby, _ = self.filter_around_freq(
            central_freq=central_freq, window=window)
        orbx_spec, freqx = self.calc_spectrum(orbx, fs=self.sampling_freq)
        orby_spec, freqy = self.calc_spectrum(orby, fs=self.sampling_freq)
        ipsdx = self.calc_integrated_spectrum(orbx_spec, inverse=inverse)
        ipsdy = self.calc_integrated_spectrum(orby_spec, inverse=inverse)
        umatx, svalsx, vhmatx = self.calc_pca(orbx)
        umaty, svalsy, vhmaty = self.calc_pca(orby)
        self.analysis['orb_freqmax'] = central_freq + window/2
        self.analysis['orb_freqmin'] = central_freq - window/2
        self.analysis['orbx_spectrum'] = orbx_spec
        self.analysis['orby_spectrum'] = orby_spec
        self.analysis['orbx_freq'] = freqx
        self.analysis['orby_freq'] = freqy
        self.analysis['orbx_ipsd'] = ipsdx
        self.analysis['orby_ipsd'] = ipsdy
        self.analysis['orbx_umat'] = umatx
        self.analysis['orbx_svals'] = svalsx
        self.analysis['orbx_vhmat'] = vhmatx
        self.analysis['orby_umat'] = umaty
        self.analysis['orby_svals'] = svalsy
        self.analysis['orby_vhmat'] = vhmaty

    def plot_orbit_spectrum(
            self, bpmidx=0, orbx=None, orby=None,
            title='', label='', figname=''):
        """."""
        if orbx:
            orbx_spec, freqx = self.calc_spectrum(orbx, fs=self.sampling_freq)
        else:
            orbx_spec = self.analysis['orbx_spectrum']
            freqx = self.analysis['orbx_freq']
        if orby:
            orby_spec, freqx = self.calc_spectrum(orby, fs=self.sampling_freq)
        else:
            orby_spec = self.analysis['orby_spectrum']
            freqy = self.analysis['orby_freq']
        fig, axs = _plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(freqx, orbx_spec[:, bpmidx], label=label)
        axs[1].plot(freqy, orby_spec[:, bpmidx], label=label)
        axs[0].set_title(title)
        axs[1].legend()
        axs[0].set_ylabel('x [um]')
        axs[1].set_ylabel('y [um]')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[0].grid(False)
        axs[1].grid(False)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        fig.show()
        return fig, axs

    def plot_orbit_integrated_spectrum(
            self, bpmidx=0, orbx=None, orby=None, inverse=False,
            title='', label='', figname=''):
        """."""
        if orbx:
            orbx_spec, freqx = self.calc_spectrum(orbx, fs=self.sampling_freq)
            ipsdx = self.calc_integrated_spectrum(orbx_spec, inverse=inverse)
        else:
            ipsdx = self.analysis['orbx_ipsd']
            freqx = self.analysis['orbx_freq']
        if orby:
            orby_spec, freqy = self.calc_spectrum(orby, fs=self.sampling_freq)
            ipsdy = self.calc_integrated_spectrum(orby_spec, inverse=inverse)
        else:
            ipsdy = self.analysis['orby_ipsd']
            freqy = self.analysis['orby_freq']
        fig, axs = _plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(freqx, ipsdx[:, bpmidx], label=label)
        axs[1].plot(freqy, ipsdy[:, bpmidx], label=label)
        axs[0].set_title(title)
        axs[1].legend()
        axs[0].set_ylabel('x [um]')
        axs[1].set_ylabel('y [um]')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[0].grid(False)
        axs[1].grid(False)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_orbit_spacial_modes(
            self, modes=[], orbx=None, orby=None, spacx0=None, spacy0=None,
            title='', label='', figname=''):
        """."""
        if orbx:
            umatx, svalsx, vhmatx = self.calc_pca(orbx)
        else:
            anly = self.analysis
            umatx, vhmatx = anly['orbx_umat'], anly['orbx_vhmat']
            svalsx = anly['orbx_svals']
        if orby:
            umaty, svalsy, vhmaty = self.calc_pca(orby)
        else:
            anly = self.analysis
            umaty, vhmaty = anly['orby_umat'], anly['orby_vhmat']
            svalsy = anly['orby_svals']

        spacx = vhmatx[modes].T*svalsx[modes]/_np.sqrt(umatx.shape[0])
        spacy = vhmaty[modes].T*svalsy[modes]/_np.sqrt(umaty.shape[0])
        spacx0 = spacx0 or spacx
        spacy0 = spacy0 or spacy
        spacx *= _np.sign(_np.sum(spacx*spacx0, axis=0))
        spacy *= _np.sign(_np.sum(spacy*spacy0, axis=0))

        fig, axs = _plt.subplots(2, 1, figsize=(12, 6))
        linesx = axs[0].plot(spacx)
        linesy = axs[1].plot(spacy)
        for mode, linx, liny in zip(modes, linesx, linesy):
            linx.set_label(label + f'mode = {mode:d}')
            liny.set_label(label + f'mode = {mode:d}')
        axs[1].legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fontsize='x-small')
        axs[0].set_ylabel('Space Modes X [um]')
        axs[1].set_ylabel('Space Modes Y [um]')
        axs[1].set_xlabel('BPM index')
        axs[0].grid(False)
        axs[1].grid(False)
        axs[0].set_title(title)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_energy_spectrum(
            self, denergy=None, title='', label='', figname=''):
        """."""
        if denergy:
            energy_spec, freq = self.calc_spectrum(
                denergy, fs=self.sampling_freq)
        else:
            energy_spec = self.analysis['energy_spectrum']
            freq = self.analysis['energy_freq']

        fig, ax = _plt.subplots(1, 1, figsize=(18, 6))
        ax.plot(freq, energy_spec, label=label)
        self._plot_ripple_rfjitter_harmonics(freq, ax)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Amplitude for DFT of $\delta(t)$')
        ax.legend(
            loc='upper right', bbox_to_anchor=(1.12, 1.02), prop={'size': 14})

        ax.set_title(title)
        ax.set_xlim([0, 5000])
        ax.set_ylim([1e-10, 1e-2])
        ax.set_yscale('log')
        ax.grid(False)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        fig.show()
        return fig, ax

    def plot_energy_integrated_psd(
            self, denergy_spec=None, inverse=True,
            title='', label='', figname=''):
        """."""
        if denergy_spec:
            intpsd = self.calc_integrated_spectrum(
                denergy_spec, inverse=inverse)
        else:
            intpsd = self.analysis['energy_ipsd']
            freq = self.analysis['freq']

        fig, ax = _plt.subplots(1, 1, figsize=(12, 6))
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


class OrbitAcquisition(OrbitAnalysis, _BaseClass):
    """."""

    BPM_TRIGGER = 'SI-Fam:TI-BPM'

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=OrbitAcquisitionParams(), isonline=isonline)
        OrbitAnalysis.__init__()

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

    def process_data_energy(
            self, central_freq=24*64, window=5, inverse=True):
        """Energy Stability Analysis."""
        self._process_orb()
        self.get_closest_orm()
        self.sampling_freq = self._get_sampling_freq(self.data)
        self.energy_stability_analysis(
            central_freq=central_freq, window=window, inverse=inverse)

    def process_data_orbit(
            self, central_freq=60, window=10, inverse=False):
        """Orbit Stability Analysis."""
        self._process_orb()
        self.sampling_freq = self._get_sampling_freq(self.data)
        self.orbit_stability_analysis(
            central_freq=central_freq, window=window, inverse=inverse)

    def _process_orb(self):
        orbx = self.data['orbx'].copy()
        orby = self.data['orby'].copy()
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.orbx, self.orby = orbx, orby
