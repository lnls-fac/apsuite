"""."""
import numpy as _np
import matplotlib.pyplot as _plt
import datetime as _datetime
import time as _time

from mathphys.functions import load_pickle
import siriuspy.clientconfigdb as _sconf
from .. import asparams as _asparams
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from siriuspy.devices import Tune, CurrInfoSI, \
    Trigger, Event, EVG, RFGen, FamBPMs
from siriuspy.diagbeam.bpm.csdev import Const as _csbpm


class OrbitAnalysis:
    """."""

    MOM_COMPACT = _asparams.SI_MOM_COMPACT
    NUM_BPMS = _asparams.SI_NUM_BPMS
    HARM_NUM = _asparams.SI_HARM_NUM
    ENERGY_SPREAD = _asparams.SI_ENERGY_SPREAD

    def __init__(self, filename='', orm_name=''):
        """Analysis of orbit over time at BPMs for a given acquisition rate.

        Args:
            filename (str, optional): filename of the pickle file with orbit
                data. Defaults to ''
            orm_name (str, optional): name of the ORM to be used as reference
                for orbit analysis. Defaults to ''
        """
        self._fname = filename
        self._data = None
        self._etax, self._etay = None, None
        self._orbx, self._orby = None, None
        self._orm_meas = None
        self._sampling_freq = None
        self._switching_freq = None
        self._rf_freq = None
        self.analysis = dict()
        self.orm_client = _sconf.ConfigDBClient(config_type='si_orbcorr_respm')
        if self.fname:
            self.load_orb()
            self.get_appropriate_orm_data(orm_name)

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

    @property
    def switching_freq(self):
        """."""
        return self._switching_freq

    @switching_freq.setter
    def switching_freq(self, val):
        self._switching_freq = val

    @property
    def rf_freq(self):
        """."""
        return self._rf_freq

    @rf_freq.setter
    def rf_freq(self, val):
        self._rf_freq = val

    def load_orb(self):
        """Load files in old format."""
        data = load_pickle(self.fname)
        timestamp = _datetime.datetime.fromtimestamp(data['timestamp'])
        gtmp = '{0:<20s} = {1:}  {2:}\n'.format
        ftmp = '{0:<20s} = {1:9.5f}  {2:}\n'.format
        stg = gtmp('filename', self.fname, '')
        stg += gtmp('timestamp', timestamp, '')
        stg += ftmp('stored_current', data['stored_current'], 'mA')
        stg += ftmp('rf_frequency', data['rf_frequency'], 'Hz')
        stg += ftmp('tunex', data['tunex'], '')
        stg += ftmp('tuney', data['tuney'], '')
        stg += gtmp('tunex_enable', bool(data['tunex_enable']), '')
        stg += gtmp('tuney_enable', bool(data['tuney_enable']), '')
        stg += gtmp('bpms_acq_rate', data['bpms_acq_rate'], '')
        stg += gtmp('bpms_switching_mode', data['bpms_switching_mode'], '')
        stg += gtmp(
            'bpms_switching_frequency', data['bpms_switching_frequency'], '')
        stg += gtmp('bpms_nrsamples_pre', data['bpms_nrsamples_pre'], '')
        stg += gtmp('bpms_nrsamples_post', data['bpms_nrsamples_post'], '')
        orbx, orby = data['orbx'], data['orby']
        # zero mean in samples dimension
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.data = data
        self.orbx, self.orby = orbx, orby
        self.rf_freq = data['rf_frequency']
        self._get_sampling_freq()
        self._get_switching_freq()
        return stg

    def get_appropriate_orm_data(self, orm_name=''):
        """Find Orbit Response Matrix measured close to data acquisition."""
        if not orm_name:
            configs = self.orm_client.find_configs()
            delays = _np.array([cfg['created'] for cfg in configs])
            delays -= self.data['timestamp']
            orm_name = configs[_np.argmin(_np.abs(delays))]['name']
        orm_meas = _np.array(
            self.orm_client.get_config_value(name=orm_name))
        orm_meas = _np.reshape(orm_meas, (2*self.NUM_BPMS, -1))
        self.rf_freq = self.data['rf_frequency']
        etaxy = orm_meas[:, -1]
        etaxy *= (-self.MOM_COMPACT*self.rf_freq)  # units of [um]
        self.etax, self.etay = etaxy[:self.NUM_BPMS], etaxy[self.NUM_BPMS:]
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
        orbx = orbx or self.orbx.copy()
        orby = orby or self.orby.copy()
        fs = self.sampling_freq
        fsw = self.switching_freq
        fil_orbx, freq = self.filter_matrix(
            orbx, fmin=0, fmax=fsw*0.9, fs=fs)
        fil_orby, _ = self.filter_matrix(
            orby, fmin=0, fmax=fsw*0.9, fs=fs)
        return fil_orbx, fil_orby, freq

    def filter_around_freq(
            self, orbx=None, orby=None, central_freq=24*64, window=5):
        """."""
        if orbx is None:
            orbx = self.orbx.copy()
        if orby is None:
            orby = self.orby.copy()
        fmin = central_freq - window/2
        fmax = central_freq + window/2
        fs = self.sampling_freq
        fil_orbx, _ = self.filter_matrix(orbx, fmin=fmin, fmax=fmax, fs=fs)
        fil_orby, _ = self.filter_matrix(orby, fmin=fmin, fmax=fmax, fs=fs)
        return fil_orbx, fil_orby

    def energy_stability_analysis(
            self, central_freq=24*64, window=5, inverse=True,
            use_eta_meas=True):
        """Calculate energy deviation and dispersion function from orbit.

        1) Filter orbit array around synchrotron frequency with some frequency
             window to obtain a filtered orbit array.
        2) Apply SVD in the filtered orbit array.
        3) Find the singular mode whose spacial signature has maximum
            correlation with a reference dispersion function from some ORM.
            Let this mode be the dispersion spacial mode.
        4) The measured dispersion function can be obtained from least-squares
            minimization of the difference between the reference dispersion
            function and the dispersion spacial mode.
        5) Calculate energy deviation over time by fitting the unfiltered
            orbit data with the dispersion function.

        Args:
            central_freq (float, optional): harmonic where synchrotron
                oscillations are excited. Units [Hz].
            window (int, optional): frequency window to filter the data.
                Units [Hz].
            inverse (bool, optional): calculate the integrated PSD with from
                lower to higher frequencies (inverse=False) or the contrary.
            use_eta_meas (bool, optional): whether to use measured eta
                function or eta function of orm to find energy deviations.
                Defaults to True.

        """
        orbx_ns, orby_ns, _ = self.remove_switching_freq()
        orbx, orby = self.filter_around_freq(
            orbx=orbx_ns, orby=orby_ns,
            central_freq=central_freq, window=window)

        orbxy_fil = _np.hstack((orbx, orby))
        _, _, vhmat = self._calc_pca(orbxy_fil)
        etaxy = _np.hstack((self.etax, self.etay))
        etaxy_nm = etaxy - _np.mean(etaxy)

        correls = []
        for mode in range(vhmat.shape[0]):
            vech_nm = vhmat[mode] - _np.mean(vhmat[mode])
            correls.append(abs(self._calc_correlation(vech_nm, etaxy_nm)))

        maxcorr_idx = _np.argmax(correls)
        vheta = vhmat[maxcorr_idx]
        vheta_nm = vheta - _np.mean(vheta)

        # Find scale factor via least-squares minimization
        gamma = _np.dot(etaxy_nm, vheta_nm)/_np.dot(etaxy_nm, etaxy_nm)
        eta_meas = vheta/gamma

        orbxy = _np.hstack((orbx_ns, orby_ns))
        eta2use = eta_meas if use_eta_meas else etaxy
        coef = _np.polynomial.polynomial.polyfit(eta2use, orbxy.T, deg=1)
        denergy = coef[1]

        energy_spec, freq = self.calc_spectrum(denergy, fs=self.sampling_freq)
        intpsd = self.calc_integrated_spectrum(energy_spec, inverse=inverse)

        self.analysis['measured_dispersion'] = eta_meas
        self.analysis['energy_freqmax'] = central_freq + window/2
        self.analysis['energy_freqmin'] = central_freq - window/2
        self.analysis['energy_deviation'] = denergy
        self.analysis['energy_spectrum'] = energy_spec
        self.analysis['energy_freq'] = freq
        self.analysis['energy_ipsd'] = intpsd

    def orbit_stability_analysis(
            self, central_freq=60, window=10, inverse=False, pca=True,
            split_planes=True):
        """Calculate orbit spectrum, integrated PSD and apply SVD in orbit
            data by filtering around a center frequency with a window.

        Args:
            central_freq (float, optional): harmonic of interested to be
                analyzed in [Hz]. Defaults to 60Hz. Units [Hz].
            window (int, optional): frequency window to filter the data.
                Units [Hz].
            inverse (bool, optional): calculate the integrated PSD with from
                lower to higher frequencies (inverse=False) or the contrary.
            pca (bool, optional): calculate SVD of orbit matrices for
                principal component analysis (PCA). Default is True.
            split_planes (bool, optional): perform PCA analysis in x and y
                planes independently. Default is True. If False, concatenates
                x and y data.

        """
        orbx_ns, orby_ns, _ = self.remove_switching_freq()
        orbx_fil, orby_fil = self.filter_around_freq(
            orbx=orbx_ns, orby=orby_ns,
            central_freq=central_freq, window=window)
        orbx_spec, freqx = self.calc_spectrum(orbx_fil, fs=self.sampling_freq)
        orby_spec, freqy = self.calc_spectrum(orby_fil, fs=self.sampling_freq)
        ipsdx = self.calc_integrated_spectrum(orbx_spec, inverse=inverse)
        ipsdy = self.calc_integrated_spectrum(orby_spec, inverse=inverse)
        self.analysis['orb_freqmax'] = central_freq + window/2
        self.analysis['orb_freqmin'] = central_freq - window/2
        self.analysis['orbx_spectrum'] = orbx_spec
        self.analysis['orby_spectrum'] = orby_spec
        self.analysis['orbx_freq'] = freqx
        self.analysis['orby_freq'] = freqy
        self.analysis['orbx_ipsd'] = ipsdx
        self.analysis['orby_ipsd'] = ipsdy
        if not pca:
            return

        if split_planes:
            umatx, svalsx, vhmatx = self._calc_pca(orbx_fil)
            umaty, svalsy, vhmaty = self._calc_pca(orby_fil)
            self.analysis['orbx_umat'] = umatx
            self.analysis['orbx_svals'] = svalsx
            self.analysis['orbx_vhmat'] = vhmatx
            self.analysis['orby_umat'] = umaty
            self.analysis['orby_svals'] = svalsy
            self.analysis['orby_vhmat'] = vhmaty
        else:
            orbxy_fil = _np.hstack((orbx_fil, orby_fil))
            umatxy, svalsxy, vhmatxy = self._calc_pca(orbxy_fil)
            self.analysis['orbxy_umat'] = umatxy
            self.analysis['orbxy_svals'] = svalsxy
            self.analysis['orbxy_vhmat'] = vhmatxy

    # plotting methods
    def plot_orbit_spectrum(
            self, bpmidx=0, orbx=None, orby=None,
            title='', label='', figname='', fig=None, axs=None, color='C0'):
        """."""
        if orbx is None:
            freqx = self.analysis['orbx_freq']
            orbx_spec = self.analysis['orbx_spectrum']
        else:
            orbx_spec, freqx = self.calc_spectrum(orbx, fs=self.sampling_freq)
        if orby is None:
            freqy = self.analysis['orby_freq']
            orby_spec = self.analysis['orby_spectrum']
        else:
            orby_spec, freqx = self.calc_spectrum(orby, fs=self.sampling_freq)

        if fig is None or axs is None:
            fig, axs = _plt.subplots(2, 1, figsize=(12, 8))
        axs[0].plot(freqx, orbx_spec[:, bpmidx], label=label, color=color)
        axs[1].plot(freqy, orby_spec[:, bpmidx], label=label, color=color)
        if title:
            axs[0].set_title(title)
        axs[0].legend(
            loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
        axs[0].set_ylabel(r'$x$ [$\mu$m]')
        axs[1].set_ylabel(r'$y$ [$\mu$m]')
        axs[1].set_xlabel('Frequency [Hz]')
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_orbit_integrated_psd(
            self, bpmidx=0, orbx=None, orby=None, inverse=False,
            title='', label='', figname='', fig=None, axs=None, color='C0',
            alpha=1.0):
        """."""
        if orbx is None:
            ipsdx = self.analysis['orbx_ipsd']
            freqx = self.analysis['orbx_freq']
        else:
            orbx_spec, freqx = self.calc_spectrum(orbx, fs=self.sampling_freq)
            ipsdx = self.calc_integrated_spectrum(orbx_spec, inverse=inverse)
        if orby is None:
            ipsdy = self.analysis['orby_ipsd']
            freqy = self.analysis['orby_freq']
        else:
            orby_spec, freqy = self.calc_spectrum(orby, fs=self.sampling_freq)
            ipsdy = self.calc_integrated_spectrum(orby_spec, inverse=inverse)
        if fig is None or axs is None:
            fig, axs = _plt.subplots(2, 1, figsize=(12, 8))
        axs[0].plot(
            freqx, ipsdx[:, bpmidx], label=label,
            color=color, alpha=alpha)
        axs[1].plot(
            freqy, ipsdy[:, bpmidx], label=label,
            color=color, alpha=alpha)
        if title:
            axs[0].set_title(title)

        if label:
            axs[0].legend(
                loc='upper right', bbox_to_anchor=(1.25, 1.02),
                prop={'size': 14})
        axs[0].set_ylabel(r'$x$ [$\mu$m]')
        axs[1].set_ylabel(r'$y$ [$\mu$m]')
        axs[1].set_xlabel('Frequency [Hz]')
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_orbit_spacial_modes(
            self, modes=[], orbx=None, orby=None, spacx0=None, spacy0=None,
            title='', label='', figname='', fig=None, axs=None, color='C0'):
        """."""
        if orbx is None:
            anly = self.analysis
            try:
                umatx, vhmatx = anly['orbx_umat'], anly['orbx_vhmat']
                svalsx = anly['orbx_svals']
            except KeyError:
                print('PCA results of x orbit missing in analysis dict.')
                return None
        else:
            umatx, svalsx, vhmatx = self._calc_pca(orbx)
        if orby is None:
            anly = self.analysis
            try:
                umaty, vhmaty = anly['orby_umat'], anly['orby_vhmat']
                svalsy = anly['orby_svals']
            except KeyError:
                print('PCA results of y orbit missing in analysis dict.')
                return None
        else:
            umaty, svalsy, vhmaty = self._calc_pca(orby)

        spacx = vhmatx[modes].T*svalsx[modes]/_np.sqrt(umatx.shape[0])
        spacy = vhmaty[modes].T*svalsy[modes]/_np.sqrt(umaty.shape[0])
        if spacx0 is None:
            spacx0 = spacx
        if spacy0 is None:
            spacy0 = spacy
        spacx *= _np.sign(_np.sum(spacx*spacx0, axis=0))
        spacy *= _np.sign(_np.sum(spacy*spacy0, axis=0))

        if fig is None or axs is None:
            fig, axs = _plt.subplots(2, 1, figsize=(12, 8))
        linesx = axs[0].plot(spacx, color=color)
        linesy = axs[1].plot(spacy, color=color)
        for mode, linx, liny in zip(modes, linesx, linesy):
            linx.set_label(label + f', mode = {mode:d}')
            liny.set_label(label + f', mode = {mode:d}')
        axs[0].legend(
            loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
        axs[0].set_ylabel(r'$x$ space modes [$\mu$m]')
        axs[1].set_ylabel(r'$y$ space modes [$\mu$m]')
        axs[1].set_xlabel('BPM index')
        if title:
            axs[0].set_title(title)
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_energy_spectrum(
            self, denergy=None,
            title='', label='', figname='', fig=None, axs=None, color='C0'):
        """."""
        if denergy is None:
            energy_spec = self.analysis['energy_spectrum']
            freq = self.analysis['energy_freq']
        else:
            energy_spec, freq = self.calc_spectrum(
                denergy, fs=self.sampling_freq)

        if fig is None or axs is None:
            fig, axs = _plt.subplots(1, 1, figsize=(18, 6))
        axs.plot(freq, energy_spec*100, label=label, color=color)
        self._plot_ripple_rfjitter_harmonics(freq, axs)
        axs.set_xlabel('Frequency [Hz]')
        axs.set_ylabel(r'Amplitude for DFT of $\delta(t)$')
        axs.legend(
            loc='upper right', bbox_to_anchor=(1.12, 1.02), prop={'size': 14})
        if title:
            axs.set_title(title)
        axs.set_yscale('log')
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def plot_energy_integrated_psd(
            self, denergy_spec=None, freq=None, inverse=True,
            title='', label='', figname='', fig=None, axs=None, color='C0'):
        """."""
        if denergy_spec is None:
            intpsd = self.analysis['energy_ipsd']
            freq = self.analysis['energy_freq']
        else:
            intpsd = self.calc_integrated_spectrum(
                denergy_spec, inverse=inverse)
            if freq is None:
                Exception('Frequency input is missing')

        if fig is None or axs is None:
            fig, axs = _plt.subplots(1, 1, figsize=(12, 6))
        freq = freq/1e3
        axs.plot(freq, intpsd*100, label=label, color=color)
        self._plot_ripple_rfjitter_harmonics(freq, axs)
        axs.legend(
            loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
        if title:
            axs.set_title(title)
        axs.set_xlabel('Frequency [kHz]')
        axs.set_ylabel(r'Sqrt of Int. Spec. [$\%$]')
        fig.tight_layout()
        if figname:
            fig.savefig(figname, dpi=300, format='pdf')
        return fig, axs

    def _get_sampling_freq(self):
        samp_freq = self.data.get('bpms_sampling_frequency')
        if samp_freq is None:
            print('bpms_sampling_frequency is not in the data.')
        self.sampling_freq = samp_freq

    def _get_switching_freq(self):
        swc_freq = self.data.get('bpms_switching_frequency')
        if swc_freq is None:
            print('bpms_switching_frequency is not in the data.')
        self.switching_freq = swc_freq

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
        self.orbit_nrpoints_before = 0
        self.orbit_nrpoints_after = 20000
        self.orbit_acq_rate = 'FAcq'
        self.orbit_acq_repeat = False

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:26s} = {1:9}  {2:s}\n'.format
        stg = ''
        stg += ftmp('trigbpm_delay', self.trigbpm_delay, '[us]')
        stg += dtmp('trigbpm_nrpulses', self.trigbpm_nrpulses, '')
        stg += ftmp('event_delay', self.event_delay, '[us]')
        stg += stmp('event_mode', self.event_mode, '')
        stg += ftmp('orbit_timeout', self.orbit_timeout, '[s]')
        stg += dtmp('orbit_nrpoints_before', self.orbit_nrpoints_before, '')
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
        OrbitAnalysis.__init__(self)

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
        ret = fambpms.mturn_config_acquisition(
            nr_points_after=prms.orbit_nrpoints_after,
            nr_points_before=prms.orbit_nrpoints_before,
            acq_rate=prms.orbit_acq_rate,
            repeat=prms.orbit_acq_repeat)
        return ret

    def acquire_data(self):
        """."""
        fambpms = self.devices['fambpms']
        evt_study = self.devices['evt_study']
        ret = self.prepare_bpms_acquisition()
        if ret < 0:
            print(f'BPM {-ret-1:d} did not finish last acquisition.')
        elif ret > 0:
            print(f'BPM {ret-1:d} is not ready for acquisition.')
        get_sum = False
        fambpms.mturn_reset_flags_and_update_initial_orbit(
            consider_sum=get_sum)
        evt_study.cmd_external_trigger()
        time0 = _time.time()
        ret = fambpms.mturn_wait_update_orbit(
            timeout=self.params.orbit_timeout, consider_sum=get_sum)
        print(f'it took {_time.time()-time0:02f}s to update bpms')
        if ret != 0:
            raise Exception(
                f'There was a problem with acquisition. Error code {ret:d}')
        orbx, orby = fambpms.get_mturn_orbit(return_sum=get_sum)

        data = dict()
        data['timestamp'] = _time.time()
        self.rf_freq = self.devices['rfgen'].frequency
        data['rf_frequency'] = self.rf_freq
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'], data['orby'] = orbx, orby
        tune = self.devices['tune']
        data['tunex'], data['tuney'] = tune.tunex, tune.tuney
        bpm0 = self.devices['fambpms'].devices[0]
        data['bpms_acq_rate'] = bpm0.acq_channel_str
        data['bpms_sampling_frequency'] = fambpms.get_sampling_frequency(
            self.rf_freq)
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpm'].delay_raw
        data['bpms_switching_mode'] = bpm0.switching_mode_str
        data['bpms_switching_frequency'] = fambpms.get_switching_frequency(
            self.rf_freq)
        data['tunex_enable'] = tune.enablex
        data['tuney_enable'] = tune.enabley
        self.data = data

    def process_data_energy(
            self, central_freq=24*64, window=5, inverse=True,
            orm_name='', use_eta_meas=True):
        """Energy Stability Analysis."""
        self._subtract_average_orb()
        self.get_appropriate_orm_data(orm_name)
        self.energy_stability_analysis(
            central_freq=central_freq, window=window, inverse=inverse,
            use_eta_meas=use_eta_meas)

    def process_data_orbit(
            self, central_freq=60, window=10, inverse=False, pca=True,
            split_planes=True):
        """Orbit Stability Analysis."""
        self._subtract_average_orb()
        self.orbit_stability_analysis(
            central_freq=central_freq, window=window,
            inverse=inverse, pca=pca, split_planes=split_planes)

    def _subtract_average_orb(self):
        orbx = self.data['orbx'].copy()
        orby = self.data['orby'].copy()
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.orbx, self.orby = orbx, orby

    def load_and_apply(self, fname, orm_name=''):
        """."""
        super().load_and_apply(fname)
        self.get_appropriate_orm_data(orm_name)
        self._get_sampling_freq()
        self._get_switching_freq()
