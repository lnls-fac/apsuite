"""."""
import numpy as _np
import scipy.fft as _sp_fft
import matplotlib.pyplot as _plt
import datetime as _datetime

import siriuspy.clientconfigdb as _sconf

from .. import asparams as _asparams

from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals


class OrbitAnalysis(_AcqBPMsSignals):
    """."""

    MOM_COMPACT = _asparams.SI_MOM_COMPACT
    NUM_BPMS = _asparams.SI_NUM_BPMS
    HARM_NUM = _asparams.SI_HARM_NUM
    ENERGY_SPREAD = _asparams.SI_ENERGY_SPREAD

    def __init__(
                self, filename='', orm_name='', isonline=False,
                ispost_mortem=False):
        """Analysis of orbit over time at BPMs for a given acquisition rate.

        Args:
            filename (str, optional): filename of the pickle file with orbit
                data. Defaults to ''
            orm_name (str, optional): name of the ORM to be used as reference
                for orbit analysis. Defaults to ''
        """
        super().__init__(isonline=isonline, ispost_mortem=ispost_mortem)
        self._fname = filename
        self._etax, self._etay = None, None
        self._orbx, self._orby = None, None
        self._orm_meas = None
        self._sampling_freq = None
        self._switching_freq = None
        self._rf_freq = None
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
        keys = self.load_and_apply(self.fname)
        if keys:
            print('The following keys were not used:')
            print('    ', str(keys))
        data = self.data
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
        stg += gtmp('acq_rate', data['acq_rate'], '')
        stg += gtmp('switching_mode', data['switching_mode'], '')
        stg += gtmp(
            'switching_frequency', data['switching_frequency'], '')
        stg += gtmp('nrsamples_pre', data['nrsamples_pre'], '')
        stg += gtmp('nrsamples_post', data['nrsamples_post'], '')
        orbx, orby = data['orbx'].copy(), data['orby'].copy()
        # zero mean in samples dimension
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.orbx, self.orby = orbx, orby
        self.rf_freq = data['rf_frequency']
        self._get_sampling_freq()
        self._get_switching_freq()
        return stg

    def load_and_apply(self, fname, orm_name=''):
        """Load and apply `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be '.pickle' for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        """
        keys = super().load_and_apply(fname)
        self.get_appropriate_orm_data(orm_name)
        self._get_sampling_freq()
        self._get_switching_freq()
        return keys

    def get_appropriate_orm_data(self, orm_name=''):
        """Find Orbit Response Matrix measured close to data acquisition."""
        if not orm_name:
            orm_meas = self.find_orm_with_closest_created_time(
                orm_client=self.orm_client, timestamp=self.data['timestamp'])
        else:
            orm_meas = _np.array(
                self.orm_client.get_config_value(name=orm_name))
        self.rf_freq = self.data['rf_frequency']
        etaxy = orm_meas[:, -1]
        etaxy *= (-self.MOM_COMPACT*self.rf_freq)  # units of [um]
        self.etax, self.etay = etaxy[:self.NUM_BPMS], etaxy[self.NUM_BPMS:]
        self.orm = orm_meas

    def calc_integrated_spectrum(self, spec, inverse=False):
        """."""
        spec_abs = _np.abs(spec)
        spec2 = spec_abs*spec_abs
        if inverse:
            intpsd = _np.sqrt(2*_np.cumsum(spec2[::-1], axis=0))[::-1]
        else:
            intpsd = _np.sqrt(2*_np.cumsum(spec2, axis=0))
        return intpsd

    def filter_switching(self, orb):
        """."""
        fsmp = self.sampling_freq
        fswt = self.switching_freq
        sw_mode = self.data['switching_mode']
        # remove switching only if switching mode was on during acquisition
        # AND the sampling frequency is greater than switching frequency
        if fsmp / fswt > 1 and sw_mode == 'switching':
            return self.filter_switching_cycles(orb, fsmp, fswt)
        return orb

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
        fil_orbx = self.filter_data_frequencies(
            orbx, fmin=fmin, fmax=fmax, fsampling=fs)
        fil_orby = self.filter_data_frequencies(
            orby, fmin=fmin, fmax=fmax, fsampling=fs)
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
        orbx_ns = self.filter_switching(self.orbx)
        orby_ns = self.filter_switching(self.orby)
        orbx, orby = self.filter_around_freq(
            orbx=orbx_ns, orby=orby_ns,
            central_freq=central_freq, window=window)

        if use_eta_meas:
            eta2use = self.calculate_eta_meas(orbx, orby, self.etax, self.etay)
        else:
            eta2use = _np.hstack((self.etax, self.etay))

        orbxy = _np.hstack((orbx_ns, orby_ns))
        coef = _np.polynomial.polynomial.polyfit(eta2use, orbxy.T, deg=1)
        denergy = coef[1]

        energy_spec, freq = self.calc_spectrum(denergy, fs=self.sampling_freq)
        intpsd = self.calc_integrated_spectrum(energy_spec, inverse=inverse)

        self.analysis['measured_dispersion'] = eta2use if use_eta_meas \
            else None
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
            central_freq (float, optional): harmonic of interest to be
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
        orbx_ns = self.filter_switching(self.orbx)
        orby_ns = self.filter_switching(self.orby)
        orbx_fil, orby_fil = self.filter_around_freq(
            orbx=orbx_ns, orby=orby_ns,
            central_freq=central_freq, window=window)
        orbx_spec, freqx = self.calc_spectrum(orbx_fil, fs=self.sampling_freq)
        orby_spec, freqy = self.calc_spectrum(orby_fil, fs=self.sampling_freq)
        ipsdx = self.calc_integrated_spectrum(orbx_spec, inverse=inverse)
        ipsdy = self.calc_integrated_spectrum(orby_spec, inverse=inverse)
        self.analysis['orbx_filtered'] = orbx_fil
        self.analysis['orby_filtered'] = orby_fil
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
            umatx, svalsx, vhmatx = self.calc_svd(orbx_fil)
            umaty, svalsy, vhmaty = self.calc_svd(orby_fil)
            self.analysis['orbx_umat'] = umatx
            self.analysis['orbx_svals'] = svalsx
            self.analysis['orbx_vhmat'] = vhmatx
            self.analysis['orby_umat'] = umaty
            self.analysis['orby_svals'] = svalsy
            self.analysis['orby_vhmat'] = vhmaty
        else:
            orbxy_fil = _np.hstack((orbx_fil, orby_fil))
            umatxy, svalsxy, vhmatxy = self.calc_svd(orbxy_fil)
            self.analysis['orbxy_umat'] = umatxy
            self.analysis['orbxy_svals'] = svalsxy
            self.analysis['orbxy_vhmat'] = vhmatxy

    def process_data_energy(
            self, central_freq=24*64, window=5, inverse=True, orm_name='',
            use_eta_meas=True):
        """Energy Stability Analysis."""
        self.subtract_average_orb()
        self.get_appropriate_orm_data(orm_name)
        self.energy_stability_analysis(
            central_freq=central_freq, window=window, inverse=inverse,
            use_eta_meas=use_eta_meas)

    def process_data_orbit(
            self, central_freq=60, window=10, inverse=False, pca=True,
            split_planes=True):
        """Orbit Stability Analysis."""
        self.subtract_average_orb()
        self.orbit_stability_analysis(
            central_freq=central_freq, window=window,
            inverse=inverse, pca=pca, split_planes=split_planes)

    def subtract_average_orb(self):
        """."""
        orbx = self.data['orbx'].copy()
        orby = self.data['orby'].copy()
        orbx -= orbx.mean(axis=0)[None, :]
        orby -= orby.mean(axis=0)[None, :]
        self.orbx, self.orby = orbx, orby

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
        axs[0].plot(
            freqx, _np.abs(orbx_spec)[:, bpmidx], label=label, color=color)
        axs[1].plot(
            freqy, _np.abs(orby_spec)[:, bpmidx], label=label, color=color)
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
            umatx, svalsx, vhmatx = self.calc_svd(orbx)
        if orby is None:
            anly = self.analysis
            try:
                umaty, vhmaty = anly['orby_umat'], anly['orby_vhmat']
                svalsy = anly['orby_svals']
            except KeyError:
                print('PCA results of y orbit missing in analysis dict.')
                return None
        else:
            umaty, svalsy, vhmaty = self.calc_svd(orby)

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
        axs.plot(freq, _np.abs(energy_spec)*100, label=label, color=color)
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
        samp_freq = self.data.get('sampling_frequency')
        if samp_freq is None:
            print('sampling_frequency is not in the data.')
        self.sampling_freq = samp_freq

    def _get_switching_freq(self):
        swc_freq = self.data.get('switching_frequency')
        if swc_freq is None:
            print('switching_frequency is not in the data.')
        self.switching_freq = swc_freq

    # static methods
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
            ax.axvline(x=rip, ls='--', lw=1, label=lab, color='k')
        for idx, jit in enumerate(rfjitt):
            lab = r'n $\times$ 64Hz' if not idx else ''
            ax.axvline(x=jit, ls='--', lw=2, label=lab, color='tab:red')

    @staticmethod
    def calculate_eta_meas(orbx, orby, etax, etay):
        """Calculate the dispersion function from measured orbits."""
        orbxy_fil = _np.hstack((orbx, orby))
        _, _, vhmat = _AcqBPMsSignals.calc_svd(orbxy_fil)
        etaxy = _np.hstack((etax, etay))
        etaxy -= _np.mean(etaxy)

        vhmat_ = vhmat - vhmat.mean(axis=1)[:, None]
        correls = _np.abs(_np.dot(vhmat_, etaxy))
        idx = _np.argmax(correls)
        vheta = vhmat[idx]
        vheta_ = vhmat_[idx]

        # Find scale factor via least-squares minimization
        gamma = _np.dot(etaxy, vheta_)/_np.dot(etaxy, etaxy)
        return vheta/gamma

    @staticmethod
    def find_orm_with_closest_created_time(orm_client, timestamp):
        """Find the measured ORM with created time closest to timestamp."""
        configs = orm_client.find_configs()
        delays = _np.array([cfg['created'] for cfg in configs])
        delays -= timestamp
        orm_name = configs[_np.argmin(_np.abs(delays))]['name']
        return _np.array(orm_client.get_config_value(name=orm_name))
