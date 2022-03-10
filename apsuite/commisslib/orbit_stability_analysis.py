"""."""
import numpy as _np
import matplotlib.pyplot as _plt
import datetime as _datetime

from mathphys.functions import load_pickle
import siriuspy.clientconfigdb as _sconf


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
        self.sampling_freq = self.get_sampling_freq(self.data)

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
        _, _, vhmat = self.calc_pca(orbxy_fil)
        corrs = []
        etaxy -= _np.mean(etaxy)
        for mode in range(vhmat.shape[0]):
            vech_nomean = vhmat[mode] - _np.mean(vhmat[mode])
            corrs.append(abs(self.calc_correlation(vech_nomean, etaxy)))
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
        fig = _plt.figure(figsize=(18, 6))
        _plt.plot(freq, energy_spec, label=label)

        rfreq = round(_np.max(freq)/60)
        ripple = _np.arange(0, rfreq) * 60
        jfreq = round(_np.max(freq)/64)
        rfjitt = _np.arange(0, jfreq) * 64

        for idx, rip in enumerate(ripple):
            lab = ''
            if not idx:
                lab = r'n $\times$ 60Hz'
            _plt.axvline(
                x=rip, ls='--', lw=1, label=lab, color='k')
        for idx, jit in enumerate(rfjitt):
            lab = ''
            if not idx:
                lab = r'n $\times$ 64Hz'
            _plt.axvline(
                x=jit, ls='--', lw=2, label=lab, color='tab:red')

        _plt.xlabel('Frequency [Hz]')
        _plt.ylabel(r'Amplitude for DFT of $\delta(t)$')
        _plt.legend(
            loc='upper right', bbox_to_anchor=(1.12, 1.02), prop={'size': 14})
        _plt.xlim([0, 5000])
        _plt.ylim([1e-10, 1e-2])
        _plt.yscale('log')
        _plt.grid(False)
        _plt.tight_layout()
        if figname:
            _plt.savefig(figname, dpi=300, format='pdf')
        _plt.show()
        return fig

    def plot_energy_integrated_psd(
            self, denergy, inverse=True, title='', label='', figname=''):
        """."""
        fig = _plt.figure(figsize=(12, 6))
        intpsd, freq = self.calc_integrated_spectrum(
            denergy, inverse=inverse)
        freq = freq/1e3
        _plt.plot(freq, intpsd, label=label)

        rfreq = round(_np.max(freq)/60)
        ripple = _np.arange(0, rfreq) * 60
        jfreq = round(_np.max(freq)/64)
        rfjitt = _np.arange(0, jfreq) * 64

        for idx, rip in enumerate(ripple):
            lab = ''
            if not idx:
                lab = r'n $\times$ 60Hz'
            _plt.axvline(
                x=rip, ls='--', lw=1, label=lab, color='k')
        for idx, jit in enumerate(rfjitt):
            lab = ''
            if not idx:
                lab = r'n $\times$ 64Hz'
            _plt.axvline(
                x=jit, ls='--', lw=2, label=lab, color='tab:red')
        # _plt.xscale('log')
        _plt.grid(False)
        _plt.axhline(
            OrbitAnalysis.ENERGY_SPREAD*0.1,
            ls='--', label=r'10$\%$ of $\sigma_{\delta}$', color='k')
        _plt.legend(
            loc='upper right', bbox_to_anchor=(1.25, 1.02), prop={'size': 14})
        _plt.title(title)
        _plt.xlabel('Frequency [kHz]')
        _plt.ylabel(r'Sqrt of Int. Spec. [$\%$]')
        _plt.xlim([0.1, 3])
        _plt.tight_layout()
        if figname:
            _plt.savefig(figname, dpi=300, format='pdf')
        _plt.show()
        return fig

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
    def get_sampling_freq(data):
        """."""
        fs = data['rf_frequency'] / OrbitAnalysis.HARM_NR
        if data['bpms_acq_rate'] == 'FOFB':
            return fs / OrbitAnalysis.FOFB_DOWNSAMPLING
        elif data['bpms_acq_rate'] == 'Monit1':
            return fs / OrbitAnalysis.MONIT1_DOWNSAMPLING

    @staticmethod
    def calc_spectrum(data, fs=1):
        """."""
        dft = _np.fft.rfft(data, axis=0)
        freq = _np.fft.rfftfreq(data.shape[0], d=1/fs)
        spec = _np.abs(dft)/data.shape[0]
        return spec, freq

    @staticmethod
    def calc_pca(data):
        """."""
        umat, svals, vhmat = _np.linalg.svd(data, full_matrices=False)
        return umat, svals, vhmat

    @staticmethod
    def calc_correlation(vec1, vec2):
        """."""
        return _np.corrcoef(vec1, vec2)[0, 1]
