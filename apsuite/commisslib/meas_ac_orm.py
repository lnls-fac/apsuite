"""Main module."""
import time as _time
from threading import Event as _Flag

import numpy as _np
import scipy.signal as _scysig
import scipy.integrate as _scyint
import scipy.optimize as _scyopt

import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from siriuspy.devices import StrengthConv, PowerSupply, BPM, CurrInfoSI, \
    Trigger, Event, EVG, RFGen
from siriuspy.sofb.csdev import SOFBFactory
import pyaccel as _pyaccel

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class ACORMParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = 5500
        self.acq_rate = 'Monit1'
        self.timeout_bpms = 10  # [s]
        self.ch_kick = 5  # [urad]
        self.cv_kick = 5  # [urad]
        self.delay_corrs = 50e-3  # [s]
        self.exc_duration = 5  # [s]

        freqs = self.find_primes(16, start=120)
        self.ch_freqs = freqs[1::2][:6]
        self.cv_freqs = freqs[::2][:8]
        self.sectors_to_measure = _np.arange(1, 21)

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += dtmp('nr_points', self.nr_points, '')
        stg += stmp('acq_rate', self.acq_rate, '')
        stg += ftmp('timeout_bpms', self.timeout_bpms, '[s]')
        stg += ftmp('ch_kick', self.ch_kick, '[urad]')
        stg += ftmp('cv_kick', self.cv_kick, '[urad]')
        stg += ftmp('delay_corrs', self.delay_corrs, '[s]')
        stg += ftmp('exc_duration', self.exc_duration, '[s]')
        stg += stmp('ch_freqs', str(self.ch_freqs), '[Hz]')
        stg += stmp('cv_freqs', str(self.cv_freqs), '[Hz]')
        stg += stmp('sectors_to_measure', str(self.sectors_to_measure), '')
        return stg

    @staticmethod
    def find_primes(n_primes, start=3):
        """."""
        primes = []
        i = start
        while True:
            for j in range(3, int(_np.sqrt(i))+1, 2):
                if not i % j:
                    break
            else:
                if i % 2:
                    primes.append(i)
            i += 1
            if len(primes) >= n_primes:
                break
        return _np.array(primes)


# TODO: Ideas for analisys:
# - try to use naff with prior filtering of data,  and making an algorithm to
#   find the frequencies closest to the desired ones
# -
class MeasACORM(_ThreadBaseClass):
    """."""

    def __init__(self, params=None, isonline=True):
        """."""
        params = ACORMParams() if params is None else params
        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        self.bpms = dict()
        self._flags = dict()

        self.sofb_data = SOFBFactory.create('SI')
        if self.isonline:
            self._create_devices()

    @classmethod
    def fitting_matrix(cls, tim, freqs, num_cycles):
        """Create the matrix used for fitting of fourier components.

        The ordering of the matriz is the following:
           mat[i, 2*j] = cos(2*pi*freqs[j]*tim[i])
           mat[i, 2*j+1] = sin(2*pi*freqs[j]*tim[i])

        Args:
            tim (numpy.ndarray): array with times
            freqs (numpy.ndarray): array with frequencies to fit.
            num_cycles (mumpy.ndarray): number of cycles of each frequency.

        Returns:
            numpy.ndarray: fitting matrix (len(tim), 2*len(freqs))

        """
        mat = _np.zeros((tim.size, 2*freqs.size))
        arg = 2*_np.pi*freqs[None, :]*tim[:, None]
        cos = _np.cos(arg)
        sin = _np.sin(arg)

        cond = arg > 2*_np.pi*num_cycles[None, :]
        cos[cond] = 0
        sin[cond] = 0
        mat[:, ::2] = cos
        mat[:, 1::2] = sin
        return mat

    def fit_fourier_components(self, data, freqs, dtim, num_cycles):
        """Fit Fourier components in signal for the given frequencies.

        Args:
            data (numpy.ndarray, NxM): signal to be fitted consisting of M
                columns of data.
            freqs (numpy.ndarray, K): K frequencies to fit Fourier components.
            dtim (numpy.ndarray, N): time vector for data columns.
            num_cycles (num.ndarray, K): number of cycles of each frequency.

        Returns:
            numpy.ndarray, KxM: Fourier amplitudes.
            numpy.ndarray, KxM: Fourier phases (phase==0 means pure sine).
            numpy.ndarray, KxM: Fourier cosine coefficients.
            numpy.ndarray, KxM: Fourier sine coefficients.

        """
        tim = _np.arange(data.shape[0]) * dtim

        mat = self.fitting_matrix(tim, freqs, num_cycles)
        coeffs, *_ = _np.linalg.lstsq(mat, data, rcond=None)

        cos = coeffs[::2]
        sin = coeffs[1::2]
        amps = _np.sqrt(cos**2 + sin**2)
        phases = _np.arctan2(cos, sin)
        return amps, phases, cos, sin

    @staticmethod
    def fit_fourier_components_naff(data, freqs0, dtim):
        """Not implemented properly yet. Please don't use this."""
        freqs = _np.zeros((len(freqs0), data.shape[1]))
        fourier = _np.zeros((len(freqs0), data.shape[1]), dtype=complex)

        for i in range(data.shape[1]):
            datum = data[:, i]
            fre, four = _pyaccel.naff.naff_general(
                datum, nr_ff=len(freqs0), is_real=True)
            fre = _np.abs(fre)
            idx = _np.argsort(fre)
            fre = fre[idx]
            four = four[idx]
            freqs[:, i] = fre/dtim
            fourier[:, i] = four

        amps = _np.abs(fourier)
        phases = _np.angle(fourier)
        cos = fourier.real
        sin = fourier.imag

        return amps, phases, cos, sin

    @staticmethod
    def calc_correlation(arr1, arr2):
        """Return the linear correlation between respective columns.

        Args:
            arr1 (numpy.ndarray, NxM): first array
            arr2 (numpy.ndarray, NxM): second array

        Returns:
            numpy.ndarray, M: correlation of the ith column of arr1 with ith
                column of arr2.

        """
        corr = (arr1 * arr2).sum(axis=0)
        corr /= _np.linalg.norm(arr1, axis=0)
        corr /= _np.linalg.norm(arr2, axis=0)
        return _np.abs(corr)

    def build_respmat(self):
        """Build response matrix from previously analysed data.

        Returns:
            numpy.ndarray, 320x281: response matrix. Missing data is filled
                with zeros.

        """
        mat = _np.zeros((320, 281), dtype=float)
        for anly in self.analysis:
            mat[:160, anly['mat_idcs']] = anly['mat_colsx']
            mat[160:, anly['mat_idcs']] = anly['mat_colsy']
        return mat

    def process_data(self, idx_ini=None, min_freq=None, max_freq=None):
        """."""
        sofb = self.sofb_data
        corr2idx = {name: i for i, name in enumerate(sofb.ch_names)}
        corr2idx.update({name: 120+i for i, name in enumerate(sofb.cv_names)})

        self.analysis = []
        for data in self.data:
            anly = dict()
            fsamp = data['rf_frequency'] / 864 / 23
            if data['bpms_acq_rate'].lower().startswith('monit'):
                fsamp /= 25
            dtim = 1/fsamp
            ch_freqs = _np.array(data['ch_frequency'])
            cv_freqs = _np.array(data['cv_frequency'])
            freqs0 = _np.r_[ch_freqs, cv_freqs]
            anly['fsamp'] = fsamp
            anly['dtim'] = dtim
            anly['freqs0'] = freqs0

            ch_ncycles = _np.array(data['ch_num_cycles'])
            cv_ncycles = _np.array(data['cv_num_cycles'])
            num_cycles = _np.r_[ch_ncycles, cv_ncycles]
            anly['num_cycles'] = num_cycles

            ch_amps = _np.array(data['ch_kick_amplitudes'])
            cv_amps = _np.array(data['cv_kick_amplitudes'])
            kicks = _np.r_[ch_amps, cv_amps]
            anly['kicks'] = kicks

            ch_idcs = _np.array([corr2idx[name] for name in data['ch_names']])
            cv_idcs = _np.array([corr2idx[name] for name in data['cv_names']])
            idcs = _np.r_[ch_idcs, cv_idcs]
            anly['mat_idcs'] = idcs

            orbx = data['orbx'].copy()
            orby = data['orby'].copy()
            orbx -= orbx.mean(axis=0)
            orby -= orby.mean(axis=0)

            tim = _np.arange(orbx.shape[0]) * dtim
            if idx_ini is None:
                idx_ini = (tim >= self.params.delay_corrs).nonzero()[0][0]
            anly['time'] = tim
            anly['idx_ini'] = idx_ini

            orbx = orbx[idx_ini:]
            orby = orby[idx_ini:]

            if min_freq is not None and max_freq is not None:
                min_freq = min_freq or 0
                max_freq = max_freq or fsamp/2
                dftx = _np.fft.rfft(orbx, axis=0)
                dfty = _np.fft.rfft(orby, axis=0)
                freq = _np.fft.rfftfreq(orbx.shape[0], d=dtim)
                idcs = (freq < min_freq) | (freq > max_freq)
                dftx[idcs] = 0
                orby = _np.fft.irfft(dfty, axis=0)
            anly['min_freq'] = min_freq
            anly['max_freq'] = max_freq

            ampx, phasex, cosx, sinx = self.fit_fourier_components(
                orbx, freqs0, dtim, num_cycles)
            ampy, phasey, cosy, siny = self.fit_fourier_components(
                orby, freqs0, dtim, num_cycles)

            signx = _np.ones(ampx.shape)
            signx[_np.abs(phasex) > (_np.pi/2)] = -1
            signy = _np.ones(ampy.shape)
            signy[_np.abs(phasey) > (_np.pi/2)] = -1

            anly['ampx'] = ampx
            anly['ampy'] = ampy
            anly['phasex'] = phasex
            anly['phasey'] = phasey
            anly['cosx'] = cosx
            anly['cosy'] = cosy
            anly['sinx'] = sinx
            anly['siny'] = siny
            anly['signx'] = signx
            anly['signy'] = signy
            anly['mat_colsx'] = (signx * ampx / kicks[:, None]).T
            anly['mat_colsy'] = (signy * ampy / kicks[:, None]).T
            self.analysis.append(anly)

    def get_data(self, chs_used, cvs_used):
        """."""
        orbx, orby = self.get_orbit()
        bpm0 = list(self.bpms.values())[0]
        csbpm = self.csbpm
        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'] = orbx
        data['orby'] = orby
        data['bpms_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpms'].delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
            bpm0.switching_mode]

        data.update({
            'ch_names': [], 'ch_amplitudes': [], 'ch_offsets': [],
            'ch_kick_amplitudes': [], 'ch_kick_offsets': [],
            'ch_frequency': [], 'ch_num_cycles': [], 'ch_cycle_type': [],
            'cv_names': [], 'cv_amplitudes': [], 'cv_offsets': [],
            'cv_kick_amplitudes': [], 'cv_kick_offsets': [],
            'cv_frequency': [], 'cv_num_cycles': [], 'cv_cycle_type': [],
            })
        for cmn in chs_used:
            data['ch_names'].append(cmn)
            cm = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_current_2_strength
            data['ch_amplitudes'].append(cm.cycle_ampl)
            data['ch_offsets'].append(cm.cycle_offset)
            data['ch_kick_amplitudes'].append(conv(cm.cycle_ampl))
            data['ch_kick_offsets'].append(conv(cm.cycle_offset))
            data['ch_frequency'].append(cm.cycle_freq)
            data['ch_num_cycles'].append(cm.cycle_num_cycles)
            data['ch_cycle_type'].append(cm.cycle_type_str)

        for cmn in cvs_used:
            data['cv_names'].append(cmn)
            cm = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_current_2_strength
            data['cv_amplitudes'].append(cm.cycle_ampl)
            data['cv_offsets'].append(cm.cycle_offset)
            data['cv_kick_amplitudes'].append(conv(cm.cycle_ampl))
            data['cv_kick_offsets'].append(conv(cm.cycle_offset))
            data['cv_frequency'].append(cm.cycle_freq)
            data['cv_num_cycles'].append(cm.cycle_num_cycles)
            data['cv_cycle_type'].append(cm.cycle_type_str)

        data['corrs_trig_delay_raw'] = self.devices['trigcorrs'].delay_raw
        return data

    def get_orbit(self):
        """."""
        orbx, orby = [], []
        for bpm_name in self.sofb_data.bpm_names:
            bpm = self.devices[bpm_name]
            orbx.append(bpm.mt_posx)
            orby.append(bpm.mt_posy)
        return _np.array(orbx).T, _np.array(orby).T

    def _create_devices(self):
        # Create objects to convert kicks to current
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.ch_names})
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.cv_names})

        # Create objects to interact with correctors
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.ch_names})
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.cv_names})

        # Create object to get stored current
        self.devices['currinfo'] = CurrInfoSI()

        # Create BPMs trigger:
        self.devices['trigbpms'] = Trigger('SI-Fam:TI-BPM')

        # Create Correctors Trigger:
        self.devices['trigcorrs'] = Trigger('SI-Glob:TI-Mags-Corrs')

        # Create event to start data acquisition sinchronously:
        self.devices['evt_study'] = Event('Study')
        self.devices['evg'] = EVG()

        # Create RF generator object
        self.devices['rfgen'] = RFGen()

        # Create BPMs
        self.bpms = dict()
        name = self.sofb_data.bpm_names[0]
        self.bpms[name] = BPM(name)
        self.csbpm = self.bpms[name].csdata

        propties_not_interested = self.bpms[name].auto_monitor_status
        propties_to_keep = ['GEN_XArrayData', 'GEN_YArrayData']
        for ppt in propties_to_keep:
            propties_not_interested.pop(ppt)

        for i, name in enumerate(self.sofb_data.bpm_names):
            if not i:
                bpm = self.bpms[name]
            else:
                bpm = BPM(name)
                self.bpms[name] = bpm
            for propty in propties_not_interested:
                bpm.set_auto_monitor(propty, False)
            for propty in propties_to_keep:
                bpm.set_auto_monitor(propty, True)
                pvo = bpm.pv_object(propty)
                self._flags[pvo.pvname] = _Flag()
                pvo.add_callback(self._set_flag)
        self.devices.update(self.bpms)

    def _do_measure(self):
        elt0 = _time.time()

        self._config_bpms(self.params.nr_points, self.params.acq_rate)
        self._config_timing(self.params.delay_corrs)

        # Shift correctors so first corrector is 01M1
        chs_shifted = self._shift_list(self.sofb_data.ch_names, 1)
        cvs_shifted = self._shift_list(self.sofb_data.cv_names, 1)

        # set operation mode to slowref
        if not self._change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return

        exc_duration = self.params.exc_duration
        ch_kick = self.params.ch_kick
        cv_kick = self.params.cv_kick
        freqh = self.params.ch_freqs
        freqv = self.params.cv_freqs

        self.data = []
        for sector in self.params.sectors_to_measure:
            elt = _time.time()

            print(f'Sector {sector:02d}:')
            slch = slice((sector-1)*6, sector*6)
            slcv = slice((sector-1)*8, sector*8)
            chs_slc = chs_shifted[slch]
            cvs_slc = cvs_shifted[slcv]

            # configure correctors
            t00 = _time.time()
            print('    Configuring Correctors...', end='')
            self._config_correctors(chs_slc, ch_kick, freqh, exc_duration)
            self._config_correctors(cvs_slc, cv_kick, freqv, exc_duration)
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # set operation mode to cycle
            t00 = _time.time()
            print('    Changing Correctors to Cycle...', end='')
            if not self._change_corrs_opmode('cycle', chs_slc + cvs_slc):
                print('Problem: Correctors not in Cycle mode.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # send event through timing system to start acquisitions
            t00 = _time.time()
            print('    Sending Timing signal...', end='')
            self._reset_flags()
            self.devices['evt_study'].cmd_external_trigger()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # Wait BPMs PV to update with new data
            t00 = _time.time()
            print('    Waiting BPMs to update...', end='')
            if not self._wait_bpms_update(timeout=self.params.timeout_bpms):
                print('Problem: timed out waiting BPMs update.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # get data
            self.data.append(self.get_data(chs_used=chs_slc, cvs_used=cvs_slc))

            # set operation mode to slowref
            t00 = _time.time()
            print('    Changing Correctors to SlowOrb...', end='')
            if not self._change_corrs_opmode('slowref', chs_slc + cvs_slc):
                print('Problem: Correctors not in SlowRef mode.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            elt -= _time.time()
            elt *= -1
            print(f'    Elapsed Time: {elt:.2f}s')
            if self._stopevt.is_set():
                print('Stopping...')
                break

        # set operation mode to slowref
        if not self._change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return

        elt0 -= _time.time()
        elt0 *= -1
        print(f'Finished!!  ET: {elt0/60:.2f}min')

    # ----------------- BPMs related methods -----------------------

    def _config_bpms(self, nr_points, acq_rate=None):
        if acq_rate is None or acq_rate.lower().startswith('monit1'):
            acq_rate = self.csbpm.AcqChan.Monit1
        else:
            acq_rate = self.csbpm.AcqChan.FOFB

        print('Configuring BPMs...', end='')
        self._cmd_bpms_acq_abort()

        for bpm in self.bpms.values():
            bpm.acq_repeat = self.csbpm.AcqRepeat.Repetitive
            bpm.acq_channel = acq_rate
            bpm.acq_trigger = self.csbpm.AcqTrigTyp.External
            bpm.acq_nrsamples_pre = 0
            bpm.acq_nrsamples_post = nr_points

        self._cmd_bpms_acq_start()
        print('Done!')

    def _cmd_bpms_acq_abort(self):
        for bpm in self.bpms.values():
            bpm.acq_ctrl = self.csbpm.AcqEvents.Abort

        for bpm in self.bpms.values():
            boo = bpm.wait_acq_finish()
            if not boo:
                print('Timed out waiting abort.')

    def _cmd_bpms_acq_start(self):
        for bpm in self.bpms.values():
            bpm.acq_ctrl = self.csbpm.AcqEvents.Start

        for bpm in self.bpms.values():
            boo = bpm.wait_acq_start()
            if not boo:
                print('Timed out waiting start.')

    def _set_flag(self, pvname, **kwargs):
        _ = kwargs
        self._flags[pvname].set()

    def _reset_flags(self):
        for flag in self._flags.values():
            flag.clear()

    def _wait_bpms_update(self, timeout=10):
        """."""
        orbx0, orby0 = self.get_orbit()
        for name, flag in self._flags.items():
            t00 = _time.time()
            if not flag.wait(timeout=timeout):
                print(f'Timed out in PV {name:s}')
                return False
            timeout -= _time.time() - t00
            timeout = max(timeout, 0)

        while timeout > 0:
            t00 = _time.time()
            orbx, orby = self.get_orbit()
            cond = _np.any(_np.all(_np.isclose(orbx0, orbx), axis=0))
            cond |= _np.any(_np.all(_np.isclose(orby0, orby), axis=0))
            if not cond:
                return True
            _time.sleep(0.1)
            timeout -= _time.time() - t00
        return False

    # ----------------- Timing related methods -----------------------

    def _config_timing(self, cm_delay):
        print('Configuring Timing...', end='')
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        evg = self.devices['evg']

        trigbpm.delay = 0.0
        trigbpm.nr_pulses = 1
        trigbpm.source = 'Study'

        trigcorr.delay = cm_delay * 1e6  # [us]
        trigcorr.nr_pulses = 1
        trigcorr.source = 'Study'

        # configure event Study to be in External mode
        evt_study.delay = 0
        evt_study.mode = 'External'

        # update event configurations in EVG
        evg.cmd_update_events()
        print('Done!')

    # ----------------- Correctors related methods -----------------------

    def _config_correctors(self, corr_names, kick, freq_vector, exc_duration):
        """."""
        for i, cmn in enumerate(corr_names):
            cmo = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_strength_2_current
            cmo.cycle_type = cmo.CYCLETYPE.Sine
            cmo.cycle_freq = freq_vector[i]
            cmo.cycle_ampl = conv(kick)
            cmo.cycle_offset = cmo.currentref_mon
            cmo.cycle_theta_begin = 0
            cmo.cycle_theta_end = 0
            cmo.cycle_num_cycles = int(exc_duration * freq_vector[i])

    def _change_corrs_opmode(self, mode, corr_names=None, timeout=10):
        """."""
        opm_sel = PowerSupply.OPMODE_SEL
        opm_sts = PowerSupply.OPMODE_STS
        mode_sel = opm_sel.Cycle if mode == 'cycle' else opm_sel.SlowRef
        mode_sts = opm_sts.Cycle if mode == 'cycle' else opm_sts.SlowRef

        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cmo = self.devices[cmn]
            cmo.opmode = mode_sel

        interval = 0.2
        for _ in range(int(timeout/interval)):
            okk = True
            corrname = ''
            for cmn in corr_names:
                cmo = self.devices[cmn]
                oki = cmo.opmode == mode_sts
                if not oki:
                    corrname = cmn
                okk &= oki
            if okk:
                return True
            _time.sleep(interval)

        print(corrname)
        return False

    @staticmethod
    def _shift_list(lst, num):
        """."""
        return lst[-num:] + lst[:-num]
