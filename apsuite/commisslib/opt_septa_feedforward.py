"""."""
import time as _time
import logging as _log

import numpy as _np
import scipy.interpolate as _scyinterp
import scipy.optimize as _scyopt
import scipy.signal as _scysig
import matplotlib.pyplot as _mplt
import pywt as _pywt

from mathphys.functions import load_pickle
from pymodels import si as _si
from siriuspy.devices import PowerSupply, SOFB, FamBPMs, Trigger
from siriuspy.clientconfigdb import ConfigDBClient as _CDBClient
from siriuspy.magnet.factory import NormalizerFactory as _Normalizer

from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams
from ..orbcorr import OrbRespmat as _OrbRespmat


class OptSeptaFFParams(_RCDSParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.wait_after_wfm_set = 1  # [s]
        self.orbit_nrpulses = 20
        self.orbit_index_start = 0
        self.orbit_index_stop = 200
        self.bpm_index_start = 0
        self.bpm_index_stop = 160
        self.orbit_planes = 'x,y'
        self.wfm_index_start = 0
        self.wfm_index_stop = 0
        self.optim_params = 'ch,cv,dly'
        self.filter_with_orm = True
        self.orm_2_filter = _np.zeros((320, 5), dtype=float)
        self.orm_indices = [0, 1, 2, 3]
        self.filter_with_wavelet = True
        self.wavelet_num_modes = 7

    def __str__(self):
        """."""
        stg = ''
        stg += '-----  Optimize Setpa FF Parameters  -----\n\n'
        stg += self._TMPF('wait_after_wfm_set', self.wait_after_wfm_set, '[s]')
        stg += self._TMPD('orbit_nrpulses', self.orbit_nrpulses, '')
        stg += self._TMPD('orbit_index_start', self.orbit_index_start, '')
        stg += self._TMPD('orbit_index_stop', self.orbit_index_stop, '')
        stg += self._TMPD('bpm_index_start', self.bpm_index_start, '')
        stg += self._TMPD('bpm_index_stop', self.bpm_index_stop, '')
        stg += self._TMPS('orbit_planes', self.orbit_planes, '')
        stg += self._TMPD('wfm_index_start', self.wfm_index_start, '')
        stg += self._TMPD('wfm_index_stop', self.wfm_index_stop, '')
        stg += self._TMPS('optim_params', self.optim_params, '')
        stg += self._TMPS('filter_with_orm', str(self.filter_with_orm), '')
        stg += self._TMPS('orm_2_filter', str(self.orm_2_filter.shape), '')
        stg += self._TMPS('orm_indices', str(self.orm_indices), '')
        stg += self._TMPS(
            'filter_with_wavelet', str(self.filter_with_wavelet), '')
        stg += self._TMPD('wavelet_num_modes', self.wavelet_num_modes, '')
        stg += '\n\n-----  RCDS Parameters  -----\n\n'
        stg += super().__str__()
        return stg

    def update_limits_and_directions(
            self, ps_low=-1, ps_high=1, dly_low=-50, dly_high=50):
        """."""
        init_pos = self.initial_position
        self.initial_search_directions = _np.eye(init_pos.size)
        self.limit_lower = _np.ones(init_pos.size) * ps_low
        self.limit_upper = _np.ones(init_pos.size) * ps_high
        if 'dly' in self.optim_params.lower():
            self.limit_lower[-1] = dly_low
            self.limit_upper[-1] = dly_high


class OptSeptaFF(_RCDS):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _RCDS.__init__(self, isonline=isonline, use_thread=use_thread)
        self.params = OptSeptaFFParams()

        if self.isonline:
            self._create_devices()

    def objective_function(self, pos=None):
        """."""
        if pos is not None:
            self.apply_position_to_machine(pos)
            _time.sleep(self.params.wait_after_wfm_set)
        else:
            pos = self.get_current_position_from_machine()
        if 'positions' not in self.data:
            self.data['positions'] = []
            self.data['obj_funcs'] = []
        self.data['positions'].append(pos)

        if self._stopevt.is_set():
            return 0.0

        res = self.get_residue_vector()
        objective = res.std()**2

        self.data['obj_funcs'].append(objective)
        return objective

    def get_residue_vector(self):
        """."""
        orb = self.measure_multiturn_orbit()

        slc_orb = slice(
            self.params.orbit_index_start, self.params.orbit_index_stop)
        slc_bpm = slice(
            self.params.bpm_index_start, self.params.bpm_index_stop)

        res = []
        if 'x' in self.params.orbit_planes.lower():
            ox_ = orb[slc_orb, :160][:, slc_bpm].copy()
            ox_ -= orb[:20, :160][:, slc_bpm].mean(axis=0)[None, :]
            res.append(ox_)
        if 'y' in self.params.orbit_planes.lower():
            oy_ = orb[slc_orb, 160:][:, slc_bpm].copy()
            oy_ -= orb[:20, 160:][:, slc_bpm].mean(axis=0)[None, :]
            res.append(oy_)
        return _np.hstack(res)

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        obj = []
        for i in range(nr_evals):
            obj.append(self.objective_function(pos))
            _log.info(f'{i+1:02d}/{nr_evals:02d}  --> obj. = {obj[-1]:.3f}')
        noise_level = _np.std(obj)
        _log.info(f'obj. = {_np.mean(obj):.3f} +- {noise_level:.3f}')
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_current_position_from_machine(self):
        """."""
        pos = []
        stt = self.params.wfm_index_start
        end = self.params.wfm_index_stop
        if 'm1ch' in self.params.optim_params:
            m1ch = self.devices['ps_m1ch']
            pos.append(m1ch.wfm[stt:end])
        if 'm2ch' in self.params.optim_params:
            m2ch = self.devices['ps_m2ch']
            pos.append(m2ch.wfm[stt:end])
        if 'm1cv' in self.params.optim_params:
            m1cv = self.devices['ps_m1cv']
            pos.append(m1cv.wfm[stt:end])
        if 'm2cv' in self.params.optim_params:
            m2cv = self.devices['ps_m2cv']
            pos.append(m2cv.wfm[stt:end])
        if 'dly' in self.params.optim_params:
            pos.append(self.devices['trigger'].delay)
        return _np.hstack(pos)

    def apply_position_to_machine(self, pos):
        """."""
        stt = self.params.wfm_index_start
        end = self.params.wfm_index_stop

        npts_wfm = end - stt
        pstt, pend = 0, npts_wfm
        if 'm1ch' in self.params.optim_params:
            m1ch = self.devices['ps_m1ch']
            wfm = m1ch.wfm
            wfm[stt:end] = pos[pstt:pend]
            m1ch.wfm = wfm
            pstt += npts_wfm
            pend += npts_wfm
        if 'm2ch' in self.params.optim_params:
            m2ch = self.devices['ps_m2ch']
            wfm = m2ch.wfm
            wfm[stt:end] = pos[pstt:pend]
            m2ch.wfm = wfm
            pstt += npts_wfm
            pend += npts_wfm
        if 'm1cv' in self.params.optim_params:
            m1cv = self.devices['ps_m1cv']
            wfm = m1cv.wfm
            wfm[stt:end] = pos[pstt:pend]
            m1cv.wfm = wfm
            pstt += npts_wfm
            pend += npts_wfm
        if 'm2cv' in self.params.optim_params:
            m2cv = self.devices['ps_m2cv']
            wfm = m2cv.wfm
            wfm[stt:end] = pos[pstt:pend]
            m2cv.wfm = wfm
            pstt += npts_wfm
            pend += npts_wfm
        if 'dly' in self.params.optim_params:
            self.devices['trigger'].delay = pos[-1]

    def measure_multiturn_orbit(self):
        """."""
        nr_avg = self.params.orbit_nrpulses
        sofb = self.devices['sofb']
        sofb.cmd_reset()
        sofb.nr_points = nr_avg
        _time.sleep(0.2)
        sofb.wait_buffer(timeout=nr_avg*0.5*2)
        orbx = sofb.mt_trajx.reshape(-1, 160)
        orby = sofb.mt_trajy.reshape(-1, 160)

        orb = _np.hstack([orbx, orby])

        if self.params.filter_with_orm:
            orb = self.filter_with_orm(
                orb, self.params.orm_2_filter,
                indices=self.params.orm_indices)
        if self.params.filter_with_wavelet:
            orb = self.filter_with_wavelet(
                orb, num_modes=self.params.wavelet_num_modes)
        return orb

    @staticmethod
    def load_orbit(fname, npts_mean=None):
        """Load orbit data file."""
        data = load_pickle(fname)
        data = data['data']
        if 'bpms_sampling_frequency' in data:
            fs = data['bpms_sampling_frequency']
        elif 'sofb_acqrate' in data:
            fs = 499667100/864
            rates = {0: 23*25, 1: 23, 2: 1, 3: 1/382, 4: 1/382}
            fs /= rates[data['sofb_acqrate']]

        orbx = data['orbx'].reshape(-1, 160)
        orby = data['orby'].reshape(-1, 160)
        orbx -= orbx[:npts_mean].mean(axis=0)[None, :]
        orby -= orby[:npts_mean].mean(axis=0)[None, :]

        orb = _np.hstack([orbx, orby])
        tim = _np.arange(orb.shape[0]) / fs
        freqs = _np.fft.rfftfreq(tim.size, d=1/fs)
        return orb, fs, tim, freqs

    @classmethod
    def filter_with_orm(cls, orb, orm, indices=None, ret_nullspace=False):
        """Filter matrix selecting the subspace reached by orbit correction.

        Note that we assume the orbit has m bpms (coluns) and n samples (rows).
        The response matrix has m bpms (rows) and k correctors (rows).

        Args:
            orb (np.ndarray, (n, m)): orbit to be filtered;
            orm (np.ndarray, (m, k)): orbit response matrix to be used for
                filtering;
            indices (slice|list|tuple|np.ndarray, optional): list of indices of
                correctors to use in reconstruction. All correctors will be
                used for orbit correction, but only the ones defined by these
                indices will be used in reconstruction. Defaults to None,
                which means all correctors will be used.
            ret_nullspace (bool, optional): If True, return the orbit that
                lives in the nullspace of the matrix, instead of the
                reacheable part. Defaults to False.

        Returns:
            orb (np.ndarray, (n, m)): the filtered orbit.

        """
        indices = indices or slice(None)
        orb_fil = (orm[:, indices] @ cls.correct_orbit(orb, orm, indices)).T
        if ret_nullspace:
            orb_fil -= orb
            orb_fil *= -1
        return orb_fil

    @classmethod
    def filter_with_svd(cls, mat, sval_indices=(0, 1, 2)):
        """Filter matrix selecting the desired singular values.

        Args:
            mat (np.ndarray, (n, m)): Matrix to be filtered.
            sval_indices (slice|list|tuple|np.ndarray, optional): indices of
                singular values to keep. Defaults to (0, 1, 2).

        Returns:
            mat_fil (np.ndarray, (n, m)): filtered matrix.
            u_fil (np.ndarray, (n, k)): left singular vectors. Where k is
                number of singular values kept;
            s_fil (np.ndarray, (k, ): singular values;
            v_fil (np.ndarray, (m, k): right singular vectors. Notice that
                we return v, not v transpose.

        """
        u, s, v = cls.do_svd(mat)

        u_fil = u[:, sval_indices]
        s_fil = s[sval_indices]
        v_fil = v[:, sval_indices]
        mat_fil = u_fil*s_fil @ v_fil.T
        return mat_fil, u_fil, s_fil, v_fil

    @staticmethod
    def filter_with_wavelet(mat, type_wave='sym5', num_modes=6, plot=False):
        """Filter matrix along lines using wavelet decomposition.

        Args:
            mat (np.ndarray, (n, m)): matrix to be filtered. Each column is
                treated independently one from another.
            type_wave (str, optional): Type of mother wavelet to use.
                Defaults to 'sym5'.
            num_modes (int, optional): Number of modes to keep. Defaults to 6.
            plot (bool, optional): Whether to plot results. Defaults to False.

        Returns:
            mat_fil (np.ndarray, (n, m)): filtered matrix.

        """
        coefs = _pywt.wavedec(mat, type_wave, mode='constant', axis=0)
        for i in range(num_modes, len(coefs)):
            coefs[i] *= 0
        mat_fil = _pywt.waverec(coefs, type_wave, mode='constant', axis=0)

        if plot:
            fig, ax = _mplt.subplots()
            ax.set_title(
                f'Wavelet filter with {type_wave:s} and {num_modes:d} modes.')
            offset = _np.arange(mat.shape[1])[None, :] * 0.05
            ax.plot(mat + offset)
            ax.plot(mat_fil + offset)
            fig.tight_layout()
            fig.show()
        return mat_fil

    @staticmethod
    def filter_with_dft(
            mat, fs=1, f_min=3.8e4, f_max=4.8e4, is_notch=True, plot=False):
        """Filter matrix along lines using DFT.

        Args:
            mat (np.ndarray, (n, m)): Matrix to be filtered.
            fs (int, optional): sampling frequency of the data. Defaults to 1.
            f_min (float, optional): Minimum frequency to keep/discard.
                Defaults to 3.8e4.
            f_max (float, optional): Maximum frequency to keep/discard.
                Defaults to 4.8e4.
            is_notch (bool, optional): Whether filter is a notch in the
                frequency range from [f_min, f_max] or a bandpass.
                Defaults to True, which means the frequency range will be
                filtered out from the data.
            plot (bool, optional): Whether to plot results. Defaults to False.

        Returns:
            mat_fil (np.ndarray, (n, m)):: _description_
        """
        dft = _np.fft.rfft(mat, axis=0)
        freqs = _np.fft.rfftfreq(mat.shape[0], d=1/fs)
        idx = (freqs <= f_min) | (freqs >= f_max)
        if is_notch:
            idx = ~idx
        dft[idx, :] = 0
        wrec = _np.fft.irfft(dft, axis=0)

        if plot:
            fig, ax = _mplt.subplots()
            ax.set_title('DFT Filtered vectors')
            offset = _np.arange(mat.shape[1])[None, :] * 0.05
            ax.plot(mat + offset)
            ax.plot(wrec + offset)
            fig.tight_layout()
            fig.show()
        return wrec

    @staticmethod
    def get_response_matrix(from_model=True):
        """Get response matrix containing the feedforward correctors and RF.

        Args:
            from_model (bool, optional): Whether to get matrix from the model
                or from ConfigDBClient ('ref_respmat'). Defaults to True.

        Returns:
            orm (np.ndarray, (320, 5)): Orbit response matrix.
            corrs_order (list(str), 4): Order of correctors in matrix.

        """
        corrs_order = ['M2-CH', 'M1-CH', 'M2-CV', 'M1-CV', 'RF']
        if from_model:
            mod = _si.create_accelerator()
            respm = _OrbRespmat(mod, acc='SI', dim='6d')

            fch = _np.array(respm.fam_data['FCH']['index']).ravel()
            fcv = _np.array(respm.fam_data['FCV']['index']).ravel()

            respm.ch_idx = fch[[0, -1]]
            respm.cv_idx = fcv[[0, -1]]

            return respm.get_respm(), corrs_order

        clt = _CDBClient(config_type='si_fastorbcorr_respm')
        orm = _np.array(clt.get_config_value('ref_respmat'))
        orm = orm[:, [0, 79, 80, 159, 160]]
        return orm, corrs_order

    @staticmethod
    def correct_orbit(orb, orm, indices):
        """Correct orbit with a given orbit response matrix.

        Note that we assume the orbit has m bpms (coluns) and n samples (rows).
        The response matrix has m bpms (rows) and k correctors (rows).

        Args:
            orb (np.ndarray, (n, m)): orbit to be filtered;
            orm (np.ndarray, (m, k)): orbit response matrix to be used for
                filtering;
            indices (slice|list|tuple|np.ndarray, optional): list of indices of
                correctors to return kicks. All correctors will be used for
                orbit correction, but only the ones defined by these indices
                will be return. Defaults to None, which means all correctors
                will be returned.

        Returns:
            kicks (np.ndarray, (l, m)): the kicks that correct the orbit.
                Where l is the size of the indices returned.

        """
        indices = indices or slice(None)
        kicks, *_ = _np.linalg.lstsq(orm, orb.T, rcond=None)
        return kicks[indices]

    @staticmethod
    def do_svd(mat):
        """Do the SVD decomposition of a matrix.

        Args:
            mat (np.ndarray, (n, m)): matrix to be decomposed.

        Returns:
            u (np.ndarray, (n, min(n, m))): left singular vectors;
            s (np.ndarray, (min(n, m), ): singular values;
            v (np.ndarray, (m, min(n, m)): right singular vectors. Notice that
                we return v, not v transpose.

        """
        u, s, v = _np.linalg.svd(mat, full_matrices=False)
        return u, s, v.T

    @staticmethod
    def plot_sing_values(u, s, v, fs, svals=None):
        """Plot singular values and vectors of matrix."""
        if svals is None:
            svals = 0, 1, 2

        fig = _mplt.figure(figsize=(9, 9))
        gs = _mplt.GridSpec(
            3, 2, top=0.98, bottom=0.08, left=0.13, right=0.98, wspace=0.02,
            hspace=0.3, width_ratios=[3, 1])
        a_s = fig.add_subplot(gs[0, :])
        a_u = fig.add_subplot(gs[1, 0])
        a_v = fig.add_subplot(gs[2, 0])
        h_u = fig.add_subplot(gs[1, 1], sharey=a_u)
        h_v = fig.add_subplot(gs[2, 1], sharey=a_v)

        a_s.set_ylabel(r'Sing. Values ($s_i$)')
        a_s.set_xlabel('#')
        a_s.set_yscale('log')
        a_s.set_xlim([-0.5, 20.5])

        a_u.set_xlabel('Time [ms]')
        a_u.set_ylabel(r'Time Sing. Vecs ($\vec{u_i}$)')

        a_v.set_xlabel('BPM Index')
        a_v.set_ylabel(r'Space Sing. Vecs ($\vec{v_i}$)')

        h_u.set_xlabel('Cnts.')
        h_v.set_xlabel('Cnts.')

        _mplt.setp(h_u.get_yticklabels(), visible=False)
        _mplt.setp(h_v.get_yticklabels(), visible=False)

        a_s.plot(s, 'ok')
        tim = _np.arange(u.shape[0]) / fs * 1e3
        for i, sing in enumerate(svals):
            cor = _mplt.cm.jet(i/len(svals))
            a_u.plot(tim, u[:, sing], 'o-', color=cor)
            a_v.plot(v[:, sing], 'o-', color=cor)

            h_u.hist(u[:, sing], bins=50, color=cor, orientation='horizontal')
            h_v.hist(v[:, sing], bins=50, color=cor, orientation='horizontal')

            a_s.plot(sing, s[sing], 'o', color=cor)

        fig.show()
        return fig, (a_u, a_s, a_v)

    def _create_devices(self):
        # Create Trigger connector:
        self.devices['trigger'] = Trigger('SI-01:TI-Mags-FFCorrs')

        # Create Power Supply connectors:
        self.devices['ps_m2ch'] = PowerSupply('SI-01M2:PS-FFCH')
        self.devices['ps_m1ch'] = PowerSupply('SI-01M1:PS-FFCH')
        self.devices['ps_m2cv'] = PowerSupply('SI-01M2:PS-FFCV')
        self.devices['ps_m1cv'] = PowerSupply('SI-01M1:PS-FFCV')

        # Create objects to interact with BPMs
        # self.devices['fambpms'] = FamBPMs()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)


class CorrectorModel:
    """."""

    def __init__(
            self, mag_name, mag_resistence=0.63905,
            mag_inductance=2.5813895e-3):
        """."""
        self.mag_name = mag_name
        self.mag_resistence = mag_resistence  # [Ohm]
        self.mag_inductance = mag_inductance  # [H]

        self.mag_cuttoff = 14e3  # [Hz]
        self.mag_delay = 0  # [s]

        energy = 2.9897  # [GeV]
        self.mag_normalizer = _Normalizer.create(
            mag_name, default_strengths_dipole=energy)

        self.mag_model_type = 'iir'  # ('iir', 'physics')
        self.mag_model_iir_num = _np.array([
            # 0, 0.00736772190177206, 0.0381989334609554, -0.116198636899350,
            # 0.0894474557595205, -0.0148332852289033, -0.00396157902847078, 0
            0, 0.00731190543657999, 0.0379093083187051, -0.115317981856320,
            0.0887697668787149, -0.0147209896017043, -0.00393155560376759, 0
            ])
        self.mag_model_iir_den = _np.array([
            # 1, -4.19172563035355, 7.21031243410994, -6.51988661626645,
            # 3.29299728245094, -0.908712830074173, 0.122036143737951,
            # -0.00500020467567861
            1, -4.19172156542397, 7.21020130413743, -6.51977136389536,
            3.29304307894445, -0.908764175487393, 0.122033349406558,
            -0.00500020467748778
            ])
        self.mag_model_duty_iir_num = _np.array([
            1.24071464455799, -5.03284626154369, 8.27859723722222,
            -7.05785644802524, 3.28450712690465, -0.789325241177209,
            0.0762102122722402
            ])
        self.mag_model_duty_iir_den = _np.array([
            1, -4.04974852456402, 6.67309977457816, -5.70003181060390,
            2.64892481109717, -0.635153655581859, 0.0629339095490494
            ])

        self.loop_period = 20e-6  # [s]
        self.loop_kp = (80 if mag_inductance > 4e-3 else 35)
        self.loop_kp *= (1 if self.loop_period > 15e-6 else 2)
        self.loop_ki = 0.2
        self.loop_dac_latency = 2  # [loop per]
        self.loop_adc_nrsamples = 2 if self.loop_period else 1

        self.chamber_conductivity = 1e6  # [S]
        self.chamber_radius = 12e-3  # [m]
        self.chamber_thickness = 300e-6  # [m]

    def __str__(self):
        """."""
        tmps = '{:20s} {:20s} {:10s}'.format
        tmpf = '{:20s} {:20.3f} {:10s}'.format
        tmpg = '{:20s} {:20.2g} {:10s}'.format
        tmpd = '{:20s} {:20d} {:10s}'.format
        stg = ''
        stg += tmps('mag_name', self.mag_name, '')
        stg += tmpf('mag_resistence', self.mag_resistence, '[Ohm]')
        stg += tmpg('mag_inductance', self.mag_inductance, '[H]')
        stg += tmpg('mag_cuttoff', self.mag_cuttoff, '[Hz]')
        stg += tmpg('mag_delay', self.mag_delay, '[s]')
        stg += tmps(
            'mag_model_type', self.mag_model_type, "('iir', 'physics')")
        stg += tmps('mag_model_iir_num', str(self.mag_model_iir_num))
        stg += tmps('mag_model_iir_den', str(self.mag_model_iir_den))
        stg += tmps('mag_model_duty_iir_num', str(self.mag_model_duty_iir_num))
        stg += tmps('mag_model_duty_iir_den', str(self.mag_model_duty_iir_den))
        stg += tmpg('loop_period', self.loop_period, '[s]')
        stg += tmpf('loop_kp', self.loop_kp, '[V/A]')
        stg += tmpf('loop_ki', self.loop_ki, '[V/A]')
        stg += tmpg('loop_dac_latency', self.loop_dac_latency, '[s]')
        stg += tmpd('loop_adc_nrsamples', self.loop_adc_nrsamples)
        stg += tmpg('chamber_conductivity', self.chamber_conductivity, '[S]')
        stg += tmpg('chamber_radius', self.chamber_radius, '[m]')
        stg += tmpg('chamber_thickness', self.chamber_thickness, '[m]')
        return stg

    def convert_current2kick(self, current):
        """."""
        return self.mag_normalizer.conv_current_2_strength(current)

    def convert_kick2current(self, kick):
        """."""
        return self.mag_normalizer.conv_strength_2_current(kick)

    def get_time_vector(self, size, T0=None):
        """."""
        if T0 is None:
            T0 = self.loop_period
        return _np.arange(size) * T0

    def get_frequency_vector(self, size, T0=None):
        """."""
        if T0 is None:
            T0 = self.loop_period
        return _np.fft.rfftfreq(size, d=T0)

    def simulate_system(
            self, tim, ref_curr, ref_volt=None, incl_magcham=False):
        """."""
        curr, volt, duty_cycle = self.evolve_current_in_time(
            ref_curr, volt=ref_volt)
        curr_upsmpl, volt_upsmpl = self.interpolate_time_evolution(
            tim, volt, curr)

        if incl_magcham:
            freq = self.get_frequency_vector(tim.size, T0=tim[1]-tim[0])
            curr_upsmpl = self.apply_corrector_transfer_function(
                freq, curr_upsmpl)
        return curr_upsmpl, volt_upsmpl, curr, volt, duty_cycle

    def apply_corrector_transfer_function(self, freqs, current):
        """."""
        tf_magnet = self.transfer_function_chamber(freqs)

        pole = self.mag_cuttoff * 2 * _np.pi
        tf_magnet *= self.transfer_function_lowpass(freqs, pole)
        tf_magnet *= self.transfer_function_delay(freqs, self.mag_delay)
        return self.apply_transfer_function_to_waveform(current, tf_magnet)

    def calc_voltage_from_current(self, current, T0=None):
        """."""
        R = self.mag_resistence
        L = self.mag_inductance
        if T0 is None:
            T0 = self.loop_period

        dcurr = _np.gradient(current, axis=0) / T0
        return current*R + L*dcurr

    def evolve_current_in_time(self, ref, volt=None):
        """Evolve system in time considering the model of the loop.

        Inputs:
            ref (numpy.ndarray, (N, )): current reference to be followed.
            volt (numpy.ndarray, (N, )): if not None simulates de behavior of
                the system in open-loop, under this voltage reference.
        """
        if self.mag_model_type.lower().startswith('iir') and volt is None:
            curr = _scysig.lfilter(
                self.mag_model_iir_num, self.mag_model_iir_den, ref)
            duty = _scysig.lfilter(
                self.mag_model_duty_iir_num, self.mag_model_duty_iir_den, ref)
            volt = self.calc_voltage_from_current(curr)
            return curr, volt, duty

        R = self.mag_resistence
        L = self.mag_inductance
        T0 = self.loop_period
        dac_latency = self.loop_dac_latency
        adc_nrsamples = self.loop_adc_nrsamples
        kp = self.loop_kp
        ki = self.loop_ki

        cur_i = _np.zeros(2)  # real current after time passage
        e_i = 0  # current error signal for the loop
        v_i = 0  # current voltage calculated as output of the loop
        # characteristic damping during one loop iteration:
        delta = R/L * T0 * _np.linspace(1/2, 1, 2)
        exp_delta = _np.exp(-delta)

        curr = []
        fixed_voltage = True
        if volt is None:
            volt = [0] * dac_latency
            fixed_voltage = False

        for i, r_i in enumerate(ref):
            curr.append(cur_i[-1])
            e_im1 = e_i
            e_i = r_i - cur_i[-adc_nrsamples:].mean()
            if not fixed_voltage:
                # Scale by R to adjust units from [A] to [V]:
                v_i += (ki*e_i + kp*(e_i-e_im1))
                volt.append(v_i)

            v_l = volt[i]/R
            cur_i = exp_delta * (cur_i[-1] - v_l)
            cur_i += v_l
        curr = _np.array(curr)
        volt = _np.array(volt)[:curr.size]
        duty = curr * 0
        return curr, volt, duty

    def interpolate_time_evolution(self, tim, digit_volt, digit_curr):
        """."""
        digit_tim = self.get_time_vector(digit_volt.size)

        if self.mag_model_type.lower().startswith('iir'):
            curr = _np.interp(tim, digit_tim, digit_curr)
            volt = _np.interp(tim, digit_tim, digit_volt)
            return curr, volt

        R = self.mag_resistence
        L = self.mag_inductance

        timt = _np.unique(_np.sort(_np.r_[digit_tim, tim]))

        voltt = _scyinterp.interp1d(
            digit_tim, digit_volt, axis=0, kind='previous',
            bounds_error=False, fill_value=(0, digit_volt[-1]))(timt)

        currt = [digit_curr[0]]
        for i in range(timt.size-1):
            vol = voltt[i]/R
            cur_i = currt[-1] - vol
            cur_i *= _np.exp(-R/L*(timt[i+1]-timt[i]))
            cur_i += vol
            currt.append(cur_i)
        currt = _np.array(currt)

        curr = _np.interp(tim, timt, currt)
        volt = _np.interp(tim, timt, voltt)
        return curr, volt

    def optimize_reference(
            self, init_ref, time_goal, goal, slc_knob=None, slc_obj=None,
            incl_magcham=True, consider_duty=False):
        """."""
        def func(x):
            ref = init_ref.copy()
            ref[slc_knob] = x
            resp_model, *_, duty = self.simulate_system(
                time_goal, ref, incl_magcham=incl_magcham)
            res1 = goal[slc_obj] - resp_model[slc_obj]
            if not consider_duty:
                return res1
            res2 = _np.tanh((_np.abs(duty)-0.95)/0.05 * 5)
            res2 += 1
            res2 *= res1.size / res2.size * goal[slc_obj].ptp() * 10
            return _np.hstack([res1, res2])

        slc_knob = slice(None) if slc_knob is None else slc_knob
        slc_obj = slice(None) if slc_obj is None else slc_obj
        x0 = init_ref[slc_knob]
        res = _scyopt.least_squares(func, x0, method='lm')

        ref = init_ref.copy()
        ref[slc_knob] = res.x
        return ref

    def get_params(self, params):
        """."""
        return [getattr(self, par) for par in params]

    def set_params(self, params, values):
        """."""
        for par, val in zip(params, values):
            setattr(self, par, val)

    def fit_system_to_step_response(
            self, time_response, response2fit, reference,
            init_time=0, final_time=_np.inf,
            params2vary=None, incl_magcham=False):
        """."""
        def objective(x):
            self.set_params(params2vary, x)
            resp_model, *_ = self.simulate_system(
                time_response, reference, incl_magcham=incl_magcham)

            idx = time_response <= final_time
            idx &= time_response >= init_time
            return response2fit[idx] - resp_model[idx]

        if params2vary is None:
            params2vary = ['loop_kp', 'loop_ki', 'mag_delay']

        x0 = _np.array(self.get_params(params2vary))

        res = _scyopt.least_squares(objective, x0, bounds=(0, _np.inf))
        return res['x'], x0, params2vary

    def transfer_function_magnet(self, freqs):
        """."""
        pole = self.mag_cuttoff * 2 * _np.pi
        tf_mag = self.transfer_function_lowpass(freqs, pole)
        tf_mag *= self.transfer_function_delay(freqs, self.mag_delay)
        return tf_mag

    def transfer_function_chamber(
            self, freqs, maxpoles=100, error_tolerance=1e-5, plot=False):
        """."""
        chamb_cond = self.chamber_conductivity
        chamb_radius = self.chamber_radius
        chamb_thick = self.chamber_thickness

        if plot:
            fig, (ax1, ax2) = _mplt.subplots(2, 1, sharex=True, figsize=(9, 6))
            ax1.set_ylabel('Magnitude')
            ax2.set_ylabel('Phase [°]')
            ax2.set_xlabel('Frequency [Hz]')
            ax1.set_xscale('log')
        mu0 = 4e-7*_np.pi  # [H/m]

        s = 2j * _np.pi * freqs
        tau = 0.5 * mu0 * chamb_cond
        tau *= chamb_radius * chamb_thick
        pole0 = 1/tau

        transf = _np.ones(freqs.shape, dtype=complex)
        for n in range(maxpoles+1):
            if n == 0:
                polen = pole0
            else:
                polen = 0.5 * pole0 * n*n * _np.pi*_np.pi
                polen *= chamb_radius/chamb_thick
            tr_old = transf.copy()
            transf *= polen / (polen + s)
            if plot:
                phase = _np.unwrap(_np.angle(transf))/_np.pi*180
                ax1.plot(freqs, _np.abs(transf))
                ax2.plot(freqs, phase)
            res = _np.abs(transf - tr_old)
            if res.max() < error_tolerance:
                break

        if plot:
            ax1.set_title(
                f'thickness = {chamb_thick:.1f} mm,  '
                f'radius = {chamb_radius:.1f} mm,  '
                f'n° poles = {n:d}')
            fig.tight_layout()

        return transf

    def digitize_waveform(self, wfm, T0_wfm, rel_dly=0.0):
        """."""
        tim = _np.arange(wfm.shape[0]) * T0_wfm
        interp = _scyinterp.interp1d(tim, wfm, axis=0)
        rate = 1 / self.loop_period

        digit_tim = _np.arange(int(tim[-1]*rate - rel_dly), dtype=float)
        digit_tim += rel_dly
        digit_tim /= rate
        digit_wfm = interp(digit_tim)

        # Oversample digitized waveform:
        interp2 = _scyinterp.interp1d(
            digit_tim, digit_wfm, axis=0, kind='previous',
            bounds_error=False, fill_value=(0, 0))
        return digit_tim, digit_wfm, interp2(tim)

    @staticmethod
    def transfer_function_delay(freqs, delay):
        """."""
        s = 2j * _np.pi * freqs
        return _np.exp(-s*delay)

    @staticmethod
    def transfer_function_lowpass(freqs, pole):
        """."""
        s = 2j * _np.pi * freqs
        return pole / (s + pole)

    @staticmethod
    def apply_transfer_function_to_waveform(wfm, transfer):
        """."""
        if wfm.ndim == 2:
            transfer = transfer[:, None]
        return _np.fft.irfft(_np.fft.rfft(wfm, axis=0) * transfer, axis=0)
