"""."""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si
from mathphys.functions import load
from siriuspy.devices import SOFB, IVU
from siriuspy.search import BPMSearch

from apsuite.commisslib.meas_bpms_signals import (
    AcqBPMsSignals as _BaseAcq,
    AcqBPMsSignalsParams as _BaseParams,
)
from apsuite.utils import ThreadedMeasBaseClass as _BaseThreaded


class ImpedanceIVUMeasParams(_BaseParams):
    """."""

    ADC_NSAMPLES_PER_TURN = 382
    HARM_NUM = 864

    def __init__(self):
        """."""
        super().__init__()
        self.acq_strategy = 'all'  # 'all' or 'odd/even'
        self.num_acquisitions = 10
        self.save_raw_data = False
        self.num_buckets_to_process = 2
        self._nrturns = 0
        self.nrturns = 500
        self.bucket_hi_charge = 1
        self.bucket_lo_charge = 530
        self.acq_rate = 'ADCSwp'
        self.signals2acq = 'ABCD'
        self.timeout = 10
        self.event_mode = 'Injection'
        self.timing_event = 'Study'

    def __str__(self):
        """."""
        stg = 'AcqBPMsSignalsParams:\n'
        stg += ''.join([f'    {l}\n' for l in super().__str__().splitlines()])
        stg += '\nImpedanceIVUMeasParams:\n'
        stg += f'    acq_strategy = {self.acq_strategy}   '
        stg += "('all', 'odd/even')\n"
        stg += f'    num_acquisitions = {self.num_acquisitions}\n'
        stg += f'    save_raw_data = {self.save_raw_data}\n'
        stg += f'    num_buckets_to_process = {self.num_buckets_to_process}\n'
        stg += f'    nrturns = {self.nrturns}\n'
        stg += '  The properties below are used to find out the position\n'
        stg += '  of the second bunch, relative to the first, during the'
        stg += '  data analysis:\n'
        stg += f'    bucket_hi_charge = {self.bucket_hi_charge}\n'
        stg += f'    bucket_lo_charge = {self.bucket_lo_charge}\n'
        return stg

    @property
    def nrturns(self):
        """."""
        return self._nrturns

    @nrturns.setter
    def nrturns(self, val):
        self._nrturns = int(val)
        self.nrpoints_after = self.nrturns * self.ADC_NSAMPLES_PER_TURN
        self.nrpoints_before = 0


class ImpedanceIVUMeas(_BaseThreaded, _BaseAcq):
    """."""

    # position, in ADC samples, to put the max. amplitude of the first bunch:
    BUN1_OFFSET = 50
    # Window, in ADC samples, where RMS to estimate bunch ampl. is calculated.
    # These values were determined during machine studies, by looking at the
    # typical antenna waveform induced by a single bunch.
    WINDOW_RMS = (-10, 20)

    def __init__(self, isonline=True):
        """."""
        _BaseThreaded.__init__(self, isonline=isonline, target=self._measure)
        _BaseAcq.__init__(self, isonline=self.isonline)
        self.params = ImpedanceIVUMeasParams()

    def create_devices(self, bpmnames=None):
        """."""
        _BaseAcq.create_devices(self, bpmnames=bpmnames)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['ivu18_08'] = IVU(IVU.DEVICES.IVU18_08SB)
        self.devices['ivu18_14'] = IVU(IVU.DEVICES.IVU18_14SB)

    def get_data(self):
        """."""
        data = super().get_data()
        data['bpm_names'] = [b.devname for b in self.devices['fambpms'].bpms]
        bns = self.devices['fambpms'].bpm_names
        data['bpm_indcs'] = np.array([bns.index(b) for b in data['bpm_names']])
        data['sofb_refx'] = self.devices['sofb'].refx
        data['sofb_refy'] = self.devices['sofb'].refy
        data['sofb_orbx'] = self.devices['sofb'].orbx
        data['sofb_orby'] = self.devices['sofb'].orby
        data['sofb_bpmxenbl'] = self.devices['sofb'].bpmxenbl
        data['sofb_bpmyenbl'] = self.devices['sofb'].bpmyenbl
        data['sofb_nr_points'] = self.devices['sofb'].nr_points
        data['sofb_kickch'] = self.devices['sofb'].kickch
        data['sofb_kickcv'] = self.devices['sofb'].kickcv
        data['sofb_kickrf'] = self.devices['sofb'].kickrf
        data['ivu18_08_gap'] = self.devices['ivu18_08'].gap
        data['ivu18_14_gap'] = self.devices['ivu18_14'].gap
        return data

    def load_and_apply(self, fname: str):
        """."""
        return _BaseThreaded.load_and_apply(self, fname)

    load_and_apply.__doc__ = _BaseThreaded.load_and_apply.__doc__

    def _measure(self):
        data = []
        fambpms = self.devices['fambpms']
        bpms = fambpms.bpms
        grp_slcs = [slice(None, None)]
        if self.params.acq_strategy.startswith('odd'):
            print(
                "Acquisition strategy 'odd/even' identified. "
                'Breaking BPMs in two groups.'
            )
            grp_slcs = [slice(0, None, 2), slice(1, None, 2)]
        try:
            for i in range(self.params.num_acquisitions):
                print(
                    f'Acquisition {i + 1:02d}/{self.params.num_acquisitions:02d}'
                )
                dt = []
                if self._stopevt.is_set():
                    break
                for grp, slc in enumerate(grp_slcs):
                    if len(grp_slcs) > 1:
                        print(f'    acquiring BPMs group {grp}...')
                    fambpms.bpms = bpms[slc]
                    if self._stopevt.is_set():
                        break
                    self.acquire_data()
                    self.data['acq_group'] = grp
                    dt.append(self.data)
                else:
                    data.extend(dt)
        except Exception:
            fambpms.bpms = bpms
            print('Problem with acquisition. Interrupting!')
            return
        fambpms.bpms = bpms
        print('Acquisitions ended. Processing data...')
        self.data = data
        self.process_data()
        self._filter_data_to_save()
        print('Finished!')

    def process_data(self, idcs_to_discard=None, return_all=False):
        """."""
        idcs_to_discard = idcs_to_discard or []
        data = self.data
        if isinstance(data, dict):
            data = [data]
        for i, dt in enumerate(data):
            if i in idcs_to_discard:
                continue
            dic = self._proc_single_data(dt, return_all=return_all)
            dt.update(dic)

    def calc_delta_orbit_2_bunches(self):
        """Calculate orbit variation between the two stored bunches.

        Returns:
            dorb: orbit deviation between the two stored bunches.
            orb1: orbit of the first stored bunch.
            orb2: orbit of the second stored bunch.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.
        """
        if not self.data:
            raise RuntimeError('Get data First.')
        orb1, orb2 = [], []
        for dt in self.data:
            if 'b1_posx' not in dt or 'b2_posx' not in dt:
                raise RuntimeError(
                    'Missing bunch positions in data. Process data first.'
                )
            orb1.append(np.vstack([dt['b1_posx'], dt['b1_posy']]))
            orb2.append(np.vstack([dt['b2_posx'], dt['b2_posy']]))
        orb1 = np.array(orb1)
        orb2 = np.array(orb2)
        dorb = orb1 - orb2
        return dorb, orb1, orb2

    def calc_current_2_bunches(self):
        """Calculate current of the two stored bunches.

        Returns:
            curr1: current of the first stored bunch.
            curr2: current of the second stored bunch.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.

        """
        if not self.data:
            raise RuntimeError('Get data First.')
        curr1, curr2 = [], []
        for dt in self.data:
            if 'b1_curr' not in dt or 'b2_curr' not in dt:
                raise RuntimeError(
                    'Missing bunch currents in data. Process data first.'
                )
            curr1.append(dt['b1_curr'])
            curr2.append(dt['b2_curr'])
        curr1 = np.array(curr1)
        curr2 = np.array(curr2)
        return curr1, curr2

    def calc_sum_signal_2_bunches(self):
        """Calculate the sum signal of the two stored bunches.

        Returns:
            sumt: sum signal of the two bunches.
            sum1: sum signal of the first stored bunch.
            sum2: sum signal of the second stored bunch.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.

        """
        if not self.data:
            raise RuntimeError('Get data First.')
        sumt, sum1, sum2 = [], [], []
        for dt in self.data:
            if 'b1_sum' not in dt or 'b2_sum' not in dt:
                raise RuntimeError(
                    'Missing sum signals in data. Process data first.'
                )
            sumt.append(dt['bt_sum'])
            sum1.append(dt['b1_sum'])
            sum2.append(dt['b2_sum'])
        sumt = np.array(sumt)
        sum1 = np.array(sum1)
        sum2 = np.array(sum2)
        return sumt, sum1, sum2

    def calc_sofb_orbit(self, isref=False):
        """Calculate the SOFB orbit.

        Returns:
            orb: The SOFB orbit.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.
        """
        if not self.data:
            raise RuntimeError('Get data First.')
        orb = []
        prop = 'sofb_' + ('ref' if isref else 'orb')
        for dt in self.data:
            orb.append(np.hstack([dt[prop + 'x'], dt[prop + 'y']]))
        return np.vstack(orb).T

    def _filter_data_to_save(self):
        if self.params.save_raw_data:
            return
        for dt in self.data:
            for ant in 'abcd':
                dt.pop('ampl' + ant)

    def _proc_single_data(self, data, return_all=False):
        of1 = self.BUN1_OFFSET
        winn, winp = self.WINDOW_RMS
        nbuc2proc = self.params.num_buckets_to_process
        bhigh = self.params.bucket_hi_charge
        blow = self.params.bucket_lo_charge
        nsamp_pturn = ImpedanceIVUMeasParams.ADC_NSAMPLES_PER_TURN
        hnum = ImpedanceIVUMeasParams.HARM_NUM

        ant_raw = np.array([data['ampl' + ant] for ant in 'abcd'])
        # [4, 382 * N, 160] --> [4, 160, 382 * N]
        ant_raw = ant_raw.swapaxes(1, 2)
        curr = data['stored_current']

        ant_abs = np.abs(ant_raw)
        ant_amax = ant_abs[..., :nsamp_pturn].argmax(axis=-1)

        nsamp2keep = ant_raw.shape[-1]
        nturn2keep = nsamp2keep // nsamp_pturn
        idx = np.arange(nsamp2keep)
        old_idx = (idx - of1 + ant_amax[..., None]) % nsamp2keep

        ant_raw2 = np.take_along_axis(ant_raw, old_idx, axis=-1)
        ant_raw2 = ant_raw2.reshape(ant_raw2.shape[:2] + (nturn2keep, -1))

        dic = {}
        if return_all:
            dic['ant_raw'] = ant_raw
            dic['ant_amax'] = ant_amax
            dic['ant_raw2'] = ant_raw2

        of2 = ((blow - bhigh) // hnum) * nsamp_pturn + of1
        slcs = [slice(of1 + winn, of1 + winp), slice(of2 + winn, of2 + winp)]
        for i in range(nbuc2proc):
            b_sigs = ant_raw2[..., slcs[i]].std(axis=-1)
            b_posx, b_posy = ImpedanceIVUMeas.calc_positions_from_amplitudes(
                b_sigs, is_adcswap_rate=True
            )
            b_sum = b_sigs.sum(axis=0)

            dic[f'b{i + 1}_posx'] = b_posx
            dic[f'b{i + 1}_posy'] = b_posy
            dic[f'b{i + 1}_sum'] = b_sum
            dic[f'b{i + 1}_sigs'] = b_sigs

        bt_sum = sum([dic[f'b{i + 1}_sum'] for i in range(nbuc2proc)])
        dic['bt_sum'] = bt_sum
        for i in range(nbuc2proc):
            dic[f'b{i + 1}_curr'] = dic[f'b{i + 1}_sum'] * curr / bt_sum

        return dic


class ImpedanceIVUAnalysis:
    """Analyze IVU transverse impedance measurements."""

    _SUPPORTED_IVUS = ('IVU18_SI08', 'IVU18_SI14')

    def __init__(self, data=None, beam_voltage=3.0e9):
        """Initialize IVU impedance analysis.

        Args:
            data (list, optional): Measurement configurations to analyze.
            beam_voltage (float): Beam energy divided by charge [V].
        """
        self.beam_voltage = float(beam_voltage)

        # Measurement and model data
        self.data = None
        self.id_name = None
        self.resp_mat = None
        self.spos_bpms = None

        # Initialize quantities calculated in the analysis
        self._reset_analysis_results()

        if data is not None:
            self.set_data(data)

    def _reset_analysis_results(self):
        """Clear quantities calculated during the analysis."""
        # Data preparation
        self.config_table = None
        self.bump_info = None

        # Orbit analysis
        self.delta_orbit = {'x': None, 'y': None}

        self.delta_orbit_stats = {'x': None, 'y': None}

        # IVU response matrix
        self.M_ivu = {'x': None, 'y': None}

        # Fits
        self.theta_fits = {'x': None, 'y': None}

        self.kperp_fits = {'x': {}, 'y': {}}

    # Accelerator-model utilities

    @staticmethod
    def get_spos_bpms():
        """Get the longitudinal positions of the BPMs in the Sirius model.

        Returns:
            spos_bpms (numpy.ndarray): BPM longitudinal positions [m].
        """
        model = si.create_accelerator()

        spos = pyaccel.lattice.find_spos(model, indices='open')
        bpm_indices = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')

        spos_bpms = np.asarray(spos[bpm_indices], dtype=float)

        return spos_bpms

    @staticmethod
    def _set_ivu_kick(model, ivu_indices, vkick):
        """Set vertical kick in the selected IVU elements.

        Args:
            model: Accelerator model.
            ivu_indices (sequence of int): Indices of the IVU elements.
            vkick (float): Vertical kick applied to each IVU element [rad].
        """
        for idx in ivu_indices:
            model[idx].pass_method = 'str_mpole_symplectic4_pass'
            model[idx].vkick_polynom = vkick

    @staticmethod
    def _calc_ivu_respmat(delta_vkick=5e-6, id_name='IVU18_SI08'):
        """Calculate the orbit response column for the selected IVU.

        The response column is the derivative of the orbits with respect
        to the IVU vertical kick. It is estimated at zero kick using the
        centered finite difference:

        [orbit(+delta_vkick/2) - orbit(-delta_vkick/2)] / delta_vkick.

        Args:
            delta_vkick (float): Difference between the positive and
                negative vertical kicks [rad].
            id_name (str): IVU name, either "IVU18_SI08" or
                "IVU18_SI14".

        Returns:
            respmat (numpy.ndarray): IVU response column with shape
                (2*n_bpms,) [m/rad]. The first half contains the
                horizontal response and the second half contains the
                vertical response.
            info (dict): Auxiliary model and response-calculation
                information.

        Raises:
            ValueError: If delta_vkick is invalid, id_name is not
                recognized, or the IVU elements cannot be identified
                in the model.
        """
        delta_vkick = float(delta_vkick)

        if not np.isfinite(delta_vkick) or delta_vkick == 0.0:
            raise ValueError('delta_vkick must be finite and nonzero.')

        model = si.create_accelerator()

        spos = pyaccel.lattice.find_spos(model, indices='open')
        bpm_indices = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')
        ivu_indices_all = pyaccel.lattice.find_indices(
            model, 'fam_name', 'IVU18'
        )

        if len(ivu_indices_all) != 4:
            raise ValueError(
                'Expected four IVU18 elements in the accelerator model, '
                f'but found {len(ivu_indices_all)}.'
            )

        id_map = {
            'IVU18_SI08': ivu_indices_all[:2],
            'IVU18_SI14': ivu_indices_all[2:],
        }

        try:
            ivu_indices = id_map[id_name]
        except KeyError:
            raise ValueError(
                f"Unknown IVU '{id_name}'. "
                'Valid options are '
                f'{list(ImpedanceIVUAnalysis._SUPPORTED_IVUS)}.'
            ) from None

        try:
            ImpedanceIVUAnalysis._set_ivu_kick(
                model, ivu_indices, +delta_vkick / 2
            )
            cod_pos = pyaccel.tracking.find_orbit4(model, indices='open')

            ImpedanceIVUAnalysis._set_ivu_kick(
                model, ivu_indices, -delta_vkick / 2
            )
            cod_neg = pyaccel.tracking.find_orbit4(model, indices='open')

        finally:
            ImpedanceIVUAnalysis._set_ivu_kick(model, ivu_indices, 0.0)

        n_bpms = len(bpm_indices)

        respmat = np.zeros(2 * n_bpms, dtype=float)

        orbit_difference = cod_pos[:, bpm_indices] - cod_neg[:, bpm_indices]

        respmat[:n_bpms] = orbit_difference[0] / delta_vkick
        respmat[n_bpms:] = orbit_difference[2] / delta_vkick

        info = {
            'model': model,
            'spos': spos,
            'spos_bpms': np.asarray(spos[bpm_indices], dtype=float),
            'bpm_indices': np.asarray(bpm_indices, dtype=int),
            'id_name': id_name,
            'ivu_indices': np.asarray(ivu_indices, dtype=int),
            'ivu_indices_all': np.asarray(ivu_indices_all, dtype=int),
            'delta_vkick': delta_vkick,
        }

        return respmat, info

    def set_data(self, data):
        """Set measurement data to be analyzed.

        Args:
            data (list or dict): List of measurement configurations,
                or one measurement configuration.

        Raises:
            TypeError: If a configuration or acquisition has an invalid type.
            ValueError: If the data structure is empty or incomplete.
        """
        if isinstance(data, dict):  # Case where a single meas. is provided
            data = [data]

        if not isinstance(data, (list, tuple)):
            raise TypeError(
                'data must be a measurement configuration or a list or tuple '
                'of measurement configurations.'
            )

        if len(data) == 0:
            raise ValueError('data must contain at least one measurement.')

        for i, data_i in enumerate(data):
            if not isinstance(data_i, dict):
                raise TypeError(f'Configuration {i} must be a dictionary.')

            if 'data' not in data_i or 'params' not in data_i:
                raise ValueError(
                    f"Configuration {i} must contain 'data' and 'params' keys."
                )

            if not isinstance(data_i['params'], dict):
                raise TypeError(
                    f"Configuration {i}: 'params' must be a dictionary."
                )

            data_meas = data_i['data']

            if not isinstance(data_meas, (list, tuple)):
                raise TypeError(
                    f"Configuration {i}: 'data' must be a list or tuple "
                    'of acquisitions.'
                )

            if len(data_meas) == 0:
                raise ValueError(
                    f'Configuration {i} must contain at least one acquisition.'
                )

            for acq, data_acq in enumerate(data_meas):
                if not isinstance(data_acq, dict):
                    raise TypeError(
                        f'Configuration {i}, acquisition {acq} '
                        'must be a dictionary.'
                    )

        self.data = list(data)
        self._reset_analysis_results()

    def load_data(self, file_paths, y0=None):
        """Load measurement data from files.

        Args:
            file_paths (str or sequence of str): Measurement file path(s).
            y0 (float or sequence of float, optional): Vertical bump value
                for each measurement [mm].

        Raises:
            TypeError: If file_paths has an unsupported type.
            ValueError: If no paths are provided or file_paths and y0 have
                incompatible lengths.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not isinstance(file_paths, (list, tuple)):
            raise TypeError(
                'file_paths must be a path or a list or tuple of paths.'
            )

        if len(file_paths) == 0:
            raise ValueError('file_paths must contain at least one file path.')

        # Prepare y0 values.
        if y0 is None:  # y0 values to be obtained from the measurement files
            y0_values = [None] * len(file_paths)

        elif np.isscalar(y0):
            if len(file_paths) != 1:
                raise ValueError(
                    'A single y0 value can only be used with one file.'
                )
            y0_values = [float(y0)]

        else:
            y0_values = list(y0)

            if len(y0_values) != len(file_paths):
                raise ValueError(
                    'file_paths and y0 must have the same length.'
                )

        data = []

        for file_path, y0_i in zip(file_paths, y0_values):
            data_i = load(file_path)

            if y0_i is not None:
                data_i['params']['y0'] = float(y0_i)

            data.append(data_i)

        self.set_data(data)

    def set_spos_bpms(self):
        """Set BPM longitudinal positions from the Sirius model.

        Returns:
            spos_bpms (numpy.ndarray): BPM longitudinal positions [m].

        Raises:
            ValueError: If no BPM positions are found or the measurement BPM
                indices are incompatible with the model.
        """
        spos_bpms = self.get_spos_bpms()

        if len(spos_bpms) == 0:
            raise ValueError(
                'No BPM longitudinal positions were found in the model.'
            )

        if self.data is not None:
            max_bpm_idx = max(
                np.max(data_acq['bpm_indcs'])
                for data_i in self.data
                for data_acq in data_i['data']
            )

            if max_bpm_idx >= len(spos_bpms):
                raise ValueError(
                    'Measurement data contain a global BPM index unavailable '
                    'in the accelerator model.'
                )

        self.spos_bpms = spos_bpms

        return spos_bpms

    def prepare_data(self, gap_key='ivu18_08_gap', infer_y0=True):
        """Prepare loaded measurement data for analysis.

        Args:
            gap_key (str): Dictionary key containing the desired IVU gap.
            infer_y0 (bool): Whether to calculate the vertical bump values
                from the SOFB orbit data.

        Raises:
            RuntimeError: If no measurement data have been loaded or required
                acquisition information is unavailable.
            ValueError: If acquisition groups or measurement dimensions are
                inconsistent.
        """
        if self.data is None:
            raise RuntimeError('Load or set data first.')

        # Any previous analysis results are no longer valid.
        self._reset_analysis_results()

        self._calculate_currents()
        self._add_charge_information()

        if infer_y0:
            self._calculate_bump_values()

        self._join_odd_even_acquisitions()

        self.config_table = self._make_config_table(gap_key=gap_key)

    def _calculate_currents(self):
        """Calculate bunch currents for all acquisitions.

        The bunch sum-signal arrays are averaged over BPMs and turns. The
        resulting mean values are used to estimate one current value for
        each bunch and acquisition.

        Raises:
            RuntimeError: If required acquisition data are unavailable.
            ValueError: If the mean total sum signal is zero.
        """
        required_keys = {'b1_sum', 'b2_sum', 'stored_current'}

        for i, data_i in enumerate(self.data):
            for acq, data_acq in enumerate(data_i['data']):
                missing_keys = required_keys.difference(data_acq)

                if missing_keys:
                    missing = ', '.join(sorted(missing_keys))
                    raise RuntimeError(
                        f'Configuration {i}, acquisition {acq}: '
                        f'missing data required to calculate currents: '
                        f'{missing}.'
                    )

                b1_sum_mean = np.mean(data_acq['b1_sum'])
                b2_sum_mean = np.mean(data_acq['b2_sum'])
                bt_sum_mean = b1_sum_mean + b2_sum_mean

                if np.isclose(bt_sum_mean, 0.0):
                    raise ValueError(
                        f'Configuration {i}, acquisition {acq}: '
                        'the mean total sum signal is zero.'
                    )

                stored_current = data_acq['stored_current']

                b1_curr = b1_sum_mean * stored_current / bt_sum_mean

                b2_curr = b2_sum_mean * stored_current / bt_sum_mean

                data_acq['b1_sum_mean'] = float(b1_sum_mean)
                data_acq['b2_sum_mean'] = float(b2_sum_mean)
                data_acq['bt_sum_mean'] = float(bt_sum_mean)

                data_acq['b1_curr'] = float(b1_curr)
                data_acq['b2_curr'] = float(b2_curr)
                data_acq['delta_curr'] = b1_curr - b2_curr

    def _add_charge_information(self):
        """Calculate bunch charges for all acquisitions.

        Raises:
            RuntimeError: If bunch-current information is unavailable.
        """
        light_speed = 299_792_458.0  # [m/s]
        circumference = 518.3899  # [m] - Sirius circumference

        rev_frequency = light_speed / circumference  # [Hz]

        required_keys = {'b1_curr', 'b2_curr'}

        for i, data_i in enumerate(self.data):
            data_meas = data_i['data']

            for acq, data_acq in enumerate(data_meas):
                missing_keys = required_keys.difference(data_acq)

                if missing_keys:
                    missing = ', '.join(sorted(missing_keys))
                    raise RuntimeError(
                        f'Configuration {i}, acquisition {acq}: '
                        'missing data required to calculate charges: '
                        f'{missing}.'
                    )

                b1_charge = 1e6 * data_acq['b1_curr'] / rev_frequency  # [nC]

                b2_charge = 1e6 * data_acq['b2_curr'] / rev_frequency  # [nC]

                data_acq['b1_charge'] = b1_charge
                data_acq['b2_charge'] = b2_charge
                data_acq['delta_charge'] = b1_charge - b2_charge

            delta_charge_mean = np.mean([
                data_acq['delta_charge'] for data_acq in data_meas
            ])  # Configuration mean delta charge [nC]

            for data_acq in data_meas:
                data_acq['delta_charge_mean'] = float(delta_charge_mean)

    def _calculate_bump_values(self):
        """Calculate vertical bump values from the SOFB orbit data.

        The two BPMs registering the bumps are identified from the SOFB
        reference orbits across configurations. Subsequently, the measured
        SOFB orbits per configuration at these BPMs are averaged over all
        acquisitions.

        The configuration whose mean bump orbit is closest to zero is then
        used as the reference for all bump calculations.

        Returns:
            y0 (numpy.ndarray): Vertical bump value for each measurement
                configuration [mm].
        """
        sofb_refs = np.asarray([
            data_i['data'][0]['sofb_refy'] for data_i in self.data
        ])

        ref_diff = sofb_refs - sofb_refs[0]

        ref_diff_mean = np.mean(np.abs(ref_diff), axis=0)

        bump_bpm_indcs = np.argsort(ref_diff_mean)[-2:]

        # Keep the BPM indices in their global order.
        bump_bpm_indcs = np.sort(bump_bpm_indcs)

        sofb_orbits = []

        for data_i in self.data:
            orbit_i = np.mean(
                [
                    data_acq['sofb_orby'][bump_bpm_indcs]
                    for data_acq in data_i['data']
                ],
                axis=0,
            )

            sofb_orbits.append(orbit_i)

        sofb_orbits = np.asarray(sofb_orbits)

        reference_cfg = np.argmin(np.abs(np.mean(sofb_orbits, axis=1)))

        bump_orbits = sofb_orbits - sofb_orbits[reference_cfg]

        # SOFB orbit values are in micrometers; y0 is stored in millimeters.
        y0 = np.mean(bump_orbits, axis=1) * 1e-3

        # Before assigning the inferred values, check if any y0 values already
        # exist in the data.

        has_existing_y0 = any('y0' in data_i['params'] for data_i in self.data)

        if has_existing_y0:
            warnings.warn(
                'Existing y0 values will be replaced by values inferred '
                'from the SOFB orbit data.',
                stacklevel=2,
            )

        for y0_i, data_i in zip(y0, self.data):
            data_i['params']['y0'] = float(y0_i)

        self.bump_info = {
            'bpm_indcs': bump_bpm_indcs,
            'reference_cfg': int(reference_cfg),
            'sofb_refs': sofb_refs,
            'sofb_orbits': sofb_orbits,
            'bump_orbits': bump_orbits,
            'y0': y0,
        }

        return y0

    def _join_bpm_groups(
        self, array1, array2, bpm_indcs1, bpm_indcs2, return_list=False
    ):
        """Join two BPM data groups using their global BPM indices.

        Args:
            array1 (array-like): Data from the first BPM group. Its first
                dimension must correspond to BPM indices.
            array2 (array-like): Data from the second BPM group. Its first
                dimension must correspond to BPM indices.
            bpm_indcs1 (array-like): Global BPM indices for the first group.
            bpm_indcs2 (array-like): Global BPM indices for the second group.
            return_list (bool): Whether to return a list instead of an array.

        Returns:
            joined_array (numpy.ndarray or list): Data from both BPM groups,
            ordered by global BPM index.

        Raises:
            ValueError: If the array dimensions do not match their BPM indices
                or the two groups contain repeated global indices.
        """
        array1 = np.asarray(array1)
        array2 = np.asarray(array2)
        bpm_indcs1 = np.asarray(bpm_indcs1, dtype=int)
        bpm_indcs2 = np.asarray(bpm_indcs2, dtype=int)

        if array1.shape[0] != len(bpm_indcs1):
            raise ValueError(
                'The first dimension of array1 must match '
                'the number of BPM indices in bpm_indcs1.'
            )

        if array2.shape[0] != len(bpm_indcs2):
            raise ValueError(
                'The first dimension of array2 must match '
                'the number of BPM indices in bpm_indcs2.'
            )

        bpm_indcs = np.concatenate([bpm_indcs1, bpm_indcs2])

        if len(np.unique(bpm_indcs)) != len(bpm_indcs):
            raise ValueError(
                'The BPM groups contain repeated global BPM indices.'
            )

        joined_array = np.concatenate([array1, array2], axis=0)

        order = np.argsort(bpm_indcs)
        joined_array = joined_array[order]

        if return_list:
            return joined_array.tolist()

        return joined_array

    def _join_odd_even_acquisitions(self):
        """Join BPM groups acquired with the odd/even strategy.

        Odd/even sub-acquisitions are paired in their acquisition order.
        BPM-resolved quantities are joined using their global BPM indices,
        while some scalar quantities are averaged between each pair.

        Raises:
            ValueError: If the odd/even acquisitions cannot be paired or their
                BPM-resolved arrays are inconsistent.
        """
        bpm_acquisition_keys = [
            'b1_posx',
            'b1_posy',
            'b2_posx',
            'b2_posy',
            'b1_sum',
            'b2_sum',
            'bt_sum',
        ]

        scalar_keys = [
            'stored_current',
            'rf_frequency',
            'tunex',
            'tuney',
            'ivu18_08_gap',
            'ivu18_14_gap',
            'b1_sum_mean',
            'b2_sum_mean',
            'bt_sum_mean',
            'b1_curr',
            'b2_curr',
            'delta_curr',
            'b1_charge',
            'b2_charge',
            'delta_charge',
            'delta_charge_mean',
        ]

        # Keys not explicitly joined are assumed to be:
        # - metadata
        # - quantities already containing all BPMs
        # They are copied from one of the odd/even acq.'s

        data_joined = []

        for data_i in self.data:
            params = data_i['params']
            data_meas = data_i['data']

            # Do nothing if this configuration was already joined.
            if params.get('acquisition_mode') == 'odd_even_joined':
                data_joined.append(data_i)
                continue

            acq_strategy = params.get('acq_strategy', 'all')

            # No joining is necessary for normal acquisitions.
            if not acq_strategy.startswith('odd'):
                data_joined.append(data_i)
                continue

            if len(data_meas) % 2 != 0:
                raise ValueError(
                    'Odd/even acquisition strategy requires '
                    'an even number of acquisitions.'
                )

            joined_meas = []

            for i in range(0, len(data_meas), 2):
                acq1 = data_meas[i]
                acq2 = data_meas[i + 1]

                bpm_indcs1 = np.asarray(acq1['bpm_indcs'])
                bpm_indcs2 = np.asarray(acq2['bpm_indcs'])

                joined_acq = acq1.copy()
                # Unmentioned keys will have values corresp. to acq1

                for key in bpm_acquisition_keys:
                    joined_acq[key] = self._join_bpm_groups(
                        acq1[key], acq2[key], bpm_indcs1, bpm_indcs2
                    )

                joined_acq['bpm_names'] = self._join_bpm_groups(
                    acq1['bpm_names'],
                    acq2['bpm_names'],
                    bpm_indcs1,
                    bpm_indcs2,
                    return_list=True,
                )

                joined_acq['bpm_indcs'] = np.sort(
                    np.concatenate([bpm_indcs1, bpm_indcs2])
                )

                for key in scalar_keys:
                    if key in acq1 and key in acq2:
                        joined_acq[key] = 0.5 * (acq1[key] + acq2[key])

                # The joined acquisition no longer belongs
                # to only one acquisition group.
                joined_acq.pop('acq_group', None)

                joined_meas.append(joined_acq)

            params_joined = params.copy()

            params_joined['num_acquisitions'] = len(joined_meas)
            params_joined['parity'] = 'all'
            params_joined['acquisition_mode'] = 'odd_even_joined'

            data_i_joined = data_i.copy()

            data_i_joined['params'] = params_joined
            data_i_joined['data'] = joined_meas

            data_joined.append(data_i_joined)

        self.data = data_joined

    def _check_consistency(self, data_i, n_acq, n_bpms, n_turns):
        """Check consistency of the measurement dimensions.

        Compare the calculated numbers of acquisitions, BPMs, and turns
        with the values stored in the measurement configuration. A warning
        is issued for each detected mismatch.

        Args:
            data_i (dict): Measurement configuration.
            n_acq (int): Calculated number of acquisitions.
            n_bpms (int): Calculated number of BPMs.
            n_turns (int): Calculated number of turns.
        """
        params = data_i['params']
        data_meas = data_i['data']

        expected_n_acq = params['num_acquisitions']
        expected_n_turns = params['_nrturns']
        indexed_n_bpms = len(data_meas[0]['bpm_indcs'])

        if n_acq != expected_n_acq:
            warnings.warn(
                f'Number of acquisitions mismatch: '
                f'calculated={n_acq}, '
                f'expected={expected_n_acq}',
                stacklevel=2,
            )

        if n_bpms != indexed_n_bpms:
            warnings.warn(
                f'BPM number mismatch: '
                f'calculated={n_bpms}, '
                f'indexed={indexed_n_bpms}',
                stacklevel=2,
            )

        if n_turns != expected_n_turns:
            warnings.warn(
                f'Number of turns mismatch: '
                f'calculated={n_turns}, '
                f'expected={expected_n_turns}',
                stacklevel=2,
            )

    def _make_config_table(self, gap_key='ivu18_08_gap'):
        """Create a table with the measurement configurations.

        Args:
            gap_key (str): Dictionary key containing the desired IVU gap.

        Returns:
            config_table (pandas.DataFrame): Table with one row per
                measurement configuration.

        Raises:
            ValueError: If a bunch-position array does not have the expected
                two-dimensional shape.
        """
        rows = []

        for i, data_i in enumerate(self.data):
            params = data_i['params']
            data_meas = data_i['data']

            first_acq = data_meas[0]

            y0 = params['y0']

            n_acq = len(data_meas)

            b1_posy = np.asarray(first_acq['b1_posy'])

            if b1_posy.ndim != 2:
                raise ValueError(
                    f"Configuration {i}: 'b1_posy' must have shape "
                    '(n_bpms, n_turns).'
                )

            n_bpms, n_turns = b1_posy.shape

            self._check_consistency(data_i, n_acq, n_bpms, n_turns)

            gap = np.mean([data_acq[gap_key] for data_acq in data_meas])

            b1_curr = np.mean([data_acq['b1_curr'] for data_acq in data_meas])

            b2_curr = np.mean([data_acq['b2_curr'] for data_acq in data_meas])

            delta_curr = np.mean([
                data_acq['delta_curr'] for data_acq in data_meas
            ])

            delta_charge = np.mean([
                data_acq['delta_charge'] for data_acq in data_meas
            ])

            rows.append({
                'cfg': i,
                'gap': gap,
                'y0': y0,
                'parity': params.get('parity', 'all'),
                'n_acq': n_acq,
                'n_bpms': n_bpms,
                'n_turns': n_turns,
                'delta_charge': delta_charge,
                'b1_curr': b1_curr,
                'b2_curr': b2_curr,
                'delta_curr': delta_curr,
            })

        return pd.DataFrame(rows)

    def calculate_delta_orbit(self, plane='y'):
        """Calculate the bunch-to-bunch orbit difference.

        For each measurement configuration, bunch positions are stacked
        into arrays with shape (n_acq, n_bpms, n_turns). The orbit
        difference is calculated as the position of bunch 1 minus the
        position of bunch 2.

        Args:
            plane (str): Transverse plane. Use "x" or "y".

        Returns:
            delta_orbit (list): Orbit-difference arrays, with one array
                per measurement configuration. Each array has shape
                (n_acq, n_bpms, n_turns) and units of micrometers.

        Raises:
            RuntimeError: If prepared measurement data or bunch-position
                information are unavailable.
            ValueError: If plane is invalid or the bunch-position data are
                incompatible.
        """
        if self.config_table is None:
            raise RuntimeError(
                'Prepare data before calculating orbit differences.'
            )

        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        key1 = f'b1_pos{plane}'
        key2 = f'b2_pos{plane}'

        delta_orbit = []

        for i, data_i in enumerate(self.data):
            data_meas = data_i['data']

            bpm_indcs_ref = np.asarray(data_meas[0]['bpm_indcs'], dtype=int)

            delta_i = []

            for acq, data_acq in enumerate(data_meas):
                missing_keys = {key1, key2, 'bpm_indcs'}.difference(data_acq)

                if missing_keys:
                    missing = ', '.join(sorted(missing_keys))
                    raise RuntimeError(
                        f'Configuration {i}, acquisition {acq}: '
                        'missing data required to calculate the orbit '
                        f'difference: {missing}.'
                    )

                bpm_indcs = np.asarray(data_acq['bpm_indcs'], dtype=int)

                if not np.array_equal(bpm_indcs, bpm_indcs_ref):
                    raise ValueError(
                        f'Configuration {i}, acquisition {acq}: '
                        'BPM indices differ from the first acquisition.'
                    )

                b1_pos = np.asarray(data_acq[key1])
                b2_pos = np.asarray(data_acq[key2])

                if b1_pos.shape != b2_pos.shape:
                    raise ValueError(
                        f'Configuration {i}, acquisition {acq}: '
                        'bunch-position arrays must have the same shape, '
                        f'but {key1} has shape {b1_pos.shape} and '
                        f'{key2} has shape {b2_pos.shape}.'
                    )

                delta_i.append(b1_pos - b2_pos)

            delta_orbit.append(np.asarray(delta_i))

        self.delta_orbit[plane] = delta_orbit
        self.delta_orbit_stats[plane] = None
        self.theta_fits[plane] = None
        self.kperp_fits[plane] = {}

        return delta_orbit

    def calculate_delta_orbit_stats(self, plane='y'):
        """Calculate statistics of the bunch-to-bunch orbit difference.

        For each measurement configuration, the mean and standard deviation
        are calculated over acquisitions and turns. The standard error of
        the mean is obtained using the total number of acquisition-turn
        samples.

        Args:
            plane (str): Transverse plane, either "x" or "y".

        Returns:
            stats (dict): Orbit-difference statistics. The "mean", "std",
                and "sem" entries are lists containing one array per meas.
                configuration, each with shape (n_bpms,). The "n_samples"
                entry is a list with the number of samples used for each
                configuration.

        Raises:
            ValueError: If plane is invalid.
            RuntimeError: If orbit differences have not been calculated for
                the selected plane.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        delta_orbit = self.delta_orbit[plane]

        if delta_orbit is None:
            raise RuntimeError(
                f'Calculate the {plane}-plane orbit differences first.'
            )

        mean = []
        std = []
        sem = []
        n_samples = []

        for delta_i in delta_orbit:
            n_acq, _, n_turns = delta_i.shape
            n_samples_i = n_acq * n_turns

            mean_i = np.mean(delta_i, axis=(0, 2))

            std_i = np.std(delta_i, axis=(0, 2))

            sem_i = std_i / np.sqrt(n_samples_i)

            mean.append(mean_i)
            std.append(std_i)
            sem.append(sem_i)
            n_samples.append(n_samples_i)

        stats = {
            'mean': mean,
            'std': std,
            'sem': sem,
            'n_samples': np.asarray(n_samples, dtype=int),
        }

        self.delta_orbit_stats[plane] = stats
        self.theta_fits[plane] = None
        self.kperp_fits[plane] = {}

        return stats

    def _get_delta_charges(self):
        """Get the mean bunch charge difference for each configuration.

        Returns:
            delta_charge (numpy.ndarray): Mean bunch charge differences,
                with one value per measurement configuration [nC].

        Raises:
            RuntimeError: If measurement data have not been prepared.
            ValueError: If any charge difference is zero or non-finite.
        """
        if self.config_table is None:
            raise RuntimeError(
                'Prepare data before accessing charge information.'
            )

        delta_charge = self.config_table['delta_charge'].to_numpy(dtype=float)

        if not np.all(np.isfinite(delta_charge)):
            raise ValueError(
                'Charge differences must contain only finite values.'
            )

        zero_charge = np.isclose(
            delta_charge, 0.0, rtol=0.0, atol=np.finfo(float).eps
        )

        if np.any(zero_charge):
            invalid_i = np.where(zero_charge)[0]
            raise ValueError(
                'Cannot normalize orbit differences because the charge '
                'difference is zero for configuration(s) '
                f'{invalid_i.tolist()}.'
            )

        return delta_charge

    def get_delta_orbit(self, plane='y', normalized=False):
        """Get raw or charge-normalized orbit differences.

        Args:
            plane (str): Transverse plane. Use "x" or "y".
            normalized (bool): Whether to divide the orbit difference of
                each configuration by its mean bunch charge difference.

        Returns:
            delta_orbit (list): Orbit-difference arrays, with one array per
                measurement configuration. Each array has shape
                (n_acq, n_bpms, n_turns). Units are micrometers if
                normalized is False and micrometers per nanocoulomb
                otherwise.

        Raises:
            ValueError: If plane is invalid or the orbit and charge data
                are incompatible.
            RuntimeError: If orbit differences are unavailable for the
                selected plane.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        delta_orbit = self.delta_orbit[plane]

        if delta_orbit is None:
            raise RuntimeError(
                f'Calculate the {plane}-plane orbit differences first.'
            )

        if not normalized:
            return delta_orbit

        delta_charge = self._get_delta_charges()

        if len(delta_orbit) != len(delta_charge):
            raise ValueError(
                'The numbers of orbit configurations and charge '
                'differences do not match.'
            )

        delta_orbit_norm = [
            delta_i / delta_charge[i] for i, delta_i in enumerate(delta_orbit)
        ]

        return delta_orbit_norm

    def get_delta_orbit_stats(self, plane='y', normalized=False):
        """Get raw or charge-normalized orbit-difference statistics.

        Args:
            plane (str): Transverse plane, either "x" or "y".
            normalized (bool): Whether to normalize the orbit statistics by
                the mean bunch charge differences.

        Returns:
            stats (dict): Orbit-difference statistics. The "mean", "std",
                and "sem" entries contain one array per measurement
                configuration, each with shape (n_bpms,). The "n_samples"
                entry contains the number of samples used for each
                configuration. Orbit units are micrometers if normalized is
                False and micrometers per nanocoulomb otherwise.

        Raises:
            ValueError: If plane is invalid or the statistics and charge
                data are incompatible.
            RuntimeError: If orbit-difference statistics are unavailable
                for the selected plane.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        stats = self.delta_orbit_stats[plane]

        if stats is None:
            raise RuntimeError(
                f'Calculate the {plane}-plane orbit statistics first.'
            )

        if not normalized:
            return stats

        delta_charge = self._get_delta_charges()

        if len(stats['mean']) != len(delta_charge):
            raise ValueError(
                'The numbers of orbit-statistics configurations and '
                'charge differences do not match.'
            )

        stats_norm = {
            'mean': [
                mean_i / delta_charge[i]
                for i, mean_i in enumerate(stats['mean'])
            ],
            'std': [
                std_i / np.abs(delta_charge[i])
                for i, std_i in enumerate(stats['std'])
            ],
            'sem': [
                sem_i / np.abs(delta_charge[i])
                for i, sem_i in enumerate(stats['sem'])
            ],
            'n_samples': stats['n_samples'].copy(),
        }

        return stats_norm

    def plot_acquisition_means(
        self,
        cfg=0,
        acqs=None,
        plane='y',
        normalized=False,
        error='sem',
        ax=None,
    ):
        """Plot mean orbit-difference profiles for selected acquisitions.

        For each selected acquisition, the orbit difference is averaged over
        turns while preserving the BPM axis.

        Args:
            cfg (int): Measurement configuration index.
            acqs (int or sequence of int, optional): Acquisition indices to
                plot. If None, all acquisitions are plotted.
            plane (str): Transverse plane. Use "x" or "y".
            normalized (bool): Whether to normalize by the mean bunch charge
                differences.
            error (str, optional): Error bars to display. Use "std", "sem",
                or None.
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            RuntimeError: If BPM positions or orbit differences are
                unavailable.
            ValueError: If plane or error is invalid.
            IndexError: If cfg or an acquisition index is outside the
                valid range.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if error not in (None, 'std', 'sem'):
            raise ValueError("error must be None, 'std', or 'sem'.")

        if self.spos_bpms is None:
            raise RuntimeError(
                'Set BPM longitudinal positions before plotting orbit data.'
            )

        delta_orbit = self.get_delta_orbit(plane=plane, normalized=normalized)

        if not 0 <= cfg < len(delta_orbit):
            raise IndexError(
                f'Configuration index {cfg} is outside the valid range.'
            )

        delta_i = delta_orbit[cfg]
        n_acq = delta_i.shape[0]

        if acqs is None:
            acqs = range(n_acq)
        elif np.isscalar(acqs):
            acqs = [int(acqs)]
        else:
            acqs = list(acqs)

        invalid_acqs = [acq for acq in acqs if not 0 <= acq < n_acq]

        if invalid_acqs:
            raise IndexError(
                f'Acquisition indices outside the valid range: {invalid_acqs}.'
            )

        bpm_indcs = np.asarray(
            self.data[cfg]['data'][0]['bpm_indcs'], dtype=int
        )
        spos_i = np.asarray(self.spos_bpms)[bpm_indcs]

        mean_by_acq = np.mean(delta_i, axis=2)

        if error == 'std':
            error_by_acq = np.std(delta_i, axis=2)
        elif error == 'sem':
            error_by_acq = np.std(delta_i, axis=2) / np.sqrt(delta_i.shape[2])
        else:
            error_by_acq = None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        else:
            fig = ax.figure

        for acq in acqs:
            if error_by_acq is None:
                ax.plot(
                    spos_i,
                    mean_by_acq[acq],
                    '.-',
                    alpha=0.6,
                    label=f'Acquisition {acq}',
                )
            else:
                ax.errorbar(
                    spos_i,
                    mean_by_acq[acq],
                    yerr=error_by_acq[acq],
                    marker='.',
                    linestyle='-',
                    capsize=2,
                    alpha=0.6,
                    label=f'Acquisition {acq}',
                )

        row = self.config_table.iloc[cfg]
        plane_name = {'x': 'Horizontal', 'y': 'Vertical'}[plane]

        if normalized:
            ylabel = (
                rf'$\langle\Delta {plane}\rangle_{{\rm turns}}$ '
                r'[$\mu$m/nC]'
            )
        else:
            ylabel = (
                rf'$\langle\Delta {plane}\rangle_{{\rm turns}}$ '
                r'[$\mu$m]'
            )

        ax.set_title(
            f'{plane_name} orbit difference between bunches by acquisition\n'
            f'Configuration {cfg}: gap = {row["gap"]:.2f} mm, '
            rf'$y_0$ = {row["y0"]:.2f} mm'
        )
        ax.set_xlabel('BPM longitudinal position [m]')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        return fig, ax

    def plot_config_means(
        self,
        cfgs=None,
        ref_cfg=None,
        plane='y',
        normalized=False,
        error='sem',
        ax=None,
    ):
        """Plot mean orbit-difference profiles for selected configurations.

        The plotted profiles are averaged over acquisitions and turns. If a
        reference configuration is provided, its profile is subtracted after
        aligning both configurations through their global BPM indices.

        Args:
            cfgs (int or sequence of int, optional): Configuration indices to
                plot. If None, all configurations are plotted.
            ref_cfg (int, optional): Reference configuration to subtract.
                If None, absolute orbit-difference profiles are plotted.
            plane (str): Transverse plane, either "x" or "y".
            normalized (bool): Whether to normalize by the mean bunch charge
                differences.
            error (str, optional): Error bars to display. Use "std", "sem",
                or None.
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            RuntimeError: If BPM positions or orbit statistics are
                unavailable.
            ValueError: If plane or error is invalid, or the reference
                configuration does not contain the required BPMs.
            IndexError: If a configuration index is outside the valid
                range.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if error not in (None, 'std', 'sem'):
            raise ValueError("error must be None, 'std', or 'sem'.")

        if self.spos_bpms is None:
            raise RuntimeError(
                'Set BPM longitudinal positions before plotting orbit data.'
            )

        stats = self.get_delta_orbit_stats(plane=plane, normalized=normalized)

        n_configs = len(stats['mean'])

        if cfgs is None:
            cfgs = range(n_configs)
        elif np.isscalar(cfgs):
            cfgs = [int(cfgs)]
        else:
            cfgs = list(cfgs)

        invalid_cfgs = [i for i in cfgs if not 0 <= i < n_configs]

        if invalid_cfgs:
            raise IndexError(
                'Configuration indices outside the valid range: '
                f'{invalid_cfgs}.'
            )

        if ref_cfg is not None and not 0 <= ref_cfg < n_configs:
            raise IndexError(
                f'Reference configuration index {ref_cfg} is outside '
                'the valid range.'
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        else:
            fig = ax.figure

        if ref_cfg is not None:
            ref_bpm_indcs = np.asarray(
                self.data[ref_cfg]['data'][0]['bpm_indcs'], dtype=int
            )

            ref_bpm_map = {
                bpm_idx: local_idx
                for local_idx, bpm_idx in enumerate(ref_bpm_indcs)
            }

            mean_ref_full = stats['mean'][ref_cfg]

            if error is not None:
                error_ref_full = stats[error][ref_cfg]

        spos_bpms = np.asarray(self.spos_bpms)

        for i in cfgs:
            bpm_indcs = np.asarray(
                self.data[i]['data'][0]['bpm_indcs'], dtype=int
            )
            spos_i = spos_bpms[bpm_indcs]

            mean_i = stats['mean'][i]

            if error is not None:
                error_i = stats[error][i]

            if ref_cfg is not None:
                missing_bpms = [
                    int(bpm_idx)
                    for bpm_idx in bpm_indcs
                    if bpm_idx not in ref_bpm_map
                ]

                if missing_bpms:
                    raise ValueError(
                        f'Reference configuration {ref_cfg} does not '
                        f'contain BPM indices required by configuration '
                        f'{i}: {missing_bpms}.'
                    )

                ref_local_indcs = np.asarray([
                    ref_bpm_map[bpm_idx] for bpm_idx in bpm_indcs
                ])

                mean_ref = mean_ref_full[ref_local_indcs]
                mean_plot = mean_i - mean_ref

                if error is not None:
                    error_ref = error_ref_full[ref_local_indcs]

                    if i == ref_cfg:
                        error_plot = np.zeros_like(error_i)
                    else:
                        error_plot = np.sqrt(error_i**2 + error_ref**2)
            else:
                mean_plot = mean_i

                if error is not None:
                    error_plot = error_i

            row = self.config_table.iloc[i]

            label = (
                f'Configuration {i}: '
                f'gap = {row["gap"]:.2f} mm, '
                rf'$y_0$ = {row["y0"]:.2f} mm'
            )

            if error is None:
                ax.plot(spos_i, mean_plot, '.-', alpha=0.7, label=label)
            else:
                ax.errorbar(
                    spos_i,
                    mean_plot,
                    yerr=error_plot,
                    marker='.',
                    linestyle='-',
                    capsize=2,
                    alpha=0.7,
                    label=label,
                )

        plane_name = {'x': 'Horizontal', 'y': 'Vertical'}[plane]

        if normalized:
            unit = r'$\mu$m/nC'
        else:
            unit = r'$\mu$m'

        if ref_cfg is None:
            title = f'{plane_name} mean orbit difference between bunches'
            ylabel = (
                rf'$\langle\Delta {plane}\rangle_'
                rf'{{\rm acqs,turns}}$ [{unit}]'
            )
        else:
            title = (
                f'{plane_name} mean orbit shift between bunches relative to '
                f'Configuration {ref_cfg}\n'
            )
            ylabel = (
                rf'$\langle\Delta {plane}\rangle'
                rf'-\langle\Delta {plane}\rangle_{{\rm ref}}$ '
                f'[{unit}]'
            )

        ax.set_title(title)
        ax.set_xlabel('BPM longitudinal position [m]')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        return fig, ax

    def calc_ivu_respmat(self, delta_vkick=5e-6, id_name='IVU18_SI08'):
        """Calculate the IVU orbit response matrix column.

        The complete response column is stored in self.resp_mat. Its first
        half contains the horizontal response and its second half contains
        the vertical response.

        Args:
            delta_vkick (float): Difference between the positive and
                negative finite-difference vertical kicks [rad].
            id_name (str): IVU name, either "IVU18_SI08" or "IVU18_SI14".

        Returns:
            resp_mat (numpy.ndarray): Full IVU response column with shape
                (2*n_bpms,) [m/rad].
            info (dict): Auxiliary model information returned by the response
                calculation.

        Raises:
            ValueError: If the finite-difference kick or id_name is invalid,
                or the IVU elements cannot be identified in the model.
        """
        resp_mat, info = self._calc_ivu_respmat(
            delta_vkick=delta_vkick, id_name=id_name
        )

        self.id_name = id_name
        self.resp_mat = resp_mat

        self.M_ivu = {'x': None, 'y': None}
        self.theta_fits = {'x': None, 'y': None}
        self.kperp_fits = {'x': {}, 'y': {}}

        return resp_mat, info

    def set_ivu_respmat(self, plane='y'):
        """Set the IVU response vector for each measurement configuration.

        The complete response matrix column is split into horizontal and
        vertical components. The selected component is then restricted to the
        global BPM indices used by each measurement configuration.

        Args:
            plane (str): Transverse plane. Use "x" or "y".

        Returns:
            m_ivu (list): IVU response arrays, with one array per measurement
                configuration. Each array contains the response values for
                the BPMs used in that configuration [m/rad].

        Raises:
            RuntimeError: If measurement data, BPM positions, or the IVU
                response column are unavailable.
            ValueError: If plane is invalid or the response-column size is
                incompatible with the BPM model.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if self.data is None:
            raise RuntimeError('Load or set measurement data first.')

        if self.spos_bpms is None:
            raise RuntimeError('Set BPM longitudinal positions first.')

        if self.resp_mat is None:
            raise RuntimeError('Calculate the IVU response matrix first.')

        n_bpms = len(self.spos_bpms)

        if len(self.resp_mat) != 2 * n_bpms:
            raise ValueError(
                'The response matrix size is incompatible with the '
                'number of BPMs.'
            )

        if plane == 'x':
            m_full = self.resp_mat[:n_bpms]
        else:
            m_full = self.resp_mat[n_bpms:]

        m_ivu = []

        for data_i in self.data:
            bpm_indcs = np.asarray(data_i['data'][0]['bpm_indcs'], dtype=int)

            m_ivu.append(m_full[bpm_indcs])

        self.M_ivu[plane] = m_ivu
        self.theta_fits[plane] = None
        self.kperp_fits[plane] = {}

        return m_ivu

    def fit_theta(self, plane='y', error='sem'):
        """Fit the IVU kick angle for each measurement configuration.

        For each configuration, the mean orbit-difference profile is
        projected onto the corresponding IVU response vector using weighted
        least squares. The fitted model is Delta u = theta*M_IVU.

        Args:
            plane (str): Transverse plane, either "x" or 'y'.
            error (str or None): Orbit uncertainty used in the fit. Use
                'std', 'sem', or None for an unweighted fit.

        Returns:
            theta_fit (dict): Fitted angles, uncertainties, orbit models,
                residuals, and goodness-of-fit quantities.

        Raises:
            RuntimeError: If orbit statistics or IVU response vectors are
                unavailable.
            ValueError: If plane or error is invalid, or the orbit and
                response vectors are incompatible.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if error not in (None, 'std', 'sem'):
            raise ValueError("error must be None, 'std', or 'sem'.")

        stats = self.delta_orbit_stats[plane]

        if stats is None:
            raise RuntimeError(
                f'Calculate the {plane}-plane orbit statistics first.'
            )

        m_ivu = self.M_ivu[plane]

        if m_ivu is None:
            raise RuntimeError(
                f'Set the {plane}-plane IVU response matrix first.'
            )

        if len(stats['mean']) != len(m_ivu):
            raise ValueError(
                'The numbers of orbit profiles and IVU response vectors '
                'do not match.'
            )

        theta = []
        theta_err = []
        model = []
        residual = []
        residual_rms = []
        chi2 = []
        reduced_chi2 = []

        for i, (delta_i, m_i) in enumerate(zip(stats['mean'], m_ivu)):
            delta_i = np.asarray(delta_i)
            m_i = np.asarray(m_i)

            if delta_i.shape != m_i.shape:
                raise ValueError(
                    f'Configuration {i}: orbit and IVU response vectors '
                    f'have incompatible shapes: {delta_i.shape} and '
                    f'{m_i.shape}.'
                )

            if error is None:
                error_i = None
                weights = np.ones_like(delta_i)
            else:
                error_i = np.asarray(stats[error][i])
                error_i = np.maximum(error_i, np.finfo(float).eps)
                weights = 1.0 / error_i**2

            denominator = np.sum(weights * m_i**2)

            if np.isclose(denominator, 0.0):
                raise ValueError(
                    f'Configuration {i}: the IVU response does not allow '
                    'the kick angle to be determined.'
                )

            numerator = np.sum(weights * delta_i * m_i)

            theta_i = numerator / denominator
            model_i = theta_i * m_i
            residual_i = delta_i - model_i

            if error_i is None:
                theta_err_i = np.nan
                chi2_i = np.nan
                reduced_chi2_i = np.nan
            else:
                theta_err_i = 1.0 / np.sqrt(denominator)

                chi2_i = np.sum((residual_i / error_i) ** 2)

                dof_i = len(delta_i) - 1

                if dof_i > 0:
                    reduced_chi2_i = chi2_i / dof_i
                else:
                    reduced_chi2_i = np.nan

            theta.append(theta_i)
            theta_err.append(theta_err_i)
            model.append(model_i)
            residual.append(residual_i)
            residual_rms.append(np.sqrt(np.mean(residual_i**2)))
            chi2.append(chi2_i)
            reduced_chi2.append(reduced_chi2_i)

        theta_fit = {
            'theta': np.asarray(theta),
            'theta_err': np.asarray(theta_err),
            'model': model,
            'residual': residual,
            'residual_rms': np.asarray(residual_rms),
            'chi2': np.asarray(chi2),
            'reduced_chi2': np.asarray(reduced_chi2),
            'error': error,
        }

        self.theta_fits[plane] = theta_fit
        self.kperp_fits[plane] = {}

        return theta_fit

    def fit_kperp(self, gap, plane='y'):
        """Fit the transverse kick factor for one IVU gap.

        The projected kick angles are normalized by the measured bunch
        charge differences and fitted as a linear function of the orbit
        bump:

            theta / Delta q = b + a*y0

        The transverse kick factor is calculated from the fitted slope.

        Args:
            gap (float): IVU gap to fit [mm].
            plane (str): Transverse plane. Use "x" or "y".

        Returns:
            kperp_fit (dict): Linear-fit and kick-factor results.

        Raises:
            RuntimeError: If fitted kick angles or their uncertainties are
                unavailable.
            ValueError: If plane is invalid, fewer than two configurations are
                available, or a selected charge difference is invalid.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        theta_fit = self.theta_fits[plane]

        if theta_fit is None:
            raise RuntimeError(f'Fit the {plane}-plane kick angles first.')

        gap_values = self.config_table['gap'].to_numpy(dtype=float)

        mask = np.isclose(gap_values, gap, rtol=0.0, atol=1e-6)
        i_selected = np.where(mask)[0]

        if len(i_selected) < 2:
            raise ValueError(
                'At least two configurations are required to fit the '
                f'kick factor at gap {gap:.3f} mm.'
            )

        y0 = self.config_table.iloc[i_selected]['y0'].to_numpy(dtype=float)

        delta_charge = self.config_table.iloc[i_selected][
            'delta_charge'
        ].to_numpy(dtype=float)

        theta = theta_fit['theta'][i_selected]
        theta_err = theta_fit['theta_err'][i_selected]

        if not np.all(np.isfinite(theta_err)) or np.any(theta_err <= 0.0):
            raise RuntimeError(
                'Positive finite kick-angle uncertainties are required to fit '
                "the kick factor. Fit the kick angles using error='std' or "
                "error='sem'."
            )

        if not np.all(np.isfinite(delta_charge)) or np.any(
            np.isclose(delta_charge, 0.0, rtol=0.0, atol=np.finfo(float).eps)
        ):
            raise ValueError('Charge differences must be finite and nonzero.')

        theta_norm = theta / delta_charge
        theta_norm_err = theta_err / np.abs(delta_charge)

        # Model: theta / Delta q = b + a*y0.
        design_matrix = np.column_stack([np.ones(len(y0)), y0])

        weights_sqrt = 1.0 / theta_norm_err

        design_matrix_weighted = design_matrix * weights_sqrt[:, None]
        theta_norm_weighted = theta_norm * weights_sqrt

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix_weighted, theta_norm_weighted, rcond=None
        )

        b = coefficients[0]
        a = coefficients[1]

        theta_norm_model = design_matrix @ coefficients
        residual = theta_norm - theta_norm_model

        covariance = np.linalg.pinv(
            design_matrix_weighted.T @ design_matrix_weighted
        )
        coefficients_err = np.sqrt(np.diag(covariance))

        b_err = coefficients_err[0]
        a_err = coefficients_err[1]

        chi2 = np.sum((residual / theta_norm_err) ** 2)

        dof = len(y0) - len(coefficients)

        if dof > 0:
            reduced_chi2 = chi2 / dof
        else:
            reduced_chi2 = np.nan

        kperp = 1e-6 * self.beam_voltage * a
        kperp_err = 1e-6 * self.beam_voltage * a_err

        kperp_fit = {
            'gap': float(gap),
            'plane': plane,
            'config_indices': i_selected,
            'y0': y0,
            'delta_charge': delta_charge,
            'delta_charge_mean': np.mean(delta_charge),
            'theta': theta,
            'theta_err': theta_err,
            'theta_norm': theta_norm,
            'theta_norm_err': theta_norm_err,
            'theta_norm_model': theta_norm_model,
            'residual': residual,
            'b': b,
            'b_err': b_err,
            'a': a,
            'a_err': a_err,
            'kperp': kperp,
            'kperp_err': kperp_err,
            'dof': dof,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
        }

        self.kperp_fits[plane][float(gap)] = kperp_fit

        return kperp_fit

    def get_kperp_results_table(self, plane='y'):
        """Create a summary table of the fitted transverse kick factors.

        Args:
            plane (str): Transverse plane, either "x" or "y".

        Returns:
            results (pandas.DataFrame): Fit results with one row per IVU gap.

        Raises:
            ValueError: If plane is invalid.
            RuntimeError: If no kick-factor fits are available for the
                selected plane.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        fits = self.kperp_fits[plane]

        if not fits:
            raise RuntimeError(
                f'No {plane}-plane kick-factor fits are available.'
            )

        rows = []

        for gap in sorted(fits):
            fit = fits[gap]

            relative_error = (
                np.abs(fit['kperp_err'] / fit['kperp'])
                if not np.isclose(fit['kperp'], 0.0)
                else np.nan
            )

            rows.append({
                'gap [mm]': fit['gap'],
                'kperp [V/(pC m)]': fit['kperp'],
                'kperp error [V/(pC m)]': fit['kperp_err'],
                'relative error [%]': 100.0 * relative_error,
                'reduced chi2': fit['reduced_chi2'],
                'dof': fit['dof'],
                'n_configs': len(fit['config_indices']),
            })

        return pd.DataFrame(rows)

    def plot_theta_fit(self, gap, plane='y', normalized=True, ax=None):
        """Plot projected kick angles and the fitted kick-factor model.

        Args:
            gap (float): IVU gap corresponding to the kick-factor fit [mm].
            plane (str): Transverse plane, either "x" or "y".
            normalized (bool): Whether to plot theta/Delta q instead of theta.
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            ValueError: If plane is invalid.
            RuntimeError: If no kick-factor fit is available for the selected
                plane and gap.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        gap_key = float(gap)

        if gap_key not in self.kperp_fits[plane]:
            raise RuntimeError(
                f'Fit the {plane}-plane kick factor at gap {gap:.3f} mm first.'
            )

        fit = self.kperp_fits[plane][gap_key]

        y0 = fit['y0']
        theta = fit['theta']
        theta_err = fit['theta_err']
        delta_charge = fit['delta_charge']

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        else:
            fig = ax.figure

        if normalized:
            values = fit['theta_norm']
            values_err = fit['theta_norm_err']

            y0_fit = np.linspace(np.min(y0), np.max(y0), 100)
            model_fit = fit['b'] + fit['a'] * y0_fit

            ylabel = (
                rf'$\hat{{\theta}}_{plane}/\Delta q$ '
                r'[$\mu$rad/nC]'
            )
        else:
            values = theta
            values_err = theta_err

            order = np.argsort(y0)
            y0_fit = y0[order]
            model_fit = delta_charge[order] * fit['theta_norm_model'][order]

            ylabel = (
                rf'$\hat{{\theta}}_{plane}$ '
                r'[$\mu$rad]'
            )

        ax.errorbar(
            y0,
            values,
            yerr=values_err,
            marker='o',
            linestyle='none',
            markersize=6,
            capsize=4,
            label='Measurements',
        )

        ax.plot(y0_fit, model_fit, '-', label='Weighted linear fit')

        id_label = self.id_name

        ax.set_title(
            rf'Projected $\hat{{\theta}}_{plane}$ versus $y_0$ ({id_label})'
            '\n'
            f'Gap = {fit["gap"]:.2f} mm, '
            rf'$k_\perp$ = {fit["kperp"]:.2f} '
            rf'$\pm$ {fit["kperp_err"]:.2f} V/(pC m)'
        )
        ax.set_xlabel(r'$y_0$ [mm]')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        return fig, ax

    def plot_ivu_projection(
        self,
        cfg,
        ref_cfg,
        plane='y',
        normalized=True,
        model='global',
        error='sem',
        ax=None,
    ):
        """Plot the reference-subtracted orbit and fitted IVU model.

        The model can use either the independently projected kick angles or
        the ones obtained from the global kick-factor fit for the selected
        IVU gap.

        Args:
            cfg (int): Configuration to plot.
            ref_cfg (int): Reference configuration to subtract.
            plane (str): Transverse plane. Use "x" or "y".
            normalized (bool): Whether to plot the orbit difference normalized
                by the mean bunch charge difference.
            model (str): Model used to predict the IVU orbit. Use
                "individual" or "global".
            error (str): Error bars to display, either "std" or "sem".
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            RuntimeError: If required orbit, response, or fit results are
                unavailable.
            ValueError: If plane, model, or error is invalid, or the
                reference configuration does not contain the required BPMs.
            IndexError: If cfg or ref_cfg is outside the valid range.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if model not in ('individual', 'global'):
            raise ValueError("model must be 'individual' or 'global'.")

        if error not in ('std', 'sem'):
            raise ValueError("error must be 'std' or 'sem'.")

        stats = self.get_delta_orbit_stats(plane=plane, normalized=False)

        n_configs = len(stats['mean'])

        if not 0 <= cfg < n_configs:
            raise IndexError(
                f'Configuration index {cfg} is outside the valid range.'
            )

        if not 0 <= ref_cfg < n_configs:
            raise IndexError(
                f'Reference configuration index {ref_cfg} is outside '
                'the valid range.'
            )

        theta_fit = self.theta_fits[plane]
        m_ivu = self.M_ivu[plane]

        if theta_fit is None:
            raise RuntimeError(f'Fit the {plane}-plane kick angles first.')

        if m_ivu is None:
            raise RuntimeError(
                f'Set the {plane}-plane IVU response matrix first.'
            )

        row = self.config_table.iloc[cfg]
        row_ref = self.config_table.iloc[ref_cfg]

        bpm_indcs = np.asarray(
            self.data[cfg]['data'][0]['bpm_indcs'], dtype=int
        )
        ref_bpm_indcs = np.asarray(
            self.data[ref_cfg]['data'][0]['bpm_indcs'], dtype=int
        )

        ref_bpm_map = {
            bpm_idx: local_idx
            for local_idx, bpm_idx in enumerate(ref_bpm_indcs)
        }

        missing_bpms = [
            int(bpm_idx) for bpm_idx in bpm_indcs if bpm_idx not in ref_bpm_map
        ]

        if missing_bpms:
            raise ValueError(
                f'Reference configuration {ref_cfg} does not contain '
                f'BPM indices required by configuration {cfg}: '
                f'{missing_bpms}.'
            )

        ref_local_indcs = np.asarray([
            ref_bpm_map[bpm_idx] for bpm_idx in bpm_indcs
        ])

        delta_cfg = stats['mean'][cfg]
        delta_ref = stats['mean'][ref_cfg][ref_local_indcs]

        error_cfg = stats[error][cfg]
        error_ref = stats[error][ref_cfg][ref_local_indcs]

        q_cfg = row['delta_charge']
        q_ref = row_ref['delta_charge']

        if (
            not np.isfinite(q_cfg)
            or not np.isfinite(q_ref)
            or np.isclose(q_cfg, 0.0)
            or np.isclose(q_ref, 0.0)
        ):
            raise ValueError(
                'The plotted and reference charge differences must be '
                'finite and nonzero.'
            )

        if normalized:
            delta_plot = delta_cfg / q_cfg - delta_ref / q_ref

            error_plot = np.sqrt(
                (error_cfg / np.abs(q_cfg)) ** 2
                + (error_ref / np.abs(q_ref)) ** 2
            )

            ylabel = (
                rf'Reference-subtracted $\Delta {plane}$ '
                r'[$\mu$m/nC]'
            )
            theta_unit = r'$\mu$rad/nC'
        else:
            charge_scale = q_cfg / q_ref

            delta_plot = delta_cfg - charge_scale * delta_ref

            error_plot = np.sqrt(
                error_cfg**2 + (charge_scale * error_ref) ** 2
            )

            ylabel = (
                rf'Reference-subtracted $\Delta {plane}$ '
                r'[$\mu$m]'
            )
            theta_unit = r'$\mu$rad'

        m_cfg = m_ivu[cfg]

        if model == 'individual':
            theta_cfg = theta_fit['theta'][cfg]
            theta_ref = theta_fit['theta'][ref_cfg]

            theta_err_cfg = theta_fit['theta_err'][cfg]
            theta_err_ref = theta_fit['theta_err'][ref_cfg]

            if normalized:
                theta_model = theta_cfg / q_cfg - theta_ref / q_ref

                theta_model_err = np.sqrt(
                    (theta_err_cfg / np.abs(q_cfg)) ** 2
                    + (theta_err_ref / np.abs(q_ref)) ** 2
                )
            else:
                charge_scale = q_cfg / q_ref

                theta_model = theta_cfg - charge_scale * theta_ref

                theta_model_err = np.sqrt(
                    theta_err_cfg**2 + (charge_scale * theta_err_ref) ** 2
                )

            model_label = 'Individual theta projection'

        else:
            gap = float(row['gap'])

            if gap not in self.kperp_fits[plane]:
                raise RuntimeError(
                    f'Fit the {plane}-plane kick factor at gap '
                    f'{gap:.3f} mm first.'
                )

            fit = self.kperp_fits[plane][gap]

            delta_y0 = row['y0'] - row_ref['y0']

            if normalized:
                theta_model = fit['a'] * delta_y0
                theta_model_err = np.abs(delta_y0) * fit['a_err']
            else:
                theta_model = q_cfg * fit['a'] * delta_y0
                theta_model_err = np.abs(q_cfg * delta_y0) * fit['a_err']

            model_label = 'Global kick-factor model'

        orbit_model = theta_model * m_cfg

        spos_cfg = np.asarray(self.spos_bpms)[bpm_indcs]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        else:
            fig = ax.figure

        ax.errorbar(
            spos_cfg,
            delta_plot,
            yerr=error_plot,
            marker='.',
            linestyle='none',
            capsize=2,
            alpha=0.6,
            label='Reference-subtracted measurement',
        )

        ax.plot(
            spos_cfg,
            orbit_model,
            '-',
            linewidth=1.8,
            label=(
                f'{model_label}: '
                rf'$\theta={theta_model:.3f}'
                rf'\pm{theta_model_err:.3f}$ '
                f'{theta_unit}'
            ),
        )

        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)

        plane_name = {'x': 'Horizontal', 'y': 'Vertical'}[plane]

        id_label = self.id_name

        ax.set_title(
            f'{plane_name} orbit distortion ({id_label})\n'
            f'Gap = {row["gap"]:.2f} mm, '
            rf'$y_0$ = {row["y0"]:.2f} mm, '
            rf'$y_{{0,\mathrm{{ref}}}}$ = '
            f'{row_ref["y0"]:.2f} mm'
        )
        ax.set_xlabel('BPM longitudinal position [m]')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        return fig, ax

    def plot_kperp_vs_gap(self, plane='y', ax=None):
        """Plot the fitted transverse kick factor versus IVU gap.

        Args:
            plane (str): Transverse plane, either "x" or 'y'.
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            ValueError: If plane is invalid.
            RuntimeError: If no kick-factor fits are available for the
                selected plane.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        fits = self.kperp_fits[plane]

        if not fits:
            raise RuntimeError(
                f'No {plane}-plane kick-factor fits are available.'
            )

        gaps = np.asarray(sorted(fits), dtype=float)

        kperp = np.asarray([fits[gap]['kperp'] for gap in gaps])

        kperp_err = np.asarray([fits[gap]['kperp_err'] for gap in gaps])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        else:
            fig = ax.figure

        ax.errorbar(
            gaps,
            kperp,
            yerr=kperp_err,
            marker='o',
            linestyle='-',
            capsize=4,
            label='Measurement',
        )

        plane_name = {'x': 'Horizontal', 'y': 'Vertical'}[plane]

        id_label = self.id_name

        ax.set_title(
            f'{plane_name} transverse kick factor versus gap ({id_label})'
        )
        ax.set_xlabel('IVU gap [mm]')
        ax.set_ylabel(r'$k_\perp$ [V/(pC m)]')
        ax.grid(True)

        if len(gaps) > 1:
            ax.legend()

        return fig, ax
