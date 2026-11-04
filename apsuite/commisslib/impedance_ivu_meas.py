"""."""

from queue import Queue as _Queue
import time as _time
import warnings as _warnings

import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt

import pyaccel
from pymodels import si as _si
from mathphys.functions import load as _load
from siriuspy.epics import CAThread as _Thread
from siriuspy.devices import SOFB, IVU, InjCtrl, BunchbyBunch

from apsuite.commisslib.meas_bpms_signals import (
    AcqBPMsSignals as _BaseAcq,
    AcqBPMsSignalsParams as _BaseParams,
)
from .bumps_scans import BumpScan as _Bump, BumpScanParams as _BumpParams
from apsuite.utils import ThreadedMeasBaseClass as _BaseThreaded


class ImpedanceIVUMeasAllBumpsParams(_BumpParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.max_bump_to_inject = 600  # [um]
        self.target_curr = 1.4  # [mA]
        self.timeout_waiting_injection = 120  # [s]

    def __str__(self):
        """."""
        stg = f'max_bump_to_inject = {self.max_bump_to_inject} [um]\n'
        stg += f'target_curr = {self.target_curr} [mA]\n'
        stg += 'timeout_waiting_injection = '
        stg += f'{self.timeout_waiting_injection} [s]\n'
        stg += super().__str__()
        return stg


class ImpedanceIVUMeasAllBumps(_Bump):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            meas_func=self._do_meas,
            isonline=isonline,
            params=ImpedanceIVUMeasAllBumpsParams(),
        )
        self.ivu_imp_meas = ImpedanceIVUMeas(isonline=isonline)
        self._thread_loop = None
        self._stop_loop = False

    def __str__(self):
        """."""
        prm = str(self.params)
        stg = 'ImpedanceIVUMeasAllBumpsParams:\n'
        stg += ''.join([f'    {lin}\n' for lin in prm.splitlines()])
        prm = str(self.ivu_imp_meas.params)
        stg += '\nImpedanceIVUMeasParams:\n'
        stg += ''.join([f'    {lin}\n' for lin in prm.splitlines()])
        return stg

    def stop(self):
        """."""
        super().stop()
        self._stop_loop = True

    def do_inject(self):
        """."""
        injctrl = self.devices['injctrl']
        dic = {'timeout': self.params.timeout_waiting_injection}

        injctrl.target_current = self.params.target_curr
        injctrl.accum_state = True
        if not self._call_and_wait(injctrl.wait_accum_start, **dic):
            injctrl.accum_state = False
            return False
        if not self._call_and_wait(injctrl.wait_accum_finish, **dic):
            injctrl.accum_state = False
            self._call_and_wait(injctrl.wait_accum_finish, **dic)
            return False

        # Do bunch cleaning
        bbbv = self.devices['bbbv']
        bbbv.bunch_clean.state = True
        _time.sleep(2)
        bbbv.bunch_clean.state = False
        return True

    def loop_over_gaps(
        self, gaps, fname_fmt=None, ivu_name='ivu18_08', timeout_move_gap=30
    ):
        """Make measurements for several IVU gaps.

        Args:
            gaps (list|numpy.ndarray, (N, )): gaps for which measurements will
                be made.
            fname_fmt (str, optional): String format for the file name. Should
                be a callable that accepts two strings, the `ivu_name` and a
                string representation of the gap, as arguments and returns a
                string. Defaults to `None`. A value of `None` means
                    :func:`'{:s}_gap_{:s}.pickle'.format`.
            ivu_name (str, optional): Name of the IVU. Defaults to 'ivu18_08'.
            timeout_move_gap (float, optional): Timeout for waiting for the
                IVU to reach the desired gap. Defaults to 30 seconds.
        """
        fname_fmt = fname_fmt or '{:s}_gap_{:s}.pickle'.format
        if not callable(fname_fmt):
            raise TypeError('`fname_fmt` is not callable.')
        if ivu_name not in self.ivu_imp_meas.devices:
            raise ValueError(
                "Invalid IVU name. Valid names: 'ivu18_08', 'ivu18_14'."
            )
        if self._thread_loop is None or not self._thread_loop.is_alive():
            self._thread_loop = _Thread(
                target=self._loop_over_gaps_thread,
                args=(gaps, fname_fmt, ivu_name, timeout_move_gap),
                daemon=True,
            )
            self._thread_loop.start()
        else:
            raise RuntimeError('A measurement is already in progress.')

    def _loop_over_gaps_thread(
        self, gaps, fname_fmt, ivu_name, timeout_move_gap
    ):
        ivu = self.ivu_imp_meas.devices[ivu_name]
        sofb = self.devices['sofb']
        for i, gap in enumerate(gaps):
            print('#' * 70)
            print(f'{i + 1:02d}/{len(gaps):02d} Gap: {gap:.3f}')
            print('    Moving IVU...')
            ivu.set_gap(gap)
            ivu.cmd_move_start()
            ivu.wait_move_start(timeout=3)
            if not ivu.wait_move_finish(timeout=timeout_move_gap):
                print('Error: IVU did not reach the desired gap.')
                return
            print('    IVU reached the desired gap.')
            print('    Injecting ...')
            self.do_inject()
            print('    Done! Correcting Orbit...')
            sofb.correct_orbit_manually(nr_iters=5, residue=4)
            print('    Done! Starting measurement...')
            self.start()
            while True:
                if self.wait_measurement(0.5):
                    if self._stop_loop:
                        self._stop_loop = False
                        print('    User Interruption. Exiting.')
                        return
                    break
            print('    Done! Saving file ...')
            gap_stg = f'{gap:.3f}'.replace('.', 'p')
            self.save_data(fname_fmt(ivu_name, gap_stg), overwrite=True)
            print('#' * 70)

    # ---------------------- auxiliary methods ----------------------

    def _create_devices(self):
        super()._create_devices()
        self.devices['injctrl'] = InjCtrl()
        self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)

    def _do_meas(self, **bump):
        if self._stopevt.is_set():
            return dict()
        if abs(bump['psy']) <= self.params.max_bump_to_inject:
            print('Injecting...')
            if not self.do_inject():
                print('Injection Failed.')
            else:
                print('Injection Finished.')
        else:
            print('Bump too high. Skipping injection.')

        self.ivu_imp_meas.start()
        while True:
            if self._stopevt.is_set():
                self.ivu_imp_meas.stop()
            if self.ivu_imp_meas.wait_measurement(timeout=0.5):
                break
        return self.ivu_imp_meas.to_dict()

    def _call_meas_func(self, **bump):
        """."""
        return self.meas_func(**bump)

    def _call_and_wait(self, func, timeout=None):
        """."""
        t0_ = _time.time()
        while _time.time() - t0_ < timeout:
            if self._stopevt.is_set():
                return False
            if func(timeout=0.5):
                return True
        return False


class ImpedanceIVUMeasParams(_BaseParams):
    """."""

    ADC_NSAMPLES_PER_TURN = 382
    HARM_NUM = 864

    def __init__(self):
        """."""
        super().__init__()
        self.acq_strategy = 'odd/even'  # 'all' or 'odd/even'
        self.num_acquisitions = 10
        self.save_raw_data = False
        self.num_buckets_to_process = 2
        self._nrturns = 0
        self.nrturns = 1000
        self.bucket_hi_charge = 1
        self.bucket_lo_charge = 530
        self.acq_rate = 'ADCSwp'
        self.signals2acq = 'ABCD'
        self.timeout = 10
        self.do_pulse_evg = True
        self.event_mode = 'External'
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
        self._data_to_proc = _Queue()
        self._data_processed = []
        self._thread_proc_data = None

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
        data['bpm_indcs'] = _np.array([
            bns.index(b) for b in data['bpm_names']
        ])
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

    def process_data(self, idcs_to_discard=None):
        """."""
        idcs_to_discard = idcs_to_discard or []
        data = self.data
        if isinstance(data, dict):
            data = [data]
        for i, dt in enumerate(data):
            if i in idcs_to_discard:
                continue
            dic = self._proc_single_data(dt)
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
            orb1.append(_np.vstack([dt['b1_posx'], dt['b1_posy']]))
            orb2.append(_np.vstack([dt['b2_posx'], dt['b2_posy']]))
        orb1 = _np.array(orb1)
        orb2 = _np.array(orb2)
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
        curr1 = _np.array(curr1)
        curr2 = _np.array(curr2)
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
        sumt = _np.array(sumt)
        sum1 = _np.array(sum1)
        sum2 = _np.array(sum2)
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
            orb.append(_np.hstack([dt[prop + 'x'], dt[prop + 'y']]))
        return _np.vstack(orb).T

    # ---------------------- private methods ----------------------

    def _measure(self):
        fambpms = self.devices['fambpms']
        bpms = fambpms.bpms
        grp_slcs = [slice(None, None)]
        if self.params.acq_strategy.startswith('odd'):
            print(
                "Acquisition strategy 'odd/even' identified. "
                'Breaking BPMs in two groups.'
            )
            grp_slcs = [slice(0, None, 2), slice(1, None, 2)]

        self._data_processed = []
        self._thread_proc_data = _Thread(
            target=self._process_data_in_thread, daemon=True
        )
        self._thread_proc_data.start()

        try:
            for i in range(self.params.num_acquisitions):
                print(
                    f'Acquisition {i + 1:02d}/{self.params.num_acquisitions:02d}'
                )
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
                    self._data_to_proc.put(self.data)
        except Exception:
            fambpms.bpms = bpms
            print('Problem with acquisition. Interrupting!')
            return
        fambpms.bpms = bpms
        print('Acquisitions ended. Processing data...')
        self._data_to_proc.put(None)
        self._thread_proc_data.join()
        self.data = self._data_processed
        print('Finished!')

    def _process_data_in_thread(self):
        while not self._stopevt.is_set():
            dtm = self._data_to_proc.get()
            if not dtm:
                break
            prc = self._proc_single_data(dtm)
            self._filter_data_to_save(dtm)
            dtm.update(prc)
            self._data_processed.append(dtm)

    def _filter_data_to_save(self, dtm):
        if self.params.save_raw_data:
            return
        for ant in 'abcd':
            dtm.pop('ampl' + ant)

    def _proc_single_data(self, data):
        """."""
        of1 = self.BUN1_OFFSET
        winn, winp = self.WINDOW_RMS
        nbuc2proc = self.params.num_buckets_to_process
        bhigh = self.params.bucket_hi_charge
        blow = self.params.bucket_lo_charge
        nsamp_pturn = ImpedanceIVUMeasParams.ADC_NSAMPLES_PER_TURN
        hnum = ImpedanceIVUMeasParams.HARM_NUM

        of2 = int(abs(blow - bhigh) / hnum * nsamp_pturn) + of1
        slcs = [slice(of1 + winn, of1 + winp), slice(of2 + winn, of2 + winp)]

        raw_arrays = [data[f'ampl{ant}'] for ant in 'abcd']

        nsamp2keep, nbpms = raw_arrays[0].shape
        nturns, _ = divmod(nsamp2keep, nsamp_pturn)
        turn_start = _np.arange(nturns, dtype=_np.intp) * nsamp_pturn

        # Create the bases of the windows to select data.
        window_bases = []
        for slc in slcs[:nbuc2proc]:
            window_pos = _np.arange(*slc.indices(nsamp_pturn), dtype=_np.intp)
            window_bases.append(turn_start[:, None] + window_pos[None, :])

        bpm_idx = _np.arange(nbpms)[..., None, None]
        bucket_sigs = [[] for _ in range(nbuc2proc)]

        for raw in raw_arrays:
            # raw has shape (samples, BPMs).
            ant_amax = _np.abs(raw[:nsamp_pturn]).argmax(axis=0)
            shift = ant_amax.astype(_np.intp, copy=False) - of1

            for sigs, base in zip(bucket_sigs, window_bases):
                # Shape: (BPMs, turns, window_size).
                sample_idx = base[None, :, :] + shift[:, None, None]
                # in-place to avoid extra array of indices.
                _np.remainder(sample_idx, nsamp2keep, out=sample_idx)
                sigs.append(raw[sample_idx, bpm_idx].std(axis=-1))

        dic = {}
        bucket_sums = []
        for i, sigs in enumerate(bucket_sigs, start=1):
            # Shape: (4, BPMs, turns).
            b_sigs = _np.stack(sigs, axis=0)

            b_posx, b_posy = ImpedanceIVUMeas.calc_positions_from_amplitudes(
                b_sigs, is_adcswap_rate=True
            )
            b_sum = b_sigs.sum(axis=0)

            dic[f'b{i}_posx'] = b_posx
            dic[f'b{i}_posy'] = b_posy
            dic[f'b{i}_sum'] = b_sum
            dic[f'b{i}_sigs'] = b_sigs
            bucket_sums.append(b_sum)

        bt_sum = sum(bucket_sums)
        dic['bt_sum'] = bt_sum

        curr = data['stored_current']
        for i, b_sum in enumerate(bucket_sums, start=1):
            dic[f'b{i}_curr'] = b_sum * curr / bt_sum

        return dic


class ImpedanceIVUAnalysis:
    """Analyze IVU transverse impedance measurements."""

    _SUPPORTED_IVUS = ('IVU18_SI08', 'IVU18_SI14')
    _GAP_ATOL = 1e-3  # [mm]

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
        self.theta_fits = {'x': {}, 'y': {}}
        self.kperp_fits = {'x': {}, 'y': {}}

    # Accelerator-model utilities

    @staticmethod
    def get_spos_bpms():
        """Get the longitudinal positions of the BPMs in the Sirius model.

        Returns:
            spos_bpms (numpy.ndarray): BPM longitudinal positions [m].
        """
        model = _si.create_accelerator()

        spos = pyaccel.lattice.find_spos(model, indices='open')
        bpm_indices = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')

        spos_bpms = _np.asarray(spos[bpm_indices], dtype=float)

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
            model[idx].vkick_polynom = vkick / len(ivu_indices)

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

        if not _np.isfinite(delta_vkick) or delta_vkick == 0.0:
            raise ValueError('delta_vkick must be finite and nonzero.')

        model = _si.create_accelerator()

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

        respmat = _np.zeros(2 * n_bpms, dtype=float)

        orbit_difference = cod_pos[:, bpm_indices] - cod_neg[:, bpm_indices]

        respmat[:n_bpms] = orbit_difference[0] / delta_vkick
        respmat[n_bpms:] = orbit_difference[2] / delta_vkick

        info = {
            'model': model,
            'spos': spos,
            'spos_bpms': _np.asarray(spos[bpm_indices], dtype=float),
            'bpm_indices': _np.asarray(bpm_indices, dtype=int),
            'id_name': id_name,
            'ivu_indices': _np.asarray(ivu_indices, dtype=int),
            'ivu_indices_all': _np.asarray(ivu_indices_all, dtype=int),
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

    def load_data(self, file_paths, multi_bump=False, append=False):
        """Load measurement configurations from files.

        Args:
            file_paths (str or sequence of str): Measurement file path(s).
            multi_bump (bool): Whether each file contains multiple bump
                configurations under 'bumps_data'. Prescribed bump settings
                are stored in each configuration's params['bump_info'].
                Defaults to False.
            append (bool): Whether to retain previously loaded configurations.
                Defalts to False.

        Raises:
            TypeError: If file_paths has an unsupported type.
            ValueError: If no file paths or measurement configurations
                are provided.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not isinstance(file_paths, (list, tuple)):
            raise TypeError(
                'file_paths must be a path or a list or tuple of paths.'
            )

        if len(file_paths) == 0:
            raise ValueError('file_paths must contain at least one file path.')

        data = []

        for file_path in file_paths:
            loaded_data = _load(file_path)

            if multi_bump:
                for bump_i in loaded_data['data']['bumps_data']:
                    data_i = bump_i['meas_data']

                    data_i['params']['bump'] = bump_i['bump'].copy()

                    data.append(data_i)
            else:
                data.append(loaded_data)

        if append and self.data is not None:
            data = self.data + data

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
                _np.max(data_acq['bpm_indcs'])
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

                b1_sum_mean = _np.mean(data_acq['b1_sum'])
                b2_sum_mean = _np.mean(data_acq['b2_sum'])
                bt_sum_mean = b1_sum_mean + b2_sum_mean

                if _np.isclose(bt_sum_mean, 0.0):
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

            delta_charge_mean = _np.mean([
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
        sofb_refs = _np.asarray([
            data_i['data'][0]['sofb_refy'] for data_i in self.data
        ])

        ref_diff = sofb_refs - sofb_refs[0]

        ref_diff_mean = _np.mean(_np.abs(ref_diff), axis=0)

        bump_bpm_indcs = _np.argsort(ref_diff_mean)[-2:]

        # Keep the BPM indices in their global order.
        bump_bpm_indcs = _np.sort(bump_bpm_indcs)

        sofb_orbits = []

        for data_i in self.data:
            orbit_i = _np.mean(
                [
                    data_acq['sofb_orby'][bump_bpm_indcs]
                    for data_acq in data_i['data']
                ],
                axis=0,
            )

            sofb_orbits.append(orbit_i)

        sofb_orbits = _np.asarray(sofb_orbits)

        reference_cfg = _np.argmin(_np.abs(_np.mean(sofb_orbits, axis=1)))

        bump_orbits = sofb_orbits - sofb_orbits[reference_cfg]

        # SOFB orbit values are in micrometers; y0 is stored in millimeters.
        y0 = _np.mean(bump_orbits, axis=1) * 1e-3

        # Before assigning the inferred values, check if any y0 values already
        # exist in the data.

        has_existing_y0 = any('y0' in data_i['params'] for data_i in self.data)

        if has_existing_y0:
            _warnings.warn(
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
        array1 = _np.asarray(array1)
        array2 = _np.asarray(array2)
        bpm_indcs1 = _np.asarray(bpm_indcs1, dtype=int)
        bpm_indcs2 = _np.asarray(bpm_indcs2, dtype=int)

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

        bpm_indcs = _np.concatenate([bpm_indcs1, bpm_indcs2])

        if len(_np.unique(bpm_indcs)) != len(bpm_indcs):
            raise ValueError(
                'The BPM groups contain repeated global BPM indices.'
            )

        joined_array = _np.concatenate([array1, array2], axis=0)

        order = _np.argsort(bpm_indcs)
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

                bpm_indcs1 = _np.asarray(acq1['bpm_indcs'])
                bpm_indcs2 = _np.asarray(acq2['bpm_indcs'])

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

                joined_acq['bpm_indcs'] = _np.sort(
                    _np.concatenate([bpm_indcs1, bpm_indcs2])
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
            _warnings.warn(
                f'Number of acquisitions mismatch: '
                f'calculated={n_acq}, '
                f'expected={expected_n_acq}',
                stacklevel=2,
            )

        if n_bpms != indexed_n_bpms:
            _warnings.warn(
                f'BPM number mismatch: '
                f'calculated={n_bpms}, '
                f'indexed={indexed_n_bpms}',
                stacklevel=2,
            )

        if n_turns != expected_n_turns:
            _warnings.warn(
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

            b1_posy = _np.asarray(first_acq['b1_posy'])

            if b1_posy.ndim != 2:
                raise ValueError(
                    f"Configuration {i}: 'b1_posy' must have shape "
                    '(n_bpms, n_turns).'
                )

            n_bpms, n_turns = b1_posy.shape

            self._check_consistency(data_i, n_acq, n_bpms, n_turns)

            gap = round(
                _np.mean([data_acq[gap_key] for data_acq in data_meas]), 3
            )

            b1_curr = _np.mean([data_acq['b1_curr'] for data_acq in data_meas])

            b2_curr = _np.mean([data_acq['b2_curr'] for data_acq in data_meas])

            delta_curr = _np.mean([
                data_acq['delta_curr'] for data_acq in data_meas
            ])

            delta_charge = _np.mean([
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

        return _pd.DataFrame(rows)

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

            bpm_indcs_ref = _np.asarray(data_meas[0]['bpm_indcs'], dtype=int)

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

                bpm_indcs = _np.asarray(data_acq['bpm_indcs'], dtype=int)

                if not _np.array_equal(bpm_indcs, bpm_indcs_ref):
                    raise ValueError(
                        f'Configuration {i}, acquisition {acq}: '
                        'BPM indices differ from the first acquisition.'
                    )

                b1_pos = _np.asarray(data_acq[key1])
                b2_pos = _np.asarray(data_acq[key2])

                if b1_pos.shape != b2_pos.shape:
                    raise ValueError(
                        f'Configuration {i}, acquisition {acq}: '
                        'bunch-position arrays must have the same shape, '
                        f'but {key1} has shape {b1_pos.shape} and '
                        f'{key2} has shape {b2_pos.shape}.'
                    )

                delta_i.append(b1_pos - b2_pos)

            delta_orbit.append(_np.asarray(delta_i))

        self.delta_orbit[plane] = delta_orbit
        self.delta_orbit_stats[plane] = None
        self.theta_fits[plane] = {}
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

            mean_i = _np.mean(delta_i, axis=(0, 2))

            std_i = _np.std(delta_i, axis=(0, 2))

            sem_i = std_i / _np.sqrt(n_samples_i)

            mean.append(mean_i)
            std.append(std_i)
            sem.append(sem_i)
            n_samples.append(n_samples_i)

        stats = {
            'mean': mean,
            'std': std,
            'sem': sem,
            'n_samples': _np.asarray(n_samples, dtype=int),
        }

        self.delta_orbit_stats[plane] = stats
        self.theta_fits[plane] = {}
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

        if not _np.all(_np.isfinite(delta_charge)):
            raise ValueError(
                'Charge differences must contain only finite values.'
            )

        zero_charge = _np.isclose(
            delta_charge, 0.0, rtol=0.0, atol=_np.finfo(float).eps
        )

        if _np.any(zero_charge):
            invalid_i = _np.where(zero_charge)[0]
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
                std_i / _np.abs(delta_charge[i])
                for i, std_i in enumerate(stats['std'])
            ],
            'sem': [
                sem_i / _np.abs(delta_charge[i])
                for i, sem_i in enumerate(stats['sem'])
            ],
            'n_samples': stats['n_samples'].copy(),
        }

        return stats_norm

    def plot_currents_vs_acquisition(self, cfgs=None, axes=None):
        """Plot both mean bunch currents versus acquisition index.

        Args:
            cfgs (int or sequence of int, optional): Configuration indices.
                If None, all configurations are plotted.
            axes (sequence of matplotlib.axes.Axes, optional): Two axes
                in which to draw. If None, a new figure is created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            axes (numpy.ndarray): Axes for bunches 1 and 2, respectively.

        Raises:
            RuntimeError: If measurement data have not been prepared.
            IndexError: If a configuration index is outside the valid range.
        """
        if self.config_table is None:
            raise RuntimeError('Prepare data before plotting bunch currents.')

        n_configs = len(self.data)

        if cfgs is None:
            cfgs = range(n_configs)
        elif _np.isscalar(cfgs):
            cfgs = [cfgs]
        else:
            cfgs = list(cfgs)

        invalid_cfgs = [i for i in cfgs if not 0 <= i < n_configs]

        if invalid_cfgs:
            raise IndexError(
                'Configuration indices outside the valid range: '
                f'{invalid_cfgs}.'
            )

        if axes is None:
            fig, axes = _plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        else:
            axes = _np.asarray(axes)
            fig = axes[0].figure

        for i in cfgs:
            row = self.config_table.iloc[i]
            data_meas = self.data[i]['data']

            acq_indices = _np.arange(len(data_meas))

            label = (
                f'gap = {row["gap"]:.2f} mm, '
                rf'$y_0$ = {row["y0"]:.2f} mm'
            )

            color = None

            for bunch, ax in enumerate(axes, start=1):
                currents = _np.asarray([
                    data_acq[f'b{bunch}_curr'] for data_acq in data_meas
                ])

                (line,) = ax.plot(
                    acq_indices, currents, '.-', color=color, label=label
                )

                # Use the same configuration color in both panels.
                color = line.get_color()

        for bunch, ax in enumerate(axes, start=1):
            ax.set_title(f'Bunch {bunch}')
            ax.set_xlabel('Acquisition index')
            ax.set_ylabel('Current [mA]')
            ax.grid(True)

        fig.suptitle('Mean bunch currents per acquisition')

        axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        return fig, axes

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
        elif _np.isscalar(acqs):
            acqs = [int(acqs)]
        else:
            acqs = list(acqs)

        invalid_acqs = [acq for acq in acqs if not 0 <= acq < n_acq]

        if invalid_acqs:
            raise IndexError(
                f'Acquisition indices outside the valid range: {invalid_acqs}.'
            )

        bpm_indcs = _np.asarray(
            self.data[cfg]['data'][0]['bpm_indcs'], dtype=int
        )
        spos_i = _np.asarray(self.spos_bpms)[bpm_indcs]

        mean_by_acq = _np.mean(delta_i, axis=2)

        if error == 'std':
            error_by_acq = _np.std(delta_i, axis=2)
        elif error == 'sem':
            error_by_acq = _np.std(delta_i, axis=2) / _np.sqrt(
                delta_i.shape[2]
            )
        else:
            error_by_acq = None

        if ax is None:
            fig, ax = _plt.subplots(figsize=(10, 5))
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
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()

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
        elif _np.isscalar(cfgs):
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
            fig, ax = _plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        if ref_cfg is not None:
            ref_bpm_indcs = _np.asarray(
                self.data[ref_cfg]['data'][0]['bpm_indcs'], dtype=int
            )

            ref_bpm_map = {
                bpm_idx: local_idx
                for local_idx, bpm_idx in enumerate(ref_bpm_indcs)
            }

            mean_ref_full = stats['mean'][ref_cfg]

            if error is not None:
                error_ref_full = stats[error][ref_cfg]

        spos_bpms = _np.asarray(self.spos_bpms)

        for i in cfgs:
            bpm_indcs = _np.asarray(
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

                ref_local_indcs = _np.asarray([
                    ref_bpm_map[bpm_idx] for bpm_idx in bpm_indcs
                ])

                mean_ref = mean_ref_full[ref_local_indcs]
                mean_plot = mean_i - mean_ref

                if error is not None:
                    error_ref = error_ref_full[ref_local_indcs]

                    if i == ref_cfg:
                        error_plot = _np.zeros_like(error_i)
                    else:
                        error_plot = _np.sqrt(error_i**2 + error_ref**2)
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
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        fig.tight_layout()

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
        self.theta_fits = {'x': {}, 'y': {}}
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
            bpm_indcs = _np.asarray(data_i['data'][0]['bpm_indcs'], dtype=int)

            m_ivu.append(m_full[bpm_indcs])

        self.M_ivu[plane] = m_ivu
        self.theta_fits[plane] = {}
        self.kperp_fits[plane] = {}

        return m_ivu

    def _prepare_gap_fit_data(
        self, gap, plane='y', error='sem', config_indices=None, gap_atol=None
    ):
        """Prepare charge-normalized orbit data for one gap scan.

        This method selects the desired configurations and organizes the
        corresponding orbit means, uncertainties, IVU response, bump values
        and charge differences into arrays used by the fitting methods.

        Args:
            gap (float): IVU gap [mm].
            plane (str): Transverse plane, either 'x' or 'y'.
            error (str): Orbit uncertainty, either 'std' or 'sem'.
            config_indices (sequence of int, optional): Explicit configuration
                indices belonging to the scan. If None, configurations are
                selected according to the requested gap.
            gap_atol (float): Absolute tolerance used to select configurations
                by gap [mm].

        Returns:
            fit_data (dict): Data and metadata needed by the free-kick and
                kick-factor fits.

        Raises:
            RuntimeError: If required analysis results are unavailable.
            ValueError: If the plane or error type is invalid, fewer than two
                configurations are selected, or arrays contain invalid values.
            IndexError: If a configuration index is outside the valid range.
        """
        if gap_atol is None:
            gap_atol = self._GAP_ATOL

        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if error not in ('std', 'sem'):
            raise ValueError("error must be 'std' or 'sem'.")

        if self.config_table is None:
            raise RuntimeError('Prepare the measurement data first.')

        if self.delta_orbit_stats[plane] is None:
            raise RuntimeError(
                f'Calculate the {plane}-plane orbit statistics first.'
            )

        if self.M_ivu[plane] is None:
            raise RuntimeError(
                f'Set the {plane}-plane IVU response matrix first.'
            )

        n_configs = len(self.config_table)

        # -------------------------------------------------------------
        # Select the configurations included in this fit.
        #
        # If explicit indices are passed, they define the scan.
        # Otherwise, all configurations at the requested gap are used.
        # -------------------------------------------------------------
        if config_indices is None:
            gap_values = self.config_table['gap'].to_numpy(dtype=float)

            config_indices = _np.flatnonzero(
                _np.isclose(gap_values, float(gap), rtol=0.0, atol=gap_atol)
            )
        else:
            config_indices = _np.asarray(config_indices, dtype=int)

        if len(config_indices) < 2:
            raise ValueError(
                'At least two configurations are required for the fit.'
            )

        invalid_indices = config_indices[
            (config_indices < 0) | (config_indices >= n_configs)
        ]

        if len(invalid_indices) > 0:
            raise IndexError(
                'Configuration indices outside the valid range: '
                f'{invalid_indices.tolist()}.'
            )

        # -------------------------------------------------------------
        # Obtain charge-normalized orbit statistics.
        #
        # The means and uncertainties returned here already have units:
        #
        #     orbit       -> um/nC
        #     orbit_error -> um/nC
        # -------------------------------------------------------------
        stats = self.get_delta_orbit_stats(plane=plane, normalized=True)

        orbit = _np.stack([stats['mean'][cfg] for cfg in config_indices])

        orbit_error = _np.stack([stats[error][cfg] for cfg in config_indices])

        # All configurations use the same BPMs and therefore the same
        # IVU response vector. We can take it from the first selected
        # configuration.
        first_cfg = config_indices[0]

        response = _np.asarray(self.M_ivu[plane][first_cfg], dtype=float)

        bpm_indices = _np.asarray(
            self.data[first_cfg]['data'][0]['bpm_indcs'], dtype=int
        )

        # np.stack already guarantees that all selected orbit arrays
        # have the same shape. We only need to confirm compatibility
        # between the orbit and response dimensions.
        if orbit.shape[1] != len(response):
            raise ValueError(
                'The orbit data and IVU response contain different '
                'numbers of BPMs.'
            )

        if len(bpm_indices) != len(response):
            raise ValueError(
                'The BPM-index array and IVU response contain different '
                'numbers of BPMs.'
            )

        # -------------------------------------------------------------
        # Validate numerical values.
        # -------------------------------------------------------------
        if not _np.all(_np.isfinite(orbit)):
            raise ValueError('Orbit means must contain only finite values.')

        if not _np.all(_np.isfinite(orbit_error)) or _np.any(
            orbit_error <= 0.0
        ):
            raise ValueError(
                'Orbit uncertainties must be finite and positive.'
            )

        if not _np.all(_np.isfinite(response)):
            raise ValueError(
                'The IVU response must contain only finite values.'
            )

        if _np.allclose(response, 0.0):
            raise ValueError('The IVU response cannot be identically zero.')

        # -------------------------------------------------------------
        # Configuration-level quantities.
        # -------------------------------------------------------------
        selected_rows = self.config_table.iloc[config_indices]

        y0 = selected_rows['y0'].to_numpy(dtype=float)

        delta_charge = selected_rows['delta_charge'].to_numpy(dtype=float)

        if not _np.all(_np.isfinite(y0)):
            raise ValueError('Bump values must contain only finite values.')

        if not _np.all(_np.isfinite(delta_charge)):
            raise ValueError(
                'Charge differences must contain only finite values.'
            )

        return {
            'gap': float(gap),
            'plane': plane,
            'error': error,
            'config_indices': config_indices,
            'bpm_indices': bpm_indices,
            'response': response,
            'orbit': orbit,
            'orbit_error': orbit_error,
            'y0': y0,
            'delta_charge': delta_charge,
            'n_bumps': len(config_indices),
            'n_bpms': len(response),
        }

    def fit_relative_kicks(
        self, gap, plane='y', error='sem', config_indices=None, gap_atol=None
    ):
        """Fit the relative charge-normalized IVU kicks for one gap scan.

        The charge-normalized orbit model is

            y_i = M*theta_i + background + epsilon_i,

        with the exact identification constraint

            sum(theta_i) = 0.

        The common BPM background and all relative kicks are estimated
        simultaneously through equality-constrained weighted least squares.

        Args:
            gap (float): IVU gap [mm].
            plane (str): Transverse plane, either 'x' or 'y'.
            error (str): Orbit uncertainty, either 'std' or 'sem'.
            config_indices (sequence of int, optional): Explicit configuration
                indices belonging to the scan. If None, select by gap.
            gap_atol (float): Absolute tolerance for gap selection [mm].

        Returns:
            kick_fit (dict): Relative kicks, background, covariances, residuals
                and goodness-of-fit quantities.
        """
        if gap_atol is None:
            gap_atol = self._GAP_ATOL

        fit_data = self._prepare_gap_fit_data(
            gap=gap,
            plane=plane,
            error=error,
            config_indices=config_indices,
            gap_atol=gap_atol,
        )

        orbit = fit_data['orbit']
        orbit_error = fit_data['orbit_error']
        response = fit_data['response']

        n_bumps = fit_data['n_bumps']
        n_bpms = fit_data['n_bpms']

        # -------------------------------------------------------------
        # Stack the orbit observations:
        #
        # y = [y_1, y_2, ..., y_N]
        #
        # Each y_i contains the n_bpms orbit values for bump i.
        # -------------------------------------------------------------
        values = orbit.ravel()
        sigma = orbit_error.ravel()
        weights = 1.0 / sigma**2

        # -------------------------------------------------------------
        # Construct the kick block:
        #
        # [ M  0  0 ... ]
        # [ 0  M  0 ... ]
        # [ 0  0  M ... ]
        #
        # Shape: (n_bumps*n_bpms, n_bumps)
        # -------------------------------------------------------------
        kick_block = _np.kron(_np.eye(n_bumps), response[:, None])

        # -------------------------------------------------------------
        # Construct the common-background block:
        #
        # [ I ]
        # [ I ]
        # [ I ]
        #
        # The same BPM-resolved background is present at every bump.
        #
        # Shape: (n_bumps*n_bpms, n_bpms)
        # -------------------------------------------------------------
        background_block = _np.tile(_np.eye(n_bpms), (n_bumps, 1))

        # Parameter order:
        #
        # p = [theta_1, ..., theta_N, b_1, ..., b_B]
        design_matrix = _np.column_stack([kick_block, background_block])

        n_parameters = design_matrix.shape[1]

        # -------------------------------------------------------------
        # Exact identification constraint:
        #
        # theta_1 + ... + theta_N = 0
        # -------------------------------------------------------------
        constraint = _np.zeros((1, n_parameters))
        constraint[0, :n_bumps] = 1.0

        # Weighted normal equations, without explicitly creating the
        # full diagonal weight matrix.
        normal_matrix = design_matrix.T @ (weights[:, None] * design_matrix)

        normal_vector = design_matrix.T @ (weights * values)

        # -------------------------------------------------------------
        # KKT system for equality-constrained least squares:
        #
        # [ A.T W A   L.T ] [p     ] = [A.T W y]
        # [ L          0   ] [lambda]   [   0   ]
        # -------------------------------------------------------------
        kkt_matrix = _np.block([
            [normal_matrix, constraint.T],
            [constraint, _np.zeros((1, 1))],
        ])

        kkt_rhs = _np.concatenate([normal_vector, _np.zeros(1)])

        solution = _np.linalg.solve(kkt_matrix, kkt_rhs)

        parameters = solution[:-1]
        lagrange_multiplier = solution[-1]

        theta = parameters[:n_bumps]
        background = parameters[n_bumps:]

        if not _np.isclose(_np.sum(theta), 0.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                'The fitted kicks do not satisfy the sum-zero constraint.'
            )

        # -------------------------------------------------------------
        # Fitted orbit model and residuals.
        # -------------------------------------------------------------
        model = (design_matrix @ parameters).reshape(n_bumps, n_bpms)

        residual = orbit - model

        residual_rms = _np.sqrt(_np.mean(residual**2, axis=1))

        # Individual chi-square contribution from each bump.
        chi2_by_bump = _np.sum((residual / orbit_error) ** 2, axis=1)

        chi2 = float(_np.sum(chi2_by_bump))

        # Independent parameter count:
        #
        # n_bumps kicks + n_bpms backgrounds - one exact constraint.
        n_independent_parameters = n_bumps + n_bpms - 1

        dof = n_bumps * n_bpms - n_independent_parameters

        reduced_chi2 = chi2 / dof if dof > 0 else _np.nan

        # Diagnostic, uniform allocation of the global degrees of freedom
        # among the bump configurations.
        dof_per_bump = dof / n_bumps

        reduced_chi2_by_bump = chi2_by_bump / dof_per_bump

        # -------------------------------------------------------------
        # Formal parameter covariance.
        #
        # For this equality-constrained problem, the upper-left block
        # of KKT^{-1} is the parameter covariance when W is the inverse
        # covariance of the observations.
        # -------------------------------------------------------------
        kkt_inverse = _np.linalg.inv(kkt_matrix)

        covariance_formal = kkt_inverse[:n_parameters, :n_parameters]

        theta_covariance_formal = covariance_formal[:n_bumps, :n_bumps]

        theta_err_formal = _np.sqrt(
            _np.maximum(0.0, _np.diag(theta_covariance_formal))
        )

        # -------------------------------------------------------------
        # Diagnostic per-bump residual inflation policy.
        #
        # This is intentionally distinguished from the formal covariance.
        # Each bump's marginal theta uncertainty is inflated according to
        # how well its orbit is represented by the common-background model.
        # -------------------------------------------------------------
        inflation_by_bump = _np.sqrt(_np.maximum(1.0, reduced_chi2_by_bump))

        theta_err = theta_err_formal * inflation_by_bump

        # Preserve the formal correlation coefficients while replacing
        # the marginal standard deviations by the inflated values.
        #
        # C_corrected = D * C_formal * D,
        # where D contains the individual inflation factors.
        inflation_matrix = _np.diag(inflation_by_bump)

        theta_covariance = (
            inflation_matrix @ theta_covariance_formal @ inflation_matrix
        )

        kick_fit = {
            'gap': fit_data['gap'],
            'plane': plane,
            'error': error,
            'config_indices': fit_data['config_indices'],
            'bpm_indices': fit_data['bpm_indices'],
            'y0': fit_data['y0'],
            'delta_charge': fit_data['delta_charge'],
            'response': response,
            'orbit': orbit,
            'orbit_error': orbit_error,
            # Relative charge-normalized kicks [urad/nC].
            'theta': theta,
            'theta_err': theta_err,
            'theta_err_formal': theta_err_formal,
            'theta_covariance': theta_covariance,
            'theta_covariance_formal': theta_covariance_formal,
            # Charge-normalized BPM background [um/nC].
            'background': background,
            # Orbit model and residuals [um/nC].
            'model': model,
            'residual': residual,
            'residual_rms': residual_rms,
            # Goodness-of-fit information.
            'chi2_by_bump': chi2_by_bump,
            'dof_per_bump': dof_per_bump,
            'reduced_chi2_by_bump': reduced_chi2_by_bump,
            'inflation_by_bump': inflation_by_bump,
            'chi2': chi2,
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            # Numerical/constraint information.
            'lagrange_multiplier': lagrange_multiplier,
            'constraint_residual': float(_np.sum(theta)),
        }

        gap_key = round(float(gap), 3)

        self.theta_fits[plane][gap_key] = kick_fit

        # The direct fit for this same gap must be recalculated.
        self.kperp_fits[plane].pop(gap_key, None)

        return kick_fit

    def fit_kperp(
        self, gap, plane='y', error='sem', config_indices=None, gap_atol=None
    ):
        """Fit the transverse kick factor directly from normalized orbit data.

        The fitted model is

            y_i = M*a*y0_i + background + epsilon_i,

        where a is the charge-normalized kick slope versus vertical bump.

        Args:
            gap (float): IVU gap [mm].
            plane (str): Transverse plane, either 'x' or 'y'.
            error (str): Orbit uncertainty, either 'std' or 'sem'.
            config_indices (sequence of int, optional): Explicit configuration
                indices belonging to the scan.
            gap_atol (float): Absolute tolerance for gap selection [mm].

        Returns:
            kperp_fit (dict): Direct slope and transverse kick-factor results.
        """
        if gap_atol is None:
            gap_atol = self._GAP_ATOL

        fit_data = self._prepare_gap_fit_data(
            gap=gap,
            plane=plane,
            error=error,
            config_indices=config_indices,
            gap_atol=gap_atol,
        )

        orbit = fit_data['orbit']
        orbit_error = fit_data['orbit_error']
        response = fit_data['response']
        y0 = fit_data['y0']

        n_bumps = fit_data['n_bumps']
        n_bpms = fit_data['n_bpms']

        if _np.allclose(y0, y0[0], rtol=0.0, atol=_np.finfo(float).eps):
            raise ValueError(
                'The bump values must vary to determine the kick-factor slope.'
            )

        values = orbit.ravel()
        sigma = orbit_error.ravel()
        weights = 1.0 / sigma**2

        # -------------------------------------------------------------
        # First design-matrix column:
        #
        # [ y0_1*M ]
        # [ y0_2*M ]
        # [   ...  ]
        #
        # Its coefficient is the slope a.
        # -------------------------------------------------------------
        slope_column = (y0[:, None] * response[None, :]).ravel()

        # Common BPM-resolved background.
        background_block = _np.tile(_np.eye(n_bpms), (n_bumps, 1))

        # Parameter order:
        #
        # p = [a, b_1, ..., b_B]
        design_matrix = _np.column_stack([slope_column, background_block])

        normal_matrix = design_matrix.T @ (weights[:, None] * design_matrix)

        normal_vector = design_matrix.T @ (weights * values)

        parameters = _np.linalg.solve(normal_matrix, normal_vector)

        a = parameters[0]
        background = parameters[1:]

        model = (design_matrix @ parameters).reshape(n_bumps, n_bpms)

        residual = orbit - model

        residual_rms = _np.sqrt(_np.mean(residual**2, axis=1))

        chi2_by_bump = _np.sum((residual / orbit_error) ** 2, axis=1)

        chi2 = float(_np.sum(chi2_by_bump))

        # Number of independent parameters:
        #
        # one slope + n_bpms background values.
        n_parameters = n_bpms + 1

        dof = n_bumps * n_bpms - n_parameters

        reduced_chi2 = chi2 / dof if dof > 0 else _np.nan

        # Formal covariance of [a, background].
        covariance_formal = _np.linalg.inv(normal_matrix)

        a_err_formal = _np.sqrt(max(0.0, covariance_formal[0, 0]))

        # Global residual inflation for the final physical slope.
        inflation = _np.sqrt(max(1.0, reduced_chi2)) if dof > 0 else 1.0

        covariance = inflation**2 * covariance_formal

        a_err = a_err_formal * inflation

        kperp = 1e-6 * self.beam_voltage * a

        kperp_err_formal = 1e-6 * self.beam_voltage * a_err_formal

        kperp_err = 1e-6 * self.beam_voltage * a_err

        # The line represented in the mean-zero convention used by
        # fit_relative_kicks().
        y0_centered = y0 - _np.mean(y0)

        theta_relative_model = a * y0_centered

        kperp_fit = {
            'gap': fit_data['gap'],
            'plane': plane,
            'error': error,
            'config_indices': fit_data['config_indices'],
            'bpm_indices': fit_data['bpm_indices'],
            'y0': y0,
            'y0_centered': y0_centered,
            'delta_charge': fit_data['delta_charge'],
            'response': response,
            'orbit': orbit,
            'orbit_error': orbit_error,
            # Slope [urad/(nC mm)].
            'a': a,
            'a_err': a_err,
            'a_err_formal': a_err_formal,
            # Relative kick line [urad/nC].
            'theta_relative_model': theta_relative_model,
            # Charge-normalized background [um/nC].
            'background': background,
            # Model and residuals [um/nC].
            'model': model,
            'residual': residual,
            'residual_rms': residual_rms,
            # Complete covariance of [a, background].
            'covariance': covariance,
            'covariance_formal': covariance_formal,
            # Goodness of fit.
            'chi2_by_bump': chi2_by_bump,
            'chi2': chi2,
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            'inflation': inflation,
            # Kick factor [V/(pC m)].
            'kperp': kperp,
            'kperp_err': kperp_err,
            'kperp_err_formal': kperp_err_formal,
        }

        gap_key = round(float(gap), 3)

        self.kperp_fits[plane][float(gap_key)] = kperp_fit

        return kperp_fit

    def get_kperp_results_table(self, plane='y'):
        """Create a summary table of the fitted transverse kick factors.

        The table includes the kick factor and slope uncertainties both
        before and after the global residual-based inflation applied by
        fit_kperp().

        Args:
            plane (str): Transverse plane, either 'x' or 'y'.

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

            kperp = fit['kperp']
            kperp_err = fit['kperp_err']

            relative_error = (
                _np.abs(kperp_err / kperp)
                if not _np.isclose(kperp, 0.0)
                else _np.nan
            )

            rows.append({
                'gap [mm]': fit['gap'],
                'kperp [V/(pC m)]': kperp,
                'kperp error [V/(pC m)]': kperp_err,
                'kperp formal error [V/(pC m)]': (fit['kperp_err_formal']),
                'relative error [%]': (100.0 * relative_error),
                'slope [urad/(nC mm)]': fit['a'],
                'slope error [urad/(nC mm)]': fit['a_err'],
                'slope formal error [urad/(nC mm)]': (fit['a_err_formal']),
                'error inflation factor': fit['inflation'],
                'chi2': fit['chi2'],
                'reduced chi2': fit['reduced_chi2'],
                'dof': fit['dof'],
                'n_bumps': len(fit['config_indices']),
                'n_bpms': len(fit['bpm_indices']),
            })

        return _pd.DataFrame(rows)

    def plot_theta_fit(self, gap, plane='y', ax=None):
        """Plot relative kicks and the fitted kick-factor model.

        The points are the relative charge-normalized kicks obtained from
        fit_relative_kicks(). Their sum is constrained to zero.

        The line is obtained from the direct background-plus-slope fit:

            theta_model = a * (y0 - mean(y0)).

        Args:
            gap (float): IVU gap corresponding to the fits [mm].
            plane (str): Transverse plane, either 'x' or 'y'.
            ax (matplotlib.axes.Axes, optional): Axes in which to draw.
                If None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            ValueError: If plane is invalid, the selected fits correspond
                to different configurations, or multiple kick-factor fits
                match the requested gap.
            RuntimeError: If either the relative-kick fit or the direct
                kick-factor fit is unavailable.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        gap_key = round(float(gap), 3)

        if gap_key not in self.theta_fits[plane]:
            raise RuntimeError(
                f'Fit the {plane}-plane relative kicks at gap '
                f'{gap_key:.3f} mm first.'
            )

        if gap_key not in self.kperp_fits[plane]:
            raise RuntimeError(
                f'Fit the {plane}-plane kick factor at gap '
                f'{gap_key:.3f} mm first.'
            )

        kick_fit = self.theta_fits[plane][gap_key]
        kperp_fit = self.kperp_fits[plane][gap_key]

        if not _np.array_equal(
            kick_fit['config_indices'], kperp_fit['config_indices']
        ):
            raise ValueError(
                'The relative-kick fit and the kick-factor fit use '
                'different configurations.'
            )

        y0 = _np.asarray(kick_fit['y0'], dtype=float)
        theta = _np.asarray(kick_fit['theta'], dtype=float)
        theta_err = _np.asarray(kick_fit['theta_err'], dtype=float)

        # The relative kicks obey sum(theta_i) = 0.
        y0_mean = _np.mean(y0)

        y0_fit = _np.linspace(_np.min(y0), _np.max(y0), 200)

        y0_fit_centered = y0_fit - y0_mean

        a = kperp_fit['a']
        a_err = kperp_fit['a_err']

        theta_model = a * y0_fit_centered

        # Uncertainty of the line due only to the slope uncertainty.
        theta_model_err = _np.abs(y0_fit_centered) * a_err

        if ax is None:
            fig, ax = _plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.errorbar(
            y0,
            theta,
            yerr=theta_err,
            marker='o',
            linestyle='none',
            markersize=6,
            capsize=4,
            label='Relative kick estimates',
        )

        ax.plot(
            y0_fit,
            theta_model,
            '-',
            linewidth=1.8,
            label='Direct background-plus-slope fit',
        )

        ax.fill_between(
            y0_fit,
            theta_model - theta_model_err,
            theta_model + theta_model_err,
            alpha=0.2,
            label=r'Fit uncertainty ($1\sigma$)',
        )

        theta_mean = _np.mean(theta)

        ax.axhline(
            theta_mean,
            color='black',
            linestyle='--',
            linewidth=1.0,
            label=(
                rf'Mean kick: '
                rf'$\overline{{\theta}}={theta_mean:.2e}$ '
                r'$\mu$rad/nC'
            ),
        )

        ax.set_title(
            rf'Relative projected $\hat{{\theta}}_{plane}$ versus $y_0$ '
            f'({self.id_name})\n'
            f'Gap = {kperp_fit["gap"]:.2f} mm, '
            rf'$k_\perp$ = {kperp_fit["kperp"]:.2f}'
            rf'$\pm$ {kperp_fit["kperp_err"]:.2f} V/(pC m)'
        )

        ax.set_xlabel(r'$y_0$[mm]')
        ax.set_ylabel(
            rf'Relative $\hat{{\theta}}_{plane}$ '
            r'[$\mu$rad/nC]'
        )

        ax.grid(True)
        ax.legend()

        return fig, ax

    def plot_ivu_projection(
        self,
        gap,
        bump,
        ref_bump,
        plane='y',
        normalized=True,
        model='global',
        error='sem',
        bump_atol=0.05,
        ax=None,
    ):
        """Plot the reference-subtracted orbit and fitted IVU model.

        The plotted and reference configurations are selected by the requested
        IVU gap and their measured bump values.

        The measured orbit can be compared with either:

            - "individual": the difference between the relative kicks obtained
            from fit_relative_kicks();
            - "global": the kick difference predicted by the direct
            kick-factor fit.

        Args:
            gap (float): IVU gap [mm].
            bump (float): Measured bump value of the configuration to plot
                [mm].
            ref_bump (float): Measured bump value of the reference
                configuration [mm].
            plane (str): Transverse plane, either "x" or "y".
            normalized (bool): Whether to plot charge-normalized orbit
                differences. If True, orbit units are micrometers per  nanocoulomb. Otherwise, orbit units are micrometers.
            model (str): Model used to predict the IVU orbit. Use
                "individual" or "global".
            error (str): Error bars to display, either "std" or "sem".
            bump_atol (float): Absolute tolerance used to associate the
                requested bump values with the measured bump values [mm].
            ax (matplotlib.axes.Axes, optional): Axes in which to draw. If
                None, a new figure and axes are created.

        Returns:
            fig (matplotlib.figure.Figure): Plot figure.
            ax (matplotlib.axes.Axes): Plot axes.

        Raises:
            RuntimeError: If orbit statistics, IVU response data or the
                requested fit are unavailable.
            ValueError: If plane, model, or error is invalid, or a unique
                configuration cannot be identified for one of the requested bump values.
        """
        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        if model not in ('individual', 'global'):
            raise ValueError("model must be 'individual' or 'global'.")

        if error not in ('std', 'sem'):
            raise ValueError("error must be 'std' or 'sem'.")

        gap_key = round(float(gap), 3)
        bump = float(bump)
        ref_bump = float(ref_bump)

        # -------------------------------------------------------------
        # Select all configurations belonging to the requested gap.
        # -------------------------------------------------------------
        gap_values = self.config_table['gap'].to_numpy(dtype=float)
        y0_values = self.config_table['y0'].to_numpy(dtype=float)

        gap_indices = _np.flatnonzero(
            _np.isclose(gap_values, gap_key, rtol=0.0, atol=self._GAP_ATOL)
        )

        if len(gap_indices) == 0:
            raise ValueError(
                f'No configurations were found at gap {gap:.3f} mm.'
            )

        # -------------------------------------------------------------
        # Select the plotted and reference configurations from their
        # measured bump values.
        # -------------------------------------------------------------
        cfg_matches = gap_indices[
            _np.isclose(y0_values[gap_indices], bump, rtol=0.0, atol=bump_atol)
        ]

        ref_matches = gap_indices[
            _np.isclose(
                y0_values[gap_indices], ref_bump, rtol=0.0, atol=bump_atol
            )
        ]

        if len(cfg_matches) != 1:
            raise ValueError(
                f'Expected one configuration near y0={bump:.3f} mm '
                f'at gap {gap:.3f} mm, but found {len(cfg_matches)}.'
            )

        if len(ref_matches) != 1:
            raise ValueError(
                f'Expected one reference configuration near '
                f'y0={ref_bump:.3f} mm at gap {gap:.3f} mm, '
                f'but found {len(ref_matches)}.'
            )

        cfg = int(cfg_matches[0])
        ref_cfg = int(ref_matches[0])

        row = self.config_table.iloc[cfg]
        row_ref = self.config_table.iloc[ref_cfg]

        measured_bump = float(row['y0'])
        measured_ref_bump = float(row_ref['y0'])

        # -------------------------------------------------------------
        # Obtain the measured orbit means and uncertainties.
        #
        # All configurations contain the same BPMs in the same order,
        # so no BPM-index remapping is necessary.
        # -------------------------------------------------------------
        stats = self.get_delta_orbit_stats(plane=plane, normalized=False)

        delta_cfg = _np.asarray(stats['mean'][cfg], dtype=float)
        delta_ref = _np.asarray(stats['mean'][ref_cfg], dtype=float)

        error_cfg = _np.asarray(stats[error][cfg], dtype=float)
        error_ref = _np.asarray(stats[error][ref_cfg], dtype=float)

        q_cfg = float(row['delta_charge'])
        q_ref = float(row_ref['delta_charge'])

        if (
            not _np.isfinite(q_cfg)
            or not _np.isfinite(q_ref)
            or _np.isclose(q_cfg, 0.0)
            or _np.isclose(q_ref, 0.0)
        ):
            raise ValueError(
                'The plotted and reference charge differences must be '
                'finite and nonzero.'
            )

        # -------------------------------------------------------------
        # Construct the measured reference-subtracted orbit.
        # -------------------------------------------------------------
        if normalized:
            delta_plot = delta_cfg / q_cfg - delta_ref / q_ref

            error_plot = _np.sqrt(
                (error_cfg / _np.abs(q_cfg)) ** 2
                + (error_ref / _np.abs(q_ref)) ** 2
            )

            ylabel = (
                rf'Reference-subtracted $\Delta {plane}$ '
                r'[$\mu$m/nC]'
            )
            theta_unit = r'$\mu$rad/nC'
        else:
            # Scale the reference orbit considering the charge difference
            # of both configurations.
            charge_scale = q_cfg / q_ref

            delta_plot = delta_cfg - charge_scale * delta_ref

            error_plot = _np.sqrt(
                error_cfg**2 + (charge_scale * error_ref) ** 2
            )

            ylabel = (
                rf'Reference-subtracted $\Delta {plane}$ '
                r'[$\mu$m]'
            )

            theta_unit = r'$\mu$rad'

        # The response vector is common to all selected configurations
        m_cfg = _np.asarray(self.M_ivu[plane][cfg], dtype=float)

        # ------------------------------------------------------------
        # Calculate the model kick difference.
        # ------------------------------------------------------------
        if model == 'individual':
            if gap_key not in self.theta_fits[plane]:
                raise RuntimeError(
                    f'Fit the {plane}-plane relative kicks at gap '
                    f'{gap_key:.3f} mm first.'
                )

            theta_fit = self.theta_fits[plane][gap_key]

            fit_config_indices = _np.asarray(
                theta_fit['config_indices'], dtype=int
            )

            cfg_local = _np.flatnonzero(fit_config_indices == cfg)
            ref_local = _np.flatnonzero(fit_config_indices == ref_cfg)

            if len(cfg_local) != 1 or len(ref_local) != 1:
                raise ValueError(
                    'Configuration indices were not found in the'
                    'relative-kick fit.'
                )

            cfg_local = int(cfg_local[0])
            ref_local = int(ref_local[0])

            theta = _np.asarray(theta_fit['theta'], dtype=float)

            theta_covariance = _np.asarray(
                theta_fit['theta_covariance'], dtype=float
            )

            # The fitted kicks are already normalized by charge.
            theta_difference = theta[cfg_local] - theta[ref_local]

            # Var(theta_i - theta_ref)
            theta_difference_var = (
                theta_covariance[cfg_local, cfg_local]
                + theta_covariance[ref_local, ref_local]
                - 2.0 * theta_covariance[cfg_local, ref_local]
            )

            theta_difference_err = _np.sqrt(max(0.0, theta_difference_var))

            if normalized:
                theta_model = theta_difference
                theta_model_err = theta_difference_err
            else:
                theta_model = q_cfg * theta_difference
                theta_model_err = _np.abs(q_cfg) * theta_difference_err

            model_label = 'Relative-kick model'

        else:
            if gap_key not in self.kperp_fits[plane]:
                raise RuntimeError(
                    f'Fit the {plane}-plane kick factor at gap '
                    f'{gap_key:.3f} mm first.'
                )

            fit = self.kperp_fits[plane][gap_key]

            delta_y0 = measured_bump - measured_ref_bump

            if normalized:
                theta_model = fit['a'] * delta_y0
                theta_model_err = _np.abs(delta_y0) * fit['a_err']
            else:
                theta_model = q_cfg * fit['a'] * delta_y0
                theta_model_err = _np.abs(q_cfg * delta_y0) * fit['a_err']

            model_label = 'Direct kick-factor model'

        orbit_model = theta_model * m_cfg

        # -------------------------------------------------------------
        # Plot.
        # -------------------------------------------------------------
        bpm_indices = _np.asarray(
            self.data[cfg]['data'][0]['bpm_indcs'],
            dtype=int,
        )

        spos = _np.asarray(
            self.spos_bpms
        )[bpm_indices]

        if ax is None:
            fig, ax = _plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        ax.errorbar(
            spos,
            delta_plot,
            yerr=error_plot,
            marker='.',
            linestyle='none',
            capsize=2,
            alpha=0.6,
            label='Reference-subtracted measurement',
        )

        ax.plot(
            spos,
            orbit_model,
            '-',
            linewidth=1.8,
            label=(
                rf'{model_label}: $\Delta\theta={theta_model:.3f}'
                rf'\pm{theta_model_err:.3f}$ {theta_unit}'
            ),
        )

        ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)

        plane_name = {'x': 'Horizontal', 'y': 'Vertical'}[plane]

        ax.set_title(
            f'{plane_name} orbit distortion ({self.id_name})\n'
            f'Gap = {gap_key:.2f} mm, '
            rf'$y_0$ = {measured_bump:.3f} mm, '
            rf'$y_{{0,\mathrm{{ref}}}}$ = '
            f'{measured_ref_bump:.3f} mm'
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

        gaps = _np.asarray(sorted(fits), dtype=float)

        kperp = _np.asarray([fits[gap]['kperp'] for gap in gaps])

        kperp_err = _np.asarray([fits[gap]['kperp_err'] for gap in gaps])

        if ax is None:
            fig, ax = _plt.subplots(figsize=(8, 5))
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
