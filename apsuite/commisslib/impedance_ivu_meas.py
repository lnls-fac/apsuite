"""."""

import numpy as np

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
        stg += '(\'all\', \'odd/even\')\n'
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
                'Acquisition strategy \'odd/even\' identified. '
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
