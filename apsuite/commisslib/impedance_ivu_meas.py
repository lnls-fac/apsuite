"""."""

import numpy as np

from siriuspy.devices import SOFB, IVU
from siriuspy.search import BPMSearch

from apsuite.commisslib.meas_bpms_signals import (
    AcqBPMsSignals as _BaseAcq,
    AcqBPMsSignalsParams as _BaseParams,
)
from apsuite.utils import ThreadedMeasBaseClass as _BaseThreaded


class MeasIVUImpedanceParams(_BaseParams):
    """."""

    ADC_NSAMPLES_PER_TURN = 382
    HARM_NUM = 864

    def __init__(self):
        """."""
        super().__init__()
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
        stg += '\nMeasIVUImpedanceParams:\n'
        stg += f'    num_acquisitions = {self.num_acquisitions}\n'
        stg += f'    save_raw_data = {self.save_raw_data}\n'
        stg += f'    num_buckets_to_process = {self.num_buckets_to_process}\n'
        stg += f'    nrturns = {self.nrturns}\n'
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


class MeasIVUImpedance(_BaseThreaded, _BaseAcq):
    """."""

    def __init__(self, isonline=True, bpmtype='all'):
        """."""
        bpmnames = BPMSearch.get_names(filters={'sec': 'SI', 'dev': 'BPM'})
        if bpmtype.startswith('odd'):
            bpmnames = bpmnames[::2]
        elif bpmtype.startswith('even'):
            bpmnames = bpmnames[1::2]
        _BaseThreaded.__init__(self, isonline=isonline, target=self._measure)
        _BaseAcq.__init__(self, isonline=self.isonline, bpmnames=bpmnames)
        self.params = MeasIVUImpedanceParams()

    def create_devices(self, bpmnames=None):
        """."""
        _BaseAcq.create_devices(self, bpmnames=bpmnames)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['ivu18_08'] = IVU(IVU.DEVICES.IVU18_08SB)
        self.devices['ivu18_14'] = IVU(IVU.DEVICES.IVU18_14SB)

    def get_data(self):
        """."""
        data = super().get_data()
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
        for i in range(self.params.num_acquisitions):
            print(
                f'Acquisition {i + 1:02d}/{self.params.num_acquisitions:02d}'
            )
            if self._stopevt.is_set():
                break
            self.acquire_data()
            data.append(self.data)

        print('Finished!')
        self.data = data
        self.process_data()
        self._filter_data_to_save()

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

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.

        Returns:
            dorb: orbit deviation between the two stored bunches.
            orb1: orbit of the first stored bunch.
            orb2: orbit of the second stored bunch.
        """
        if not self.data:
            raise RuntimeError('Get data First.')
        orb1, orb2 = [], []
        for dt in self.data:
            if 'b1_posx' not in dt or 'b2_posx' not in dt:
                raise RuntimeError(
                    'Missing bunch positions in data. Process data firts.'
                )
            orb1.append(np.vstack([dt['b1_posx'], dt['b1_posy']]))
            orb2.append(np.vstack([dt['b2_posx'], dt['b2_posy']]))
        orb1 = np.array(orb1)
        orb2 = np.array(orb2)
        dorb = orb1 - orb2
        return dorb, orb1, orb2

    def calc_current_2_bunches(self):
        """Calculate current of the two stored bunches.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.

        Returns:
            curr1: current of the first stored bunch.
            curr2: current of the second stored bunch.

        """
        if not self.data:
            raise RuntimeError('Get data First.')
        curr1, curr2 = [], []
        for dt in self.data:
            if 'b1_curr' not in dt or 'b2_curr' not in dt:
                raise RuntimeError(
                    'Missing bunch currents in data. Process data firts.'
                )
            curr1.append(dt['b1_curr'])
            curr2.append(dt['b2_curr'])
        curr1 = np.array(curr1)
        curr2 = np.array(curr2)
        return curr1, curr2

    def calc_sofb_orbit(self, isref=False):
        """Calculate the SOFB orbit.

        Raises:
            RuntimeError: If there is no data acquired.
            RuntimeError: If data is not processed yet.

        Returns:
            orb: The SOFB orbit.
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
        b1_offset = 50
        windowp = 20
        windown = -10
        nbuc2proc = self.params.num_buckets_to_process
        bhigh = self.params.bucket_hi_charge
        blow = self.params.bucket_lo_charge
        nsamp_pturn = MeasIVUImpedanceParams.ADC_NSAMPLES_PER_TURN
        hnum = MeasIVUImpedanceParams.HARM_NUM

        ant_raw = np.array([data['ampl' + ant] for ant in 'abcd'])
        ant_raw = ant_raw.swapaxes(
            1, 2
        )  # [4, 382 * N, 160] --> [4, 160, 382 * N]
        curr = data['stored_current']

        ant_abs = np.abs(ant_raw)
        ant_amax = ant_abs[..., :nsamp_pturn].argmax(axis=-1)

        nsamp2keep = ant_raw.shape[-1]
        nturn2keep = nsamp2keep // nsamp_pturn
        idx = np.arange(nsamp2keep)
        old_idx = (idx - b1_offset + ant_amax[..., None]) % nsamp2keep

        ant_raw2 = np.take_along_axis(ant_raw, old_idx, axis=-1)
        ant_raw2 = ant_raw2.reshape(ant_raw2.shape[:2] + (nturn2keep, -1))

        dic = {}
        if return_all:
            dic['ant_raw'] = ant_raw
            dic['ant_amax'] = ant_amax
            dic['ant_raw2'] = ant_raw2

        b2_offset = abs(blow - bhigh) / hnum * nsamp_pturn
        b2_offset = int(b2_offset) + b1_offset
        slcs = [
            slice(b1_offset + windown, b1_offset + windowp),
            slice(b2_offset + windown, b2_offset + windowp),
        ]
        pref = lambda x: f'b{x + 1}_'
        for i in range(nbuc2proc):
            b_sigs = ant_raw2[..., slcs[i]].std(axis=-1)
            b_posx, b_posy = MeasIVUImpedance.calc_positions_from_amplitudes(
                b_sigs
            )
            b_sum = b_sigs.sum(axis=0)

            dic[pref(i) + 'posx'] = b_posx
            dic[pref(i) + 'posy'] = b_posy
            dic[pref(i) + 'sum'] = b_sum
            if return_all:
                dic[pref(i) + 'sigs'] = b_sigs

        bt_sum = sum([dic[pref(i) + 'sum'] for i in range(nbuc2proc)])
        dic['bt_sum'] = bt_sum
        for i in range(nbuc2proc):
            dic[pref(i) + 'curr'] = dic[pref(i) + 'sum'] * curr / bt_sum

        return dic

    @staticmethod
    def _find_peak(ant_amp, search_reg, npts=5):
        slc = slice(*search_reg)
        amax = ant_amp[..., slc].argmax(axis=-1) + (search_reg[0] or 0)
        amax = np.expand_dims(amax, axis=-1)
        x = np.arange(-npts, npts + 1)
        slc = amax + x
        ant_amp_slc = np.take_along_axis(ant_amp, slc, axis=-1)
        coefs = np.polynomial.polynomial.polyfit(
            x, ant_amp_slc.reshape(-1, ant_amp_slc.shape[-1]).T, deg=2
        ).T.reshape(ant_amp_slc.shape[:-1] + (-1,))

        xmax = -coefs[..., 1] / (2 * coefs[..., 2])
        ymax = coefs[..., 0] + coefs[..., 1] * xmax + coefs[..., 2] * xmax**2
        return ymax, amax, xmax, coefs
