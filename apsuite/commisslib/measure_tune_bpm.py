"Methods for measure tunes by bpm data"

import numpy as _np
import time as _time
import pyaccel as _pa
from numpy.fft import rfft, rfftfreq, rfftn
import matplotlib.pyplot as _plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from siriuspy.sofb.csdev import SOFBFactory

from siriuspy.devices import CurrInfoBO, \
    Trigger, Event, EVG, RFGen, SOFB, PowerSupplyPU, FamBPMs

from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class BPMeasureParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        # self.n_bpms = 50  # Numbers of BPMs
        self.nr_points_after = 0
        self.nr_points_before = 10000
        self.bpms_timeout = 30  # [s] (?)
        self.DigBOmode = 'Extraction'  # or 'Injection'


class BPMeasure(_ThreadBaseClass):
    """."""
    def __init__(self, params=None, isonline=True):
        """."""
        params = BPMeasureParams() if params is None else params
        # Do I need to set a target in the below line?
        super().__init__(params=params, isonline=isonline)
        self.sofb_data = SOFBFactory.create('BO')
        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['bobpms'] = FamBPMs(FamBPMs.DEVICES.BO)
        # self.devices['event'] = Event('Study')
        self.devices['event'] = Event('DigBO')
        self.devices['evg'] = EVG()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.BO)
        self.devices['trigbpm'] = Trigger('BO-Fam:TI-BPM')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()
        self.devices['ejekckr'] = PowerSupplyPU(PowerSupplyPU.
                                                DEVICES.BO_EJE_KCKR)

    def get_orbit_data(self):
        """Get orbit data from BPMs in TbT acquisition rate

        BPMs must be configured to listen DigBO event and the DigBO
        event must be in Injection mode."""

        prms = self.params
        bobpms = self.devices['bobpms']
        bobpms.mturn_config_acquisition(
            nr_points_after=prms.nr_points_after,
            nr_points_before=prms.nr_points_before,
            acq_rate='TbT', repeat=False, external=False)
        # I put external=False because DigBO must be in Injection mode
        # This make sense?
        bobpms.mturn_reset_flags()
        # DigBO putted in Extraction mode
        self.devices['event'].mode = prms.DigBOmode
        ret = bobpms.mturn_wait_update_flags(timeout=prms.bpms_timeout)
        if ret != 0:
            print(f'Problem waiting BPMs update. Error code: {ret:d}')
            return dict()

        orbx, orby = bobpms.get_mturn_orbit()
        # I am not sure if i really need to store sofb data
        chs_names = [self.sofb_data.ch_names[idx] for idx in prms.chs_idx]
        data = dict()
        data['timestamp'] = _time.time()
        data['chs_off'] = chs_names
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'], data['orby'] = orbx, orby
        data['mt_acq_rate'] = 'TbT'
        data['rf_frequency'] = self.devices['rfgen'].frequency
        self.data = data

    def load_orbit(self, data=None, orbx=None, orby=None):
        """Load orbit data into the object. You can pass the
        intire data dictionary or just the orbits. If data argument
        is provided, orbx and orby become optional"""

        if not hasattr(self, 'data'):
            self.data = dict()

        if data is not None:
            self.data = data
        if orbx is not None:
            self.data['orbx'] = orbx
        if orby is not None:
            self.data['orby'] = orby

    def dft(self, bpm_index=0):
        "Apply a dft in a single bpm"
        orbx = self.data['orbx'][:, bpm_index]
        orby = self.data['orby'][:, bpm_index]

        x_beta = orbx - orbx.mean(axis=0)
        y_beta = orby - orby.mean(axis=0)

        N = x_beta.shape[0]
        freqs = rfftfreq(N)

        spectrumx = _np.abs(rfft(x_beta))*2*_np.pi/N
        spectrumy = _np.abs(rfft(y_beta))*2*_np.pi/N

        return spectrumx, spectrumy, freqs

    def naff_tunes(self, dn=None, window_param=1):
        """Computes the tune evolution from the BPMs matrix with a moving
            window of length dn.
           If dn is not passed, the tunes are computed using all points."""

        x = self.data['orbx']
        y = self.data['orby']
        N = x.shape[0]

        if dn is None:
            return self.tune_by_naff(x, y)

        else:
            tune1_list = []
            tune2_list = []
            slices = _np.arange(0, N, dn)
            for idx in range(len(slices)-1):
                idx1, idx2 = slices[idx], slices[idx+1]
                sub_x = x[idx1:idx2, :]
                sub_y = y[idx1:idx2, :]
                tune1, tune2 = self.tune_by_naff(sub_x, sub_y, window_param=1,
                                                 decimal_only=True)
                tune1_list.append(tune1)
                tune2_list.append(tune2)

        return _np.array(tune1_list), _np.array(tune2_list)

    def spectrogram(self, dn=None):
        """."""
        x = self.data['orbx'] - self.data['orbx'].mean(axis=0)
        y = self.data['orby'] - self.data['orby'].mean(axis=0)
        N = x.shape[0]

        if dn is None:
            dn = N//50

        dn = int(dn)

        freqs = rfftfreq(dn)
        tune1_matrix = _np.zeros([freqs.size, N-dn])
        tune2_matrix = tune1_matrix.copy()

        for n in range(N-dn):
            sub_x = x[n:n+dn, :]
            espectra_by_bpm_x = _np.abs(rfftn(sub_x, axes=[0]))

            sub_y = y[n:n+dn, :]
            espectra_by_bpm_y = _np.abs(rfftn(sub_y, axes=[0]))

            tune1_matrix[:, n] = _np.mean(espectra_by_bpm_x, axis=1)
            tune2_matrix[:, n] = _np.mean(espectra_by_bpm_y, axis=1)
        tune_matrix = tune1_matrix + tune2_matrix

        # normalizing this matrix to get a better heatmap plot:
        tune_matrix = (tune_matrix - tune_matrix.min()) / \
            (tune_matrix.max() - tune_matrix.min())

        # plots spectogram
        _plot_heatmap(tune_matrix, freqs)

        return tune1_matrix, tune2_matrix, freqs

    @staticmethod
    def tune_by_naff(x, y, window_param=1, decimal_only=True):
        """."""
        M = x.shape[1]
        beta_osc_x = x - _np.mean(x, axis=0)
        beta_osc_y = y - _np.mean(y, axis=0)

        Ax = beta_osc_x.ravel()
        Ay = beta_osc_y.ravel()

        freqx, _ = _pa.naff.naff_general(Ax, is_real=True, nr_ff=1,
                                         window=window_param)
        freqy, _ = _pa.naff.naff_general(Ay, is_real=True, nr_ff=1,
                                         window=window_param)
        tune1, tune2 = M*freqx, M*freqy

        if decimal_only is False:
            return tune1, tune2
        else:
            tune1, tune2 = _np.abs(tune1 % 1), _np.abs(tune2 % 1)
            if tune1 > 0.5:
                tune1 = _np.abs(1-tune1)
            if tune2 > 0.5:
                tune2 = _np.abs(1-tune2)
            return tune1 % 1, tune2 % 1


def _plot_heatmap(tune_matrix, freqs):
    ax = _plt.subplot()
    extent = [0, tune_matrix.shape[1], freqs[1:].min(), freqs[1:].max()]
    im = ax.imshow(tune_matrix[1:, :], cmap='hot', aspect='auto',
                   extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    _plt.colorbar(im, cax=cax)
    ax.set_ylabel('Frac. Freq')
    ax.set_xlabel('Turns')
    _plt.show()
