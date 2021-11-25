"Tune measure module"

import numpy as _np
import pyaccel as _pa
import pymodels as _pm
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


class BPMeasure:
    def __init__(self, bo=None, Ax=None, Ay=None):
        """."""
        self._bo = bo
        self._Ax = Ax
        self._Ay = Ay
        self._QF_KL_default = 0.18862733  # Default KL of QF quadrupoles

    def create_booster(self, QF_KL=None, KsL=None):
        """."""
        self._bo = _pm.bo.create_accelerator(energy=3e9)
        self._bo.cavity_on = True
        self._bo.radiation_on = True
        self._bo.vchamber_on = True

        self._famdata = _pm.bo.get_family_data(self._bo)

        # Introducing coupling:
        if KsL is not None:
            qs_idx = self._famdata['QS']['index']
            self._bo[qs_idx[0][0]].KsL = KsL

        # Changing QF quadrupole forces:
        if QF_KL is not None:
            qf_idx = _np.array(self._famdata['QF']['index']).flatten()
            for qf in qf_idx:
                self._bo[qf].KL = QF_KL

        # Getting optics information about the machine
        self._et, _ = _pa.optics.calc_edwards_teng(self._bo)
        self._coupling_coef, _ = _pa.optics.estimate_coupling_parameters(
                                 self._et)
        self._eqparams = _pa.optics.EqParamsFromBeamEnvelope(self._bo)

    def create_initial_bunch(self, n_part=1000, offset=None):
        """
        Creates a bunch in the booster

        Args:
        n_part (int, optional) : number of particles in the bunch.
        offset (np.array, optional) : column np.array with [x,x',y,y',de,dl]
         offset in relation to close orbit
        """
        if offset is None:
            offset = _np.array([[0], [0], [0], [0], [0], [0]])

        emit1, emit2 = self._eqparams.emit1, _np.abs(self._eqparams.emit2)
        sigmae, sigmal = self._eqparams.espread0, self._eqparams.bunlen
        self._bunch = _pa.tracking.generate_bunch(n_part=n_part, emit1=emit1,
                                                  emit2=emit2,
                                                  optics=self._et[0],
                                                  sigmae=sigmae, sigmas=sigmal)
        co = _pa.tracking.find_orbit6(accelerator=self._bo, indices='closed')
        self._bunch += co[:, [0]] + offset

    def tracking_and_get_bpmdata(self, N_turns=40):
        """."""
        bpm_idx = self._famdata['BPM']['index']
        M = len(bpm_idx)

        x_measures = _np.zeros([N_turns, M])   # (Turn x BPM)
        y_measures = _np.zeros(x_measures.shape)

        for n in range(N_turns):
            part_out, *_ = _pa.tracking.line_pass(
                accelerator=self._bo, particles=self._bunch, indices='closed',
                parallel=True
                )
            centroid = _np.mean(part_out, axis=1)
            bpm_measures = centroid[:, bpm_idx].reshape(6, M)
            x_measures[n, :] = bpm_measures[0, :]  # Selects  x at all BPM's
            y_measures[n, :] = bpm_measures[2, :]  # Selects  y at all BpM's
            self._bunch = part_out[:, :, -1]

        return x_measures, y_measures


# ----- Functions to estimates tune by BPM data --------- #
def DFT(betatron_osc):
    """."""
    N = _np.size(betatron_osc)

    # Fourier Transform
    yf = _np.abs(rfft(betatron_osc))
    tunes = rfftfreq(N)

    yf_normalized = yf/(yf.max() - yf.min())
    # signal processing: finding proeminent peaks
    peaks, _ = find_peaks(yf_normalized)
    y_peaks = yf_normalized[peaks]
    tune_peaks = tunes[peaks]
    if tune_peaks.size != 0:
        maxpeak = _np.argmax(y_peaks)
    else:
        return 0

    return tune_peaks[maxpeak]


def tune_by_DFT(x_measures, y_measures):
    """Estimates tunes using mixed BPM data and Fourier Transform"""

    M = x_measures.shape[1]

    tunesx = _np.zeros(M)
    tunesy = _np.zeros(M)

    beta_osc_x = x_measures - _np.mean(x_measures, axis=0)
    beta_osc_y = y_measures - _np.mean(y_measures, axis=0)

    for j in range(M):
        tunesx[j] = DFT(beta_osc_x[:, j])
        tunesy[j] = DFT(beta_osc_y[:, j])

    return tunesx.mean(), tunesy.mean()


def tune_by_NAFF(x_measures, y_measures, window_param=None, decimal_only=True):
    """Estimates tunes using mixed BPM data and Numerical Analysis
    of Fundamental Frequencies"""

    M = x_measures.shape[1]

    beta_osc_x = x_measures - _np.mean(x_measures, axis=0)
    beta_osc_y = y_measures - _np.mean(y_measures, axis=0)

    Ax = beta_osc_x.ravel()
    Ay = beta_osc_y.ravel()

    # May we could include the below lines at naff code:
    px = Ax.size//6
    if Ax.size % 6 < 1:
        px -= 1
    px = 6*px+1
    Ax = Ax[:px]
    Ay = Ay[:px]

    freqx, _ = _pa.naff.naff_general(Ax, is_real=True, nr_ff=2, window=1)
    freqy, _ = _pa.naff.naff_general(Ay, is_real=True, nr_ff=2, window=1)
    tunex, tuney = M*freqx[0], M*freqy[0]

    if decimal_only is False:
        return tunex, tuney
    else:
        tunex, tuney = _np.abs(tunex % 1), _np.abs(tuney % 1)
        if tunex > 0.5:
            tunex = _np.abs(1-tunex)
        if tuney > 0.5:
            tuney = _np.abs(1-tuney)
        return tunex, tuney
