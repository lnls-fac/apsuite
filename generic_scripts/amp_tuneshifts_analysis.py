"""."""
import numpy as _np
import matplotlib.pyplot as _mplt
from scipy.optimize import curve_fit

import pyaccel as _pa
from pymodels import si as _si
from apsuite.optics_analysis.tune_correction import TuneCorr
from apsuite.optics_analysis.chromaticity_correction import ChromCorr
from apsuite.utils import DataBaseClass
from mathphys.functions import load_pickle, save_pickle

import os


class TbTData(DataBaseClass):
    """."""

    def __init__(self, fname=None):
        """."""
        self.data = dict()
        if fname is not None:
            self.load_and_apply(fname=fname)

    def load_and_apply(self, fname: str):
        """."""
        try:
            super().load_and_apply(fname)
        except KeyError:
            data = load_pickle(fname)
            keys = set(data.keys())
            if 'sofb_tbt' not in keys:
                is_data = set(['trajx', 'trajy', 'trajsum'])
            else:
                is_data = set('sofb_tbt')
            is_params = keys - is_data
            self.data = {key: data[key] for key in is_data}
            self.params = {key: data[key] for key in is_params}

    def process_data(self, get_dft=True):
        """Get mean-subtracted trajctories in [mm] and calculate the DFT.

        Store the trajectories as atributes `trajx` and `trajy` and the DFTs
         as `dftx` `dfty`.

        Args:
            get_dft (bool, optional): Whether to calculate the Discrete
             Fourier Transform (DFT). Defaults to True.
        """
        try:
            trajx, trajy = self.data['trajx']*1e-3, self.data['trajy']*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        except KeyError:
            trajx = self.data['sofb_tbt']['x']*1e-3
            trajy = self.data['sofb_tbt']['y']*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        self.data['trajx'], self.data['trajy'] = trajx, trajy
        if get_dft:
            self.data['dftx'] = self._calculate_dft(trajx)
            self.data['dfty'] = self._calculate_dft(trajy)

    def fit_hist_mat(self, model=None):
        """."""
        hist_mat = self.data['trajx']
        nturns = hist_mat.shape[0]
        nbpms = hist_mat.shape[-1]
        dft = self.data['dftx']
        tune_guesses = self._get_dft_peaks_tunes(dft, nturns)
        projections = self._calc_projections_hist_mat(hist_mat, tune_guesses)
        amplitude_guesses = _np.linalg.norm(projections, axis=1)
        phase_guesses = _np.arctan2(projections[:, 0], projections[:, -1])

        p0 = _np.concatenate((amplitude_guesses[:, None],
                              tune_guesses[:, None],
                              phase_guesses[:, None]), axis=1)

        params = _np.zeros((nbpms, 3), dtype=float)
        n = _np.arange(nturns)

        for i, param_guess in enumerate(p0):
            params[i, :], _ = curve_fit(self._TbT_model,
                                        xdata=n,
                                        ydata=hist_mat[:nturns, i],
                                        p0=param_guess)
        if model is None:
            model = _si.create_accelerator()
            model.radiation_on = False
            model.cavity_on = False
            model.vchamber_on = False

        famdata = _si.get_family_data(model)
        bpms_idcs = _pa.lattice.flatten(famdata['BPM']['index'])
        twiss, *_ = _pa.optics.calc_twiss(
            accelerator=model, indices=bpms_idcs)
        betax, betay = twiss.betax, twiss.betay

        fitted_tunes = params[:, 1]
        fitted_J = (params[:, 0]**4).sum() / (betax * params[:, 0]**2).sum()
        # J is calculated as in eq. (9) of the reference X.R. Resende and M.B. Alves and L. Liu and F.H. de Sá. Equilibrium and Nonlinear Beam Dynamics Parameters From Sirius Turn-by-Turn BPM Data. In Proc. IPAC'21. DOI: 10.18429/JACoW-IPAC2021-TUPAB219

        string = f'avg tune {fitted_tunes.mean():.3f}'
        string += f'+- {fitted_tunes.std():.3f} (std)'
        print(string)

        self.data['J'] = fitted_J
        self.data['tunes'] = fitted_tunes

    def _calculate_dft(self, traj, nturns=None):
        """Calculate the Discrete Fourier Transform (DFT) history matrix.

        Args:
            data (n x m array): history matrix where n is the number of turns
             and m is the number of BPMS
            nturns (int, optional): number of turns to consider. Defaults to
             None, in which case, data.shape[0] is used.

        Returns:
            nturns x m  DFT array: _description_
        """
        if nturns is None or nturns > traj.shape[0]:
            nturns = traj.shape[0]
        return _np.fft.rfft(a=traj[:nturns], axis=0)

    def _get_dft_peaks_tunes(self, dft, nturns):
        """Identify the tunes at which the DFT amplitudes peaks.

        Args:
            dft (array): DFT of the history matrix or of a single BPM TbT
             time-series
            nturns (int): Number of turns considered for the calculation of
             the DFT.

        Returns:
            array: tune guesses for each BPM or for a single BPM.
        """
        tunes = _np.fft.rfftfreq(n=nturns)
        idcs = _np.abs(dft).argmax(axis=0)
        tune_guesses = _np.array([tunes[idc] for idc in idcs], dtype=float)
        return tune_guesses

    def _calc_projections_hist_mat(self, data, tune_guesses):
        """Calculate each BPM time-series sine and cosine components.

        Calculates the cosine and and amplitudes for each BPM time series by
         projecting data into these functions at the specified tunes.

        Args:
            data (ndarray): n x m history matrix
            tune_guesses (array): m-array containing the tunes at wich to
             project each BPMs time-series. The i-th tune entry corresponds to
             the tune at which the i-th BPM time series is projected.

        Returns:
            ndarray: m x 2 array containing the cosine and sine amplitudes
             for each one of the m BPMs time series.
        """
        data_projections = _np.zeros((data.shape[-1], 2))
        for i, bpm_data in enumerate(data.T):
            data_projections[i, :] = self._calc_projections(
                bpm_data, tune_guesses[i])
        return data_projections

    def _calc_projections(self, data, tune):
        """Calculate BPM raw data sine and cosine components.

        Calculates the cosine and sine and amplitudes by projecting data into
        these functions at the specified tune.

        Args:
            data (n-array): data vector
            tune (float): freuqency at which to project

        Returns:
            tuple: the data's amplitudes along cosine and sine components
        """
        N = data.shape[0]
        n = _np.arange(N)
        cos, sin = self._get_cos_and_sin_arrays(tune, n)
        cos_proj = data[None, :] @ cos[:, None]
        sin_proj = data[None, :] @ sin[:, None]
        norm = 2/N
        return cos_proj.item() * norm, sin_proj.item() * norm

    def _get_cos_and_sin_arrays(self, tune, n):
        """."""
        cos = _np.cos(2 * _np.pi * tune * n)
        sin = _np.sin(2 * _np.pi * tune * n)
        return cos, sin

    def _TbT_model(self, n, amplitude, tune, phase):
        """."""
        return amplitude * _np.sin(2 * _np.pi * tune * n + phase)

    def plot_spectrum(self, traj='trajx', bpm_idx=None):
        raise NotImplementedError
