"""."""
import numpy as _np
import matplotlib.pyplot as _mplt
from scipy.optimize import curve_fit

import pyaccel as _pa
from pymodels import si as _si
from apsuite.utils import DataBaseClass
from mathphys.functions import load_pickle

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

    @staticmethod
    def _process_data(data, get_dft=True):
        """."""
        try:
            # trajx, trajy = data['trajx']*1e-3, data['trajy']*1e-3
            trajx, trajy = data['trajx'][0]*1e-3, data['trajy'][0]*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        except KeyError:
            trajx = data['sofb_tbt']['x']*1e-3
            trajy = data['sofb_tbt']['y']*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        if get_dft:
            dftx = TbTData._calculate_dft(trajx)
            dfty = TbTData._calculate_dft(trajy)
            return trajx, trajy, dftx, dfty
        return trajx, trajy

    def process_data(self, get_dft=True):
        """Get mean-subtracted trajctories in [mm] and calculate the DFT.

        Store the trajectories as atributes `trajx` and `trajy` and the DFTs
         as `dftx` `dfty`.

        Args:
            get_dft (bool, optional): Whether to calculate the Discrete
             Fourier Transform (DFT). Defaults to True.
        """
        data = TbTData._process_data(self.data, get_dft=get_dft)
        self.data['trajx'], self.data['trajy'] = data[0], data[1]
        if get_dft:
            self.data['dftx'], self.data['dfty'] = data[2], data[3]

    def fit_hist_mat(self, traj='xy', from_turn=0, to_turn=15, model=None):
        """."""
        for axis in traj:
            self.data['J'+axis], self.data['tunes'+axis] = \
                TbTData._fit_hist_mat(data=self.data, traj=axis,
                                      from_turn=from_turn, to_turn=to_turn,
                                      model=model)

    @staticmethod
    def _fit_hist_mat(data, traj='x', from_turn=0, to_turn=15, model=None):
        """."""
        hist_mat = data['traj'+traj]
        dft = data['dft'+traj]

        tune_guesses = TbTData._get_dft_peaks_tunes(dft,
                                                    nturns=hist_mat.shape[0])
        hist_mat = hist_mat[from_turn:to_turn + 1, :]
        nturns = hist_mat.shape[0]
        nbpms = hist_mat.shape[-1]

        projections = TbTData._calc_projections_hist_mat(
            hist_mat, tune_guesses)
        amplitude_guesses = _np.linalg.norm(projections, axis=1)
        phase_guesses = _np.arctan2(projections[:, 0], projections[:, -1])

        p0 = _np.concatenate((amplitude_guesses[:, None],
                              tune_guesses[:, None],
                              phase_guesses[:, None]), axis=1)

        params = _np.zeros((nbpms, 3), dtype=float)
        n = _np.arange(nturns)

        for i, param_guess in enumerate(p0):
            params[i, :], _ = curve_fit(TbTData._TbT_model,
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
        # J is calculated as in eq. (9) of the reference X.R. Resende and M.B.
        # Alves and L. Liu and F.H. de Sá. Equilibrium and Nonlinear Beam
        # Dynamics Parameters From Sirius Turn-by-Turn BPM Data. In Proc.
        # IPAC'21. DOI: 10.18429/JACoW-IPAC2021-TUPAB219

        string = f'avg tune {fitted_tunes.mean():.4f}'
        string += f' +- {fitted_tunes.std():.4f} (std)'
        # TODO: add function to evaluate quality of the fit against BPMs data
        print(string)

        return fitted_J, fitted_tunes

    @staticmethod
    def _calculate_dft(traj, nturns=None):
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

    @staticmethod
    def _get_dft_peaks_tunes(dft, nturns):
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

    @staticmethod
    def _calc_projections_hist_mat(data, tune_guesses):
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
            data_projections[i, :] = TbTData._calc_projections(
                bpm_data, tune_guesses[i])
        return data_projections

    @staticmethod
    def _calc_projections(data, tune):
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
        cos, sin = TbTData._get_cos_and_sin_arrays(tune, n)
        cos_proj = data[None, :] @ cos[:, None]
        sin_proj = data[None, :] @ sin[:, None]
        norm = 2/N
        return cos_proj.item() * norm, sin_proj.item() * norm

    @staticmethod
    def _get_cos_and_sin_arrays(tune, n):
        """."""
        cos = _np.cos(2 * _np.pi * tune * n)
        sin = _np.sin(2 * _np.pi * tune * n)
        return cos, sin

    @staticmethod
    def _TbT_model(n, amplitude, tune, phase):
        """."""
        return amplitude * _np.sin(2 * _np.pi * tune * n + phase)

    def plot_spectrum(self, traj='trajx', bpm_idx=None):
        """."""
        raise NotImplementedError


class ADTSAnalysis():
    """."""

    def __init__(self, dir=None):
        """."""
        if dir is not None:
            self.dir = dir
            self.get_files(dir=dir)

    def get_files(self, dir: str):
        """."""
        self.dir = dir
        self.scan_files()

    def scan_files(self):
        """Scan the working directory and update the `files` property.

        `files` property is a dict with the kicks as keys and the TbT data
         of that kick as value.
        """
        files = sorted(os.listdir(path=self.dir))
        file_dict = {}
        for i, file in enumerate(files):
            slc = slice(file.find('urad')-3, file.find('urad'))
            # slc = slice(file.find('=') + 1, file.find('=') + 4)
            kick = file[slc]
            file_dict[kick] = file
        self.files = file_dict

    def remove_file(self, kick):
        """Remove a file from the list of files to be analyzed.

        Args:
            kick (str or int): file with this kick file is removed
        """
        del self.files[str(kick)]

    def add_file(self, kick):
        """."""
        raise NotImplementedError

    def process_data(self, get_dft=True):
        """."""
        self.data = dict()
        self.data['x'] = dict()
        self.data['y'] = dict()
        for kick_key, file in self.files.items():
            self.data['x'][kick_key] = dict()
            self.data['y'][kick_key] = dict()
            raw_data = load_pickle(self.dir+file)
            proc_data = TbTData._process_data(data=raw_data, get_dft=get_dft)
            self.data['x'][kick_key]['trajx'] = proc_data[0]
            self.data['y'][kick_key]['trajy'] = proc_data[1]
            if get_dft:
                self.data['x'][kick_key]['dftx'] = proc_data[2]
                self.data['y'][kick_key]['dfty'] = proc_data[3]

    def fit_data(self, traj='xy', from_turn=0, to_turn=15, model=None):
        """."""
        for axis in traj:
            for kick in self.files.keys():
                print(f'Fitting {kick} urad file')
                data = self.data[axis][kick]
                self.data[axis][kick]['J'],  self.data[axis][kick]['tunes'] = \
                    TbTData()._fit_hist_mat(data=data,
                                            traj=axis,
                                            from_turn=from_turn,
                                            to_turn=to_turn,
                                            model=model)
