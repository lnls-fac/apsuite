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

    # @staticmethod
    def process_data(self, auto_filtering=False):
        """."""
        try:
            trajx = self.data['trajx'].copy()*1e-3
            trajy = self.data['trajy'].copy()*1e-3
            # trajx, trajy = self.data['trajx'][0]*1e-3, self.data['trajy'][0]*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        except KeyError:
            trajx = self.data['sofb_tbt']['x'].copy()*1e-3
            trajy = self.data['sofb_tbt']['y'].copy()*1e-3
            trajx -= trajx.mean(axis=0)
            trajy -= trajy.mean(axis=0)
        self.data['dftx'] = TbTData._calculate_dft(trajx)
        self.data['dfty'] = TbTData._calculate_dft(trajy)

        self.data['trajx'] = trajx
        self.data['trajy'] = trajy

        # self.filter_data2()

        # tunexpeaks = TbTData._get_dft_peaks_tunes(
        #     self.data['dftx'], nturns=trajx.shape[0])
        # tuneypeaks = TbTData._get_dft_peaks_tunes(
        #     self.data['dfty'], nturns=trajy.shape[0])

        # recalculate_peaks = False
        # for idc, (tunex, tuney) in enumerate(zip(tunexpeaks, tuneypeaks)):
        #     if tunex != tuney:
        #         # band-pass filter around the peaks
        #         self.filter_data(trajs='xy', central_tunes=(tunex, tuney),
        #                          bands=(0.005, 0.01),
        #                          keep_within_range=(True, True),
        #                          bpm_idc=idc)
        #     else:
        #         # band-pass around tunex peak & filter-out tunex from y
        #         print('equal')
        #         self.filter_data(trajs='xy', central_tunes=(tunex, tunex),
        #                          bands=(0.005, 0.01),
        #                          keep_within_range=(True, False),
        #                          bpm_idc=idc)
        #         recalculate_peaks = recalculate_peaks or True

        # if recalculate_peaks:
        #     tuneypeaks = TbTData._get_dft_peaks_tunes(
        #         self.data['dfty'], nturns=trajy.shape[0])
        #     print(tuneypeaks)
        #     for idc, tuney in enumerate(tuneypeaks):
        #         self.filter_data(trajs='y', central_tunes=tuney, bands=0.05,
        #                          keep_within_range=True, bpm_idc=idc)

    # def process_data(self, auto_filtering=True):
    #     """Get mean-subtracted trajctories in [mm] and calculate the DFT.

    #     Store the trajectories as atributes `trajx` and `trajy` and the DFTs
    #      as `dftx` `dfty`.

    #     Args:
    #         get_dft (bool, optional): Whether to calculate the Discrete
    #          Fourier Transform (DFT). Defaults to True.
    #     """
    #     data = TbTData._process_data(self.data,auto_filtering=auto_filtering)
    #     self.data['trajx'], self.data['trajy'] = data[0], data[1]
    #     self.data['dftx'], self.data['dfty'] = data[2], data[3]

    def filter_data2(self):
        """."""
        dftx, dfty = self.data['dftx'], self.data['dfty']
        xpeaks_idcs = _np.argmax(_np.abs(dftx), axis=0)
        ypeaks_idcs = _np.argmax(_np.abs(dfty), axis=0)
        tunesx = _np.fft.rfftfreq(self.data['trajx'].shape[0])
        tunesy = _np.fft.rfftfreq(self.data['trajy'].shape[0])

        tunesx_at_peaks = list()
        tunesy_at_peaks = list()
        for idcx, idcy in zip(xpeaks_idcs, ypeaks_idcs):
            tunesx_at_peaks.append(tunesx[idcx])
            tunesy_at_peaks.append(tunesy[idcy])
        tunesx_at_peaks = _np.array(tunesx_at_peaks)
        tunesy_at_peaks = _np.array(tunesy_at_peaks)

        # identify BPMs whose spectra overlap
        overlap = _np.abs(tunesy_at_peaks - tunesx_at_peaks) < 0.06
        # Band-pass filter on trajx spectra
        dftx = self.filter_around_peak(dftx, xpeaks_idcs,
                                       keep_within=True, n=2)
        # Band-pass filter around y peaks for tunesy w/ no overlap with tunesx
        dfty[:, ~overlap] = self.filter_around_peak(dfty[:, ~overlap],
                                                    ypeaks_idcs[~overlap],
                                                    keep_within=True,
                                                    n=2)
        # Filter out tunex peak from trajy to remove coupling artifacts
        dfty[:, overlap] = self.filter_around_peak(dfty[:, overlap],
                                                   ypeaks_idcs[overlap],
                                                   keep_within=False, n=4)
        ypeaks_idcs = _np.argmax(_np.abs(dfty), axis=0)
        dfty = self.filter_around_peak(dfty, ypeaks_idcs,
                                       keep_within=True, n=2)

        self.data['dftx'], self.data['dfty'] = dftx, dfty
        self.data['trajx'] = _np.fft.irfft(dftx, axis=0)
        self.data['trajy'] = _np.fft.irfft(dfty, axis=0)

    def filter_around_peak(self, dft, peak_idcs, keep_within=True, n=2):
        """."""
        for i, (idc, bpm_dft) in enumerate(zip(peak_idcs, dft.T)):
            bpm_spec = _np.abs(bpm_dft)
            peak = bpm_spec[idc]
            idcs = _np.argwhere(bpm_spec < peak / n)
            idx = idcs[_np.argmin(_np.abs(idcs - idc))][0]
            if idx < idc:
                lower_lim = idx
                upper_lim = idc + (idc - lower_lim)
            else:
                lower_lim = idc - (idx - idc)
                upper_lim = idx
            if keep_within:
                dft[:lower_lim, i] = 0
                dft[upper_lim:, i] = 0
            else:
                dft[lower_lim:upper_lim+1, i] = 0
        return dft

    def filter_data(self, trajs='xy', central_tunes=(0.08, 0.14),
                    bands=(0.03, 0.03), keep_within_range=(True, True),
                    bpm_idc=None):
        """."""
        is_tuple = isinstance(central_tunes, tuple)
        is_tuple = is_tuple and isinstance(bands, tuple)
        is_tuple = is_tuple and isinstance(keep_within_range, tuple)
        if len(trajs) > 1 and (not is_tuple):
            raise TypeError('must provide central tunes, bands and whether to keep_within range for both planes')

        elif len(trajs)==1:
            central_tunes = (central_tunes,)
            bands, keep_within_range = (bands,), (keep_within_range,)

        for traj, tune, bd, kp in zip(trajs, central_tunes, bands,
                                      keep_within_range):
            dft = self.data['dft'+traj].copy()
            if bpm_idc is not None:
                dft = dft[:, bpm_idc]
            tunes = _np.fft.rfftfreq(n=self.data['traj'+traj].shape[0])
            tunemin = tune - bd
            tunemax = tune + bd
            if kp:
                idcs = (tunes < tunemin) | (tunes > tunemax)
                dft[idcs] = 0
            else:
                idcs = (tunes > tunemin) & (tunes < tunemax)
                dft[idcs] = 0
            if bpm_idc is not None:
                self.data['dft'+traj][:, bpm_idc] = dft
                self.data['traj'+traj][:, bpm_idc] = _np.fft.irfft(dft, axis=0)
            else:
                self.data['dft'+traj] = dft
                self.data['traj'+traj] = _np.fft.irfft(dft, axis=0)

    def fit_hist_mat(self, traj='xy',
                     from_turn=None, to_turn=None, model=None):
        """."""
        is_tuple = isinstance(from_turn, tuple) and isinstance(to_turn, tuple)
        if len(traj) == 2 and not is_tuple:
            raise TypeError(
                'Both from_turn and to_turn must be tuples for len(traj)>1')
        for i, axis in enumerate(traj):
            self.data['J'+axis], params = \
                TbTData._fit_hist_mat(data=self.data, traj=axis,
                                      from_turn=from_turn[i],
                                      to_turn=to_turn[i],
                                      model=model)
            self.data['tunes'+axis] = params[:, 1]
            self.data['fitted_amps'+axis] = params[:, 0]
            self.data['fitted_phases'+axis] = params[:, -1]

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
        beta = twiss.betax if traj == 'x' else twiss.betay

        fitted_tunes = params[:, 1]
        fitted_J = (params[:, 0]**4).sum() / (beta * params[:, 0]**2).sum()
        # J is calculated as in eq. (9) of the reference X.R. Resende and M.B.
        # Alves and L. Liu and F.H. de Sá. Equilibrium and Nonlinear Beam
        # Dynamics Parameters From Sirius Turn-by-Turn BPM Data. In Proc.
        # IPAC'21. DOI: 10.18429/JACoW-IPAC2021-TUPAB219

        string = f'avg tune {traj} {fitted_tunes.mean():.4f}'
        string += f' +- {fitted_tunes.std():.4f} (std)'
        print(string)
        return fitted_J, params

    @staticmethod
    def _calculate_dft(traj, nturns=None):
        """Calculate the Discrete Fourier Transform (DFT) of the history
          matrix.

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
        peaks_tunes = _np.array([tunes[idc] for idc in idcs], dtype=float)
        return peaks_tunes

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

    def plot_spectrum(self, bpm_idx=None):
        """."""
        fig, ax = _mplt.subplots(1, 2, sharey=True)
        for i, plane in enumerate(['x', 'y']):
            abs_dft = _np.abs(self.data['dft'+plane])[:, bpm_idx]
            tunes = _np.fft.rfftfreq(n=self.data['traj'+plane].shape[0])
            ax[i].plot(tunes, abs_dft)
            ax[i].set_xlabel(f'{plane} tunes')
            peak_tune = tunes[abs_dft.argmax()]
            ax[i].axvline(x=peak_tune, color='r', linestyle='--', alpha=0.6,
                          label=f'peak @ {peak_tune:.3f}')
            ax[i].legend()
        fig.supylabel(f'abs(DFT) [mm]')
        fig.tight_layout()
        _mplt.show()

    def plot_trajctories(self, bpm_idx=None, from_turn=None, to_turn=None):
        """."""
        fig, ax = _mplt.subplots(1, 2, sharey=False)
        for i, plane in enumerate(['x', 'y']):
            ax[i].plot(self.data['traj'+plane][:, bpm_idx])
            ax[i].set_title(f'{plane} trajectory')
            from_t, to_t = from_turn[i], to_turn[i]
            ax[i].set_xlim(from_t, to_t+1)
            x_ticks = _np.arange(from_t, to_t+1, step=(to_t-from_t)//5)
            ax[i].set_xticks(x_ticks)
        fig.supylabel(f'trajctory [mm]')
        fig.supxlabel(f'turns')
        fig.tight_layout()
        _mplt.show()

    def evaluate_fit(self, traj='xy', from_turn=(0, 0), to_turn=(20, 20)):
        """."""
        for i, axis in enumerate(traj):
            amps = self.data['fitted_amps'+axis]
            tunes = self.data['tunes'+axis]
            phases = self.data['fitted_phases'+axis]
            hist_mat = self.data['traj'+axis][from_turn[i]:to_turn[i] + 1, :]
            nturns = hist_mat.shape[0]
            n = _np.arange(nturns)
            fit = _np.zeros_like(hist_mat)
            for i, (amp, tune, phase) in enumerate(zip(amps, tunes, phases)):
                fit[:, i] = TbTData()._TbT_model(n, amp, tune, phase)
            diff = hist_mat - fit
            bpms_rms_error = diff.std(axis=0)
            avg_rms_errors = bpms_rms_error.mean()
            print(f'average RMS errors for BPMs time-series: {avg_rms_errors}')
            self.data['fit'+axis] = fit

    def compare_fit_with_data(
            self, bpm_idx=0, traj='xy', from_turn=(0, 0), to_turn=(20, 20)):
        """."""
        fig, ax = _mplt.subplots(1, len(traj), sharey=False)
        for i, plane in enumerate(traj):
            from_t, to_t = from_turn[i], to_turn[i]
            arg = _np.arange(from_t, to_t+1)
            ax[i].plot(
                arg,
                self.data['traj'+plane][from_turn[i]:to_turn[i] + 1, bpm_idx],
                'o-', color='blue')
            ax[i].plot(
                arg,
                self.data['fit'+plane][:, bpm_idx], 'o--', color='red')
            ax[i].set_title(f'{plane} trajectory')
            ax[i].set_xlim(from_t, to_t+1)
            x_ticks = _np.arange(from_t, to_t+1, step=(to_t-from_t)//5)
            ax[i].set_xticks(x_ticks)
        fig.supylabel(f'trajectory [mm]')
        fig.supxlabel(f'turns')
        fig.tight_layout()
        _mplt.show()


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
