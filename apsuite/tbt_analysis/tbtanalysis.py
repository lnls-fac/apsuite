#!/usr/bin/env python-sirius
"""."""

import pickle as _pickle
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.optimize as _opt

from pyaccel.naff import naff_general as _naff_general
from ..utils import FrozenClass as _FrozenClass


class TbTAnalysis(_FrozenClass):
    """."""

    KICKTYPE_X = 'X'
    KICKTYPE_Y = 'Y'
    NOM_CHROMX = 2.5
    NOM_CHROMY = 2.5
    NOM_CHROMX_ERR = 0.0
    NOM_CHROMY_ERR = 0.0

    # https://wiki-sirius.lnls.br/mediawiki/index.php/Parameter:SI_optics_natural_energy_spread
    NOM_ESPREAD = 0.08436/100

    def __init__(self, kicktype=None, data=None, data_fname=None):
        """."""
        # --- data attributes ---
        self._data = data
        self._data_fname = data_fname

        # --- select type of tbtanalysis: 'X' or 'Y'
        self._select_kicktype = kicktype # This will be updated with data
        
        # --- data selection attributes ---
        self._select_idx_bpm = 0
        self._select_idx_kick = 0
        self._select_idx_turn_start = 0
        self._select_idx_turn_stop = None
        
        self._tunes_frac = 0.0
        self._tunes_frac_err = 0.0
        
        self._rx_offset = None
        self._rx0 = 0.0
        self._tunex_frac = 0.0
        self._chromx_decoh = 0.0
        self._chromx = TbTAnalysis.NOM_CHROMX
        self._mux = 0.0
        self._rx0_err = 0.0
        self._tunex_frac_err = 0.0
        self._chromx_decoh_err = 0.0
        self._chromx_err = 0.0
        self._mux_err = 0.0

        self._ry_offset = None
        self._ry0 = 0.0
        self._tuney_frac = 0.0
        self._chromy_decoh = 0.0
        self._chromy = TbTAnalysis.NOM_CHROMY
        self._muy = 0.0
        self._ry0_err = 0.0
        self._tuney_frac_err = 0.0
        self._chromy_decoh_err = 0.0
        self._chromy_err = 0.0
        self._muy_err = 0.0

        if not self._data and self._data_fname:
            self.data_load_raw()

        if self._select_kicktype is None:
            self._select_kicktype = self._data.get('kicktype', TbTAnalysis.KICKTYPE_X)
        self.chromx = self._data.get('chromx', TbTAnalysis.NOM_CHROMX)
        self.chromy = self._data.get('chromy', TbTAnalysis.NOM_CHROMY)
        self.chromx_err = self._data.get('chromx_err', TbTAnalysis.NOM_CHROMX_ERR)
        self.chromy_err = self._data.get('chromy_err', TbTAnalysis.NOM_CHROMY_ERR)
        self.select_idx_turn_stop = self.data_nr_turns
        self._select_update_offsets()

        # self.fit_simulann_reset()

        # load parameters of 2020-05-05 data (idx_bpm 0, idx_kick 0)
        # self.load_parameters_20200505()
        # self._init_analysis()

        # nominal energy spread
        # self.espread = TbTAnalysis.NOM_ESPREAD

        # freeze class attributes, as to alert class users of wrong settler names used by mistake
        self._freeze()

    # --- data methods ---

    @property
    def data_trajsum(self):
        """Return trajsum data."""
        if 'trajsum' in self._data:
            return self._data['trajsum']
        return None

    @property
    def data_kicks(self):
        """Return kick values."""
        if 'kicks' in self._data:
            return self._data['kicks']
        return None

    @property
    def data_nr_kicks(self):
        """Return number of kicks."""
        if not self._data:
            return None
        return self._data['trajx'].shape[0]

    @property
    def data_nr_turns(self):
        """Return number of turns in data."""
        if not self._data:
            return None
        return self._data['trajx'].shape[1]

    @property
    def data_nr_bpms(self):
        """."""
        if not self._data:
            return None
        return self._data['trajx'].shape[2]

    def data_load_raw(self, fname=None):
        """Load raw data from pick file."""
        if fname is None:
            fname = self._data_fname
        else:
            self._data_fname = fname
        with open(fname, 'rb') as fil:
            was_none = self._data is None
            self._data = _pickle.load(fil)
            if was_none:
                self._select_idx_kick = 0
                self._select_idx_turn_start = 0
                self._select_idx_turn_stop = self.data_nr_turns
                self._select_idx_bpm = 0
            else:
                self._select_idx_kick = min(self._select_idx_kick, self.data_nr_kicks)
                self._select_idx_turn_start = min(self._select_idx_turn_start, self.data_nr_turns)
                self._select_idx_turn_stop = min(self._select_idx_turn_stop, self.data_nr_turns)
                self._select_idx_bpm = min(self._select_idx_bpm, self.data_nr_bpms)

        if self._select_kicktype is None:
            self._select_kicktype = self._data.get('kicktype', TbTAnalysis.KICKTYPE_X)
        self._select_update_offsets()

    @property
    def data_trajx(self):
        """Return trajx data."""
        if 'trajx' in self._data:
            return self._data['trajx']
        return None

    @property
    def data_trajy(self):
        """Return trajy data."""
        if 'trajy' in self._data:
            return self._data['trajy']
        return None

    # --- data selection methods for analysis ---

    @property
    def select_kicktype(self):
        """Return selected kick type for analysis, either 'X' or 'Y'."""
        return self._select_kicktype

    @select_kicktype.setter
    def select_kicktype(self, value):
        """Set selected kicktype."""
        self._select_kicktype = value

    @property
    def select_idx_kick(self):
        """Return selected kick index."""
        return self._select_idx_kick

    @select_idx_kick.setter
    def select_idx_kick(self, value):
        """Set selected kick index."""
        self._select_idx_kick = value
        self._select_update_offsets()

    @property
    def select_idx_bpm(self):
        """Return selected bpm index."""
        return self._select_idx_bpm

    @select_idx_bpm.setter
    def select_idx_bpm(self, value):
        """Set selected bpm index."""
        self._select_idx_bpm = value
        self._select_update_offsets()

    @property
    def select_idx_turn_start(self):
        """Return selected turn start index."""
        return self._select_idx_turn_start

    @select_idx_turn_start.setter
    def select_idx_turn_start(self, value):
        """Set selected turn start index."""
        self._select_idx_turn_start = value
        self._select_update_offsets()

    @property
    def select_idx_turn_stop(self):
        """Return selected turn stop index."""
        return self._select_idx_turn_stop

    @select_idx_turn_stop.setter
    def select_idx_turn_stop(self, value):
        """Set selected turn stop index."""
        self._select_idx_turn_stop = value
        self._select_update_offsets()

    def select_get_traj(self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            traj_mea = self.data_trajx[select_idx_kick, turns_sel, select_idx_bpm]
        else:
            traj_mea = self.data_trajy[select_idx_kick, turns_sel, select_idx_bpm]
        return traj_mea

    # --- beam parameters ---
      
    @property
    def tunes_frac(self):
        """Return fitting parameter tunes_frac."""
        return self._tunes_frac

    @tunes_frac.setter
    def tunes_frac(self, value):
        """Set fitting parameter tunes_frac."""
        self._tunes_frac = value

    @property
    def espread(self):
        """."""
        # chrom_decoh = 2 * chrom * espread / tunes_frac
        if self._select_kicktype != TbTAnalysis.KICKTYPE_Y:
            return self.chromx_decoh * self.tunes_frac / 2 / self.chromx
        else:
            return self.chromy_decoh * self.tunes_frac / 2 / self.chromy

    @espread.setter
    def espread(self, value):
        """."""
        if self._select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self._chromx_decoh = 2 * self.chromx * value / self.tunes_frac
        else:
            self._chromy_decoh = 2 * self.chromy * value / self.tunes_frac

    # --- beam parameter errors ---

    @property
    def tunes_frac_err(self):
        """Return fitting parameter tunes_frac_err."""
        return self._tunes_frac_err

    @tunes_frac_err.setter
    def tunes_frac_err(self, value):
        """Set fitting parameter tunes_frac_err."""
        self._tunes_frac_err = value

    @property
    def espread_err(self):
        """."""
        if self._select_kicktype != TbTAnalysis.KICKTYPE_Y:
            el1 = (self.chromx_decoh_err * self.tunes_frac/2/self.chromx)**2
            el2 = (self.tunes_frac_err * self.chromx_decoh/2/self.chromx)**2
            el3 = (self.chromx_err * self.chromx_decoh*self.tunes_frac/2/self.chromx**2)**2
        else:
            el1 = (self.chromy_decoh_err * self.tunes_frac/2/self.chromy)**2
            el2 = (self.tunes_frac_err * self.chromy_decoh/2/self.chromy)**2
            el3 = (self.chromy_err * self.chromy_decoh*self.tunes_frac/2/self.chromy**2)**2
        return _np.sqrt(el1 + el2 + el3)

    # --- horizontal beam parameters ---

    @property
    def rx_offset(self):
        """Return offset value of trajx selected data."""
        return self._rx_offset

    @property
    def rx0(self):
        """."""
        return self._rx0

    @rx0.setter
    def rx0(self, value):
        """."""
        self._rx0 = value

    @property
    def tunex_frac(self):
        """Return fitting parameter tunex_frac."""
        """Return offset value of trajy selected data."""
        return self._tunex_frac

    @tunex_frac.setter
    def tunex_frac(self, value):
        """Set fitting parameter tunex_frac."""
        self._tunex_frac = value

    @property
    def chromx_decoh(self):
        """."""
        return self._chromx_decoh

    @chromx_decoh.setter
    def chromx_decoh(self, value):
        """."""
        self._chromx_decoh = value

    @property
    def chromx(self):
        """."""
        return self._chromx

    @chromx.setter
    def chromx(self, value):
        """."""
        self._chromx = value

    @property
    def mux(self):
        """."""
        return self._mux

    @mux.setter
    def mux(self, value):
        """."""
        self._mux = value

    # --- horizontal beam parameter errors ---

    @property
    def rx0_err(self):
        """."""
        return self._rx0_err

    @rx0_err.setter
    def rx0_err(self, value):
        """."""
        self._rx0_err = value

    @property
    def tunex_frac_err(self):
        """Return fitting parameter tunex_frac_err."""
        return self._tunex_frac_err

    @tunex_frac_err.setter
    def tunex_frac_err(self, value):
        """Set fitting parameter tunex_frac_err."""
        self._tunex_frac_err = value

    @property
    def chromx_decoh_err(self):
        """."""
        return self._chromx_decoh_err

    @chromx_decoh_err.setter
    def chromx_decoh_err(self, value):
        """."""
        self._chromx_decoh_err = value

    @property
    def chromx_err(self):
        """."""
        return self._chromx_err

    @chromx_err.setter
    def chromx_err(self, value):
        """."""
        self._chromx_err = value

    @property
    def mux_err(self):
        """."""
        return self._mux_err

    @mux_err.setter
    def mux_err(self, value):
        """."""
        self._mux_err = value

    # --- vertical beam parameters ---

    @property
    def ry_offset(self):
        return self._ry_offset
   
    @property
    def ry0(self):
        """."""
        return self._ry0

    @ry0.setter
    def ry0(self, value):
        """."""
        self._ry0 = value

    @property
    def tuney_frac(self):
        """Return fitting parameter tuney_frac."""
        return self._tuney_frac

    @tuney_frac.setter
    def tuney_frac(self, value):
        """Set fitting parameter tuney_frac."""
        self._tuney_frac = value

    @property
    def chromy_decoh(self):
        """."""
        return self._chromy_decoh

    @chromy_decoh.setter
    def chromy_decoh(self, value):
        """."""
        self._chromy_decoh = value

    @property
    def chromy(self):
        """."""
        return self._chromy

    @chromy.setter
    def chromy(self, value):
        """."""
        self._chromy = value

    @property
    def muy(self):
        """."""
        return self._muy

    @muy.setter
    def muy(self, value):
        """."""
        self._muy = value

    # --- vertical beam parameter errors ---

    @property
    def ry0_err(self):
        """."""
        return self._ry0_err

    @ry0_err.setter
    def ry0_err(self, value):
        """."""
        self._ry0_err = value

    @property
    def tuney_frac_err(self):
        """Return fitting parameter tuney_frac_err."""
        return self._tuney_frac_err

    @tuney_frac_err.setter
    def tuney_frac_err(self, value):
        """Set fitting parameter tuney_frac_err."""
        self._tuney_frac_err = value

    @property
    def chromy_decoh_err(self):
        """."""
        return self._chromy_decoh_err

    @chromy_decoh_err.setter
    def chromy_decoh_err(self, value):
        """."""
        self._chromy_decoh_err = value

    @property
    def chromy_err(self):
        """."""
        return self._chromy_err

    @chromy_err.setter
    def chromy_err(self, value):
        """."""
        self._chromy_err = value

    @property
    def muy_err(self):
        """."""
        return self._muy_err

    @muy_err.setter
    def muy_err(self, value):
        """."""
        self._muy_err = value

    # --- search methods ---

    def search_tunes(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None,
            peak_frac=0.999,
            plot_flag=False):
        """."""
        # selection of data to analyse
        select_idx_turn_stop = \
            self.data_nr_turns if select_idx_turn_stop is None else select_idx_turn_stop

        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        
        if select_idx_kick != self.select_idx_kick:
            self.select_idx_kick = select_idx_kick
        if select_idx_bpm != self.select_idx_bpm:
            self.select_idx_bpm = select_idx_bpm

        traj_mea = self.select_get_traj(
            select_idx_kick=select_idx_kick, select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start, select_idx_turn_stop=select_idx_turn_stop)
        
        # search tunes using FFT on selected data
        title = 'FFT, nr_turns: {}, idx_kick: {}, idx_bpm: {}'.format(
            select_idx_turn_stop, select_idx_kick, select_idx_bpm)
        fft, tune, tunes = TbTAnalysis.calc_fft(traj_mea, peak_frac, plot_flag, title)
        _ = fft

        # naff
        # intv = select_idx_turn_start + int((self.data_nr_turns - 1 ) // 6)
        # size = intv*6 + 1
        # signal = traj_mea[:size] - _np.mean(traj_mea[:size])
        # freqs, fourier = _naff_general(signal=signal, is_real=True, nr_ff=1, window=1)
        # tune = freqs

        # set tunes
        self.tunes_frac = tunes
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.tunex_frac = tune
        else:
            self.tuney_frac = tune

    def search_r0_mu(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """."""
        select_idx_turn_stop = 40 if select_idx_turn_stop is None else select_idx_turn_stop

        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)

        if select_idx_kick != self.select_idx_kick:
            self.select_idx_kick = select_idx_kick
        if select_idx_bpm != self.select_idx_bpm:
            self.select_idx_bpm = select_idx_bpm

        params, offset, traj_mea = self._get_fit_inputs(
            select_idx_kick, select_idx_bpm,
            select_idx_turn_start, select_idx_turn_stop)

        traj_mea = traj_mea - offset
        args = [select_idx_turn_start, select_idx_turn_stop, offset]

        _, cn_, sn_, *_ = TbTAnalysis.calc_traj_chrom(params, *args)
        # linear algebra to solve for mu and r0
        a11, a12 = + _np.sum(cn_ * cn_), - _np.sum(cn_ * sn_)
        a21, a22 = - a12, - _np.sum(sn_ * sn_)
        b11, b21 = _np.sum(traj_mea * cn_), _np.sum(traj_mea * sn_)
        mata = _np.array([[a11, a12], [a21, a22]])
        matb = _np.array([b11, b21])
        c0_, s0_ = _np.linalg.solve(mata, matb)
        r0 = _np.sqrt(c0_**2 + s0_**2)
        mu = _np.arctan(s0_ / c0_)

        # check if inversion is necessary
        cos1 = cn_ * _np.cos(mu) - sn_ * _np.sin(mu)  # cos(2 * pi * tune_frac * turn + mu)
        cos2 = cn_ * _np.cos(mu + _np.pi) - sn_ * _np.sin(mu + _np.pi)  # cos(2 * pi * tune_frac * turn + mu + pi)
        vec1 = r0 * cos1 - traj_mea
        vec2 = r0 * cos2 - traj_mea
        if _np.sum(vec2**2) < _np.sum(vec1**2):
            mu += _np.pi

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.rx0 = r0
            self.mux = mu
        else:
            self.ry0 = r0
            self.muy = mu

    def search_init(self, plot_flag=False):
        """."""
        # search tunes with all data
        self.select_idx_turn_stop = self.data_nr_turns
        self.search_tunes(peak_frac = 0.999, plot_flag=False)

        # search beta and mu with 3 periods of betatron oscillations
        self.espread = TbTAnalysis.NOM_ESPREAD * 1.0
        self.select_idx_turn_stop = int(3 / self.tunex_frac)
        self.search_r0_mu()

    # --- fitting methods: common ---

    def fit_leastsqr(self):
        """."""
        init_params, offset, traj_mea = self._get_fit_inputs()
        
        fit_params = _opt.least_squares(
            fun=self._calc_residue_vector,
            x0=init_params,
            args=(
                traj_mea,
                self.select_idx_turn_start,
                self.select_idx_turn_stop,
                offset),
            method='lm')

        fit_errors = TbTAnalysis._calc_leastsqr_fitting_error(fit_params)

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.tunes_frac = fit_params['x'][0]
            self.tunex_frac = fit_params['x'][1]
            self.chromx_decoh = fit_params['x'][2]
            self.rx0 = fit_params['x'][3]
            self.mux = fit_params['x'][4]
            # errors
            self.tunes_frac_err = fit_errors[0]
            self.tunex_frac_err = fit_errors[1]
            self.chromx_decoh_err = fit_errors[2]
            self.rx0_err = fit_errors[3]
            self.mux_err = fit_errors[4]
        else:
            self.tunes_frac = fit_params['x'][0]
            self.tuney_frac = fit_params['x'][1]
            self.chromy_decoh = fit_params['x'][2]
            self.ry0 = fit_params['x'][3]
            self.muy = fit_params['x'][4]
            # errors
            self.tunes_frac_err = fit_errors[0]
            self.tuney_frac_err = fit_errors[1]
            self.chromy_decoh_err = fit_errors[2]
            self.ry0_err = fit_errors[3]
            self.muy_err = fit_errors[4]

    def fit_residue(self):
        """."""
        params, offset, traj_mea = self._get_fit_inputs()

        residue_vec = self._calc_residue_vector(
            params, traj_mea, self.select_idx_turn_start, self.select_idx_turn_stop, offset)

        return _np.sqrt(_np.sum(residue_vec**2)/len(residue_vec))

    def fit_trajs(self):
        """."""
        params, offset, traj_mea = self._get_fit_inputs()
        args = [self.select_idx_turn_start, self.select_idx_turn_stop, offset]
        traj_fit, *_ = TbTAnalysis.calc_traj_chrom(params, *args)
        return traj_mea, traj_fit

    def fit_run(self, plot_flag=False):
        """."""
        # initial search for tunes, r0 and mu
        self.search_init(plot_flag)

        # search all parameters with 1 period of synchrotron oscillations
        self.select_idx_turn_stop = int(1 / self.tunes_frac)
        self.fit_leastsqr()

    # --- analysis methods ---

    def analysis_run(self, select_idx_kick=0, bpm_indices=None):
        """."""
        if bpm_indices is None:
            bpm_indices = _np.arange(self.data_nr_bpms)
        vec = _np.zeros(len(bpm_indices))
        rx0, mux, tunex_frac, tunes_frac, espread, residue = 0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec
        self.select_idx_kick = select_idx_kick
        for idx, idx_bpm in enumerate(bpm_indices):
            self.select_idx_bpm = idx_bpm
            self.fit_run()
            # store params
            rx0[idx] = self.rx0
            mux[idx] = self.mux
            tunex_frac[idx] = self.tunex_frac
            tunes_frac[idx] = self.tunes_frac - int(self.tunes_frac)
            espread[idx] = self.espread
            residue[idx] = self.fit_residue()

        # unwrap phase
        mux = _np.unwrap(mux)
        changed = True
        while changed:
            changed = False
            for i in range(1,len(mux)):
                if mux[i] < mux[i-1]:
                    changed = True
                    mux[i:] += 0.5

        return bpm_indices, rx0, mux, tunex_frac, tunes_frac, espread, residue


    # --- aux. public methods ---

    @staticmethod
    def calc_fft(data, peak_frac=0.7, plot_flag=False, title=None):
        """."""
        # print()
        data = data - _np.mean(data)
        fft = _np.abs(_np.fft.rfft(data))
        idx = _np.argmax(fft)
        tunex = idx / len(data)
        tunes1, tunes2 = 0.003, 0.003
        for i in range(idx+1, len(fft)):
            val1n, val0, val1p = fft[i-1:i+2]
            # print(val1n, val0, val1p)
            if val1n/val0 < peak_frac and val1p/val0 < peak_frac:
                tunes1 = i / len(data) - tunex
                break
        for i in range(idx-1, len(fft), -1):
            val1n, val0, val1p = fft[i-1:i+2]
            if val1n/val0 < peak_frac and val1p/val0 < peak_frac:
                tunes2 = tunex - i / len(data)
                break
        tunes = 0.5 * (tunes1 + tunes2)
        # print(tunex - tunes2, tunex + tunes1)
        if plot_flag:
            ind = _np.arange(len(fft)) / len(data)
            _plt.plot(ind, fft, '-b')
            _plt.plot(ind, fft, 'ob')
            _plt.xlabel('tunex')
            _plt.ylabel('Abs FFT')
            _plt.grid()
            if title:
                title += ' tunex: {:.5f}  tunes: {:.5f}'.format(tunex, tunes)
                _plt.title(title)
            _plt.show()
        return fft, tunex, tunes

    @staticmethod
    def calc_traj_chrom(params, *args):
        """BPM averaging due to longitudinal dynamics decoherence.

        nu ~ nu0 + chrom * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        tunes_frac = params[0]
        tune_frac = params[1]
        chromx_decoh = params[2]
        r0 = params[3]
        mu = params[4]

        select_idx_turn_start, select_idx_turn_stop, offset = args
        turn = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        # cos = _np.cos(2 * _np.pi * tune_frac * turn + mu)
        cn = _np.cos(2 * _np.pi * tune_frac * turn)
        sn = _np.sin(2 * _np.pi * tune_frac * turn)
        cos = cn * _np.cos(mu) - sn * _np.sin(mu)
        alp = chromx_decoh * _np.sin(_np.pi * tunes_frac * turn)
        exp = _np.exp(-alp**2/2.0)
        traj = r0 * exp * cos + offset
        return traj, cn, sn, alp, exp, cos

    def __str__(self):
        """."""
        # self._self2simulann()
        sel_nr_turns = self.select_idx_turn_stop - self.select_idx_turn_start
        rst = ''
        rst += '.--- Data -------\n'
        rst += '| nr_kicks       : {}\n'.format(self.data_nr_kicks)
        rst += '| nr_turns       : {}\n'.format(self.data_nr_turns)
        rst += '| nr_bpms        : {}\n'.format(self.data_nr_bpms)
        rst += '| rx_offset      : {:.5f} um\n'.format(self.rx_offset)
        rst += '|--- Selection -- \n'
        rst += '| idx_kick       : {}\n'.format(self.select_idx_kick)
        rst += '| idx_bpm        : {}\n'.format(self.select_idx_bpm)
        rst += '| idx_turn_start : {}\n'.format(self.select_idx_turn_start)
        rst += '| idx_turn_stop  : {}\n'.format(self.select_idx_turn_stop)
        rst += '| sel_nr_turns   : {}\n'.format(sel_nr_turns)
        rst += '| kick           : {:+.1f} urad\n'.format(
            self.data_kicks[self.select_idx_kick]*1e3)
        # rst += '| niter          : {}\n'.format(self.fit_simulann_niter)
        rst += '|-- Fit Params -- \n'
        rst += '| espread        : {:.5f} %\n'.format(100*self.espread)
        rst += '| tunes_frac     : {:.6f}\n'.format(self.tunes_frac)
        rst += '| ----- X ------ \n'
        rst += '| chromx         : {:+.3f} ± {:.3f}\n'.format(self.chromx, self.chromx_err)
        rst += '| tunex_frac     : {:.6f}\n'.format(self.tunex_frac)
        rst += '| rx0            : {:.5f} um\n'.format(self.rx0)
        rst += '| mux            : {:.5f} rad\n'.format(self.mux)
        rst += '| chromx_decoh   : {:.5f}\n'.format(self.chromx_decoh)
        rst += '| ----- Y ------ \n'
        rst += '| chromy         : {:+.3f} ± {:.3f}\n'.format(self.chromy, self.chromy_err)
        rst += '| tuney_frac     : {:.6f}\n'.format(self.tuney_frac)
        rst += '| ry0            : {:.5f} um\n'.format(self.ry0)
        rst += '| muy            : {:.5f} rad\n'.format(self.muy)
        rst += '| chromy_decoh   : {:.5f}\n'.format(self.chromy_decoh)
        rst += '|--------------- \n'
        rst += '| residue        : {:.5f} um\n'.format(self.fit_residue())
        rst += ' ---------------- \n'

        return rst

    # --- private methods ---

    def _select_update_offsets(self):
        """."""
        kicktype = self.select_kicktype
        self.select_kicktype = TbTAnalysis.KICKTYPE_X
        trajx = self.select_get_traj()
        self._rx_offset = _np.mean(trajx)
        self.select_kicktype = TbTAnalysis.KICKTYPE_Y
        trajy = self.select_get_traj()
        self._ry_offset = _np.mean(trajy)
        self.select_kicktype = kicktype

    def _select_get_traj(self, select_type=None,
                  select_idx_kick=None, select_idx_bpm=None,
                  select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_type = 'X' if select_type is None else select_type
        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        if select_type == 'X':
            traj_mea = self.data_trajx[select_idx_kick, turns_sel, select_idx_bpm]
        else:
            traj_mea = self.data_trajy[select_idx_kick, turns_sel, select_idx_bpm]
        return traj_mea

    def _get_sel_args(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        select_idx_kick = self.select_idx_kick if select_idx_kick is None else select_idx_kick
        select_idx_turn_start = \
            self.select_idx_turn_start if select_idx_turn_start is None else select_idx_turn_start
        select_idx_turn_stop = \
            self.select_idx_turn_stop if select_idx_turn_stop is None else select_idx_turn_stop
        select_idx_bpm = self.select_idx_bpm if select_idx_bpm is None else select_idx_bpm
        return select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop

    def _get_fit_params(self):
        """."""
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            tune_frac = self.tunex_frac
            chrom_decoh = self.chromx_decoh
            r0 = self.rx0
            mu = self.mux
        else:
            tune_frac = self.tuney_frac
            chrom_decoh = self.chromy_decoh
            r0 = self.ry0
            mu = self.muy

        params = [
            self.tunes_frac, tune_frac,
            chrom_decoh, r0, mu]

        return params

    def _get_fit_inputs(self,
        select_idx_kick=None, select_idx_bpm=None,
        select_idx_turn_start=None, select_idx_turn_stop=None):
        """."""
        params = self._get_fit_params()
        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        traj_mea = self.select_get_traj(
            select_idx_kick=select_idx_kick,
            select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop)
        offset = _np.mean(traj_mea)
        return params, offset, traj_mea

    @staticmethod
    def _calc_leastsqr_fitting_error(fit_params):
        """."""
        # based on fitting error calculation of scipy.optimization.curve_fit
        # do Moore-Penrose inverse discarding zero singular values.
        _, smat, vhmat = _np.linalg.svd(
            fit_params['jac'], full_matrices=False)
        thre = _np.finfo(float).eps * max(fit_params['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = _np.dot(vhmat.T / (smat*smat), vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_params['fun'])
        cost = 2 * fit_params['cost']  # res.cost is half sum of squares!
        popt = fit_params['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(_np.nan)
            print('# of fitting parameters larger than # of data points!')
        return _np.sqrt(_np.diag(pcov))

    @staticmethod
    def _calc_residue_vector(
            params, traj_mea,
            select_idx_turn_start, select_idx_turn_stop, offset):
        """."""
        args = [select_idx_turn_start, select_idx_turn_stop, offset]
        traj_fit, *_ = TbTAnalysis.calc_traj_chrom(params, *args)
        traj_res = traj_mea - traj_fit
        return traj_res
