#!/usr/bin/env python-sirius
"""."""

import pickle as _pickle
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import least_squares as _least_squares

from ..utils import FrozenClass as _FrozenClass
from .tbtsimulann import TbTSimulAnneal as _TbTSimulAnneal


class TbTAnalysis(_FrozenClass):
    """."""

    KICKTYPE_X = 'X'
    KICKTYPE_Y = 'Y'
    KICKTYPE_XY = 'XY'
    NOM_CHROMX = 2.5
    NOM_CHROMY = 2.5
    NOM_ESPREAD = 8.515e-4

    def __init__(self, kicktype=None, data=None, data_fname=None):
        """."""
        # --- data attributes ---
        self._data = data
        self._data_fname = data_fname

        # --- select type of tbtanalysis: 'X' or 'Y'
        self._select_kicktype = kicktype  # This will be updated with data

        # --- data selection attributes ---
        self._select_idx_bpm = 0
        self._select_idx_kick = 0
        self._select_idx_turn_start = 0
        self._select_idx_turn_stop = 500

        self._simulann = None
        self._analysis = None

        self._rx_offset = None
        self._rx0 = 0.0
        self._tunex_frac = 0.0
        self._chromx_decoh = 0.0
        self._chromx = 0.0
        self._mux = 0.0
        self._rx0_err = 0.0
        self._tunex_frac_err = 0.0
        self._chromx_err = 0.0
        self._mux_err = 0.0

        self._ry_offset = None
        self._ry0 = 0.0
        self._tuney_frac = 0.0
        self._chromy_decoh = 0.0
        self._chromy = 0.0
        self._muy = 0.0
        self._ry0_err = 0.0
        self._tuney_frac_err = 0.0
        self._chromy_err = 0.0
        self._muy_err = 0.0

        if not self._data and self._data_fname:
            self.data_load_raw()
        else:
            if self._select_kicktype is None:
                self._select_kicktype = self._data.get('kicktype', TbTAnalysis.KICKTYPE_X)
            self._select_update_offsets()

        self.fit_simulann_reset()

        # load parameters of 2020-05-05 data (idx_bpm 0, idx_kick 0)
        self.load_parameters_20200505()
        self._init_analysis()

        # freeze class attributes, as to alert class users of wrong settler
        # names used by mistake
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

    def select_get_traj(
            self, select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_idx_kick, select_idx_bpm, select_idx_turn_start, \
            select_idx_turn_stop = self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start,
                select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            traj_mea = self.data_trajx[
                select_idx_kick, turns_sel, select_idx_bpm]
        else:
            traj_mea = self.data_trajy[
                select_idx_kick, turns_sel, select_idx_bpm]
        return traj_mea

    # --- objects ---

    @property
    def simulann(self):
        """Return TbTSimuAnnel object."""
        self._self2simulann()
        return self._simulann

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
        return self._espread

    @espread.setter
    def espread(self, value):
        """."""
        self._espread = value

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
        return self._espread_err

    @espread_err.setter
    def espread_err(self, value):
        """."""
        self._espread_err = value

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
        """."""
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
            select_idx_kick=select_idx_kick,
            select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop)

        # search tunes using FFT on selected data
        title = 'FFT, nr_turns: {}, idx_kick: {}, idx_bpm: {}'.format(
            select_idx_turn_stop, select_idx_kick, select_idx_bpm)
        fft, tune, tunes = TbTAnalysis.calc_fft(traj_mea, plot_flag, title)
        _ = fft

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
        select_idx_turn_stop = 20 if select_idx_turn_stop \
            is None else select_idx_turn_stop

        select_idx_kick, select_idx_bpm, select_idx_turn_start, \
            select_idx_turn_stop = self._get_sel_args(
                select_idx_kick,
                select_idx_bpm,
                select_idx_turn_start,
                select_idx_turn_stop)

        if select_idx_kick != self.select_idx_kick:
            self.select_idx_kick = select_idx_kick
        if select_idx_bpm != self.select_idx_bpm:
            self.select_idx_bpm = select_idx_bpm

        traj_mea = self.select_get_traj(
            select_idx_kick=select_idx_kick,
            select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop)
        traj_mea = traj_mea - _np.mean(traj_mea)

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            tune_frac = self.tunex_frac
        else:
            tune_frac = self.tuney_frac

        turn = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        cn_ = _np.cos(2 * _np.pi * tune_frac * turn)
        sn_ = _np.sin(2 * _np.pi * tune_frac * turn)
        a11, a12 = + _np.sum(cn_ * cn_), - _np.sum(cn_ * sn_)
        a21, a22 = - a12, - _np.sum(sn_ * sn_)
        b11, b21 = _np.sum(traj_mea * cn_), _np.sum(traj_mea * sn_)
        mata = _np.array([[a11, a12], [a21, a22]])
        matb = _np.array([b11, b21])
        c0_, s0_ = _np.linalg.solve(mata, matb)
        rx = _np.sqrt(c0_**2 + s0_**2)
        mu = _np.arcsin(s0_ / rx)

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.rx0 = rx
            self.mux = mu
        else:
            self.ry0 = rx
            self.muy = mu

    def search_chromx_decoh(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """."""
        select_idx_turn_stop = 500 if select_idx_turn_stop \
            is None else select_idx_turn_stop

        select_idx_kick, select_idx_bpm, select_idx_turn_start, \
            select_idx_turn_stop = self._get_sel_args(
                select_idx_kick,
                select_idx_bpm,
                select_idx_turn_start,
                select_idx_turn_stop)

        if select_idx_kick != self.select_idx_kick:
            self.select_idx_kick = select_idx_kick
        if select_idx_bpm != self.select_idx_bpm:
            self.select_idx_bpm = select_idx_bpm

        # print(select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)

        self.chromx_decoh = 0
        obj1 = self.fit_simulann_residue
        self.chromx_decoh = 0.1
        obj2 = self.fit_simulann_residue
        factor = 2.0 if obj2 < obj1 else 0.5

        # first find a parameter interval
        niter = 0
        while niter < 16:
            self.chromx_decoh *= factor
            obj1 = self.fit_simulann_residue
            print(self._chromx_decoh, obj1)
            if obj1 < obj2:
                obj2 = obj1
            else:
                break
            niter += 1
        par2 = self.chromx_decoh
        par1 = self.chromx_decoh / factor / factor
        pars = _np.linspace(par1, par2, 30)

        # get best value in interval
        best_par, best_obj = par2, self.fit_simulann_residue
        for par in pars:
            self.chromx_decoh = par
            obj = self.simulann.calc_obj_fun()
            # print(par, obj)
            if obj < best_obj:
                best_par = par
                best_obj = obj
        print(best_par)
        self.chromx_decoh = best_par

    def search_init(
            self, select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None,
            plot_flag=False):
        """."""
        self.search_tunes(
            select_idx_kick=select_idx_kick, select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop,
            plot_flag=plot_flag)

        self.search_r0_mu(
            select_idx_kick=select_idx_kick, select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop)

        # self.search_chromx_decoh(
        #     select_idx_kick=select_idx_kick, select_idx_bpm=select_idx_bpm,
        #     select_idx_turn_start=select_idx_turn_start, select_idx_turn_stop=select_idx_turn_stop)

        self.espread = TbTAnalysis.NOM_ESPREAD
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.chromx = TbTAnalysis.NOM_CHROMX
        else:
            self.chromy = TbTAnalysis.NOM_CHROMY

    # --- fitting methods: simulated annealing ---

    @property
    def fit_simulann_niter(self):
        """Return number of iterations for fitting."""
        return self._simulann.niter

    @fit_simulann_niter.setter
    def fit_simulann_niter(self, value):
        """Set number of iterations for fitting."""
        self._simulann.niter = value

    def fit_simulann_reset(self):
        """."""
        simulann = _TbTSimulAnneal(
            self, _TbTSimulAnneal.TYPES.CHROMX)
        self._simulann = simulann
        value_max = 1.1 * _np.amax(self.data_trajx)
        lower = _np.array([0.0, 0.0, 0.0, 0.0, -_np.pi])
        upper = _np.array([0.05, 0.5, 100.0, value_max, +_np.pi])
        delta = _np.array(
            [0.00003, 0.001, 0.01, 0.01*value_max, 2*_np.pi*0.01])
        self._self2simulann()
        simulann.niter = 1 if simulann.niter == 0 else simulann.niter
        simulann.limits_upper = upper
        simulann.limits_lower = lower
        simulann.deltas = delta

    def fit_simulann_start(self, niter=1, print_flag=True):
        """."""
        self.simulann.niter = niter
        self.simulann.start(print_flag=print_flag)
        self._simulann2self()

    def fit_simulann_run(self, niter=None, select_idx_bpm=None, select_idx_kick=None, log_flag=True):
        """."""
        niter = 30000 if niter is None else niter
        self.fit_simulann_reset()
        if select_idx_bpm:
            self.select_idx_bpm = select_idx_bpm
        if select_idx_kick:
            self.select_idx_kick = select_idx_kick
        simulann = self.simulann

        # search parameters
        niter_, nrpts = 0, 20
        factor = _np.exp(_np.log(0.1)/nrpts)
        simulann.niter = int(round(niter / nrpts))
        if log_flag:
            sfmt = '#{:02d}/{:02d} iter {:06d} : obj_func = {:7.2f} um'
            print(sfmt.format(0, nrpts, 0, simulann.calc_obj_fun()))
        for k in range(nrpts):
            simulann.start(print_flag=False)
            niter_ += simulann.niter
            if log_flag:
                print(sfmt.format(k, nrpts, niter_, simulann.calc_obj_fun()))
            simulann.deltas *= factor
        self._simulann2self()

    @property
    def fit_simulann_residue(self):
        """Return fit residue."""
        return self.simulann.calc_obj_fun()

    def fit_simulann_plot(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None,
            title=None):
        """."""
        select_idx_kick, select_idx_bpm, select_idx_turn_start, \
            select_idx_turn_stop = self._get_sel_args(
                select_idx_kick,
                select_idx_bpm,
                select_idx_turn_start,
                select_idx_turn_stop)

        traj_mea = self.select_get_traj(
            select_idx_kick=select_idx_kick,
            select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start,
            select_idx_turn_stop=select_idx_turn_stop)

        # parms = {
        #     'select_idx_turn_start': select_idx_turn_start,
        #     'select_idx_turn_stop': select_idx_turn_stop,
        #     'tunes_frac': self.tunes_frac,
        #     'rx_offset': self.rx_offset,
        #     'tunex_frac': self.tunex_frac,
        #     'chromx_decoh': self.chromx_decoh,
        #     'rx0': self.rx0,
        #     'mux': self.mux,
        #     'ry_offset': self.ry_offset,
        #     'tuney_frac': self.tuney_frac,
        #     'chromy_decoh': self.chromy_decoh,
        #     'ry0': self.ry0,
        #     'muy': self.muy,
        # }

        # if self.select_kicktype == self.KICKTYPE_Y:
        #     traj_fit = self.simann.calc_trajy(**parms)
        # else:
        #     traj_fit = self.simann.calc_trajx(**parms)

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            tunet_frac = self.tunex_frac
            chrom = self.chromx
            r0 = self.rx0
            mu = self.mux
            r_offset = self.rx_offset
        else:
            tunet_frac = self.tuney_frac
            chrom = self.chromy
            r0 = self.ry0
            mu = self.muy
            r_offset = self.ry_offset

        parms = [
            self.tunes_frac,
            tunet_frac, chrom,
            self.espread, r0, mu]
        args = [
            self.select_idx_turn_start,
            self.select_idx_turn_stop,
            r_offset]

        traj_fit = TbTAnalysis.calc_traj_chrom(parms, *args)
        # offset = self.rx_offset
        # args = [select_idx_turn_start, select_idx_turn_stop, offset]
        # trajx_fit = TbTAnalysis.calc_trajx_chromx(parms, *args)

        trajx_mea = self.select_get_trajx(
            select_idx_kick=select_idx_kick, select_idx_bpm=select_idx_bpm,
            select_idx_turn_start=select_idx_turn_start, select_idx_turn_stop=select_idx_turn_stop)

        parms = {
            'select_idx_turn_start': select_idx_turn_start,
            'select_idx_turn_stop': select_idx_turn_stop,
            'rx_offset': self.rx_offset,
            'tunes_frac': self.tunes_frac,
            'tunex_frac': self.tunex_frac,
            'chromx_decoh': self.chromx_decoh,
            'rx0': self.rx0,
            'mux': self.mux,
        }
        trajx_fit = self.simulann.calc_trajx(**parms)
        if title is None:
            title = \
                'kick idx: {:03d}, '.format(select_idx_kick) + \
                'bpm idx: {:03d}, '.format(select_idx_bpm)
        label_residue = 'Residue (rms: {:.2f} um)'.format(self.fit_simulann_residue)
        ind = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        traj_res = traj_mea - traj_fit
        _plt.plot(ind, traj_mea, color='C0', label='TbT raw data')
        _plt.plot(ind, traj_mea, 'o', color='C0')
        _plt.plot(ind, traj_fit, color='C1', label='Fitting')
        _plt.plot(ind, traj_fit, 'o', color='C1')
        _plt.plot(ind, traj_res, label=label_residue, color='C2')
        _plt.legend()
        _plt.xlabel('turn')
        _plt.ylabel('BPM Average [um]')
        _plt.title(title)
        _plt.grid()
        _plt.show()

    # --- fitting methods: least squared ---

    def fit_leastsqr_start(self):
        """."""
        init_params, offset, traj_mea = self._get_leastsqr_inputs()

        fit_params = _least_squares(
            fun=self._calc_leastsqr_residue,
            x0=init_params,
            args=(
                traj_mea,
                self.select_idx_turn_start,
                self.select_idx_turn_stop,
                offset),
            method='lm')

        fit_errors = TbTAnalysis._calc_leastsqr_fitting_error(fit_params)

        self.tunes_frac = fit_params['x'][0]
        self.tunes_frac_err = fit_errors[0]
        self.espread = fit_params['x'][3]
        self.espread_err = fit_errors[3]

        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            self.tunex_frac = fit_params['x'][1]
            self.chromx = fit_params['x'][2]
            self.chromx_decoh = 2*self.chromx*self.espread/self.tunes_frac
            self.rx0 = fit_params['x'][4]
            self.mux = fit_params['x'][5]

            self.tunex_frac_err = fit_errors[1]
            self.chromx_err = fit_errors[2]
            self.rx0_err = fit_errors[4]
            self.mux_err = fit_errors[5]
        else:
            self.tuney_frac = fit_params['x'][1]
            self.chromy = fit_params['x'][2]
            self.chromy_decoh = 2*self.chromy*self.espread/self.tunes_frac
            self.ry0 = fit_params['x'][4]
            self.muy = fit_params['x'][5]

            self.tuney_frac_err = fit_errors[1]
            self.chromy_err = fit_errors[2]
            self.ry0_err = fit_errors[4]
            self.muy_err = fit_errors[5]

    def fit_leastsqr_residue(self):
        """."""
        params, offset, traj_mea = self._get_leastsqr_inputs()

        return self._calc_leastsqr_residue(
            params, traj_mea, self.select_idx_turn_start, self.select_idx_turn_stop, offset)

    def fit_leastsqr_traj(self):
        """."""
        params, offset, traj_mea = self._get_leastsqr_inputs()
        args = [self.select_idx_turn_start, self.select_idx_turn_stop, offset]
        traj_fit = TbTAnalysis._calc_traj_chrom(params, *args)
        return traj_mea, traj_fit

    # --- analysis methods ---

    def analysis_all_bpms(self, niter=30000, fname=None, plot_flag=False):
        """."""
        self.load_parameters_20200505()
        self._init_analysis()
        for select_idx_bpm in range(0, self.data_nr_bpms):
            if select_idx_bpm > 0:
                self.select_idx_bpm = select_idx_bpm
                self.search_r0_mu()
                self.fit_simulann_run(niter=niter, log_flag=False)

            print('bpm idx: {:03d}, residue = {:.2f} um'.format(
                self.select_idx_bpm, self.fit_simulann_residue))
            print(self)
            if plot_flag:
                self.fit_simulann_plot()

            if fname is not None:
                self.analysis_save(fname)

    # --- aux. public methods ---

    @staticmethod
    def calc_fft(data, plot_flag=False, title=None):
        """."""
        data = data - _np.mean(data)
        fft = _np.abs(_np.fft.rfft(data))
        idx = _np.argmax(fft)
        tunex = idx / len(data)
        tunes1, tunes2 = 0.003, 0.003
        for i in range(idx+1, len(fft)):
            val1n, val0, val1p = fft[i-1:i+2]
            if val1n/val0 < 0.7 and val1p/val0 < 0.7:
                tunes1 = i / len(data) - tunex
                break
        for i in range(idx-1, len(fft), -1):
            val1n, val0, val1p = fft[i-1:i+2]
            if val1n/val0 < 0.7 and val1p/val0 < 0.7:
                tunes2 = tunex - i / len(data)
                break
        tunes = 0.5 * (tunes1 + tunes2)
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

    def __str__(self):
        """."""
        self._self2simulann()
        sel_nr_turns = self.select_idx_turn_stop - self.select_idx_turn_start
        rst = ''
        rst += '.--- TBT --------\n'
        rst += '| nr_kicks       : {}\n'.format(self.data_nr_kicks)
        rst += '| nr_turns       : {}\n'.format(self.data_nr_turns)
        rst += '| nr_bpms        : {}\n'.format(self.data_nr_bpms)
        rst += '| rx_offset      : {:.5f} um\n'.format(self.rx_offset)
        rst += '|--- FIT Sel ---- \n'
        rst += '| idx_kick       : {}\n'.format(self.select_idx_kick)
        rst += '| idx_bpm        : {}\n'.format(self.select_idx_bpm)
        rst += '| idx_turn_start : {}\n'.format(self.select_idx_turn_start)
        rst += '| idx_turn_stop  : {}\n'.format(self.select_idx_turn_stop)
        rst += '| sel_nr_turns   : {}\n'.format(sel_nr_turns)
        rst += '| kick           : {:+.1f} urad\n'.format(
            self.data_kicks[self.select_idx_kick]*1e3)
        rst += '| niter          : {}\n'.format(self.fit_simulann_niter)
        rst += '|-- FIR  Parms -- \n'
        rst += '| tunes_frac     : {:.6f}\n'.format(self.tunes_frac)
        rst += '| tunex_frac     : {:.6f}\n'.format(self.tunex_frac)
        rst += '| rx0            : {:.5f} um\n'.format(self.rx0)
        rst += '| mux            : {:.5f} rad\n'.format(self.mux)
        rst += '| chromx_decoh   : {:.5f}\n'.format(self.chromx_decoh)
        # rst += '|--- KXX ------- \n'
        # rst += '| tunex0_frac    : {:.6f}\n'.format(self.tunex0_frac)
        # rst += '| rxk            : {:.5f} um\n'.format(self.rxk)
        # rst += '| kxx_decoh      : {:.2f}\n'.format(self.kxx_decoh)
        # rst += '| kxx_ratio      : {:.2f}\n'.format(self.kxx_ratio)
        rst += '|--- OBJ --------\n'
        rst += '| obj_func       : {:.5f} um\n'.format(self.fit_simulann_residue)
        rst += ' ---------------- \n'

        return rst

    def load_parameters_20200505(self):
        """."""
        # fname = './2020-05/2020-05-05/tune_shift_with_amplitude.pickle'
        self.select_idx_kick = 0
        self.select_idx_bpm = 0
        self.select_idx_turn_stop = 500
        self.tunes_frac = 0.003156

        self.tunex_frac = 0.131403
        self.rx0 = 454.10347
        self.mux = 1.65361
        self.chromx_decoh = 1.39313

        self.tuney_frac = 0.131403
        self.ry0 = 454.10347
        self.muy = 1.65361
        self.chromy_decoh = 1.39313

    def analysis_save(self, fname):
        """."""
        self._analysis['select_idx_kick'].append(self.select_idx_kick)
        self._analysis['idx_bpm'].append(self.select_idx_bpm)
        self._analysis['select_idx_turn_start'].append(
            self.select_idx_turn_start)
        self._analysis['select_idx_turn_stop'].append(
            self.select_idx_turn_stop)
        self._analysis['tunes_frac'].append(self.tunes_frac)
        self._analysis['tunex_frac'].append(self.tunex_frac)
        self._analysis['chromx_decoh'].append(self.chromx_decoh)
        self._analysis['rx0'].append(self.rx0)
        self._analysis['mux'].append(self.mux)
        self.select_kicktype = TbTAnalysis.KICKTYPE_X
        self._analysis['rx_offset'].append(self.rx_offset)
        self._analysis['trajx_std'].append(_np.std(self.select_get_trajx()))
        self._analysis['residue'].append(self.fit_simulann_residue)
        with open(fname, 'wb') as fil:
            _pickle.dump(self._analysis, fil)

    def analysis_load(self, fname):
        """."""
        with open(fname, 'rb') as fil:
            self._analysis = _pickle.dump(self._analysis, fil)

    def analysis_set(self, index):
        """."""
        data = self._analysis
        self.select_idx_kick = data['select_idx_kick'][index]
        self.select_idx_bpm = data['select_idx_bpm'][index]
        self.select_idx_turn_start = data['select_idx_turn_start'][index]
        self.select_idx_turn_stop = data['select_idx_turn_stop'][index]
        self.tunes_frac = data['tunes_frac'][index]
        self.tunex_frac = data['tunex_frac'][index]
        self.chromx_decoh = data['chromx_decoh'][index]
        self.rx0 = data['rx0'][index]
        self.mux = data['mux'][index]

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

    def _select_get_traj(
            self, select_type=None,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_type = 'X' if select_type is None else select_type
        select_idx_kick, select_idx_bpm, select_idx_turn_start, \
            select_idx_turn_stop = self._get_sel_args(
                select_idx_kick,
                select_idx_bpm,
                select_idx_turn_start,
                select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        if select_type == 'X':
            traj_mea = self.data_trajx[
                select_idx_kick, turns_sel, select_idx_bpm]
        else:
            traj_mea = self.data_trajy[
                select_idx_kick, turns_sel, select_idx_bpm]
        return traj_mea

    def _self2simulann(self):
        """."""
        if self._simulann.fit_type == _TbTSimulAnneal.TYPES.CHROMX:
            pos = _np.array([
                self.tunes_frac,
                self.tunex_frac,
                self.chromx_decoh,
                self.rx0,
                self.mux])
        elif self._simulann.fit_type == _TbTSimulAnneal.TYPES.KXX:
            pos = _np.array([
                self.tunex0_frac,
                self.rxk,
                self.kxx_decoh,
                self.kxx_ratio])
        self._simulann.position = pos

    def _simulann2self(self):
        """."""
        pos = self._simulann.position
        if self._simulann.fit_type == _TbTSimulAnneal.TYPES.CHROMX:
            self.tunes_frac = pos[0]
            self.tunex_frac = pos[1]
            self.chromx_decoh = pos[2]
            self.rx0 = pos[3]
            self.mux = pos[4]
        else:
            self.tunex0_frac = pos[0]
            self.rxk = pos[1]
            self.kxx_decoh = pos[2]
            self.kxx_ratio = pos[3]

    def _get_sel_args(
            self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        select_idx_kick = self.select_idx_kick if select_idx_kick \
            is None else select_idx_kick
        select_idx_turn_start = \
            self.select_idx_turn_start if select_idx_turn_start \
            is None else select_idx_turn_start
        select_idx_turn_stop = \
            self.select_idx_turn_stop if select_idx_turn_stop \
            is None else select_idx_turn_stop
        select_idx_bpm = self.select_idx_bpm if select_idx_bpm \
            is None else select_idx_bpm
        return select_idx_kick, select_idx_bpm, select_idx_turn_start,select_idx_turn_stop

    def _init_analysis(self):
        """."""
        self._analysis = {
            'select_idx_kick': [],
            'select_idx_bpm': [],
            'select_idx_turn_start': [],
            'select_idx_turn_stop': [],
            'tunes_frac': [],
            'tunex_frac': [],
            'chromx_decoh': [],
            'rx0': [],
            'mux': [],
            'rx_offset': [],
            'trajx_std': [],
            'residue': [],
        }

    def _get_leastsqr_inputs(self):
        """."""
        if self.select_kicktype != TbTAnalysis.KICKTYPE_Y:
            tune_frac = self.tunex_frac
            chrom = self.chromx
            r0 = self.rx0
            mu = self.mux
            offset = self.rx_offset
        else:
            tune_frac = self.tuney_frac
            chrom = self.chromy
            r0 = self.ry0
            mu = self.muy
            offset = self.ry_offset

        params = [
            self.tunes_frac, tune_frac,
            chrom, self.espread, r0, mu]

        traj_mea = self.select_get_traj(
            select_idx_kick=self.select_idx_kick,
            select_idx_bpm=self.select_idx_bpm,
            select_idx_turn_start=self.select_idx_turn_start,
            select_idx_turn_stop=self.select_idx_turn_stop)

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
    def _calc_traj_chrom(params, *args):
        """BPM averaging due to longitudinal dynamics decoherence.

        nu ~ nu0 + chrom * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        tunes_frac = params[0]
        tune_frac = params[1]
        chrom = params[2]
        espread = params[3]
        r0 = params[4]
        mu = params[5]

        select_idx_turn_start, select_idx_turn_stop, offset = args
        turn = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        cos = _np.cos(2 * _np.pi * tune_frac * turn + mu)
        chromx_decoh = 2 * chrom * espread / tunes_frac
        alp = chromx_decoh * _np.sin(_np.pi * tunes_frac * turn)
        exp = _np.exp(-alp**2/2.0)
        traj = r0 * exp * cos + offset
        return traj

    @staticmethod
    def _calc_leastsqr_residue(
            params, traj_mea,
            select_idx_turn_start, select_idx_turn_stop, offset):
        """."""
        args = [select_idx_turn_start, select_idx_turn_stop, offset]
        traj_fit = TbTAnalysis._calc_traj_chrom(params, *args)
        traj_res = traj_mea - traj_fit
        return _np.sqrt(traj_res * traj_res / len(traj_res))
