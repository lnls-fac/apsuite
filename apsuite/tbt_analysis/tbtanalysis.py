#!/usr/bin/env python-sirius
"""."""

import pickle as _pickle
import numpy as _np
import matplotlib.pyplot as _plt

from .tbtsimann import TbTSimulAnneal as _TbTSimulAnneal


class TbTAnalysis:
    """."""

    def __init__(self, data=None, data_fname=None):
        """."""
        self._data = data
        self._data_fname = data_fname
        self._idx_bpm = 0
        self._idx_kick = 0
        self._idx_turn_start = None
        self._idx_turn_stop = None
        self._rx_offset = None
        self._tunes_frac = 0.0
        self._tunex_frac = 0.0
        self._chromx_decoh = 0.0
        self._rx0 = 0.0
        self._mux = 0.0
        self._simann = None
        self._analysis = None

        if not self._data and self._data_fname:
            self.load_data_raw()

        self.fit_reset()

        # loads parameters of 2020-05-05 data (idx_bpm 0, idx_kick 0)
        self.load_parameters_20200505()
        self._init_analysis()

    # --- methods to retrieve raw data ---

    @property
    def trajx(self):
        """Return trajx data."""
        if 'trajx' in self._data:
            return self._data['trajx']
        return None

    @property
    def trajy(self):
        """Return trajy data."""
        if 'trajy' in self._data:
            return self._data['trajy']
        return None

    @property
    def trajsum(self):
        """Return trajsum data."""
        if 'trajsum' in self._data:
            return self._data['trajsum']
        return None

    @property
    def kicks(self):
        """Return kick values."""
        if 'kicks' in self._data:
            return self._data['kicks']
        return None

    @property
    def nr_kicks(self):
        """Return number of kicks."""
        if not self._data:
            return None
        return self._data['trajx'].shape[0]

    @property
    def nr_turns(self):
        """Return number of turns in data."""
        if not self._data:
            return None
        return self._data['trajx'].shape[1]

    @property
    def nr_bpms(self):
        """."""
        if not self._data:
            return None
        return self._data['trajx'].shape[2]

    def load_data_raw(self, fname=None):
        """Load raw data from pick file."""
        if fname is None:
            fname = self._data_fname
        else:
            self._data_fname = fname
        with open(fname, 'rb') as fil:
            was_none = self._data is None
            self._data = _pickle.load(fil)
            if was_none:
                self._idx_kick = 0
                self._idx_turn_start = 0
                self._idx_turn_stop = self.nr_turns
                self._idx_bpm = 0
            else:
                self._idx_kick = min(self._idx_kick, self.nr_kicks)
                self._idx_turn_start = min(self._idx_turn_start, self.nr_turns)
                self._idx_turn_stop = min(self._idx_turn_stop, self.nr_turns)
                self._idx_bpm = min(self._idx_bpm, self.nr_bpms)
        trajx = self.sel_trajx()
        self._rx_offset = _np.mean(trajx)

    # --- methods that select raw data for analysis ---

    @property
    def idx_kick(self):
        """Return selected kick index."""
        return self._idx_kick

    @idx_kick.setter
    def idx_kick(self, value):
        """Set selected kick index."""
        self._idx_kick = value
        trajx = self.sel_trajx()
        self._rx_offset = _np.mean(trajx)

    @property
    def idx_bpm(self):
        """Return selected bpm index."""
        return self._idx_bpm

    @idx_bpm.setter
    def idx_bpm(self, value):
        """Set selected bpm index."""
        self._idx_bpm = value
        trajx = self.sel_trajx()
        self._rx_offset = _np.mean(trajx)

    @property
    def idx_turn_start(self):
        """Return selected turn start index."""
        return self._idx_turn_start

    @idx_turn_start.setter
    def idx_turn_start(self, value):
        """Set selected turn start index."""
        self._idx_turn_start = value
        trajx = self.sel_trajx()
        self._rx_offset = _np.mean(trajx)

    @property
    def idx_turn_stop(self):
        """Return selected turn stop index."""
        return self._idx_turn_stop

    @idx_turn_stop.setter
    def idx_turn_stop(self, value):
        """Set selected turn stop index."""
        self._idx_turn_stop = value
        trajx = self.sel_trajx()
        self._rx_offset = _np.mean(trajx)

    def sel_trajx(self,
                  idx_kick=None, idx_bpm=None,
                  idx_turn_start=None, idx_turn_stop=None):
        """Get selected trajx data."""
        idx_kick, idx_bpm, idx_turn_start, idx_turn_stop = \
            self._get_sel_args(
                idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)
        turns_sel = _np.arange(idx_turn_start, idx_turn_stop)
        trajx_mea = self.trajx[idx_kick, turns_sel, idx_bpm]
        return trajx_mea

    # --- fitting parameters methods ---

    @property
    def simann(self):
        """Return TbTSimuAnnel object."""
        self._self2simann()
        return self._simann

    @property
    def rx_offset(self):
        """Return offset value of trajx selected data."""
        return self._rx_offset

    @property
    def tunes_frac(self):
        """Return fitting parameter tunes_frac."""
        return self._tunes_frac

    @tunes_frac.setter
    def tunes_frac(self, value):
        """Set fitting parameter tunes_frac."""
        self._tunes_frac = value

    @property
    def tunex_frac(self):
        """Return fitting parameter tunex_frac."""
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
    def rx0(self):
        """."""
        return self._rx0

    @rx0.setter
    def rx0(self, value):
        """."""
        self._rx0 = value

    @property
    def mux(self):
        """."""
        return self._mux

    @mux.setter
    def mux(self, value):
        """."""
        self._mux = value

    # --- search methods ---

    def search_tunes(
            self,
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None,
            plot_flag=False):
        """."""
        # print(self)
        # plot_flag = True

        idx_turn_stop = \
            self.nr_turns if idx_turn_stop is None else idx_turn_stop

        idx_kick, idx_bpm, idx_turn_start, idx_turn_stop = \
            self._get_sel_args(
                idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        self.idx_kick = idx_kick
        self.idx_bpm = idx_bpm

        # print('tunes: ', idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        trajx_mea = self.sel_trajx(
            idx_kick=idx_kick, idx_bpm=idx_bpm,
            idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop)

        title = 'FFT, nr_turns: {}, idx_kick: {}, idx_bpm: {}'.format(
            idx_turn_stop, idx_kick, idx_bpm)
        fft, tunex, tunes = TbTAnalysis.calc_fft(trajx_mea, plot_flag, title)
        _ = fft

        self.tunex_frac = tunex
        self.tunes_frac = tunes

    def search_rx0_mux(
            self,
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None):
        """."""
        idx_turn_stop = 20 if idx_turn_stop is None else idx_turn_stop

        idx_kick, idx_bpm, idx_turn_start, idx_turn_stop = \
            self._get_sel_args(
                idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        self.idx_kick = idx_kick
        self.idx_bpm = idx_bpm

        # print(self)
        # print('rx0_mux: ', idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        trajx_mea = self.sel_trajx(
            idx_kick=idx_kick, idx_bpm=idx_bpm,
            idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop)
        trajx_mea = trajx_mea - _np.mean(trajx_mea)

        tunex_frac = self.tunex_frac
        turn = _np.arange(idx_turn_start, idx_turn_stop)
        cn_ = _np.cos(2 * _np.pi * tunex_frac * turn)
        sn_ = _np.sin(2 * _np.pi * tunex_frac * turn)
        a11, a12 = + _np.sum(cn_ * cn_), - _np.sum(cn_ * sn_)
        a21, a22 = - a12, - _np.sum(sn_ * sn_)
        b11, b21 = _np.sum(trajx_mea * cn_), _np.sum(trajx_mea * sn_)
        mata = _np.array([[a11, a12], [a21, a22]])
        matb = _np.array([b11, b21])
        c0_, s0_ = _np.linalg.solve(mata, matb)
        rx0 = _np.sqrt(c0_**2 + s0_**2)
        mux = _np.arcsin(s0_ / rx0)

        self.rx0 = rx0
        self.mux = mux

    def search_chromx_decoh(
            self,
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None):
        """."""
        idx_turn_stop = 500 if idx_turn_stop is None else idx_turn_stop

        idx_kick, idx_bpm, idx_turn_start, idx_turn_stop = \
            self._get_sel_args(
                idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        self.idx_kick = idx_kick
        self.idx_bpm = idx_bpm

        # print(idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        self.chromx_decoh = 0
        obj1 = self.fit_residue
        self.chromx_decoh = 0.1
        obj2 = self.fit_residue
        factor = 2.0 if obj2 < obj1 else 0.5

        # first find a parameter interval
        niter = 0
        while niter < 16:
            self.chromx_decoh *= factor
            obj1 = self.fit_residue
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
        best_par, best_obj = par2, self.fit_residue
        for par in pars:
            self.chromx_decoh = par
            obj = self.simann.calc_obj_fun()
            # print(par, obj)
            if obj < best_obj:
                best_par = par
                best_obj = obj
        print(best_par)
        self.chromx_decoh = best_par

    def search_init(
            self,
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None,
            plot_flag=False):
        """."""
        self.search_tunes(
            idx_kick=idx_kick, idx_bpm=idx_bpm,
            idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop,
            plot_flag=plot_flag)

        self.search_rx0_mux(
            idx_kick=idx_kick, idx_bpm=idx_bpm,
            idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop)

        # self.search_chromx_decoh(
        #     idx_kick=idx_kick, idx_bpm=idx_bpm,
        #     idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop)

    # --- fitting methods ---

    @property
    def fit_niter(self):
        """Return number of iterations for fitting."""
        return self._simann.niter

    @fit_niter.setter
    def fit_niter(self, value):
        """Set number of iterations for fitting."""
        self._simann.niter = value

    def fit_reset(self):
        """."""
        simann = _TbTSimulAnneal(
            self, _TbTSimulAnneal.TYPES.CHROMX)
        self._simann = simann
        value_max = 1.1 * _np.amax(self.trajx)
        lower = _np.array([0.0, 0.0, 0.0, 0.0, -_np.pi])
        upper = _np.array([0.05, 0.5, 100.0, value_max, +_np.pi])
        delta = _np.array(
            [0.00003, 0.001, 0.01, 0.01*value_max, 2*_np.pi*0.01])
        self._self2simann()
        simann.niter = 1 if simann.niter == 0 else simann.niter
        simann.limits_upper = upper
        simann.limits_lower = lower
        simann.deltas = delta

    def fit_start(self, niter=1, print_flag=True):
        """."""
        self.simann.niter = niter
        self.simann.start(print_flag=print_flag)
        self._simann2self()

    @property
    def fit_residue(self):
        """Return fit residue."""
        return self.simann.calc_obj_fun()

    def fit_plot(
            self,
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None,
            title=None):
        """."""
        idx_kick, idx_bpm, idx_turn_start, idx_turn_stop = \
            self._get_sel_args(
                idx_kick, idx_bpm, idx_turn_start, idx_turn_stop)

        trajx_mea = self.sel_trajx(
            idx_kick=idx_kick, idx_bpm=idx_bpm,
            idx_turn_start=idx_turn_start, idx_turn_stop=idx_turn_stop)

        parms = {
            'idx_turn_start': idx_turn_start,
            'idx_turn_stop': idx_turn_stop,
            'rx_offset': self.rx_offset,
            'tunes_frac': self.tunes_frac,
            'tunex_frac': self.tunex_frac,
            'chromx_decoh': self.chromx_decoh,
            'rx0': self.rx0,
            'mux': self.mux,
        }
        trajx_fit = self.simann.calc_trajx(**parms)
        if title is None:
            title = \
                'kick idx: {:03d}, '.format(idx_kick) + \
                'bpm idx: {:03d}, '.format(idx_bpm)
        label_residue = 'Residue (rms: {:.2f} um)'.format(self.fit_residue)
        ind = _np.arange(idx_turn_start, idx_turn_stop)
        trajx_res = trajx_mea - trajx_fit
        _plt.plot(ind, trajx_mea, color='red', label='TbT raw data')
        _plt.plot(ind, trajx_mea, 'o', color='red')
        _plt.plot(ind, trajx_fit, color='orange', label='Fitting')
        _plt.plot(ind, trajx_fit, 'o', color='orange')
        _plt.plot(ind, trajx_res, label=label_residue)
        _plt.legend()
        _plt.xlabel('turn')
        _plt.ylabel('BPM Average [um]')
        _plt.title(title)
        _plt.grid()
        _plt.show()

    def fit_run(self, niter=None, idx_bpm=None, idx_kick=None, log_flag=True):
        """."""
        niter = 30000 if niter is None else niter
        self.fit_reset()
        if idx_bpm:
            self.idx_bpm = idx_bpm
        if idx_kick:
            self.idx_kick = idx_kick
        simann = self.simann

        # search parameters
        niter_, nrpts = 0, 20
        factor = _np.exp(_np.log(0.1)/nrpts)
        simann.niter = int(round(niter / nrpts))
        if log_flag:
            sfmt = '#{:02d}/{:02d} iter {:06d} : obj_func = {:7.2f} um'
            print(sfmt.format(0, nrpts, 0, simann.calc_obj_fun()))
        for k in range(nrpts):
            simann.start(print_flag=False)
            niter_ += simann.niter
            if log_flag:
                print(sfmt.format(k, nrpts, niter_, simann.calc_obj_fun()))
            simann.deltas *= factor
        self._simann2self()

    # --- analysis methods ---

    def analysis_all_bpms(self, niter=30000, fname=None, plot_flag=False):
        """."""
        self.load_parameters_20200505()
        self._init_analysis()
        for idx_bpm in range(0, self.nr_bpms):
            if idx_bpm > 0:
                self.idx_bpm = idx_bpm
                self.search_rx0_mux()
                self.fit_run(niter=niter, log_flag=False)

            print('bpm idx: {:03d}, residue = {:.2f} um'.format(
                self.idx_bpm, self.fit_residue))
            print(self)
            if plot_flag:
                self.fit_plot()

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
        self._self2simann()
        sel_nr_turns = self.idx_turn_stop - self.idx_turn_start
        rst = ''
        rst += '.--- TBT --------\n'
        rst += '| nr_kicks       : {}\n'.format(self.nr_kicks)
        rst += '| nr_turns       : {}\n'.format(self.nr_turns)
        rst += '| nr_bpms        : {}\n'.format(self.nr_bpms)
        rst += '| rx_offset      : {:.5f} um\n'.format(self.rx_offset)
        rst += '|--- FIT Sel ---- \n'
        rst += '| idx_kick       : {}\n'.format(self.idx_kick)
        rst += '| idx_bpm        : {}\n'.format(self.idx_bpm)
        rst += '| idx_turn_start : {}\n'.format(self.idx_turn_start)
        rst += '| idx_turn_stop  : {}\n'.format(self.idx_turn_stop)
        rst += '| sel_nr_turns   : {}\n'.format(sel_nr_turns)
        rst += '| kick           : {:+.1f} urad\n'.format(
            self.kicks[self.idx_kick]*1e3)
        rst += '| niter          : {}\n'.format(self.fit_niter)
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
        rst += '| obj_func       : {:.5f} um\n'.format(self.fit_residue)
        rst += ' ---------------- \n'

        return rst

    def load_parameters_20200505(self):
        """."""
        # fname = './2020-05/2020-05-05/tune_shift_with_amplitude.pickle'
        self.idx_kick = 0
        self.idx_bpm = 0
        self.idx_turn_stop = 500
        self.tunes_frac = 0.003156
        self.tunex_frac = 0.131403
        self.rx0 = 454.10347
        self.mux = 1.65361
        self.chromx_decoh = 1.39313

    def analysis_save(self, fname):
        """."""
        self._analysis['idx_kick'].append(self.idx_kick)
        self._analysis['idx_bpm'].append(self.idx_bpm)
        self._analysis['idx_turn_start'].append(self.idx_turn_start)
        self._analysis['idx_turn_stop'].append(self.idx_turn_stop)
        self._analysis['tunes_frac'].append(self.tunes_frac)
        self._analysis['tunex_frac'].append(self.tunex_frac)
        self._analysis['chromx_decoh'].append(self.chromx_decoh)
        self._analysis['rx0'].append(self.rx0)
        self._analysis['mux'].append(self.mux)
        self._analysis['rx_offset'].append(self.rx_offset)
        self._analysis['trajx_std'].append(_np.std(self.sel_trajx()))
        self._analysis['residue'].append(self.fit_residue)
        with open(fname, 'wb') as fil:
            _pickle.dump(self._analysis, fil)

    def analysis_load(self, fname):
        """."""
        with open(fname, 'rb') as fil:
            self._analysis = _pickle.dump(self._analysis, fil)

    def analysis_set(self, index):
        """."""
        data = self._analysis
        self.idx_kick = data['idx_kick'][index]
        self.idx_bpm = data['idx_bpm'][index]
        self.idx_turn_start = data['idx_turn_start'][index]
        self.idx_turn_stop = data['idx_turn_stop'][index]
        self.tunes_frac = data['tunes_frac'][index]
        self.tunex_frac = data['tunex_frac'][index]
        self.chromx_decoh = data['chromx_decoh'][index]
        self.rx0 = data['rx0'][index]
        self.mux = data['mux'][index]

    # --- private methods ---

    def _self2simann(self):
        """."""
        if self._simann.fit_type == _TbTSimulAnneal.TYPES.CHROMX:
            pos = _np.array([
                self.tunes_frac,
                self.tunex_frac,
                self.chromx_decoh,
                self.rx0,
                self.mux])
        elif self._simann.fit_type == _TbTSimulAnneal.TYPES.KXX:
            pos = _np.array([
                self.tunex0_frac,
                self.rxk,
                self.kxx_decoh,
                self.kxx_ratio])
        self._simann.position = pos

    def _simann2self(self):
        """."""
        pos = self._simann.position
        if self._simann.fit_type == _TbTSimulAnneal.TYPES.CHROMX:
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
            idx_kick=None, idx_bpm=None,
            idx_turn_start=None, idx_turn_stop=None):
        idx_kick = self.idx_kick if idx_kick is None else idx_kick
        idx_turn_start = \
            self.idx_turn_start if idx_turn_start is None else idx_turn_start
        idx_turn_stop = \
            self.idx_turn_stop if idx_turn_stop is None else idx_turn_stop
        idx_bpm = self.idx_bpm if idx_bpm is None else idx_bpm
        return idx_kick, idx_bpm, idx_turn_start, idx_turn_stop

    def _init_analysis(self):
        """."""
        self._analysis = {
            'idx_kick': [],
            'idx_bpm': [],
            'idx_turn_start': [],
            'idx_turn_stop': [],
            'tunes_frac': [],
            'tunex_frac': [],
            'chromx_decoh': [],
            'rx0': [],
            'mux': [],
            'rx_offset': [],
            'trajx_std': [],
            'residue': [],
        }
