#!/usr/bin/env python-sirius
"""."""

import pickle as _pickle
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.optimize as _opt
import scipy.signal as _scysig

from pyaccel.optics import calc_twiss as _calc_twiss
from pyaccel.naff import naff_general as _naff_general
from pymodels import si as _si
from ..utils import FrozenClass as _FrozenClass
from ..optics_analysis.tune_correction import TuneCorr as _TuneCorr


class TbT(_FrozenClass):
    """."""
    ATYPE_CHROMX = 'CHROMX'
    ATYPE_CHROMY = 'CHROMY'
    ATYPE_KXX = 'KXX'
    ATYPE_KYY = 'KYY'

    NOM_HARMONIC_NR = 864
    NOM_ALPHA = 1.645e-4
    NOM_ESPREAD = 0.08436/100
    NOM_EMITTANCE = 0.25e-3  # [um.rad]
    NOM_COUPLING = 3.0/100
    NOM_KXX_DECOH_NORM = -0.0145  # dtune/action [1/(rad.um)] for 2.4 chroms
    NOM_KXY_DECOH_NORM = +0.0083  # dtune/action [1/(rad.um)] for 2.4 chroms
    NOM_KYY_DECOH_NORM = +0.0377  # dtune/action [1/(rad.um)] for 2.4 chroms
    NOM_KYX_DECOH_NORM = +0.0083  # dtune/action [1/(rad.um)] for 2.4 chroms

    NOM_CHROMX = 2.4
    NOM_CHROMY = 2.4
    NOM_CHROM_ERR = 0.02
    NOM_TUNE_ERR = 1e-4

    def __init__(self, kicktype=None, data=None, data_fname=None):
        """."""
        # --- data attributes ---
        self._data = data
        self._data_fname = data_fname

        # --- select type of tbtanalysis: 'CHROMX' or 'CHROMY'
        self._select_kicktype = kicktype # This will be updated with data
        
        # --- data selection attributes ---
        self._select_idx_bpm = 0
        self._select_idx_kick = 0
        self._select_idx_turn_start = 0
        self._select_idx_turn_stop = None
        
        self._tunes_frac = 0.0
        self._tunes_frac_err = 0.0

        self._chrom = TbT.NOM_CHROMX
        self._tune_frac = 0.0
        self._chrom_decoh = 0.0
        self._r0 = 0.0
        self._mu = 0.0
        self._beta = 0.0
        self._eta = 0.0
        self._sigma = 0.0
        self._k_decoh = 0.0

        self._chrom_err = 0.0
        self._tune_frac_err = 0.0
        self._chrom_decoh_err = 0.0
        self._r0_err = 0.0
        self._mu_err = 0.0
        self._sigma_err = 0.0
        self._k_decoh_err = 0.0

        if not self._data and self._data_fname:
            self.data_load_raw()

        if self._select_kicktype is None:
            self._select_kicktype = self._data.get('kicktype', TbT.ATYPE_CHROMX)
        if self.select_plane_x:
            self.chrom = self._data.get('chromx', TbT.NOM_CHROMX)
            self.chrom_err = self._data.get('chromx_err', TbT.NOM_CHROM_ERR)
        else:
            self.chrom = self._data.get('chromy', TbT.NOM_CHROMX)
            self.chrom_err = self._data.get('chromy_err', TbT.NOM_CHROM_ERR)
        
        self.select_idx_turn_stop = self.data_nr_turns

        # --- model ---
        self._model_twiss = None
        self._model_bpms_idx = None
        self.init_twiss_from_model()

        # freeze class attributes, as to alert class users of wrong settler names used by mistake
        self._freeze()

    # --- data methods ---

    @property
    def data_fname(self):
        """."""
        return self._data_fname

    @property
    def data(self):
        """."""
        return self._data

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
            self._select_kicktype = self._data.get('kicktype', TbT.ATYPE_CHROMX)

    @property
    def data_traj(self):
        """."""
        if self.select_plane_x:
            return self._data['trajx']
        else:
            return self._data['trajy']

    # --- data selection methods for analysis ---

    @property
    def select_plane_x(self):
        return self._select_kicktype in (TbT.ATYPE_CHROMX, TbT.ATYPE_KXX)

    @property
    def select_plane_y(self):
        return self._select_kicktype in (TbT.ATYPE_CHROMY, TbT.ATYPE_KYY)

    @property
    def select_kicktype(self):
        """Return selected kick type for analysis."""
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

    @property
    def select_idx_bpm(self):
        """Return selected bpm index."""
        return self._select_idx_bpm

    @select_idx_bpm.setter
    def select_idx_bpm(self, value):
        """Set selected bpm index."""
        self._select_idx_bpm = value

    @property
    def select_idx_turn_start(self):
        """Return selected turn start index."""
        return self._select_idx_turn_start

    @select_idx_turn_start.setter
    def select_idx_turn_start(self, value):
        """Set selected turn start index."""
        self._select_idx_turn_start = value

    @property
    def select_idx_turn_stop(self):
        """Return selected turn stop index."""
        return self._select_idx_turn_stop

    @select_idx_turn_stop.setter
    def select_idx_turn_stop(self, value):
        """Set selected turn stop index."""
        self._select_idx_turn_stop = value

    def select_get_traj(self,
            select_idx_kick=None, select_idx_bpm=None,
            select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        traj_mea = self.data_traj[select_idx_kick, turns_sel, select_idx_bpm]
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
        return self.chrom_decoh * self.tunes_frac / 2 / self.chrom

    @espread.setter
    def espread(self, value):
        """."""
        self._chrom_decoh = 2 * self.chrom * value / self.tunes_frac

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
        el1 = (self.chrom_decoh_err * self.tunes_frac/2/self.chrom)**2
        el2 = (self.tunes_frac_err * self.chrom_decoh/2/self.chrom)**2
        el3 = (self.chrom_err * self.chrom_decoh*self.tunes_frac/2/self.chrom**2)**2
        return _np.sqrt(el1 + el2 + el3)

    # --- horizontal beam parameters ---

    @property
    def r0(self):
        """."""
        return self._r0

    @r0.setter
    def r0(self, value):
        """."""
        self._r0 = value

    @property
    def tune_frac(self):
        """Return fitting parameter tune_frac."""
        return self._tune_frac

    @tune_frac.setter
    def tune_frac(self, value):
        """Set fitting parameter tune_frac."""
        self._tune_frac = value

    @property
    def tune0_frac(self):
        """Return parameter tune0_frac."""
        k_decoh = self.k_decoh / self.beta
        tune0_frac = self.tune_frac - k_decoh*(4*self.sigma**2+self.r0**2)
        return tune0_frac

    @property
    def dtune_frac(self):
        """."""
        k_decoh = self.k_decoh / self.beta
        return k_decoh*(4*self.sigma**2+self.r0**2)

    @property
    def chrom_decoh(self):
        """."""
        return self._chrom_decoh

    @chrom_decoh.setter
    def chrom_decoh(self, value):
        """."""
        self._chrom_decoh = value

    @property
    def chrom(self):
        """."""
        return self._chrom

    @chrom.setter
    def chrom(self, value):
        """."""
        self._chrom = value

    @property
    def mu(self):
        """."""
        return self._mu

    @mu.setter
    def mu(self, value):
        """."""
        self._mu = value

    @property
    def beta(self):
        """Return parameter beta."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """Set parameter beta."""
        self._beta = value

    @property
    def eta(self):
        """Return parameter eta."""
        return self._eta

    @eta.setter
    def eta(self, value):
        """Set parameter eta."""
        self._eta = value

    @property
    def sigma(self):
        """Return fitting parameter sigma."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """Set fitting parameter sigma."""
        self._sigma = value

    @property
    def k_decoh(self):
        """Return fitting parameter k_decoh."""
        return self._k_decoh

    @k_decoh.setter
    def k_decoh(self, value):
        """Set fitting parameter k_decoh."""
        self._k_decoh = value

    # --- horizontal beam parameter errors ---

    @property
    def r0_err(self):
        """."""
        return self._r0_err

    @r0_err.setter
    def r0_err(self, value):
        """."""
        self._r0_err = value

    @property
    def tune_frac_err(self):
        """Return fitting parameter tune_frac_err."""
        return self._tune_frac_err

    @tune_frac_err.setter
    def tune_frac_err(self, value):
        """Set fitting parameter tune_frac_err."""
        self._tune_frac_err = value

    @property
    def tune0_frac_err(self):
        """Return parameter tune0_frac_err."""
        k_decoh = self.k_decoh / self.beta
        el1 = self.tune_frac_err
        el2 = (4*self.sigma**2+self.r0**2)*self.k_decoh_err/self.beta
        el3 = 8*k_decoh*self.sigma*self.sigma_err
        el4 = 2*k_decoh*self.r0*self.r0_err
        error = _np.sqrt(el1**2+el2**2+el3**2+el4**4)
        return error

    @property
    def dtune_frac_err(self):
        """."""
        k_decoh = self.k_decoh / self.beta
        el2 = (4*self.sigma**2+self.r0**2)*self.k_decoh_err/self.beta
        el3 = 8*k_decoh*self.sigma*self.sigma_err
        el4 = 2*k_decoh*self.r0*self.r0_err
        error = _np.sqrt(el2**2+el3**2+el4**4)
        return error

    @property
    def chrom_decoh_err(self):
        """."""
        return self._chrom_decoh_err

    @chrom_decoh_err.setter
    def chrom_decoh_err(self, value):
        """."""
        self._chrom_decoh_err = value

    @property
    def chrom_err(self):
        """."""
        return self._chrom_err

    @chrom_err.setter
    def chrom_err(self, value):
        """."""
        self._chrom_err = value

    @property
    def mu_err(self):
        """."""
        return self._mu_err

    @mu_err.setter
    def mu_err(self, value):
        """."""
        self._mu_err = value

    @property
    def sigma_err(self):
        """Return fitting parameter sigma_err."""
        return self._sigma_err

    @sigma_err.setter
    def sigma_err(self, value):
        """Set fitting parameter sigma_err."""
        self._sigma_err = value

    @property
    def k_decoh_err(self):
        """Return fitting parameter k_decoh_err."""
        return self._k_decoh_err

    @k_decoh_err.setter
    def k_decoh_err(self, value):
        """Set fitting parameter k_decoh_err."""
        self._k_decoh_err = value

    # --- model ---

    def init_k_decoh(self):
        """."""
        if self.select_plane_x:
            self.k_decoh = TbT.NOM_KXX_DECOH_NORM
        else:
            self.k_decoh = TbT.NOM_KYY_DECOH_NORM
            
    def init_twiss_from_model(self, update=False, goal_tunes=None):
        """."""
        if update or goal_tunes or self._model_twiss is None:
            self._model_twiss, self._model_bpms_idx = \
                TbT.calc_model_twiss(goal_tunes)
        bpms_idx = self._model_bpms_idx
        k = TbT.NOM_COUPLING
        emit0 = TbT.NOM_EMITTANCE
        if self.select_plane_x:
            emit = emit0 * 1 / (1 + k)
            self.beta = 1e6 * self._model_twiss.betax[bpms_idx[self.select_idx_bpm]]
            self.eta = 1e6 * self._model_twiss.etax[bpms_idx[self.select_idx_bpm]]
            self.sigma = _np.sqrt(emit * self.beta + 0*(self.eta * self.espread)**2)
        else:
            emit = emit0 * k / (1 + k)
            self.beta = 1e6 * self._model_twiss.betay[bpms_idx[self.select_idx_bpm]]
            self.eta = 1e6 * self._model_twiss.etay[bpms_idx[self.select_idx_bpm]]
        self.sigma = _np.sqrt(emit * self.beta + 0*(self.eta * self.espread)**2)
        
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
        fft, tune, tunes = TbT.calc_fft(traj_mea, peak_frac, plot_flag, title)
        _ = fft

        # naff
        # intv = select_idx_turn_start + int((self.data_nr_turns - 1 ) // 6)
        # size = intv*6 + 1
        # signal = traj_mea[:size] - _np.mean(traj_mea[:size])
        # freqs, fourier = _naff_general(signal=signal, is_real=True, nr_ff=1, window=1)
        # tune = freqs

        # set tunes
        self.tunes_frac = tunes
        self.tune_frac = tune

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

        _, cn_, sn_, *_ = TbT.calc_traj_chrom(params, *args)
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
        res1, res2 = _np.sum(vec1**2), _np.sum(vec2**2)
        if res2 < res1:
            mu += _np.pi
        self.r0 = r0
        self.mu = mu

    def search_init(self, plot_flag=False):
        """."""
        # search tunes with all data
        self.select_idx_turn_stop = self.data_nr_turns
        self.search_tunes(peak_frac = 0.999, plot_flag=False)

        # search beta and mu with 3 periods of betatron oscillations
        self.espread = TbT.NOM_ESPREAD * 1.0
        
        self.select_idx_turn_stop = int(3 / self.tune_frac)
    
        self.search_r0_mu()

    def filter_data(self, tune=None, tune_sigma=0.01, real_flag=True):
        """."""
        if tune is None:
            self.search_tunes()
            tune = self.tune_frac
        
        key = 'trajx' if self.select_plane_x else 'trajy'
        for idxkick in range(self.data_nr_kicks):
            for idxbpm in range(self.data_nr_bpms):
                data = self._data[key][idxkick][:, idxbpm]
                data_f = TbT.calc_filter(data, tune, tune_sigma, real_flag)
                self._data[key][idxkick][:, idxbpm] = data_f

    # --- fitting methods: common ---

    def fit_leastsqr(self):
        """."""
        init_params, offset, traj_mea = self._get_fit_inputs()
        
        tune_frac = self.tune_frac
        beta = self.beta
        sigma = self.sigma

        fit_params = _opt.least_squares(
            fun=self._calc_residue_vector,
            x0=init_params,
            args=(
                self._select_kicktype,
                traj_mea,
                self.select_idx_turn_start,
                self.select_idx_turn_stop,
                offset, tune_frac, beta, sigma),
            method='lm')

        fit_errors = TbT.calc_leastsqr_fitting_error(fit_params)

        self._set_from_params(fit_params, fit_errors)

    def fit_residue(self):
        """."""
        params, offset, traj_mea = self._get_fit_inputs()

        tune_frac = self.tune_frac
        beta = self.beta
        sigma = self.sigma

        residue_vec = self._calc_residue_vector(
            params, self._select_kicktype, traj_mea,
            self.select_idx_turn_start, self.select_idx_turn_stop, offset, tune_frac, beta, sigma)

        return _np.sqrt(_np.sum(residue_vec**2)/len(residue_vec))

    def fit_trajs(self):
        """."""
        params, offset, traj_mea = self._get_fit_inputs()
        args = [
            self.select_idx_turn_start, self.select_idx_turn_stop, offset, 
            self.tune_frac, self.beta, self.sigma]

        traj_fit, *_ = TbT.calc_traj(self.select_kicktype, params, *args)
        return traj_mea, traj_fit

    def fit_run_chrom(self, plot_flag=False):
        """."""
        # initial search for tunes, r0 and mu
        self.search_init(plot_flag)

        # search all parameters with 1 period of synchrotron oscillations
        self.select_idx_turn_stop = int(1 / self.tunes_frac)
        # self.select_idx_turn_stop = 280

        self.fit_leastsqr()

    def fit_run_tuneshift(self):
        """."""
        # change analysis type
        if self.select_kicktype == TbT.ATYPE_CHROMX:
            self.select_kicktype = TbT.ATYPE_KXX
        elif self.select_kicktype == TbT.ATYPE_CHROMY:
            self.select_kicktype = TbT.ATYPE_KYY
        self.select_idx_turn_stop = self.data_nr_turns

        # set initial fit parameters
        self.init_twiss_from_model()
        self.init_k_decoh()
        
        # does fitting
        self.fit_leastsqr()

    # --- analysis methods ---

    def analysis_run_chrom(self, select_idx_kick=0, bpm_indices=None, unwrap=False):
        """."""
        if bpm_indices is None:
            bpm_indices = _np.arange(self.data_nr_bpms)
        vec = _np.zeros(len(bpm_indices))
        residue = 0*vec
        r0, mu, tune_frac, tunes_frac, espread, chrom_decoh = \
            0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec
        r0_err, mu_err, tune_frac_err, tunes_frac_err, espread_err, chrom_decoh_err = \
            0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec
        self.select_idx_kick = select_idx_kick
        for idx, idx_bpm in enumerate(bpm_indices):
            # print(idx, self._select_kicktype)
            self.select_idx_bpm = idx_bpm
            self.fit_run_chrom()
            residue[idx] = self.fit_residue()
            # store params
            tunes_frac[idx] = self.tunes_frac - int(self.tunes_frac)
            espread[idx] = self.espread
            tunes_frac_err[idx] = self.tunes_frac_err - int(self.tunes_frac_err)
            espread_err[idx] = self.espread_err
            r0[idx] = self.r0
            mu[idx] = self.mu
            tune_frac[idx] = self.tune_frac
            chrom_decoh[idx] = self.chrom_decoh
            r0_err[idx] = self.r0_err
            mu_err[idx] = self.mu_err
            tune_frac_err[idx] = self.tune_frac_err
            chrom_decoh_err[idx] = self.chrom_decoh_err
                
        # unwrap phase    
        if unwrap:
            mu = _np.unwrap(mu)
            changed = True
            while changed:
                changed = False
                for i in range(1,len(mu)):
                    if mu[i] < mu[i-1]:
                        changed = True
                        mu[i:] += 0.5

        params = [r0, mu, tune_frac, tunes_frac, espread, chrom_decoh]
        params_err = [r0_err, mu_err, tune_frac_err, tunes_frac_err, espread_err, chrom_decoh_err]
        return bpm_indices, residue, params, params_err

    def analysis_run_tuneshift(self, select_idx_kick=0, bpm_indices=None):
        """."""
        if self.select_plane_x:
            ktype1, ktype2 = self.ATYPE_CHROMX, self.ATYPE_KXX
        else:
            ktype1, ktype2 = self.ATYPE_CHROMY, self.ATYPE_KYY
        if bpm_indices is None:
            bpm_indices = _np.arange(self.data_nr_bpms)
        vec = _np.zeros(len(bpm_indices))
        residue = 0*vec
        r0, mu, dtune_frac, tunes_frac, espread, k_decoh, sigma = \
            0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec
        r0_err, mu_err, dtune_frac_err, tunes_frac_err, espread_err, k_decoh_err, sigma_err = \
            0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec, 0*vec
        self.select_idx_kick = select_idx_kick
        for idx, idx_bpm in enumerate(bpm_indices):
            print(idx, ktype1, ktype2)
            self.select_kicktype = ktype1
            self.select_idx_bpm = idx_bpm
            self.fit_run_chrom()
            self.select_kicktype = ktype2
            self.fit_run_tuneshift()
            residue[idx] = self.fit_residue()
            # store params
            tunes_frac[idx] = self.tunes_frac - int(self.tunes_frac)
            espread[idx] = self.espread
            tunes_frac_err[idx] = self.tunes_frac_err
            espread_err[idx] = self.espread_err
            r0[idx] = self.r0
            mu[idx] = self.mu
            dtune_frac[idx] = self.tune_frac - self.tune0_frac
            k_decoh[idx] = self.k_decoh
            sigma[idx] = self.sigma
            # store params errors
            r0_err[idx] = self.r0_err
            mu_err[idx] = self.mu_err
            dtune_frac_err[idx] = self.dtune_frac_err
            k_decoh_err[idx] = self.k_decoh_err
            sigma_err[idx] = self.sigma_err
                
            # traj_mea, traj_fit = self.fit_trajs()
            # _plt.plot(traj_mea)
            # _plt.plot(traj_fit)
            # _plt.show()

        params = [r0, mu, dtune_frac, tunes_frac, espread, k_decoh, sigma]
        params_err = [r0_err, mu_err, dtune_frac_err, tunes_frac_err, espread_err, k_decoh_err, sigma_err]
        return bpm_indices, residue, params, params_err

    # --- aux. public methods ---

    @staticmethod
    def calc_filter(data, tune, tune_sigma, real_flag=True):

        offset = _np.mean(data)
        signal = data - offset
        # data_anal = _np.array(_scysig.hilbert(signal))
        data_anal = signal
        # # calculate DFT:
        if real_flag:
            data_dft = _np.fft.rfft(data_anal)
            freq = _np.fft.rfftfreq(data_anal.shape[0])
        else:
            data_dft = _np.fft.fft(data_anal)
            freq = _np.fft.fftfreq(data_anal.shape[0])
        center_freq = tune
        sigma_freq = tune_sigma

        # Apply Gaussian filter to get only the synchrotron frequency
        H = _np.exp(-(freq - center_freq)**2/2/sigma_freq**2)
        H += _np.exp(-(freq + center_freq)**2/2/sigma_freq**2)
        H /= H.max()
        data_dft *= H
        # get the processed data by inverse DFT
        if real_flag:
            data_anal = _np.real(_np.fft.irfft(data_dft))
        else:
            data_anal = _np.fft.ifft(data_dft)
        return data_anal + offset

    @staticmethod
    def calc_fft(data, peak_frac=0.7, plot_flag=False, title=None):
        """."""
        # plot_flag = True
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
    def calc_traj(kicktype, params, *args):
        """."""
        if kicktype in (TbT.ATYPE_CHROMX, TbT.ATYPE_CHROMY):
            args = args[:3]
            return TbT.calc_traj_chrom(params, *args)
        elif kicktype in (TbT.ATYPE_KXX, TbT.ATYPE_KYY):
            return TbT.calc_traj_tuneshift(params, *args)

    @staticmethod
    def calc_traj_chrom(params, *args):
        """BPM averaging due to longitudinal dynamics decoherence.

        nu ~ nu0 + chrom * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        tunes_frac = params[0]
        tune_frac = params[1]
        chrom_decoh = params[2]
        r0 = params[3]
        mu = params[4]

        select_idx_turn_start, select_idx_turn_stop, offset = args
        n = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        cn = _np.cos(2 * _np.pi * tune_frac * n)
        sn = _np.sin(2 * _np.pi * tune_frac * n)
        cos = cn * _np.cos(mu) - sn * _np.sin(mu)
        alp = chrom_decoh * _np.sin(_np.pi * tunes_frac * n)
        exp = _np.exp(-alp**2/2.0)
        traj = r0 * exp * cos + offset
        return traj, cn, sn, alp, exp, cos

    @staticmethod
    def calc_traj_tuneshift(params, *args):
        """BPM averaging due to tune-shift decoherence.

        nu ~ nu0 + k_decoh * a**2
        See Laurent Nadolski Thesis, Chapter 4, pg. 123, Eq. 4.28
        """
        select_idx_turn_start, select_idx_turn_stop, offset, tune_frac, beta, sigma = args

        tunes_frac = params[0]
        chrom_decoh = params[1]
        r0 = params[2]
        mu = params[3]
        k_decoh = params[4]/beta

        b = (r0**2/(sigma**2)/2)
        a = 4*_np.pi*k_decoh*sigma**2
        tune0_frac = tune_frac - k_decoh*(4*sigma**2+r0**2)
        n = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        theta = a*n
        fa0 = 1/(1+theta**2)
        fp0 = 2*_np.pi*(tune0_frac)*n + (mu - _np.pi/2)
        fp1 = 2*_np.arctan(theta)
        fp2 = b*theta*fa0
        fa1 = _np.exp(-b*theta**2*fa0)

        alp = chrom_decoh * _np.sin(_np.pi * tunes_frac * n)
        exp = _np.exp(-alp**2/2.0)

        traj = -r0 * fa0 * fa1 * exp * _np.sin(fp0 + fp1 + fp2) + offset
        return traj, alp, exp

    @staticmethod
    def calc_model_twiss(goal_tunes=None):
        """."""
        # model
        si = _si.create_accelerator()

        if goal_tunes is not None:
            print('correcting model tunes...')
            tc = _TuneCorr(si, 'SI')
            print('init tunes: ', tc.get_tunes())
            print('goal tunes: ', goal_tunes)
            tc.correct_parameters(goal_tunes)
            print('corr tunes: ', tc.get_tunes())
            print()

        twiss, *_ = _calc_twiss(si)
        fam_data = _si.get_family_data(si)
        bpms_idx = [v[0] for v in fam_data['BPM']['index']]
        return twiss, _np.array(bpms_idx)

    def __str__(self):
        """."""
        # self._self2simulann()
        sel_nr_turns = self.select_idx_turn_stop - self.select_idx_turn_start
        rst = ''
        rst += '.--- Data -------\n'
        rst += '| nr_kicks       : {}\n'.format(self.data_nr_kicks)
        rst += '| nr_turns       : {}\n'.format(self.data_nr_turns)
        rst += '| nr_bpms        : {}\n'.format(self.data_nr_bpms)
        rst += '|--- Selection -- \n'
        rst += '| kicktype       : {}\n'.format(self.select_kicktype)
        rst += '| idx_kick       : {}\n'.format(self.select_idx_kick)
        rst += '| idx_bpm        : {}\n'.format(self.select_idx_bpm)
        rst += '| idx_turn_start : {}\n'.format(self.select_idx_turn_start)
        rst += '| idx_turn_stop  : {}\n'.format(self.select_idx_turn_stop)
        rst += '| sel_nr_turns   : {}\n'.format(sel_nr_turns)
        rst += '| kick           : {:+.1f} urad\n'.format(
            self.data_kicks[self.select_idx_kick]*1e3)
        # rst += '| niter          : {}\n'.format(self.fit_simulann_niter)
        rst += '|-- Fit Params -- \n'
        rst += '| espread        : {:.5f} ± {:.5f} %\n'.format(100*self.espread, 100*self.espread_err)
        rst += '| tunes_frac     : {:.6f} ± {:.6f}\n'.format(self.tunes_frac, self.tunes_frac_err)
        rst += '| --------------- \n'
        rst += '| chrom         : {:+.3f} ± {:.3f}\n'.format(self.chrom, self.chrom_err)
        rst += '| tune0_frac    : {:.6f} ± {:.6f}\n'.format(self.tune0_frac, self.tune0_frac_err)
        rst += '| tune_frac     : {:.6f} ± {:.6f}\n'.format(self.tune_frac, self.tune_frac_err)
        rst += '| dtune_frac    : {:.6f} ± {:.6f}\n'.format(self.dtune_frac, self.dtune_frac_err)
        rst += '| r0            : {:.5f} ± {:.5f} um\n'.format(self.r0, self.r0_err)
        rst += '| mu            : {:.5f} ± {:.5f} rad\n'.format(self.mu, self.mu_err)
        rst += '| chrom_decoh   : {:.5f} ± {:.5f}\n'.format(self.chrom_decoh, self.chrom_decoh_err)
        rst += '| beta          : {:.2f} m\n'.format(self.beta/1e6)
        rst += '| eta           : {:.2f} cm\n'.format(self.eta/1e4)
        rst += '| sigma         : {:.2f} ± {:.2f} um\n'.format(self.sigma, self.sigma_err)
        rst += '| k_decoh      : {:.5f} ± {:.5f} 1/um\n'.format(self.k_decoh, self.k_decoh_err)
        rst += '|--------------- \n'
        rst += '| residue        : {:.5f} um\n'.format(self.fit_residue())
        rst += ' ---------------- \n'

        return rst

    # --- private methods ---

    def _select_get_traj(self, select_type=None,
                  select_idx_kick=None, select_idx_bpm=None,
                  select_idx_turn_start=None, select_idx_turn_stop=None):
        """Get selected traj data."""
        select_type = TbT.ATYPE_CHROMX if select_type is None else select_type
        select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop = \
            self._get_sel_args(
                select_idx_kick, select_idx_bpm, select_idx_turn_start, select_idx_turn_stop)
        turns_sel = _np.arange(select_idx_turn_start, select_idx_turn_stop)
        traj_mea = self.data_traj[select_idx_kick, turns_sel, select_idx_bpm]
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
        if self.select_kicktype in (TbT.ATYPE_CHROMX, TbT.ATYPE_CHROMY):
            params = [
                self.tunes_frac, self.tune_frac,
                self.chrom_decoh, self.r0, self.mu]
        else:
            params = [
                self.tunes_frac, 
                self.chrom_decoh, self.r0, 
                self.mu, self.k_decoh]
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
    def calc_leastsqr_fitting_error(fit_params):
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
            params, kicktype, traj_mea,
            select_idx_turn_start, select_idx_turn_stop, offset, tune_frac, beta, sigma):
        """."""
        args = [select_idx_turn_start, select_idx_turn_stop, offset, tune_frac, beta, sigma]
        traj_fit, *_ = TbT.calc_traj(kicktype, params, *args)
        traj_res = traj_mea - traj_fit
        return traj_res

    def _set_from_params(self, fit_params, fit_errors):
        if self._select_kicktype in (TbT.ATYPE_CHROMX, TbT.ATYPE_CHROMY):
            self.tunes_frac = fit_params['x'][0]
            self.tune_frac = fit_params['x'][1]
            self.chrom_decoh = fit_params['x'][2]
            self.r0 = fit_params['x'][3]
            self.mu = fit_params['x'][4]
            # errors
            self.tunes_frac_err = fit_errors[0]
            self.tune_frac_err = fit_errors[1]
            self.chrom_decoh_err = fit_errors[2]
            self.r0_err = fit_errors[3]
            self.mu_err = fit_errors[4]
        else:
            self.tunes_frac = fit_params['x'][0]
            self.chrom_decoh = fit_params['x'][1]
            self.r0 = fit_params['x'][2]
            self.mu = fit_params['x'][3]
            self.k_decoh = fit_params['x'][4]
            # errors
            self.tunes_frac_err = fit_errors[0]
            self.chrom_decoh_err = fit_errors[1]
            self.r0_err = fit_errors[2]
            self.mu_err = fit_errors[3]
            self.k_decoh_err = fit_errors[4]
