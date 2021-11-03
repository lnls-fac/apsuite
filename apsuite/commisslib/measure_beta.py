"""Main module."""
import time as _time
import math
from copy import deepcopy as _dcopy
from mathphys.functions import get_namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply, Tune, SOFB
import pyaccel
from pymodels import si

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BetaParams(_ParamsBaseClass):
    """."""

    # Measurements data can be found in Lucas I. Balthazar's e-mail
    # sent in 10/30/2020.

    # Currents used in measurements performed at IMA's lab.
    DELTA_CURRENT = {
        'QDA': 1.37,
        'QDB1': 1.40,
        'QDP1': 1.40,
        'QDB2': 1.48,
        'QDP2': 1.48,

        'QFA': 0.88,
        'Q1': 0.86,
        'Q2': 0.95,
        'Q3': 0.86,
        'Q4': 0.93,

        'QFB': 0.59,
        'QFP': 0.59,
        }

    # Integrated quadrupole field variations measured at IMA's lab.
    DELTA_GL = {
        'QDA': -6.9e-2,
        'QDB1': -7.0e-2,
        'QDP1': -7.0e-2,
        'QDB2': -7.1e-2,
        'QDP2': -7.1e-2,

        'QFA': 3.6e-2,
        'Q1':  3.9e-2,
        'Q2':  4.0e-2,
        'Q3':  3.9e-2,
        'Q4':  4.2e-2,

        'QFB': 4.0e-2,
        'QFP': 4.0e-2,
        }

    def __init__(self):
        """."""
        super().__init__()
        self.nr_measures = 1
        self.quad_deltakl = 0.01  # [1/m]
        self.wait_quadrupole = 1  # [s]
        self.wait_tune = 3  # [s]
        self.timeout_quad_turnon = 10  # [s]
        self.recover_tune = True
        self.recover_tune_tol = 1e-4
        self.recover_tune_maxiter = 3
        self.quad_nrcycles = 0
        self.time_wait_quad_cycle = 0.5  # [s]
        self.sofb_nriters = 2
        self.sofb_residue = 0.1  # [um]

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_measures', self.nr_measures, '')
        stg += ftmp('quad_deltakl [1/m]', self.quad_deltakl, '')
        stg += ftmp('wait_quadrupole [s]', self.wait_quadrupole, '')
        stg += ftmp('wait_tune [s]', self.wait_tune, '')
        stg += ftmp('timeout_quad_turnon [s]', self.timeout_quad_turnon, '')
        stg += ftmp('recover_tune', self.recover_tune, '')
        stg += ftmp('recover_tune_tol', self.recover_tune_tol, '')
        stg += dtmp(
            'recover_tune_maxiter', self.recover_tune_maxiter, '')
        stg += ftmp('quad_nrcycles', self.quad_nrcycles, '')
        stg += ftmp(
            'time_wait_quad_cycle [s]', self.time_wait_quad_cycle, '')
        stg += dtmp('sofb_nriters', self.sofb_nriters, '')
        stg += ftmp('sofb_residue [um]', self.sofb_residue, '')
        return stg


class MeasBeta(_BaseClass):
    """."""

    METHODS = get_namedtuple('Methods', ['Analytic', 'Numeric'])
    MEASUREMENT = get_namedtuple('Process', ['Current', 'KL'])
    ANALYSIS = get_namedtuple('Analysis', ['IMA', 'Excdata'])
    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(
            self, model, famdata=None, isonline=True,
            meas_method=None, calc_method=None, anly_method=None):
        """."""
        super().__init__(
            params=BetaParams(), target=self._meas_beta, isonline=isonline)
        self.quads_betax = []
        self.quads_betay = []
        self._calc_method = MeasBeta.METHODS.Numeric
        self._meas_method = MeasBeta.MEASUREMENT.Current
        self._anly_method = MeasBeta.ANALYSIS.Excdata
        if self.isonline:
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.data['quadnames'] = list()
        self.data['cycling'] = dict()
        self.data['betax_int'] = dict()
        self.data['betay_int'] = dict()
        self.data['measure'] = dict()
        self.data['tunex_ref'] = 0
        self.data['tuney_ref'] = 0
        self._quads2meas = list()
        self.model = model
        self.meas_method = meas_method
        self.calc_method = calc_method
        self.anly_method = anly_method
        self.famdata = famdata or si.get_family_data(model)
        self._initialize_data()
        if self.isonline:
            self._connect_to_objects()

    @property
    def calc_method_str(self):
        """."""
        return MeasBeta.METHODS._fields[self._calc_method]

    @property
    def calc_method(self):
        """."""
        return self._calc_method

    @calc_method.setter
    def calc_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._calc_method = int(value in MeasBeta.METHODS._fields[1])
        elif int(value) in MeasBeta.METHODS:
            self._calc_method = int(value)

    @property
    def meas_method_str(self):
        """."""
        return MeasBeta.MEASUREMENT._fields[self._meas_method]

    @property
    def meas_method(self):
        """."""
        return self._meas_method

    @meas_method.setter
    def meas_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._meas_method = int(value in MeasBeta.MEASUREMENT._fields[1])
        elif int(value) in MeasBeta.MEASUREMENT:
            self._meas_method = int(value)

    @property
    def anly_method_str(self):
        """."""
        return MeasBeta.ANALYSIS._fields[self._anly_method]

    @property
    def anly_method(self):
        """."""
        return self._anly_method

    @anly_method.setter
    def anly_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._anly_method = int(value in MeasBeta.ANALYSIS._fields[1])
        elif int(value) in MeasBeta.ANALYSIS:
            self._anly_method = int(value)

    @property
    def measuredquads(self):
        """."""
        return sorted(self.data['measure'])

    @property
    def quads2meas(self):
        """."""
        if self._quads2meas:
            return self._quads2meas
        return sorted(
            set(self.data['quadnames']) - self.data['measure'].keys())

    @quads2meas.setter
    def quads2meas(self, quadslist):
        """."""
        self._quads2meas = _dcopy(quadslist)

    def _connect_to_objects(self):
        """."""
        for qname in self.data['quadnames']:
            if qname not in self.devices:
                self.devices[qname] = PowerSupply(qname)

    def _initialize_data(self):
        """."""
        quadnames, quadsidx = MeasBeta.get_quads(self.model, self.famdata)
        self.data['quadnames'] = quadnames
        twi, *_ = pyaccel.optics.calc_twiss(self.model, indices='open')

        if self.calc_method == MeasBeta.METHODS.Analytic:
            for idx, qname in zip(quadsidx, quadnames):
                L = self.model[idx].length
                K = self.model[idx].K
                Kr = np.sqrt(abs(K))
                bxi = twi.betax[idx]
                byi = twi.betay[idx]
                axi = twi.alphax[idx]
                ayi = twi.alphay[idx]
                gxi = (1+axi*axi)/bxi
                gyi = (1+ayi*ayi)/byi
                self.data['betax_int'][qname] = self._calc_beta_integral(
                    K, L, bxi, axi, gxi)
                self.data['betay_int'][qname] = self._calc_beta_integral(
                    -K, L, byi, ayi, gyi)
        elif self.calc_method == MeasBeta.METHODS.Numeric:
            nux0, nuy0, *_ = pyaccel.optics.get_frac_tunes(self.model)
            dkl = self.params.quad_deltakl
            for idx, qname in zip(quadsidx, quadnames):
                korig = self.model[idx].KL
                if 'QF' in self.model[idx].fam_name:
                    deltakl = +dkl
                else:
                    deltakl = -dkl
                self.model[idx].KL = korig + deltakl
                nuxp, nuyp, *_ = pyaccel.optics.get_frac_tunes(self.model)
                dnux = nuxp - nux0
                dnuy = nuyp - nuy0
                self.data['betax_int'][qname] = +4 * np.pi * dnux/deltakl
                self.data['betay_int'][qname] = -4 * np.pi * dnuy/deltakl
                self.model[idx].KL = korig

    @staticmethod
    def _calc_beta_integral(K, L, beta0, alpha0, gamma0):
        Kr = np.sqrt(abs(K))
        fac1 = math.sin(2*Kr*L) if K > 0 else math.sinh(2*Kr*L)
        fac2 = (math.sin(Kr*L))**2 if K > 0 else \
            (math.sinh(Kr*L))**2
        beta_int = (beta0 + gamma0/K)/2
        beta_int += fac1/Kr/L/4 * (beta0 - gamma0/K)
        beta_int -= alpha0*fac2/abs(K)/L
        return beta_int

    def _meas_beta(self):
        """."""
        sofb = self.devices['sofb']
        ti0 = _time.time()

        _time.sleep(self.params.wait_tune)
        self.data['tunex_ref'] = self.devices['tune'].tunex
        self.data['tuney_ref'] = self.devices['tune'].tuney

        autocorr = sofb.autocorrsts
        if autocorr:
            print('SOFB Loop is enable in SOFB, disabling it...')
            sofb.cmd_turn_off_autocorr()

        self._cycle_quads()

        for quadname in self.quads2meas:
            if self._stopevt.is_set():
                return
            print('\n  measuring quad: ' + quadname, end=' ')
            self._meas_beta_single_quad(quadname)

        if autocorr:
            print('SOFB Loop was enable, restoring original state...')
            sofb.cmd_turn_on_autocorr()

        print('finished!')
        tif = _time.time()
        print('time elapsed: {:.2f} min'.format((tif-ti0)/60))

    @staticmethod
    def get_cycling_curve():
        """."""
        return [+1, 0]

    def _cycle_quads(self):
        tune = self.devices['tune']
        deltakl = self.params.quad_deltakl
        cycling_curve = MeasBeta.get_cycling_curve()
        tunex_cycle = []
        tuney_cycle = []
        print('\n preparing all quads: ', end='')
        for cynum in range(self.params.quad_nrcycles):
            print('\n   cycle: {0:02d}/{1:02d} --> '.format(
                cynum+1, self.params.quad_nrcycles), end='')
            for quadname in self.data['quadnames']:
                if self._stopevt.is_set():
                    print('exiting...')
                    break
                print('\n  cycling quad ' + quadname, end=' ')
                quad = self.devices[quadname]
                korig = quad.strength
                for fac in cycling_curve:
                    quad.strength = korig + deltakl*fac
                    if fac:
                        _time.sleep(self.params.time_wait_quad_cycle)
                tunex_cycle.append(tune.tunex)
                tuney_cycle.append(tune.tuney)

        self.data['cycling']['tunex'] = np.array(tunex_cycle)
        self.data['cycling']['tuney'] = np.array(tuney_cycle)
        print(' Ok!')

    def _recover_tune(self, meas, quadname):
        print('recovering tune...')
        tunex0 = meas['tunex_ini'][0]
        tuney0 = meas['tuney_ini'][0]
        dkl_applied = self._select_deltakl(meas)

        dnux = meas['tunex_pos'][-1] - meas['tunex_ini'][-1]
        dnuy = meas['tuney_pos'][-1] - meas['tuney_ini'][-1]
        cxx = dnux/dkl_applied
        cyy = dnuy/dkl_applied

        _time.sleep(self.params.wait_tune)
        tunex_now = self.devices['tune'].tunex
        tuney_now = self.devices['tune'].tuney
        dtunex = tunex_now - tunex0
        dtuney = tuney_now - tuney0
        dtune_init = np.sqrt(dtunex*dtunex + dtuney*dtuney)
        kl_init = self.devices[quadname].strength

        tol = self.params.recover_tune_tol
        niter = self.params.recover_tune_maxiter

        for _ in range(niter):
            dtune_mod = np.sqrt(dtunex*dtunex + dtuney*dtuney)
            print(
                '   delta tune x, y: {0:.6f}, {1:.6f} ({2:.6f})'.format(
                    dtunex, dtuney, dtune_mod))
            if dtune_mod < tol:
                return True

            # minimization of squares [cx cy]*dkl = [dtunex dtuney]
            dkl = (cxx * dtunex + cyy * dtuney)/(cxx*cxx + cyy*cyy)
            if np.abs(dkl) > np.abs(dkl_applied):
                print('   deltakl calculated is too big!')
                return False
            self.devices[quadname].strength -= dkl

            _time.sleep(self.params.wait_quadrupole)
            _time.sleep(self.params.wait_tune)

            tunex_now = self.devices['tune'].tunex
            tuney_now = self.devices['tune'].tuney
            dtunex = tunex_now - tunex0
            dtuney = tuney_now - tuney0

        if dtune_mod > dtune_init:
            print('   recovering loop increased tune error, reverting changes')
            self.devices[quadname].strength = kl_init
        return False

    def _meas_beta_single_quad(self, quadname):
        """."""
        quad = self.devices[quadname]
        tune = self.devices['tune']
        sofb = self.devices['sofb']

        print('Correcting Orbit...')
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_nriters,
            residue=self.params.sofb_residue)
        print('Done!')

        if not quad.pwrstate:
            print('turning quadrupole ' + quadname + ' On', end='')
            quad.cmd_turn_on(self.params.timeout_quad_turnon)

        if not quad.pwrstate:
            print('\n    error: quadrupole ' + quadname + ' is Off.')
            self._stopevt.set()
            print('    exiting...')
            return

        deltakl = self.params.quad_deltakl
        curr_orig = quad.current
        kl_orig = quad.strength
        quadname = _PVName(quadname)
        dcurr = self.params.DELTA_CURRENT[quadname.dev]
        dkl_ima = self.params.DELTA_GL[quadname.dev] / self.model.brho

        cycling_curve = MeasBeta.get_cycling_curve()

        # measurement always increases the power supply current
        if 'QD' in quadname:
            deltakl *= -1

        tunex_ini, tunex_neg, tunex_pos = [], [], []
        tuney_ini, tuney_neg, tuney_pos = [], [], []
        tunex_wfm_ini, tunex_wfm_neg, tunex_wfm_pos = [], [], []
        tuney_wfm_ini, tuney_wfm_neg, tuney_wfm_pos = [], [], []

        for nrmeas in range(self.params.nr_measures):

            if self._stopevt.is_set():
                print('exiting...')
                break
            print('   meas. {0:02d}/{1:02d} --> '.format(
                nrmeas+1, self.params.nr_measures), end='')

            tunex_ini.append(tune.tunex)
            tuney_ini.append(tune.tuney)
            tunex_wfm_ini.append(tune.tunex_wfm)
            tuney_wfm_ini.append(tune.tuney_wfm)
            for j, fac in enumerate(cycling_curve):
                if self.meas_method == self.MEASUREMENT.Current:
                    quad.current = curr_orig + dcurr*fac
                elif self.meas_method == self.MEASUREMENT.KL:
                    quad.strength = kl_orig + deltakl*fac
                _time.sleep(self.params.wait_quadrupole)
                if not j:
                    kl_new = quad.strength
                    curr_new = quad.current
                    dkl_exc = kl_new - kl_orig
                    dcurr = curr_new - curr_orig
                    if 'QD' in quadname:
                        print(' -dk ', end='')
                    else:
                        print(' +dk ', end='')
                    _time.sleep(self.params.wait_tune)
                    tunex_pos.append(tune.tunex)
                    tuney_pos.append(tune.tuney)
                    tunex_wfm_pos.append(tune.tunex_wfm)
                    tuney_wfm_pos.append(tune.tuney_wfm)
            dnux = tunex_pos[-1] - tunex_ini[-1]
            dnuy = tuney_pos[-1] - tuney_ini[-1]
            print('--> dnux = {:.5f}, dnuy = {:.5f}'.format(dnux, dnuy))

        meas = dict()
        meas['tunex_ini'] = np.array(tunex_ini)
        meas['tuney_ini'] = np.array(tuney_ini)
        meas['tunex_neg'] = np.array(tunex_neg)
        meas['tuney_neg'] = np.array(tuney_neg)
        meas['tunex_pos'] = np.array(tunex_pos)
        meas['tuney_pos'] = np.array(tuney_pos)
        meas['tunex_wfm_ini'] = np.array(tunex_wfm_ini)
        meas['tuney_wfm_ini'] = np.array(tuney_wfm_ini)
        meas['tunex_wfm_neg'] = np.array(tunex_wfm_neg)
        meas['tuney_wfm_neg'] = np.array(tuney_wfm_neg)
        meas['tunex_wfm_pos'] = np.array(tunex_wfm_pos)
        meas['tuney_wfm_pos'] = np.array(tuney_wfm_pos)
        meas['delta_kl'] = deltakl
        meas['dkl_ima'] = dkl_ima
        meas['dkl_exc'] = dkl_exc
        meas['dcurr'] = dcurr

        if self.params.recover_tune:
            tune_recovered = self._recover_tune(meas, quadname)
            if tune_recovered:
                print('tune recovered!')
            else:
                print('could not recover tune for :{:s}'.format(quadname))
            meas['tune_recovered'] = tune_recovered

        self.data['measure'][quadname] = meas

    def _select_deltakl(self, datameas):
        if self.meas_method == self.MEASUREMENT.Current:
            if self.anly_method == self.ANALYSIS.IMA:
                dkl = datameas['dkl_ima']
            elif self.anly_method == self.ANALYSIS.Excdata:
                dkl = datameas['dkl_exc']
        if self.meas_method == self.MEASUREMENT.KL:
            dkl = datameas['delta_kl']
        return dkl

    def process_data(self, mode='pos', discardpoints=None):
        """."""
        for quad in self.data['measure']:
            self.analysis[quad] = self.calc_beta(
                quad, mode=mode, discardpoints=discardpoints)

    def calc_beta(
            self, quadname, mode='pos', discardpoints=None):
        """."""
        anl = dict()
        datameas = self.data['measure'][quadname]

        if mode.lower().startswith('symm'):
            dnux = datameas['tunex_pos'] - datameas['tunex_neg']
            dnuy = datameas['tuney_pos'] - datameas['tuney_neg']
        elif mode.lower().startswith('pos'):
            dnux = datameas['tunex_pos'] - datameas['tunex_ini']
            dnuy = datameas['tuney_pos'] - datameas['tuney_ini']
        else:
            dnux = datameas['tunex_ini'] - datameas['tunex_neg']
            dnuy = datameas['tuney_ini'] - datameas['tuney_neg']

        usepts = set(range(dnux.shape[0]))
        if discardpoints is not None:
            usepts = set(usepts) - set(discardpoints)
        usepts = sorted(usepts)

        dkl = self._select_deltakl(datameas)
        anl['betasx'] = +4 * np.pi * dnux[usepts] / dkl
        anl['betasy'] = -4 * np.pi * dnuy[usepts] / dkl
        anl['betax_ave'] = np.mean(anl['betasx'])
        anl['betay_ave'] = np.mean(anl['betasy'])
        return anl

    @staticmethod
    def get_quads(model, fam_data):
        """."""
        quads_idx = _dcopy(fam_data['QN']['index'])
        quads_idx = np.array([idx[len(idx)//2] for idx in quads_idx])

        qnames = list()
        for qidx in quads_idx:
            name = model[qidx].fam_name
            idc = fam_data[name]['index'].index([qidx, ])
            sub = fam_data[name]['subsection'][idc]
            inst = fam_data[name]['instance'][idc]
            qname = 'SI-{0:s}:PS-{1:s}-{2:s}'.format(sub, name, inst)
            qnames.append(qname.strip('-'))
        return qnames, quads_idx

    def get_betas(self, quads=None):
        """."""
        beta_data = dict()
        quads = quads or self.data['quadnames']
        indcs, nom_bx, nom_by = [], [], []
        mes_bx, mes_by, bx_ave, by_ave = [], [], [], []
        for quad in quads:
            if quad not in self.analysis:
                continue
            indcs.append(self.data['quadnames'].index(quad))
            nom_bx.append(self.data['betax_int'][quad])
            nom_by.append(self.data['betay_int'][quad])
            mes_bx.append(self.analysis[quad]['betasx'])
            mes_by.append(self.analysis[quad]['betasy'])
            bx_ave.append(self.analysis[quad]['betax_ave'])
            by_ave.append(self.analysis[quad]['betay_ave'])

        spos = pyaccel.lattice.find_spos(self.model)
        qnidx = np.array(self.famdata['QN']['index']).ravel()

        beta_data['indices'] = np.array(indcs)
        beta_data['quad_spos'] = spos[qnidx[indcs]]
        beta_data['betax_nom'] = np.array(nom_bx)
        beta_data['betay_nom'] = np.array(nom_by)
        beta_data['betax_meas'] = np.array(mes_bx)
        beta_data['betay_meas'] = np.array(mes_by)
        beta_data['betax_meas_ave'] = np.array(bx_ave)
        beta_data['betay_meas_ave'] = np.array(by_ave)
        return beta_data

    def plot_betas(self, quads=None, title='', scale=1):
        """."""
        fig = plt.figure(figsize=(9, 7))
        grids = mpl_gs.GridSpec(ncols=1, nrows=2, figure=fig)
        grids.update(
            left=0.1, right=0.95, bottom=0.15, top=0.9,
            hspace=0.0, wspace=0.35)

        ax1 = fig.add_subplot(grids[0, 0])
        ax2 = fig.add_subplot(grids[1, 0], sharex=ax1)

        if title:
            fig.suptitle(title)

        beta_data = self.get_betas(quads=quads)
        quad_spos = beta_data['quad_spos']

        ax1.plot(quad_spos, beta_data['betax_nom'], '-', color='C0')
        ax1.plot(
            quad_spos, beta_data['betax_meas_ave']*scale, '--', color='C1')
        ax1.plot(quad_spos, beta_data['betax_meas']*scale, '.', color='C1')

        ax2.plot(quad_spos, beta_data['betay_nom'], '-', color='C0')
        ax2.plot(
            quad_spos, beta_data['betay_meas_ave']*scale, '--', color='C1')
        ax2.plot(quad_spos, beta_data['betay_meas']*scale, '.', color='C1')

        ax1.set_ylabel(r'$\beta_x$ [m]')
        ax2.set_ylabel(r'$\beta_y$ [m]')
        ax2.set_xlabel('s [m]')
        ax1.legend(
            ['nominal', 'measurement'], loc='lower right',
            bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2)
        ax1.grid(True, ls='--')
        ax2.grid(True, ls='--')
        fig.show()

    def plot_beta_beating(self, quads=None, title='', scale=1):
        """."""
        fig = plt.figure(figsize=(9, 7))
        grids = mpl_gs.GridSpec(ncols=1, nrows=1, figure=fig)
        grids.update(
            left=0.1, right=0.95, bottom=0.15, top=0.9,
            hspace=0.0, wspace=0.35)

        ax1 = fig.add_subplot(grids[0, 0])

        if title:
            fig.suptitle(title)

        beta_data = self.get_betas(quads=quads)
        quad_spos = beta_data['quad_spos']

        bbeatx = beta_data['betax_meas_ave']*scale - beta_data['betax_nom']
        bbeatx /= beta_data['betax_nom']

        bbeaty = beta_data['betay_meas_ave']*scale - beta_data['betay_nom']
        bbeaty /= beta_data['betay_nom']

        ax1.plot(quad_spos, bbeatx * 100, '.-', color='tab:blue')
        ax1.plot(quad_spos, bbeaty * 100, '.-', color='tab:red')

        bbeatx_std = np.std(bbeatx) * 100
        bbeaty_std = np.std(bbeaty) * 100

        strx = r'$\Delta \beta_x/\beta_x$ ={:.2f}% (std)'
        stry = r'$\Delta \beta_y/\beta_y$ ={:.2f}% (std)'
        textstr = '\n'.join(
            (strx.format(bbeatx_std), stry.format(bbeaty_std)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

        # place a text box in upper left in axes coords
        ax1.text(
            0.35, 0.95, textstr, transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

        ax1.set_ylabel(r'$\Delta\beta/\beta$ [%]')
        ax1.set_xlabel('s [m]')
        ax1.legend(
            ['horizontal', 'vertical'], loc='lower right',
            bbox_to_anchor=(1, 1), fontsize='x-small', ncol=2)
        ax1.grid(True, ls='--')
        fig.show()
