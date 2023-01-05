from copy import deepcopy as _dcopy
import numpy as np
import pandas as pd
from mathphys.functions import load_pickle, save_pickle

import pyaccel
from pymodels import si

import siriuspy.clientconfigdb as servconf

from apsuite.loco.utils import LOCOUtils
from apsuite.loco.config import LOCOConfigSI
from apsuite.loco.main import LOCO

from apsuite.orbcorr import OrbRespmat
from apsuite.optics_analysis.tune_correction import TuneCorr

import matplotlib.pyplot as plt
import matplotlib.gridspec as mpl_gs
import matplotlib.cm as cm
from matplotlib import rcParams, rc
import matplotlib.style
import matplotlib as mpl
from fpdf import FPDF
import datetime

rc('font', **{'size': 14})


class LOCOAnalysis():
    """."""

    def __init__(self, fname_setup, fname_fit):
        """."""
        self.fname_setup = fname_setup
        self.fname_fit = fname_fit
        self.loco_setup = None
        self.loco_fit = None
        self.nom_model = None
        self.twi_nom = None
        self.twi_fit = None
        self.disp_meas = None

    def get_setup(self):
        """."""
        loco_setup = load_pickle(self.fname_setup)
        if 'data' in loco_setup:
            loco_setup = loco_setup['data']
        self.loco_setup = loco_setup

    def calc_twiss(self):
        """."""
        self.twi_fit, *_ = pyaccel.optics.calc_twiss(
            self.loco_fit['fit_model'], indices='open')
        # edteng_fit, *_ = pyaccel.optics.calc_edwards_teng(
        #     self.loco_fit['fit_model'], indices='open')

        if self.nom_model is None:
            self.nom_model, _ = self.get_nominal_model()

        self.twi_nom, *_ = pyaccel.optics.calc_twiss(
            self.nom_model, indices='open')
        # edteng_nom, *_ = pyaccel.optics.calc_edwards_teng(
        #     self.nom_model, indices='open')

    def get_nominal_model(self):
        """."""
        # Get nominal model
        simod = si.create_accelerator()
        simod.cavity_on = True
        simod.radiation_on = True

        # Adjust tunes to match measured ones
        tunex_goal = 49 + self.loco_setup['tunex']
        tuney_goal = 14 + self.loco_setup['tuney']

        # print('--- correcting si tunes...')
        tunecorr = TuneCorr(
            simod, 'SI', method='Proportional', grouping='TwoKnobs')
        tunecorr.get_tunes(simod)
        # print('    tunes init  : ', tunecorr.get_tunes(simod))
        tunemat = tunecorr.calc_jacobian_matrix()
        tunecorr.correct_parameters(
            model=simod,
            goal_parameters=np.array([tunex_goal, tuney_goal]),
            jacobian_matrix=tunemat)
        # print('    tunes final : ', tunecorr.get_tunes(simod))

        # Get nominal orbit matrix and dispersion
        matrix_nominal = OrbRespmat(simod, 'SI', '6d').get_respm()

        alpha0 = pyaccel.optics.get_mcf(simod)
        # print('momentum compaction: {:e}'.format(alpha0))
        idx = pyaccel.lattice.find_indices(
            simod, 'pass_method', 'cavity_pass')[0]
        # rf_freq = setup['rf_frequency']
        rf_freq = simod[idx].frequency

        rfline = matrix_nominal[:, -1]
        disp_nominal = self._get_dispersion(rfline, alpha0, rf_freq)
        self.nom_model = simod
        return simod, disp_nominal

    def get_loco_results(self):
        """."""
        loco_data = load_pickle(self.fname_fit)

        config = loco_data['config']
        model_fitting = loco_data['fit_model']
        gain_bpm = loco_data['gain_bpm']
        gain_corr = loco_data['gain_corr']
        roll_bpm = loco_data['roll_bpm']

        self.loco_fit = loco_data

        matrix_fitting = OrbRespmat(model_fitting, 'SI', '6d').get_respm()
        matrix_fitting = LOCOUtils.apply_all_gain(
            matrix_fitting, gain_bpm, roll_bpm, gain_corr)

        alpha_fit = pyaccel.optics.get_mcf(model_fitting)
        idx = pyaccel.lattice.find_indices(
            model_fitting, 'pass_method', 'cavity_pass')[0]
        rf_freq_mod = model_fitting[idx].frequency
        rf_freq = self.loco_setup['rf_frequency']

        rfline_fit = matrix_fitting[:, -1]
        disp_fit = self._get_dispersion(rfline_fit, alpha_fit, rf_freq_mod)
        rfline_meas = config.goalmat[:, -1]
        disp_meas = self._get_dispersion(rfline_meas, alpha_fit, rf_freq)
        self.disp_meas = disp_meas
        return loco_data, matrix_fitting, disp_fit, disp_meas

# ============= static methods =============

    @staticmethod
    def get_famidx_quad(model):
        """."""
        famidx = []
        famlist = [
            'QFA', 'QDA', 'QFB', 'QDB1', 'QDB2', 'QFP', 'QDP1', 'QDP2',
            'Q1', 'Q2', 'Q3', 'Q4']
        famdata = si.get_family_data(model)
        for fam_name in famlist:
            famidx.append(famdata[fam_name]['index'])
        return famidx

    @staticmethod
    def get_famidx_sext(model):
        """."""
        famidx = []
        famdata = si.get_family_data(model)
        for fam_name in si.families.families_sextupoles:
            famidx.append(famdata[fam_name]['index'])
        return famidx

    @staticmethod
    def get_attribute_quad(model):
        """."""
        kl_strength = []
        ksl_strength = []
        famidx = LOCOAnalysis.get_famidx_quad(model)
        for q in famidx:
            kl_strength.append(pyaccel.lattice.get_attribute(model, 'KL', q))
            ksl_strength.append(pyaccel.lattice.get_attribute(model, 'KsL', q))
        kl_strength = np.array(kl_strength, dtype=list)
        ksl_strength = np.array(ksl_strength, dtype=list)
        return kl_strength, ksl_strength

    @staticmethod
    def get_attribute_sext(model):
        """."""
        kl_strength = []
        sl_strength = []
        ksl_strength = []
        famidx = LOCOAnalysis.get_famidx_sext(model)
        for q in famidx:
            kl_strength.append(pyaccel.lattice.get_attribute(model, 'KL', q))
            sl_strength.append(pyaccel.lattice.get_attribute(model, 'SL', q))
            ksl_strength.append(pyaccel.lattice.get_attribute(model, 'KsL', q))
        kl_strength = np.array(kl_strength)
        sl_strength = np.array(sl_strength)
        ksl_strength = np.array(ksl_strength)
        return kl_strength, sl_strength, ksl_strength

    @staticmethod
    def plot_histogram(
            diff_nominal, diff_loco,
            kickx=15, kicky=15*1.5, kickrf=15*5*1e6, save=False, fname=None):
        """."""
        fig = plt.figure(figsize=(12, 8))
        gs = mpl_gs.GridSpec(2, 2)
        gs.update(left=0.12, right=0.98, top=0.97, bottom=0.10, hspace=0.25)
        axx = plt.subplot(gs[0, 0])
        axy = plt.subplot(gs[0, 1])
        ayy = plt.subplot(gs[1, 1])
        ayx = plt.subplot(gs[1, 0])

        nbpm, nch = 160, 120

        dnom = diff_nominal.copy()
        dloc = diff_loco.copy()

        dnom[:, :nch] *= kickx
        dnom[:, nch:-1] *= kicky
        dnom[:, -1] *= kickrf

        dloc[:, :nch] *= kickx
        dloc[:, nch:-1] *= kicky
        dloc[:, -1] *= kickrf

        rmsnomyx = np.sqrt(np.mean(dnom[nbpm:, :nch]**2))
        rmsnomxy = np.sqrt(np.mean(dnom[:nbpm, nch:-1]**2))
        rmsnomxx = np.sqrt(np.mean(dnom[:nbpm, :nch]**2))
        rmsnomyy = np.sqrt(np.mean(dnom[nbpm:, nch:-1]**2))
        rmslocyx = np.sqrt(np.mean(dloc[nbpm:, :nch]**2))
        rmslocxy = np.sqrt(np.mean(dloc[:nbpm, nch:-1]**2))
        rmslocxx = np.sqrt(np.mean(dloc[:nbpm, :nch]**2))
        rmslocyy = np.sqrt(np.mean(dloc[nbpm:, nch:-1]**2))

        ayx.hist(
            dnom[nbpm:, :nch].flatten(), bins=100*2,
            label=r'nom. $\sigma_{{yx}} = {:.2f}\mu$m'.format(rmsnomyx),
            density=True)
        ayx.hist(
            dloc[nbpm:, :nch].flatten(), bins=100*2,
            label=r'fit. $\sigma_{{yx}} = {:.2f}\mu$m'.format(rmslocyx),
            density=True)
        ayx.set_xlabel(r'$\Delta y$ [$\mu$m]')
        ayx.set_ylabel(r'# of $M_{yx}$ elements')

        axy.hist(
            dnom[:nbpm, nch:-1].flatten(), bins=90*2,
            label=r'nom. $\sigma_{{xy}} = {:.2f}\mu$m'.format(rmsnomxy),
            density=True)
        axy.hist(
            dloc[:160, nch:-1].flatten(), bins=90*2,
            label=r'fit. $\sigma_{{xy}} = {:.2f}\mu$m'.format(rmslocxy),
            density=True)
        axy.set_xlabel(r'$\Delta x$ [$\mu$m]')
        axy.set_ylabel(r'# of $M_{xy}$ elements')

        axx.hist(
            dnom[:nbpm, :nch].flatten(), bins=100*2,
            label=r'nom. $\sigma_{{xx}} = {:.2f}\mu$m'.format(rmsnomxx),
            density=True)
        axx.hist(
            dloc[:nbpm, :nch].flatten(), bins=100*2,
            label=r'fit. $\sigma_{{xx}} = {:.2f}\mu$m'.format(rmslocxx),
            density=True)
        axx.set_xlabel(r'$\Delta x$ [$\mu$m]')
        axx.set_ylabel(r'# of $M_{xx}$ elements')

        ayy.hist(
            dnom[nbpm:, nch:-1].flatten(), bins=90*2,
            label=r'nom. $\sigma_{{yy}} = {:.2f}\mu$m'.format(rmsnomyy),
            density=True)
        ayy.hist(
            dloc[nbpm:, nch:-1].flatten(), bins=90*2,
            label=r'fit. $\sigma_{{yy}} = {:.2f}\mu$m'.format(rmslocyy),
            density=True)
        ayy.set_xlabel(r'$\Delta y$ [$\mu$m]')
        ayy.set_ylabel(r'# of $M_{yy}$ elements')

        axx.legend(loc='upper right')
        axy.legend(loc='upper right')
        ayx.legend(loc='upper right')
        ayy.legend(loc='upper right')
        if save:
            fig.savefig(fname + '.png', dpi=300, format='png')

    @staticmethod
    def plot_quadrupoles_gradients_by_family(
            nom_model, fit_model, save=False, fname=None):
        """."""
        fig = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        kfit_mean = []
        knom_mean = []
        kfit_std = []
        maxmin = []
        kfit, *_ = LOCOAnalysis.get_attribute_quad(fit_model)
        knom, *_ = LOCOAnalysis.get_attribute_quad(nom_model)
        count = 0
        famlist = [
            'QFA', 'QDA', 'QFB', 'QDB1', 'QDB2', 'QFP', 'QDP1', 'QDP2',
            'Q1', 'Q2', 'Q3', 'Q4']
        for q, _ in enumerate(famlist):
            clr = plt.cm.jet(q/len(famlist))
            npoints = len(kfit[q])
            kf = np.array(kfit[q])
            kn = np.array(knom[q])
            init = count
            final = init + npoints - 1
            span = np.linspace(init, final, npoints)
            perc = (kf - kn)/kn * 100
            p = ax1.plot(span, perc, '-o', label=famlist[q], color=clr)
            ax1.plot(
                span, np.repeat(np.mean(perc), npoints), '-',
                color=clr)
            ax1.plot(
                span, np.repeat(np.mean(perc) + np.std(perc), npoints), '--',
                color=clr)
            ax1.plot(
                span, np.repeat(np.mean(perc) - np.std(perc), npoints), '--',
                color=clr)
            count += len(kfit[q])
            kfit_mean.append(np.mean(perc))
            kfit_std.append(np.std(perc))
            knom_mean.append(np.mean(perc))
            maxmin.append(np.ptp(perc))
        stats = {
            'family name': famlist,
            'avg [%]': kfit_mean,
            'std [%]': kfit_std,
            'p2p [%]': maxmin,
        }
        df_stats = pd.DataFrame.from_dict(stats).round(4)
        ax1.set_xlabel('quadrupole index')
        ax1.set_ylabel('quadrupole variation [%]')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(alpha=0.5, linestyle='--')
        plt.tight_layout()
        if save:
            if fname is None:
                fig.savefig(
                    'quadrupoles_gradients.png', format='png', dpi=1200)
            else:
                fig.savefig(fname+'.png', format='png', dpi=300)
        return df_stats

    @staticmethod
    def plot_quadrupoles_gradients_by_s(
            nom_model, fit_model, save=False, fname=None):
        """."""
        fig = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        spos = pyaccel.lattice.find_spos(nom_model)
        fam_nom = si.get_family_data(nom_model)
        qnlist_nom = fam_nom['QN']['index']
        qnidx = np.array(qnlist_nom).flatten()
        knom = np.array(pyaccel.lattice.get_attribute(
            nom_model, 'KL', qnlist_nom))

        fam_fit = si.get_family_data(fit_model)
        qnlist_fit = fam_fit['QN']['index']
        kfit = np.array(pyaccel.lattice.get_attribute(
            fit_model, 'KL', qnlist_fit))
        perc = (kfit-knom)/knom * 100
        ax1.plot(spos[qnidx], perc, '.-', label='deviation')
        for idx in range(21):
            ax1.axvline(spos[-1]/20 * idx, ls='--', color='k', lw=1)
        ax1.set_xlabel('s [m]')
        ax1.set_ylabel('relative variation [%]')
        plt.tight_layout()
        if save:
            if fname is None:
                fig.savefig('quadrupoles_gradients.png', format='png', dpi=600)
            else:
                fig.savefig(fname+'.png', format='png', dpi=300)
        return kfit, knom, perc

    @staticmethod
    def plot_skew_quadrupoles(
            nom_model, fit_model, save=False, fname=None):
        """."""
        _ = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        # gs.update(left=0.10, right=0.85, hspace=0, wspace=0.25)
        ax1 = plt.subplot(gs[0, 0])

        spos = pyaccel.lattice.find_spos(nom_model)
        fam_nom = si.get_family_data(nom_model)
        qslist_nom = fam_nom['QS']['index']
        qsidx = np.array(qslist_nom).flatten()
        knom = np.array(pyaccel.lattice.get_attribute(
            nom_model, 'KsL', qslist_nom))

        fam_fit = si.get_family_data(fit_model)
        qslist_fit = fam_fit['QS']['index']
        kfit = np.array(pyaccel.lattice.get_attribute(
            fit_model, 'KsL', qslist_fit))
        if sum(knom):
            percentage = (kfit-knom)/knom * 100
        else:
            percentage = kfit
        ax1.plot(spos[qsidx], percentage, '.-')

        for idx in range(21):
            ax1.axvline(spos[-1]/20 * idx, ls='--', color='k', lw=1)
        ax1.set_xlabel('s [m]')
        ax1.set_ylabel('integrated skew quadrupole [1/m]')
        plt.tight_layout()
        if save:
            if fname is None:
                plt.savefig(
                    'skew_quadrupoles_skew_gradients.png',
                    format='png', dpi=300)
            else:
                plt.savefig(fname+'.png', format='png', dpi=300)
        return kfit, knom, percentage

    def plot_gain(self, save=False, fname=None):
        """."""
        _, axs = plt.subplots(3, 1, figsize=(8, 10))

        gain_bpm = self.loco_fit['gain_bpm']
        gain_corr = self.loco_fit['gain_corr']
        roll_bpm = self.loco_fit['roll_bpm']

        color_corr = 'tab:red'
        color_gain_bpm = 'tab:blue'
        color_roll_bpm = 'tab:green'

        axs[0].plot(gain_bpm, '.-', color=color_gain_bpm)
        avg, std,  = np.mean(gain_bpm), np.std(gain_bpm)
        axs[0].axhline(avg, ls='-', color=color_gain_bpm)
        axs[0].axhline(avg + std, ls='--', color=color_gain_bpm)
        axs[0].axhline(avg - std, ls='--', color=color_gain_bpm)
        axs[0].grid(True)
    #     axs[0].set_xlabel('bpm idx')
        axs[0].set_ylabel('gain')
        axs[0].set_title('BPM Gains')

        axs[1].plot(gain_corr, '.-', color=color_corr)
        avg, std,  = np.mean(gain_corr), np.std(gain_corr)
        axs[1].axhline(avg, ls='-', color=color_corr)
        axs[1].axhline(avg + std, ls='--', color=color_corr)
        axs[1].axhline(avg - std, ls='--', color=color_corr)
        axs[1].grid(True)
    #     axs[1].set_xlabel('corrector idx')
        axs[1].set_ylabel('gain')
        axs[1].set_title('Corrector Gains')

        axs[2].plot(roll_bpm*1e3, '.-', color=color_roll_bpm)
        avg, std,  = np.mean(roll_bpm)*1e3, np.std(roll_bpm)*1e3
        axs[2].axhline(avg, ls='-', color=color_roll_bpm)
        axs[2].axhline(avg + std, ls='--', color=color_roll_bpm)
        axs[2].axhline(avg - std, ls='--', color=color_roll_bpm)

        axs[2].grid(True)
        axs[2].set_xlabel('index')
        axs[2].set_ylabel('roll [mrad]')
        axs[2].set_title('BPM Roll')
        plt.tight_layout()
        if save:
            if fname is None:
                plt.savefig('gains.png', format='png', dpi=300)
            else:
                plt.savefig(fname+'.png', format='png', dpi=300)

    def beta_and_tune(self):
        """."""
        twi_nom, twi_fit = self.twi_nom, self.twi_fit
        if twi_nom.spos.size != twi_fit.spos.size:
            if twi_fit.spos.size > twi_nom.spos.size:
                spos = twi_fit.spos
                betax_nom = np.interp(spos, twi_nom.spos, twi_nom.betax)
                betay_nom = np.interp(spos, twi_nom.spos, twi_nom.betay)
                betax_fit = twi_fit.betax
                betay_fit = twi_fit.betay
            else:
                spos = twi_nom.spos
                betax_fit = np.interp(spos, twi_fit.spos, twi_fit.betax)
                betay_fit = np.interp(spos, twi_fit.spos, twi_fit.betay)
                betax_nom = twi_nom.betax
                betay_nom = twi_nom.betay
        else:
            spos = twi_nom.spos
            betax_nom, betay_nom = twi_nom.betax, twi_nom.betay
            betax_fit, betay_fit = twi_fit.betax, twi_fit.betay

        beta_beatx = (betax_fit - betax_nom)/betax_nom * 100
        beta_beaty = (betay_fit - betay_nom)/betay_nom * 100

        _ = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

        ax.plot(twi_nom.spos, beta_beatx, label='Horizontal')
        ax.plot(twi_nom.spos, beta_beaty, label='Vertical')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = r"$\Delta \beta_x/\beta_x$ = {:.2f}% (std)".format(
            np.std(beta_beatx))
        textstr += "\n"
        textstr += r"$\Delta \beta_y/\beta_y$ = {:.2f}% (std)".format(
            np.std(beta_beaty))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

        # place a text box in upper left in axes coords
        ax.text(0.8, 0.17, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        ax.set_xlabel('s [m]')
        ax.set_ylabel(r'$\Delta \beta/\beta$ [%]')
        # ax.set_title(r'Beta-beating - Step $\Delta K$ constraint')
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            'beta_beating.png', dpi=300)

        tunex_meas = 49 + self.loco_setup['tunex']
        tuney_meas = 14 + self.loco_setup['tuney']

        tunex_nom = twi_nom.mux[-1]/2/np.pi
        tuney_nom = twi_nom.muy[-1]/2/np.pi

        tunex_fit = twi_fit.mux[-1]/2/np.pi
        tuney_fit = twi_fit.muy[-1]/2/np.pi

        diffx = tunex_meas - tunex_fit
        diffy = tuney_meas - tuney_fit

        tunex_list = [tunex_nom, tunex_meas, tunex_fit, diffx]
        tuney_list = [tuney_nom, tuney_meas, tuney_fit, diffy]

        names = [
            'initial nom. model', 'measured values',
            'LOCO model', 'meas. - LOCO']
        df_tunes = pd.DataFrame(
            {'name': names, 'tunex': tunex_list, 'tuney': tuney_list})

        beta_list = [beta_beatx, beta_beaty]
        avg_list = [float(np.mean(val)) for val in beta_list]
        std_list = [float(np.std(val)) for val in beta_list]
        ptp_list = [float(np.ptp(val)) for val in beta_list]

        stats = {
            'beta beating': ['x', 'y'],
            'avg [%]': avg_list,
            'std [%]': std_list,
            'p2p [%]': ptp_list}
        df_betabeat = pd.DataFrame.from_dict(stats)
        tmp = df_betabeat.select_dtypes(include=[np.number])
        df_betabeat.loc[:, tmp.columns] = np.round(tmp, 4)
        return df_tunes.round(6), df_betabeat

    def dispersion(self, disp_meas=None):
        """."""
        _ = plt.figure(figsize=(14, 8))
        gs = mpl_gs.GridSpec(2, 1)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])

        bpm_idx_nom = np.array(
            pyaccel.lattice.find_indices(
                self.nom_model, 'fam_name', 'BPM')).flatten()

        bpm_idx_fit = np.array(
            pyaccel.lattice.find_indices(
                self.loco_fit['fit_model'], 'fam_name', 'BPM')).flatten()

        twi_nom, twi_fit = self.twi_nom, self.twi_fit

        disp0x = twi_nom.etax[bpm_idx_nom]*100
        disp0y = twi_nom.etay[bpm_idx_nom]*100

        dispx_fit = twi_fit.etax[bpm_idx_fit]*100
        dispy_fit = twi_fit.etay[bpm_idx_fit]*100

        if disp_meas is None:
            disp_meas = self.disp_meas
        dispx_meas = disp_meas[:160]
        dispy_meas = disp_meas[160:]

        errx_mf = dispx_meas - dispx_fit
        erry_mf = dispy_meas - dispy_fit

        errx_nf = disp0x - dispx_fit
        erry_nf = disp0y - dispy_fit

        errx_nm = disp0x - dispx_meas
        erry_nm = disp0y - dispy_meas

        errx_list = [errx_mf, errx_nf, errx_nm]
        erry_list = [erry_mf, erry_nf, erry_nm]
        # 10 factor [cm] -> [mm]
        avg_listx = [float(np.mean(val))*10 for val in errx_list]
        avg_listy = [float(np.mean(val))*10 for val in erry_list]
        std_listx = [float(np.std(val))*10 for val in errx_list]
        std_listy = [float(np.std(val))*10 for val in erry_list]
        ptp_listx = [float(np.ptp(val))*10 for val in errx_list]
        ptp_listy = [float(np.ptp(val))*10 for val in erry_list]

        stats = {
            'diff': ['meas - fit', 'nom - fit', 'nom - meas'],
            'avg x [mm]': avg_listx,
            'std x [mm]': std_listx,
            'p2p x [mm]': ptp_listx,
            'avg y [mm]': avg_listy,
            'std y [mm]': std_listy,
            'p2p y [mm]': ptp_listy}
        df_disp = pd.DataFrame.from_dict(stats)
        tmp = df_disp.select_dtypes(include=[np.number])
        df_disp.loc[:, tmp.columns] = np.round(tmp, 4)

        spos = twi_nom.spos[bpm_idx_nom]

        ax1.plot(spos, disp0x, '.-', label='nominal', linewidth=1)
        ax1.plot(spos, dispx_fit, '.-', label='fitting', linewidth=1)
        ax1.plot(spos, dispx_meas, '.-', label='meas', linewidth=1)
        ax1.plot(spos, dispx_fit-dispx_meas, '.-', label='error', linewidth=1)

        ax2.plot(spos, disp0y, '.-', label='nominal', linewidth=1)
        ax2.plot(spos, dispy_fit, '.-', label='fitting', linewidth=1)
        ax2.plot(spos, dispy_meas, '.-', label='meas', linewidth=1)
        ax2.plot(spos, dispy_fit-dispy_meas, '.-', label='error', linewidth=1)

        ax1.tick_params(axis='x', which='minor', bottom=True)
        ax1.tick_params(axis='y', which='minor', bottom=True)
        ax1.minorticks_on

        # Edit the major and minor ticks of the x and y axes
        ax1.xaxis.set_tick_params(
            which='major', size=5, width=1, direction='in', top='on')
        ax1.xaxis.set_tick_params(
            which='minor', size=7, width=1, direction='in', top='on')
        ax1.yaxis.set_tick_params(
            which='major', size=5, width=1, direction='in', right='on')
        ax1.yaxis.set_tick_params(
            which='minor', size=7, width=1, direction='in', right='on')

        # ax1.set_xlabel('bpm idx')
        ax1.set_ylabel(r'$\eta_x$ [cm]')
        # ax1.set_title('Dispersion function at BPMs')
        ax1.legend(bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=12)

        ax2.tick_params(axis='x', which='minor', bottom=True)
        ax2.tick_params(axis='y', which='minor', bottom=True)
        ax2.minorticks_on

        # Edit the major and minor ticks of the x and y axes
        ax2.xaxis.set_tick_params(
            which='major', size=5, width=1, direction='in', top='on')
        ax2.xaxis.set_tick_params(
            which='minor', size=7, width=1, direction='in', top='on')
        ax2.yaxis.set_tick_params(
            which='major', size=5, width=1, direction='in', right='on')
        ax2.yaxis.set_tick_params(
            which='minor', size=7, width=1, direction='in', right='on')

        ax2.set_xlabel('s [m]')
        ax2.set_ylabel(r'$\eta_y$ [cm]')
        plt.tight_layout()
        plt.savefig('dispersion.png', dpi=300)
        return df_disp

    def emittance(self):
        """."""
        eqnom = pyaccel.optics.EqParamsFromBeamEnvelope(self.nom_model)
        eqfit = pyaccel.optics.EqParamsFromBeamEnvelope(
            self.loco_fit['fit_model'])

        m2pm = 1e12
        names = [
            'x [pm.rad]', 'y [pm.rad]',
            'ratio [%]']
        emit_nom_list = [
                eqnom.emit1*m2pm, eqnom.emit2*m2pm,
                eqnom.emit2/eqnom.emit1*100]
        emit_fit_list = [
                eqfit.emit1*m2pm, eqfit.emit2*m2pm,
                eqfit.emit2/eqfit.emit1*100]
        emit_nom_list = [float(abs(val)) for val in emit_nom_list]
        emit_fit_list = [float(abs(val)) for val in emit_fit_list]
        emits = {
            'emittance': names,
            'initial nom model': emit_nom_list,
            'LOCO model': emit_fit_list}
        df_emits = pd.DataFrame.from_dict(emits)
        tmp = df_emits.select_dtypes(include=[np.number])
        df_emits.loc[:, tmp.columns] = np.round(tmp, 4)
        return df_emits

# ============= private methods =============

    @staticmethod
    def _get_dispersion(rfline, alpha, rf_freq):
        return - alpha * rf_freq * rfline * 1e2
