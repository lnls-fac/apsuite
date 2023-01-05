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
# rc('text', usetex=True)
# rcParams['savefig.dpi'] = 300
# from mpl_toolkits.mplot3d import Axes3D


class LOCOReport(FPDF):
    """."""

    def __init__(self, loco_data=None):
        """."""
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.loco_data = loco_data

    def header(self):
        """."""
        self.image('header.jpg', x=10, y=6, w=40, h=15)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 6, 'SIRIUS LOCO Report', 0, 0, 'C')
        self.set_font('Arial', '', 12)
        today = datetime.date.today()
        self.cell(0, 6, '{:s}'.format(today.strftime("%Y-%m-%d")), 0, 0, 'R')
        self.ln(10)

    def footer(self):
        """."""
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_title(self, title, loc_y=None):
        """."""
        self.set_font('Arial', '', 12)
        self.set_fill_color(215)
        if loc_y is not None:
            self.set_y(loc_y)
        self.cell(0, 6, f'{title:s}', 0, 1, 'C', 1)
        self.ln(2)

    def page_body(self, images):
        """."""
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        # if len(images) == 3:
        #     self.image(images[0], 15, 25, self.WIDTH - 30)
        #     self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        #     self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        # elif len(images) == 2:
        #     self.image(images[0], 15, 25, self.WIDTH - 30)
        #     self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        # else:
        self.image(images[0], 15, 50, self.WIDTH - 30)
        self.image(images[1], 15, 150, self.WIDTH - 30)

    def quadrupole_fit(self):
        """."""
        self.image('quad_by_family.png', 10, 120, self.WIDTH - 30)
        self.image('quad_by_s.png', 10, 200, self.WIDTH - 30)

    def loco_fingerprint(self):
        """."""
        setup = self.loco_data['setup']
        self.set_font('Arial', 'B', 10)
        table_cell_width = self.WIDTH/3
        table_cell_height = 5

        self.ln(table_cell_height)
        self.set_font('Arial', '', 10)
        tstamp = datetime.datetime.fromtimestamp(setup['timestamp'])
        tstamp = tstamp.strftime('%Y-%m-%d %H:%M:%S')
        data = (
            ('Timestamp', tstamp),
            ('Stored current', f"{setup['stored_current']:.2f} mA"),
            ('Orbit response matrix on ServConf', setup['orbmat_name']),
            ('RF Frequency', f"{setup['rf_frequency']/1e6:.6f} MHz"),
            ('Measured tune x', f"{setup['tunex']:.6f}"),
            ('Measured tune y', f"{setup['tuney']:.6f}"),
            ('SOFB buffer average', f"{setup['sofb_nr_points']:d}"),
            )
        for row in data:
            self.set_x(self.WIDTH/6)
            for datum in row:
                self.cell(
                    table_cell_width, table_cell_height,
                    str(datum), align='C', border=0)
            self.ln(table_cell_height)

    def config_table(self):
        """."""
        self.set_font('Arial', 'B', 10)
        table_cell_width = self.WIDTH/3.8
        table_cell_height = 5

        # self.ln(table_cell_height*5)
        columns = ('Config. Property', 'Value')
        self.set_x(self.WIDTH/3.8)
        for col in columns:
            self.cell(
                table_cell_width, table_cell_height,
                col, align='C', border=1)
        self.ln(table_cell_height)

        config = self.loco_data['config']
        data = (
            ('Tracking dimension', config.dim),
            ('Include dispersion', config.use_dispersion),
            ('Include diagonal blocks', config.use_diagonal),
            ('Include off-diagonal blocks', config.use_offdiagonal),
            ('Minimization method', config.min_method_str),
            ('Lambda LM',  f'{config.lambda_lm:.2e}'),
            ('Fixed lambda LM', f'{config.fixed_lambda:.2e}'),
            ('Jacobian manipulation', config.inv_method_str),
            ('Constraint delta KL total', config.constraint_deltak_total),
            ('Constraint delta KL step', config.constraint_deltak_step),
            ('Singular values method', config.svd_method_str),
            ('SV to be used:', config.svd_sel),
            ('SV threshold (s/smax):', config.svd_thre),
            ('Tolerance delta', config.tolerance_delta),
            ('Tolerance overfit', config.tolerance_overfit),
            ('Dipoles normal gradients', config.fit_dipoles),
            ('Quadrupoles normal gradients', config.fit_quadrupoles),
            ('Sextupoles normal gradients', config.fit_sextupoles),
            ('Use dipoles as families', config.use_dip_families),
            ('Use quadrupoles as families', config.use_quad_families),
            ('Dipoles skew gradients', config.fit_dipoles_coupling),
            ('Quadrupoles skew gradients', config.fit_quadrupoles_coupling),
            ('Sextupoles skew gradients', config.fit_sextupoles_coupling),
            ('Skew quadrupoles skew gradients', config.fit_skew_quadrupoles),
            ('Girders longitudinal shifts', config.fit_girder_shift),
            ('Fit BPM gains', config.fit_gain_bpm),
            ('Fit Corrector gains', config.fit_gain_corr),
            ('Fit BPM roll error', config.fit_roll_bpm),
            ('Horizontal delta kicks',
                str(config.delta_kickx_meas*1e6) + ' urad'),
            ('Vertical delta kicks',
                str(config.delta_kicky_meas*1e6) + ' urad'),
            ('RF delta frequency', str(config.delta_frequency_meas) + ' Hz'),
        )

        self.set_font('Arial', '', 10)
        for row in data:
            self.set_x(self.WIDTH/3.8)
            for datum in row:
                self.cell(
                    table_cell_width, table_cell_height,
                    str(datum), align='C', border=1)
            self.ln(table_cell_height)

    def add_fingerprint(self):
        """."""
        self.add_page()
        self.page_title('LOCO fingerprint')
        self.loco_fingerprint()
        self.page_title('LOCO configuration setup', loc_y=70)
        self.set_y(80)
        self.config_table()

    # def add_config(self):
    #     """."""
    #     self.add_page()
    #     self.page_title('LOCO configuration setup')
    #     self.config_table()

    def add_images_to_page(self, images):
        """."""
        self.add_page()
        self.page_body(images)

    def add_quadfit(self, df_stats):
        """."""
        self.add_page()
        self.page_title('Normal quadrupoles variations')
        self.quadrupole_fit()
        self.df_to_table(df_stats)

    def add_skewquadfit(self):
        """."""
        self.add_page()
        self.page_title('Skew quadrupoles variations')
        self.image('skewquad_by_s.png', 10, 120, self.WIDTH - 30)

    def df_to_table(self, df):
        """."""
        table_cell_width = 30
        table_cell_height = 5

        self.ln(table_cell_height)
        self.set_font('Arial', 'B', 10)
        self.set_x(self.WIDTH/4)

        # Loop over to print column names
        cols = df.columns
        for col in cols:
            self.cell(
                table_cell_width, table_cell_height, col, align='C', border=1)
        cols = [c.replace(' ', '_') for c in cols]
        cols = [c.replace('[%]', '') for c in cols]
        df.columns = cols
        # Line break
        self.ln(table_cell_height)
        # Loop over to print each data in the table
        for row in df.itertuples():
            self.set_font('Arial', '', 10)
            self.set_x(self.WIDTH/4)
            for col in cols:
                value = str(getattr(row, col))
                self.cell(
                    table_cell_width, table_cell_height, value,
                    align='C', border=1)
            self.ln(table_cell_height)


class LOCOAnalysis():
    """."""

    def __init__(self, fname_setup, fname_fit):
        """."""
        self.fname_setup = fname_setup
        self.fname_fit = fname_fit
        self.loco_setup = None
        self.loco_fit = None

    def get_setup(self):
        """."""
        loco_setup = load_pickle(self.fname_setup)
        if 'data' in loco_setup:
            loco_setup = loco_setup['data']
        self.loco_setup = loco_setup

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
        return simod, disp_nominal

    def get_loco_results(self):
        """."""
        loco_data = load_pickle(self.fname_fit)

        config = loco_data['config']
        model_fitting = loco_data['fit_model']
        gain_bpm = loco_data['gain_bpm']
        gain_corr = loco_data['gain_corr']
        roll_bpm = loco_data['roll_bpm']
        # energy_shift = loco_data['energy_shift']
        # chi_history = loco_data['chi_history']

        self.loco_fit = loco_data

        matrix_fitting = OrbRespmat(model_fitting, 'SI', '6d').get_respm()
        matrix_fitting = LOCOUtils.apply_all_gain(
            matrix_fitting, gain_bpm, roll_bpm, gain_corr)

        alpha_fit = pyaccel.optics.get_mcf(model_fitting)
        # print('momentum compaction: {:e}'.format(alpha_fit))
        idx = pyaccel.lattice.find_indices(
            model_fitting, 'pass_method', 'cavity_pass')[0]
        rf_freq_mod = model_fitting[idx].frequency
        rf_freq = self.loco_setup['rf_frequency']

        rfline_fit = matrix_fitting[:, -1]
        disp_fit = self._get_dispersion(rfline_fit, alpha_fit, rf_freq_mod)
        rfline_meas = config.goalmat[:, -1]
        disp_meas = self._get_dispersion(rfline_meas, alpha_fit, rf_freq)
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
        kl_strength = np.array(kl_strength)
        ksl_strength = np.array(ksl_strength)
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
        # fig.tight_layout()
        # plt.show()
        if save:
            fig.savefig(fname + '.png', dpi=300, format='png')
        # plt.savefig('loco_histogram2.png', format='png', dpi=300)

    @staticmethod
    def plot_quadrupoles_gradients_by_family(
            nom_model, fit_model, save=False, fname=None):
        """."""
        fig = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
    #     gs.update(left=0.10, right=0.85, hspace=0, wspace=0.25)
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
            # print(
            #     '{:5s}: mean {:.4f}, std {:.4f}, maxmin {:.4f}'.format(
            #         famlist[q], np.mean(perc), np.std(perc), np.ptp(perc)))
        stats = {
            'family name': famlist,
            'average [%]': kfit_mean,
            'std [%]': kfit_std,
            'peak to peak [%]': maxmin,
        }
        df_stats = pd.DataFrame.from_dict(stats).round(4)
        ax1.set_xlabel('quadrupole index')
        ax1.set_ylabel('quadrupole variation [%]')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(alpha=0.5, linestyle='--')
        # plt.show()
        # plt.tight_layout(True)
        plt.tight_layout()
        if save:
            if fname is None:
                fig.savefig(
                    'quadrupoles_gradients.png', format='png', dpi=1200)
            else:
                fig.savefig(fname+'.png', format='png', dpi=300)
        return df_stats

    @staticmethod
    def plot_quadrupoles_gradients_qn_order(
            nom_model, fit_model, save=False, fname=None):
        """."""
        fig = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        # gs.update(left=0.10, right=0.85, hspace=0, wspace=0.25)
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
        # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
        plt.tight_layout()
        # plt.show()
        if save:
            if fname is None:
                fig.savefig('quadrupoles_gradients.png', format='png', dpi=600)
            else:
                fig.savefig(fname+'.png', format='png', dpi=300)
        return kfit, knom, perc

    def plot_skew_quadrupoles(
            config, nom_model, fit_model, save=False, fname=None):
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

    # ============= private methods =============

    @staticmethod
    def _get_dispersion(rfline, alpha, rf_freq):
        return - alpha * rf_freq * rfline * 1e2


def run():
    """."""
    folder = '2022-05-24-SI_LOCO/'
    fname_setup = folder + 'respmat_endofmay22_bpms_03sector_switching_issue'
    fname_fit = folder + 'fitting_endofmay22_bpms_03sector_switching_issue'
    loco_anly = LOCOAnalysis(fname_setup=fname_setup, fname_fit=fname_fit)
    loco_anly.get_setup()
    mod, disp_nom = loco_anly.get_nominal_model()
    loco_data, orm_fit, disp_fit, disp_meas = loco_anly.get_loco_results()
    loco_data['setup'] = loco_anly.loco_setup
    config = loco_data['config']
    dnomi = config.matrix - config.goalmat
    dloco = orm_fit - config.goalmat
    loco_anly.plot_histogram(dnomi, dloco, save=True, fname='test1')
    df_stats = loco_anly.plot_quadrupoles_gradients_by_family(
        mod, loco_data['fit_model'], save=True, fname='quad_by_family')
    loco_anly.plot_quadrupoles_gradients_qn_order(
        mod, loco_data['fit_model'], save=True, fname='quad_by_s')
    loco_anly.plot_skew_quadrupoles(
        mod, loco_data['fit_model'], save=True, fname='skewquad_by_s')
    # images = [['test1.png'], ['test2.png'], ['test3.png']]
    # images = [['test1.png'], 'test2.png', 'test3.png']
    pdf = LOCOReport(loco_data)
    pdf.add_fingerprint()
    # pdf.add_config()
    # pdf.add_images_to_page([images[1], images[2]])
    pdf.add_quadfit(df_stats)
    pdf.add_skewquadfit()
    # pdf.add_images_to_page(images[2])
    pdf.output('test.pdf', 'F')


if __name__ == "__main__":
    run()
