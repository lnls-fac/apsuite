"""."""

import matplotlib.cm as cm
import matplotlib.gridspec as mpl_gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaccel
from mathphys.functions import load as _load, save as _save
from matplotlib import rc
from pymodels import si

from apsuite.loco.utils import LOCOUtils
from apsuite.optics_analysis.tune_correction import TuneCorr
from apsuite.orbcorr import OrbRespmat

rc("font", **{"size": 14})

# difference between sector 20 B1 end and model start marker
SECTOR_SHIFT = -5.017  # [m]
DEFAULT_FIG_DPI = 100


class LOCOAnalysis:
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
        self.edteng_nom = None
        self.edteng_fit = None
        self.disp_meas = None
        self.famdata = None

    def get_setup(self):
        """."""
        loco_setup = _load(self.fname_setup)
        if "data" in loco_setup:
            loco_setup = loco_setup["data"]
        self.loco_setup = loco_setup

    def calc_twiss(self):
        """."""
        self.twi_fit, *_ = pyaccel.optics.calc_twiss(
            self.loco_fit["fit_model"], indices="open"
        )

        if self.nom_model is None:
            self.nom_model, _ = self.get_nominal_model()

        self.twi_nom, *_ = pyaccel.optics.calc_twiss(
            self.nom_model, indices="open"
        )

    def calc_edteng(self):
        """."""
        self.edteng_fit, *_ = pyaccel.optics.calc_edwards_teng(
            self.loco_fit["fit_model"], indices="open"
        )

        if self.nom_model is None:
            self.nom_model, _ = self.get_nominal_model()

        self.edteng_nom, *_ = pyaccel.optics.calc_edwards_teng(
            self.nom_model, indices="open"
        )

    def get_nominal_model(self):
        """."""
        # Get nominal model
        simod = si.create_accelerator()
        simod.cavity_on = True
        simod.radiation_on = True

        # Adjust tunes to match measured ones
        tunex_goal = 49 + self.loco_setup["tunex"]
        tuney_goal = 14 + self.loco_setup["tuney"]

        # print('--- correcting si tunes...')
        tunecorr = TuneCorr(
            simod, "SI", method="Proportional", grouping="TwoKnobs"
        )
        tunecorr.get_tunes(simod)
        # print('    tunes init  : ', tunecorr.get_tunes(simod))
        tunemat = tunecorr.calc_jacobian_matrix()
        tunecorr.correct_parameters(
            model=simod,
            goal_parameters=np.array([tunex_goal, tuney_goal]),
            jacobian_matrix=tunemat,
        )
        # print('    tunes final : ', tunecorr.get_tunes(simod))

        # Get nominal orbit matrix and dispersion
        matrix_nominal = OrbRespmat(simod, "SI", True).get_respm()

        alpha0 = pyaccel.optics.get_mcf(simod)
        idx = pyaccel.lattice.find_indices(
            simod, "pass_method", "cavity_pass"
        )[0]
        # rf_freq = setup['rf_frequency']
        rf_freq = simod[idx].frequency

        rfline = matrix_nominal[:, -1]
        disp_nominal = self._get_dispersion(rfline, alpha0, rf_freq)
        self.nom_model = simod

        self.famdata = si.get_family_data(simod)
        return simod, disp_nominal

    def get_loco_results(self):
        """."""
        loco_data = _load(self.fname_fit)

        config = loco_data["config"]
        model_fitting = loco_data["fit_model"]
        gain_bpm = loco_data["gain_bpm"]
        gain_corr = loco_data["gain_corr"]
        roll_bpm = loco_data["roll_bpm"]

        self.loco_fit = loco_data

        matrix_fitting = OrbRespmat(model_fitting, "SI", True).get_respm()
        matrix_fitting = LOCOUtils.apply_all_gain(
            matrix_fitting, gain_bpm, roll_bpm, gain_corr
        )

        alpha_fit = pyaccel.optics.get_mcf(model_fitting)
        idx = pyaccel.lattice.find_indices(
            model_fitting, "pass_method", "cavity_pass"
        )[0]
        rf_freq_mod = model_fitting[idx].frequency
        rf_freq = self.loco_setup["rf_frequency"]

        rfline_fit = matrix_fitting[:, -1]
        disp_fit = self._get_dispersion(rfline_fit, alpha_fit, rf_freq_mod)
        rfline_meas = config.goalmat[:, -1]
        disp_meas = self._get_dispersion(rfline_meas, alpha_fit, rf_freq)
        self.disp_meas = disp_meas
        return loco_data, matrix_fitting, disp_fit, disp_meas

    # ============= static methods =============

    def get_famidx_quad(self):
        """."""
        famidx = []
        famlist = [
            "QFA",
            "QDA",
            "QFB",
            "QDB1",
            "QDB2",
            "QFP",
            "QDP1",
            "QDP2",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ]
        for fam_name in famlist:
            famidx.append(self.famdata[fam_name]["index"])
        return famidx

    def get_famidx_sext(self):
        """."""
        famidx = []
        for fam_name in si.families.families_sextupoles:
            famidx.append(self.famdata[fam_name]["index"])
        return famidx

    def get_attribute_quad(self, model):
        """."""
        kl_strength = []
        ksl_strength = []
        famidx = self.get_famidx_quad()
        for q in famidx:
            kl_strength.append(pyaccel.lattice.get_attribute(model, "KL", q))
            ksl_strength.append(pyaccel.lattice.get_attribute(model, "KsL", q))
        kl_strength = np.array(kl_strength, dtype=list)
        ksl_strength = np.array(ksl_strength, dtype=list)
        return kl_strength, ksl_strength

    def get_attribute_sext(self, model):
        """."""
        kl_strength = []
        sl_strength = []
        ksl_strength = []
        famidx = self.get_famidx_sext()
        for q in famidx:
            kl_strength.append(pyaccel.lattice.get_attribute(model, "KL", q))
            sl_strength.append(pyaccel.lattice.get_attribute(model, "SL", q))
            ksl_strength.append(pyaccel.lattice.get_attribute(model, "KsL", q))
        kl_strength = np.array(kl_strength)
        sl_strength = np.array(sl_strength)
        ksl_strength = np.array(ksl_strength)
        return kl_strength, sl_strength, ksl_strength

    @staticmethod
    def plot_histogram(
        diff_nominal,
        diff_loco,
        kickx=15,
        kicky=15 * 1.5,
        kickrf=15 * 5 * 1e6,
        fname=None,
    ):
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

        rmsnomyx = np.sqrt(np.mean(dnom[nbpm:, :nch] ** 2))
        rmsnomxy = np.sqrt(np.mean(dnom[:nbpm, nch:-1] ** 2))
        rmsnomxx = np.sqrt(np.mean(dnom[:nbpm, :nch] ** 2))
        rmsnomyy = np.sqrt(np.mean(dnom[nbpm:, nch:-1] ** 2))
        rmslocyx = np.sqrt(np.mean(dloc[nbpm:, :nch] ** 2))
        rmslocxy = np.sqrt(np.mean(dloc[:nbpm, nch:-1] ** 2))
        rmslocxx = np.sqrt(np.mean(dloc[:nbpm, :nch] ** 2))
        rmslocyy = np.sqrt(np.mean(dloc[nbpm:, nch:-1] ** 2))

        bins = 200
        ayx.hist(
            dnom[nbpm:, :nch].flatten(),
            bins=bins,
            label=r"nom. $\sigma_{{yx}} = {:.2f}\mu$m".format(rmsnomyx),
            density=True,
        )
        ayx.hist(
            dloc[nbpm:, :nch].flatten(),
            bins=bins,
            label=r"fit. $\sigma_{{yx}} = {:.2f}\mu$m".format(rmslocyx),
            density=True,
            alpha=0.7,
        )
        ayx.set_xlabel(r"$\Delta y$ [$\mu$m]")
        ayx.set_ylabel(r"$M_{yx}$")

        axy.hist(
            dnom[:nbpm, nch:-1].flatten(),
            bins=bins,
            label=r"nom. $\sigma_{{xy}} = {:.2f}\mu$m".format(rmsnomxy),
            density=True,
        )
        axy.hist(
            dloc[:160, nch:-1].flatten(),
            bins=bins,
            label=r"fit. $\sigma_{{xy}} = {:.2f}\mu$m".format(rmslocxy),
            density=True,
            alpha=0.7,
        )
        axy.set_xlabel(r"$\Delta x$ [$\mu$m]")
        axy.set_ylabel(r"$M_{xy}$")

        axx.hist(
            dnom[:nbpm, :nch].flatten(),
            bins=bins,
            label=r"nom. $\sigma_{{xx}} = {:.2f}\mu$m".format(rmsnomxx),
            density=True,
        )
        axx.hist(
            dloc[:nbpm, :nch].flatten(),
            bins=bins,
            label=r"fit. $\sigma_{{xx}} = {:.2f}\mu$m".format(rmslocxx),
            density=True,
            alpha=0.7,
        )
        axx.set_xlabel(r"$\Delta x$ [$\mu$m]")
        axx.set_ylabel(r"$M_{xx}$")

        ayy.hist(
            dnom[nbpm:, nch:-1].flatten(),
            bins=bins,
            label=r"nom. $\sigma_{{yy}} = {:.2f}\mu$m".format(rmsnomyy),
            density=True,
        )
        ayy.hist(
            dloc[nbpm:, nch:-1].flatten(),
            bins=bins,
            label=r"fit. $\sigma_{{yy}} = {:.2f}\mu$m".format(rmslocyy),
            density=True,
            alpha=0.7,
        )
        ayy.set_xlabel(r"$\Delta y$ [$\mu$m]")
        ayy.set_ylabel(r"$M_{yy}$")

        axx.legend(loc="upper right", fontsize=11)
        axy.legend(loc="upper right", fontsize=11)
        ayx.legend(loc="upper right", fontsize=11)
        ayy.legend(loc="upper right", fontsize=11)
        if fname:
            fig.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)

    def plot_3d_fitting(self, diff1, diff2, fname):
        """."""
        nbpm, ncorr = 2 * 160, 120 + 160 + 1
        idxbpm = np.linspace(0, nbpm - 1, nbpm)
        idxcorr = np.linspace(0, ncorr - 1, ncorr)
        corrs, bpms = np.meshgrid(idxcorr, idxbpm)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")

        diff1[:, :120] *= 15e-6
        diff1[:, 120:280] *= 1.5 * 15e-6
        diff1[:, -1] *= 5 * 15
        ax1.plot_surface(
            bpms,
            corrs,
            np.abs(diff1) * 1e6,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        diff1[:, :120] /= 15e-6
        diff1[:, 120:280] /= 1.5 * 15e-6
        diff1[:, -1] /= 5 * 15
        ax1.set_xlabel("BPM index", fontsize=10, labelpad=15)
        ax1.set_ylabel("Corr. index", fontsize=10, labelpad=15)
        ax1.set_zlabel(r"$|\chi|$ [$\mu$m]", labelpad=15)
        ax1.set_title("Measured - Nominal")

        ax2 = fig.add_subplot(122, projection="3d")
        diff2[:, :120] *= 15e-6
        diff2[:, 120:280] *= 1.5 * 15e-6
        diff2[:, -1] *= 5 * 15
        ax2.plot_surface(
            bpms,
            corrs,
            np.abs(diff2) * 1e6,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        diff2[:, :120] /= 15e-6
        diff2[:, 120:280] /= 1.5 * 15e-6
        diff2[:, -1] /= 5 * 15

        ax2.set_xlabel("BPM index", fontsize=10, labelpad=15)
        ax2.set_ylabel("Corr. index", fontsize=10, labelpad=15)
        ax2.set_zlabel(r"$|\chi|$ [$\mu$m]", labelpad=15)
        ax2.set_title("Measured - LOCO Fit")
        plt.tight_layout()
        if fname:
            fig.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)

    def plot_quadrupoles_gradients_by_family(
        self, nom_model, fit_model, fname=None
    ):
        """."""
        fig = plt.figure(figsize=(12, 5))
        gs = mpl_gs.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        kfit_mean = []
        knom_mean = []
        kfit_std = []
        maxmin = []
        kfit, *_ = self.get_attribute_quad(fit_model)
        knom, *_ = self.get_attribute_quad(nom_model)
        count = 0
        famlist = [
            "QFA",
            "QDA",
            "QFB",
            "QDB1",
            "QDB2",
            "QFP",
            "QDP1",
            "QDP2",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ]
        for q, _ in enumerate(famlist):
            clr = plt.cm.jet(q / len(famlist))
            npoints = len(kfit[q])
            kf = np.array(kfit[q])
            kn = np.array(knom[q])
            init = count
            final = init + npoints - 1
            span = np.linspace(init, final, npoints)
            perc = (kf - kn) / kn * 100
            ax1.plot(span, perc, ".-", label=famlist[q], color=clr)
            ax1.plot(span, np.repeat(np.mean(perc), npoints), "-", color=clr)
            ax1.plot(
                span,
                np.repeat(np.mean(perc) + np.std(perc), npoints),
                "--",
                color=clr,
            )
            ax1.plot(
                span,
                np.repeat(np.mean(perc) - np.std(perc), npoints),
                "--",
                color=clr,
            )
            count += len(kfit[q])
            kfit_mean.append(np.mean(perc))
            kfit_std.append(np.std(perc))
            knom_mean.append(np.mean(perc))
            maxmin.append(np.ptp(perc))
        stats = {
            "family name": famlist,
            "avg [%]": kfit_mean,
            "std [%]": kfit_std,
            "p2p [%]": maxmin,
        }
        df_stats = pd.DataFrame.from_dict(stats).round(4)
        ax1.set_xlabel("quadrupole index")
        ax1.set_ylabel("$\Delta K/K_0$ [%]")
        ax1.set_title("Quadrupoles changes grouped by family")
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1.0))
        ax1.grid(alpha=0.5, linestyle="--")
        plt.tight_layout()
        if fname:
            fig.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)
        return df_stats

    def save_quadrupoles_variations(
        self, nom_model, fit_model, fname_family, fname_trims
    ):
        """."""
        fam = self.famdata
        qn = np.array(fam["QN"]["index"]).ravel()
        kl = np.array(
            pyaccel.lattice.get_attribute(fit_model, "KL", indices=qn)
        ).flatten()
        kl_nom = np.array(
            pyaccel.lattice.get_attribute(nom_model, "KL", indices=qn)
        ).ravel()
        dkl = kl - kl_nom

        quad_families = si.families.families_quadrupoles()
        quadfam_idx = dict()
        qn = fam["QN"]["index"]
        for famname in quad_families:
            qfam = fam[famname]["index"]
            quadfam_idx[famname] = [qn.index(qidx) for qidx in qfam]

        quadfam_averages = dict()
        for famname in quad_families:
            quadfam_averages[famname] = np.mean(dkl[quadfam_idx[famname]])

        _save(data=quadfam_averages, fname=fname_family, overwrite=False)

        dkl_no_average = dkl
        for qnames in quad_families:
            idx_list = quadfam_idx[qnames]
            dkl_no_average[idx_list] -= quadfam_averages[qnames]

        np.savetxt(fname_trims + ".txt", dkl_no_average)

    def save_skew_quadrupoles_variations(self, nom_model, fit_model, fname=""):
        """."""
        fam = self.famdata
        qs = np.array(fam["QS"]["index"]).flatten()
        ksl_fit = np.array(
            pyaccel.lattice.get_attribute(fit_model, "KsL", indices=qs)
        ).flatten()
        ksl_nom = np.array(
            pyaccel.lattice.get_attribute(nom_model, "KsL", indices=qs)
        ).flatten()
        dksl = ksl_fit - ksl_nom

        np.savetxt(fname + ".txt", dksl)

    def plot_quadrupoles_gradients_by_s(
        self, nom_model, fit_model, fname=None
    ):
        """."""
        fig = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        spos = pyaccel.lattice.find_spos(nom_model)
        fam_nom = self.famdata
        qnlist_nom = fam_nom["QN"]["index"]
        qnidx = np.array(qnlist_nom).flatten()
        knom = np.array(
            pyaccel.lattice.get_attribute(nom_model, "KL", qnlist_nom)
        )

        fam_fit = si.get_family_data(fit_model)
        qnlist_fit = fam_fit["QN"]["index"]
        kfit = np.array(
            pyaccel.lattice.get_attribute(fit_model, "KL", qnlist_fit)
        )
        perc = (kfit - knom) / knom * 100
        ax1.plot(
            spos[qnidx], perc, ".-", label="deviation", color="tab:orange"
        )

        xdelta = spos[-1] / 20
        ax1 = self._create_sectors_vlines(
            ax1, xdelta=xdelta, yloc=perc.max() * 0.9
        )
        ax1.set_xlabel("s [m]")
        ax1.set_ylabel("$\Delta K/K_0$ [%]")
        ax1.set_title("Quadrupoles changes along the ring")
        plt.tight_layout()
        if fname:
            fig.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)
        return kfit, knom, perc

    def plot_skew_quadrupoles(self, nom_model, fit_model, fname=None):
        """."""
        _ = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])

        spos = pyaccel.lattice.find_spos(nom_model)
        fam_nom = self.famdata
        qslist_nom = fam_nom["QS"]["index"]
        qsidx = np.array(qslist_nom).flatten()
        knom = np.array(
            pyaccel.lattice.get_attribute(nom_model, "KsL", qslist_nom)
        )

        fam_fit = si.get_family_data(fit_model)
        qslist_fit = fam_fit["QS"]["index"]
        kfit = np.array(
            pyaccel.lattice.get_attribute(fit_model, "KsL", qslist_fit)
        )
        if sum(knom):
            percentage = (kfit - knom) / knom * 100
        else:
            percentage = kfit
        ax1.plot(spos[qsidx], percentage, ".-", color="tab:green")

        xdelta = spos[-1] / 20
        ax1 = self._create_sectors_vlines(
            ax1, xdelta=xdelta, yloc=percentage.max() * 0.9
        )
        ax1.set_xlabel("s [m]")
        ax1.set_ylabel("$\Delta$KsL [1/m]")
        ax1.set_title("Skew quadrupoles changes along the ring")
        plt.tight_layout()
        if fname:
            plt.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)
        return kfit, knom, percentage

    def plot_gain(self, fname=None):
        """."""
        _, axs = plt.subplots(3, 1, figsize=(14, 14))

        if self.nom_model is None:
            self.nom_model, _ = self.get_nominal_model()

        spos = pyaccel.lattice.find_spos(self.nom_model)
        fam = self.famdata
        bpm_idx = np.array(fam["BPM"]["index"]).ravel()
        ch_idx = np.array(fam["CH"]["index"]).ravel()
        cv_idx = np.array(fam["CV"]["index"]).ravel()

        gain_bpm = self.loco_fit["gain_bpm"]
        gain_corr = self.loco_fit["gain_corr"]
        roll_bpm = self.loco_fit["roll_bpm"]

        color_ch = "tab:blue"
        color_cv = "tab:red"
        color_gainx_bpm = "tab:blue"
        color_gainy_bpm = "tab:red"
        color_roll_bpm = "tab:green"
        axs[0].plot(
            spos[bpm_idx],
            gain_bpm[:160],
            ".-",
            color=color_gainx_bpm,
            alpha=0.5,
            label="x",
        )
        axs[0].plot(
            spos[bpm_idx],
            gain_bpm[:160],
            ".-",
            color=color_gainy_bpm,
            alpha=0.5,
            label="y",
        )
        axs[0].legend(loc="lower right")
        axs[0].grid(alpha=0.5, linestyle="--")
        axs[0].set_ylabel("gain")
        axs[0].set_title("BPM Gains")

        axs[1].plot(spos[bpm_idx], roll_bpm * 1e3, ".-", color=color_roll_bpm)
        axs[1].grid(alpha=0.5, linestyle="--")
        axs[1].set_xlabel("index")
        axs[1].set_ylabel("roll [mrad]")
        axs[1].set_title("BPM Roll")
        xdelta = spos[-1] / 20
        axs[1] = self._create_sectors_vlines(
            axs[1], xdelta=xdelta, yloc=roll_bpm.max() * 0.95 * 1e3
        )

        axs[2].plot(
            spos[ch_idx], gain_corr[:120], ".-", color=color_ch, label="CH"
        )
        axs[2].plot(
            spos[cv_idx], gain_corr[120:], ".-", color=color_cv, label="CV"
        )
        axs[2].legend(loc="lower right")
        axs[2].grid(alpha=0.5, linestyle="--")
        axs[2].set_xlabel("s [m]")
        axs[2].set_ylabel("gain")
        axs[2].set_title("Corrector Gains")
        xdelta = spos[-1] / 20
        axs[2] = self._create_sectors_vlines(
            axs[2], xdelta=xdelta, yloc=gain_corr.max()
        )
        plt.tight_layout()
        if fname:
            plt.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)

    def beta_and_tune(self, twiss=False, fname=None):
        """."""
        names = ["betax", "betay", "mux", "muy"]
        nom, fit = self.twi_nom, self.twi_fit
        if not twiss:
            names = ["beta1", "beta2", "mu1", "mu2"]
            nom, fit = self.edteng_nom, self.edteng_fit

        if nom.spos.size != fit.spos.size:
            if fit.spos.size > nom.spos.size:
                spos = fit.spos
                betax_nom = np.interp(spos, nom.spos, getattr(nom, names[0]))
                betay_nom = np.interp(spos, nom.spos, getattr(nom, names[1]))
                betax_fit = getattr(fit, names[0])
                betay_fit = getattr(fit, names[1])
            else:
                spos = nom.spos
                betax_fit = np.interp(spos, fit.spos, getattr(fit, names[0]))
                betay_fit = np.interp(spos, fit.spos, getattr(fit, names[1]))
                betax_nom = getattr(nom, names[0])
                betay_nom = getattr(nom, names[1])
        else:
            spos = nom.spos
            betax_nom = getattr(nom, names[0])
            betay_nom = getattr(nom, names[1])
            betax_fit = getattr(fit, names[0])
            betay_fit = getattr(fit, names[1])

        beta_beatx = (betax_fit - betax_nom) / betax_nom * 100
        beta_beaty = (betay_fit - betay_nom) / betay_nom * 100

        _ = plt.figure(figsize=(12, 4))
        gs = mpl_gs.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

        spos = nom.spos
        ax.plot(spos, beta_beatx, label="Horizontal", color="tab:blue")
        ax.plot(spos, beta_beaty, label="Vertical", color="tab:red")

        xdelta = spos[-1] / 20
        max_val = np.max([beta_beatx.max(), beta_beaty.max()])
        ax = self._create_sectors_vlines(ax, xdelta=xdelta, yloc=max_val * 0.9)
        ax.set_xlabel("s [m]")
        ax.set_ylabel(r"$\Delta \beta/\beta$ [%]")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.5, linestyle="--")
        plt.tight_layout()
        if fname:
            plt.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)

        tunex_meas = 49 + round(self.loco_setup["tunex"], 4)
        tuney_meas = 14 + round(self.loco_setup["tuney"], 4)

        tunex_nom = getattr(nom, names[2])[-1] / 2 / np.pi
        tuney_nom = getattr(nom, names[3])[-1] / 2 / np.pi

        tunex_fit = getattr(fit, names[2])[-1] / 2 / np.pi
        tuney_fit = getattr(fit, names[3])[-1] / 2 / np.pi

        diffx = tunex_meas - tunex_fit
        diffy = tuney_meas - tuney_fit

        tunex_list = [tunex_nom, tunex_meas, tunex_fit, diffx]
        tuney_list = [tuney_nom, tuney_meas, tuney_fit, diffy]

        names = [
            "initial nom. model",
            "measured values",
            "LOCO model",
            "meas. - LOCO",
        ]
        df_tunes = pd.DataFrame({
            "name": names,
            "tunex": tunex_list,
            "tuney": tuney_list,
        })

        beta_list = [beta_beatx, beta_beaty]
        avg_list = [float(np.mean(val)) for val in beta_list]
        std_list = [float(np.std(val)) for val in beta_list]
        ptp_list = [float(np.ptp(val)) for val in beta_list]

        stats = {
            "beta beating": ["x", "y"],
            "avg [%]": avg_list,
            "std [%]": std_list,
            "p2p [%]": ptp_list,
        }
        df_betabeat = pd.DataFrame.from_dict(stats)
        tmp = df_betabeat.select_dtypes(include=[np.number])
        df_betabeat.loc[:, tmp.columns] = np.round(tmp, 4)
        return df_tunes.round(4), df_betabeat

    def dispersion(self, disp_meas=None, twiss=False, fname=None):
        """."""
        _ = plt.figure(figsize=(14, 6))
        gs = mpl_gs.GridSpec(2, 1)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])

        bpm_idx_nom = np.array(
            pyaccel.lattice.find_indices(self.nom_model, "fam_name", "BPM")
        ).flatten()

        bpm_idx_fit = np.array(
            pyaccel.lattice.find_indices(
                self.loco_fit["fit_model"], "fam_name", "BPM"
            )
        ).flatten()

        names = ["etax", "etay"]
        nom, fit = self.twi_nom, self.twi_fit
        if not twiss:
            names = ["eta1", "eta2"]
            nom, fit = self.edteng_nom, self.edteng_fit

        disp0x = getattr(nom, names[0])[bpm_idx_nom] * 100
        disp0y = getattr(nom, names[1])[bpm_idx_nom] * 100

        dispx_fit = getattr(fit, names[0])[bpm_idx_fit] * 100
        dispy_fit = getattr(fit, names[1])[bpm_idx_fit] * 100

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
        avg_listx = [float(np.mean(val)) * 10 for val in errx_list]
        avg_listy = [float(np.mean(val)) * 10 for val in erry_list]
        std_listx = [float(np.std(val)) * 10 for val in errx_list]
        std_listy = [float(np.std(val)) * 10 for val in erry_list]
        ptp_listx = [float(np.ptp(val)) * 10 for val in errx_list]
        ptp_listy = [float(np.ptp(val)) * 10 for val in erry_list]

        stats = {
            "difference": ["meas - fit", "nom - fit", "nom - meas"],
            "avg x [mm]": avg_listx,
            "std x [mm]": std_listx,
            "p2p x [mm]": ptp_listx,
            "avg y [mm]": avg_listy,
            "std y [mm]": std_listy,
            "p2p y [mm]": ptp_listy,
        }
        df_disp = pd.DataFrame.from_dict(stats)
        tmp = df_disp.select_dtypes(include=[np.number])
        df_disp.loc[:, tmp.columns] = np.round(tmp, 4)
        spos = nom.spos[bpm_idx_nom]

        ax1.plot(spos, disp0x, ".-", label="nominal", linewidth=1)
        ax1.plot(spos, dispx_fit, ".-", label="fitting", linewidth=1)
        ax1.plot(spos, dispx_meas, ".-", label="meas", linewidth=1)
        ax1.plot(
            spos, dispx_fit - dispx_meas, ".-", label="error", linewidth=1
        )

        ax2.plot(spos, disp0y, ".-", label="nominal", linewidth=1)
        ax2.plot(spos, dispy_fit, ".-", label="fitting", linewidth=1)
        ax2.plot(spos, dispy_meas, ".-", label="meas", linewidth=1)
        ax2.plot(
            spos, dispy_fit - dispy_meas, ".-", label="error", linewidth=1
        )

        ax1.set_ylabel(r"$\eta_x$ [cm]")
        ax1.legend(bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=12)
        ax1.grid(alpha=0.5, linestyle="--")

        ax2.set_xlabel("s [m]")
        ax2.set_ylabel(r"$\eta_y$ [cm]")
        ax2.grid(alpha=0.5, linestyle="--")

        diff1 = abs(dispy_meas)
        diff2 = abs(dispy_fit - dispy_meas)
        max_val = np.max([diff1.max(), diff2.max()])
        xdelta = spos[-1] / 20
        self._create_sectors_vlines(ax1, xdelta=xdelta, annotate=False)
        self._create_sectors_vlines(ax2, xdelta=xdelta, yloc=max_val * 0.9)
        plt.tight_layout()
        if fname:
            plt.savefig(fname + ".png", format="png", dpi=DEFAULT_FIG_DPI)
        return df_disp

    def emittance_and_coupling(self):
        """."""
        eqnom = pyaccel.optics.EqParamsFromBeamEnvelope(self.nom_model)
        eqfit = pyaccel.optics.EqParamsFromBeamEnvelope(
            self.loco_fit["fit_model"]
        )

        if self.edteng_fit is None or self.edteng_nom is None:
            self.calc_edteng()

        min_sep_nom, *_ = pyaccel.optics.estimate_coupling_parameters(
            self.edteng_nom
        )
        min_sep_fit, *_ = pyaccel.optics.estimate_coupling_parameters(
            self.edteng_fit
        )

        m2pm = 1e12
        names = [
            "emit_x [pm.rad]",
            "emit_y [pm.rad]",
            "emit_ratio [%]",
            "min_tune_sep [%]",
        ]
        emit_nom_list = [
            eqnom.emit1 * m2pm,
            eqnom.emit2 * m2pm,
            eqnom.emit2 / eqnom.emit1 * 100,
            min_sep_nom * 100,
        ]
        emit_fit_list = [
            eqfit.emit1 * m2pm,
            eqfit.emit2 * m2pm,
            eqfit.emit2 / eqfit.emit1 * 100,
            min_sep_fit * 100,
        ]
        emit_nom_list = [float(abs(val)) for val in emit_nom_list]
        emit_fit_list = [float(abs(val)) for val in emit_fit_list]
        emits = {
            "parameter": names,
            "initial nom model": emit_nom_list,
            "LOCO model": emit_fit_list,
        }
        df_emits = pd.DataFrame.from_dict(emits)
        tmp = df_emits.select_dtypes(include=[np.number])
        df_emits.loc[:, tmp.columns] = np.round(tmp, 4)
        return df_emits

    # ============= private methods =============

    @staticmethod
    def _get_dispersion(rfline, alpha, rf_freq):
        return -alpha * rf_freq * rfline * 1e2

    @staticmethod
    def _create_sectors_vlines(ax, xdelta, yloc=0, annotate=True):
        for idx in range(20):
            ax.axvline(xdelta * idx + SECTOR_SHIFT, ls="--", color="k", lw=1)
            if annotate:
                ax.annotate(
                    f"{idx + 1:02d}",
                    size=10,
                    xy=(xdelta * (idx + 1 / 5), yloc),
                )
        return ax
