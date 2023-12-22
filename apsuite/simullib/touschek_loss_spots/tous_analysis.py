"""tous_analysis."""
from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute, find_indices, find_spos
import touschek_pack.functions as to_fu
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.cm as _cm
from mathphys.beam_optics import beam_rigidity as _beam_rigidity
import pandas as _pd
import scipy.integrate as scyint
import pyaccel as _pyaccel
from mathphys.functions import load_pickle


class TousAnalysis:
    """Class for the analysis of electron losses along the ring."""

    def __init__(
        self, accelerator, energies_off=None, beam_energy=None, n_turns=7
    ):
        """Parameters necessary to define the class."""
        if energies_off is None:
            energy_off = _np.linspace(0, 0.046, 460)  # physical limitants
            deltas = _np.linspace(0, 0.1, 400)  # energy off for tracking

        if beam_energy is None:  # defining beta factor
            beam_energy = _beam_rigidity(energy=3)[2]
            beta = beam_energy

        self._model_fit = accelerator
        self._model = pymodels.si.create_accelerator()

        self._amp_and_limidx = None
        self._sc_accps = None
        self._accep = None
        self._inds_pos = load_pickle(
            "/".join(__file__.split("/")[:-1]) + "/ph_lim.pickle"
        )
        self._inds_neg = load_pickle(
            "/".join(__file__.split("/")[:-1]) + "/ph_lim_neg.pickle"
        )
        self._amps_pos = None
        self._amps_neg = None
        self.num_part = 50000
        self.energy_dev_min = 1e-4

        self.beta = beta  # beta factor
        self.h_pos = get_attribute(self._model_fit, "hmax", indices="closed")
        self.h_neg = get_attribute(self._model_fit, "hmin", indices="closed")
        self.ltime = Lifetime(self._model_fit)
        self._off_energy = energy_off  # (linear model) en_dev to amplitudes
        self.nturns = n_turns
        self._deltas = deltas
        self.spos = find_spos(self._model_fit, indices="closed")
        self.scraph_inds = find_indices(self._model, "fam_name", "SHVC")
        self.scrapv_inds = find_indices(self._model, "fam_name", "SVVC")

    @property
    def accelerator(self):
        """."""
        return self._model_fit

    @accelerator.setter
    def accelerator(self, new_model):
        """."""
        self._model_fit = new_model

    @property
    def nom_model(self):
        """Defines nominal model.

        Some calculus involves nominal model without coupling and vertical
        dispersion corretion.
        """
        return self._model

    @property
    def accep(self):
        """Defines Touschek energy acceptance."""
        if self._accep is None:
            self._accep = py_op.calc_touschek_energy_acceptance(
                self.accelerator
            )
        return self._accep

    @property
    def s_calc(self):
        """Defines property.

        Defines s position and get the energy accpetance both at each 10
        meters.
        """
        if self._sc_accps is None:
            self._sc_accps = to_fu.get_scaccep(self.accelerator, self.accep)
        return self._sc_accps

    @property
    def amp_and_limidx(self):
        """Defines 4 properties.

        Positive and negative amplitudes.
        physical limitants for positive and negative e_dev
        """
        if self._amp_and_limidx is None:
            self._model.cavity_on = False
            self._model.radiation_on = False

            self._amps_pos, self._inds_pos = to_fu.calc_amp(
                self._model, self.off_energy, self.h_pos, self.h_neg
            )

            self._amps_neg, self._inds_neg = to_fu.calc_amp(
                self._model, -self.off_energy, self.h_pos, self.h_neg
            )

            self._amp_and_limidx = True

        return self._amp_and_limidx

    @property
    def off_energy(self):
        """."""
        return self._off_energy

    @off_energy.setter
    def off_energy(self, accep):
        """If necessary redefines the e_dev for tracking.

        The property defines the cutoof e_dev with tous_accep
        accep = new touschek acceptance.
        """
        accep_pos, accep_neg = accep
        accep_lim = _np.max(_np.maximum(accep_pos, _np.abs(accep_neg)))
        steps = int(accep_lim * 10000)  # choosen to be the number of steps
        self._off_energy = _np.linspace(0, accep_lim, steps)

    @property
    def deltas(self):
        """."""
        return self._deltas

    @deltas.setter
    def deltas(self, dev_percent, steps=400):
        """Redefines the e_dev for tracking.

        dev_percent = energy deviation in percents [%].
        """
        dev_percent /= 100
        self._deltas = _np.linspace(0, dev_percent, steps)

    # the four properties defining below are to be static
    @property
    def inds_pos(self):
        """."""
        return self._inds_pos

    @property
    def inds_neg(self):
        """."""
        return self._inds_neg

    @property
    def amp_pos(self):
        """."""
        return self._amps_pos

    @property  # negative amplitudes from linear model
    def amp_neg(self):
        """."""
        return self._amps_neg

    def get_amps_idxs(self):  # Defines various parameters
        """Defines 3 self params at same time."""
        return self.amp_and_limidx, self.accep, self.s_calc

    def set_vchamber_scraper(self, vchamber):
        """Function for setting the vchamber apperture."""
        model = self._model
        scph_inds = self.scraph_inds
        scpv_inds = self.scrapv_inds

        for iten in scph_inds:
            model[iten].hmin = vchamber[0]
            model[iten].hmax = vchamber[1]
        for iten in scpv_inds:
            model[iten].vmin = vchamber[2]
            model[iten].vmax = vchamber[3]

    def _single_pos_track(self, single_spos, par):
        """Single position tracking."""
        self._model.cavity_on = True
        self._model.radiation_on = True
        self._model.vchamber_on = True
        s = self.spos

        index = _np.argmin(_np.abs(s - single_spos))
        if "pos" in par:
            res = to_fu.track_eletrons_d(
                self.deltas,
                self.nturns,
                index,
                self._model,
                pos_x=1e-5,
                pos_y=3e-6,
            )
        elif "neg" in par:
            res = to_fu.track_eletrons_d(
                -self.deltas,
                self.nturns,
                index,
                self._model,
                pos_x=1e-5,
                pos_y=3e-6,
            )

        return res

    def _get_weighting_tous(self, single_spos, npt=5000):
        """."""
        scalc, daccp, daccn = to_fu.get_scaccep(self.accelerator, self.accep)
        bf = self.beta  # bf:beta factor
        lt = self.ltime
        b1 = lt.touschek_data["touschek_coeffs"]["b1"]
        b2 = lt.touschek_data["touschek_coeffs"]["b2"]

        taup, taun = (bf * daccp) ** 2, (bf * daccn) ** 2
        idx = _np.argmin(_np.abs(scalc - single_spos))
        taup_0, taun_0 = taup[idx], taun[idx]
        kappap_0 = _np.arctan(_np.sqrt(taup_0))
        kappan_0 = _np.arctan(_np.sqrt(taun_0))

        kappa_pos = _np.linspace(kappap_0, _np.pi / 2, npt)
        kappa_neg = _np.linspace(kappan_0, _np.pi / 2, npt)

        deltp = _np.tan(kappa_pos) / bf
        deltn = _np.tan(kappa_neg) / bf
        fdensp = to_fu.f_function_arg_mod(
            kappa_pos, kappap_0, b1[idx], b2[idx], norm=False
        )

        fdensn = to_fu.f_function_arg_mod(
            kappa_neg, kappan_0, b1[idx], b2[idx], norm=False
        )

        # eliminating negative values
        indp = _np.where(fdensp < 0)[0]
        indn = _np.where(fdensn < 0)[0]
        fdensp[indp] = 0
        fdensn[indn] = 0

        ind = _np.intp(_np.where(fdensp > 1e-2)[0])
        fdensp = fdensp[ind]
        deltp = deltp[ind]
        ind = _np.intp(_np.where(fdensn > 1e-2)[0])
        fdensn = fdensn[ind]
        deltn = deltn[ind]

        self.deltas = deltp[-1] * 1e2

        return fdensp, fdensn, deltp, deltn

    def _get_trackndens(self, single_spos, par):
        """Concatenates tracking and touschek loss dens."""
        # if len(to_fu.t_list(single_spos)) != 1: # Não sei se isso é útil
        #     raise ValueError('This function suports only one s position')

        fdensp, fdensn, deltp, deltn = self._get_weighting_tous(single_spos)

        fp = fdensp.squeeze()
        fn = fdensn.squeeze()

        dic = self._single_pos_track(single_spos, par)
        deltas = dic["energy_deviation"]
        delta_ = _np.abs(_np.diff(deltas)[0])

        if "pos" in par:
            return dic, fp * delta_, deltp * 1e2
        elif "neg" in par:
            return dic, fn * delta_, -deltn * 1e2

    # this function plot the graphic of tracking and the touschek scattering
    # distribution for one single position

    def plot_track_lossdens(self, single_spos, par, accep):
        """Plot the results of tracking and the tous. scat. loss density.

        single_spos =                            single possition.
        par         = defines the analysis (positive or negative).
        accep       =              touschek scattering acceptance.
        """
        dic, fp, dp = self._get_trackndens(single_spos, par)
        s = self.spos

        if "pos" in par:
            inds = _np.intp(self.inds_pos)
        elif "neg" in par:
            inds = _np.intp(self.inds_neg)
        index = _np.argmin(_np.abs(s - single_spos))

        to_fu.plot_track_d(
            self.accelerator,
            dic,
            inds,
            self.off_energy,
            par,
            index,
            accep,
            dp,
            fp,
        )

    def plot_normtousd(self, spos):
        """Touschek scattering loss density.

        spos = desired s positions (list or numpy.array)
        """
        spos_ring = self.spos
        dic = to_fu.norm_cutacp(
            self._model_fit, spos, 5000, self._accep, norm=True
        )

        fdensp, fdensn = dic["fdensp"], dic["fdensn"]
        deltasp, deltasn = dic["deltasp"], dic["deltasn"]

        _, ax = _plt.subplots(figsize=(10, 5))
        ax.set_title(
            "Probability density analytically calculated", fontsize=20
        )
        ax.grid(True, alpha=0.5, ls="--", color="k")
        ax.xaxis.grid(False)
        ax.set_xlabel(r"$\delta$ [%]", fontsize=25)
        ax.set_ylabel("PDF", fontsize=25)
        ax.tick_params(axis="both", labelsize=28)

        for idx, _ in enumerate(spos):
            array_fdens = fdensp[idx]
            # pega o primeiro item onde ocorre a condição passada
            # como argumento para a função numpy.where
            index = _np.intp(_np.where(array_fdens <= 1e-2)[0][1])

            # this block selects the best index
            # for plot the density distribution
            if not idx:
                best_index = index
            else:
                if best_index < index:  # eu não entendi direito porquê disso
                    best_index = index
                else:
                    pass

        for idx, s in enumerate(spos):
            mod_ind = _np.argmin(_np.abs(spos_ring - s))

            fdenspi = fdensp[idx][:best_index]
            fdensni = fdensn[idx][:best_index]
            deltaspi = deltasp[idx][:best_index] * 1e2
            deltasni = -deltasn[idx][:best_index] * 1e2

            color = _cm.gist_rainbow(idx / len(spos))
            not_desired = [
                "calc_mom_accep",
                "mia",
                "mib",
                "mip",
                "mb1",
                "mb2",
                "mc",
            ]

            while self._model_fit[mod_ind].fam_name in not_desired:
                mod_ind += 1

            fam_name = self._model_fit[mod_ind].fam_name
            s_stri = _np.round(spos_ring[mod_ind], 2)
            stri = f"{fam_name} em {s_stri} m"

            ax.plot(deltaspi, fdenspi, label=stri, color=color)
            ax.plot(deltasni, fdensni, color=color)

        ax.legend(loc="best", fontsize=20)

    def plot_histograms(self, l_spos):
        """Touschek scattering density from Monte-Carlo simulation.

        l_spos = desired s positions (list or array).
        """
        s = self.spos
        accep = self.accep
        model = self._model_fit

        tup = to_fu.histgms(
            self._model_fit,
            l_spos,
            self.num_part,
            accep,
            self.energy_dev_min,
            cutaccep=False,
        )

        hp, hn, idx_model = tup

        fig, ax = _plt.subplots(
            ncols=len(l_spos), nrows=1, figsize=(30, 10), sharey=True
        )
        fig.suptitle(
            "Probability density calculated by Monte-Carlo simulation",
            fontsize=20,
        )

        for index, iten in enumerate(idx_model):
            color = _cm.jet(index / len(idx_model))
            ay = ax[index]
            if not index:
                ay.set_ylabel("PDF", fontsize=25)

            ay.grid(True, alpha=0.5, ls="--", color="k")
            ay.xaxis.grid(False)
            ay.set_xlabel(r"$\delta$ [%]", fontsize=25)
            ay.tick_params(axis="both", labelsize=18)

            stri = f"{model[iten].fam_name:s}, {s[iten]:.2f}"
            ay.hist(hp[index], density=True, bins=200, color=color, label=stri)
            ay.hist(hn[index], density=True, bins=200, color=color)
            _plt.tight_layout()
            ay.legend()

    def _get_track_def(self, l_scattered_pos, scrap, vchamber):
        """Tracking for getting the loss profile along the ring.

        l_scattered_pos = scattered positions (list or numpy.array).
        scrap = if True, the vchamber's height will be changed.
        vchmaber = defines the new vchamber's apperture.
        """
        all_track = []
        indices = []
        spos = self.spos

        self._model.radiation_on = True
        self._model.cavity_on = True
        self._model.vchamber_on = True

        if scrap:
            self.set_vchamber_scraper(vchamber)

        for _, scattered_pos in enumerate(l_scattered_pos):
            index = _np.argmin(_np.abs(scattered_pos - spos))
            indices.append(index)
            dic = to_fu.track_eletrons_d(
                self._deltas, self.nturns, index, self._model
            )
            all_track.append(dic)

        hx = self._model_fit[self.scraph_inds[0]].hmax
        hn = self._model_fit[self.scraph_inds[0]].hmin
        vx = self._model_fit[self.scrapv_inds[0]].vmax
        vn = self._model_fit[self.scrapv_inds[0]].vmin
        vchamber = [hx, hn, vx, vn]

        self.set_vchamber_scraper(vchamber)

        return all_track, indices

    def _concat_track_lossrate(self, l_scattered_pos, scrap, vchamber):
        # não consegui resolvero erro que o ruff indicou nessa função
        """Generating the data for the plot."""
        all_track, indices = self._get_track_def(
            l_scattered_pos, scrap, vchamber
        )
        spos = self.spos
        fact = 0.03

        tous_rate = self.ltime.touschek_data["rate"]  # scattering rate
        prob, lostp, all_lostp = [], [], []

        # comentar os dois métodos implementados.
        # um deles parece ser mais rápido.
        # for j, single_track in enumerate(all_track):
        for j, dic in enumerate(all_track):
            index = indices[j]

            lostinds = dic["element_lost"]
            deltas = dic["energy_deviation"]

            # lostinds = _np.zeros(len(single_track))
            # deltas = _np.zeros(len(single_track))
            # for idx, iten in enumerate(single_track):
            #     _, ellost, delta = iten
            #     lostinds[idx] = ellost
            #     deltas[idx] = delta
            # lostinds = _np.intp(lostinds)

            lost_positions = _np.round(spos[lostinds], 2)

            step = int((deltas[0] + deltas[-1]) / fact)
            itv_track = _np.linspace(deltas[0], deltas[-1], step)

            data = _pd.DataFrame({"lost_pos_by_tracking": lost_positions})
            # dataframe that storages the tracking data
            lost_pos_column = (
                data.groupby("lost_pos_by_tracking").groups
            ).keys()
            data = _pd.DataFrame({"lost_pos_by_tracking": lost_pos_column})
            # this step agroups the lost_positions

            itv_delta = []
            for current, next_iten in zip(itv_track, itv_track[1:]):
                stri = f"{current*1e2:.2f} % < delta < {next_iten*1e2:.2f} %"
                data[stri] = _np.zeros(len(list(lost_pos_column)))  # this step
                # creates new columns in the dataframe and fill with zeros

                itv_delta.append((current, next_iten))
                # Next step must calculate each matrix element from the
                # dataframe

            var = list(data.index)
            if var == lost_pos_column:
                pass
            else:
                data = data.set_index("lost_pos_by_tracking")

            for idx, lost_pos in enumerate(lost_positions):  # essas duas
                # estruturas de repetição são responsáveis por calcular
                # o percentual dos eletrons que possuem um determinado desvio
                # de energia e se perdem em um intervalo de desvio de energia
                # específico
                delta = deltas[idx]
                # lps = []
                for i, interval in enumerate(itv_delta):
                    if not i:  # subtle difference: <= in first iteraction
                        if interval[0] <= delta <= interval[1]:
                            stri = f"{interval[0]*1e2:.2f} % < delta < {interval[1]*1e2:.2f} %"
                            data.loc[lost_pos, stri] += 1

                    else:
                        if interval[0] < delta <= interval[1]:
                            stri = f"{interval[0]*1e2:.2f} % < delta < {interval[1]*1e2:.2f} %"
                            data.loc[lost_pos, stri] += 1

            data = data / len(deltas)

            npt = int((spos[-1] - spos[0]) / 0.1)

            scalc = _np.linspace(spos[0], spos[-1], npt)
            rate_nom_lattice = _np.interp(spos, scalc, tous_rate)

            lost_pos_df = []
            part_prob = []
            # Calculates the loss probablity by tracking
            for indx, iten in data.iterrows():
                t_prob = 0
                for idx, m in enumerate(iten):
                    t_prob += m
                    if idx == iten.count() - 1:
                        # appends the probability after sum
                        part_prob.append(t_prob)
                        lost_pos_df.append(indx)

            lost_pos_df = _np.array(lost_pos_df)
            part_prob = _np.array(part_prob)

            # Calculates the absolute probability for electron loss
            # by touschek scattering rate
            prob.append(part_prob * rate_nom_lattice[index])
            lostp.append(lost_pos_df)

            # Aqui eu pego as posições em que os elétrons foram perdidos
            # e armazeno todas em uma grande lista sem repetição de qualquer
            # posição perdida
            if not j:
                all_lostp = lost_pos_df
            else:
                boolean_array = _np.isin(lost_pos_df, all_lostp)
                for ind, boolean_indc in enumerate(boolean_array):
                    if not boolean_indc:
                        all_lostp = _np.append(all_lostp, lost_pos_df[ind])

        return all_lostp, prob, lostp

    def _f_scat_table(self, l_scattered_pos, scrap, vchamber):
        """Generates the heat map of loss positions."""
        dic_res = {}
        all_lostp, prob, lostp = self._concat_track_lossrate(
            l_scattered_pos, scrap, vchamber
        )
        n_scat = _np.round(l_scattered_pos, 2)

        for idx, scattered_pos in enumerate(n_scat):
            scat_data = []
            bool_array = _np.isin(all_lostp, lostp[idx])

            for j, boolean in enumerate(bool_array):
                if boolean:
                    index = _np.intp(
                        _np.where(lostp[idx] == all_lostp[j])[0][0]
                    )
                    scat_data.append(prob[idx][index])
                else:
                    scat_data.append(0)

            if not idx:
                dic_res["lost_positions"] = all_lostp
                stri = f"{scattered_pos}"
                dic_res[stri] = scat_data
            else:
                stri = f"{scattered_pos}"
                dic_res[stri] = scat_data

            # df = _pd.DataFrame(dic_res)

        return dic_res

    def get_scat_dict(
        self, l_scattered_pos, reording_key, scrap, vchamber
    ):
        """Get the reordered dictionary."""
        dic = self._f_scat_table(l_scattered_pos, scrap, vchamber)

        zip_tuples = zip(*[dic[chave] for chave in dic])
        new_tuples = sorted(
            zip_tuples, key=lambda x: x[list(dic.keys()).index(reording_key)]
        )
        zip_ordered = zip(*new_tuples)

        new_dict = {
            chave: list(valores)
            for chave, valores in zip(dic.keys(), zip_ordered)
        }

        return new_dict

    def get_loss_profile(self, dic):
        """Integrates the lost positions for all scattering points.

        dic = cointains the lost positions and the scattered points.
        """
        spos = self.spos

        df = _pd.DataFrame(dic)
        a = df.set_index("lost_positions")

        scat_pos = _np.array(a.columns, dtype=float)

        indices = []
        for iten in scat_pos:
            ind = _np.argmin(_np.abs(spos - iten))
            indices.append(ind)

        summed = []
        for idx, _ in a.iterrows():
            sum_row = scyint.trapz(a.loc[idx], spos[indices])
            summed.append(sum_row)

        _, ax = _plt.subplots(
            figsize=(13, 7), gridspec_kw={"hspace": 0.2, "wspace": 0.2}
        )
        ax.set_title("loss rate integral along the ring", fontsize=16)

        ax.set_xlabel("lost position [m]", fontsize=16)
        ax.set_ylabel("loss rate [1/s]", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)

        ax.plot(list(a.index), summed, color="navy")
        _pyaccel.graphics.draw_lattice(
            self._model_fit, offset=-1e-6, height=1e-6, gca=True
        )

    def get_loss_profilel(self, l_dic):
        """Comparing distinct loss profiles.

        l_dic = list of dictionaries of loss profiles.
        """
        lista = []
        for dic in l_dic:
            s = self.spos

            df = _pd.DataFrame(dic)
            a = df.set_index("lost_positions")

            scat_pos = _np.array(a.columns, dtype=float)

            indices = []
            for iten in scat_pos:
                ind = _np.argmin(_np.abs(s - iten))
                indices.append(ind)

            summed = []
            for idx, _ in a.iterrows():
                sum_row = scyint.trapz(a.loc[idx], s[indices])
                summed.append(sum_row)

            lista.append((a.index, summed))

        return lista

    def plot_scat_dict(self, new_dic):
        """Heatmap plot indicating the warm points of loss along the ring.

        new_dic = contains the reordered dict with lost positions and
        scattered points.
        """
        s = self.spos

        df = _pd.DataFrame(new_dic)
        df = df.set_index("lost_positions")

        val = df.values.copy()
        idx = val != 0.0
        val[idx] = _np.log10(val[idx])
        val[~idx] = val[idx].min()

        fig, ax = _plt.subplots(figsize=(10, 10))

        y = _np.linspace(0, s[-1], df.shape[0] + 1)
        x = _np.linspace(0, s[-1], df.shape[1] + 1)
        x_mesh, y_mesh = _np.meshgrid(x, y)

        heatmp = ax.pcolor(x_mesh, y_mesh, val, cmap="jet", shading="flat")

        cbar = _plt.colorbar(heatmp)
        cbar.set_label("Loss rate [1/s] in logarithmic scale", rotation=90)

        ax.set_title("Loss profile", fontsize=16)

        ax.set_xlabel("scattered positions [m]", fontsize=16)
        ax.set_ylabel("lost positions [m]", fontsize=16)

        fig.tight_layout()
        _plt.gca().set_aspect("equal")
        _plt.show()
