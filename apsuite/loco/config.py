"""."""

from mathphys.functions import get_namedtuple as _get_namedtuple
from copy import deepcopy as _dcopy

import numpy as _np

import pyaccel as _pyaccel

from ..orbcorr import OrbRespmat as _OrbRespmat
from .utils import LOCOUtils as _LOCOUtils


class LOCOConfig:
    """LOCO configuration template."""

    DEFAULT_DELTA_KL = 1e-6  # [1/m]
    DEFAULT_DELTA_KSL = 1e-6  # [1/m]
    DEFAULT_DELTA_DIP_HKICK = 1e-6  # [rad]
    DEFAULT_DELTA_RF = 100  # [Hz]
    DEFAULT_SVD_THRESHOLD = 1e-6
    DEFAULT_DELTAK_NORMALIZATION = 1e-3
    DEFAULT_GIRDER_SHIFT = 1e-6  # [m]

    FAMNAME_RF = "SRFCav"

    INVERSION = _get_namedtuple("Methods", ["Normal", "Transpose"])
    MINIMIZATION = _get_namedtuple(
        "Methods", ["GaussNewton", "LevenbergMarquardt"]
    )
    SVD = _get_namedtuple("Methods", ["Selection", "Threshold"])

    def __init__(self, **kwargs):
        """."""
        self._inversion = LOCOConfig.INVERSION.Normal
        self._minimization = LOCOConfig.MINIMIZATION.GaussNewton
        self._svd_method = LOCOConfig.SVD.Selection
        self.model = None
        self.dim = None
        self.respm = None
        self.goalmat = None
        self.measured_dispersion = None
        self.delta_kickx_meas = None
        self.delta_kicky_meas = None
        self.delta_frequency_meas = None
        self.fitting_method = None
        self.parallel = True  # must be False, True or an integer (nr of cores)
        self.lambda_lm = None
        self.fixed_lambda = None
        self.use_dispersion = None
        self.use_offdiagonal = None
        self.use_diagonal = None
        self.use_dip_families = None
        self.use_quad_families = None
        self.use_sext_families = None
        self.dipoles_to_fit = None
        self.quadrupoles_to_fit = None
        self.skew_quadrupoles_to_fit = None
        self.sextupoles_to_fit = None
        self.svd_sel = None
        self.svd_thre = None
        self.fit_quadrupoles = None
        self.fit_sextupoles = None
        self.fit_dipoles = None
        self.fit_quadrupoles_coupling = None
        self.fit_sextupoles_coupling = None
        self.fit_dipoles_coupling = None
        self.fit_gain_bpm = None
        self.fit_roll_bpm = None
        self.fit_gain_corr = None
        self.fit_dipoles_hkick = None
        self.fit_energy_shift = None
        self.fit_girder_shift = None
        self.constraint_deltakl_total = None
        self.constraint_deltakl_step = None
        self.fit_skew_quadrupoles = None
        self.cavidx = None
        self.matrix = None
        self.idx_cav = None
        self.idx_bpm = None
        self.gain_bpm = None
        self.gain_corr = None
        self.roll_bpm = None
        self.roll_corr = None
        self.vector = None
        self.quad_indices_kl = None
        self.quad_indices_ksl = None
        self.sext_indices_kl = None
        self.sext_indices_ksl = None
        self.dip_indices_kl = None
        self.dip_indices_ksl = None
        self.dip_indices_hkick = None
        self.b1_indices_kl = None
        self.b2_indices_kl = None
        self.bc_indices_kl = None
        self.skew_quad_indices_ksl = None
        self.gir_indices = None
        self.k_nrsets = None
        self.weight_bpm = None
        self.weight_corr = None
        self.weight_deltakl = None
        self.weight_dispx = None
        self.weight_dispy = None
        self.deltakl_normalization = None
        self.tolerance_delta = None
        self.tolerance_overfit = None
        self.nr_fit_parameters = None

        self._process_input(kwargs)

    def __str__(self):
        """."""
        stmp = "{0:35s}: {1:}  {2:s}\n".format
        ftmp = "{0:35s}: {1:3.2f}  {2:s}\n".format
        dtmp = "{0:35s}: {1:3d}  {2:s}\n".format
        etmp = "{0:35s}: {1:e}  {2:s}\n".format

        stg = stmp("Tracking dimension", self.dim, "")
        stg += stmp("Include dispersion", self.use_dispersion, "")
        stg += stmp("Include diagonal", self.use_diagonal, "")
        stg += stmp("Include off-diagonal", self.use_offdiagonal, "")
        stg += stmp("Minimization method", self.min_method_str, "")
        stg += etmp("Lambda LM", self.lambda_lm, "")
        stg += stmp("Fixed lambda LM", self.fixed_lambda, "")
        stg += stmp("Jacobian manipulation", self.inv_method_str, "")
        stg += stmp(
            "Constraint delta KL total", self.constraint_deltakl_total, ""
        )
        stg += stmp(
            "Constraint delta KL step", self.constraint_deltakl_step, ""
        )
        stg += etmp(
            "Constraint delta KL normalization", self.deltakl_normalization, ""
        )
        stg += stmp("Singular values method", self.svd_method_str, "")

        if self.svd_method == LOCOConfig.SVD.Selection:
            if self.svd_sel is not None:
                stg += dtmp("SV to be used:", self.svd_sel, "")
            else:
                stg += stmp("SV to be used", "All", "")
        if self.svd_method == LOCOConfig.SVD.Threshold:
            stg += ftmp("SV threshold (s/s_max):", self.svd_thre, "")

        stg += etmp("Tolerance delta", self.tolerance_delta, "")
        stg += etmp("Tolerance overfit", self.tolerance_overfit, "")

        stg += ftmp(
            "H. kicks used to measure", self.delta_kickx_meas * 1e6, "[urad]"
        )
        stg += ftmp(
            "V. kicks used to measure", self.delta_kicky_meas * 1e6, "[urad]"
        )
        stg += ftmp(
            "RF freq. variation used to measure",
            self.delta_frequency_meas,
            "[Hz]",
        )

        stg += stmp("Dipoles normal gradients", self.fit_dipoles, "")
        stg += stmp("Quadrupoles normal gradients", self.fit_quadrupoles, "")
        stg += stmp("Sextupoles normal gradients", self.fit_sextupoles, "")

        stg += stmp("Use dipoles as families", self.use_dip_families, "")
        stg += stmp("Use quadrupoles as families", self.use_quad_families, "")
        stg += stmp("Use sextupoles as families", self.use_sext_families, "")

        stg += stmp("Dipoles skew gradients", self.fit_dipoles_coupling, "")
        stg += stmp(
            "Quadrupoles skew gradients", self.fit_quadrupoles_coupling, ""
        )
        stg += stmp(
            "Sextupoles skew gradients", self.fit_sextupoles_coupling, ""
        )
        stg += stmp(
            "Skew quadrupoles skew gradients", self.fit_skew_quadrupoles, ""
        )
        stg += stmp("Girders longitudinal shifts", self.fit_girder_shift, "")

        stg += stmp("BPM gains", self.fit_gain_bpm, "")
        stg += stmp("Corrector gains", self.fit_gain_corr, "")
        stg += stmp("BPM roll", self.fit_roll_bpm, "")
        stg += dtmp("Nr. fit parameters:", self.nr_fit_parameters, "")
        return stg

    @property
    def acc(self):
        """."""
        raise NotImplementedError

    @property
    def nr_bpm(self):
        """."""
        raise NotImplementedError

    @property
    def nr_ch(self):
        """."""
        raise NotImplementedError

    @property
    def nr_cv(self):
        """."""
        raise NotImplementedError

    @property
    def nr_corr(self):
        """."""
        return self.nr_ch + self.nr_cv

    @property
    def famname_quadset(self):
        """."""
        raise NotImplementedError

    @property
    def famname_sextset(self):
        """."""
        raise NotImplementedError

    @property
    def famname_dipset(self):
        """."""
        raise NotImplementedError

    @property
    def inv_method(self):
        """."""
        return self._inversion

    @inv_method.setter
    def inv_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._inversion = int(value in LOCOConfig.INVERSION._fields[1])
        elif int(value) in LOCOConfig.INVERSION:
            self._inversion = int(value)

    @property
    def inv_method_str(self):
        """."""
        return LOCOConfig.INVERSION._fields[self._inversion]

    @property
    def min_method(self):
        """."""
        return self._minimization

    @min_method.setter
    def min_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._minimization = int(
                value in LOCOConfig.MINIMIZATION._fields[1]
            )
        elif int(value) in LOCOConfig.MINIMIZATION:
            self._minimization = int(value)

    @property
    def min_method_str(self):
        """."""
        return LOCOConfig.MINIMIZATION._fields[self._minimization]

    @property
    def svd_method(self):
        """."""
        return self._svd_method

    @svd_method.setter
    def svd_method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._svd_method = int(value in LOCOConfig.SVD._fields[1])
        elif int(value) in LOCOConfig.SVD:
            self._svd_method = int(value)

    @property
    def svd_method_str(self):
        """."""
        return LOCOConfig.SVD._fields[self._svd_method]

    def update(self):
        """."""
        self.update_model(self.model, self.dim)
        self.update_matrix(self.use_dispersion)
        self.update_goalmat(
            self.goalmat, self.use_dispersion, self.use_offdiagonal
        )
        self.update_gain()
        self.update_quad_knobs(self.use_quad_families)
        self.update_sext_knobs(self.use_sext_families)
        self.update_dip_knobs(self.use_dip_families)
        self.update_girder_knobs()
        self.update_skew_quad_knobs()
        self.update_weight()
        self.update_svd(self.svd_method, self.svd_sel, self.svd_thre)
        self.nr_fit_parameters = self.get_nr_fit_parameters()

    def update_model(self, model, dim):
        """."""
        self.dim = dim
        self.model = _dcopy(model)
        self.model.cavity_on = dim == "6d"
        self.model.radiation_on = False
        self.respm = _OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self._create_indices()

    def update_svd(
        self, svd_method, svd_sel=None, svd_thre=None, flat_print=False
    ):
        """."""
        self.svd_sel = svd_sel
        self.svd_thre = svd_thre
        if svd_method == LOCOConfig.SVD.Selection:
            if svd_sel is not None:
                if flat_print:
                    print(
                        "svd_selection: {:d} values will be used.".format(
                            self.svd_sel
                        )
                    )
            else:
                if flat_print:
                    print("svd_selection: all values will be used.")
        if svd_method == LOCOConfig.SVD.Threshold:
            if svd_thre is None:
                self.svd_thre = LOCOConfig.DEFAULT_SVD_THRESHOLD
            if flat_print:
                print("svd_threshold: {:f}".format(self.svd_thre))

    def update_goalmat(self, goalmat, use_dispersion, use_offdiagonal):
        """."""
        # init goalmat
        if goalmat is None:
            goalmat = _dcopy(self.matrix)

        # coupling
        self.use_offdiagonal = use_offdiagonal
        if not use_offdiagonal:
            self.goalmat = _LOCOUtils.remove_offdiagonal(
                goalmat, self.nr_bpm, self.nr_ch, self.nr_cv
            )
        else:
            self.goalmat = _dcopy(goalmat)

        # dispersion
        self.use_dispersion = use_dispersion
        if not self.use_dispersion:
            self.goalmat[:, -1] *= 0

    def update_matrix(self, use_dispersion):
        """."""
        self.matrix = _LOCOUtils.respm_calc(
            self.model, self.respm, use_dispersion
        )

    def update_gain(
        self, gain_bpm=None, gain_corr=None, roll_bpm=None, roll_corr=None
    ):
        """."""
        # bpm
        if gain_bpm is None:
            if self.gain_bpm is None:
                self.gain_bpm = _np.ones(2 * self.nr_bpm)
        else:
            if isinstance(gain_bpm, (int, float)):
                self.gain_bpm = _np.ones(2 * self.nr_bpm) * gain_bpm
            else:
                print("setting initial bpm gain...")
                self.gain_bpm = gain_bpm
        if roll_bpm is None:
            if self.roll_bpm is None:
                self.roll_bpm = _np.zeros(self.nr_bpm)
        else:
            if isinstance(roll_bpm, (int, float)):
                self.roll_bpm = _np.ones(self.nr_bpm) * roll_bpm
            else:
                print("setting initial bpm roll...")
                self.roll_bpm = roll_bpm
        # corr
        if gain_corr is None:
            if self.gain_corr is None:
                self.gain_corr = _np.ones(self.nr_corr)
        else:
            if isinstance(gain_corr, (int, float)):
                self.gain_bpm = _np.ones(self.nr_corr) * gain_corr
            else:
                print("setting initial corrector gain...")
                self.gain_corr = gain_corr
        if roll_corr is None:
            if self.roll_corr is None:
                self.roll_corr = _np.zeros(self.nr_corr)
        else:
            if isinstance(roll_corr, (int, float)):
                self.roll_corr = _np.ones(self.nr_bpm) * roll_corr
            else:
                self.roll_corr = roll_corr

        self.matrix = _LOCOUtils.apply_all_gain(
            matrix=self.matrix,
            gain_bpm=self.gain_bpm,
            roll_bpm=self.roll_bpm,
            gain_corr=self.gain_corr,
        )
        self.vector = self.matrix.ravel()

    def update_weight(self):
        """."""
        # bpm
        bpmw = self.weight_bpm
        if bpmw is None:
            bpmw = _np.ones((2 * self.nr_bpm, self.nr_corr + 1))
        elif isinstance(bpmw, (int, float)):
            weight_bpm = _np.ones((2 * self.nr_bpm, self.nr_corr + 1))
            weight_bpm *= bpmw / 2 / self.nr_bpm
            bpmw = weight_bpm
        elif isinstance(bpmw, _np.ndarray) and bpmw.ndim == 1:
            bpmw = _np.tile(bpmw[:, None], self.nr_corr + 1)
        self.weight_bpm = bpmw

        # corr
        corrw = self.weight_corr
        if corrw is None:
            corrw = _np.ones(self.nr_corr + 1)
        elif isinstance(corrw, (int, float)):
            weight_corr = _np.ones(self.nr_corr + 1)
            weight_corr *= corrw / (self.nr_corr + 1)
            corrw = weight_corr
        self.weight_corr = corrw

        nknb = 0
        if self.fit_quadrupoles:
            nknb += len(self.quad_indices_kl)
        if self.fit_dipoles:
            nknb += len(self.dip_indices_kl)
        if self.fit_sextupoles:
            nknb += len(self.sext_indices_kl)

        # delta kl constraint weight
        if self.weight_deltakl is None:
            self.weight_deltakl = _np.ones(nknb)
        elif isinstance(self.weight_deltakl, (int, float)):
            self.weight_deltakl = _np.ones(nknb) * self.weight_deltakl

        if self.deltakl_normalization is None:
            self.deltakl_normalization = (
                LOCOConfig.DEFAULT_DELTAK_NORMALIZATION
            )

        if self.weight_dispx is None:
            self.weight_dispx = 1

        if self.weight_dispy is None:
            self.weight_dispy = 1

    def update_quad_knobs(self, use_families):
        """."""
        if self.quadrupoles_to_fit is None:
            self.quadrupoles_to_fit = self.famname_quadset
        else:
            setquadfit = set(self.quadrupoles_to_fit)
            setquadall = set(self.famname_quadset)
            if not setquadfit.issubset(setquadall):
                raise Exception("invalid quadrupole name used to fit!")
        if use_families is None:
            use_families = False
        self.use_quad_families = use_families
        if self.use_quad_families:
            self.quad_indices_kl = [None] * len(self.quadrupoles_to_fit)
            self.quad_indices_ksl = []
            for idx, fam_name in enumerate(self.quadrupoles_to_fit):
                fam = self.respm.fam_data
                self.quad_indices_kl[idx] = fam[fam_name]["index"]
                self.quad_indices_ksl += self.quad_indices_kl[idx]
            self.quad_indices_ksl.sort()
        else:
            self.quad_indices_kl = []
            for fam_name in self.quadrupoles_to_fit:
                self.quad_indices_kl += self.respm.fam_data[fam_name]["index"]
            self.quad_indices_kl.sort()
            self.quad_indices_ksl = self.quad_indices_kl

    def update_sext_knobs(self, use_families):
        """."""
        if self.sextupoles_to_fit is None:
            self.sextupoles_to_fit = self.famname_sextset
        else:
            setsextfit = set(self.sextupoles_to_fit)
            setsextall = set(self.famname_sextset)
            if not setsextfit.issubset(setsextall):
                raise Exception("invalid sextupole name used to fit!")
        if use_families is None:
            use_families = False
        self.use_sext_families = use_families
        if self.use_sext_families:
            self.sext_indices_kl = [None] * len(self.sextupoles_to_fit)
            self.sext_indices_ksl = []
            for idx, fam_name in enumerate(self.sextupoles_to_fit):
                fam = self.respm.fam_data
                self.sext_indices_kl[idx] = fam[fam_name]["index"]
                self.sext_indices_ksl += self.sext_indices_kl[idx]
            self.sext_indices_ksl.sort()
        else:
            self.sext_indices_kl = []
            for fam_name in self.sextupoles_to_fit:
                fam = self.respm.fam_data
                self.sext_indices_kl += fam[fam_name]["index"]
            self.sext_indices_kl.sort()
            self.sext_indices_ksl = self.sext_indices_kl

    def update_skew_quad_knobs(self):
        """."""
        if self.skew_quadrupoles_to_fit is None:
            self.skew_quad_indices_ksl = self.respm.fam_data["QS"]["index"]
        else:
            skewquadfit = set(self.skew_quadrupoles_to_fit)
            skewquadall = set(self.famname_skewquadset)
            if not skewquadfit.issubset(skewquadall):
                raise Exception("invalid skew quadrupole name used to fit!")
            else:
                self.skew_quad_indices_ksl = []
                for fam_name in self.skew_quadrupoles_to_fit:
                    fam = self.respm.fam_data
                    self.skew_quad_indices_ksl += fam[fam_name]["index"]
                idx_all = _np.array(self.respm.fam_data["QS"]["index"]).ravel()
                idx_sub = _np.array(self.skew_quad_indices_ksl).ravel()
                self.skew_quad_indices_ksl = list(set(idx_sub) & set(idx_all))
                self.skew_quad_indices_ksl.sort()

    def update_b1_knobs(self):
        """."""
        self.b1_indices_kl = self.respm.fam_data["B1"]["index"]

    def update_b2_knobs(self):
        """."""
        self.b2_indices_kl = self.respm.fam_data["B2"]["index"]

    def update_bc_knobs(self):
        """."""
        self.bc_indices_kl = self.respm.fam_data["BC"]["index"]

    def update_dip_knobs(self, use_families):
        """."""
        if self.dipoles_to_fit is None:
            self.dipoles_to_fit = self.famname_dipset
        else:
            setdipfit = set(self.dipoles_to_fit)
            setdipall = set(self.famname_dipset)
            if not setdipfit.issubset(setdipall):
                raise Exception("invalid dipole name used to fit!")
        if use_families is None:
            use_families = False
        self.use_dip_families = use_families
        if self.use_dip_families:
            self.dip_indices_kl = [None] * len(self.dipoles_to_fit)
            self.dip_indices_ksl = []
            self.dip_indices_hkick = []
            for idx, fam_name in enumerate(self.dipoles_to_fit):
                fam = self.respm.fam_data
                self.dip_indices_kl[idx] = fam[fam_name]["index"]
                self.dip_indices_ksl += self.dip_indices_kl[idx]
                self.dip_indices_ksl.sort()
                self.dip_indices_hkick += self.dip_indices_kl[idx]
                self.dip_indices_hkick.sort()
        else:
            self.dip_indices_kl = []
            for fam_name in self.dipoles_to_fit:
                self.dip_indices_kl += self.respm.fam_data[fam_name]["index"]
                self.dip_indices_kl.sort()
                self.dip_indices_ksl = self.dip_indices_kl
                self.dip_indices_hkick = self.dip_indices_kl

    def update_girder_knobs(self):
        """."""
        self.gir_indices = _pyaccel.lattice.find_indices(
            self.model, "fam_name", "girder"
        )
        self.gir_indices = _np.reshape(self.gir_indices, (-1, 2))

    def get_nr_fit_parameters(self):
        """."""
        idx = 0
        if self.fit_quadrupoles:
            idx += len(self.quad_indices_kl)
        if self.fit_sextupoles:
            idx += len(self.sext_indices_kl)
        if self.fit_dipoles:
            idx += len(self.dip_indices_kl)
        if self.fit_quadrupoles_coupling:
            idx += len(self.quad_indices_ksl)
        if self.fit_sextupoles_coupling:
            idx += len(self.sext_indices_ksl)
        if self.fit_dipoles_coupling:
            idx += len(self.dip_indices_ksl)
        if self.fit_gain_bpm:
            idx += 2 * self.nr_bpm
        if self.fit_roll_bpm:
            idx += self.nr_bpm
        if self.fit_gain_corr:
            idx += self.nr_corr
        if self.fit_dipoles_hkick:
            idx += len(self.dip_indices_hkick)
        if self.fit_energy_shift:
            idx += self.nr_corr
        if self.fit_skew_quadrupoles:
            idx += len(self.skew_quad_indices_ksl)
        if self.fit_girder_shift:
            idx += self.gir_indices.shape[0]
        return idx

    def _process_input(self, kwargs):
        for key, value in kwargs.items():
            if key == "model" and "dim" in kwargs:
                model, dim = kwargs["model"], kwargs["dim"]
                self.update_model(model, dim)
            elif key == "dim":
                pass
            elif key == "svd_method" and (
                "svd_sel" in kwargs or "svd_thre" in kwargs
            ):
                svd_method = kwargs["svd_method"]
                svd_sel = kwargs["svd_sel"] if "svd_sel" in kwargs else None
                svd_thre = kwargs["svd_thre"] if "svd_thre" in kwargs else None
                self.update_svd(svd_method, svd_sel, svd_thre)
            setattr(self, key, value)

    def _create_indices(self):
        """."""
        self.idx_cav = _pyaccel.lattice.find_indices(
            self.model, "fam_name", self.FAMNAME_RF
        )[0]
        self.idx_bpm = _pyaccel.lattice.find_indices(
            self.model, "fam_name", "BPM"
        )


class LOCOConfigSI(LOCOConfig):
    """Sirius Storage Ring LOCO configuration."""

    @property
    def acc(self):
        """."""
        return "SI"

    @property
    def nr_bpm(self):
        """."""
        return 160

    @property
    def nr_ch(self):
        """."""
        return 120

    @property
    def nr_cv(self):
        """."""
        return 160

    @property
    def famname_dipset(self):
        """."""
        return ["B1", "B2", "BC"]

    @property
    def famname_quadset(self):
        """."""
        return [
            "QFA",
            "QDA",
            "QDB2",
            "QFB",
            "QDB1",
            "QDP2",
            "QFP",
            "QDP1",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ]

    @property
    def famname_sextset(self):
        """."""
        return [
            "SDA0",
            "SDB0",
            "SDP0",
            "SDA1",
            "SDB1",
            "SDP1",
            "SDA2",
            "SDB2",
            "SDP2",
            "SDA3",
            "SDB3",
            "SDP3",
            "SFA0",
            "SFB0",
            "SFP0",
            "SFA1",
            "SFB1",
            "SFP1",
            "SFA2",
            "SFB2",
            "SFP2",
        ]

    @property
    def famname_skewquadset(self):
        """."""
        return [
            "SFA0",
            "SDB0",
            "SDP0",
            "SDA2",
            "SDB2",
            "FC2",
            "SDP2",
            "SDA3",
            "SDB3",
            "SDP3",
        ]


class LOCOConfigBO(LOCOConfig):
    """Sirius Booster LOCO configuration."""

    @property
    def acc(self):
        """."""
        return "BO"

    @property
    def nr_bpm(self):
        """."""
        return 50

    @property
    def nr_ch(self):
        """."""
        return 25

    @property
    def nr_cv(self):
        """."""
        return 25

    @property
    def famname_dipset(self):
        """."""
        return ["B"]

    @property
    def famname_quadset(self):
        """."""
        return ["QF", "QD"]

    @property
    def famname_sextset(self):
        """."""
        return ["SF", "SD"]

    @property
    def famname_skewquadset(self):
        """."""
        return ["QS"]
