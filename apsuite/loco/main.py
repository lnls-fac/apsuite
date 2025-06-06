"""."""

from copy import deepcopy as _dcopy
import time as _time
import numpy as _np

from siriuspy.namesys import SiriusPVName as _PVName
import pyaccel as _pyaccel

from .utils import LOCOUtils as _LOCOUtils
from .config import LOCOConfig as _LOCOConfig


class LOCO:
    """Main LOCO algorithm."""

    DEFAULT_TOL_DELTA = 1e-6
    DEFAULT_TOL_OVERFIT = 1e-6
    DEFAULT_REDUC_THRESHOLD = 5 / 100
    DEFAULT_LAMBDA_LM = 1e-3
    DEFAULT_MAX_LAMBDA_LM = 1e6
    DEFAULT_MIN_LAMBDA_LM = 1e-6
    DEFAULT_INCR_RATE_LAMBDA_LM = 10
    DEFAULT_DECR_RATE_LAMBDA_LM = 10

    def __init__(self, config=None, save_jacobian_matrices=False):
        """."""
        if config is not None:
            self.config = config
        else:
            self.config = _LOCOConfig()

        if (
            self.config.min_method
            == _LOCOConfig.MINIMIZATION.LevenbergMarquardt
        ):
            if self.config.lambda_lm is None:
                self.config.lambda_lm = LOCO.DEFAULT_LAMBDA_LM

        check_problem = (
            self.config.min_method
            == _LOCOConfig.MINIMIZATION.LevenbergMarquardt
        )
        check_problem &= self.config.inv_method == _LOCOConfig.INVERSION.Normal

        if check_problem:
            raise ValueError(
                "Levenberg-Marquardt works with Transpose Inversion only!"
            )

        self._model = None
        self._matrix = None
        self._nr_k_sets = None

        self._jloco_gain_bpm = None
        self._jloco_roll_bpm = None
        self._jloco_gain_corr = None
        self._jloco_kl = None
        self._jloco_kl_quad = None
        self._jloco_kl_sext = None
        self._jloco_kl_dip = None
        self._jloco_kl_b1 = None
        self._jloco_kl_b2 = None
        self._jloco_kl_bc = None

        self._jloco_ksl = None
        self._jloco_ksl_quad = None
        self._jloco_ksl_sext = None
        self._jloco_ksl_dip = None
        self._jloco_ksl_b1 = None
        self._jloco_ksl_b2 = None
        self._jloco_ksl_bc = None
        self._jloco_ksl_skew_quad = None

        self._jloco_hkick_b1 = None
        self._jloco_hkick_b2 = None
        self._jloco_hkick_bc = None
        self._jloco_hkick_dip = None

        self._jloco_energy_shift = None
        self._jloco_girder_shift = None

        self._jloco = None
        self._jloco_inv = None
        self._jloco_u = None
        self._jloco_s = None
        self._jloco_vt = None
        self._lm_dmat = None
        self._deltakl_mat = None

        self._dip_kl_inival = None
        self._dip_kl_deltas = None
        self._quad_kl_inival = None
        self._quad_kl_deltas = None
        self._quad_kl_deltas_step = None
        self._sext_kl_inival = None
        self._sext_kl_deltas = None

        self._dip_ksl_inival = None
        self._dip_ksl_deltas = None
        self._quad_ksl_inival = None
        self._quad_ksl_deltas = None
        self._sext_ksl_inival = None
        self._sext_ksl_deltas = None
        self._skew_quad_ksl_inival = None
        self._skew_quad_ksl_deltas = None
        self._skew_quad_ksl_deltas_step = None

        self._dip_hkick_inival = None
        self._dip_hkick_deltas = None

        self._energy_shift_inival = None
        self._energy_shift_deltas = None
        self._girders_shift_inival = None
        self._girders_shift_deltas = None

        self._gain_bpm_inival = self.config.gain_bpm
        self._gain_bpm_delta = None
        self._roll_bpm_inival = self.config.roll_bpm
        self._roll_bpm_delta = None
        self._gain_corr_inival = self.config.gain_corr
        self._gain_corr_delta = None

        self._chi_init = None
        self._chi = None
        self._chi_history = []
        self._tol = None
        self._tol_delta = None
        self._tol_overfit = None
        self._reduc_threshold = None
        self._res_history = []
        self._kldelta_history = []
        self._ksldelta_history = []

        self.fitmodel = None
        self.chi_history = None
        self.bpm_gain = None
        self.bpm_roll = None
        self.corr_gain = None
        self.energy_shift = None
        self.residue_history = None
        self.girder_shift = None
        self.kldelta_history = None
        self.ksldelta_history = None

        self.save_jacobian_matrices = save_jacobian_matrices

    def update(
        self,
        fname_jloco_kl=None,
        fname_inv_jloco_kl=None,
        fname_jloco_kl_quad=None,
        fname_jloco_kl_sext=None,
        fname_jloco_kl_dip=None,
        fname_jloco_ksl_quad=None,
        fname_jloco_ksl_sext=None,
        fname_jloco_ksl_dip=None,
        fname_jloco_hkick_dip=None,
        fname_jloco_ksl_skewquad=None,
        fname_jloco_girder_shift=None,
    ):
        """."""
        print("update config...")
        self.update_config()
        if fname_inv_jloco_kl is not None:
            print("setting jloco inverse input...")
            self._jloco_inv = _LOCOUtils.load_data(fname=fname_inv_jloco_kl)
        else:
            print("update jloco...")
            self.update_jloco(
                fname_jloco_kl=fname_jloco_kl,
                fname_jloco_kl_quad=fname_jloco_kl_quad,
                fname_jloco_kl_sext=fname_jloco_kl_sext,
                fname_jloco_kl_dip=fname_jloco_kl_dip,
                fname_jloco_ksl_quad=fname_jloco_ksl_quad,
                fname_jloco_ksl_sext=fname_jloco_ksl_sext,
                fname_jloco_ksl_dip=fname_jloco_ksl_dip,
                fname_jloco_hkick_dip=fname_jloco_hkick_dip,
                fname_jloco_ksl_skewquad=fname_jloco_ksl_skewquad,
                fname_jloco_girder_shift=fname_jloco_girder_shift,
            )
            print("update svd...")
            self.update_svd()
        print("update fit...")
        self.update_fit()

    def update_config(self):
        """."""
        self.config.update()
        # reset model
        self._model = _dcopy(self.config.model)
        self._matrix = _dcopy(self.config.matrix)

    def _handle_dip_fit_kl(self, fname_jloco_kl_dip):
        # calculate K jacobian for dipole
        if self.config.fit_dipoles:
            if fname_jloco_kl_dip is None:
                print("calculating dipoles klmatrix...")
                time0 = _time.time()
                self._jloco_kl_dip = _LOCOUtils.jloco_calc_kl_dip(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading dipole klmatrix...")
                jloco_kl_dip_dict = _LOCOUtils.load_data(fname_jloco_kl_dip)
                self._jloco_kl_dip = self._convert_dict2array(
                    jloco_kl_dip_dict, "dipole"
                )

    def _handle_quad_fit_kl(self, fname_jloco_kl_quad):
        # calculate K jacobian for quadrupole
        if self.config.fit_quadrupoles:
            if fname_jloco_kl_quad is None:
                print("calculating quadrupoles klmatrix...")
                time0 = _time.time()
                self._jloco_kl_quad = _LOCOUtils.jloco_calc_kl_quad(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading quadrupoles klmatrix...")
                jloco_kl_quad_dict = _LOCOUtils.load_data(fname_jloco_kl_quad)
                self._jloco_kl_quad = self._convert_dict2array(
                    jloco_kl_quad_dict, "quadrupole"
                )

    def _convert_dict2array(self, jlocodict, magtype, is_normal=True):
        if "knobs_order" in jlocodict:
            knobs_order = jlocodict["knobs_order"]
        else:
            raise Exception("knobs_order missing in jloco dictionary!")
        if magtype == "dipole":
            if self.config.dipoles_to_fit is not None:
                magstofit = self.config.dipoles_to_fit
            else:
                magstofit = self.config.famname_dipset
            if is_normal:
                index = self.config.dip_indices_kl
            else:
                index = self.config.dip_indices_ksl
        elif magtype == "quadrupole":
            if self.config.quadrupoles_to_fit is not None:
                magstofit = self.config.quadrupoles_to_fit
            else:
                magstofit = self.config.famname_quadset
            if is_normal:
                index = self.config.quad_indices_kl
            else:
                index = self.config.quad_indices_ksl
        elif magtype == "sextupole":
            if self.config.sextupoles_to_fit is not None:
                magstofit = self.config.sextupoles_to_fit
            else:
                magstofit = self.config.famname_sextset
            if is_normal:
                index = self.config.sext_indices_kl
            else:
                index = self.config.sext_indices_ksl
        elif magtype == "skew_quadrupole":
            if self.config.skew_quadrupoles_to_fit is not None:
                magstofit = self.config.skew_quadrupoles_to_fit
            else:
                magstofit = self.config.famname_skewquadset
            index = self.config.skew_quad_indices_ksl
        quadfam = self.config.use_quad_families
        quadfam &= magtype == "quadrupole"
        quadfam &= is_normal

        dipfam = self.config.use_dip_families
        dipfam &= magtype == "dipole"
        dipfam &= is_normal

        sextfam = self.config.use_sext_families
        sextfam &= magtype == "sextupole"
        sextfam &= is_normal
        if dipfam or quadfam or sextfam:
            jloco = []
            for quad in magstofit:
                famcols = [
                    val for key, val in jlocodict.items() if quad in key
                ]
                jloco.append(sum(famcols))
            jloco = _np.array(jloco).T
        else:
            last_key = list(jlocodict.keys())[-1]
            jloco = _np.zeros((len(jlocodict[last_key]), len(index)))

            for idx, name in enumerate(knobs_order):
                name = _PVName(name)
                if name.dev not in magstofit:
                    jlocodict.pop(name, None)
                    knobs_order.pop(idx)

            for idx, name in enumerate(knobs_order):
                jloco[:, idx] = jlocodict[name]
        return jloco

    def _handle_sext_fit_kl(self, fname_jloco_kl_sext):
        # calculate K jacobian for sextupole
        if self.config.fit_sextupoles:
            if fname_jloco_kl_sext is None:
                print("calculating sextupoles klmatrix...")
                time0 = _time.time()
                self._jloco_kl_sext = _LOCOUtils.jloco_calc_kl_sext(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading sextupoles klmatrix...")
                jloco_kl_sext_dict = _LOCOUtils.load_data(fname_jloco_kl_sext)
                self._jloco_kl_sext = self._convert_dict2array(
                    jloco_kl_sext_dict, "sextupole"
                )

    def _handle_dip_fit_ksl(self, fname_jloco_ksl_dip):
        # calculate Ks jacobian for dipole
        if self.config.fit_dipoles_coupling:
            if fname_jloco_ksl_dip is None:
                print("calculating dipoles kslmatrix...")
                time0 = _time.time()
                self._jloco_ksl_dip = _LOCOUtils.jloco_calc_ksl_dipoles(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading dipole kslmatrix...")
                jloco_ksl_dip_dict = _LOCOUtils.load_data(fname_jloco_ksl_dip)
                self._jloco_ksl_dip = self._convert_dict2array(
                    jloco_ksl_dip_dict, "dipole", is_normal=False
                )

    def _handle_skewquad_fit_ksl(self, fname_jloco_ksl_skewquad):
        # calculate Ks jacobian for skew quadrupoles
        if self.config.fit_skew_quadrupoles:
            if fname_jloco_ksl_skewquad is None:
                print("calculating skew quadrupoles kslmatrix...")
                time0 = _time.time()
                self._jloco_ksl_skew_quad = _LOCOUtils.jloco_calc_ksl_skewquad(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading skew quadrupoles kslmatrix...")
                jloco_ksl_skewquad_dict = _LOCOUtils.load_data(
                    fname_jloco_ksl_skewquad
                )
                self._jloco_ksl_skew_quad = self._convert_dict2array(
                    jloco_ksl_skewquad_dict, "skew_quadrupole", is_normal=False
                )

    def _handle_quad_fit_ksl(self, fname_jloco_ksl_quad):
        # calculate Ks jacobian for quadrupole
        if self.config.fit_quadrupoles_coupling:
            if fname_jloco_ksl_quad is None:
                print("calculating quadrupoles kslmatrix...")
                time0 = _time.time()
                self._jloco_ksl_quad = _LOCOUtils.jloco_calc_ksl_quad(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading quadrupoles kslmatrix...")
                jloco_ksl_quad_dict = _LOCOUtils.load_data(
                    fname_jloco_ksl_quad
                )
                self._jloco_ksl_quad = self._convert_dict2array(
                    jloco_ksl_quad_dict, "quadrupole", is_normal=False
                )

    def _handle_sext_fit_ksl(self, fname_jloco_ksl_sext):
        # calculate Ks jacobian for sextupole
        if self.config.fit_sextupoles_coupling:
            if fname_jloco_ksl_sext is None:
                print("calculating sextupoles kslmatrix...")
                time0 = _time.time()
                self._jloco_ksl_sext = _LOCOUtils.jloco_calc_ksl_sextupoles(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading sextupoles kslmatrix...")
                jloco_ksl_sext_dict = _LOCOUtils.load_data(
                    fname_jloco_ksl_sext
                )
                self._jloco_ksl_sext = self._convert_dict2array(
                    jloco_ksl_sext_dict, "sextupole", is_normal=False
                )

    def _handle_dip_fit_hkick(self, fname_jloco_hkick_dip=None):
        # calculate hkick jacobian for dipole
        if self.config.fit_dipoles_hkick:
            if fname_jloco_hkick_dip is None:
                print("calculating dipoles hkick matrix...")
                self._jloco_hkick_dip = _LOCOUtils.jloco_calc_hkick_dipoles(
                    self.config, self._model
                )
            else:
                print("loading dipole hkick matrix...")
                self._jloco_hkick_dip = _LOCOUtils.load_data(
                    fname_jloco_hkick_dip
                )["jloco_klmatrix"]

    def _handle_girder_shift(self, fname_jloco_girder_shift=None):
        # calculate hkick jacobian for dipole
        if self.config.fit_girder_shift:
            if fname_jloco_girder_shift is None:
                print("calculating girder shift matrix...")
                time0 = _time.time()
                self._jloco_girder_shift = _LOCOUtils.jloco_calc_girders(
                    self.config, self._model
                )
                dtime = _time.time() - time0
                print("it took {:.2f} min to calculate".format(dtime / 60))
            else:
                print("loading girder shift matrix...")
                self._jloco_girder_shift = _LOCOUtils.load_data(
                    fname_jloco_girder_shift
                )["jloco_klmatrix"]

    def _get_ps_names(self, idx, sub):
        name_list = []
        for num, ind in enumerate(idx):
            name = "SI-"
            name += sub[num]
            name += ":PS-"
            if type(ind) == list:
                name += self._model[ind[0]].fam_name
            else:
                name += self._model[ind].fam_name
            name_list.append(name)
        return name_list

    def create_new_jacobian_dict(self, jloco, idx, sub):
        """."""
        newjloco = dict()
        name_list = self._get_ps_names(idx, sub)
        newjloco["knobs_order"] = name_list
        for num, name in enumerate(name_list):
            newjloco[name] = jloco[:, num]
        return newjloco

    def save_jacobian(self):
        """."""
        if self.config.fit_dipoles or self.config.fit_dipoles_coupling:
            idx_bn = self.config.respm.fam_data["BN"]["index"]
            sub_bn = self.config.respm.fam_data["BN"]["subsection"]
        if self.config.fit_quadrupoles or self.config.fit_quadrupoles_coupling:
            idx_qn = self.config.respm.fam_data["QN"]["index"]
            sub_qn = self.config.respm.fam_data["QN"]["subsection"]
        if self.config.fit_sextupoles or self.config.fit_sextupoles_coupling:
            idx_sn = self.config.respm.fam_data["SN"]["index"]
            sub_sn = self.config.respm.fam_data["SN"]["subsection"]

        if self.config.fit_dipoles:
            jloco_kl_dip = self.create_new_jacobian_dict(
                self._jloco_kl_dip, idx_bn, sub_bn
            )
            print("saving jacobian KL for dipoles")
            _LOCOUtils.save_data("6d_KL_dipoles", jloco_kl_dip)
        if self.config.fit_quadrupoles:
            jloco_kl_quad = self.create_new_jacobian_dict(
                self._jloco_kl_quad, idx_qn, sub_qn
            )
            print("saving jacobian KL for quadrupoles")
            _LOCOUtils.save_data("6d_KL_quadrupoles_trims", jloco_kl_quad)
        if self.config.fit_sextupoles:
            jloco_kl_sext = self.create_new_jacobian_dict(
                self._jloco_kl_sext, idx_sn, sub_sn
            )
            print("saving jacobian KL for sextupoles")
            _LOCOUtils.save_data("6d_KL_sextupoles", jloco_kl_sext)

        if self.config.fit_skew_quadrupoles:
            idx_qs = self.config.respm.fam_data["QS"]["index"]
            sub_qs = self.config.respm.fam_data["QS"]["subsection"]
            idx_qs = self.config.skew_quad_indices_ksl
            selidx = []
            for sel in self.config.skew_quad_indices_ksl:
                selidx.append(idx_qs.index([sel]))
            idx_qs = [idx_qs[idx] for idx in selidx]
            sub_qs = [sub_qs[idx] for idx in selidx]
            jloco_ksl_skewquad = self.create_new_jacobian_dict(
                self._jloco_ksl_skew_quad, idx_qs, sub_qs
            )
            print("saving jacobian KsL for skew quadrupoles")
            _LOCOUtils.save_data("6d_KsL_skew_quadrupoles", jloco_ksl_skewquad)

        if self.config.fit_dipoles_coupling:
            jloco_ksl_dip = self.create_new_jacobian_dict(
                self._jloco_ksl_dip, idx_bn, sub_bn
            )
            print("saving jacobian KsL for dipoles")
            _LOCOUtils.save_data("6d_KsL_dipoles", jloco_ksl_dip)
        if self.config.fit_quadrupoles_coupling:
            jloco_ksl_quad = self.create_new_jacobian_dict(
                self._jloco_ksl_quad, idx_qn, sub_qn
            )
            print("saving jacobian KsL for quadrupoles")
            _LOCOUtils.save_data("6d_KsL_quadrupoles", jloco_ksl_quad)
        if self.config.fit_sextupoles_coupling:
            jloco_ksl_sext = self.create_new_jacobian_dict(
                self._jloco_ksl_sext, idx_sn, sub_sn
            )
            print("saving jacobian KsL for sextupoles")
            _LOCOUtils.save_data("6d_KsL_sextupoles", jloco_ksl_sext)

    def update_jloco(
        self,
        fname_jloco_kl=None,
        fname_jloco_kl_dip=None,
        fname_jloco_kl_quad=None,
        fname_jloco_kl_sext=None,
        fname_jloco_ksl_dip=None,
        fname_jloco_ksl_quad=None,
        fname_jloco_ksl_sext=None,
        fname_jloco_hkick_dip=None,
        fname_jloco_ksl_skewquad=None,
        fname_jloco_girder_shift=None,
    ):
        """."""
        # calc jloco linear parts
        self._jloco_gain_bpm, self._jloco_roll_bpm, self._jloco_gain_corr = (
            _LOCOUtils.jloco_calc_linear(self.config, self._matrix)
        )

        if fname_jloco_kl is not None:
            self._jloco_kl = _LOCOUtils.load_data(fname_jloco_kl)[
                "jloco_klmatrix"
            ]
        else:
            self._handle_dip_fit_hkick(fname_jloco_hkick_dip)

            self._handle_dip_fit_kl(fname_jloco_kl_dip)
            self._handle_quad_fit_kl(fname_jloco_kl_quad)
            self._handle_sext_fit_kl(fname_jloco_kl_sext)

            self._handle_dip_fit_ksl(fname_jloco_ksl_dip)
            self._handle_quad_fit_ksl(fname_jloco_ksl_quad)
            self._handle_sext_fit_ksl(fname_jloco_ksl_sext)

            self._handle_skewquad_fit_ksl(fname_jloco_ksl_skewquad)

            self._handle_girder_shift(fname_jloco_girder_shift)

            if self.save_jacobian_matrices:
                self.save_jacobian()

        if self.config.fit_energy_shift:
            print("calculating energy shift matrix...")
            self._jloco_energy_shift = _LOCOUtils.jloco_calc_energy_shift(
                self.config, self._model
            )
        else:
            self._jloco_energy_shift = _np.zeros((
                self._matrix.size,
                self.config.nr_corr,
            ))

        # merge J submatrices
        self._jloco = _LOCOUtils.jloco_merge_linear(
            self.config,
            self._jloco_kl_quad,
            self._jloco_kl_sext,
            self._jloco_kl_dip,
            self._jloco_ksl_quad,
            self._jloco_ksl_sext,
            self._jloco_ksl_dip,
            self._jloco_gain_bpm,
            self._jloco_roll_bpm,
            self._jloco_gain_corr,
            self._jloco_hkick_dip,
            self._jloco_energy_shift,
            self._jloco_ksl_skew_quad,
            self._jloco_girder_shift,
        )

        # apply weight
        self._jloco = _LOCOUtils.jloco_apply_weight(
            self._jloco,
            self.config.weight_bpm,
            self.config.weight_corr,
            self.config.weight_dispx,
            self.config.weight_dispy,
        )

        if not self.config.use_dispersion:
            jloco_temp = _np.reshape(
                self._jloco,
                (2 * self.config.nr_bpm, self.config.nr_corr + 1, -1),
            )
            jloco_temp[:, -1, :] *= 0

        if self.config.constraint_deltakl_total:
            self.calc_jloco_deltakl_constraint()
            self._jloco = _np.vstack((self._jloco, self._deltakl_mat))

        self._jloco_u, self._jloco_s, self._jloco_vt = _np.linalg.svd(
            self._jloco, full_matrices=False
        )
        # print('save singular values for jloco')
        # _np.savetxt('svalues_j_lambda0_w0.txt', self._jloco_s)
        self._filter_svd_jloco()

    def _filter_svd_jloco(self):
        print("filtering singular values jloco...")
        umat, smat, vtmat = self._jloco_u, self._jloco_s, self._jloco_vt
        if self.config.svd_method == self.config.SVD.Threshold:
            bad_sv = smat / _np.max(smat) < self.config.svd_thre
            smat[bad_sv] = 0
        elif self.config.svd_method == self.config.SVD.Selection:
            smat[self.config.svd_sel :] = 0
        self._jloco = _np.dot(umat * smat[None, :], vtmat)

    def calc_lm_dmat(self):
        """."""
        ncols = self._jloco.shape[1]
        self._lm_dmat = _np.zeros(ncols)
        for col in range(ncols):
            self._lm_dmat[col] = _np.linalg.norm(self._jloco[:, col])
        self._lm_dmat = _np.diag(self._lm_dmat)

    def calc_jloco_deltakl_constraint(self):
        """."""
        sigma_deltak = self.config.deltakl_normalization
        ncols = self._jloco.shape[1]
        nknobs = 0

        if self.config.fit_quadrupoles:
            if self.config.use_quad_families:
                nknobs += len(self.config.quadrupoles_to_fit)
            else:
                nknobs += len(self.config.quad_indices_kl)
        if self.config.fit_dipoles:
            if self.config.use_dip_families:
                nknobs += len(self.config.dipoles_to_fit)
            else:
                nknobs += len(self.config.dip_indices_kl)
        if self.config.fit_sextupoles:
            if self.config.use_sext_families:
                nknobs += len(self.config.sextupoles_to_fit)
            else:
                nknobs += len(self.config.sext_indices_kl)

        deltakl_mat = _np.zeros((nknobs, ncols))
        for knb in range(nknobs):
            deltakl_mat[knb, knb] = self.config.weight_deltakl[knb]
        self._deltakl_mat = deltakl_mat / sigma_deltak

    def update_svd(self):
        """."""
        if self.config.inv_method == _LOCOConfig.INVERSION.Transpose:
            print("svd decomposition Jt * J")
            if self.config.min_method == _LOCOConfig.MINIMIZATION.GaussNewton:
                matrix2invert = self._jloco.T @ self._jloco
            elif (
                self.config.min_method
                == _LOCOConfig.MINIMIZATION.LevenbergMarquardt
            ):
                self.calc_lm_dmat()
                dmat = self._lm_dmat
                matrix2invert = self._jloco.T @ self._jloco
                matrix2invert += self.config.lambda_lm * dmat.T @ dmat
            if self.config.constraint_deltakl_step:
                self.calc_jloco_deltakl_constraint()
                matrix2invert += self._deltakl_mat.T @ self._deltakl_mat
            umat, smat, vtmat = _np.linalg.svd(
                matrix2invert, full_matrices=False
            )
        elif self.config.inv_method == _LOCOConfig.INVERSION.Normal:
            umat, smat, vtmat = self._jloco_u, self._jloco_s, self._jloco_vt
        # print('saving singular values for jtj loco')
        # _np.savetxt('svalues_jtj_lambda1e-3_w1e3.txt', smat)
        ismat = 1 / smat
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        if self.config.svd_method == self.config.SVD.Selection:
            ismat[self.config.svd_sel :] = 0
        self._jloco_inv = _np.dot(vtmat.T * ismat[None, :], umat.T)

    def update_fit(self):
        """."""
        # k inival and deltas
        if self.config.use_quad_families:
            self._quad_kl_inival = _LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.quad_indices_kl
            )
        else:
            self._quad_kl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KL", self.config.quad_indices_kl
                )
            )

        if self.config.use_dip_families:
            self._dip_kl_inival = _LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.dip_indices_kl
            )
        else:
            self._dip_kl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KL", self.config.dip_indices_kl
                )
            )

        if self.config.use_sext_families:
            self._sext_kl_inival = _LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.sext_indices_kl
            )
        else:
            self._sext_kl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KL", self.config.sext_indices_kl
                )
            )

        if self.config.use_offdiagonal:
            self._quad_ksl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KsL", self.config.quad_indices_ksl
                )
            )
            self._sext_ksl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KsL", self.config.sext_indices_ksl
                )
            )
            self._dip_ksl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KsL", self.config.dip_indices_ksl
                )
            )
            self._skew_quad_ksl_inival = _np.atleast_1d(
                _pyaccel.lattice.get_attribute(
                    self._model, "KsL", self.config.skew_quad_indices_ksl
                )
            )
            self._quad_ksl_deltas = _np.zeros(
                len(self.config.quad_indices_ksl)
            )
            self._sext_ksl_deltas = _np.zeros(
                len(self.config.sext_indices_ksl)
            )
            self._dip_ksl_deltas = _np.zeros(len(self.config.dip_indices_ksl))
            self._skew_quad_ksl_deltas = _np.zeros(
                len(self.config.skew_quad_indices_ksl)
            )

        self._energy_shift_inival = _np.zeros(self.config.nr_corr)
        self._energy_shift_deltas = _np.zeros(self.config.nr_corr)

        self._quad_kl_deltas = _np.zeros(len(self.config.quad_indices_kl))
        self._sext_kl_deltas = _np.zeros(len(self.config.sext_indices_kl))
        self._dip_kl_deltas = _np.zeros(len(self.config.dip_indices_kl))

        self._girders_shift_inival = _np.zeros(
            self.config.gir_indices.shape[0]
        )
        self._girders_shift_deltas = _np.zeros(
            self.config.gir_indices.shape[0]
        )

        # bpm inival and deltas
        if self._gain_bpm_inival is None:
            self._gain_bpm_inival = _np.ones(2 * self.config.nr_bpm)
        if self._roll_bpm_inival is None:
            self._roll_bpm_inival = _np.zeros(self.config.nr_bpm)
        self._gain_bpm_delta = _np.zeros(2 * self.config.nr_bpm)
        self._roll_bpm_delta = _np.zeros(self.config.nr_bpm)

        # corr inival and deltas
        if self._gain_corr_inival is None:
            self._gain_corr_inival = _np.ones(self.config.nr_corr)
        self._gain_corr_delta = _np.zeros(self.config.nr_corr)

        check_case = self._gain_bpm_inival is not None
        check_case |= self._roll_bpm_inival is not None
        check_case |= self._gain_corr_inival is not None

        if check_case:
            self._matrix = _LOCOUtils.apply_all_gain(
                matrix=self._matrix,
                gain_bpm=self._gain_bpm_inival,
                roll_bpm=self._roll_bpm_inival,
                gain_corr=self._gain_corr_inival,
            )

        self._chi = self.calc_chi()
        self._chi_init = self._chi
        print("chi_init: {0:.6f} um".format(self._chi_init))

        self._tol_delta = self.config.tolerance_delta or LOCO.DEFAULT_TOL_DELTA
        self._tol_overfit = (
            self.config.tolerance_overfit or LOCO.DEFAULT_TOL_DELTA
        )
        self._reduc_threshold = LOCO.DEFAULT_REDUC_THRESHOLD

    def _calc_residue(self):
        matrix_diff = self.config.goalmat - self._matrix
        if not self.config.use_diagonal:
            matrix_diff = _LOCOUtils.remove_diagonal(matrix_diff, 160, 120)
        matrix_diff = _LOCOUtils.apply_all_weight(
            matrix_diff,
            self.config.weight_bpm,
            self.config.weight_corr,
            self.config.weight_dispx,
            self.config.weight_dispy,
        )
        res = matrix_diff.ravel()
        if self.config.constraint_deltakl_total:
            kldeltas = []
            if self.config.fit_quadrupoles:
                kldeltas = _np.hstack((kldeltas, self._quad_kl_deltas))
            if self.config.fit_dipoles:
                kldeltas = _np.hstack((kldeltas, self._dip_kl_deltas))
            if self.config.fit_sextupoles:
                kldeltas = _np.hstack((kldeltas, self._sext_kl_deltas))
            wmat = self.config.weight_deltakl
            wmat /= self.config.deltakl_normalization
            kldeltas *= -wmat
            res = _np.hstack((res, kldeltas))
        return res

    def run_fit(self, niter=1):
        """."""
        self._chi = self._chi_init
        for _iter in range(niter):
            self._chi_history.append(self._chi)
            self._kldelta_history.append(self._quad_kl_deltas_step)
            self._ksldelta_history.append(self._skew_quad_ksl_deltas_step)
            print("iter # {}/{}".format(_iter + 1, niter))
            res = self._calc_residue()
            self._res_history.append(res)
            if self.config.inv_method == _LOCOConfig.INVERSION.Transpose:
                param_new = _np.dot(
                    self._jloco_inv, _np.dot(self._jloco.T, res)
                )
            elif self.config.inv_method == _LOCOConfig.INVERSION.Normal:
                param_new = _np.dot(self._jloco_inv, res)
            param_new = param_new.ravel()
            model_new, matrix_new = self._calc_model_matrix(param_new)
            chi_new = self.calc_chi(matrix_new)
            print("chi: {0:.6f} um".format(chi_new))
            if _np.isnan(chi_new):
                print("chi is NaN!")
                break
            if chi_new < self._chi:
                case_delta = _np.abs(chi_new - self._chi) < self._tol_delta
                case_overfit = chi_new < self._tol_overfit
                if case_delta or case_overfit:
                    if case_delta:
                        print("chi reduction is lower than delta tolerance...")
                    if case_overfit:
                        print("chi is lower than overfitting tolerance...")
                    break
                else:
                    self._update_state(model_new, matrix_new, chi_new)
                    # print('recalculating jloco...')
                    # self.update_jloco()
                    # self.update_svd()
                    if (
                        self.config.min_method
                        == _LOCOConfig.MINIMIZATION.LevenbergMarquardt
                    ):
                        if not self.config.fixed_lambda:
                            self._recalculate_inv_jloco(case="good")
            else:
                # print('recalculating jloco...')
                # self.update_jloco()
                # self.update_svd()
                if (
                    self.config.min_method
                    == _LOCOConfig.MINIMIZATION.GaussNewton
                ):
                    factor = self._try_refactor_param(param_new)
                    if factor <= self._reduc_threshold:
                        # could not converge at current iteration!
                        break
                elif (
                    self.config.min_method
                    == _LOCOConfig.MINIMIZATION.LevenbergMarquardt
                ):
                    if self.config.lambda_lm <= LOCO.DEFAULT_MIN_LAMBDA_LM:
                        break
                    if not self.config.fixed_lambda:
                        self._try_refactor_lambda(chi_new)
                    else:
                        break
                    if self.config.lambda_lm >= LOCO.DEFAULT_MAX_LAMBDA_LM:
                        break
            if self._chi < self._tol_overfit:
                print("chi is lower than overfitting tolerance...")
                break
        self._create_output_vars()
        print("Finished!")

    def _recalculate_inv_jloco(self, case="good"):
        if case == "good":
            self.config.lambda_lm /= LOCO.DEFAULT_DECR_RATE_LAMBDA_LM
        elif case == "bad":
            self.config.lambda_lm *= LOCO.DEFAULT_INCR_RATE_LAMBDA_LM
        dmat = self._lm_dmat
        matrix2invert = self._jloco.T @ self._jloco
        matrix2invert += self.config.lambda_lm * dmat.T @ dmat
        if self.config.constraint_deltakl_step:
            matrix2invert += self._deltakl_mat.T @ self._deltakl_mat
        umat, smat, vtmat = _np.linalg.svd(matrix2invert, full_matrices=False)
        ismat = 1 / smat
        if self.config.svd_method == self.config.SVD.Selection:
            ismat[self.config.svd_sel :] = 0
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        self._jloco_inv = _np.dot(vtmat.T * ismat[None, :], umat.T)

    def calc_chi(self, matrix=None):
        """."""
        if matrix is None:
            matrix = self._matrix
        dmatrix = matrix - self.config.goalmat
        # dmatrix = _LOCOUtils.apply_all_weight(dmatrix, self.config.weight_bpm, self.config.weight_corr)
        dmatrix[:, : self.config.nr_ch] *= self.config.delta_kickx_meas
        dmatrix[:, self.config.nr_ch : -1] *= self.config.delta_kicky_meas
        dmatrix[:, -1] *= self.config.delta_frequency_meas
        chi2 = _np.sum(dmatrix * dmatrix) / dmatrix.size
        return _np.sqrt(chi2) * 1e6  # m to um

    def _create_output_vars(self):
        """."""
        self.fitmodel = _dcopy(self._model)
        self.chi_history = self._chi_history
        self.bpm_gain = self._gain_bpm_inival + self._gain_bpm_delta
        self.bpm_roll = self._roll_bpm_inival + self._roll_bpm_delta
        self.corr_gain = self._gain_corr_inival + self._gain_corr_delta
        self.energy_shift = (
            self._energy_shift_inival + self._energy_shift_deltas
        )
        self.residue_history = self._res_history
        self.girder_shift = (
            self._girders_shift_inival + self._girders_shift_deltas
        )
        self.kldelta_history = self._kldelta_history

    def clear_output_vars(self):
        """."""
        self.fitmodel = None
        self.bpm_gain = None
        self.bpm_roll = None
        self.corr_gain = None
        self.energy_shift = None
        self.chi_history = []
        self.residue_history = []
        self._chi_history = []
        self._res_history = []
        self.girder_shift = []
        self.kldelta_history = []
        self._kldelta_history = []

    def _calc_model_matrix(self, param):
        """."""
        model = _dcopy(self._model)
        config = self.config
        param_dict = _LOCOUtils.param_select(config, param)
        one_knob = False
        if "dipoles_gradient" in param_dict:
            self._dip_kl_deltas += param_dict["dipoles_gradient"]
            # update local model
            if self.config.use_dip_families:
                set_dip_kldelta = _LOCOUtils.set_dipset_kldelta
            else:
                set_dip_kldelta = _LOCOUtils.set_dipmag_kldelta
            for idx, idx_set in enumerate(config.dip_indices_kl):
                set_dip_kldelta(
                    model,
                    idx_set,
                    self._dip_kl_inival[idx],
                    self._dip_kl_deltas[idx],
                )
            one_knob = True
        if "quadrupoles_gradient" in param_dict:
            # update quadrupole delta
            self._quad_kl_deltas += param_dict["quadrupoles_gradient"]
            self._quad_kl_deltas_step = param_dict["quadrupoles_gradient"]
            # update local model
            if self.config.use_quad_families:
                set_quad_kldelta = _LOCOUtils.set_quadset_kldelta
            else:
                set_quad_kldelta = _LOCOUtils.set_quadmag_kldelta
            for idx, idx_set in enumerate(config.quad_indices_kl):
                set_quad_kldelta(
                    model,
                    idx_set,
                    self._quad_kl_inival[idx],
                    self._quad_kl_deltas[idx],
                )
            one_knob = True
        if "sextupoles_gradient" in param_dict:
            # update sextupole delta
            self._sext_kl_deltas += param_dict["sextupoles_gradient"]
            # update local model
            if self.config.use_sext_families:
                set_sext_kdelta = _LOCOUtils.set_quadset_kdelta
            else:
                set_sext_kdelta = _LOCOUtils.set_quadmag_kdelta
            for idx, idx_set in enumerate(config.sext_indices):
                set_sext_kdelta(
                    model,
                    idx_set,
                    self._sext_kl_inival[idx],
                    self._sext_kl_deltas[idx],
                )
            one_knob = True
        if "quadrupoles_coupling" in param_dict:
            # update quadrupole Ks delta
            self._quad_ksl_deltas += param_dict["quadrupoles_coupling"]
            # update local model
            set_quad_ksldelta = _LOCOUtils.set_quadmag_ksldelta
            for idx, idx_set in enumerate(config.quad_indices_ksl):
                set_quad_ksldelta(
                    model,
                    idx_set,
                    self._quad_ksl_inival[idx],
                    self._quad_ksl_deltas[idx],
                )
            one_knob = True
        if "sextupoles_coupling" in param_dict:
            # update sextupole Ks delta
            self._sext_ksl_deltas += param_dict["sextupoles_coupling"]
            # update local model
            set_quad_ksldelta = _LOCOUtils.set_quadmag_ksldelta
            for idx, idx_set in enumerate(config.sext_indices_ksl):
                set_quad_ksldelta(
                    model,
                    idx_set,
                    self._sext_ksl_inival[idx],
                    self._sext_ksl_deltas[idx],
                )
            one_knob = True
        if "dipoles_coupling" in param_dict:
            # update dipoles Ks delta
            self._dip_ksl_deltas += param_dict["dipoles_coupling"]
            # update local model
            for idx, idx_set in enumerate(config.dip_indices_ksl):
                _LOCOUtils.set_dipmag_ksldelta(
                    model,
                    idx_set,
                    self._dip_ksl_inival[idx],
                    self._dip_ksl_deltas[idx],
                )
            one_knob = True
        if "dipoles_hkicks" in param_dict:
            # update dipoles hkick delta
            self._dip_hkick_deltas += param_dict["dipoles_hkick"]
            # update local model
            for idx, idx_set in enumerate(config.dip_indices_hkick):
                _LOCOUtils.set_dipmag_hkick(
                    model,
                    idx_set,
                    self._dip_hkick_inival[idx],
                    self._dip_hkick_deltas[idx],
                )
            one_knob = True
        if "skew_quadrupoles" in param_dict:
            # update skew quadrupoles
            self._skew_quad_ksl_deltas += param_dict["skew_quadrupoles"]
            self._skew_quad_ksl_deltas_step = param_dict["skew_quadrupoles"]
            # update local model
            set_quad_ksldelta = _LOCOUtils.set_quadmag_ksldelta
            for idx, idx_set in enumerate(config.skew_quad_indices_ksl):
                set_quad_ksldelta(
                    model,
                    idx_set,
                    self._skew_quad_ksl_inival[idx],
                    self._skew_quad_ksl_deltas[idx],
                )
            one_knob = True
        if "girders_shift" in param_dict:
            # update girders shift
            self._girders_shift_deltas += param_dict["girders_shift"]
            # update model
            _LOCOUtils.set_girders_long_shift(
                model,
                config.gir_indices,
                self._girders_shift_inival + self._girders_shift_deltas,
            )
            one_knob = True

        if one_knob:
            matrix = _LOCOUtils.respm_calc(
                model, config.respm, config.use_dispersion
            )
        else:
            matrix = _dcopy(self.config.matrix)

        if "energy_shift" in param_dict:
            # update energy shift
            self._energy_shift_deltas += param_dict["energy_shift"]
            matrix = _LOCOUtils.add_dispersion_to_respm(
                matrix,
                self._energy_shift_inival + self._energy_shift_deltas,
                config.measured_dispersion,
            )

        if "gain_bpm" in param_dict:
            # update gain delta
            self._gain_bpm_delta += param_dict["gain_bpm"]
            gain = self._gain_bpm_inival + self._gain_bpm_delta
            matrix = _LOCOUtils.apply_bpm_gain(matrix, gain)

        if "roll_bpm" in param_dict:
            # update roll delta
            self._roll_bpm_delta += param_dict["roll_bpm"]
            roll = self._roll_bpm_inival + self._roll_bpm_delta
            matrix = _LOCOUtils.apply_bpm_roll(matrix, roll)

        if "gain_corr" in param_dict:
            # update gain delta
            self._gain_corr_delta += param_dict["gain_corr"]
            gain = self._gain_corr_inival + self._gain_corr_delta
            matrix = _LOCOUtils.apply_corr_gain(matrix, gain)
        return model, matrix

    def _try_refactor_param(self, param_new):
        """."""
        factor = 0.5
        _iter = 1
        while factor > self._reduc_threshold:
            print("chi was increased! Trial {0:d}".format(_iter))
            print("applying {0:0.4f} %".format(100 * factor))
            model_new, matrix_new = self._calc_model_matrix(factor * param_new)
            chi_new = self.calc_chi(matrix_new)
            print("chi: {0:.6f} um".format(chi_new))
            if chi_new < self._chi:
                self._update_state(model_new, matrix_new, chi_new)
                break
            factor /= 2
            _iter += 1
        return factor

    def _try_refactor_lambda(self, chi_new):
        """."""
        _iter = 0
        while (
            chi_new > self._chi
            and self.config.lambda_lm < LOCO.DEFAULT_MAX_LAMBDA_LM
        ):
            print("chi was increased! Trial {0:d}".format(_iter))
            print("applying lambda {0:0.4e}".format(self.config.lambda_lm))
            self._recalculate_inv_jloco(case="bad")
            res = self._calc_residue()
            param_new = _np.dot(self._jloco_inv, _np.dot(self._jloco.T, res))
            param_new = param_new.ravel()
            model_new, matrix_new = self._calc_model_matrix(param_new)
            chi_new = self.calc_chi(matrix_new)
            print("chi: {0:.6f} um".format(chi_new))
            if chi_new < self._chi:
                self._update_state(model_new, matrix_new, chi_new)
                break
            _iter += 1

    def _update_state(self, model_new, matrix_new, chi_new):
        """."""
        self._model = model_new
        self._matrix = matrix_new
        self._chi = chi_new
