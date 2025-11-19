"""."""

import matplotlib.pyplot as _mplt
import numpy as _np
import pyaccel as _pa
from pyaccel.optics.miscellaneous import (
    get_chromaticities as _get_chromaticities,
    get_mcf as _get_mcf,
    get_rf_frequency as _get_rf_frequency,
)
from pymodels import si as _si
from scipy.linalg import block_diag

from apsuite.optics_analysis import ChromCorr, TuneCorr
from apsuite.optimization.least_squares import (
    LeastSquaresOptimize,
    LeastSquaresParams,
)
from apsuite.orbcorr import OrbRespmat


class NOECOParams(LeastSquaresParams):
    """."""

    SEXT_FAMS = (
        'SDA0',
        'SDB0',
        'SDP0',
        'SFA0',
        'SFB0',
        'SFP0',
        'SDA1',
        'SDB1',
        'SDP1',
        'SDA2',
        'SDB2',
        'SDP2',
        'SDA3',
        'SDB3',
        'SDP3',
        'SFA1',
        'SFB1',
        'SFP1',
        'SFA2',
        'SFB2',
        'SFP2',
    )
    SEXT_FAMS_ACHROM = SEXT_FAMS[:6]
    SEXT_FAMS_CHROM = SEXT_FAMS[6:]

    def __init__(self):
        """."""
        self.denergy_oeorm_calc = 1e-2  # 1 %
        self.tunex = 49.16
        self.tuney = 14.22
        self.chromx = 3.11
        self.chromy = 2.66

        self.sextfams2fit = self.SEXT_FAMS_CHROM
        self.fit_sexts = True
        self.fit_gain_bpms = True
        self.fit_coup_bpms = True
        self.fit_gain_corr = True

        self._use_diag_blocks = None
        self._use_hor_disp = None
        self._use_ver_disp = None
        self._use_offdiag_blocks = None
        self.oeorm_mask = None

        self.update_jacobian_sexts = False
        self.update_jacobian_gains = True

        self.nr_bpms = 160
        self.nr_chs = 120
        self.nr_cvs = 160
        self.nr_corrs = self.nr_chs + self.nr_cvs

        self.init_gain_corr = _np.ones(self.nr_corrs)
        self.init_gain_bpms = _np.ones(2 * self.nr_bpms)  # Gx, Gy
        self.init_coup_bpms = _np.zeros(2 * self.nr_bpms)  # Cc, Cy
        self.chrom_weights = _np.array([1, 1])
        self.disp_weight = 1
        super().__init__()

        self.use_diag_blocks = True
        self.use_offdiag_blocks = False
        self.use_hor_disp = True
        self.use_ver_disp = False
        self.use_chroms = False

    @property
    def nr_sexts(self):
        """."""
        return len(self.sextfams2fit)

    @property
    def use_diag_blocks(self):
        """."""
        return self._use_diag_blocks

    @use_diag_blocks.setter
    def use_diag_blocks(self, value):
        """."""
        self._use_diag_blocks = value
        self._update_oeorm_mask()

    @property
    def use_offdiag_blocks(self):
        """."""
        return self._use_offdiag_blocks

    @use_offdiag_blocks.setter
    def use_offdiag_blocks(self, value):
        """."""
        self._use_offdiag_blocks = value
        if not value:
            self.fit_coup_bpms = value  # don't fit coupling
        self._update_oeorm_mask()

    @property
    def use_hor_disp(self):
        """."""
        return self._use_hor_disp

    @use_hor_disp.setter
    def use_hor_disp(self, value):
        """."""
        self._use_hor_disp = value
        self._update_oeorm_mask()

    @property
    def use_ver_disp(self):
        """."""
        return self._use_ver_disp

    @use_ver_disp.setter
    def use_ver_disp(self, value):
        """."""
        self._use_ver_disp = value
        self._update_oeorm_mask()

    def _update_oeorm_mask(self):
        """."""
        self.oeorm_mask = self._get_oeorm_mask()

    def _get_oeorm_mask(self):
        """."""
        n, m = self.nr_bpms, self.nr_corrs + 1
        mask = _np.zeros((2 * n, m), dtype=bool)

        if self.use_diag_blocks:
            mask[:n, : self.nr_chs] = True
            mask[n:, self.nr_chs : self.nr_corrs] = True

        if self.use_offdiag_blocks:
            mask[:n, self.nr_chs : self.nr_corrs] = True
            mask[n:, : self.nr_chs] = True

        if self.use_hor_disp:
            mask[:n, -1] = True

        if self.use_ver_disp:
            mask[n:, -1] = True

        return mask


class NOECOFit(LeastSquaresOptimize):
    """."""

    def __init__(
        self, oeorm_goal=None, bpms_noise=None, use_thread=True, isonline=False
    ):
        """."""
        params = NOECOParams()

        self._oeorm_goal = None
        self._chroms_goal = _np.array([params.chromx, params.chromy])
        self._bpms_noise = None
        self._model = None
        self.famdata = None
        self._orbmat = None
        self._tunecorr = None
        self._chromcorr = None
        self.jacobian_sexts = None
        self.jacobian_gains = None

        super().__init__(
            params=params,
            merit_figure_goal=None,
            use_thread=use_thread,
            isonline=isonline,
        )

        self.oeorm_goal = oeorm_goal
        self.bpms_noise = bpms_noise

    @property
    def model(self):
        """."""
        return self._model

    @model.setter
    def model(self, model):
        """."""
        if not isinstance(model, _pa.accelerator.Accelerator):
            raise TypeError('model must be pyaccel.accelerator.Accelerator')
        self._model = model
        self._model.radiation_on = 1
        self._model.cavity_on = True
        self.famdata = _si.get_family_data(self._model)
        self.create_corr_objects()

    @property
    def oeorm_goal(self):
        """."""
        return self._oeorm_goal

    @oeorm_goal.setter
    def oeorm_goal(self, oeorm):
        """."""
        if oeorm is None:
            raise ValueError('oeorm_goal must not be None.')
        self._oeorm_goal = oeorm

    @property
    def chroms_goal(self):
        """."""
        return self._chroms_goal

    @chroms_goal.setter
    def chroms_goal(self, chroms):
        """."""
        if chroms is None:
            raise ValueError('chroms_goal must not be None.')
        self._chroms_goal = chroms

    @property
    def merit_figure_goal(self):
        """."""
        figgoal = self.oeorm_goal.ravel()[self.params.oeorm_mask.ravel()]
        if self.params.use_chroms:
            chroms = (self.params.chrom_weights * self.chroms_goal).ravel()
            figgoal = _np.concatenate((figgoal, chroms))
        return figgoal

    @merit_figure_goal.setter
    def merit_figure_goal(self, merit_figure):
        """."""
        pass  # compatibility only

    @property
    def bpms_noise(self):
        """."""
        return self._bpms_noise

    @bpms_noise.setter
    def bpms_noise(self, bpms_noise):
        """."""
        if bpms_noise is None:
            return
        self._bpms_noise = bpms_noise
        if bpms_noise.squeeze().ndim == 1:
            self.params.errorbars = self._get_errorbars(bpms_noise)
        else:
            self.params.errorbars = bpms_noise[self.params.oeorm_mask].ravel()

    def create_model(self):
        """."""
        self.model = _si.create_accelerator()

    def create_corr_objects(self):
        """."""
        if self._model is None:
            raise RuntimeError(
                'self.model is None. Call create_model() or set model first'
            )

        self._orbmat = OrbRespmat(model=self._model, acc='SI', use6dtrack=True)
        self._tunecorr = TuneCorr(model=self._model, acc='SI')
        self._chromcorr = ChromCorr(model=self._model, acc='SI')

    def adjust_model(self, tunes=None, chroms=None):
        """."""
        if tunes is None:
            tunes = self.params.tunex, self.params.tuney
        if chroms is None:
            chroms = self.params.chromx, self.params.chromy

        cav = self._model.cavity_on
        rad = self._model.radiation_on
        self._model.cavity_on = False
        self._model.radiation_on = False
        self._tunecorr.correct_parameters(tunes, model=self._model)
        self._chromcorr.correct_parameters(chroms, model=self._model)
        self._model.cavity_on = cav
        self._model.radiation_on = rad

    def _set_delta_energy_offset(self, energy_offset, model=None):
        """."""
        if model is None:
            model = self.model
        freq0 = _get_rf_frequency(model)
        alpha = _get_mcf(model)
        df = -alpha * freq0 * energy_offset
        self._set_delta_rf_frequency(df, model)

    def _set_delta_rf_frequency(self, delta_frequency, model=None):
        if model is None:
            model = self.model
        freq0 = _get_rf_frequency(model)
        # print(f'Changing RF freq. by {delta_frequency:.4e}')
        self._set_rf_frequency(freq0 + delta_frequency, model)

    def _set_rf_frequency(self, frequency, model=None):
        """."""
        if model is None:
            model = self.model
        cav_idx = _si.get_family_data(model)['SRFCav']['index'][0][0]
        # print(f'Setting RF freq. to {frequency:.8e}')
        model[cav_idx].frequency = frequency

    def _initialization(self):
        if self._model is None:
            print('Cannot start optimization without machine model.')
            return False

        if self._oeorm_goal is None:
            print('Cannot start optimization without oeorm_goal')
            return False

        if not len(self.params.initial_position):
            print('No initial position specified. Using default pos.')
            pos = self.get_initial_pos()
            if not len(pos):
                print('No parameters to fit!')
                return False
            self.params.initial_position = pos
        return True

    def _get_errorbars(self, bpms_noise):
        n, m = self.params.nr_bpms, self.params.nr_corrs + 1
        errorbar = _np.ones(2 * n * m)
        for i, sigma in enumerate(bpms_noise, start=0):
            errorbar[i * m : (i + 1) * m] = _np.repeat(sigma, m)
        return errorbar[self.params.oeorm_mask.ravel()]

    def get_strengths(self):
        """."""
        strengths = list()
        for fam in self.params.sextfams2fit:
            idc = _pa.lattice.flatten(self.famdata[fam]['index'])[0]
            strengths.append(self.model[idc].SL)
        return _np.array(strengths)

    def set_strengths(self, strengths):
        """."""
        for stren, fam in zip(strengths, self.params.sextfams2fit):
            idcs = _pa.lattice.flatten(self.famdata[fam]['index'])
            _pa.lattice.set_attribute(self.model, 'SL', idcs, stren)

    def get_oeorm(
        self,
        model=None,
        strengths=None,
        delta=1e-2,
        normalize_rf=True,
        ravel=False,
    ):
        """."""
        if model is None:
            model = self.model

        if strengths is not None:
            strengths0 = self.get_strengths()
            self.set_strengths(strengths)

        rf_freq0 = _get_rf_frequency(model)
        self._set_delta_energy_offset(energy_offset=delta, model=model)
        # TODO: model not being used here!
        mat_pos = self._orbmat.get_respm(add_rfline=True)

        self._set_rf_frequency(frequency=rf_freq0, model=model)

        self._set_delta_energy_offset(energy_offset=-delta, model=model)
        # TODO: model not being used here!
        mat_neg = self._orbmat.get_respm(add_rfline=True)

        self._set_rf_frequency(frequency=rf_freq0, model=model)

        oeorm = (mat_pos - mat_neg) / 2 / delta

        if normalize_rf:
            oeorm[:, -1] *= 1e6  # [um/Hz]

        if ravel:
            oeorm = _np.ravel(oeorm)

        if strengths is not None:
            self.set_strengths(strengths0)

        return oeorm

    def get_chroms(self, strengths=None):
        """."""
        if strengths is not None:
            strengths0 = self.get_strengths()
            self.set_strengths(strengths)

        chroms = _get_chromaticities(self.model)

        if strengths is not None:
            self.set_strengths(strengths0)

        return _np.array(chroms)

    def get_chrom_optics(self, strengths=None):
        """."""
        # NOTE: get this to pyaccel maybe?
        if strengths is not None:
            strengths0 = self.get_strengths()
            self.set_strengths(strengths)

        model = self.model

        cav = model.cavity_on
        rad = model.radiation_on
        model.cavity_on = False
        model.radiation_on = False
        bpms_idcs = _pa.lattice.flatten(self.famdata['BPM']['index'])

        delta = 1e-3
        twissp, *_ = _pa.optics.calc_twiss(
            model, indices=bpms_idcs, energy_offset=delta / 2
        )
        twissn, *_ = _pa.optics.calc_twiss(
            model, indices=bpms_idcs, energy_offset=-delta / 2
        )

        model.cavity_on = cav
        model.radiation_on = rad

        betax = (twissp.betax - twissn.betax) / delta
        betay = (twissp.betay - twissn.betay) / delta
        mux = (twissp.mux - twissn.mux) / delta
        muy = (twissp.muy - twissn.muy) / delta
        etax = (twissp.etax - twissn.etax) / delta

        if strengths is not None:
            self.set_strengths(strengths0)

        return betax, betay, mux, muy, etax

    def calc_merit_figure(self, pos=None):
        """."""
        if pos is None:
            pos = self.get_initial_pos()
        strengths, *_ = self.parse_params_from_pos(pos)
        merit_fig = self.get_oeorm(
            strengths=strengths,
            delta=self.params.denergy_oeorm_calc,
            normalize_rf=True,
            ravel=False,
        )

        if self.params.disp_weight:
            merit_fig[:, -1] *= self.params.disp_weight
        merit_fig = merit_fig[self.params.oeorm_mask].ravel()

        if self.params.use_chroms:
            chroms = self.get_chroms(strengths)
            chroms *= self.params.chrom_weights
            merit_fig = _np.concatenate((merit_fig, chroms.ravel()))

        return merit_fig

    def calc_residual(self, pos=None, merit_figure_goal=None):
        """."""
        if pos is None:
            pos = self.get_initial_pos()

        if merit_figure_goal is None:
            merit_figure_goal = self.oeorm_goal

        if merit_figure_goal.ndim == 1:
            merit_figure_goal = self.reshape_merit_fig(merit_figure_goal)

        merit_figure_goal = self.apply_gains(merit_figure_goal, pos)

        if self.params.disp_weight:
            merit_figure_goal[:, -1] *= self.params.disp_weight

        merit_figure_goal = merit_figure_goal[self.params.oeorm_mask].ravel()

        if self.params.use_chroms:
            merit_figure_goal = _np.concatenate((
                merit_figure_goal,
                (self.params.chrom_weights * self.chroms_goal).ravel(),
            ))

        return super().calc_residual(pos, merit_figure_goal)

    def reshape_merit_fig(self, merit_fig):
        """."""
        merit_fig = merit_fig.reshape((
            2 * self.params.nr_bpms,
            self.params.nr_corrs + 1,
        ))
        return merit_fig

    def calc_jacobian_sexts(self, strengths=None, step=1e-4):
        """."""
        # TODO: bug when calculating sextupole jacobian
        # with errorbars & block selections
        # array shape issue
        if strengths is None:
            strengths, *_ = self.parse_params_from_pos(self.get_default_pos())
        return super().calc_jacobian(strengths, step)

    def calc_jacobian_gains(self, pos=None):
        """."""
        if pos is None:
            pos = self.get_default_pos()

        oeorm = self.oeorm_goal
        jacobian = None

        if self.params.fit_gain_corr:
            # apply BPMs gains and calculate corrs gains jacobian
            oeorm_bpms_gains = self.apply_gains(
                oeorm, pos, bpms=True, corrs=False
            )
            jacobian_gains_corrs = _np.vstack([
                _np.diag(m) for m in oeorm_bpms_gains
            ])[self.params.oeorm_mask.ravel()]

            if self.params.errorbars is not None:
                jacobian_gains_corrs /= self.params.errorbars[:, None]
            jacobian = jacobian_gains_corrs[:, : self.params.nr_corrs]

        oeorm_corr_gains = self.apply_gains(oeorm, pos, bpms=False, corrs=True)
        # BPM gains Jacobian jth col = measured OEORM jth line, padded with
        # zeros
        # J_gains = [[oeorm_corr_gains[0, :].T, zeros, ...      zeros        ],
        #            [zeros, oeorm_corr_gains[1, :].T, ...      zeros        ],
        #            [zeros, zeros, oeorm_corr_gains[2, :].T    zeros        ],
        #            [  .      .            .                     .          ],
        #            [  .      .            .                     .          ],
        #            [zeros,  zeros,       ...    oeorm_corr_gains[320, :].T]]
        # this is implemented below
        if self.params.fit_gain_bpms:
            jacobian_gains_bpms = block_diag(*oeorm_corr_gains).T
            jacobian_gains_bpms = jacobian_gains_bpms[
                self.params.oeorm_mask.ravel()
            ]
            if self.params.errorbars is not None:
                jacobian_gains_bpms /= self.params.errorbars[:, None]

            if jacobian is None:
                jacobian = jacobian_gains_bpms
            else:
                jacobian = _np.hstack((jacobian, jacobian_gains_bpms))

        # couplings Jacobian jth col = jth line of OEORM with swapped upper and
        # lower blocks. Eg:
        #       OEORM =      [[Mxx, Mxy], [Myx, Myy]],
        #       OEORM_swap = [[Myx, Myy], [Mxx, Mxy]]
        # implementeded below
        if self.params.fit_coup_bpms:
            jacobian_coup_bpms = block_diag(
                *_np.vstack((
                    oeorm_corr_gains[self.params.nr_bpms :],  # TODO: coups
                    oeorm_corr_gains[: self.params.nr_bpms],  # and no disps
                ))
            ).T
            jacobian_coup_bpms = jacobian_gains_corrs[
                self.params.oeorm_mask.ravel()
            ]
            if self.params.errorbars is not None:
                jacobian_coup_bpms /= self.params.errorbars[:, None]

            if jacobian is None:
                jacobian = jacobian_coup_bpms
            else:
                jacobian = _np.hstack((jacobian, jacobian_coup_bpms))

        if self.params.use_chroms:
            jacobian = _np.vstack((
                jacobian,
                _np.zeros((2, jacobian.shape[-1])),
            ))

        return -jacobian  # oposite sign convention for numeric & analytic jacs

    def calc_jacobian(self, pos=None, step=1e-4):
        """."""
        if pos is None:
            pos = self.get_default_pos()

        jacobians = []

        # --- sexts ---
        if self.params.fit_sexts:
            if self.jacobian_sexts is not None:
                jacobian_sexts = self.jacobian_sexts

            update_sexts = self.params.update_jacobian_sexts
            if self.jacobian_sexts is None or update_sexts:
                strengths, *_ = self.parse_params_from_pos(pos)
                jacobian_sexts = self.calc_jacobian_sexts(strengths, step)

            jacobians.append(jacobian_sexts)

        # --- gains ---
        fit_gains = (
            self.params.fit_gain_bpms
            or self.params.fit_coup_bpms
            or self.params.fit_gain_corr
        )

        if fit_gains:
            if self.jacobian_gains is not None:
                jacobian_gains = self.jacobian_gains

            update_gains = self.params.update_jacobian_gains
            if self.jacobian_gains is None or update_gains:
                jacobian_gains = self.calc_jacobian_gains(pos)

            jacobians.append(jacobian_gains)

        jacobian = None
        if jacobians:
            jacobian = _np.hstack(jacobians)

        return jacobian

    def get_oeorm_blocks(self, oeorm):
        """."""
        nr_bpms = self.params.nr_bpms
        nr_ch = self.params.nr_chs
        nr_cv = self.params.nr_cvs

        xx = oeorm[:nr_bpms, :nr_ch]
        xy = oeorm[:nr_bpms, nr_ch : nr_ch + nr_cv]
        dispx = oeorm[:nr_bpms, -1]
        yx = oeorm[nr_bpms:, :nr_ch]
        yy = oeorm[nr_bpms:, nr_ch : nr_ch + nr_cv]
        dispy = oeorm[nr_bpms:, -1]

        return xx, xy, dispx, yx, yy, dispy

    def apply_gains(self, oeorm, pos=None, bpms=True, corrs=True):
        """."""
        if pos is None:
            pos = self.get_default_pos()

        oeorm_ = oeorm.copy()

        _, *ret = self.parse_params_from_pos(pos)
        gains_corrs, gains_bpms, coup_bpms = ret

        if corrs and gains_corrs is not None:
            oeorm_ = oeorm_ * _np.r_[gains_corrs, 1]

        if bpms and (gains_bpms is not None or coup_bpms is not None):
            Gb = self.bpms_gains_matrix(gains_bpms, coup_bpms)
            oeorm_ = Gb @ oeorm_

        return oeorm_

    def bpms_gains_matrix(self, gains, coups):
        """."""
        if gains is None:
            gains = _np.ones(2 * self.params.nr_bpms)
        if coups is None:
            coups = _np.zeros(2 * self.params.nr_bpms)

        gx = _np.diag(gains[: self.params.nr_bpms])
        gy = _np.diag(gains[self.params.nr_bpms :])
        cx = _np.diag(coups[: self.params.nr_bpms])
        cy = _np.diag(coups[self.params.nr_bpms :])
        return _np.block([[gx, cx], [cy, gy]])

    def get_default_pos(self):
        """."""
        pos = (
            self.params.initial_position
            if len(self.params.initial_position)
            else self.get_initial_pos()
        )
        return pos

    def get_initial_pos(self):
        """."""
        pos = []
        if self.params.fit_sexts:
            pos = _np.r_[pos, self.get_strengths()]
        if self.params.fit_gain_corr:
            pos = _np.r_[pos, self.params.init_gain_corr]
        if self.params.fit_gain_bpms:
            pos = _np.r_[pos, self.params.init_gain_bpms]
        if self.params.fit_coup_bpms:
            pos = _np.r_[pos, self.params.init_coup_bpms]
        return pos

    def parse_params_from_pos(self, pos):
        """."""
        sexts = None
        gains_corrs = None
        gains_bpms = None
        coup_bpms = None
        n = 0
        if self.params.fit_sexts:
            n = self.params.nr_sexts
            sexts = pos[:n]
        if self.params.fit_gain_corr:
            gains_corrs = pos[n : n + self.params.nr_corrs]
            n += self.params.nr_corrs
        if self.params.fit_gain_bpms:
            gains_bpms = pos[n : n + 2 * self.params.nr_bpms]
            n += 2 * self.params.nr_bpms
        if self.params.fit_coup_bpms:
            coup_bpms = pos[n : n + 2 * self.params.nr_bpms]
        return sexts, gains_corrs, gains_bpms, coup_bpms

    def plot_strengths(
        self,
        strengths=None,
        fig=None,
        ax=None,
        label=None,
        diff_from_init=False,
        percentual_diff=True,
    ):
        """."""
        if fig is None or ax is None:
            fig, ax = _mplt.subplots(figsize=(12, 4))
        if strengths is None:
            strengths = self.get_strengths()

        strengths_ = strengths.copy()

        if diff_from_init:
            str0, *_ = self.parse_params_from_pos(self.params.initial_position)
            strengths_ -= str0
            if percentual_diff:
                strengths_ /= str0 / 100

        ax.plot(strengths_, 'o-', mfc='none', label=label)
        tick_label = self.params.sextfams2fit
        tick_pos = list(range(len(tick_label)))
        ax.set_xticks(tick_pos, labels=tick_label, rotation=45)
        line_label = 'SL'
        if diff_from_init:
            line_label = 'delta ' + line_label
        if diff_from_init and percentual_diff:
            line_label = 'percentual ' + line_label + ' [\%]'
        else:
            line_label = line_label + r' [m$^{-2}$]'
        ax.set_ylabel(line_label)
        ax.set_xlabel('chromatic sextupole families')
        if label:
            ax.legend()
        fig.tight_layout()
        fig.show()
        return fig, ax

    def plot_matrix_diff(self, mat1=None, mat2=None, residual=None, title=''):
        """."""
        if mat1 is not None and mat2 is not None:
            residual = _np.abs(mat1 - mat2)

        if residual.ndim == 1:
            residual.reshape(2 * self.params.nr_bpms, self.params.nr_corrs + 1)

        x = _np.arange(residual.shape[1])  # colunas
        y = _np.arange(residual.shape[0])  # linhas
        X, Y = _np.meshgrid(x, y)

        fig = _mplt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, _np.abs(residual), cmap='viridis')

        ax.set_xlabel('corrs. index')
        ax.set_ylabel('BPMs index')
        ax.set_zlabel('OEORM abs. diff [m/rad]')

        _mplt.show()

        return fig, ax

    def plot_chrom_optics(
        self,
        betax=None,
        betay=None,
        mux=None,
        muy=None,
        etax=None,
        fig=None,
        axes=None,
        title='',
        diff_from_init=False,
    ):
        """."""
        if fig is None:
            fig, axes = _mplt.subplots(3, 1, sharex=True, figsize=(12, 10))

        if (betax, betay, mux, muy, etax) == (None, None, None, None, None):
            betax, betay, mux, muy, etax = self.get_chrom_optics()

        betax_ = betax.copy()
        betay_ = betay.copy()
        mux_ = mux.copy()
        muy_ = muy.copy()
        etax_ = etax.copy()

        if diff_from_init:
            # plot deviation from initial strengths optics
            str0, *_ = self.parse_params_from_pos(self.params.initial_position)
            betax0, betay0, mux0, muy0, etax0 = self.get_chrom_optics(str0)
            betax_ -= betax0
            betay_ -= betay0
            mux_ -= mux0
            muy_ -= muy0
            etax_ -= etax0

        ax0 = axes[0]
        ax0.set_title('chromatic beta')
        ax0.set_ylabel(r'$\beta_x^{(1)}$ [m]', color='blue')
        ax0.plot(betax_, color='blue', label='hor')
        ax0.tick_params(axis='y', labelcolor='blue')

        ax0tw = ax0.twinx()
        ax0tw.set_ylabel(r'$\beta_y^{(1)}$ [m]', color='red')
        ax0tw.plot(betay_, color='red', label=r'$\beta_y$')
        ax0tw.tick_params(axis='y', labelcolor='red')

        axes[1].set_title('2nd-order dispersion')
        axes[1].set_ylabel(r'$\eta^{(2)}$ [cm]')
        axes[1].plot(etax_ * 100, color='green')

        ax1 = axes[2]
        ax1.set_title('chromatic phase advance')
        ax1.set_xlabel('BPMs indices')
        ax1.set_ylabel(r'$\phi_x^{(1)}$ [rad]', color='blue')
        ax1.plot(mux_, color='blue', label=r'$\beta_y$')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax1tw = ax1.twinx()
        ax1tw.set_ylabel(r'$\phi_y^{(1)}$ [rad]', color='red')
        ax1tw.plot(muy_, color='red', label=r'$\mu_y$')
        ax1tw.tick_params(axis='y', labelcolor='red')

        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.show()

        return fig, axes
