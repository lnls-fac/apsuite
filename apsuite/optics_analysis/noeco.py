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
        super().__init__()
        self.denergy_oeorm_calc = 1e-2  # 1 %
        self.tunex = 49.16
        self.tuney = 14.22
        self.chromx = 3.11
        self.chromy = 2.5

        self.sextfams2fit = self.SEXT_FAMS_CHROM
        self.fit_sexts = True
        self.fit_gain_bpms = True
        self.fit_coup_bpms = True
        self.fit_gain_corr = True

        self.use_diag_blocks = True
        self.use_hor_disp = True
        self.use_ver_disp = False
        self.use_offdiag_blocks = False

        self.update_jacobian_sexts = False
        self.update_jacobian_gains = True

        self.nr_bpms = 160
        self.nr_chs = 120
        self.nr_cvs = 160
        self.nr_corrs = self.nr_chs + self.nr_cvs

        self.init_gain_corr = _np.ones(self.nr_corrs)
        self.init_gain_bpms = _np.ones(2 * self.nr_bpms)  # Gx, Gy
        self.init_coup_bpms = _np.zeros(2 * self.nr_bpms)  # Cc, Cy

    @property
    def nr_sexts(self):
        """."""
        return len(self.sextfams2fit)


class NOECOFit(LeastSquaresOptimize):
    """."""

    def __init__(
        self, oeorm_goal=None, bpms_noise=None, use_thread=True, isonline=False
    ):
        """."""
        params = NOECOParams()

        self._oeorm_goal = None
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
        oeorm_ = self._select_matrix_blocks(oeorm)
        self._oeorm_goal = oeorm_
        self.merit_figure_goal = oeorm_.ravel()

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
        self.params.errorbars = self._get_errorbars(bpms_noise)

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

    def _select_matrix_blocks(self, oeorm):
        n, m = self.params.nr_bpms, self.params.nr_corrs + 1
        xx, xy, dispx, yx, yy, dispy = self.get_oeorm_blocks(oeorm)
        oeorm_ = _np.zeros((2 * n, m))
        if self.params.use_diag_blocks:
            oeorm_[:n, : self.params.nr_chs] = xx
            oeorm_[n:, self.params.nr_chs : self.params.nr_corrs] = yy
        if self.params.use_offdiag_blocks:
            oeorm_[:n, self.params.nr_chs : self.params.nr_corrs] = xy
            oeorm_[n:, : self.params.nr_chs] = yx
        if self.params.use_hor_disp:
            oeorm_[:n, -1] = dispx
        if self.params.use_ver_disp:
            oeorm_[n:, -1] = dispy
        return oeorm_.reshape((2 * n, m))

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
        return errorbar

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

        oeorm = self._select_matrix_blocks(oeorm)

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

    def calc_merit_figure(self, pos=None):
        """."""
        if pos is None:
            pos = self.get_initial_pos()
        strengths, *_ = self.parse_params_from_pos(pos)
        return self.get_oeorm(
            strengths=strengths,
            delta=self.params.denergy_oeorm_calc,
            normalize_rf=True,
            ravel=True,
        )

    def calc_residual(self, pos=None, merit_figure_goal=None):
        """."""
        if pos is None:
            pos = self.get_initial_pos()

        if merit_figure_goal is None:
            merit_figure_goal = self.oeorm_goal

        merit_figure_goal = self.apply_gains(merit_figure_goal, pos).ravel()

        return super().calc_residual(pos, merit_figure_goal)

    def calc_jacobian_sexts(self, strengths=None, step=1e-4):
        """."""
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
            ])
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
                    oeorm_corr_gains[self.params.nr_bpms :],
                    oeorm_corr_gains[: self.params.nr_bpms],
                ))
            ).T
            if self.params.errorbars is not None:
                jacobian_coup_bpms /= self.params.errorbars[:, None]

            if jacobian is None:
                jacobian = jacobian_coup_bpms
            else:
                jacobian = _np.hstack((jacobian, jacobian_coup_bpms))

        return jacobian

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
        ravel = False
        if oeorm_.ndim == 1:
            ravel = True
            oeorm_ = oeorm_.reshape((
                2 * self.params.nr_bpms,
                self.params.nr_corrs + 1,
            ))

        _, *ret = self.parse_params_from_pos(pos)
        gains_corrs, gains_bpms, coup_bpms = ret

        if corrs and gains_corrs is not None:
            oeorm_ = oeorm_ * _np.r_[gains_corrs, 1]

        if bpms and (gains_bpms is not None or coup_bpms is not None):
            Gb = self.bpms_gains_matrix(gains_bpms, coup_bpms)
            oeorm_ = Gb @ oeorm_

        return oeorm_.ravel() if ravel else oeorm_

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

    def plot_strengths(self, strengths=None, fig=None, ax=None, label=None):
        """."""
        if fig is None or ax is None:
            fig, ax = _mplt.subplots(figsize=(12, 4))
        if strengths is None:
            strengths = self.get_strengths()

        ax.plot(strengths, 'o-', mfc='none', label=label)
        tick_label = self.params.sextfams2fit
        tick_pos = list(range(len(tick_label)))
        ax.set_xticks(tick_pos, labels=tick_label, rotation=45)
        ax.set_ylabel(r'SL [m$^{-2}$]')
        ax.set_xlabel('chromatic sextupole families')
        if label:
            ax.legend()
        fig.tight_layout()
        fig.show()
        return fig, ax
