"""."""

import matplotlib.pyplot as _mplt
import numpy as _np
import pyaccel as _pa
from pyaccel.optics.miscellaneous import (
    get_chromaticities,
    get_mcf,
    get_rf_frequency
)
from pymodels import si as _si

from apsuite.optics_analysis import ChromCorr, TuneCorr
from apsuite.optimization.least_squares import (
    LeastSquaresOptimize,
    LeastSquaresParams
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


class NOECOFit(LeastSquaresOptimize):
    """."""

    def __init__(
        self,
        merit_figure_goal=None,
        jacobian=None,
        use_thread=True,
        isonline=False,
    ):
        """."""
        params = NOECOParams()
        super().__init__(
            params, merit_figure_goal, jacobian, use_thread, isonline
        )
        self._model = None
        self.famdata = None

        self._orbmat = None
        self._tunecorr = None
        self._chromcorr = None

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
        self.famdata = _si.get_family_data(self._model)
        self.create_corr_objects()

    def create_model(self):
        """."""
        self.model = _si.create_accelerator()

    def create_corr_objects(self):
        """."""
        if self.model is None:
            raise RuntimeError(
                'self.model is None. Call create_model() or set model first'
            )

        if self._orbmat is None:
            self._orbmat = OrbRespmat(
                model=self.model, acc='SI', use6dtrack=True
            )
        if self._tunecorr is None:
            self._tunecorr = TuneCorr(model=self.model, acc='SI')
        if self._chromcorr is None:
            self._chromcorr = ChromCorr(model=self.model, acc='SI')

    def adjust_model(self, tunes=None, chroms=None):
        """."""
        if tunes is None:
            tunes = self.params.tunex, self.params.tuney
        if chroms is None:
            chroms = self.params.chromx, self.params.chromy

        self._tunecorr.correct_parameters(tunes, model=self.model)
        self._chromcorr.correct_parameters(chroms, model=self.model)

    def _set_energy_offset(self, energy_offset, model=None):
        """."""
        if model is None:
            model = self.model
        freq0 = get_rf_frequency(model)
        alpha = get_mcf(model)
        df = -alpha * freq0 * energy_offset
        self._set_delta_rf_frequency(df, model)

    def _set_delta_rf_frequency(self, delta_frequency, model=None):
        if model is None:
            model = self.model
        freq0 = get_rf_frequency(model)
        print(f'Changing RF freq. by {delta_frequency:.4e}')
        self._set_rf_frequency(freq0 + delta_frequency, model)

    def _set_rf_frequency(self, frequency, model=None):
        """."""
        if model is None:
            model = self.model
        cav_idx = _si.get_family_data(model)['SRFCav']['index'][0][0]
        print(f'Setting RF freq. to {frequency:.8e}')
        model[cav_idx].frequency = frequency

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
        idempotent=True,
        delta=1e-2,
        normalize_rf=True,
        ravel=False,
    ):
        """."""
        if model is None:
            model = self.model

        if strengths is not None:
            pos0 = self.get_strengths()
            self.set_strengths(strengths)

        rf_freq0 = get_rf_frequency(self.model)
        self._set_energy_offset(energy_offset=delta, model=self.model)
        mat_pos = self._orbmat.get_respm(add_rfline=True)

        self._set_rf_frequency(frequency=rf_freq0, model=self.model)

        self._set_energy_offset(energy_offset=-delta, model=self.model)
        mat_neg = self._orbmat.get_respm(add_rfline=True)

        self._set_rf_frequency(frequency=rf_freq0, model=self.model)

        oeorm = (mat_pos - mat_neg) / 2 / delta

        if normalize_rf:
            oeorm[:, -1] *= 1e6  # [um/Hz]

        if ravel:
            oeorm = _np.ravel(oeorm)

        if strengths is not None and idempotent:
            self.set_strengths(pos0)

        return oeorm

    def get_chroms(self, strengths=None, idempotent=True):
        """."""
        if strengths is not None:
            pos0 = self.get_strengths()
            self.set_strengths(strengths)

        chroms = get_chromaticities(self.model)

        if strengths is not None and idempotent:
            self.set_strengths(pos0)

        return _np.array(chroms)

    def calc_merit_figure(self, pos):
        """."""
        oeorm = self.get_oeorm(
            strengths=pos,
            idempotent=True,
            delta=self.params.denergy_oeorm_calc,
            normalize_rf=True,
            ravel=True,
        )
        return oeorm

    def get_optimization_pos(self):
        """."""
        pos = self.get_strengths()
        # other params will go here as well (gains, rolls, etc)
        return pos

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
