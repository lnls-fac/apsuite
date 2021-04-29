"""TbT Simulation Annealing."""

import numpy as _np

from ..optimization import SimulAnneal as _SimulAnneal
from .calctraj import calc_traj_chrom as _calc_traj_chrom


class TbTSimulAnneal(_SimulAnneal):
    """."""

    class TYPES:
        """."""

        CHROMX = 'chromx'
        CHROMY = 'chromy'
        KXX = 'kxx'  # NOTE: implementation not finished

    def __init__(self, tbtanalysis, fit_type=TYPES.CHROMX, use_thread=False):
        """."""
        self._tbtanalysis = tbtanalysis
        self._fit_type = fit_type

        super().__init__(save=False, use_thread=use_thread)

    @property
    def fit_type(self):
        """."""
        return self._fit_type

    def initialization(self):
        """."""

    def calc_obj_fun(self):
        """."""
        tbt = self._tbtanalysis
        types = TbTSimulAnneal.TYPES
        if self._fit_type == types.CHROMX:
            params = [
                tbt.tunes_frac,
                tbt.tunex_frac,
                tbt.chromx,
                tbt.espread,
                tbt.rx0,
                tbt.mux
            ]
            args = (tbt.idx_turn_start, tbt.idx_turn_stop, tbt.rx_offset)
            traj_fit = _calc_traj_chrom(params, *args)
        elif self._fit_type == types.CHROMY:
            params = [
                tbt.tunes_frac,
                tbt.tuney_frac,
                tbt.chromy,
                tbt.espread,
                tbt.ry0,
                tbt.muy
            ]
            args = (tbt.idx_turn_start, tbt.idx_turn_stop, tbt.ry_offset)
            traj_fit = _calc_traj_chrom(params, *args)
        elif self._fit_type == TbTSimulAnneal.TYPES.KXX:
            parms = {
                'offset': tbt.rx_offset,
                'turn_start': tbt.idx_turn_start,
                'turn_stop': tbt.idx_turn_stop,
                'tunex_frac': self.position[0],
                'rxk': self.position[1],
                'kxx_decoh': self.position[2],
                'kxx_ratio': self.position[3],
            }
            traj_fit = TbTSimulAnneal._calc_traj_kxx(**parms)
        traj_mea = tbt.select_get_traj(
            idx_kick=tbt.idx_kick, idx_bpm=tbt.idx_bpm,
            idx_turn_start=tbt.idx_turn_start,
            idx_turn_stop=tbt.idx_turn_stop)
        traj_res = traj_mea - traj_fit
        residue = _np.sqrt(_np.sum(traj_res**2)/len(traj_res))
        return residue

    def calc_trajx(self, **parms):
        """."""
        types = TbTSimulAnneal.TYPES
        if self._fit_type in (types.CHROMX, types.CHROMY):
            _parms = [
                parms['tunes_frac'], parms['tune_frac'],
                parms['chrom'], parms['espread'],
                parms['r0'], parms['mu'],
            ]
            _args = (parms['offset'], )
            return _calc_traj_chrom(parms, *_args)
        elif self._fit_type == TbTSimulAnneal.TYPES.KXX:
            return TbTSimulAnneal._calc_traj_kxx(**parms)

    def start(self, print_flag=True):
        """."""
        super().start(print_flag=print_flag)

    @staticmethod
    def _calc_traj_kxx(**parms):
        """BPM averaging due to horizontal tune shift with amplitude.

        nux ~ nux0 + kxx * amplitude_x
        See Laurent Nadolski Thesis, Chapter 4, pg. 123, Eq. 4.28
        """
        rx_offset = parms['rx_offset']
        nr_turns = parms['nr_turns']
        tunex_frac = parms['tunex_frac']
        rxk = parms['rxk']
        kxx_decoh = parms['kxx_decoh']
        kxx_ratio = parms['kxx_ratio']

        turn = _np.arange(0, nr_turns)
        theta = turn / kxx_decoh
        invftheta = 1 / (1 + theta**2)
        sinf = _np.sin(
            2*_np.pi*turn*tunex_frac + 2 * _np.arctan(theta) +
            0.5 * kxx_ratio * theta * invftheta)
        funcx = _np.exp(- 0.5 * kxx_ratio * theta**2 * invftheta) * invftheta
        trajx = - rxk * funcx * sinf + rx_offset

        return trajx
