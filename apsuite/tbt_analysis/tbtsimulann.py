"""TbT Simulation Annealing."""

import numpy as _np

from apsuite.optimization import SimulAnneal as _SimulAnneal


class TbTSimulAnneal(_SimulAnneal):
    """."""

    class TYPES:
        """."""

        CHROMX = 'chromx'
        CHROMY = 'chromy'
        KXX = 'kxx'  # NOTE: implementation not finished

    def __init__(self, tbtfit, fit_type=TYPES.CHROMX, use_thread=False):
        """."""
        self._tbtfit = tbtfit
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
        tbt = self._tbtfit
        if self._fit_type == TbTSimulAnneal.TYPES.CHROMX:
            parms = {
                'rx_offset': tbt.rx_offset,
                'select_idx_turn_start': tbt.select_idx_turn_start,
                'select_idx_turn_stop': tbt.select_idx_turn_stop,
                'tunes_frac': self.position[0],
                'tunex_frac': self.position[1],
                'chromx_decoh': self.position[2],
                'rx0': self.position[3],
                'mux': self.position[4],
            }
            trajx_fit = TbTSimulAnneal._calc_trajx_chromx(**parms)
        elif self._fit_type == TbTSimulAnneal.TYPES.KXX:
            parms = {
                'rx_offset': tbt.rx_offset,
                'turn_start': tbt.idx_turn_start,
                'turn_stop': tbt.idx_turn_stop,
                'tunex_frac': self.position[0],
                'rxk': self.position[1],
                'kxx_decoh': self.position[2],
                'kxx_ratio': self.position[3],
            }
            trajx_fit = TbTSimulAnneal._calc_trajx_kxx(**parms)
        trajx_mea = tbt.select_get_traj(
            select_idx_kick=tbt.select_idx_kick,
            select_idx_bpm=tbt.select_idx_bpm,
            select_idx_turn_start=tbt.select_idx_turn_start,
            select_idx_turn_stop=tbt.select_idx_turn_stop)
        trajx_res = trajx_mea - trajx_fit
        residue = _np.sqrt(_np.sum(trajx_res**2)/len(trajx_res))
        return residue

    def calc_trajx(self, **parms):
        """."""
        if self._fit_type == TbTSimulAnneal.TYPES.CHROMX:
            return TbTSimulAnneal._calc_trajx_chromx(**parms)
        elif self._fit_type == TbTSimulAnneal.TYPES.KXX:
            return TbTSimulAnneal._calc_trajx_kxx(**parms)

    def calc_trajy(self, **parms):
        """."""
        if self._fit_type == TbTSimulAnneal.TYPES.CHROMX:
            return TbTSimulAnneal._calc_trajy_chromy(**parms)
        elif self._fit_type == TbTSimulAnneal.TYPES.KXX:
            return TbTSimulAnneal._calc_trajx_kxx(**parms)

    def start(self, print_flag=True):
        """."""
        super().start(print_flag=print_flag)

    @staticmethod
    def _calc_trajx_chromx(**parms):
        """BPM averaging due to longitudinal dynamics decoherence.

        nux ~ nux0 + chromx * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        rx_offset = parms['rx_offset']
        idx_turn_start = parms['select_idx_turn_start']
        idx_turn_stop = parms['select_idx_turn_stop']
        tunes_frac = parms['tunes_frac']
        tunex_frac = parms['tunex_frac']
        chromx_decoh = parms['chromx_decoh']
        rx0 = parms['rx0']
        mux = parms['mux']

        turn = _np.arange(idx_turn_start, idx_turn_stop)
        cos = _np.cos(2 * _np.pi * tunex_frac * turn + mux)
        alp = chromx_decoh * _np.sin(_np.pi * tunes_frac * turn)
        exp = _np.exp(-alp**2/2.0)
        trajx = rx0 * exp * cos + rx_offset
        return trajx

    @staticmethod
    def _calc_trajy_chromy(**parms):
        """BPM averaging due to longitudinal dynamics decoherence.

        nuy ~ nuy0 + chromy * delta_energy
        See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
        """
        ry_offset = parms['ry_offset']
        idx_turn_start = parms['select_idx_turn_start']
        idx_turn_stop = parms['select_idx_turn_stop']
        tunes_frac = parms['tunes_frac']
        tuney_frac = parms['tuney_frac']
        chromy_decoh = parms['chromy_decoh']
        ry0 = parms['ry0']
        muy = parms['muy']

        turn = _np.arange(idx_turn_start, idx_turn_stop)
        cos = _np.cos(2 * _np.pi * tuney_frac * turn + muy)
        alp = chromy_decoh * _np.sin(_np.pi * tunes_frac * turn)
        exp = _np.exp(-alp**2/2.0)
        trajy = ry0 * exp * cos + ry_offset
        return trajy

    @staticmethod
    def _calc_trajx_kxx(**parms):
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
