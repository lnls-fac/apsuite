"""."""

import numpy as _np


def calc_traj_chrom(params, *args):
    """BPM averaging due to longitudinal dynamics decoherence.

    nu ~ nu0 + chrom * delta_energy
    See Laurent Nadolski Thesis, Chapter 4, pg. 121, Eq. 4.15
    """
    tunes_frac = params[0]
    tune_frac = params[1]
    chrom = params[2]
    espread = params[3]
    r0 = params[4]
    mu = params[5]

    select_idx_turn_start, select_idx_turn_stop, offset = args
    turn = _np.arange(select_idx_turn_start, select_idx_turn_stop)
    cos = _np.cos(2 * _np.pi * tune_frac * turn + mu)
    chromx_decoh = 2 * chrom * espread / tunes_frac
    alp = chromx_decoh * _np.sin(_np.pi * tunes_frac * turn)
    exp = _np.exp(-alp**2/2.0)
    traj = r0 * exp * cos + offset
    return traj
