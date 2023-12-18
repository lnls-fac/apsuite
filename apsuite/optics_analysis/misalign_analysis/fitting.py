"""Fitting module to run dispersion fitting and analisys."""

import numpy as _np
from pymodels import si as _si

from apsuite.orbcorr import OrbitCorr as _OrbitCorr

from .functions import calc_disp as _disp, calc_pinv as _pinv, \
    rmk_orbit_corr as _correct_orbit, set_errors as _set_errors \
    # get_errors as _get_errors,


def fit(
    base,
    dispy_meta,
    nr_iters=5,
    orbcorr_obj=None,
    orbcorr_jac=None,
    svals="auto",
    svd_cut=1e-3,
    model=None,
):
    """."""
    imat, goal, oc, jac, mod = _handle_input(
        base,
        dispy_meta,
        nr_iters,
        orbcorr_obj,
        orbcorr_jac,
        svals,
        svd_cut,
        model,
    )

    dispy, deltas = _fitting_loop(base, imat, goal, oc, jac, mod, nr_iters)

    return dispy, deltas, mod


def _handle_input(
    base, dispy_meta, nr_iters, orbcorr_obj, orbcorr_jac, svals, cut, model
):
    mat = base.resp_mat
    imat = _pinv(mat, svals=svals, cut=cut)

    if (
        isinstance(dispy_meta, (list, tuple, _np.ndarray))
        and len(dispy_meta) == 160
    ):
        dispy_meta = _np.array(dispy_meta)
    else:
        raise ValueError("Invalid Dispy")

    nr_iters = int(nr_iters)

    if all(i is not None for i in [model, orbcorr_obj]):
        raise ValueError("too much args: model and orbcorr_obj")

    if model is not None:
        mod = model
        oc = _OrbitCorr(model, "SI")
    elif model is None and orbcorr_obj is None:
        mod = _si.create_accelerator()
        oc = _OrbitCorr(mod, "SI")
    else:
        oc = orbcorr_obj
        mod = oc.respm.model

    if orbcorr_jac is not None:
        jac = orbcorr_jac
    else:
        jac = oc.get_jacobian_matrix()

    return imat, dispy_meta, oc, jac, mod


def _fitting_loop(base, imat, dispy_meta, oc, jac, mod, nr_iters):
    count = 0
    fulldeltas = _np.zeros(len(base))
    for _ in range(nr_iters):
        disp = _disp(mod)
        ddispy = dispy_meta - disp[160:]
        deltas = _np.dot(imat, ddispy)
        fulldeltas += deltas
        _set_errors(mod, base, fulldeltas)
        try:
            _correct_orbit(oc, jac)
        except Exception:
            fulldeltas -= deltas
            _set_errors(mod, base, fulldeltas)
            _correct_orbit(oc, jac)
            break
        count += 1
    finaldispy = _disp(mod)[160:]
    return finaldispy, fulldeltas, count
