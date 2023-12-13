"""Fitting module to run dispersion fitting and analisys"""
import numpy as _np
from apsuite.orbcorr import OrbitCorr as _OrbitCorr
from .functions import set_errors as _set_errors, \
    get_errors as _get_errors, rmk_correct_orbit as _correct_orbit, \
    calc_pinv as _pinv, calc_disp as _disp
from pymodels import si as _si

def fit(base, dispy_meta, **kwargs):
    r"""Fitting: fit vertical dispersion in SIRIUS model with misalignments.

    Args.:
    base (Base object)
    dispy_meta: (numpy 1D array, size=160)

    **kargs:
    model: pymodels SIRIUS model (unnecessary when orbcorr_obj is passed!).
    orbcorr_obj: Apsuite's OrbitCorr SI object.
    orbcorr_jacobian: jacobian matrix for orbit correction.
    svals: integer number to limit the singular values for fitting.
    cut: floating point to limit the number of singular values for fitting.
    nr_iters: integer fot set the number of iterations in the fitting.
    TrueApply (boolean): apply the fit or not in the model
    return_conv (boolean): return the step-by-step of the fitting.

    Return:
    > fitted_dispy, deltas
        If "return_conv" = False and model or orbcorr_obj is passed:
    or
    > fitted_dispy, deltas, fitting_dict_evolution
        If "return_conv" = True
    or
    > [...], model
        If neither model or orbcorr_obj is passed:

    """
    model = None;
    nr_iters = 10;
    svals = "auto";
    cut = 5e-3;
    return_conv = False;
    trueapply = True
    oc = None;
    jac = None
    if "model" in kwargs:
        model = kwargs['model']
    if "orbcorr_obj" in kwargs:
        oc = kwargs['orbcorr_obj']
        model = oc.respm.model
        if oc.respm.model != model:
            print("""> Passed model is different from the \
                  model in OrbitCorr object. \
                  Now using model from OrbitCorr object!""")
    if "orbcorr_jacobian" in kwargs:
        jac = kwargs['orbcorr_jacobian']
    if "TrueApply" in kwargs:
        trueapply = kwargs['TrueApply']
        if not isinstance(trueapply, bool):
            raise ValueError('TrueApply should be True or False!')
    rtn_model_flag = False
    if model is None:
        model = _si.create_accelerator()
        rtn_model_flag = True
        if trueapply == False:
            trueapply = True
            print('When model isnt passed, a new model is\
                   created and returned with applied fit')
    if oc is None:
        oc = _OrbitCorr(model, 'SI')
    if jac is None:
        jac = oc.get_jacobian_matrix()
    if "svals" in kwargs:
        svals = kwargs['svals']
    if "cut" in kwargs:
        cut = kwargs['cut']
    if "nr_iters" in kwargs:
        nr_iters = kwargs['nr_iters']
    if "return_conv" in kwargs:
        return_conv = kwargs['return_conv']
        if not isinstance(return_conv, bool):
            raise ValueError('return_conv should be True or False!')

    mat = _np.zeros_like(base.resp_mat)
    mat += base.resp_mat
    imat, u, smat, vt, nr_sv = _pinv(mat, svals=svals, cut=cut,
                                    return_svd=True)
    init_errors = _get_errors(model, base)
    init_kicks = oc.get_kicks()
    fulldeltas = _np.zeros(len(base))
    if return_conv:
        evo = {'deltas':[], 'dispersion':[_disp(model)],
               'rms_deltas':[], 'rms_dispy':[], 'orbcorr_status':[]}
    for i in range(nr_iters):
        disp = _disp(model)
        ddispy = dispy_meta - disp[160:]
        deltas = _np.dot(imat, ddispy)
        fulldeltas += deltas
        _set_errors(model, base, fulldeltas)
        try:
            oc_flag, _ = _correct_orbit(oc, jac)
            if return_conv:
                evo['deltas'].append(deltas)
                evo['rms_deltas'].append(_np.std(deltas))
                evo['rms_dispy'].append(_np.std(disp[160:]))
                evo['dispersion'].append(disp)
                evo['orbcorr_status'].append(oc_flag)
        except:
            fulldeltas -= deltas
            _set_errors(model, base, fulldeltas)
            oc_flag, _ = _correct_orbit(oc, jac)
            break
        count = i+1
    finaldispy = _disp(model)[160:]
    diff = dispy_meta - finaldispy
    rms_diff = _np.std(diff)
    coef = _np.corrcoef(finaldispy, dispy_meta)[0, 1]
    print("Fitting done!\n"+fr'Total iterations: {count:d}')
    print(f"RMS diff = {rms_diff*1e3:.3f} [mm]")
    print(f"Correlation coef. = {coef*1e2:.1f} [%]")
    if not trueapply:
        _set_errors(model, base, init_errors)
        oc.set_kicks(init_kicks)
    res = [finaldispy, fulldeltas]
    if return_conv:
     res.append(evo)
    if rtn_model_flag:
        res.append(model)
    return res
