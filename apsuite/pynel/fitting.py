"""Fitting module to run dispersion fitting and analisys"""
import numpy as _np
from apsuite.orbcorr import OrbitCorr as _OrbitCorr
from .misc_functions import apply_deltas as _apply_deltas, \
    calc_vdisp as _calc_vdisp, rmk_correct_orbit as _rmk_correct_orbit,\
    calc_rms as _calc_rms, revoke_deltas as _revoke_deltas, calc_pinv as _calc_pinv

_IJMAT = None 

def s_iter(model, disp_meta, base, n_iter, svals="auto", cut=1e-3, Orbcorr="auto"):
    imat, _, smat, _, num_svals = _calc_pinv(base.resp_mat, svals, cut)
    print("N_svals =", num_svals)
    deltas = _np.zeros(len(base.buttons))
    disp = _np.zeros(160)
    if Orbcorr == "auto":
        oc = _OrbitCorr(model, acc="SI")
        oc_jacob_mat = oc.get_jacobian_matrix()
        oc_inv_jacob_mat = oc.get_inverse_matrix(jacobian_matrix=oc_jacob_mat)
        oc.params.maxnriters = 100
    elif isinstance(Orbcorr, tuple) and isinstance(Orbcorr[0], _OrbitCorr):
        oc = Orbcorr[0]
        oc_inv_jacob_mat = Orbcorr[1]
    else:
        raise ValueError(
            "Orbcorr in wrong format: should be a tuple of (_OrbitCorr obj, inverse_jacobian_matrix)"
        )
    for i in range(n_iter):
        disp = _calc_vdisp(model)
        diff = disp_meta - disp
        delta = imat @ diff
        deltas += delta
        _apply_deltas(model=model, base=base, deltas=deltas)
        _rmk_correct_orbit(oc, inverse_jacobian_matrix=oc_inv_jacob_mat)
    disp = _calc_vdisp(model)
    print(f"RMS residue = {_calc_rms(disp-disp_meta):f}")
    print(f"Corr. coef. = {_np.corrcoef(disp, disp_meta)[1,0]*100:.3f}%")
    _revoke_deltas(model, base)
    _rmk_correct_orbit(oc, inverse_jacobian_matrix=oc_inv_jacob_mat)
    return disp, deltas, smat, num_svals

def f_iter_Y(
    model, disp_meta, base, n_iter, svals="auto", cut=1e-3, Orbcorr="auto"
):
    imat, _, smat, _, num_svals = _calc_pinv(base.resp_mat, svals, cut)
    # print("N_svals =", num_svals)
    deltas = _np.zeros(len(base.buttons))
    disp = _np.zeros(160)
    for i in range(n_iter):
        for j, b in enumerate(base.buttons):
            disp += deltas[j] * b.signature
        diff = disp_meta - disp
        delta = imat @ diff
        deltas += delta
    _apply_deltas(model, base, -deltas)
    if Orbcorr == "auto":
        oc = _OrbitCorr(model, acc="SI")
        oc_jacob_mat = oc.get_jacobian_matrix()
        oc_inv_jacob_mat = oc.get_inverse_matrix(jacobian_matrix=oc_jacob_mat)
        oc.params.maxnriters = 100
    elif isinstance(Orbcorr, tuple) and isinstance(Orbcorr[0], _OrbitCorr):
        oc = Orbcorr[0]
        oc_inv_jacob_mat = Orbcorr[1]
    else:
        raise ValueError("Orbcorr in wrong format: should be a tuple of (_OrbitCorr obj, inverse_jacobian_matrix)")
    _rmk_correct_orbit(oc, oc_inv_jacob_mat)
    disp = _calc_vdisp(model)
    # print(f"RMS residue = {_calc_rms(disp-disp_meta):f}")
    # print(f"Corr. coef. = {_np.corrcoef(disp, disp_meta)[1,0]*100:.3f}%")
    return disp, deltas, smat, num_svals

def sf_iter_Y(
    disp_meta, base, n_iter, svals="auto", cut=1e-3
):
    imat, *_ = _calc_pinv(base.resp_mat, svals, cut)
    deltas = _np.zeros(len(base.buttons))
    disp = _np.zeros(160)

    for i in range(n_iter):
        for j, b in enumerate(base.buttons):
            disp += deltas[j] * b.signature
        diff = disp_meta - disp
        delta = imat @ diff
        deltas += delta

    for j, b in enumerate(base.buttons):
        disp += deltas[j] * b.signature
    return disp, deltas

def dev_fit(model, disp_meta, base, n_iter, inv_jacob_mat='std', True_Apply=True, svals="auto", cut=1e-3):
    """Returns: disp, deltas, smat, num_svals, rms_res, corr_coef, total_iter"""
    imat, _, smat, _, num_svals = _calc_pinv(base.resp_mat, svals, cut)
    print("N_svals =", num_svals)
    deltas = _np.zeros(len(base.buttons))
    disp = _np.zeros(160)
    OrbcorrObj = _OrbitCorr(model, 'SI')
    if isinstance(inv_jacob_mat, str):
        if inv_jacob_mat == 'std':
            inv_jacob_mat = _IJMAT
        elif inv_jacob_mat == 'auto':
            _jacob_mat = OrbcorrObj.get_jacobian_matrix()
            inv_jacob_mat = OrbcorrObj.get_inverse_matrix(_jacob_mat)
        else:
            raise ValueError('inv_jacob_mat should be "std", "auto", or a "numpy.ndarray" with shape: (320, 281)')
    elif isinstance(inv_jacob_mat, (_np.ndarray)):
        pass
    else:
        raise ValueError('inv_jacob_mat should be "std", "auto", or a "numpy.ndarray" with shape: (320, 281)')
    total_iter = 0
    for i in range(n_iter):
        disp = _calc_vdisp(model); 
        diff = disp_meta - disp;
        delta = imat @ diff; 
        deltas += delta
        _apply_deltas(model=model, base=base, deltas=deltas);
        try:
            total_iter += 1
            _rmk_correct_orbit(OrbcorrObj, inverse_jacobian_matrix=inv_jacob_mat); 
        except:
            break
    try:
        disp_exit = _calc_vdisp(model)
    except: 
        disp_exit = disp
    rms_res = _calc_rms(disp_exit-disp_meta)
    corr_coef = _np.corrcoef(disp_exit, disp_meta)[1,0]*100
    print(f"RMS residue = {rms_res:f}")
    print(f"Corr. coef. = {corr_coef:.3f}%")
    if not True_Apply:
        _revoke_deltas(model, base)
        _rmk_correct_orbit(OrbcorrObj, inverse_jacobian_matrix=inv_jacob_mat); 
    return disp_exit, deltas, smat, num_svals, rms_res, corr_coef, total_iter

def fit(model, disp_meta, base, n_iter, svals="auto", cut=1e-3, Orbcorr="auto"):
    """
    Function for fitting a 'meta vertical dispersion' into a model, using some Base.
    > model: Accelerator
    > disp_meta: Numpy array with shape (160,)
    > base: Base object
    Returns: disp, deltas, smat, num_svals, rms_res, corr_coef
    """
    imat, _, smat, _, num_svals = _calc_pinv(base.resp_mat, svals, cut)
    print("N_svals =", num_svals)
    deltas = _np.zeros(len(base.buttons))
    disp = _np.zeros(160)
    if Orbcorr == "auto":
        oc = _OrbitCorr(model, acc="SI")
        oc_jacob_mat = oc.get_jacobian_matrix()
        oc_inv_jacob_mat = oc.get_inverse_matrix(jacobian_matrix=oc_jacob_mat)
        oc.params.maxnriters = 100
    elif isinstance(Orbcorr, tuple) and isinstance(Orbcorr[0], _OrbitCorr):
        oc = Orbcorr[0]
        oc_inv_jacob_mat = Orbcorr[1]
    else:
        raise ValueError(
            "Orbcorr in wrong format: should be a tuple of (_OrbitCorr obj, inverse_jacobian_matrix)"
        )
    for i in range(n_iter):
        disp = _calc_vdisp(model)
        diff = disp_meta - disp
        delta = imat @ diff
        deltas += delta
        _apply_deltas(model=model, base=base, deltas=deltas)
        _rmk_correct_orbit(oc, inverse_jacobian_matrix=oc_inv_jacob_mat)
    disp = _calc_vdisp(model)
    rms_res = _calc_rms(disp-disp_meta)
    corr_coef = _np.corrcoef(disp, disp_meta)[1,0]*100
    print(f"RMS residue = {rms_res:f}")
    print(f"Corr. coef. = {corr_coef:.3f}%")
    _revoke_deltas(model, base)
    _rmk_correct_orbit(oc, inverse_jacobian_matrix=oc_inv_jacob_mat)
    return disp, deltas, smat, num_svals, rms_res, corr_coef