"""."""

from collections import namedtuple as _namedtuple
from copy import deepcopy as _dcopy
import numpy as _np


class KnobTypes:
    """."""

    def __init__(self, focusing, defocusing):
        """."""
        self.focusing = focusing
        self.defocusing = defocusing

    @property
    def all(self):
        """."""
        return self.focusing + self.defocusing


class BaseCorr():
    """."""

    _STRENGTH_TYPE = ''
    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    GROUPING = _namedtuple('Grouping', ['Individual', 'TwoKnobs'])(0, 1)
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model, acc=None, method=None, grouping=None):
        """."""
        self.model = model
        self.acc = acc.upper()
        self._method = BaseCorr.METHODS.Proportional
        self._grouping = BaseCorr.GROUPING.TwoKnobs
        self.knobs = KnobTypes([], [])
        self.fam = None
        self.method = method
        self.grouping = grouping

    @property
    def grouping(self):
        """."""
        return self._grouping

    @grouping.setter
    def grouping(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._grouping = int(value in BaseCorr.GROUPING._fields[1])
        elif int(value) in BaseCorr.GROUPING:
            self._grouping = int(value)

    @property
    def grouping_str(self):
        """."""
        return BaseCorr.GROUPING._fields[self._grouping]

    @property
    def method(self):
        """."""
        return self._method

    @method.setter
    def method(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self._method = int(value in BaseCorr.METHODS._fields[1])
        elif int(value) in BaseCorr.METHODS:
            self._method = int(value)

    @property
    def method_str(self):
        """."""
        return BaseCorr.METHODS._fields[self._method]

    def _get_parameter(self, model=None):
        """."""
        raise NotImplementedError

    def calc_jacobian_matrix(self, model=None):
        """."""
        raise NotImplementedError

    def _get_strength(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs.all
        stren = []
        for knb in knobs:
            stren_mag = []
            for mag in self.fam[knb]['index']:
                stren_seg = []
                for seg in mag:
                    stren_seg.append(getattr(
                        model[seg], self._STRENGTH_TYPE))
                stren_mag.append(sum(stren_seg))
            stren.append(_np.mean(stren_mag))
        return _np.array(stren)

    def correct_parameters(self, goal_parameters, model=None,
                           jacobian_matrix=None, tol=1e-6, nr_max=10,
                           nsv=None):
        """."""
        if model is None:
            model = self.model
        if self.method not in BaseCorr.METHODS:
            raise Exception('Invalid correction method!')

        if None in goal_parameters:
            # use current model values for missing goal parameters
            goal_parameters = _np.array(goal_parameters)  # as not to modify input
            param_now = self._get_parameter(model)
            sel = (goal_parameters == None)
            goal_parameters[sel] = param_now[sel]

        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        else:
            jmat = _dcopy(jacobian_matrix)

        nominal_stren = self._get_strength(model)
        if self._method == BaseCorr.METHODS.Proportional:
            jmat *= nominal_stren
        if self._grouping == BaseCorr.GROUPING.TwoKnobs:
            jmat = self._group_2knobs_matrix(jmat)

        umat, smat, vmat = _np.linalg.svd(jmat, full_matrices=False)
        ismat = 1/smat
        ismat[_np.isnan(ismat)] = 0
        ismat[_np.isinf(ismat)] = 0
        if nsv is not None:
            ismat[nsv:] = 0
        ismat = _np.diag(ismat)
        invmat = -1 * _np.dot(_np.dot(vmat.T, ismat), umat.T)
        param_new = self._get_parameter(model)
        dparam = param_new - goal_parameters
        if _np.sqrt(_np.mean(dparam*dparam)) < tol:
            return BaseCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dstren = _np.dot(invmat, dparam)
            self._add_delta_stren(dstren, model=model)
            param_new = self._get_parameter(model)
            dparam = param_new - goal_parameters
            if _np.sqrt(_np.mean(dparam*dparam)) < tol:
                break
        else:
            return BaseCorr.CORR_STATUS.Fail
        return BaseCorr.CORR_STATUS.Sucess

    def _add_delta_stren(self, delta_stren, model=None):
        """."""
        if model is None:
            model = self.model

        for idx_knb, knb in enumerate(self.knobs.all):
            if self._grouping == BaseCorr.GROUPING.TwoKnobs:
                if knb in self.knobs.focusing:
                    delta = delta_stren[0]
                elif knb in self.knobs.defocusing:
                    delta = delta_stren[1]
            else:
                delta = delta_stren[idx_knb]
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    stren = getattr(model[seg], self._STRENGTH_TYPE)
                    if self._method == BaseCorr.METHODS.Proportional:
                        stren *= (1 + delta/len(mag))
                    else:
                        stren += delta/len(mag)
                    setattr(model[seg], self._STRENGTH_TYPE, stren)

    def _group_2knobs_matrix(self, jacobian_matrix=None):
        """."""
        if jacobian_matrix is None:
            jacobian_matrix = self.calc_jacobian_matrix(self.model)

        jacobian_2knobs_matrix = _np.zeros((2, 2))
        nfocus = len(self.knobs.focusing)

        for nfoc, _ in enumerate(self.knobs.focusing):
            jacobian_2knobs_matrix[:, 0] += jacobian_matrix[:, nfoc]
        for ndf, _ in enumerate(self.knobs.defocusing):
            jacobian_2knobs_matrix[:, 1] += jacobian_matrix[:, ndf+nfocus]
        return jacobian_2knobs_matrix
