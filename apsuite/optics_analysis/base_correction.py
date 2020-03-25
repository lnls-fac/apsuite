"""."""

from collections import namedtuple as _namedtuple
from copy import deepcopy as _dcopy
import numpy as np

__KNOBTYPES = _namedtuple('KnobTypes', ['Focusing', 'Defocusing'])


class KnobTypes(__KNOBTYPES):
    """."""

    @property
    def ALL(self):
        """."""
        return self.Focusing + self.Defocusing


class BaseCorr():
    """."""

    METHODS = _namedtuple('Methods', ['Additional', 'Proportional'])(0, 1)
    GROUPING = _namedtuple('Grouping', ['Individual', 'TwoKnobs'])(0, 1)
    CORR_STATUS = _namedtuple('CorrStatus', ['Fail', 'Sucess'])(0, 1)

    def __init__(self, model=None, acc=None, method=None, grouping=None):
        """."""
        self.model = model
        self.acc = acc
        self._method = BaseCorr.METHODS.Proportional
        self._grouping = BaseCorr.GROUPING.TwoKnobs
        self.knobs = None
        self.fam = None
        self.strength_type = None
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
        elif isinstance(value, str):
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
        elif isinstance(value, str):
            self._method = int(value in BaseCorr.METHODS._fields[1])
        elif int(value) in BaseCorr.METHODS:
            self._method = int(value)

    @property
    def method_str(self):
        """."""
        return BaseCorr.METHODS._fields[self._method]

    def define_knobs(self, f_knobs, d_knobs, strength_type):
        """."""
        self.knobs = KnobTypes(Focusing=f_knobs, Defocusing=d_knobs)
        self.strength_type = strength_type

    def get_parameter(self, model=None):
        """."""
        raise NotImplementedError

    def calc_jacobian_matrix(self, model=None):
        """."""
        raise NotImplementedError

    def get_strength(self, model=None, knobs=None):
        """."""
        if model is None:
            model = self.model
        if knobs is None:
            knobs = self.knobs.ALL
        stren = []
        for knb in knobs:
            stren_mag = []
            for mag in self.fam[knb]['index']:
                stren_seg = []
                for seg in mag:
                    stren_seg.append(getattr(model[seg], self.strength_type))
                stren_mag.append(sum(stren_seg))
            stren.append(np.mean(stren_mag))
        return np.array(stren)

    def correct_parameters(self,
                           goal_parameters,
                           model=None,
                           jacobian_matrix=None,
                           tol=1e-6,
                           nr_max=10,
                           nsv=None):
        """."""
        if model is None:
            model = self.model
        if self.method not in BaseCorr.METHODS:
            raise Exception('Invalid correction method!')

        if jacobian_matrix is None:
            jmat = self.calc_jacobian_matrix(model)
        else:
            jmat = _dcopy(jacobian_matrix)

        nominal_stren = self.get_strength(model)
        if self._method == BaseCorr.METHODS.Proportional:
            jmat *= nominal_stren
        if self._grouping == BaseCorr.GROUPING.TwoKnobs:
            jmat = self._group_2knobs_matrix(jmat)

        U, S, V = np.linalg.svd(jmat, full_matrices=False)
        iS = 1/S
        iS[np.isnan(iS)] = 0
        iS[np.isinf(iS)] = 0
        if nsv is not None:
            iS[nsv:] = 0
        iS = np.diag(iS)
        invmat = -1 * np.dot(np.dot(V.T, iS), U.T)
        param_new = self.get_parameter(model)
        dparam = param_new - goal_parameters
        if np.sum(dparam*dparam) < tol:
            return BaseCorr.CORR_STATUS.Sucess

        for _ in range(nr_max):
            dstren = np.dot(invmat, dparam)
            self._add_delta_stren(dstren, model=model)
            param_new = self.get_parameter(model)
            dparam = param_new - goal_parameters
            if np.sum(dparam*dparam) < tol:
                break
        else:
            return BaseCorr.CORR_STATUS.Fail
        return BaseCorr.CORR_STATUS.Sucess

    def _add_delta_stren(self, delta_stren, model=None):
        """."""
        if model is None:
            model = self.model

        for idx_knb, knb in enumerate(self.knobs.ALL):
            if self._grouping == BaseCorr.GROUPING.TwoKnobs:
                if knb in self.knobs.Focusing:
                    delta = delta_stren[0]
                elif knb in self.knobs.Defocusing:
                    delta = delta_stren[1]
            else:
                delta = delta_stren[idx_knb]
            for mag in self.fam[knb]['index']:
                for seg in mag:
                    stren = getattr(model[seg], self.strength_type)
                    if self._method == BaseCorr.METHODS.Proportional:
                        stren *= (1 + delta/len(mag))
                    else:
                        stren += delta/len(mag)
                    setattr(model[seg], self.strength_type, stren)

    def _group_2knobs_matrix(self, jacobian_matrix=None):
        """."""
        if jacobian_matrix is None:
            jacobian_matrix = self.calc_jacobian_matrix(self.model)

        jacobian_2knobs_matrix = np.zeros((2, 2))
        nfocus = len(self.knobs.Focusing)

        for nf, _ in enumerate(self.knobs.Focusing):
            jacobian_2knobs_matrix[:, 0] += jacobian_matrix[:, nf]
        for ndf, _ in enumerate(self.knobs.Defocusing):
            jacobian_2knobs_matrix[:, 1] += jacobian_matrix[:, ndf+nfocus]
        return jacobian_2knobs_matrix
