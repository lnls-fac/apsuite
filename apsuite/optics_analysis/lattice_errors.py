from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

import mathphys as _mp


class Error:

    def __init__(self, avg=None, sigma=None):
        self._mean = avg
        self._sigma = sigma

    def __bool__(self):
        return not (self._sigma is None and self._mean is None)

    @property
    def mean(self):
        return self._mean or 0.0

    @mean.setter
    def mean(self, value):
        self._mean = float(value)

    @property
    def sigma(self):
        return self._sigma or 1.0

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)


class BaseError(ABC):

    DIST_TYPE = _mp.functions.get_namedtuple(
        'DistributionType', ('Normal', 'Uniform'))

    def __init__(self):
        self.indcs = [[], ]
        self.fam_name = ''
        self.dist_type = self.DIST_TYPE.Normal
        self.cutoff = 3
        self.mis_x = Error()
        self.mis_y = Error()
        self.rot_roll = Error()
        self.rot_pitch = Error()
        self.rot_yaw = Error()
        self.error_mis_x = np.array()
        self.error_mis_y = np.array()
        self.error_rot_roll = np.array()
        self.error_rot_pitch = np.array()
        self.error_rot_yaw = np.array()

    @staticmethod
    def generate_errors(nerr, dist_type, errortype, cutoff):
        if not errortype:
            return None

        if dist_type not in BaseError.DIST_TYPE:
            raise ValueError(
                'dist_type must be one of ' + str(BaseError.DIST_TYPE))
        typ = BaseError.DIST_TYPE._fields[dist_type].lower()

        errs = _mp.functions.generate_random_numbers(
            nerr, dist_type=typ, cutoff=cutoff)

        errs *= errortype.sigma
        errs += errortype.mean
        return errs


    @abstractmethod
    def apply_errors(self, lattice, fraction=1):
        return NotImplemented


class MagnetError(BaseError):

    def __init__(self):
        super().__init__()
        self.mean_exc_main = None
        self.sigma_exc_main = None
        self.errors_exc_main = np.array()
        self.multi_main_monomial = 0
        self.multi_main_isskew = False
        self.multi_r0 = 12e-3
        self.multi_monomials = np.array()
        self.mean_multi_normal = np.array()
        self.mean_multi_skew = np.array()
        self.sigma_multi_normal = np.array()
        self.sigma_multi_skew = np.array()
        self.errors_multi_normal = np.array()
        self.errors_multi_skew = np.array()


class BPMError(BaseError):
    pass


class GirderError(BaseError):
    pass
