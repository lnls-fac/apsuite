#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
import time as _time
import copy as _copy

import pyaccel as _pyaccel
import pymodels as _pymodels
from apsuite.orbcorr import OrbitCorr, CorrParams
from apsuite.optics_analysis import TuneCorr, OpticsCorr, CouplingCorr
from apsuite.commisslib.measure_bba import BBAParams
from mathphys.functions import save_pickle, load_pickle
from mathphys import units


class ConfigErrors:
    """Class to configure general types of errors."""

    class Constants:
        UM = 1e-6
        MRAD = 1e-3
        PERCENT = 1e-2

    def __init__(self):
        """Errors attributes"""
        self._fam_names = []
        self._sigma_x = 0
        self._sigma_y = 0
        self._sigma_roll = 0
        self._sigma_pitch = 0
        self._sigma_yaw = 0
        self._sigmas_dict = dict()
        self._um = self.Constants.UM
        self._mrad = self.Constants.MRAD
        self._percent = self.Constants.PERCENT

    @property
    def fam_names(self):
        """Families which the errors are going to be applied.

        Returns:
            list of string: List with the names of the families
        """
        return self._fam_names

    @fam_names.setter
    def fam_names(self, value):
        self._fam_names = value

    @property
    def sigma_x(self):
        """Standard deviation of the x position errors.

        Returns:
            float: [um]
        """
        return self._sigma_x

    @sigma_x.setter
    def sigma_x(self, value):
        """Standard deviation of the x position errors.

        Args:
            value (float): The input value must be in [um].
        """
        self._sigma_x = value * self._um

    @property
    def sigma_y(self):
        """Standard deviation of the y position errors.

        Returns:
            float: [um]
        """
        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, value):
        """Standard deviation of the y position errors.

        Args:
            value (float): The input value must be in [um].
        """
        self._sigma_y = value * self._um

    @property
    def sigma_roll(self):
        """Standard deviation of the roll errors.

        Returns:
            float: [mrad]
        """
        return self._sigma_roll

    @sigma_roll.setter
    def sigma_roll(self, value):
        """Standard deviation of the roll errors.

        Args:
            value (float): The input value must be in [mrad]..
        """
        self._sigma_roll = value * self._mrad

    @property
    def sigma_pitch(self):
        """Standard deviation of the pitch errors.

        Returns:
            float: [mrad]
        """
        return self._sigma_pitch

    @sigma_pitch.setter
    def sigma_pitch(self, value):
        """Standard deviation of the roll errors.

        Args:
            value (float): The input value must be in [mrad]..
        """
        self._sigma_pitch = value * self._mrad

    @property
    def sigma_yaw(self):
        """Standard deviation of the yaw errors.

        Returns:
            float: [mrad]
        """
        return self._sigma_yaw

    @sigma_yaw.setter
    def sigma_yaw(self, value):
        """Standard deviation of the roll errors.

        Args:
            value (float): The input value must be in [mrad]..
        """
        self._sigma_yaw = value * self._mrad

    @property
    def sigmas_dict(self):
        """Dictionary with names and values of the configured errors.

        Returns:
            dictionary: Keys are the types of the errors, values are the std.
        """
        return self._sigmas_dict

    @sigmas_dict.setter
    def sigmas_dict(self, value):
        """Dictionary with names and values of the configured errors.

        Args:
            value (dict - key:string, value:float): Keys are the types of the
                errors, values are the std.
        """
        self._sigmas_dict = value


class MultipolesErrors(ConfigErrors):
    """Class to configure multipole errors."""

    def __init__(self):
        """Multipole errors attributes."""
        super().__init__()
        self._sigma_excit = 0
        self._r0 = 12e-3  # [m]
        self._multipoles_dict = None
        self._normal_multipoles_order = []
        self._skew_multipoles_order = []
        self._sigma_multipoles_n = []
        self._sigma_multipoles_s = []

    @property
    def sigma_excit(self):
        """Standard deviation of the excitation errors.

        Returns:
            float: [%]
        """
        return self._sigma_excit

    @sigma_excit.setter
    def sigma_excit(self, value):
        """Standard deviation of the excitation errors.

        Args:
            value (float): Must be in [%]
        """
        self._sigma_excit = value * self._percent

    @property
    def r0(self):
        """ Transverse horizontal position where
            the multipoles are normalized.

        Returns:
            float: [meter]
        """
        return self._r0

    @r0.setter
    def r0(self, value):
        """Transverse horizontal position where
            the multipoles are normalized.

        Args:
            value (float): [meter]
        """
        self._r0 = value

    @property
    def normal_multipoles_order(self):
        """List with multipoles order in which the error are going to be
             applied.

        Returns:
            list of integers: no unit
        """
        return self._normal_multipoles_order

    @normal_multipoles_order.setter
    def normal_multipoles_order(self, value):
        self._normal_multipoles_order = value

    @property
    def skew_multipoles_order(self):
        """List with multipoles order in which the error are going to be
             applied.

        Returns:
            list of integers: no unit
        """
        return self._skew_multipoles_order

    @skew_multipoles_order.setter
    def skew_multipoles_order(self, value):
        self._skew_multipoles_order = value

    @property
    def sigma_multipoles_n(self):
        """List of standard deviation of the normalized normal multipole errors

        Returns:
            list of floats: no unit
        """
        return self._sigma_multipoles_n

    @sigma_multipoles_n.setter
    def sigma_multipoles_n(self, value):
        self._sigma_multipoles_n = value

    @property
    def sigma_multipoles_s(self):
        """List of standard deviation of the normalized skew multipole errors

        Returns:
            list of floats: no unit
        """
        return self._sigma_multipoles_s

    @sigma_multipoles_s.setter
    def sigma_multipoles_s(self, value):
        self._sigma_multipoles_s = value

    @property
    def multipoles_dict(self):
        """Dictionary whose keys are the type and the order of the multipole
            and the items are the errors std value.

        Returns:
            dictionary: example: multipoles_dict['normal][3] will return the
                std of the normalized normal sextupolar error.
        """
        return self._multipoles_dict

    @multipoles_dict.setter
    def multipoles_dict(self, value):
        self._multipoles_dict = value

    def create_multipoles_dict(self):
        """Create the dictionary struct for multipoles errors."""
        n_multipoles_order_dict = dict()
        s_multipoles_order_dict = dict()
        for i, order in enumerate(self.normal_multipoles_order):
            n_multipoles_order_dict[order] = self.sigma_multipoles_n[i]
        for i, order in enumerate(self.skew_multipoles_order):
            s_multipoles_order_dict[order] = self.sigma_multipoles_s[i]
        self.multipoles_dict = dict()
        self.multipoles_dict['normal'] = n_multipoles_order_dict
        self.multipoles_dict['skew'] = n_multipoles_order_dict
        self.multipoles_dict['r0'] = self._r0


class DipolesErrors(MultipolesErrors):
    """Class to configure dipole errors."""

    def __init__(self):
        """Dipole errors attributes."""
        super().__init__()
        self._sigma_kdip = 0  # [%]
        self._set_default_dipole_config()

    @property
    def sigma_kdip(self):
        """Standard deviation of the field gradient errors.

        Returns:
            float: [%]
        """
        return self._sigma_kdip

    @sigma_kdip.setter
    def sigma_kdip(self, value):
        """Standard deviation of the field gradient errors.

        Args:
            value (float): [%]
        """
        self._sigma_kdip = value * self._percent

    def _set_default_dipole_config(self):
        """Create a default configuration for dipole errors."""
        self.fam_names = ['B1', 'B2', 'BC']
        self.sigma_x = 40  # [um]
        self.sigma_y = 40  # [um]
        self.sigma_roll = 0.30  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]
        self.sigma_excit = 0.05  # [%]
        self.sigma_kdip = 0.10  # [%]
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        sigmas_dict['excit'] = self.sigma_excit
        sigmas_dict['kdip'] = self.sigma_kdip
        sigmas_dict['multipoles'] = self.multipoles_dict
        self.sigmas_dict = sigmas_dict


class QuadsErrors(MultipolesErrors):
    """Class to configure quadrupole errors."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._set_default_quad_config()

    def _set_default_quad_config(self):
        """Create a default configuration for quadrupole errors."""
        self.fam_names = ['QFA', 'QDA', 'Q1', 'Q2', 'Q3', 'Q4', 'QDB1',
                          'QFB',  'QDB2', 'QDP1', 'QFP', 'QDP2']
        self.sigma_x = 40  # [um]
        self.sigma_y = 40  # [um]
        self.sigma_roll = 0.30  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]
        self.sigma_excit = 0.05  # [%]
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        sigmas_dict['excit'] = self.sigma_excit
        sigmas_dict['multipoles'] = self.multipoles_dict
        self.sigmas_dict = sigmas_dict


class QuadsSkewErrors(MultipolesErrors):
    """Class to configure skew quadrupole errors."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._set_default_quad_config()

    def _set_default_quad_config(self):
        """Create a default configuration for skew quadrupole errors."""
        self.fam_names = ['QS']
        self.sigma_x = 0  # [um]
        self.sigma_y = 0  # [um]
        self.sigma_roll = 0  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]
        self.sigma_excit = 0.05  # [%]
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        sigmas_dict['excit'] = self.sigma_excit
        sigmas_dict['multipoles'] = self.multipoles_dict
        self.sigmas_dict = sigmas_dict


class SextsErrors(MultipolesErrors):
    """Class to configure sextupole errors."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._set_default_sext_config()

    def _set_default_sext_config(self):
        """Create a default configuration for sextupole errors."""
        self.fam_names = ['SFA0', 'SDA0', 'SDA1', 'SFA1', 'SDA2', 'SDA3',
                          'SFA2', 'SFB2', 'SDB3', 'SDB2', 'SFB1', 'SDB1',
                          'SDB0', 'SFB0', 'SFP2', 'SDP3', 'SDP2', 'SFP1',
                          'SDP1', 'SDP0', 'SFP0']
        self.sigma_x = 40  # [um]
        self.sigma_y = 40  # [um]
        self.sigma_roll = 0.17  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]
        self.sigma_excit = 0.05  # [%]
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [4, 5, 6, 7]
        self.skew_multipoles_order = [4, 5, 6, 7]
        self.create_multipoles_dict()

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        sigmas_dict['excit'] = self.sigma_excit
        sigmas_dict['multipoles'] = self.multipoles_dict
        self.sigmas_dict = sigmas_dict


class GirderErrors(ConfigErrors):
    """Class to configure girder errors."""

    def __init__(self):
        """Constructor"""
        super().__init__()
        self._set_default_girder_config()

    def _set_default_girder_config(self):
        """Create a default configuration for girder errors."""
        self.fam_names = ['girder']
        self.sigma_x = 80  # [um]
        self.sigma_y = 80  # [um]
        self.sigma_roll = 0.30  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        self.sigmas_dict = sigmas_dict


class BPMErrors(ConfigErrors):
    """Class to configure bpm errors."""

    def __init__(self):
        """Constructor"""
        super().__init__()
        self._set_default_bpm_config()

    def _set_default_bpm_config(self):
        """Create a default configuration for bpm errors."""
        self.fam_names = ['BPM']
        self.sigma_x = 20  # [um]
        self.sigma_y = 20  # [um]
        self.sigma_roll = 0.30  # [mrad]
        self.sigma_pitch = 0  # [mrad]
        self.sigma_yaw = 0  # [mrad]

        sigmas_dict = dict()
        sigmas_dict['posx'] = self.sigma_x
        sigmas_dict['posy'] = self.sigma_y
        sigmas_dict['roll'] = self.sigma_roll
        sigmas_dict['pitch'] = self.sigma_pitch
        sigmas_dict['yaw'] = self.sigma_yaw
        self.sigmas_dict = sigmas_dict
