"""Standart "PyModels Sirius model" data collection"""

import pymodels as _pymodels
import numpy as _np
from pyaccel.lattice import find_spos as _find_spos, find_indices as _find_indices
import os as _os
from copy import deepcopy as _deepcopy
from mathphys.functions import load_pickle as _load_pickle

_path_model = _os.path.join(_os.path.dirname(__file__), "model_SI_V25_04_v1.17.0.pickle")
if _pymodels.__version__ == '1.17.0':
    __model = _load_pickle(_path_model)
else:
    __model = _pymodels.si.create_accelerator()
    __model.radiation_on = True
    __model.vchamber_on = True
    __model.cavity_on = True

def MODEL_BASE():
    """Generate a standart (BASE) SIRIUS model with radiation, cavity and vchambers ON."""
    return _deepcopy(__model)

__mi_idx = sorted(_find_indices(__model, 'fam_name', 'mib') +\
                _find_indices(__model, 'fam_name', 'mip') +\
                _find_indices(__model, 'fam_name', 'mia') + [len(__model)])

__spos = _find_spos(__model, indices='closed')
def SI_SPOS():
    """Get the default SIRIUS longitudinal coordinates of the lattice elements"""
    return _deepcopy(__spos)

def SI_SEXTUPOLES():
    """Get the default SIRIUS sextupoles names"""
    return ['SDA0','SDA1','SDA2','SDA3','SFA0','SFA1','SFA2',
            'SDB0','SDB1','SDB2','SDB3','SFB0','SFB1','SFB2',
            'SDP0','SDP1','SDP2','SDP3','SFP0','SFP1','SFP2']

def SI_DIPOLES():
    """Get the default SIRIUS dipoles names"""
    return ['B1', 'B2', 'BC']

def SI_QUADRUPOLES():
    """Get the default SIRIUS quadrupoles names"""
    return ['QFA','QDA',
            'QFB','QDB1','QDB2',
            'QFP','QDP1','QDP2',
            'Q1','Q2','Q3','Q4']

# if _famdata is None:
#     print('getting_famdata')
#     _famdata = _pymodels.si.families.get_family_data(_model)

def SI_FAMDATA():
    """Get the default SIRIUS families data"""
    return _deepcopy(_pymodels.si.families.get_family_data(__model))

__bpmidx = _find_indices(__model, 'fam_name', 'BPM')
def BPMIDX():
    """Get the default SIRIUS BPM's indices"""
    return _deepcopy(__bpmidx)

def STD_ELEMS_HALB():
    """Get the default SIRIUS names of the elements in the 'HA-LB' or 'LB-HA' sectors"""
    return ['SFA0', 'QFA', 'SDA0', 'QDA', 'B1',
            'SDA1', 'Q1', 'SFA1', 'Q2', 'SDA2','B2',
            'SDA3', 'Q3', 'SFA2', 'Q4', 
            'BC',
            'Q4', 'SFB2', 'Q3', 'SDB3', 
            'B2','SDB2', 'Q2', 'SFB1', 'Q1', 'SDB1', 
            'B1','QDB1', 'SDB0', 'QFB', 'SFB0', 'QDB2']

def STD_ELEMS_LBLP():
    """Get the default SIRIUS names of the elements in the 'LB-LP' or 'LP-LB' sectors"""
    return ['QDB2', 'SFB0', 'QFB', 'SDB0', 'QDB1', 'B1',
            'SDB1', 'Q1', 'SFB1', 'Q2', 'SDB2', 'B2',
            'SDB3', 'Q3', 'SFB2', 'Q4',
            'BC',
            'Q4', 'SFP2', 'Q3', 'SDP3', 
            'B2','SDP2', 'Q2', 'SFP1', 'Q1', 'SDP1', 
            'B1','QDP1', 'SDP0', 'QFP', 'SFP0', 'QDP2']

def STD_TYPES():
    """Returns the default modification types: 
    'dr'  : Rotation roll misalignment (theta)
    'dx'  : Tranverse-horizontal misalignment (X)
    'dy'  : Tranverse-vertical misalignment (Y)
    'drp' : Rotation pitch misalignment 
    'dry' : Rotation yaw misalignment
    """
    return [ 'dx', 'dy', 'dr', 'drp', 'dry'] #, 'dksl'])

__all_sects = list(range(1, 21))
def STD_SECTS():
    """Returns the default sectors of SIRIUS ring (sectors from 1 to 20)"""
    return _deepcopy(__all_sects)

def STD_ELEMS():
    """Get the default SIRIUS names of dipoles, quadrupoles and sextupoles in the ring"""
    return ['B1','B2','BC',
            'QFA','QDA',
            'QFB','QDB1','QDB2',
            'QFP','QDP1','QDP2',
            'Q1','Q2','Q3','Q4',
            'SDA0','SDA1','SDA2','SDA3','SFA0','SFA1','SFA2',
            'SDB0','SDB1','SDB2','SDB3','SFB0','SFB1','SFB2',
            'SDP0','SDP1','SDP2','SDP3','SFP0','SFP1','SFP2']

__sect_spos = __spos[__mi_idx]
def SI_SECT_SPOS():
    """Get the longitudinal coordinates at the start of each sector"""
    return _deepcopy(__sect_spos)

def SI_SECT_INDICES():
    """Get the indices at the start of each sector"""
    return _deepcopy(__mi_idx)

def SI_SECTOR_TYPES():
    """Get the default SIRIUS sector types"""
    return ['HighBetaA -> LowBetaB', 
            'LowBetaB -> LowBetaP', 
            'LowBetaP -> LowBetaB', 
            'LowBetaB -> HighBetaA']

def NothingHere():
    """You will find nothing here..."""
    print('Nothing Here!')

# #_orbcorr_jacob_mat = _OrbitCorr(_model, 'SI').get_jacobian_matrix()
__orbcorr_jacob_mat = _load_pickle(_os.path.join(_os.path.dirname(__file__), 'orbcorr_jacobian_SI_V25_04_v1.17.0.pickle')) # pre-saved jacobian for MODEL_BASE
def STD_ORBCORR_JACOBIAN():
    """Return the inverse matrix of the Orbit Correction Jacobian of the standart pymodels SIRIUS model"""
    return _deepcopy(__orbcorr_jacob_mat)

__deltas = {'dx':  {'B':40e-6, 'Q':40e-6, 'S':40e-6}, 'dy':  {'B':40e-6, 'Q':40e-6, 'S':40e-6}, 
 'dr':  {'B':0.3e-3, 'Q':0.3e-3, 'S':0.17e-3}, 'drp': {'B':0.3e-3, 'Q':0.3e-3, 'S':0.17e-3}, 
 'dry': {'B':0.3e-3, 'Q':0.3e-3, 'S':0.17e-3}}
def STD_ERROR_DELTAS():
    """Default misalignment and rotation expected error"""
    return _deepcopy(__deltas)

__full_buttons = _load_pickle(_os.path.join(_os.path.dirname(__file__), "full_buttons_04_09_23.pickle"))
def COMPLETE_BUTTONS_VERTICAL_DISPERSION():
    return _deepcopy(__full_buttons)