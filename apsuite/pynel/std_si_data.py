"""Standart "PyModels Sirius model" data collection"""

import pymodels as _pymodels
import numpy as _np
from pyaccel.lattice import find_spos as _find_spos, find_indices as _find_indices
from copy import deepcopy as _deepcopy
from pyaccel import lattice as _latt

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

def SI_FAMDATA():
    """Get the default SIRIUS families data"""
    return _deepcopy(_pymodels.si.families.get_family_data(__model))

__bpmidx = _find_indices(__model, 'fam_name', 'BPM')
def BPMIDX():
    """Get the default SIRIUS BPM's indices"""
    return _deepcopy(__bpmidx)

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

__deltas = {
    'dx':  {'B':40e-6, 'Q':40e-6, 'S':40e-6}, 
    'dy':  {'B':40e-6, 'Q':40e-6, 'S':40e-6}, 
    'dr':  {'B':0.3e-3, 'Q':0.3e-3, 'S':0.3e-3},
    'drp': {'B':0.3e-3, 'Q':0.3e-3, 'S':0.3e-3}, 
    'dry': {'B':0.3e-3, 'Q':0.3e-3, 'S':0.3e-3}}

def STD_ERROR_DELTAS():
    """Default misalignment and rotation expected error"""
    return _deepcopy(__deltas)