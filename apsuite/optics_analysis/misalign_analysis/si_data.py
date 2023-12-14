"""Standart "PyModels Sirius model" data collection."""

from copy import deepcopy as _deepcopy

import pymodels as _pymodels
from pyaccel.lattice import find_indices as _find_indices, \
    find_spos as _find_spos

__model = _pymodels.si.create_accelerator()
__model.radiation_on = True
__model.vchamber_on = True
__model.cavity_on = True


def model_base():
    """Generate standart SIRIUS model with radiation, cavity and vchambers ON.

    Returns:
        pyaccel.accelerator: standart SIRIUS model
    """
    return _deepcopy(__model)


__mi_idx = sorted(
    _find_indices(__model, "fam_name", "mib")
    + _find_indices(__model, "fam_name", "mip")
    + _find_indices(__model, "fam_name", "mia")
    + [len(__model)]
)

__spos = _find_spos(__model, indices="closed")


def si_spos():
    """Default SIRIUS longitudinal coordinates of the lattice elements."""
    return _deepcopy(__spos)


def si_sextupoles():
    """Default SIRIUS sextupoles names."""
    return [
        "SDA0",
        "SDA1",
        "SDA2",
        "SDA3",
        "SFA0",
        "SFA1",
        "SFA2",
        "SDB0",
        "SDB1",
        "SDB2",
        "SDB3",
        "SFB0",
        "SFB1",
        "SFB2",
        "SDP0",
        "SDP1",
        "SDP2",
        "SDP3",
        "SFP0",
        "SFP1",
        "SFP2",
    ]


def si_dipoles():
    """Default SIRIUS dipoles families."""
    return ["B1", "B2", "BC"]


def si_quadrupoles():
    """Default SIRIUS quadrupoles families."""
    return [
        "QFA",
        "QDA",
        "QFB",
        "QDB1",
        "QDB2",
        "QFP",
        "QDP1",
        "QDP2",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]


def si_famdata():
    """Default SIRIUS families data."""
    return _deepcopy(_pymodels.si.families.get_family_data(__model))


__bpmidx = _find_indices(__model, "fam_name", "BPM")


def si_bpmidx():
    """Default SIRIUS BPM's indices."""
    return _deepcopy(__bpmidx)


def std_misaligment_types():
    """Default misalignment types.

    'dr'  : Rotation roll misalignment (theta)
    'dx'  : Tranverse-horizontal misalignment (X)
    'dy'  : Tranverse-vertical misalignment (Y)
    'drp' : Rotation pitch misalignment
    'dry' : Rotation yaw misalignment
    """
    return ["dx", "dy", "dr", "drp", "dry"]  # , 'dksl'])


__all_sects = list(range(1, 21))


def si_sectors():
    """Default sectors of SIRIUS ring."""
    return _deepcopy(__all_sects)


def si_elems():
    """Default SIRIUS elements families."""
    return [
        "B1",
        "B2",
        "BC",
        "QFA",
        "QDA",
        "QFB",
        "QDB1",
        "QDB2",
        "QFP",
        "QDP1",
        "QDP2",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "SDA0",
        "SDA1",
        "SDA2",
        "SDA3",
        "SFA0",
        "SFA1",
        "SFA2",
        "SDB0",
        "SDB1",
        "SDB2",
        "SDB3",
        "SFB0",
        "SFB1",
        "SFB2",
        "SDP0",
        "SDP1",
        "SDP2",
        "SDP3",
        "SFP0",
        "SFP1",
        "SFP2",
    ]


__deltas = {
    "dx": {"B": 40e-6, "Q": 40e-6, "S": 40e-6},
    "dy": {"B": 40e-6, "Q": 40e-6, "S": 40e-6},
    "dr": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
    "drp": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
    "dry": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
}


def std_misaligment_tolerance():
    """Default misalignment and rotation expected error."""
    return _deepcopy(__deltas)
