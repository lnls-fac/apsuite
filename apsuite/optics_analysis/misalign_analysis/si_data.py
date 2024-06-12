"""Standart "PyModels Sirius model" data collection."""

from copy import deepcopy as _deepcopy

import pymodels as _pymodels
from pyaccel.lattice import find_indices as _find_indices, \
    find_spos as _find_spos


def std_misaligment_tolerance():
    """Default misalignment and rotation expected error."""
    return {
        "dx": {"B": 40e-6, "Q": 40e-6, "S": 40e-6},
        "dy": {"B": 40e-6, "Q": 40e-6, "S": 40e-6},
        "dr": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
        "drp": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
        "dry": {"B": 0.3e-3, "Q": 0.3e-3, "S": 0.3e-3},
    }


def std_misaligment_types():
    """Default misalignment types.

    'dr'  : Rotation roll misalignment (theta)
    'dx'  : Tranverse-horizontal misalignment (X)
    'dy'  : Tranverse-vertical misalignment (Y)
    'drp' : Rotation pitch misalignment
    'dry' : Rotation yaw misalignment
    """
    return ["dx", "dy", "dr", "drp", "dry"]


def si_sectors():
    """Default sectors of SIRIUS ring."""
    return list(range(1, 21))


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


def si_elems():
    """Default SIRIUS elements families."""
    return si_dipoles() + si_quadrupoles() + si_sextupoles()


__model = None
__bpmidx = None
__spos = None


def __update_model():
    globals()["__model"].radiation_on = True
    globals()["__model"].vchamber_on = True
    globals()["__model"].cavity_on = True
    globals()["__spos"] = _find_spos(globals()["__model"], indices="closed")
    globals()["__bpmidx"] = _find_indices(
        globals()["__model"], "fam_name", "BPM"
    )
    # print("si_data -> model is", type(globals()["__model"]))


def __set_model(model=None):
    """."""
    # print("entered si_data set model")
    if model is None:
        # print("si_data model -> none")
        globals()["__model"] = _pymodels.si.create_accelerator()
    else:
        # print("si_data model -> model")
        globals()["__model"] = _deepcopy(model)
    __update_model()


def get_model():
    """SIRIUS model.

    Returns:
        pyaccel.accelerator: standart SIRIUS model
    """
    if __model is None:
        __set_model()
    return __model


def si_spos():
    """Default SIRIUS longitudinal coordinates of the lattice elements."""
    if __model is None:
        __set_model()
    return _deepcopy(__spos)


def si_famdata():
    """Default SIRIUS families data."""
    if __model is None:
        __set_model()
    return _deepcopy(_pymodels.si.families.get_family_data(__model))


def si_bpmidx():
    """Default SIRIUS BPM's indices."""
    if __model is None:
        __set_model()
    return _deepcopy(__bpmidx)
