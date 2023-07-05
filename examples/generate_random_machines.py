#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
import pymodels
from apsuite import lattice_errors_new
from apsuite.orbcorr import OrbitCorr, CorrParams
from mathphys.functions import save_pickle, load_pickle
import idanalysis.optics as opt
from apsuite.dynap import DynapXY

if __name__ == '__main__':

    dips_error = lattice_errors_new.DipolesErrors()
    quads_error = lattice_errors_new.QuadsErrors()
    quads_skew_error = lattice_errors_new.QuadsSkewErrors()
    sexts_error = lattice_errors_new.SextsErrors()
    girder_error = lattice_errors_new.GirderErrors()
    bpms_error = lattice_errors_new.BpmsErrors()
    error_configs = [dips_error, quads_error, sexts_error, quads_skew_error,
                     bpms_error, girder_error]

    # get family data
    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)
    # create a seed
    seed = 4738

    nr_mach = 20
    # create manage errors object
    lattice_errors = lattice_errors_new.ManageErrors()
    lattice_errors.nr_mach = nr_mach
    lattice_errors.nominal_model = model
    lattice_errors.famdata = famdata
    lattice_errors.reset_seed()
    lattice_errors.seed = seed
    lattice_errors.error_configs = error_configs
    lattice_errors.cutoff = 1
    errors = lattice_errors.generate_errors(save_errors=True)
    lattice_errors.load_error_file(
        str(nr_mach) + '_errors_seed_'+str(lattice_errors.seed))

    lattice_errors.load_jacobians = False
    lattice_errors.save_jacobians = True
    lattice_errors.configure_corrections()
    nr_steps = 3

    lattice_errors.ramp_with_ids = False

    lattice_errors.apply_girder = True
    lattice_errors.rescale_girder = 1

    lattice_errors.do_bba = True
    lattice_errors.do_opt_corr = True
    lattice_errors.ocorr_params.minsingval = 0.2
    lattice_errors.ocorr_params.maxnriters = 15
    lattice_errors.ocorr_params.tolerance = 1e-9
    lattice_errors.ocorr_params.maxdeltakickch = 50e-6
    lattice_errors.ocorr_params.maxdeltakickcv = 50e-6

    data_mach = lattice_errors.generate_machines(nr_steps=nr_steps)
