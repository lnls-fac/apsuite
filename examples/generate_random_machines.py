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

    # Configure errors
    dips_error = lattice_errors_new.DipolesErrors()
    quads_error = lattice_errors_new.QuadsErrors()
    quads_skew_error = lattice_errors_new.QuadsSkewErrors()
    sexts_error = lattice_errors_new.SextsErrors()
    girder_error = lattice_errors_new.GirderErrors()
    bpms_error = lattice_errors_new.BpmsErrors()
    error_configs = [dips_error, quads_error, sexts_error, quads_skew_error,
                     bpms_error, girder_error]

    # Create nominal model and get family data
    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)

    # Create manage errors object
    lattice_errors = lattice_errors_new.ManageErrors()
    nr_mach = 20
    lattice_errors.nr_mach = nr_mach
    lattice_errors.nominal_model = model
    lattice_errors.famdata = famdata
    lattice_errors.reset_seed()
    lattice_errors.error_configs = error_configs
    lattice_errors.cutoff = 1

    # Generate errors and load file
    errors = lattice_errors.generate_errors(save_errors=True)
    lattice_errors.load_error_file(
        str(nr_mach) + '_errors_seed_'+str(lattice_errors.seed))

    # If running for the first time there will be no jacobian to load
    lattice_errors.load_jacobians = False
    lattice_errors.save_jacobians = True

    # Configure orbit corretion
    lattice_errors.orbcorr_params.minsingval = 0.2
    lattice_errors.orbcorr_params.maxnriters = 15
    lattice_errors.orbcorr_params.tolerance = 1e-9
    lattice_errors.orbcorr_params.maxdeltakickch = 50e-6
    lattice_errors.orbcorr_params.maxdeltakickcv = 50e-6
    lattice_errors.orbcorr_params.maxkickch = 300e-6  # rad
    lattice_errors.orbcorr_params.maxkickcv = 300e-6  # rad
    lattice_errors.configure_corrections()

    # Apply errors in all machines
    nr_steps = 5
    data_mach = lattice_errors.generate_machines(nr_steps=nr_steps)
