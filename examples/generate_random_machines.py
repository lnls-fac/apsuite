#!/usr/bin/env python-sirius

import pymodels
from apsuite import lattice_errors

if __name__ == '__main__':

    # Configure errors
    dips_error = lattice_errors.DipolesErrors()
    quads_error = lattice_errors.QuadsErrors()
    quads_skew_error = lattice_errors.QuadsSkewErrors()
    sexts_error = lattice_errors.SextsErrors()
    girder_error = lattice_errors.GirderErrors()
    bpms_error = lattice_errors.BPMErrors()
    error_configs = [dips_error, quads_error, sexts_error, quads_skew_error,
                     bpms_error, girder_error]

    # Create nominal model and get family data
    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)

    # Create GenerateErrors object
    nr_mach = 20
    generate_errors = lattice_errors.GenerateErrors()
    generate_errors.nr_mach = nr_mach
    generate_errors.generate_new_seed()
    print(generate_errors.seed)
    generate_errors.famdata = famdata
    generate_errors.error_configs = error_configs
    generate_errors.cutoff = 1
    *_, = generate_errors.generate_errors(save_errors=True)
    errors = generate_errors.load_error_file(
        str(nr_mach) + '_errors_seed_' + str(generate_errors.seed))

    # Create GenerateMachines object
    random_machines = lattice_errors.GenerateMachines()
    random_machines.nr_mach = nr_mach
    random_machines.nominal_model = model
    random_machines.famdata = famdata
    random_machines.seed = generate_errors.seed
    random_machines.fam_errors_dict = errors

    # If running for the first time there will be no jacobian to load
    random_machines.load_jacobians = True
    random_machines.save_jacobians = False

    # Configure orbit corretion
    random_machines.orbcorr_params.minsingval = 0.2
    random_machines.orbcorr_params.maxnriters = 15
    random_machines.orbcorr_params.tolerance = 1e-9
    random_machines.orbcorr_params.maxdeltakickch = 50e-6
    random_machines.orbcorr_params.maxdeltakickcv = 50e-6
    random_machines.orbcorr_params.maxkickch = 300e-6  # rad
    random_machines.orbcorr_params.maxkickcv = 300e-6  # rad
    random_machines.configure_corrections()

    # Apply errors in all machines
    nr_steps = 5
    data_mach = random_machines.generate_machines(nr_steps=nr_steps)
