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
    error_configs = [dips_error, quads_error, quads_skew_error,
                     sexts_error, bpms_error, girder_error]

    nr_mach = 20

    # Create nominal model and get family data
    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)

    # Create GenerateErrors object
    generate_errors = lattice_errors.GenerateErrors()
    generate_errors.nr_mach = nr_mach
    generate_errors.generate_new_seed()
    generate_errors.seed = 614080
    generate_errors.reset_seed()
    print(generate_errors.seed)
    generate_errors.famdata = famdata
    generate_errors.error_configs = error_configs
    generate_errors.cutoff = 1
    _ = generate_errors.generate_errors(save_errors=True)

    # Configure parameters:
    machineparams = lattice_errors.MachinesParams()

    # If running for the first time there will be no jacobian to load
    machineparams.load_jacobians = True
    machineparams.save_jacobians = False

    # Do corrections after application of multipole errors
    machineparams.do_multipoles_corr = True

    # Do optics correction
    machineparams.do_optics_corr = True

    # Do coupling correction
    machineparams.do_coupling_corr = True

    # Do singular value ramp
    machineparams.do_singval_ramp = True

    # Configure parameters for orbit correction
    machineparams.orbcorr_params.minsingval = 0
    machineparams.orbcorr_params.tikhonovregconst = 1
    machineparams.orbcorr_params.orbrmswarnthres = 20e-6  # rad
    machineparams.orbcorr_params.numsingval = 281
    machineparams.orbcorr_params.maxnriters = 15
    machineparams.orbcorr_params.convergencetol = 1e-9
    machineparams.orbcorr_params.maxdeltakickch = 50e-6
    machineparams.orbcorr_params.maxdeltakickcv = 50e-6
    machineparams.orbcorr_params.maxkickch = 300e-6  # rad
    machineparams.orbcorr_params.maxkickcv = 300e-6  # rad

    # Configure parameters for optics correction
    machineparams.optcorr_params.nr_singval = 80
    machineparams.optcorr_params.tolerance = 1e-8

    # Configure parameters for coupling correction
    machineparams.coupcorr_params.nr_singval = 80
    machineparams.coupcorr_params.tolerance = 1e-8
    machineparams.coupcorr_params.weight_dispy = 1e5

    # Create GenerateMachines object
    random_machines = lattice_errors.GenerateMachines(machineparams)

    # Define number of machines
    random_machines.nr_mach = nr_mach

    # Nominal model for reference
    random_machines.nominal_model = model

    # Family data of nominal model
    random_machines.famdata = famdata
    random_machines.seed = generate_errors.seed

    # Load errors
    errors = generate_errors.load_error_file(
        str(nr_mach) + '_errors_seed_' + str(generate_errors.seed))
    random_machines.fam_errors_dict = errors

    random_machines.configure_corrections()

    # Apply errors in all machines
    nr_steps = 3
    data_mach = random_machines.generate_machines(nr_steps=nr_steps)
