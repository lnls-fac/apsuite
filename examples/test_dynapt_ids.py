#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
import pymodels
from apsuite import lattice_errors_new
from apsuite.orbcorr import OrbitCorr, CorrParams
from mathphys.functions import save_pickle, load_pickle
import idanalysis.optics as opt

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
    seed = 424242

    # create manage errors object
    lattice_errors = lattice_errors_new.ManageErrors()
    lattice_errors.nr_mach = 20
    lattice_errors.nominal_model = model
    lattice_errors.famdata = famdata
    # lattice_errors.reset_seed()
    lattice_errors.seed = seed
    lattice_errors.error_configs = error_configs
    lattice_errors.cutoff = 1
    errors = lattice_errors.generate_errors(save_errors=True)
    lattice_errors.load_error_file('20_errors_seed_'+str(lattice_errors.seed))

    lattice_errors.load_jacobians = True
    lattice_errors.save_jacobians = False
    lattice_errors.ramp_sextupoles = True
    print(lattice_errors.orbcorr_dim)
    print('ramp corrs:', lattice_errors._ramp_corrections)
    lattice_errors.configure_corrections()

    if lattice_errors.ramp_sextupoles is True:
        nr_steps = 3
    else:
        nr_steps = 8

    lattice_errors.ramp_with_ids = False
    lattice_errors.nr_mach = 20
    # correction with girder errors: 8 steps
    lattice_errors.apply_girder = True
    lattice_errors.rescale_girder = 1

    lattice_errors.do_bba = True
    lattice_errors.ocorr_params.minsingval = 0.5
    lattice_errors.ocorr_params.maxnriters = 10
    lattice_errors.ocorr_params.tolerance = 1e-9
    data_mach = lattice_errors.generate_machines(nr_steps=nr_steps)

    filenames = [
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap00p0-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap13p1-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap26p2-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg13p125_gap13p1-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg13p125_gap26p2-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg26p250_gap13p1-shifted_on_axis.txt',
        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg26p250_gap26p2-shifted_on_axis.txt'
    ]


    # Calc DA without ID symmetrization
    for i, filename in enumerate(filenames):
        ids = list()
        IDModel = pymodels.si.IDModel
        delta52 = IDModel(
            subsec=IDModel.SUBSECTIONS.ID10SB,
            file_name=filename,
            fam_name='DELTA52', nr_steps=40,
            rescale_kicks=1, rescale_length=1)
        ids.append(delta52)

        lattice_errors.ids = ids

        #  Calc DA
        nturns = 2048
        nrtheta = 9
        mindeltar = 0.1e-3
        x, y = np.zeros((20, nrtheta)), np.zeros((20, nrtheta))
        for mach in np.arange(0, 20):
            model_ = lattice_errors.models[mach]
            model = lattice_errors.insert_kickmap(model_)
            model.radiation_on = 0
            model.cavity_on = False
            model.vchamber_on = True
            x_, y_ = opt.calc_dynapt_xy(model,
                                        nrturns=nturns,
                                        nrtheta=nrtheta,
                                        mindeltar=mindeltar,
                                        print_flag=False)

            x[mach, :] = x_
            y[mach, :] = y_
        filename = 'x_y_dynapt_test_424242_DELTA52_nonsymm_config_'
        filename += str(i)
        save_pickle((x, y), filename)

    # Calc DA with ID symmetrization
    for i, filename in enumerate(filenames):
        ids = list()
        IDModel = pymodels.si.IDModel
        delta52 = IDModel(
            subsec=IDModel.SUBSECTIONS.ID10SB,
            file_name=filename,
            fam_name='DELTA52', nr_steps=40,
            rescale_kicks=1, rescale_length=1)
        ids.append(delta52)

        lattice_errors.ids = ids
        lattice_errors.corr_ids()

        #  Calc DA
        nturns = 2048
        nrtheta = 9
        mindeltar = 0.1e-3
        x, y = np.zeros((20, nrtheta)), np.zeros((20, nrtheta))
        for mach in np.arange(0, 20):
            model = lattice_errors.models[mach]
            model.radiation_on = 0
            model.cavity_on = False
            model.vchamber_on = True
            x_, y_ = opt.calc_dynapt_xy(model,
                                        nrturns=nturns,
                                        nrtheta=nrtheta,
                                        mindeltar=mindeltar,
                                        print_flag=False)

            x[mach, :] = x_
            y[mach, :] = y_
        filename = 'x_y_dynapt_test_424242_DELTA52_symm_config_' + str(i)
        save_pickle((x, y), filename)
