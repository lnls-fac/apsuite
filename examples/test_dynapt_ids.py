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
    seed = 424242

    nr_mach = 20
    # create manage errors object
    lattice_errors = lattice_errors_new.ManageErrors()
    lattice_errors.nr_mach = nr_mach
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

    lattice_errors.ramp_with_ids = True
    lattice_errors.nr_mach = 20

    lattice_errors.apply_girder = True
    lattice_errors.rescale_girder = 1

    lattice_errors.do_bba = True
    lattice_errors.ocorr_params.minsingval = 0.5
    lattice_errors.ocorr_params.maxnriters = 15
    lattice_errors.ocorr_params.tolerance = 1e-9
    # data_mach = lattice_errors.generate_machines(nr_steps=nr_steps)

    filenames = [


        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap00p0-shifted_on_axis.txt',
        # '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap13p1-shifted_on_axis.txt',
        # '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phasepos00p000_gap26p2-shifted_on_axis.txt',
        # '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg13p125_gap13p1-shifted_on_axis.txt',
        # '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg13p125_gap26p2-shifted_on_axis.txt',
        # '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg26p250_gap13p1-shifted_on_axis.txt',

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

        fname = '20_machines_seed_' + str(seed) + '_sext_ramp.pickle'
        data_mach = load_pickle(fname)
        mach = 0
        model_ = data_mach[mach]['model']
        model = lattice_errors.insert_kickmap(model_)
        model.radiation_on = 0
        model.cavity_on = False
        model.vchamber_on = True

        dynapxy = DynapXY(model)
        dynapxy.params.x_nrpts = 40
        dynapxy.params.y_nrpts = 20
        dynapxy.params.nrturns = 2*1024
        print(dynapxy)
        dynapxy.do_tracking()
        dynapxy.process_data()
        fig, axx, ayy = dynapxy.make_figure_diffusion(
            orders=(1, 2, 3, 4),
            nuy_bounds=(14.12, 14.45),
            nux_bounds=(49.05, 49.50))

        figname = 'dynapt_DELTA52' + str(seed) + '_config_' + str(i) + '.png'
        fig.savefig(figname, dpi=300, format='png')
        fig.clf()
