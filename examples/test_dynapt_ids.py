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

    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)

    seed = 4738

    nr_mach = 20
    # create manage errors object
    lattice_errors = lattice_errors_new.ManageErrors()
    lattice_errors.nr_mach = nr_mach
    lattice_errors.nominal_model = model
    lattice_errors.famdata = famdata

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

    lattice_errors.ocorr_params.minsingval = 0.2
    lattice_errors.ocorr_params.maxnriters = 15
    lattice_errors.ocorr_params.tolerance = 1e-9

    filenames = [


        '/home/gabriel/repos/idanalysis/scripts/delta52/results/model/kickmaps/kickmap-ID_width45_phasepos00p000_gap00p0-shifted_on_axis.txt',

        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg13p125_gap26p2-shifted_on_axis.txt',

        '/home/gabriel/repos/idanalysis/scripts/delta52/results/measurements/kickmaps/kickmap-ID_width45_phaseneg26p250_gap13p1-shifted_on_axis.txt',

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

        nturns = 2048
        nrtheta = 9
        mindeltar = 0.1e-3
        x, y = np.zeros((20, nrtheta)), np.zeros((20, nrtheta))
        for mach in np.arange(0, 20):
            msg = 'calculating DA for config ' + str(i) + ' and machine ' + str(mach+1)
            print(msg)
            model = data_mach[mach]['model']
            model = lattice_errors.insert_kickmap(model)
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

        filename = 'dynapt_DELTA52_nc_' + str(seed) + '_config' + str(i)
        save_pickle((x, y), filename, overwrite=True)
