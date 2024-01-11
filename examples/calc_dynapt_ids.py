#!/usr/bin/env python-sirius

import numpy as np

import pymodels
from apsuite import lattice_errors
import idanalysis.optics as opt

from idanalysis.analysis import StorageRingAnalysis
from idanalysis.idsdata import DELTA52Data

from mathphys.functions import save_pickle, load_pickle


if __name__ == "__main__":
    # Create nominal model and get family data
    model = pymodels.si.create_accelerator()
    model.cavity_on = False
    model.radiation_on = 0
    model.vchamber_on = False
    famdata = pymodels.si.families.get_family_data(model)

    seed = 902080

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

    # Force correction
    machineparams.force_orb_correction = True

    # Configure parameters for orbit correction
    machineparams.orbcorr_params.minsingval = 0.3
    machineparams.orbcorr_params.tikhonovregconst = 1
    machineparams.orbcorr_params.orbrmswarnthres = 20e-6  # rad
    machineparams.orbcorr_params.numsingval = 281
    machineparams.orbcorr_params.maxnriters = 30
    machineparams.orbcorr_params.convergencetol = 1e-9
    machineparams.orbcorr_params.maxdeltakickch = 50e-6
    machineparams.orbcorr_params.maxdeltakickcv = 50e-6
    machineparams.orbcorr_params.maxkickch = 300e-6  # rad
    machineparams.orbcorr_params.maxkickcv = 300e-6  # rad

    # Configure parameters for optics correction
    machineparams.optcorr_params.nr_singval = 65
    machineparams.optcorr_params.tolerance = 1e-8

    # Configure parameters for coupling correction
    machineparams.coupcorr_params.nr_singval = 40
    machineparams.coupcorr_params.tolerance = 1e-8
    machineparams.coupcorr_params.weight_dispy = 1e5

    # Create GenerateMachines object
    nr_mach = 5
    random_machines = lattice_errors.GenerateMachines(machineparams)
    random_machines.nr_mach = nr_mach
    random_machines.nominal_model = model
    random_machines.famdata = famdata
    random_machines.seed = seed

    random_machines.configure_corrections()

    nr_steps = 5

    deltadata = DELTA52Data()

    filenames = [

        deltadata.get_kickmap_filename(meas_flag=True, dp=0, dgv=0),
        deltadata.get_kickmap_filename(meas_flag=True, dp=0, dgv=13.125),
        deltadata.get_kickmap_filename(meas_flag=True, dp=0, dgv=26.25),
        deltadata.get_kickmap_filename(meas_flag=True, dp=-13.125, dgv=13.125),
        deltadata.get_kickmap_filename(meas_flag=True, dp=-13.125, dgv=26.25),
        deltadata.get_kickmap_filename(meas_flag=True, dp=-26.25, dgv=13.125),
        deltadata.get_kickmap_filename(meas_flag=True, dp=-26.25, dgv=26.25)

    ]

    for i, kmap_fname in enumerate(filenames):
        ring_anl = StorageRingAnalysis()
        ring_anl.add_id_to_model(
            kmap_fname=kmap_fname, subsec=deltadata.subsecs[0],
            fam_name=deltadata.id_famname)

        random_machines.ids = ring_anl.ids
        random_machines.corr_ids(sufix=str(i))
        fname = "SI_" + str(nr_mach) + "_machines_seed_" + str(seed)
        fname += "_" + deltadata.id_famname + "_symm" + str(i) + ".pickle"
        data_mach = load_pickle(fname)

        nturns = 2048
        nrtheta = 9
        mindeltar = 0.1e-3
        x, y = np.zeros((20, nrtheta)), np.zeros((20, nrtheta))
        for mach in np.arange(0, nr_mach):
            msg = (
                "calculating DA for config "
                + str(i)
                + " and machine "
                + str(mach + 1)
            )
            print(msg)
            model = data_mach[mach]["model"]
            model.radiation_on = 1
            model.cavity_on = True
            model.vchamber_on = True
            x_, y_ = opt.calc_dynapt_xy(
                model,
                nrturns=nturns,
                nrtheta=nrtheta,
                mindeltar=mindeltar,
                print_flag=False,
            )

            x[mach, :] = x_
            y[mach, :] = y_

        filename = "dynapt_" + str(seed) + "_config" + str(i)
        save_pickle((x, y), filename, overwrite=True)
