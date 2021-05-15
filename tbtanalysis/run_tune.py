#!/usr/bin/env python-sirius

"""."""

import numpy as np
import matplotlib.pyplot as plt
from pymodels import si
import pyaccel
from pyaccel.naff import naff_general as _naff_general


from lib import create_tbt



def analysis_tune(folder, name, kicktype, kickidx):
    
    tbt = create_tbt(folder+fname, kicktype)
    data = tbt.select_get_traj()
    tbt.calc_fft(data, peak_frac=0.99)


if __name__ == "__main__":
    
    save_flag = True
    print_flag = True
    plot_flag = False

    # --- multibunch horizontal - after cycling ---

    folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    fname = 'tbt_data_horizontal_m050urad_after_cycle.pickle'
    parms = analysis_tune(folder, fname, kicktype='CHROMX', kickidx=0)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m100urad_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m150urad_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m200urad_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m250urad_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # # --- multibunch small amplitude kicks - before cycling ---
    
    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m005urad_chrom=2p5.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m010urad_chrom=2p5.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_horizontal_m025urad_chrom=2p5.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- multibunch vertical - after cycling ---

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_100volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_150volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_200volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_250volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_300volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_350volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_400volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-03-SI_commissioning-equilibrium_parameters_tbt/'
    # fname = 'tbt_data_vertical_450volts_after_cycle.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # --- single-bunch horizontal ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m025urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m050urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m100urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m150urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m200urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # fname = 'tbt_data_horizontal_m250urad_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMX', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)


    # --- single-bunch vertical ---

    # folder = '2021-05-04-SI_commissioning-equilibrium_parameters_tbt_single_bunch/'
    # !!! halted! fname = 'tbt_data_vertical_050volts_single_bunch.pickle'
    # parms = analysis_chrom(folder, fname, kicktype='CHROMY', kickidx=0, save_flag=save_flag, print_flag=print_flag, plot_flag=plot_flag)
    

