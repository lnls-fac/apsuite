#!/usr/bin/env python3
"""."""

import numpy as np
import matplotlib.pyplot as plt
import pyaccel
import pymodels

from apsuite.optics_analysis.tune_correction import TuneCorr


class Params:
    """."""

    def __init__(self):
        """."""
        # --- input ---
        self.nr_particles = 1000
        self.nr_turns = 300

        self.si_cavity_on = True
        self.si_radiation_on = True
        self.si_vchamber_on = True
        self.si_coupling = 1 * 0.01
        self.si_tunex_goal = 49.115
        self.si_tuney_goal = 14.213
        self.si_injdpkckr = -0.35 / 1000  # [rad]
        self.si_sfa0_frac = 0
        self.si_plot_turns = None
        self.plot_title = None


def si_create_model(params):
    """."""
    print('--- creating model...')

    lat_mod = pymodels.si.lattice
    si_lattice = lat_mod.create_lattice()

    # fix bc high field vaccum limits
    indices = pyaccel.lattice.find_indices(si_lattice, 'fam_name', 'BC')
    for i in indices:
        ang = si_lattice[i].angle
        lng = si_lattice[i].length
        rho = lng/ang
        if rho < 5:
            si_lattice[i].hmax = +4/1000
            si_lattice[i].hmin = -4/1000
            si_lattice[i].vmax = +4/1000
            si_lattice[i].vmin = -4/1000
        # print(i, si_lattice[i].fam_name, rho)

    # hmax = 1e3 * np.array(pyaccel.lattice.get_attribute(si_lattice, 'hmax'))
    # hmin = 1e3 * np.array(pyaccel.lattice.get_attribute(si_lattice, 'hmin'))
    # plt.plot(hmin, '.', color='black')
    # plt.plot(hmax, '.', color='black')
    # plt.show()

    accelerator = pyaccel.accelerator.Accelerator(
        lattice=si_lattice,
        energy=lat_mod.energy,
        harmonic_number=lat_mod.harmonic_number,
        cavity_on=params.si_cavity_on,
        radiation_on=params.si_radiation_on,
        vchamber_on=params.si_vchamber_on,
    )
    params.si_model = accelerator

    # for i in range(len(params.si_model)):
    #     if params.si_model[i].fam_name in ('mia', 'mib', 'mc', 'mip', 'l240'):
    #         print(i, params.si_model[i].fam_name)

    # raise

    # for i in range(1900, 1950):
    #     print(i)
    #     print(accelerator[i])
    #     print()

    # raise

def si_correct_tunes(params):
    """."""
    # correct tune
    print('--- correcting si tunes...')
    tunecorr = TuneCorr(
        params.si_model, 'SI', method='Proportional', grouping='TwoKnobs')
    tunecorr.get_tunes(params.si_model)
    print('    tunes init  : ', tunecorr.get_tunes(params.si_model))
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
        model=params.si_model,
        goal_parameters=np.array([params.si_tunex_goal, params.si_tuney_goal]),
        jacobian_matrix=tunemat)
    print('    tunes final : ', tunecorr.get_tunes(params.si_model))


def si_bunch_generate(params):
    """."""
    # bunch equilibrium in SI
    si_coupling = params.si_coupling
    nr_particles = params.nr_particles
    si_model = params.si_model

    print('--- calculating si equilibrium parameters...')
    si_eparam = pyaccel.optics.EquilibriumParameters(si_model)
    emit0 = si_eparam.emit0
    emitx = (1.0 / (1.0 + si_coupling)) * emit0
    emity = (si_coupling / (1.0 + si_coupling)) * emit0
    sigmae = si_eparam.espread0
    sigmas = si_eparam.bunch_length
    print('    emitax: {:.4f} nm.rad'.format(emitx/1e-9))
    print('    emitay: {:.4f} nm.rad'.format(emity/1e-9))
    print('    sigmae: {:.4f} %'.format(sigmae*100))
    print('    sigmas: {:.4f} mm'.format(sigmas*1e3))

    # model at specific starting point
    print('--- calculating si twiss...')
    twiss, _ = pyaccel.optics.calc_twiss(si_model)
    # bc = pyaccel.lattice.find_indices(
    #     si_model, 'fam_name', 'BC')
    idx = 0

    twiss0 = twiss[idx]

    # generate bunch at SI extraction point
    print('--- generating bunch...')
    bunch = pyaccel.tracking.generate_bunch(
        emitx, emity, sigmae, sigmas, twiss0, nr_particles-1, cutoff=100)
    zero = np.zeros((6, 1))
    bunch = np.hstack((zero, bunch))

    return bunch


def si_change_sfa0(params):
    """."""
    print('--- changing SFA0...')

    chromx, chromy = pyaccel.optics.get_chromaticities(params.si_model)
    print('   chroms init  : ', chromx, chromy)

    idx = pyaccel.lattice.find_indices(params.si_model, 'fam_name', 'SFA0')
    idx = idx[-1]
    params.si_model[idx].polynom_b[2] *= params.si_sfa0_frac
    # print(params.si_model[idx].polynom_b)

    chromx, chromy = pyaccel.optics.get_chromaticities(params.si_model)
    print('   chroms final : ', chromx, chromy)


def si_track(params, bunch):
    """."""
    si_model = params.si_model

    print('--- one-turn tracking with dipolear kick')
    # track one turn with kick
    idx = pyaccel.lattice.find_indices(
        si_model, 'fam_name', 'InjDpKckr')[0]
    si_model[idx].hkick_polynom = params.si_injdpkckr  # add dipolar kick
    rout, lost_flag, lost_element, lost_plane = \
        pyaccel.tracking.line_pass(
            si_model, bunch, 'closed')
    _ = lost_flag, lost_element, lost_plane
    si_model[idx].hkick_polynom = 0.0  # zero dipolar kick
    bunch = rout[:, :, -1]

    # plt.plot(1e3*rout[1, 0, :])
    # plt.xlabel('element')
    # plt.ylabel('px [mrad]')
    # plt.show()

    # tracking SI turns
    print('--- tracking in si...')
    rout, lost_flag, lost_turn, lost_element, lost_plane = \
        pyaccel.tracking.ring_pass(
            si_model, bunch, params.nr_turns, turn_by_turn='open')
    _ = lost_flag, lost_element, lost_plane

    print(lost_turn)
    lostat = np.zeros(len(si_model))
    print(lostat)
    for i in range(len(lost_turn)):
        if lost_turn[i] < params.nr_turns:
            lostat[lost_element[i]] += 1

    sidx = np.argsort(lostat)
    idx = np.array([i for i in range(len(lostat))])
    idx = idx[sidx]
    lostat = lostat[sidx]
    print('--- where particles are lost ---')
    for ind, nrl in zip(idx, lostat):
        if nrl > 0:
            print('{:04d} {:<15s} {}'.format(ind, si_model[ind].fam_name, nrl))


    # plot number of surviving particles along turns
    params.lost_turn = lost_turn
    plot_lost(params, lost_turn)

    # plot phase space in each turn
    plot_phase_space(params, rout)


def plot_bunch(text, bunch):
    """."""
    print('--- plotting ' + text)
    plt.plot(1e3*bunch[0, :], 1e3*bunch[1, :], '.', label='X')
    plt.plot(1e3*bunch[2, :], 1e3*bunch[3, :], '.', label='Y')
    plt.xlabel('r [mm]')
    plt.ylabel('p [mrad]')
    plt.legend()
    plt.grid()
    plt.title(text)
    plt.show()


def plot_phase_space(params, rout):
    """."""
    nr_turns = rout.shape[2]
    for turn in range(nr_turns):
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', color=[0.9, 0.9, 0.9])

    for turn in params.si_plot_turns:
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', label='turn = {}'.format(turn))

    plt.xlabel('rx [mm]')
    plt.ylabel('px [mrad]')
    plt.grid()
    plt.legend()
    strf = 'Phase Space ' + params.plot_title + '(nr_particles: {}, nr_turns {})'
    plt.title(strf.format(params.nr_particles, params.nr_turns))
    plt.show()


def plot_lost(params, lost_turn):
    """."""
    print('--- plotting lost data...')
    # nr_particles, nr_turns = rout.shape[-2:]
    lost_turn = np.array(lost_turn)
    sel = lost_turn == params.nr_turns-1  # trick
    lost_turn[sel] = params.nr_turns + 1  # trick

    survivals = params.nr_particles * np.ones(params.nr_turns)
    for turn in range(params.nr_turns):
        nrl = np.sum(lost_turn == turn)
        if turn == 0:
            survivals[turn] = params.nr_particles - nrl
        else:
            survivals[turn] = survivals[turn-1] - nrl

    survivals *= 100/params.nr_particles
    plt.plot(survivals, 'o')
    plt.xlabel('turn')
    plt.ylabel('surviving [%]')
    plt.title('Surviving particles {}'.format(params.plot_title))
    plt.grid()
    plt.show()


def run():
    """."""
    # define parameters
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.si_coupling = 1 * 0.01
    params.si_tunex_goal = 49.115
    params.si_tuney_goal = 14.213
    params.si_injdpkckr = -0.2 / 1000  # [rad]
    params.si_plot_turns = (0, 1, 2, 3, 6, 9, 15, 18, 24, 27, 33, 36)
    params.si_sfa0_frac = 0.0
    params.plot_title = 'Kick -0.2mrad, SFA 0 %'

    si_create_model(params)
    si_correct_tunes(params)
    si_change_sfa0(params)

    bunch = si_bunch_generate(params)
    plot_bunch('initial bunch', bunch)
    si_track(params, bunch)


def si_lattice_shift(params, idx):
    """."""
    lat_mod = pymodels.si.lattice
    si_lattice = lat_mod.create_lattice()

    # add obstruction
    # indices = pyaccel.lattice.find_indices(si_lattice, 'fam_name', 'mc')
    for i in range(159, 171):
        # si_lattice[i].hmax += params.hmax_delta
        # si_lattice[i].hmin += params.hmin_delta
        si_lattice[i].hmax = +4/1000 + params.hmax_delta
        si_lattice[i].hmin = -4/1000 - params.hmin_delta

    si_lattice = pyaccel.lattice.shift(si_lattice, idx)

    accelerator = pyaccel.accelerator.Accelerator(
        lattice=si_lattice,
        energy=lat_mod.energy,
        harmonic_number=lat_mod.harmonic_number,
        cavity_on=params.si_cavity_on,
        radiation_on=params.si_radiation_on,
        vchamber_on=params.si_vchamber_on,
    )
    return accelerator



# def run():
#     """."""
#     params = Params()
#     params.nr_particles = 100
#     params.nr_turns = 300
#     params.si_cavity_on = True
#     params.si_radiation_on = True
#     params.si_vchamber_on = True
#     params.si_coupling = 1*0.01
#     # NOTE: pos and injection angle can be varied using these
#     # two parameters
#     params.hmax_delta = -0.5/1000 * 6  # [m]
#     params.hmin_delta = +0.5/1000 * 0  # [m]

#     # generate bunch and model
#     bunch, si_model = si_bunch_generate(params)
#     plot_bunch('bunch at begining of high field BC', bunch)

#     # add dipolar kick
#     idx = pyaccel.lattice.find_indices(
#         si_model, 'fam_name', 'InjDpKckr')[0]
#     si_model[idx].hkick_polynom = -0.35/1000

#     # track one turn
#     rout0, lost_flag, lost_element, lost_plane = \
#         pyaccel.tracking.line_pass(
#             si_model, bunch, 'closed')
#     _ = lost_flag, lost_element, lost_plane
#     bunch = rout0[:, :, -1]

#     # data = 1e3*rout0[1, 0, :]
#     # print(data)
#     # plt.plot(data)
#     # plt.show()
#     # return

#     # turn dipolar kick off
#     si_model[idx].hkick_polynom = 0

#     # tracking SI turns
#     print('--- tracking in si...')
#     rout, lost_flag, lost_turn, lost_element, lost_plane = \
#         pyaccel.tracking.ring_pass(
#             si_model, bunch, params.nr_turns, turn_by_turn='open')
#     _ = lost_flag, lost_element, lost_plane

#     print(lost_turn)

#     # plot number of surviving particles along turns
#     params.lost_turn = lost_turn
#     plot_lost(params, lost_turn)

#     # plot phase space in each turn
#     plot_phase_space(params, rout)


run()
