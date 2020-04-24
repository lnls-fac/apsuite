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
        self.bo_coupling = 1*0.01

        self.si_cavity_on = True
        self.si_radiation_on = True
        self.si_vchamber_on = True
        self.si_coupling = 1*0.01

        # NOTE: pos and injection angle can be varied using these
        # two parameters
        error_inj_rx = 0.0/1000  # [m]
        error_inj_px = 0.0/1000  # [rad]

        self.transf_ts2si_drx = -17.35/1000 + error_inj_rx  # [m]
        self.transf_ts2si_dpx = 2.4/1000 + error_inj_px  # [rad]

        self.nlk_kick_error = 0.3/1000 * 0  # [rad]
        self.hmax_delta = -4.5/1000 * 0  # [m]
        self.hmin_delta = +4.5/1000 * 0  # [m]

        # --- output ---
        self.bunch_si_injpoint = None
        self.bunch_si_after_nlk = None
        self.lost_turn = None


def si_bunch_generate(params):
    """."""
    # bunch equilibrium in SI
    si_coupling = params.si_coupling
    nr_particles = params.nr_particles
    si_model = pymodels.si.create_accelerator()

    # correct tune
    print('--- correcting si tunes...')
    tunecorr = TuneCorr(
        si_model, 'SI', method='Proportional', grouping='TwoKnobs')
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
        model=si_model,
        goal_parameters=np.array([49.115, 14.213]),
        jacobian_matrix=tunemat)
    print('    ', tunecorr.get_tunes(si_model))

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
    idx = 159

    # print(idx)
    # return
    # for i in range(144, 186):
    #     ang = si_model[i].angle
    #     len = si_model[i].length
    #     rad = -1 if ang == 0 else len / ang
    #     print(i, si_model[i].fam_name, si_model[i].angle, rad)
    # return

    twiss0 = twiss[idx]

    # generate bunch at SI extraction point
    print('--- generating bunch at si center...')
    bunch = pyaccel.tracking.generate_bunch(
        emitx, emity, sigmae, sigmas, twiss0, nr_particles-1, cutoff=100)
    zero = np.zeros((6, 1))
    bunch = np.hstack((zero, bunch))

    # shift model
    si_model = si_lattice_shift(params, idx)

    return bunch, si_model


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
    plt.title('Surviving particles')
    plt.grid()
    plt.show()


def plot_phase_space(params, rout):
    """."""
    nr_turns = rout.shape[2]
    for turn in range(nr_turns):
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', color=[0.9, 0.9, 0.9])

    # for turn in (0, 1, 2, 3, 4, 5, 6, 15, 24, 33):
    for turn in (0, 6, 9, 15, 18, 24, 27, 33, 36):
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', label='turn = {}'.format(turn))

    # minnr = min(10, nr_turns)
    # rx_ = 1e3*rout[0, :, minnr]
    # px_ = 1e3*rout[1, :, minnr]
    # plt.plot(rx_, px_, '.', color='grey', label='turn > {}'.format(minnr))
    # for turn in range(minnr):
    #     rx_ = 1e3*rout[0, :, turn]
    #     px_ = 1e3*rout[1, :, turn]
    #     plt.plot(rx_, px_, '.', label='turn = {}'.format(turn))
    plt.xlabel('rx [mm]')
    plt.ylabel('px [mrad]')
    plt.grid()
    plt.legend()
    strf = 'Phase Space at beg of high field BC (nr_particles: {}, nr_turns {})'
    plt.title(strf.format(params.nr_particles, params.nr_turns))
    plt.show()


def run():
    """."""
    params = Params()
    params.nr_particles = 100
    params.nr_turns = 300
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.si_coupling = 1*0.01
    # NOTE: pos and injection angle can be varied using these
    # two parameters
    params.hmax_delta = -0.5/1000 * 6  # [m]
    params.hmin_delta = +0.5/1000 * 0  # [m]

    # generate bunch and model
    bunch, si_model = si_bunch_generate(params)
    plot_bunch('bunch at begining of high field BC', bunch)

    # add dipolar kick
    idx = pyaccel.lattice.find_indices(
        si_model, 'fam_name', 'InjDpKckr')[0]
    si_model[idx].hkick_polynom = -0.35/1000

    # track one turn
    rout0, lost_flag, lost_element, lost_plane = \
        pyaccel.tracking.line_pass(
            si_model, bunch, 'closed')
    _ = lost_flag, lost_element, lost_plane
    bunch = rout0[:, :, -1]

    # data = 1e3*rout0[1, 0, :]
    # print(data)
    # plt.plot(data)
    # plt.show()
    # return

    # turn dipolar kick off
    si_model[idx].hkick_polynom = 0

    # tracking SI turns
    print('--- tracking in si...')
    rout, lost_flag, lost_turn, lost_element, lost_plane = \
        pyaccel.tracking.ring_pass(
            si_model, bunch, params.nr_turns, turn_by_turn='open')
    _ = lost_flag, lost_element, lost_plane

    print(lost_turn)

    # plot number of surviving particles along turns
    params.lost_turn = lost_turn
    plot_lost(params, lost_turn)

    # plot phase space in each turn
    plot_phase_space(params, rout)


run()
