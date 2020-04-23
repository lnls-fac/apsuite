#!/usr/bin/env python3
"""."""


import numpy as np
import matplotlib.pyplot as plt
import pyaccel
import pymodels


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

        # NOTE: pos and injection angle can be varied using these
        # two parameters
        error_inj_rx = 0.0/1000  # [m]
        error_inj_px = 0.0/1000  # [rad]

        self.transf_ts2si_drx = -17.35/1000 + error_inj_rx  # [m]
        self.transf_ts2si_dpx = 2.4/1000 + error_inj_px  # [rad]

        self.nlk_kick_error = 0.3/1000 * 0  # [rad]
        self.scrapper_h_max_delta = -2.0/1000 * 0  # [m]
        self.scrapper_h_min_delta = +0.0/1000  # [m]

        # --- output ---
        self.bunch_si_injpoint = None
        self.bunch_si_after_nlk = None
        self.lost_turn = None


def si_bunch_injection(params=None):
    """."""
    # parameters
    if params is None:
        params = Params()
    print('--- defining parameters')
    print('    nr_particles : {}'.format(params.nr_particles))
    print('    nr_turns     : {}'.format(params.nr_turns))
    print('    bo_coupling  : {:.2f} %'.format(100*params.bo_coupling))

    # create initial bunch at BO extraction point.
    bunch = bo_bunch_generate(params.nr_particles, params.bo_coupling)
    plot_bunch('bunch at BO extraction point (EjeSeptF)', bunch)

    # transport bunch across TS
    bunch = ts_bunch_transport(params, bunch)
    params.bunch_si_injpoint = bunch
    plot_bunch('bunch in SI injection point', bunch)

    # transport bunch from injpoint to NLK, add kick
    bunch = si_transport_inj2nlk(params, bunch)
    params.bunch_si_after_nlk = bunch
    plot_bunch('bunch in SI right after NLK Kick', bunch)

    # create shifted SI model (right after NLK)
    si_model = si_lattice_shift_after_nlk(params)

    # tracking SI turns
    print('--- tracking in si...')
    rout, lost_flag, lost_turn, lost_element, lost_plane = \
        pyaccel.tracking.ring_pass(
            si_model, bunch, params.nr_turns, turn_by_turn='open')
    _ = lost_flag, lost_element, lost_plane

    # plot number of surviving particles along turns
    params.lost_turn = lost_turn
    plot_lost(params, lost_turn)

    # plot phase space in each turn
    plot_phase_space2(params, rout)


def si_transport_inj2nlk(params, bunch):
    """."""
    print('--- transporting in si from injection point to nlk...')
    # transport from InjSepF to InjNLKckr
    params.si_vchamber_on = False  # pinocchio vchamber not in the model?!
    si_model = si_lattice_shift_inj(params)
    params.si_vchamber_on = True

    nlk_idx = pyaccel.lattice.find_indices(
        si_model, 'fam_name', 'InjNLKckr')[0]

    # NOTE: This seems buggy!!!
    # inds = np.array([0 for i in range(nlk_idx + 1)])
    # bunch, lost_flag, lost_element, lost_plane = \
    #     pyaccel.tracking.line_pass(si_model, bunch, indices=ind)
    bunch, lost_flag, lost_element, lost_plane = \
        pyaccel.tracking.line_pass(si_model, bunch, indices='open')
    _ = lost_flag, lost_element, lost_plane

    # print tracking data for center particle
    strf = '    {:<12s} -  rx:{:+07.3f} mm  px:{:+07.3f} mrad, hmin: {:.3f}'
    for i in range(nlk_idx+1):
        rx_ = 1e3*bunch[0, 0, i]
        px_ = 1e3*bunch[1, 0, i]
        fname = si_model[i].fam_name
        hmin = 1e3*si_model[i].hmin
        print(strf.format(fname, rx_, px_, hmin))

    # add NLK kick
    bunch = bunch[:, :, nlk_idx+1]
    bunch[1, :] += -np.mean(bunch[1, :]) + params.nlk_kick_error
    return bunch


def si_lattice_shift_inj(params):
    """."""
    lat_mod = pymodels.si.lattice
    si_lattice = lat_mod.create_lattice()
    injseptf_idx = pyaccel.lattice.find_indices(
        si_lattice, 'fam_name', 'InjSeptF')[0]
    si_lattice = pyaccel.lattice.shift(si_lattice, injseptf_idx)
    accelerator = pyaccel.accelerator.Accelerator(
        lattice=si_lattice,
        energy=lat_mod.energy,
        harmonic_number=lat_mod.harmonic_number,
        cavity_on=params.si_cavity_on,
        radiation_on=params.si_radiation_on,
        vchamber_on=params.si_vchamber_on,
    )
    return accelerator


def si_lattice_shift_after_nlk(params):
    """."""
    lat_mod = pymodels.si.lattice
    si_lattice = lat_mod.create_lattice()
    nlk_idx = pyaccel.lattice.find_indices(
        si_lattice, 'fam_name', 'InjNLKckr')[0]
    si_lattice = pyaccel.lattice.shift(si_lattice, nlk_idx+1)

    # add scrapper error
    scraph = pyaccel.lattice.find_indices(si_lattice, 'fam_name', 'ScrapH')
    for i in scraph:
        si_lattice[i].hmax += params.scrapper_h_max_delta
        si_lattice[i].hmin += params.scrapper_h_min_delta

    accelerator = pyaccel.accelerator.Accelerator(
        lattice=si_lattice,
        energy=lat_mod.energy,
        harmonic_number=lat_mod.harmonic_number,
        cavity_on=params.si_cavity_on,
        radiation_on=params.si_radiation_on,
        vchamber_on=params.si_vchamber_on,
    )
    return accelerator


def bo_bunch_generate(nr_particles, bo_coupling):
    """."""
    bo_energy = 3e9  # [eV]

    # bunch equilibrium in BO
    print('--- calculating bo equilibrium parameters...')
    bo_model = pymodels.bo.create_accelerator(energy=bo_energy)
    bo_eparam = pyaccel.optics.EquilibriumParameters(bo_model)
    emit0 = bo_eparam.emit0
    emitx = (1.0 / (1.0 + bo_coupling)) * emit0
    emity = (bo_coupling / (1.0 + bo_coupling)) * emit0
    sigmae = bo_eparam.espread0
    sigmas = bo_eparam.bunch_length
    print('    emitax: {:.4f} nm.rad'.format(emitx/1e-9))
    print('    emitay: {:.4f} nm.rad'.format(emity/1e-9))
    print('    sigmae: {:.4f} %'.format(sigmae*100))
    print('    sigmas: {:.4f} mm'.format(sigmas*1e3))

    # optics as BO extraction point
    print('--- calculating bo twiss...')
    twiss, _ = pyaccel.optics.calc_twiss(bo_model)
    ejeseptf_idx = pyaccel.lattice.find_indices(
        bo_model, 'fam_name', 'EjeSeptF')[0]
    twiss0 = twiss[ejeseptf_idx]

    # generate bunch at BO extraction point
    print('--- generating bunch at bo extraction point...')
    bunch = pyaccel.tracking.generate_bunch(
        emitx, emity, sigmae, sigmas, twiss0, nr_particles-1, cutoff=100)
    zero = np.zeros((6, 1))
    bunch = np.hstack((zero, bunch))
    return bunch


def ts_bunch_transport(params, bunch):
    """."""
    pymodels.ts.accelerator.default_cavity_on = True
    pymodels.ts.accelerator.default_radiation_on = True
    pymodels.ts.accelerator.default_vchamber_on = True
    ts_model, _ = pymodels.ts.create_accelerator()

    bunch, lost_flag, lost_turn, lost_element, lost_plane = \
        pyaccel.tracking.ring_pass(ts_model, bunch)
    _ = lost_flag, lost_turn, lost_element, lost_plane

    # transform TS->SI geometry
    bunch[0, :] += params.transf_ts2si_drx
    bunch[1, :] += params.transf_ts2si_dpx

    return bunch


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
    nr_particles = rout.shape[1]
    for particle in range(nr_particles):
        rx_ = 1e3*rout[0, particle, :]
        px_ = 1e3*rout[1, particle, :]
        plt.plot(rx_, px_, '.')
    plt.xlabel('rx [mm]')
    plt.ylabel('px [mrad]')
    plt.title('Phase Space at InjSeptF ({} particles, {} turns)'.format(
        params.nr_particles, params.nr_turns))
    plt.show()


def plot_phase_space2(params, rout):
    """."""
    nr_turns = rout.shape[2]
    for turn in range(nr_turns):
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', color='grey')
    minnr = min(10, nr_turns)
    rx_ = 1e3*rout[0, :, minnr]
    px_ = 1e3*rout[1, :, minnr]
    plt.plot(rx_, px_, '.', color='grey', label='turn > {}'.format(minnr))
    for turn in range(minnr):
        rx_ = 1e3*rout[0, :, turn]
        px_ = 1e3*rout[1, :, turn]
        plt.plot(rx_, px_, '.', label='turn = {}'.format(turn))
    plt.xlabel('rx [mm]')
    plt.ylabel('px [mrad]')
    plt.grid()
    plt.legend()
    strf = 'Phase Space right after NLKick (nr_particles: {}, nr_turns)'
    plt.title(strf.format(params.nr_particles, params.nr_turns))
    plt.show()


def run1():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.3/1000 * 0  # [rad]
    params.scrapper_h_max_delta = -2.0/1000 * 0  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


def run2():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.3/1000 * 0  # [rad]
    params.scrapper_h_max_delta = -2.0/1000 * 1  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


def run3():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.3/1000 * 1  # [rad]
    params.scrapper_h_max_delta = -2.0/1000 * 1  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


def run4():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.3/1000 * 1  # [rad]
    params.scrapper_h_max_delta = -2.0/1000 * 0  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


def run5():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.2/1000 * 1  # [rad]
    params.scrapper_h_max_delta = -2.0/1000 * 0  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


def run6():
    """."""
    params = Params()
    params.nr_particles = 1000
    params.nr_turns = 300
    params.bo_coupling = 1*0.01
    params.si_cavity_on = True
    params.si_radiation_on = True
    params.si_vchamber_on = True
    params.transf_ts2si_drx = -17.35/1000  # [m]
    params.transf_ts2si_dpx = 2.4/1000  # [rad]
    params.nlk_kick_error = 0.3/1000 * 0  # [rad]
    params.scrapper_h_max_delta = -3.0/1000 * 1  # [m]
    params.scrapper_h_min_delta = +0.0/1000  # [m]
    si_bunch_injection(params)


run6()
