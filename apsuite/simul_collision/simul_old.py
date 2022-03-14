#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
import pyaccel
from pymodels import si
plt.rcParams.update({'font.size': 16})


def create_model(plane, max_length=0.01):
    """."""
    mod = si.create_accelerator(ids_vchamber=False)
    mod.cavity_on = False
    mod.radiation_on = False
    mod.vchamber_on = False
    mod = pyaccel.lattice.refine_lattice(mod, max_length=max_length)
    famname = 'InjDpKckr' if plane == 'x' else 'PingV'
    ping = pyaccel.lattice.find_indices(mod, 'fam_name', famname)[0]
    mod = pyaccel.lattice.shift(mod, start=ping)
    mib = pyaccel.lattice.find_indices(mod, 'fam_name', 'mib')[7]
    spos = pyaccel.lattice.find_spos(mod, indices='open')
    chamb_len = 2.6
    mod = pyaccel.lattice.insert_marker_at_position(
        mod, 'end_delta22', spos[mib] + chamb_len / 2)
    mod = pyaccel.lattice.insert_marker_at_position(
        mod, 'end_delta22', spos[mib] - chamb_len / 2)
    spos = pyaccel.lattice.find_spos(mod, indices='open')
    dlt22 = pyaccel.lattice.find_indices(mod, 'fam_name', 'end_delta22')
    # 6.5mm is the length of skew squared vacuum chamber
    side = 6.5e-3 / np.sqrt(2)
    idcs = np.arange(dlt22[0], dlt22[1])
    for idx in idcs:
        elem = mod[idx]
        elem.hmin, elem.hmax, elem.vmin, elem.vmax = [-side, side, -side, side]

    data = dict()
    data['hmax'] = pyaccel.lattice.get_attribute(mod, 'hmax')
    data['hmin'] = pyaccel.lattice.get_attribute(mod, 'hmin')
    data['vmax'] = pyaccel.lattice.get_attribute(mod, 'vmax')
    data['vmin'] = pyaccel.lattice.get_attribute(mod, 'vmin')
    data['spos'] = pyaccel.lattice.find_spos(mod)
    data['plane'] = plane
    data['model'] = mod
    data['dlt22'] = dlt22
    return data


def create_bunch(data, n_part):
    """."""
    mod = data['model']
    twiss, *_ = pyaccel.optics.calc_twiss(mod)
    emit0 = 0.25e-9  # [nm.rad]
    coup = 2 / 100
    emitx = emit0 * 1 / (1 + coup)
    emity = emit0 * coup / (1 + coup)
    sigmae = 0.084 / 100
    sigmas = 2.5 / 1000  # [m]
    bunch = np.array([])
    if n_part > 1:
        bunch = pyaccel.tracking.generate_bunch(
            n_part=n_part - 1,
            emit1=emitx, emit2=emity, sigmae=sigmae, sigmas=sigmas,
            optics=twiss[0],
            cutoff=10)
    origin = np.zeros((6, 1))
    data['bunch'] = np.hstack((origin, bunch))
    return data


def track(data, kick, nrturns):
    """."""
    rin = data['bunch']
    kicknorm = 1 / (1 + rin[4, :])
    if data['plane'] == 'x':
        rin[1, :] = kick * kicknorm
    else:
        rin[3, :] = kick * kicknorm
    mod = data['model']
    rout = np.zeros((nrturns, 6, rin.shape[1], len(mod)))
    for n in range(nrturns):
        rout_, lostflag, lostelem, lostplane = \
            pyaccel.tracking.line_pass(
                mod, rin, indices='closed', parallel=False)
        rin = rout_[:, :, -1]
        rout[n, :, :, :] = rout_[:, :, :-1]
    data['nrturns'] = nrturns
    data['kick'] = kick
    data['traj'] = rout
    return data


def check_collision(data, plane):
    """."""
    if plane == 'x':
        vcmax = data['hmax']
        vcmin = data['hmin']
        coord = 0
    else:
        vcmax = data['vmax']
        vcmin = data['vmin']
        coord = 2
    traj = data['traj'].copy()
    for n in range(traj.shape[0]):
        pn = traj[n, coord, :, :]
        # xn[np.isnan(xn)] = 0
        cond1 = pn <= vcmin[None, :]
        cond2 = pn >= vcmax[None, :]
        mask = cond1 | cond2
        # mask = np.isnan(xn)
        _, lostidx = np.where(mask)
        if any(lostidx):
            # lostidx = lostidx.reshape(mask.shape)
            idx = lostidx[0]
            vec = traj[n, :, :, idx]
            return [plane, n, idx, vec]
    return None


def kick_time_profile(nrpts=1000, tmax=None):
    """."""
    pulse_duration = 1.5e-6
    tmax = tmax or pulse_duration
    time = np.linspace(0, tmax, nrpts)
    kick = np.sin(np.pi/2*time/pulse_duration)
    return time, kick


def bunch_2_kick(bunch_idx, peak_kick=1.0):
    """."""
    time, kick = kick_time_profile(tmax=3e-6)
    bunch_spacing = 2e-9
    buncht = bunch_spacing * np.asarray(bunch_idx)
    # if isinstance(bunch_idx, int):
    #     buncht = bunch_spacing * bunch_idx
    # else:
    #     buncht = bunch_spacing * np.asarray(bunch_idx)
    return buncht, peak_kick * np.interp(buncht, time, kick)


def run_track(bunch_idx, plane='x', 
              n_part=300, peak_kick=-8e-3, max_length=0.01):
    """."""
    data = create_model(plane, max_length=max_length)
    data = create_bunch(data, n_part=n_part)

    # pulse_duration = 1.5e-6
    # omega = 2/np.pi/pulse_duration
    # bunch_spacing = 2e-9
    # bunch_idx = 400
    # kick = peak_kick * np.cos(np.pi/2*bunch_idx*bunch_spacing/pulse_duration)

    # bunch_idx = 0
    _, kick = bunch_2_kick(bunch_idx, peak_kick)

    data = track(data, kick, 1)
    spos, traj = data['spos'], data['traj']
    nturn = 0
    coord = 0
    coldata = check_collision(data, plane=plane)
    turn_lost = coldata[1]
    dlt_10cm = int(10*0.01/max_length)
    idx_stat = coldata[2] - dlt_10cm
    vec6d_lost = traj[turn_lost, :, :, idx_stat]

    sizex = np.std(vec6d_lost[0, :])
    sizey = np.std(vec6d_lost[2, :])
    divx = np.std(vec6d_lost[1, :])
    divy = np.std(vec6d_lost[3, :])
    centxl = np.mean(vec6d_lost[1, :])
    centyl = np.mean(vec6d_lost[3, :])
    # lost_angles.append(centxl)
    # lost_sizesx.append(sizex)

    # plt.figure()
    # lost_angles = np.array(lost_angles)
    # plt.plot(kicks*1e3, lost_angles*1e3, 'o-')
    # plt.xlabel('kicks dpkicker [mrad]')
    # plt.ylabel('angle @ lost point')
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(
        spos, 1e3 * traj[nturn, coord, :, :].T, '-', lw=1, alpha=0.1,
        color='tab:blue')
    ax.plot(spos, 1e3 * data['hmax'], color='k', ls='--')
    ax.plot(spos, 1e3 * data['hmin'], color='k', ls='--')
    ax.plot(
        spos[idx_stat], 1e3 * traj[turn_lost, coord, 0, idx_stat],
        'o', color='red')
    ax.set_ylim([-18, 18])
    ax.set_xlim([0, spos[idx_stat]*1.5])
    pyaccel.graphics.draw_lattice(
        data['model'], gca=ax, height=3, offset=0)
    stg = ''
    stg = f'Bunch Index: {bunch_idx:d}, '
    stg += f'Lost at s = {spos[idx_stat]:.4f} m, '
    dpos = spos[idx_stat+dlt_10cm] - spos[idx_stat]
    stg += rf'$\Delta$s = {dpos*100:.1f} cm'
    stg += '\n'
    stg += f"DipKicker = {data['kick']*1e3:.2f} mrad, "
    stg += r'$\sigma_x$ = '
    stg += f'{sizex*1e6:.1f} um, '
    stg += r'$\sigma_y$ = '
    stg += f'{sizey*1e6:.1f} um, '
    stg += '\n'
    stg += rf'AngleX: ({centxl*1e3:.3} $\pm$ {divx*1e3:.3}) mrad, '
    stg += rf'AngleY: ({centyl*1e6:.2} $\pm$ {divy*1e6:.2}) urad'

    ax.set_title(stg, fontsize=12)
    ax.set_xlabel('s [m]')
    ax.set_ylabel('x [mm]')
    plt.tight_layout()
    plt.savefig(f'beam_collision_kickx_bunch{bunch_idx:d}.png', dpi=300)
    plt.show()


def plot_kick(peak_kick=-8e-3):
    """."""
    time, kick = kick_time_profile(tmax=3e-6)
    bunches = np.arange(0, 864, 50)
    buncht, bunchkicks = bunch_2_kick(bunches, peak_kick=peak_kick)
    plt.hlines(y=0, xmin=-0.5, xmax=0.0, color='C0')
    plt.plot(1e6*time, 1e3*kick*peak_kick)
    plt.hlines(y=0.0, xmin=3.0, xmax=3.5, color='C0')
    plt.hlines(y=-9.0, xmin=0, xmax=864*2e-9*1e6, color='C1', label='1-turn', linestyles='dashed')
    plt.plot(1e6*buncht, 1e3*bunchkicks, 'o', label='bunches')

    plt.xlabel('time [us]')
    plt.ylabel('kick [mrad]')
    plt.title('Dipolar kick at 864 buckets\n(bunches separated by 50 buckets)')
    plt.xlim(-0.5, 3.5)
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('dipolar-kick.png')
    plt.show()


if __name__ == '__main__':
    """."""
    # plot_kick()
    run_track(bunch_idx=150, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=230, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=300, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=400, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=500, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=600, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=700, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=740, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=749, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
    run_track(bunch_idx=750, plane='x', n_part=300, peak_kick=-8e-3, max_length=0.05)
