#!/usr/bin/env python-sirius

from attr import attributes
import numpy as _np
import matplotlib.pyplot as _plt
import pyaccel as _pyaccel
from pymodels import si as _si

_plt.rcParams.update({'font.size': 16})


class CollisionSimul:
    """."""

    MAX_KICK = -8e-3  # [rad]

    def __init__(self, plane, refine_max_len=0.01):
        """Create model with pulsed dipolar kick as first element."""
        self._plane = plane
        self._pinger_ind = None
        self._refine_max_len = refine_max_len
        self._kick = None
        self._bunchnr = None

        # create model
        self._model = self._create_model(
            cavity_on=True, radiation_on=True,
            vchamber_on=True, ids_vchamber=True)

        # init nominal equilibrium parameters
        self._emit0 = 0.25e-9  # [nm.rad]
        self._coup = 2 / 100
        self._emitx = self._emit0 * 1 / (1 + self._coup)  # [nm.rad]
        self._emity = self._emit0 * self._coup / (1 + self._coup)  # [nm.rad]
        self._sigmae = 0.084 / 100
        self._sigmas = 2.5 / 1000  # [m]

        # init lattice attributes
        self._twiss = None
        self._spos = None
        self._hmax = None
        self._hmin = None
        self._vmax = None
        self._vmin = None
        self._spos = None

        # bunch and traj attributes
        self._bunch = None
        self._traj = None
        self._lostparticles = None
        self._collision_distro = None

        # add delta22 vchamber
        self._delta22_vchamber_side, \
        self._delta22_vchamber_len, \
        self._delta22_idx = self._add_delta22_vchamber()

        self._update_model()

    @property
    def plane(self):
        """."""
        return self._plane

    @property
    def spos(self):
        """."""
        return self._twiss.spos

    @property
    def hmax(self):
        """."""
        return self._hmax

    @property
    def hmin(self):
        """."""
        return self._hmin

    @property
    def vmax(self):
        """."""
        return self._vmax

    @property
    def vmin(self):
        """."""
        return self._vmin

    @property
    def bunch(self):
        """."""
        return self._bunch

    @property
    def bunchnr(self):
        """."""
        return self._bunchnr

    @property
    def traj(self):
        """."""
        return self._traj

    @property
    def lostparticles(self):
        """."""
        return self._lostparticles

    @property
    def nr_particles(self):
        """."""
        if self._bunch is not None:
            return self._bunch.shape[1]
        return None

    @property
    def pinger_ind(self):
        """."""
        return self._pinger_ind

    @property
    def model(self):
        """."""
        return self._model

    @property
    def kick(self):
        """."""
        return self._kick

    def create_bunch(self, nr_particles, fixed_point=True):
        """."""
        bunch = _np.array([])
        if nr_particles > 1:
            bunch = _pyaccel.tracking.generate_bunch(
                n_part=nr_particles - 1,
                emit1=self._emitx, emit2=self._emity,
                sigmae=self._sigmae, sigmas=self._sigmas,
                optics=self._twiss[0],
                cutoff=10)
        if fixed_point:
            fixed_point_at_pinger = _pyaccel.tracking.find_orbit(
                accelerator=self.model, indices=self.pinger_ind)
            origin = fixed_point_at_pinger[:, [0]]
        else:
            origin = _np.zeros((6, 1))

        # self._bunch = _np.hstack((origin, bunch))
        self._bunch = bunch + origin

    def run_tracking(self, bunchnr=None, nrturns=1):
        """."""
        if self.bunch is None:
            raise Exception('Uninitialized bunch!')

        if bunchnr is not None:
            bunch_spacing = 2e-9
            pulse_duration = 1.5e-6
            time = bunchnr * bunch_spacing
            kick = CollisionSimul.MAX_KICK * _np.cos((_np.pi/2)*time/pulse_duration)

        # set pinger kick
        self._bunchnr = bunchnr
        self._kick = kick
        pinger_ind = self._pinger_ind
        lens = _np.sum(_pyaccel.lattice.get_attribute(self._model, 'length', pinger_ind))
        for idx in pinger_ind:
            elem = self._model[idx]
            if self._plane == 'x':
                elem.hkick_polynom = (elem.length/lens) * kick
            else:
                elem.vkick_polynom = (elem.length/lens) * kick

        # do tracking nturns
        rin = self.bunch.copy()
        rout = _np.zeros((nrturns, 6, rin.shape[1], len(self._model)))
        for nidx in range(nrturns):
            # NOTE: check if parallel runs ok
            rout_, lostflag, lostelem, lostplane = \
                _pyaccel.tracking.line_pass(
                    self._model, rin, indices='closed', parallel=False)
            rin = rout_[:, :, -1]
            rout[nidx, :, :, :] = rout_[:, :, :-1]
            _ = lostflag, lostelem, lostplane
        self._traj = rout

    def check_collisions(self):
        """."""
        if self.traj is None:
            raise Exception('Pasrticles not tracked!!')
        nrturns, nrcoors, nrparticles, nrelems = self.traj.shape
        _ = nrcoors, nrelems
        lostparticles = dict()
        for nturn in range(nrturns):
            lostparticles[nturn] = dict()
            ptraj = self._traj[nturn]
            # NOTE: generalize to include verticla case
            ptrajx = _np.isnan(ptraj[0])
            for part in range(nrparticles):
                elemidx = _np.where(ptrajx[part])[0][0]
                prx = ptraj[0, part, elemidx-1]  # posx
                ppx = ptraj[1, part, elemidx-1]  # angx
                if ppx > 0:
                    # loss at upper vchamber
                    dpos = (self.hmax[elemidx] - prx)/ppx
                else:
                    # loss at lower vchamber
                    dpos = (self.hmin[elemidx] - prx)/ppx
                lpos = self.spos[elemidx-1] + dpos  # extrapolated spos at loss
                if elemidx not in lostparticles[nturn]:
                    lostparticles[nturn][elemidx] = [(part, lpos), ]
                else:
                    lostparticles[nturn][elemidx].append((part, lpos))
        self._lostparticles = lostparticles

        # for nturn in lostparticles:
        #     for elem in lostparticles[nturn]:
        #         part, lpos = zip(*lostparticles[nturn][elem])
        #         stg = ''
        #         stg += f'nturn:{nturn:02d} elem:{elem:04d} '
        #         stg += f'nrpart:{len(part):04d} '
        #         stg += f'lpos_avg:{_np.mean(lpos):} lpos_std:{_np.std(lpos)}'
        #         print(stg)

        self._group_adjacent_loss_points()

    def calc_collision_distro(self):
        """."""
        self._collision_distro = dict()
        for nturn in self._lostparticles:
            traj = self.traj[nturn]
            for elem in self._lostparticles[nturn]:
                data = self._lostparticles[nturn][elem]
                part, lpos = zip(*data)
                lpos_avg, lpos_std = _np.mean(lpos), _np.std(lpos)
                stg = ''
                # stg += f'nturn:{nturn:02d} elem:{elem:04d} '
                # stg += f'nrpart:{len(part):04d} '
                # stg += f'lpos_avg:{lpos_avg:.5f} lpos_std:{lpos_std:.5f}'
                # print(stg)
                datarx = traj[0, part, elem-1]
                datapx = traj[1, part, elem-1]
                datary = traj[2, part, elem-1]
                datapy = traj[3, part, elem-1]
                _, rx_std = _np.mean(datarx), _np.std(datarx)
                _, ry_std = _np.mean(datary), _np.std(datary)
                px_avg, px_std = _np.mean(datapx), _np.std(datapx)
                py_avg, py_std = _np.mean(datapy), _np.std(datapy)
                stg = ''
                stg += f'nturn  : {nturn:02d}\n'
                stg += f'elem   : {elem-1:04d}\n'
                stg += f'dist   : {100*(lpos_avg - self.spos[elem-1]):.2f} cm\n'
                stg += f'nrpart : {len(part):04d}/{traj.shape[1]:04d}\n'
                stg += f'lpos   : ({lpos_avg:.4f} +/- {lpos_std:.4f}) m\n'
                stg += f'sigmax : {1e6*rx_std:.1f} um\n'
                stg += f'anglex : ({1e3*px_avg:.2f} +/- {1e3*px_std:.2f}) mrad\n'
                stg += f'sigmay : {1e6*ry_std:.1f} um\n'
                stg += f'angley : ({1e6*py_avg:.2f} +/- {1e6*py_std:.2f}) urad\n'
                print(stg)
                # NOTE: this will overwrite previous elem data
                self._collision_distro[nturn] = dict(
                    elem=elem-1,
                    dist=(lpos_avg - self.spos[elem-1]),
                    nrpart=len(part),
                    lpos_avg=lpos_avg,
                    lpos_std=lpos_std,
                    sigmax=rx_std,
                    sigmay=ry_std,
                    anglex_avg=px_avg,
                    anglex_std=px_std,
                    angley_avg=py_avg,
                    angley_std=py_std,
                )

    def plot_bunch(self):
        """."""
        bunch = self._bunch
        _plt.plot(1e6*bunch[0, :], 1e6*bunch[1, :], 'o')
        _plt.show()

    def plot_traj(self, nturn):
        """."""
        if self._lostparticles is None:
            self.check_collisions()
        if self._collision_distro is None:
            self.calc_collision_distro()

        coord = 0
        spos = self.spos
        traj = self.traj[nturn]

        if nturn in self.lostparticles:
            lost_elemidx = max(self.lostparticles[nturn].keys())
        else:
            lost_elemidx = len(self._model)

        fig, axs = _plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(
            spos, 1e3 * traj[coord, :, :].T, '-', lw=1, alpha=0.1,
            color='tab:blue')
        axs.plot(spos, 1e3 * self.hmax, color='k', ls='--')
        axs.plot(spos, 1e3 * self.hmin, color='k', ls='--')
        _pyaccel.graphics.draw_lattice(
            self._model, gca=axs, height=3, offset=0)
        axs.set_ylim([-18, 18])
        axs.set_xlim([0, spos[lost_elemidx]*1.5])

        stat = self._collision_distro[nturn]
        stg = ''
        stg += f'Bunch Nr.: {self.bunchnr}, '
        stg += f'DipKick = {1e3*self.kick:.3f} mrad, '
        stg += rf'Lost {stat["nrpart"]}/{self.nr_particles} at s = {stat["lpos_avg"]:.4f} m $\pm$ {1e3*stat["lpos_std"]:.2f} mm'
        stg += '\n'
        stg += f'At {1e2*stat["dist"]:.2f} cm from vchamber: '    
        stg += r'$\sigma_x$ = '
        stg += f'{1e6*stat["sigmax"]:.1f} um, '
        stg += r'$\sigma_y$ = '
        stg += f'{1e6*stat["sigmay"]:.1f} um\n'
        stg += rf'angX = ({1e3*stat["anglex_avg"]:.3f} $\pm$ {1e3*stat["anglex_std"]:.3f}) mrad, '
        stg += rf'angY = ({1e6*stat["angley_avg"]:.2f} $\pm$ {1e6*stat["angley_std"]:.2f}) urad '
        axs.set_title(stg, fontsize=12)
        axs.set_xlabel('s [m]')
        axs.set_ylabel('x [mm]')
        _plt.tight_layout()
        _plt.savefig(f'beam_collision_kickx_bunch_{self.bunchnr:03d}.png', dpi=300)
        _plt.show()

    def _create_model(self,
                      cavity_on, radiation_on,
                      vchamber_on, ids_vchamber):
        """."""
        # create model
        model = _si.create_accelerator(ids_vchamber=ids_vchamber)
        model.cavity_on = cavity_on
        model.radiation_on = radiation_on
        model.vchamber_on = vchamber_on

        # refine lattice
        model = _pyaccel.lattice.refine_lattice(
            model, max_length=self._refine_max_len)

        # shift model according to selected plane
        famname = 'InjDpKckr' if self._plane == 'x' else 'PingV'
        pinger_ind = _pyaccel.lattice.find_indices(model, 'fam_name', famname)
        model = _pyaccel.lattice.shift(model, start=pinger_ind[0])
        self._pinger_ind = _pyaccel.lattice.find_indices(model, 'fam_name', famname)

        return model

    def _add_delta22_vchamber(self):
        self._update_model()
        # NOTE: add vcahmber delta22 for any straight?
        # add delta22 vchamber at 16SB
        chamb_side = 6.5e-3 / _np.sqrt(2)  # [m]
        chamb_len = 2.6
        chamb_sb_idx = 7

        sscenter = self._mib[chamb_sb_idx]
        spos = self.spos
        self._model = _pyaccel.lattice.insert_marker_at_position(
            self._model, 'end_delta22', spos[sscenter] + chamb_len / 2)
        self._model = _pyaccel.lattice.insert_marker_at_position(
            self._model, 'end_delta22', spos[sscenter] - chamb_len / 2)
        self._update_model()
        dlt22 = _pyaccel.lattice.find_indices(
            self._model, 'fam_name', 'end_delta22')
        idcs = _np.arange(dlt22[0], dlt22[1])
        for idx in idcs:
            elem = self._model[idx]
            elem.hmin, elem.hmax, elem.vmin, elem.vmax = \
                [-chamb_side, chamb_side, -chamb_side, chamb_side]

        return chamb_side, chamb_len, dlt22

    def _update_model(self):
        model = self._model
        self._hmax = _pyaccel.lattice.get_attribute(model, 'hmax')
        self._hmin = _pyaccel.lattice.get_attribute(model, 'hmin')
        self._vmax = _pyaccel.lattice.get_attribute(model, 'vmax')
        self._vmin = _pyaccel.lattice.get_attribute(model, 'vmin')
        self._mib = _pyaccel.lattice.find_indices(model, 'fam_name', 'mib')
        self._twiss, _ = _pyaccel.optics.calc_twiss(model)

    def _group_adjacent_loss_points(self):
        for nturn in self._lostparticles:
            lostp = self._lostparticles[nturn]
            elemind = _np.array(sorted(lostp.keys()))
            groups = []
            for elem in elemind:
                # print(elem, groups)
                found = False
                for gelem in groups:
                    if min(abs(_np.array(gelem) - elem)) == 1:
                        gelem.append(elem)
                        gelem = list(set(gelem))
                        found = True
                if not found:
                    groups.append([elem])
            # print(groups)
            data = dict()
            for gelem in groups:
                firstelem = min(gelem)
                data[firstelem] = []
                for ele in gelem:
                    data[firstelem].extend(self._lostparticles[nturn][ele])
            self._lostparticles[nturn] = data


def run():
    """."""
    csimul = CollisionSimul('x', refine_max_len=0.01)
    csimul.create_bunch(nr_particles=500, fixed_point=False)
    csimul.run_tracking(bunchnr=0, nrturns=1)
    csimul.plot_traj(nturn=0)


def plot_initial_train():
    """."""
    data = {
        '0': [1.6998, 8.03],
        '10': [1.7004, 8.61],
        '30': [1.7034, 8.62],
        '50': [1.7074, 8.35],
        '70': [1.7171, 8.52],
        '90': [1.7271, 8.68],
        '100': [1.7333, 8.92],
        '120': [1.7488, 8.28],
        '150': [1.7768, 8.25],
        '200': [1.8373, 7.88],
        '250': [1.9171, 7.89],
        '300': [2.0257, 9.48],
    }

    for label, gaussm in data.items():
        _plt.vlines(int(label), gaussm[0] - gaussm[1]/1000, gaussm[0] + gaussm[1]/1000)
        _plt.plot([int(label)], [gaussm[0]], 'o', color='C0')
    _plt.xlabel('Bunch Number')
    _plt.ylabel('Impact Position [m]')
    _plt.grid()
    stg = 'Impact Position of Kicked Bunches on Vacuum Chamber\n'
    stg += '(vertical bars represent 1-sigma beam distribution)' 
    _plt.title(stg)
    _plt.show()



    s = _np.linspace(1.6998-10/1000, 2.0257+10/1000, 100)
    for label, gaussm in data.items():
        avg, std = gaussm
        std /= 1000
        gauss = _np.exp(-0.5*(s-avg)**2/std)
        _plt.plot(s, gauss, label='bunch ' + label)
    _plt.xlabel('Impact position [m]')
    _plt.ylabel('Particle density [a.u.]')
    _plt.legend()
    _plt.title('Impact Position of Kicked Bunches on Vacuum Chamber')
    _plt.show()


if __name__ == "__main__":
    run()
    # plot_initial_train()
