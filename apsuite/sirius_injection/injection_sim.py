"""Sirius injection simulation module"""
import numpy as np
import pyaccel as pa
import pymodels as pm
from plots import plot_phase_diagram, plot_phase_diagram2, plot_xtrajectories


class Injection:
    """Class which simulates the Sirius injection system"""

    def __init__(self, bo=None, ts=None, si=None):
        """."""
        self._bo = bo
        self._ts = ts
        self._si = si
        self._nparticles = 1000         # nr_particles in simulation
        self._bunch = np.zeros([6, self._nparticles])
        self._bo_coupling = 0.10        # booster transverse coupling (<1.0)
        self._bo_kickex_idx = 0
        self._bo_kickex_kick = 2.461e-3  # booster extraction kick [rad]
        self._bo_twiss = np.nan
        self._bo_eqparams = np.nan
        self._nturns = 213              # number of turns to track
        # For instance, I prefer local plot and verbose flags
        self._ts_chamber_rx_at_bo = 22e-3   # [m]   (rx of center of TS vacuum
        # - chamber w.r.t. TS coord. system)
        self._ts_chamber_px_at_bo = 5.0e-3  # [rad] (px of center of TS vacuum
        # - chamber w.r.t. TS coord. system)
        self._ts_init_twiss = np.zeros([2098])
        self._si_chamber_rx_at_ts = 0.01935  # [m]   (as measured at TS
        # - coordinates)
        self._si_chamber_px_at_ts = -2.84e-3  # [rad] (as measured at SI
        # - coordinates)
        self._si_nlk_strength = 0.727
        self._si_nlk_pulse = [1.0, 0.00, 0.00]  # temporal profile of nlk pulse
        self._si_nlk_physaccp = [0.0095, np.infty]  # horizontal and vertical
        # of VChamber at booster extraction
        self._injsep_idx = 0
        self._nlk_idx = 0

    def create_injection_lattice(self):
        """Creates the all the lattices involved in the injection process"""

        # Importing booster
        self._bo = pm.bo.create_accelerator(energy=3e9)  # Booster at ramp end
        self._bo.radiation_on = True
        self._bo.cavity_on = True
        self._bo.vchamber_on = True

        # Importing Transport Line
        self._ts, self._ts_init_twiss = pm.ts.create_accelerator(
            optics_mode="M2")  # Maybe is better change to "M2"
        # Remember that only the ts first index is a model
        self._ts.radiation_on = True
        self._ts.vchamber_on = True

        # Importing Sirius
        self._si = pm.si.create_accelerator()
        self._si.radiation_on = True
        self._si.cavity_on = True
        self._si.vchamber_on = True

        # Shifting the Booster start to the Ejection Kicker
        self._bo_kickex_idx = pa.lattice.find_indices(
            lattice=self._bo, attribute_name='fam_name', value='EjeKckr')
        self._bo = pa.lattice.shift(lattice=self._bo,
                                    start=self._bo_kickex_idx[0])
        self._bo_kickex_idx = pa.lattice.find_indices(
            lattice=self._bo, attribute_name='fam_name', value='EjeKckr')

        # Shifting the Sirius start to the septum index
        self._injsep_idx = pa.lattice.find_indices(
            lattice=self._si, attribute_name='fam_name', value='InjSeptF')
        self._si = pa.lattice.shift(lattice=self._si,
                                    start=self._injsep_idx[0])
        self._injsep_idx = pa.lattice.find_indices(
            lattice=self._si, attribute_name='fam_name', value='InjSeptF')

        # Seting the VChamber horizontal acceptance equal to the nlk
        self._nlk_idx = pa.lattice.find_indices(
            lattice=self._si, attribute_name='fam_name', value='InjNLKckr')
        self._si[self._nlk_idx[0]].hmin = -self._si_nlk_physaccp[0]
        self._si[self._nlk_idx[0]].hmax = +self._si_nlk_physaccp[0]

    def create_initial_bunch(self, plot=True, verbose=True):
        """."""
        self._bo_twiss, *_ = pa.optics.calc_edwards_teng(self._bo)
        self._bo_eqparams = pa.optics.EqParamsFromBeamEnvelope(self._bo)
        e1 = self._bo_eqparams.emit1
        e2 = abs(self._bo_eqparams.emit2)  # Avoiding numeric fluctuations
        energy_spread = self._bo_eqparams.espread0
        bunlen = self._bo_eqparams.bunlen

        bunch = pa.tracking.generate_bunch(
            emit1=e1, emit2=e2, sigmae=energy_spread,
            sigmas=bunlen, optics=self._bo_twiss[self._injsep_idx[0]],
            n_part=self._nparticles, cutoff=3)
        # orbit on ejection point:
        closed_orbit = pa.tracking.find_orbit6(accelerator=self._bo)
        self._bunch = bunch + closed_orbit

        if verbose:
            print('Bunch created with ', np.shape(self._bunch)[1],
                  ' particles \n')
        if plot:
            plot_phase_diagram2(
                bunch=self._bunch, local_twiss=self._bo_twiss[0],
                eqparams=self._bo_eqparams,
                title='Initial bunch at extraction kicker',
                closed_orbit=closed_orbit, emmit_error=True
                )

    def eject_from_booster(self, plot=True, verbose=True):
        """."""
        # Setting kick angle
        kick_length = self._bo[self._bo_kickex_idx].length
        self._bo[self._bo_kickex_idx[0]].polynom_b[0] = \
            -self._bo_kickex_kick/kick_length
        if verbose:
            print('- setting bo extraction kick to {:.3f} mrad \n'.format(
                self._bo_kickex_kick*1e3))

        # Transports bunch from extraction kicker to extraction thin septum
        self._bo_ejesepta_idx = pa.lattice.find_indices(
            self._bo, attribute_name='fam_name', value='EjeSeptF')

        self.part_out, lost_flag, _, _ = pa.tracking.line_pass(
            self._bo, particles=self._bunch,
            indices=np.arange(self._bo_kickex_idx[0],
                              self._bo_ejesepta_idx[0]+1))
        # Bunch after extraction_kicker
        bunch = self.part_out[:, :, self._bo_kickex_idx[0]+1]

        # Plots bunch after extraction kicker
        if plot:
            plot_phase_diagram(bunch, title="bunch after extraction kicker")

        if lost_flag and verbose:
            self.lost_particles = np.sum(np.isnan(bunch[0, :]))
            print('Particles lost at kicker to septum trajectory:',
                  self.lost_particles)
        if plot:
            plot_xtrajectories(
                self._bo, self.part_out, self._bo_kickex_idx,
                self._bo_ejesepta_idx,
                title="Horizontal trajectory from BO extraction"
                "kicker to extraction septum'"
                )

        # gets bunch at entrance of thin extraction septum
        self._bunch = self.part_out[:, :, self._bo_ejesepta_idx[0]]
        if plot:
            plot_phase_diagram(
                self._bunch,
                title='bunch at entrance of extraction septa (BO coordinates)'
                )

    def transport_along_ts(self, plot=True, verbose=True):
        """."""
        # translation of bunch coordinates from BO to TS
        ts_chamber_at_bo = np.array(
            [[self._ts_chamber_rx_at_bo], [self._ts_chamber_px_at_bo],
             [0], [0], [0], [0]]
            )
        self._bunch = self._bunch - ts_chamber_at_bo

        if plot:
            plot_phase_diagram(self._bunch, title='bunch at beginning of TS')

        # adds error in thin and thick BO extraction septa
        # (INCOMPLETE!!! right now they have ideal pulses)
        # Maybe I could use lattice.add_erro

        # transports bunch through TS
        # self._ts_twiss, *_ = pa.optics.calc_edwards_teng(self._ts,
        #                                                 self._ts_init_twiss)

        self.part_out, lost_flag, *_ = pa.tracking.line_pass(
            self._ts, self._bunch, indices='open'
            )

        if plot:
            plot_xtrajectories(self._ts, self.part_out, 0,
                               len(self._ts)-1,
                               title="bunch transported along TS")

        # bunch at the end of TS
        self._bunch = self.part_out[:, :, -1]
        if plot:
            plot_phase_diagram(self._bunch, title='bunch at end of TS')

        # lost particles and trajectories
        if lost_flag and verbose:
            self.lost_particles = \
                np.sum(np.isnan(self._bunch[0, :])) - self.lost_particles
            print('Particles losts at TS =', self.lost_particles)

    def inject_into_si_and_transports_to_nlk(self, plot=True, verbose=True):

        # Translation of bunch coordinates from TS to SI
        co = pa.tracking.find_orbit6(self._si)
        beam_long_center_si_inj = np.nanmean(self._bunch[5, :])
        si_chamber_at_ts = np.array(
            [[self._si_chamber_rx_at_ts],
             [self._si_chamber_px_at_ts],
             [0], [0], [0],
             [beam_long_center_si_inj-co[5][0]]]
            )
        # Bunch at injection point
        self._bunch = self._bunch - si_chamber_at_ts
        centroid = 1e3*np.nanmean(self._bunch, axis=1)  # nm
        print("- beam centroid at si injpoint (rx,px)(ry,py):"
              "(%+.3f mm, %+.3f mrad) (%+.3f mm, %+.3f mrad)\n"
              % (centroid[0], centroid[1], centroid[2], centroid[3]))

        # plots bunch at SI injection point
        if plot:
            plot_phase_diagram(self._bunch, closed_orbit=co,
                               title='bunch at Sirius Injection')

        # transports bunch from injection point to nlk
        self._nlk_idx = pa.lattice.find_indices(
            self._si, 'fam_name', 'InjNLKckr')
        inj_2_nlk = self._si[0:self._nlk_idx[0]+1]
        self.part_out, lost_flag, *_ = pa.tracking.line_pass(
            inj_2_nlk, self._bunch, indices='open'
            )
        # bunch at entrance of nlk
        self._bunch = self.part_out[:, :, -1]

        if lost_flag and verbose:
            self.lost_particles = \
                 np.sum(np.isnan(self._bunch[0, :])) - self.lost_particles
            print('Particles losts at injection point to nlk transport =',
                  self.lost_particles)
            centroid = 1e3*np.nanmean(self._bunch, axis=1)  # nm
            print("- beam centroid at entrance of si nlk (rx,px):"
                  "(%+.3f mm, %+.3f mrad) (%+.3f mm, %+.3f mrad)\n"
                  % (centroid[0], centroid[1], centroid[2], centroid[3]))

        # shifts si so that it starts at nlk
        self._si = pa.lattice.shift(self._si, self._nlk_idx[0])
        # p.si{p.si_nlk_idx}.VChamber(1) = p.si_nlk_physaccp(1);
        # I dont implement the above line because I think that the actual
        # Sirius model has the vaacum chamber with true dimensions at all ring.

    def sets_nlk_and_kicks_beam(self):
        pass

    def vary_si_nlk_strength(self):
        pass

    def vary_skewquad(self):
        pass
