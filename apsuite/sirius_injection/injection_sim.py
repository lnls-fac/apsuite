import numpy as np
from numpy.core.numeric import NaN
import pyaccel as pa
import pymodels as pm
from plots import plot_phase_diagram, plot_phase_diagram2, plot_xtrajectories

class Injection:
    """Class which simulates the Sirius injection system"""

    def __init__(self, bo=NaN, ts=NaN, si=NaN):
        """."""
        self.bo = bo
        self.ts = ts
        self.si = si
        self._nparticles = 1000         # nr_particles in simulation
        self._bo_coupling = 0.10        # booster transverse coupling (<1.0)
        self._bo_kickex_kick = 2.461e-3 # booster extraction kick [rad]
        self._nturns = 213              # number of turns to track
        #For instance, I prefer local plot and verbose flags
        self._si_nlk_strength = 0.727
        self._si_nlk_pulse = [1.0, 0.00, 0.00] # temporal profile of nlk pulse
        self._si_nlk_physaccp = [0.0095, np.infty] #horizontal and vertical of VChamber at booster extraction
        self._ts_chamber_rx_at_bo = 22e-3   # [m]   (rx of center of TS vacuum chamber w.r.t. TS coord. system)
        self._ts_chamber_px_at_bo = 5.0e-3  # [rad] (px of center of TS vacuum chamber w.r.t. TS coord. system)
        self._si_chamber_rx_at_ts = 0.01935 # [m]   (as measured at TS coordinates)
        self._si_chamber_px_at_ts = -2.84e-3 # [rad] (as measured at SI coordinates)
    
     #Remains add all the properties and setters
    def create_injection_lattice(self):
        """Creates the all the lattices involved in the injection process"""

        #Importing booster
        self.bo = pm.bo.create_accelerator(energy=3e9) #Booster at ramp end
        self.bo.radiation_on = True
        self.bo.cavity_on = True
        self.bo.vchamber_on = True

        # Importing Transport Line
        self.ts = pm.ts.create_accelerator()
        #Remember that only the ts first index is a model
        self.ts[0].radiation_on = True
        self.ts[0].vchamber_on = True

        # Importing Sirius
        self.si = pm.si.create_accelerator()
        self.si.radiation_on = True
        self.si.cavity_on = True
        self.si.vchamber_on = True

        # Shifting the Booster start to the Ejection Kicker
        self.bo_kickex_idx = pa.lattice.find_indices(
            lattice=self.bo, attribute_name='fam_name', value='EjeKckr')
        self.bo = pa.lattice.shift(lattice=self.bo, start=self.bo_kickex_idx[0])
        self.bo_kickex_idx = pa.lattice.find_indices(
            lattice=self.bo, attribute_name='fam_name', value='EjeKckr')
        
        # Shifting the Sirius start to the septum index
        self.injsep_idx = pa.lattice.find_indices(
            lattice=self.si, attribute_name='fam_name', value='InjSeptF')
        self.si = pa.lattice.shift(lattice=self.si, start=self.injsep_idx[0])
        self.injsep_idx = pa.lattice.find_indices(
            lattice=self.si, attribute_name='fam_name', value='InjSeptF')

        # Seting the VChamber horizontal acceptance equal to the nlk
        self._nlk_idx = pa.lattice.find_indices(
            lattice=self.si, attribute_name='fam_name', value='InjNLKckr')
        self.si[self._nlk_idx[0]].hmin = -self._si_nlk_physaccp[0]
        self.si[self._nlk_idx[0]].hmax = +self._si_nlk_physaccp[0]


    def create_initial_bunch(self, plot=True, verbose = True, refine=False): 
        
        self.bo_twiss,*_ = pa.optics.calc_twiss(self.bo)
        self.bo_eqparams = pa.optics.EquilibriumParametersOhmiFormalism(self.bo)
        ex = self.bo_eqparams.emitx
        ey = abs(self.bo_eqparams.emity) #the abs is avoid negative numeric erros around 'true zero' vertical emmitance
        energy_spread = self.bo_eqparams.espread0
        bunlen = self.bo_eqparams.bunlen
        
        bunch = pa.tracking.generate_bunch(emitx=ex, emity=ey,
            sigmae = energy_spread,sigmas=bunlen,twi=self.bo_twiss[0],
            n_part=self._nparticles, cutoff=3)
        closed_orbit = pa.tracking.find_orbit6(accelerator = self.bo) #orbit on ejection point
        self.bunch = bunch + closed_orbit

        if verbose:
            print('Bunch created with ',np.shape(self.bunch)[1],' particles \n')
        if plot:
            plot_phase_diagram2(bunch=self.bunch,local_twiss=self.bo_twiss[0],eqparams=self.bo_eqparams,
                title = 'Initial bunch at extraction kicker', closed_orbit= closed_orbit, emmit_error= True)
        

    def eject_from_booster(self, plot=True, verbose=True):
        
        #Setting kick angle
        kick_length = self.bo[self.bo_kickex_idx].length
        self.bo[self.bo_kickex_idx[0]].polynom_b[0] = -self._bo_kickex_kick / kick_length
        if verbose:
            print('- setting bo extraction kick to {:.3f} mrad \n'.format(self._bo_kickex_kick*1e3))

        #Transports bunch from extraction kicker to extraction thin septum
        self.bo_ejesepta_idx = pa.lattice.find_indices(self.bo,
            attribute_name='fam_name', value='EjeSeptF')
        self.part_out, lost_flag, lost_element, lost_plane = pa.tracking.line_pass(self.bo,
            particles=self.bunch, indices=np.arange(self.bo_kickex_idx[0],self.bo_ejesepta_idx[0]+1))
        bunch = self.part_out[:,:,self.bo_kickex_idx[0]+1] #Bunch after extraction_kicker
        
        # Plots bunch after extraction kicker
        if plot:
            plot_phase_diagram(bunch,title="'bunch after extraction kicker'")

        if lost_flag and verbose:
            print('Particles lost at kicker to septum trajectory:', np.sum(np.isnan(bunch[0,:])) )
        if plot:
            plot_xtrajectories(self.bo, self.part_out, self.bo_kickex_idx, 
            self.bo_ejesepta_idx, title = "Horizontal trajectory from BO extraction kicker to extraction septum'")
    
        # gets bunch at entrance of thin extraction septum
        self.bunch = self.part_out[:,:,self.bo_ejesepta_idx[0]]
        if plot:
            plot_phase_diagram(self.bunch,title='bunch at entrance of extraction septa (BO coordinates)')
        

    def transport_along_ts(self,plot=True,verbose=True):
        # translation of bunch coordinates from BO to TS
        ts_chamber_at_bo = np.array([[self._ts_chamber_rx_at_bo], [self._ts_chamber_px_at_bo],[0],[0],[0],[0]])
        self.bunch = self.bunch - ts_chamber_at_bo
        if plot:
            plot_phase_diagram2(self.bunch,self.ts[1],self.bo_eqparams,title='bunch at beginning of TS',emmit_error=True)

        # adds error in thin and thick BO extraction septa
        # (INCOMPLETE!!! right now they have ideal pulses)
        # Maybe I could use lattice.add_erro

        #transports bunch through TS
        self.part_out, lost_flag,*_  = pa.tracking.line_pass(self.ts[0],self.bunch,
            indices = 'open')
        
        #lost particles and trajectories
        if lost_flag and verbose:
            print('Particles losts at TS =',np.sum(np.isnan(self.part_out[0,:,-1])))
            
        if plot:
            plot_xtrajectories(self.ts[0], self.part_out, 0, 
            len(self.ts[0])-1, title = "bunch transported along TS")
        
        #bunch at the end of TS
        self.bunch = self.part_out[:,:,-1]
        self.ts_twiss,*_ = pa.optics.calc_twiss(self.ts[0],self.ts[1])
        if plot:
            plot_phase_diagram2(self.bunch,self.ts_twiss[-1],self.bo_eqparams,title='bunch at end of TS',emmit_error=True)
    
    def inject_into_si_and_transports_to_nlk(self,plot=True,verbose=True):
        
        #transports bunch through TS
        self.part_out, lost_flag,*_  = pa.tracking.line_pass(self.ts[0],self.bunch,
            indices = 'open')
        
        #lost particles and trajectories
        if lost_flag and verbose:
            print('There was particles lost at ts')
            
        if plot:
            plot_xtrajectories(self.ts[0], self.part_out, 0, 
            len(self.ts[0])-1, title = "bunch transported along TS")
        
        #bunch at the end of TS
        self.bunch = self.part_out[:,:,-1]
        self.ts_twiss,*_ = pa.optics.calc_twiss(self.ts[0],self.ts[1])
        if plot:
            plot_phase_diagram2(self.bunch,self.ts_twiss[-1],self.bo_eqparams,title='bunch at end of TS',emmit_error=True)
    
    def inject_into_si_and_transports_to_nlk(self,plot=True,verbose=True):
        
        # Translation of bunch coordinates from TS to SI
        co = pa.tracking.find_orbit6(self.si)
        beam_long_center_si_inj = np.mean(self.bunch[5,:])
        si_chamber_at_ts = np.array([[self._si_chamber_rx_at_ts],[self._si_chamber_px_at_ts],
            [0],[0],[0],[beam_long_center_si_inj-co[5][0]]])
        self.bunch = self.bunch - si_chamber_at_ts #Bunch at injection point
        #remains adds a print centroid of the bunch here
        #
        
        # plots bunch at SI injection point
        if plot:
            plot_phase_diagram(self.bunch,closed_orbit=co,title='bunch at Si Injection')

        pass
        # transports bunch from injection point to nlk
        self._nlk_idx = pa.lattice.find_indices(
            self.si, 'fam_name', 'InjNLKckr')
        inj_2_nlk = self.si[0:self._nlk_idx[0]+1]
        trajectories = pa.tracking.line_pass(inj_2_nlk,self.bunch,)

        
    def sets_nlk_and_kicks_beam(self):
        pass
    
    def vary_si_nlk_strength(self):
        pass

    def vary_skewquad(self):
        pass