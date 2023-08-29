from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute
import touschek_pack.functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.beam_optics import beam_rigidity


class Tous_analysis():

    def __init__(self,accelerator):
        self._acc = accelerator
        self._beta = beam_rigidity(energy=3)[2] # defining beta to tranformate the energy deviation
        self.h_pos = get_attribute(self._acc, 'hmax', indices='closed') # getting the vchamber's height
        self.h_neg = get_attribute(self._acc, 'hmin', indices='closed')
        self._ltime = Lifetime(self._acc)
        fam = pymodels.si.get_family_data(self._acc) # Geting the name of the element we desire to study
        self.lname = list(fam.keys()) # geting  
        self.index = tousfunc.el_idx_collector(self._acc, self.lname) # this calculation is relativily fast to execute, 
        
        self._sc_accps = None # check
        self._accep = None # check        
        off_array = _np.linspace(0,0.046, 460) 
        self._ener_off = off_array # interval of energy deviation for calculating the amplitudes and idx_limitants from linear model
        self._nturns = None # check
        self._deltas = None  # check

        self._lamppn_idx = None # this parameter defines 4 analysis' elements calculated by the linear model 


    # Defining the energy acceptance. This method also needs a setter to change the value of the acceptance by a different inserted model
    @property
    def accep(self):
        if self._accep is None:
            self._accep = py_op.calc_touschek_energy_acceptance(self._acc)
        return self._accep

    # Defining the s position, positive and negative accptances along the ring at each 10 meters.
    # the description above is actually the returns of the function
    @property
    def scalc(self):
        if self._sc_accps is None:
            self._sc_accps = tousfunc.get_scaccep(self._acc, self._accep)
        return self._sc_accps

    # Defining the energy array to get the maximum amplitudes and the limitant indices by the linear model
    @property
    def ener_off(self):
        return self._ener_off
    
    @ener_off.setter
    def ener_off(self, new):
        self._ener_off = new
        return self._ener_off
    
    # Defining the number of turns to realize the tracking simulation
    @property
    def nturns(self):
        return self._nturns
    
    @nturns.setter
    def nturns(self, new_turns):
        self._nturns = new_turns
        return self._nturns

    @property
    def deltas(self):
        return self._deltas
    
    @deltas.setter
    def deltas(self, new_deltas):
        self._deltas = new_deltas
        return self._deltas

    @property
    def lamppn_idx(self):
        if self._lamppn_idx is None:
            model = pymodels.si.create_accelerator()
            model.cavity_on = False
            model.radiation_on = False
            self.lmd_amp_pos, self.idx_lim_pos = Tous_analysis.calc_amp(model, self.ener_off, self.h_pos, self.h_neg)
            self.lmd_amp_neg, self.idx_lim_neg = Tous_analysis.calc_amp(model, -self.ener_off, self.h_pos, self.h_neg)
        
        return self.lmd_amp_pos, self.idx_lim_pos, self.lmd_amp_neg, self.idx_lim_neg
    
    def return_tracked(self,s_position, par):
        lspos = _np.array(s_position, dtype=object)
        if 'pos' in par:
            res, ind = tousfunc.trackm_elec(self._acc,self._deltas,self._nturns,lspos)
        elif 'neg' in par:
            res, ind = tousfunc.trackm_elec(self._acc,-self._deltas,self._nturns,lspos)
        
        return res, ind
    
    def get_weighting_tous(self, s_position, npt=5000):
        
        scalc, daccp, daccn  = tousfunc.get_scaccep(self._acc, self._accep)
        bf = self._beta
        ltime = self._ltime
        b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']
        
        taup, taun = (bf* daccp)**2, (bf*daccn)**2
        idx = _np.argmin(_np.abs(scalc-s_position))
        taup_0, taun_0 = taup[idx], taun[idx]
        kappap_0 =  _np.arctan(_np.sqrt(taup_0))
        kappan_0 =  _np.arctan(_np.sqrt(taun_0))

        kappa_pos = _np.linspace(kappap_0, _np.pi/2, npt)
        kappa_neg = _np.linspace(kappan_0, _np.pi/2, npt)

        deltp = _np.tan(kappa_pos)/bf
        deltn = _np.tan(kappa_neg)/bf
        getdp = tousfunc.f_function_arg_mod(kappa_pos, kappap_0, b1[idx], b2[idx],norm=False)
        getdn = tousfunc.f_function_arg_mod(kappa_neg, kappan_0, b1[idx], b2[idx],norm=False)

        # eliminating negative values
        indp = _np.where(getdp<0)
        indn = _np.where(getdn<0)
        getdp[indp] == 0
        getdn[indn] == 0

        # this function will return the weighting factors in the scalc position convertion 
        
        return getdp, getdn, deltp, deltn

    def fast_aquisition(self, s_position, par):

        res, ind = Tous_analysis.return_tracked(s_position, par)
        
        turn_lost, elem_lost, delta = _np.zeros(len(res)), _np.zeros(len(res)), _np.zeros(len(res))
        for idx, iten in enumerate(res):
            turn_lost[idx] = iten[0]
            elem_lost[idx] = iten[1]
            delta[idx] = iten[2]

        fdensp, fdensn, deltp, deltn = Tous_analysis.get_weighting_tous(s_position)
        Ddeltas = _np.diff(delta)

        for i, iten in enumerate(delta):
            indp = _np.where(delta[i] < deltp < delta[i+1])
            fdensp[indp] = fdensp * Ddeltas[i]
            

            indn = _np.where(delta[i] < deltn < delta[i+1])
            fdensn[indn] = fdensn * Ddeltas[i]

        return res

    
    # tudo bem que desenvolver um código que obtenha diversos trackings para todos os pontos do anel é de fato uma coisa legal
    # mas para fazer isso é necessário empregar muito tempo de simulação
    # dessa forma, é possível definir funções que serão implementadas para realizar simulações rápidas 
    # e é possível definir funções que vão realizar simulações mais demoradas (provavelmente neste caso será realizada uma sondagem ao longo
    # de todo anel de armazenamento)

    # tenho de pensar em como realizar essas simulações rápidas, uma vez que eu já possuo as funções e como 
    # e como podemos saber em quais pontos específicos é bom realizar esta análise

