from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute, find_indices, find_spos
import touschek_pack.functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.beam_optics import beam_rigidity as _beam_rigidity


class Tous_analysis():

    def __init__(self,accelerator):
        self._acc = accelerator
        self._beta = _beam_rigidity(energy=3)[2] # defining beta to tranformate the energy deviation
        self.h_pos = get_attribute(self._acc, 'hmax', indices='closed') # getting the vchamber's height
        self.h_neg = get_attribute(self._acc, 'hmin', indices='closed')
        self._ltime = Lifetime(self._acc)
        self._lname = ['BC', 'Q1', 'SDA0'] # names defined by default. it can be modified as the users desires
        
        self._sc_accps = None 
        self._accep = None         
        off_array = _np.linspace(0,0.046, 460) 
        self._ener_off = off_array # interval of energy deviation for calculating the amplitudes and idx_limitants from linear model
        self._nturns = None
        self._deltas = None

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
    def lname(self):
        return self._lname
    
    @lname.setter
    def lname(self, call_lname): # call_name is the list of element names (passed by the user) that someone desires to know the distribution
        self._lname = call_lname
        return self._lname

    @property
    def lamppn_idx(self):
        if self._lamppn_idx is None:
            model = pymodels.si.create_accelerator()
            model.cavity_on = False
            model.radiation_on = False
            self.lmd_amp_pos, self.idx_lim_pos = tousfunc.calc_amp(model, self.ener_off, self.h_pos, self.h_neg)
            self.lmd_amp_neg, self.idx_lim_neg = tousfunc.calc_amp(model, -self.ener_off, self.h_pos, self.h_neg)
        
        return self.lmd_amp_pos, self.idx_lim_pos, self.lmd_amp_neg, self.idx_lim_neg
    
    def return_single_tracked(self,s_position, par):
        model = pymodels.si.create_accelerator()
        model.cavity_on = True
        model.radiation_on = True
        
        # alterar depois a função que é utilizada nesta função para realizar o tracking.

        if 'pos' in par:
            res, ind = tousfunc.trackm_elec(model,self._deltas,self._nturns,s_position)
        elif 'neg' in par:
            res, ind = tousfunc.trackm_elec(model,-self._deltas,self._nturns,s_position)
        
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
        indp = indp[0]
        indn = indn[0]
        getdp[indp] == 0
        getdn[indn] == 0

        # this function will return the weighting factors in the scalc position convertion
        
        return getdp, getdn, deltp, deltn
    
    # In general, this function is usefull when we desire to calculate the touschek scattering weighting function for one specific point along the ring

    def fast_aquisition(self, s_position, par):
        # this raise blocks to runing the program if the list of s position has more than 1 element
        if len(tousfunc.t_list(s_position)) != 1:
            raise Exception('This function suports only one s position')
        
        lspos = tousfunc.t_list(s_position)
        res, ind = self.return_tracked(lspos, par)
        res = res[0]
        turn_lost, elem_lost, delta = _np.zeros(len(res)), _np.zeros(len(res)), _np.zeros(len(res))

        for idx, iten in enumerate(res):
            tlost, elmnlost, delt = iten
            turn_lost[idx] = tlost
            elem_lost[idx] = elmnlost
            delta[idx] = delt

        Ddeltas = _np.diff(delta)[0]
        fdensp, fdensn, deltp, deltn = self.get_weighting_tous(s_position)
        
        fp = fdensp.squeeze()
        fn = fdensn.squeeze()

        fp *= Ddeltas
        fn *= Ddeltas
        deltp *= 1e2
        deltn *= 1e2
        
        return fp, fn, deltp, deltn
    
    # vale mencionar que a fast aquisition da forma como esta definida já está funcional
    

#    a função complete aquisition poderia receber uma lista com os nomes dos elementos que se 
#    deseja estudar, mas poderia também passar uma lista de elementos quaisquer
#    como fazer para a função calcular uma hora um uma hora outro ?

# uma coisa é certa é melhor definir estas condições antes do programa realizar os calculos

# eu poderia fazer um função para vincular o nome do elemento as posições s ao longo do anel
# isso parece ser bem util caso alguem deseje estudar um elemento em um ponto ja especificado

    def comp_aq(self, lname_or_spos, par):
        
        param = tousfunc.char_check(lname_or_spos)

        if param is str:
            
            indices = tousfunc.el_idx_collector(self._acc, lname_or_spos)
            spos = find_spos(self._acc, indices='closed')
            res, ind = self.return_tracked(lspos, par)

            calc_dp, calc_dn, delta_p, delta_n = tousfunc.nnorm_cutacp(self._acc, lspos, 
                                                                    npt=5000, getsacp=getsacp)

            # chama a função tousfunc.el_idx_collector para selecionar os indices dos elementos em que se 
            # deseja realizar as análises 
            # indices esses que serão os pontos iniciais para a realização do tracking
            pass
        elif param is float:
            pass
            # se o usuário desejar obter o estudo ao longo de todo o anel ele simplesmente 
            # pode colocar como input todas as posições s que vem do modelo nominal

    # remember that ind is the index that represents the initial position where tracking begins
        getsacp = tousfunc.get_scaccep(self._acc, self._accep)

        
        

        



    #o resultado para este res será uma lista com diversas tuplas então agora eu tenho que me perguntar como
    #eu vou organizar isso 
    
    
    # e se eu fizesse a função dessa classe já pensando na possibilidade do calculo ser realizado para apenas um ponto do anel ou para varios ?
    # caso eu seja questionado sobre isso, posso justificar que para apenas um ponto do anel os cálculos são executados mais rapidamente.


        # As mensagens deixadas aqui são referentes a modificações que eu preciso realizar nesta classe com novas funcionalidades

        # Proximos passos
        #  preciso implementar o gráfico da distribuição junto com o gráfico do tracking
        # esses gráficos podem ser mostrados juntamente com o tracking o separados  
        #  preciso implementar o complete aquisition
        #  e ainda preciso pensar em como será o input dessa função
        # se ela vai usar o get_family_data ou find_indices (saber quando cada um será usado)
        # eu também deveria fazer a pesagem também por meio da simulação monte carlo que foi implementada há algum tempo
    

