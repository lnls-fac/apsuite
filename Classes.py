from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute, find_indices, find_spos
import touschek_pack.functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.beam_optics import beam_rigidity as _beam_rigidity
import pandas as _pd


class Tous_analysis():

    def __init__(self, accelerator, energies_off=None, beam_energy=None, n_turns=7):
        if energies_off is None:
            energy_off = _np.linspace(0,0.046, 460) # used when calculating physical limitants by the linear model
            deltas = _np.linspace(0,0.1,400) # used in tracking simulation

        if beam_energy is None: # beam_energy is associated to the beta factor of the electrons of the storage ring 
            beta = _beam_rigidity(energy=3)[2]  # defining by default the energy of the beam

        self._model_fit = accelerator
        self._model = pymodels.si.create_accelerator() # No necessity to change this model, because it's an internal tool for calculations.

        self._amp_and_limidx = None
        self._sc_accps = None
        self._accep = None      
        self._inds_pos = None
        self._inds_neg = None
        self._amps_pos = None
        self._amps_neg = None

        self._beta = beta # beta is a constant necessary to the calculations
        self._h_pos = get_attribute(self._model_fit, 'hmax', indices='closed') # getting the vchamber's height
        self._h_neg = get_attribute(self._model_fit, 'hmin', indices='closed')
        self._ltime = Lifetime(self._model_fit)
        # self._lname = ['BC', 'Q1', 'SDA0'] # names defined by default. it can be modified as the users desires
        
        self._off_energy = energy_off # interval of energy deviation for calculating the amplitudes and idx_limitants from linear model
        self._nturns = n_turns # defined like this by the standard
        self._deltas = deltas # defined by the standard 
        self._spos = find_spos(self._model_fit, indices='closed')


    # Defining the energy acceptance. This method also needs a setter to change the value of the acceptance by a different inserted model
    @property
    def accelerator(self):
        return self._model_fit
    
    @accelerator.setter
    def accelerator(self, new_model): #This line defines a new accelerator if the user desires
        self._model_fit = new_model

    # defining the nominal accelerator
    @property
    def nom_model(self):
        return self._model

    # defining the heights of the vchamber, this properties vary the lenght of the array hpos and hneg
    @property
    def hpos(self):
        return self._h_pos
    
    @hpos.setter
    def hpos(self, indices):
        self._h_pos = get_attribute(self.accelerator, 'hmax', indices=indices)

    @property
    def hneg(self):
        return self._h_neg
    
    @hneg.setter
    def hneg(self, indices):
        self._h_neg = get_attribute(self.accelerator, 'hmin', indices=indices)

    @property
    def accep(self):
        if self._accep is None:
            self._accep = py_op.calc_touschek_energy_acceptance(self.accelerator)
        return self._accep

    # Defining the s position, positive and negative accptances along the ring at each 10 meters.
    # the description above is actually the returns of the function
    @property
    def s_calc(self):
        if self._sc_accps is None:
            self._sc_accps = tousfunc.get_scaccep(self.accelerator, self.accep)
        return self._sc_accps
    
    # This property calculates the physical limitants by the prediction of the linear model
    @property
    def amp_and_limidx(self):
        if self._amp_and_limidx is None:
            self.nom_model.cavity_on = False # this step is necessary to define if the 
            self.nom_model.radiation_on = False
            self._amps_pos, self._inds_pos = tousfunc.calc_amp(self.nom_model,
                                                               self.off_energy, self.hpos, self.hneg)
            self._amps_neg, self._inds_neg = tousfunc.calc_amp(self.nom_model,
                                                               -self.off_energy, self.hpos, self.hneg)
            self._amp_and_limidx =  True

        return self._amp_and_limidx
    
    # Defining the energy array to get the maximum amplitudes and the limitant indices by the linear model

    @property
    def ltime(self):
        return self._ltime
    
    @property
    def beam_energy(self):
        return self._beta
    
    @beam_energy.setter
    def beam_energy(self, new_beam_energy): # the user could change the energy of the beam if it is necessary
        self._beta = _beam_rigidity(energy=new_beam_energy)[2]

    @property
    def pos_vchamber(self):
        return self.h_pos
    
    @pos_vchamber.setter
    def pos_vchamber(self, indices): # here indices must be a string, closed or open, the user could define if it is necessary
        self.h_pos = find_spos(self._model_fit, indices=indices)

    @property
    def neg_vchamber(self):
        return self.h_neg
    
    @neg_vchamber.setter
    def neg_vchamber(self, indices):
        self.h_neg = find_spos(self._model_fit, indices=indices)

    @property
    def off_energy(self):
        return self._off_energy
    
    # the cutoff for energy deviation is the energy acceptance limit
    @off_energy.setter
    def off_energy(self, accep): # pass a new energy acceptance tuple of arrays if the user desire
        accep_pos, accep_neg = accep
        accep_lim = _np.max(_np.maximum(accep_pos, _np.abs(accep_neg)))
        steps = int(accep_lim*10000) # choosed to be the number of steps
        self._off_energy = _np.linspace(0, accep_lim, steps)

    @property
    def nturns(self):
        return self._nturns
    
    @nturns.setter
    def nturns(self, new_turns):
        self._nturns = new_turns # changes in the nturns to simulate with tracking

    @property
    def deltas(self):
        return self._deltas
    
    @deltas.setter
    def deltas(self, dev_percent,steps=400):
        dev_percent /= 100
        self._deltas = _np.linspace(0,dev_percent, steps) # if the user desires to make a change in the quantity of energ. dev. in tracking simulation
    
    @property
    def lname(self):
        return self._lname
    
    @lname.setter
    def lname(self, call_lname): # call_name is the list of element names (passed by the user) that someone desires to know the distribution
        self._lname = call_lname
    
    @property
    def spos(self):
        return self._spos
    
    @spos.setter
    def spos(self, indices): # if the user desires to make a change in the indices in the s position array / the variable indices must be a string 'closed' or 'open'
        self._spos = find_spos(self._model_fit, indices=indices)

    # the four properties defining below are to be fixed
    # this indices are obtained from the linear approach for the physical limitants
    @property
    def inds_pos(self):
        return self._inds_pos
    
    @property
    def inds_neg(self):
        return self._inds_neg
    
    # This two propertier will help if the user wants to plot the amplitudes calculated by the linear model
    @property
    def amp_pos(self):
        return self._amps_pos
    
    @property
    def amp_neg(self):
        return self._amps_neg

    
    # define aceitancia de energia, define também o fator de conversão que é sempre necessário das posições s do modelo nominal para scalc, e além disso calcula
    # as amplitudes e os indices limitantes

    # vai ser melhor retornar todos os parametros que eu preciso de uma vez, 
    # esses parametros são utilizados em momentos diferentes mas todos eles são necessários para a realização das análises tanto rapida quanto a completa
    # mas inicialmente não tem como o usuário saber que ele precisa definir estes parâmetros para visualização e análise dos gráficos entao
    # eu ainda preciso pensar em como tornar esta biblioteca mais simples de ser utilizada por alguem que nao conhece a fundo o código
    # isso significa que ao longo do codigo eu vou chamar as funções como se esses parametros ja tivessem sido definidos, mas na verdade eles so são 
    # definidos pelo usuário quando a classe é instanciada e a função abaixo é chamada definido estes 3 parametros ao mesmo tempo
    
    # def get_scaccep(self):
    #     return self.accep
    
    # def get_scalc(self):
    #     return self.s_calc let this code here to remind me how I may call a function in a class

    def get_amps_idxs(self): # this step calls and defines 3 disctinct getters
        return self.amp_and_limidx, self.accep, self.s_calc

    def return_sinpos_track(self,single_spos, par):
        
        self.nom_model.cavity_on = True
        self.nom_model.radiation_on = True
        self.nom_model.vchamber_on = True
        spos = self.spos
        
        index = _np.argmin(_np.abs(spos-single_spos))
        if 'pos' in par:
            res = tousfunc.track_eletrons(self.deltas,self.nturns,
                                               index, self.nom_model, pos_x=1e-5, pos_y=3e-6)
        elif 'neg' in par:
            res = tousfunc.track_eletrons(-self.deltas,self._nturns,
                                               index, self.nom_model, pos_x=1e-5, pos_y=3e-6)
        
        return res
    

    def return_compos_track(self, lspos, par):
        self.nom_model.cavity_on = True
        self.nom_model.radiation_on = True
        self.nom_model.vchamber_on = True

        if 'pos' in par:
            res = tousfunc.trackm_elec(self.nom_model, self.deltas,
                                            self.nturns, lspos)
        elif 'neg' in par:
            res = tousfunc.trackm_elec(self.nom_model, -self.deltas,
                                            self.nturns, lspos)
        return res
        
    
    def get_weighting_tous(self, single_spos, npt=5000):
        
        scalc, daccp, daccn  = tousfunc.get_scaccep(self.accelerator, self.accep)
        bf = self.beam_energy
        ltime = self._ltime
        b1, b2 = ltime.touschek_data['touschek_coeffs']['b1'],ltime.touschek_data['touschek_coeffs']['b2']
        
        taup, taun = (bf* daccp)**2, (bf*daccn)**2
        idx = _np.argmin(_np.abs(scalc-single_spos))
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
        indp = _np.where(getdp<0)[0]
        indn = _np.where(getdn<0)[0]
        getdp[indp] = 0
        getdn[indn] = 0

        ind = _np.intp(_np.where(getdp>1e-2)[0])
        getdp = getdp[ind]
        deltp = deltp[ind]
        ind = _np.intp(_np.where(getdn>1e-2)[0])
        getdn = getdn[ind]
        deltn = deltn[ind]

        # defining the energy deviation limit until tracking will be performed

        self.deltas = deltp[-1]*1e2

        # this function will return the weighting factors in the scalc position convertion
        
        return getdp, getdn, deltp, deltn
    
    # In general, this function is usefull when we desire to calculate the touschek scattering weighting function for one specific point along the ring

    def fast_aquisition(self, single_spos, par):
        # this raise blocks to runing the program if the list of s position has more than 1 element
        if len(tousfunc.t_list(single_spos)) != 1:
            raise Exception('This function suports only one s position')

        fdensp, fdensn, deltp, deltn = self.get_weighting_tous(single_spos)
        
        fp = fdensp.squeeze()
        fn = fdensn.squeeze()
        
        res = self.return_sinpos_track(single_spos, par)
        delta = _np.zeros(len(res))

        for index, iten in enumerate(res):
            tlost, ellost, delt = iten
            delta[index] = delt

        Ddeltas = _np.diff(delta)[0]

        if 'pos' in par:
            return res, fp*Ddeltas, deltp *1e2
        elif 'neg' in par:
            return res, fn*Ddeltas, deltn*1e2
        else:
            return res, fp*Ddeltas, fn*Ddeltas, deltp*1e2, deltn*1e2
    
    # vale mencionar que a fast aquisition da forma como esta definida já está funcional
    

#    a função complete aquisition poderia receber uma lista com os nomes dos elementos que se 
#    deseja estudar, mas poderia também passar uma lista de elementos quaisquer
#    como fazer para a função calcular uma hora um uma hora outro ?

# uma coisa é certa é melhor definir estas condições antes do programa realizar os calculos

# eu poderia fazer um função para vincular o nome do elemento as posições s ao longo do anel
# isso parece ser bem util caso alguem deseje estudar um elemento em um ponto ja especificado

    def complete_aquisition(self, lname_or_spos, par):
        param = tousfunc.char_check(lname_or_spos)
        getsacp = tousfunc.get_scaccep(self._model_fit, self._accep)
        spos = self._spos

        if issubclass(param, str): # if user pass a list of element names
            
            all_indices = tousfunc.el_idx_collector(self._model_fit, lname_or_spos)
            all_indices = _np.array(all_indices, dtype=object)
            ress = []
            scatsdis = []

            for indices in all_indices:
                
                res = self.return_compos_track(spos[indices], par)
                scat_dis = tousfunc.nnorm_cutacp(self._model_fit, spos[indices],
                                                 npt=5000, getsacp=getsacp)
                ress.append(res)
                scatsdis.append(scat_dis)

        # if user pass a list of positions (it can be all s posistions if the user desires)
        elif issubclass(param, float):
            ress = self.return_compos_track(lname_or_spos, par)
            scat_dis = tousfunc.nnorm_cutacp(self._model_fit, spos[indices],
                                             npt=5000, getsacp=getsacp)
            
        return ress, scat_dis
    
    # this function plot the graphic of tracking and the touschek scattering distribution for one single position
    def plot_analysis_at_position(self, single_spos, par, accep):
        # defining some params top plot the tracking and the scattering distribution
        # In a first approach I dont have to be concerned in this decision structure because all variables necessary for the calculation will be defined
        # I will may let this part of the code if I know a best method or decide to make a change here

        # if self._amp_and_limidx is None:
        #     amps_and_idxs = self.amp_and_limidx
        # else:
        #     amps_and_idxs = self._amp_and_limidx

        res, fp, dp = self.fast_aquisition(single_spos, par)
        spos = self.spos
        index = _np.argmin(_np.abs(spos-single_spos))
        tousfunc.plot_track(self.accelerator, res, _np.intp(self.inds_pos),
                            self.off_energy, par, index, accep, dp, fp)
        

    #if the user desires to know all the scattering events along the ring, 
    #only its necessary to do is to pass the 

    # remember that ind is the index that represents the initial position where tracking begins
        


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

        # eu vou definir alguns parametros de uma forma que talvez não seja ideal 
        # o que eu vou fazer vai ser definir todos os parametros que eu preciso de uma vez que eu preciso por meio de uma função
    

def get_table(self,l_scattered_pos):
    
    fact = 0.03
    spos = self._spos

    ltime = Lifetime(self._model_fit)
    tous_rate = ltime.touschek_data['rate']
    
    self._model.radiation_on = True
    self._model.cavity_on = True
    self.vchamber_on = True

    for j, scattered_pos in enumerate(l_scattered_pos):

        index = _np.argmin(_np.abs(scattered_pos-spos))
        res = tousfunc.track_eletrons(self._deltas, self._nturns, index, self._model)

        lostinds, deltas = _np.zeros(len(res)), _np.zeros(len(res))
        for idx,iten in enumerate(res):
            _, ellost, delta = iten
            lostinds[idx] = ellost
            deltas[idx] = delta # alguns elétrons possuem desvio de energia abaixo da aceitancia e acabam não sendo perdidos

        lostinds = _np.intp(lostinds)
        lost_positions = spos[lostinds]
        lost_positions = _np.round(lost_positions, 2)

        step = int((deltas[0]+deltas[-1])/fact)
        itv_track = _np.linspace(deltas[0], deltas[-1], step) # method learned by fac repositories

        data = _pd.DataFrame({'lost_pos_by_tracking': lost_positions}) # create the dataframe that is obtained by tracking
        lost_pos_column = (data.groupby('lost_pos_by_tracking').groups).keys()
        data = _pd.DataFrame({'lost_pos_by_tracking':lost_pos_column}) # this step agroups the lost_positions

        # scat_lost_df = pd.DataFrame(f'{s}':) # dataframe will contain the scattered positions and the lost positions after scattering

        itv_delta = []
        for current, next_iten in zip(itv_track, itv_track[1:]):
            data['{:.2f} % < delta < {:.2f} %'.format(current*1e2, next_iten*1e2)] = _np.zeros(len(list(lost_pos_column))) # this step creates new columns in the dataframe and fill with zeros
            itv_delta.append((current, next_iten))
            # Next step must calculate each matrix element from the dataframe

        var = list(data.index) 
        # quando as duas variáveis são iguais isso acab resultando em um erro então estou colocando essa condição.
        if var == lost_pos_column:
            pass
        else:
            data = data.set_index('lost_pos_by_tracking')


        for idx, lost_pos in enumerate(lost_positions): # essas duas estruturas de repetição são responsáveis por calcular 
            # o percentual dos eletrons que possuem um determinado desvio de energia e se perdem em um intervalo de desvio de energia específico
            delta = deltas[idx]
            lps = []
            for j, interval in enumerate(itv_delta):
                if j == 0:
                    if interval[0]<= delta <= interval[1]:
                        data.loc[lost_pos, '{:.2f} % < delta < {:.2f} %'.format(interval[0]*1e2, interval[1]*1e2)] += 1
                else:
                    if interval[0]< delta <= interval[1]:
                        data.loc[lost_pos, '{:.2f} % < delta < {:.2f} %'.format(interval[0]*1e2, interval[1]*1e2)] += 1

        data = data / len(deltas)

        npt = int((spos[-1]-spos[0])/0.1)


        scalc = _np.linspace(spos[0], spos[-1], npt)
        rate_nom_lattice = _np.interp(spos, scalc, tous_rate)

        dic_res = {}
        lost_pos_df = []
        part_prob = []
        for index, iten in data.iterrows():
            t_prob = 0
            for idx, m in enumerate(iten):
                t_prob += m
                if idx == iten.count()-1:
                    part_prob.append(t_prob)
                    lost_pos_df.append(index)

        lost_pos_df = _np.array(lost_pos_df)
        part_prob = _np.array(part_prob)


        dic_res['lost_position'] = lost_pos_df
        dic_res['{}'.format(scattered_pos)] = part_prob * rate_nom_lattice[index]

        _pd.DataFrame(dic_res).set_index('lost_position').transpose()