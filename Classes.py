from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute, find_indices, find_spos, find_dict
import touschek_pack.functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.beam_optics import beam_rigidity as _beam_rigidity
import pandas as _pd
import scipy.integrate as scyint
import pyaccel as _pyaccel
import matplotlib.gridspec as gs

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
        self._num_part = 50000
        self._energy_dev_min = 1e-4

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
    
    @property
    def num_part(self):
        return self._num_part
    
    @num_part.setter
    def num_part(self, new_num_part):
        self._num_part = new_num_part

    @property
    def energy_dev_mcs(self):
        return self._energy_dev_min
    
    @energy_dev_mcs.setter
    def energy_dev_mcs(self, new_energy_dev):
        self._energy_dev_min = new_energy_dev


    
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
        
        self._model.cavity_on = True
        self._model.radiation_on = True
        self._model.vchamber_on = True
        spos = self._spos
        
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
        bf = self.beam_energy # bf is the beta factor
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

    # remember that ind is the index that represents the initial position where tracking begins

    # this function is used to compare the PDF of distinct s positions along the storage ring 
    def plot_normtousd(self, spos):
        
        spos_ring = self._spos
        model = self._model_fit
        accep = self._accep
        dic = tousfunc.norm_cutacp(self._model_fit, 
                             spos, 5000, accep, norm=True)
        
        fdensp = dic['fdensp']
        fdensn = dic['fdensp']
        deltasp = dic['deltasp']
        deltasn = dic['deltasn']

        fig, ax = _plt.subplots(figsize=(10,5))
        ax.set_title('Densidade de probabilidade para posições distintas da rede magnética')
        ax.grid(True, alpha=0.5, ls='--', color='k')
        ax.xaxis.grid(False)
        ax.set_xlabel(r'$\delta$ [%]', fontsize=14)
        ax.set_ylabel('PDF', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        # ap_ind = []

        for idx, s in enumerate(spos):
            
            array_fdens = fdensp[idx]
            index = _np.intp(_np.where(array_fdens <= 1e-2)[0][1])

            # apind.append(index)
            # this block selects the best index for plot the density distribution
            if not idx:
                best_index = index
            else:
                if best_index < index:
                    best_index = index
                else:
                    pass

        for idx, s in enumerate(spos):

            mod_ind = _np.argmin(_np.abs(spos_ring-s))

            fdenspi = fdensp[idx][:best_index]
            fdensni = fdensn[idx][:best_index]
            deltaspi = deltasp[idx][:best_index]*1e2
            deltasni = -deltasn[idx][:best_index]*1e2

            color = _plt.cm.gist_rainbow(idx/len(spos))
            not_desired = ['calc_mom_accep',
                           'mia', 'mib', 'mip',
                           'mb1', 'mb2', 'mc']
        
            while model[mod_ind].fam_name in not_desired:
                mod_ind += 1

            ax.plot(deltaspi, fdenspi,label='{} em {} m'.format(model[mod_ind].fam_name, _np.round(spos_ring[mod_ind], 2)),color=color)
            ax.plot(deltasni, fdensni, color=color )
            # ap_ind.append(mod_ind)

        ax.legend(loc='best', fontsize=12)

    #  this function plots the histograms returned by the monte carlo simulation
    def plot_histograms(self, l_spos):
        spos = self._spos
        accep = self.accep
        model = self._model_fit

        tup = tousfunc.histgms(self._model_fit, l_spos, self._num_part, accep,
                               self._energy_dev_min,cutaccep=False)
        
        hp, hn, idx_model = tup
        

        fig, ax = _plt.subplots(ncols=len(l_spos), nrows=1,figsize=(10,5), sharey=True)
        fig.suptitle('Densidade de probabilidade a partir da simulação Monte-Carlo')
        

        for index, iten in enumerate(idx_model):
            color = _plt.cm.jet(index/len(idx_model))
            ay = ax[index]
            if not index:
                ay.set_ylabel('PDF', fontsize=14)

            ay.grid(True, alpha=0.5, ls='--', color='k')
            ay.xaxis.grid(False)
            ay.set_xlabel(r'$\delta$ [%]', fontsize=14)
            ay.tick_params(axis='both', labelsize=12)
            ay.hist(hp[index], density=True, bins=200, color=color,
                    label='element:{}, pos:{:.2f} [m]'.format(model[iten].fam_name, spos[iten]))
            ay.hist(hn[index], density=True, bins=200, color=color)
            ay.legend()


    def get_track(self,l_scattered_pos):
        
        all_track = []
        indices = []
        spos = self._spos
        
        self._model.radiation_on = True
        self._model.cavity_on = True
        self._model.vchamber_on = True

        for _, scattered_pos in enumerate(l_scattered_pos):

            index = _np.argmin(_np.abs(scattered_pos-spos))
            indices.append(index)
            res = tousfunc.track_eletrons(self._deltas, self._nturns, index, self._model)
            all_track.append(res)

        return all_track, indices
    
    def find_data(self, l_scattered_pos):

        all_track, indices = self.get_track(l_scattered_pos)
        spos = self._spos
        
        fact = 0.03
        ltime = Lifetime(self._model_fit)
        tous_rate = ltime.touschek_data['rate']
        
        prob = []
        lostp = []
        all_lostp = []

        for j, iten in enumerate(all_track):

            index = indices[j]
            scattered_pos = l_scattered_pos[j]
            
            lostinds, deltas = _np.zeros(len(iten)), _np.zeros(len(iten))
            for idx,iten in enumerate(iten):
                _, ellost, delta = iten
                lostinds[idx] = ellost
                deltas[idx] = delta # alguns elétrons possuem desvio de energia abaixo da aceitancia e acabam não sendo perdidos

            lostinds = _np.intp(lostinds)
            lost_positions = _np.round(spos[lostinds], 2)

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
                for i, interval in enumerate(itv_delta):
                    if i == 0:
                        if interval[0]<= delta <= interval[1]:
                            data.loc[lost_pos, '{:.2f} % < delta < {:.2f} %'.format(interval[0]*1e2, interval[1]*1e2)] += 1
                    else:
                        if interval[0]< delta <= interval[1]:
                            data.loc[lost_pos, '{:.2f} % < delta < {:.2f} %'.format(interval[0]*1e2, interval[1]*1e2)] += 1

            data = data / len(deltas)

            npt = int((spos[-1]-spos[0])/0.1)

            scalc = _np.linspace(spos[0], spos[-1], npt)
            rate_nom_lattice = _np.interp(spos, scalc, tous_rate)

            lost_pos_df = []
            part_prob = []
            for indx, iten in data.iterrows():
                t_prob = 0
                for idx, m in enumerate(iten):
                    t_prob += m
                    if idx == iten.count()-1:
                        part_prob.append(t_prob)
                        lost_pos_df.append(indx)

            lost_pos_df = _np.array(lost_pos_df)
            part_prob = _np.array(part_prob)

            prob.append(part_prob * rate_nom_lattice[index])
            lostp.append(lost_pos_df)

            if not j:
                all_lostp = lost_pos_df
            else:
                boolean_array = _np.isin(lost_pos_df, all_lostp)
                for ind, boolean_indc in enumerate(boolean_array):
                    if not boolean_indc:
                        all_lostp = _np.append(all_lostp, lost_pos_df[ind])

        return all_lostp, prob, lostp
            # dataframe = _pd.DataFrame(dic_res)
    
    def get_table(self,l_scattered_pos):
        
        dic_res = {}
        all_lostp, prob, lostp = self.find_data(l_scattered_pos)
        n_scat = _np.round(l_scattered_pos, 2)

        for idx, scattered_pos in enumerate(n_scat):

            scat_data = []
            bool_array = _np.isin(all_lostp, lostp[idx])
            
            for j, boolean in enumerate(bool_array):
                if boolean:
                    index = _np.intp(_np.where(lostp[idx] == all_lostp[j])[0][0])
                    scat_data.append(prob[idx][index])
                else:
                    scat_data.append(0)

            if not idx:
                dic_res['lost_positions'] = all_lostp
                dic_res['{}'.format(scattered_pos)] = scat_data
            else:
                dic_res['{}'.format(scattered_pos)] = scat_data

            # df = _pd.DataFrame(dic_res)

        return dic_res
    
    def get_reordered_dict(self, l_scattered_pos, reording_key): # chatgpt code to reorder the dictionary

        dic = self.get_table(l_scattered_pos)

        zip_tuples = zip(*[dic[chave] for chave in dic])
        new_tuples = sorted(zip_tuples, key=lambda x: x[list(dic.keys()).index(reording_key)])
        zip_ordered = zip(*new_tuples)

        new_dict = {chave: list(valores) for chave, valores in zip(dic.keys(), zip_ordered)}

        return new_dict
    
    def get_lost_profile(self, dic):

        # dic = self.get_reordered_dict(l_scattered_pos, reording_key)
        spos = self._spos

        df = _pd.DataFrame(dic)
        a = df.set_index('lost_positions')

        scat_pos = _np.array(a.columns, dtype=float)
        
        indices = []
        for iten in scat_pos:
            ind =  _np.argmin(_np.abs(spos-iten))
            indices.append(ind)

        summed = []
        for idx, iten in a.iterrows():
            sum_row = scyint.trapz(a.loc[idx], spos[indices])
            summed.append(sum_row)

        fig, ax = _plt.subplots(figsize=(10,5),gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        ax.set_title('loss rate integral along the ring', fontsize=16)

        ax.set_xlabel('lost position [m]', fontsize=16)
        ax.set_ylabel('loss rate [1/s]', fontsize=16)

        ax.plot(list(a.index), summed, color='orange')
        _pyaccel.graphics.draw_lattice(self._model_fit,
                                       offset=-1e-6, height=1e-6, gca=True)

    
    def plot_scat_table(self, l_scattered_pos, new_dic, n_r,n_c=1):
        spos = self._spos
        # new_dic = self.get_reordered_dict(l_scattered_pos, 'lost_positions')

        if len(l_scattered_pos)%2 == 0:
            array = _np.arange(l_scattered_pos.size)
            j = _np.intp(_np.where(array == int((len(l_scattered_pos)/2 + 1)))[0])
            array_j = l_scattered_pos[j]
            index = _np.argmin(_np.abs(spos-array_j))

            lists = list(find_dict(self._model_fit, 'fam_name').values())

        else:
            array = _np.arange(l_scattered_pos.size)
            j = _np.where(array == int((len(l_scattered_pos)+1)/2))[0]
            array_j = l_scattered_pos[j]
            index = _np.argmin(_np.abs(spos-array_j))

            lists = list(find_dict(self._model_fit, 'fam_name').values())

        for i, l in enumerate(lists):
            if _np.isin(index, l).item():
                fam_name = list(find_dict(self._model_fit, 'fam_name').keys())[i]

        df = _pd.DataFrame(new_dic)
        df = df.set_index('lost_positions')

        fig, ax = _plt.subplots(n_c, n_r, figsize=(10, 6))
        ax.set_title('Loss profile')
        
        # legend = ax.text(0.5, -0.1, '{}'.format(fam_name), size=12, color='black',
        #           ha='center', va='center', transform=_plt.gca().transAxes)

        # # Ajustando a legenda (posicionamento)
        # legend.set_bbox({'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'black'})

        ax.set_xlabel('scattered positions [m]', fontsize=16)
        ax.set_ylabel('lost positions [m]', fontsize=16)

        heatmap = ax.pcolor(df, cmap='jet')  
        _plt.colorbar(heatmap)

        step1 = int(len(new_dic.keys())/5)
        arr1 = df.columns.values[::step1]

        _plt.xticks(_np.arange(df.shape[1])[::step1] + 0.5, arr1, fontsize=12)

        step2 = int(len(new_dic['lost_positions'])/5)
        arr2 = df.index.values[::step2]

        _plt.yticks(_np.arange(df.shape[0])[::step2] + 0.5, arr2, fontsize=12)

        fig.tight_layout()
        
        _plt.show()  


# # This function will probably will be in my repositories
# # I dont know if I will use it again, but it seems to me that it could be uselfull in some point

# def extract_delt(groups, deltas):
#     c = 0
#     big_list = []
#     for lists in groups:
#         lil_list = []
#         for _ in lists:
#             lil_list.append(c)
#             c+=1
            
#         big_list.append(lil_list)
    
#     sep_deltas = []
#     comp_l = []

#     for iten in big_list:
#         sep_deltas.append(deltas[iten])
#         comp_l.append(len(iten))


#     return sep_deltas, comp_l


