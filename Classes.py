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

        if beam_energy is None: # defining beta factor
            beta = _beam_rigidity(energy=3)[2]  

        self._model_fit = accelerator
        self._model = pymodels.si.create_accelerator()

        self._amp_and_limidx = None
        self._sc_accps = None
        self._accep = None
        self._inds_pos = None
        self._inds_neg = None
        self._amps_pos = None
        self._amps_neg = None
        self.num_part = 50000
        self.energy_dev_min = 1e-4

        self.beta = beta # beta is a constant necessary to the calculations
        self.h_pos = get_attribute(self._model_fit, 'hmax', indices='closed') 
        self.h_neg = get_attribute(self._model_fit, 'hmin', indices='closed')
        self.ltime = Lifetime(self._model_fit)
        # self._lname = ['BC', 'Q1', 'SDA0'] # it can be modified as the users desires
        self._off_energy = energy_off # en_dev to calculate amplitudes and idx_limitants from linear model
        self.nturns = n_turns 
        self._deltas = deltas  
        self.spos = find_spos(self._model_fit, indices='closed')
        self.scraph_inds = find_indices(self._model, 'fam_name', 'SHVC')
        self.scrapv_inds = find_indices(self._model, 'fam_name', 'SVVC')


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
                                                               self.off_energy, self.h_pos, self.h_neg)
            self._amps_neg, self._inds_neg = tousfunc.calc_amp(self.nom_model,
                                                               -self.off_energy, self.h_pos, self.h_neg)
            self._amp_and_limidx =  True

        return self._amp_and_limidx

    @property
    def off_energy(self):
        return self._off_energy
    
    # the cutoff for energy deviation is the energy acceptance limit
    @off_energy.setter
    def off_energy(self, accep): # pass a new energy acceptance tuple of arrays if the user desire
        accep_pos, accep_neg = accep
        accep_lim = _np.max(_np.maximum(accep_pos, _np.abs(accep_neg)))
        steps = int(accep_lim*10000) # choosen to be the number of steps
        self._off_energy = _np.linspace(0, accep_lim, steps)

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

    def get_amps_idxs(self): # this step calls and defines 3 disctinct getters
        return self.amp_and_limidx, self.accep, self.s_calc
    
    def set_vchamber_scraper(self, vchamber):
        model = self._model
        scph_inds = self.scraph_inds
        scpv_inds = self.scrapv_inds

        for iten in scph_inds:
            model[iten].hmin = vchamber[0]
            model[iten].hmax = vchamber[1]
        for iten in scpv_inds:
            model[iten].vmin = vchamber[2]
            model[iten].vmax = vchamber[3]

    def return_sinpos_track(self,single_spos, par):
        
        self._model.cavity_on = True
        self._model.radiation_on = True
        self._model.vchamber_on = True
        s = self.spos
        
        index = _np.argmin(_np.abs(s-single_spos))
        if 'pos' in par:
            res = tousfunc.track_eletrons(self.deltas,self.nturns,
                                               index, self.nom_model, pos_x=1e-5, pos_y=3e-6)
        elif 'neg' in par:
            res = tousfunc.track_eletrons(-self.deltas,self.nturns,
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
        bf = self.beta # bf is the beta factor
        lt = self.ltime
        b1, b2 = lt.touschek_data['touschek_coeffs']['b1'],lt.touschek_data['touschek_coeffs']['b2']
        
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

        self.deltas = deltp[-1]*1e2
        
        return getdp, getdn, deltp, deltn

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
        s = self.spos

        if issubclass(param, str): # if user pass a list of element names
            
            all_indices = tousfunc.el_idx_collector(self._model_fit, lname_or_spos)
            all_indices = _np.array(all_indices, dtype=object)
            ress = []
            scatsdis = []

            for indices in all_indices:
                
                res = self.return_compos_track(s[indices], par)
                scat_dis = tousfunc.nnorm_cutacp(self._model_fit, s[indices],
                                                 npt=5000, getsacp=getsacp)
                ress.append(res)
                scatsdis.append(scat_dis)

        # if user pass a list of positions (it can be all s posistions if the user desires)
        elif issubclass(param, float):
            ress = self.return_compos_track(lname_or_spos, par)
            scat_dis = tousfunc.nnorm_cutacp(self._model_fit, s[indices],
                                             npt=5000, getsacp=getsacp)
            
        return ress, scat_dis
    
    # this function plot the graphic of tracking and the touschek scattering distribution for one single position
    def plot_analysis_at_position(self, single_spos, par, accep,filename):

        res, fp, dp = self.fast_aquisition(single_spos, par)
        s = self.spos
        index = _np.argmin(_np.abs(s-single_spos))
        tousfunc.plot_track(self.accelerator, res, _np.intp(self.inds_pos),
                            self.off_energy, par, index, accep, dp, fp, filename)

    def plot_normtousd(self, spos,filename): # user must provide a list of s positions
        
        spos_ring = self.spos
        dic = tousfunc.norm_cutacp(self._model_fit, 
                             spos, 5000, self._accep, norm=True)
        
        fdensp, fdensn = dic['fdensp'], dic['fdensn'] 
        deltasp, deltasn = dic['deltasp'], dic['deltasn']

        fig, ax = _plt.subplots(figsize=(10,5))
        ax.set_title('Probability density analytically calculated', fontsize=20)
        ax.grid(True, alpha=0.5, ls='--', color='k')
        ax.xaxis.grid(False)
        ax.set_xlabel(r'$\delta$ [%]', fontsize=25)
        ax.set_ylabel('PDF', fontsize=25)
        ax.tick_params(axis='both', labelsize=28)

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
        
            while self._model_fit[mod_ind].fam_name in not_desired:
                mod_ind += 1

            ax.plot(deltaspi, fdenspi,
                    label='{} em {} m'.format(self._model_fit[mod_ind].fam_name, _np.round(spos_ring[mod_ind], 2)),color=color)
            ax.plot(deltasni, fdensni, color=color )
            # ap_ind.append(mod_ind)

        ax.legend(loc='best', fontsize=20)
        fig.savefig(filename, dpi=150)

    #  this function plots the histograms returned by the monte carlo simulation
    
    def plot_histograms(self, l_spos): # user must provide a list of s positions
        s = self.spos
        accep = self.accep
        model = self._model_fit

        tup = tousfunc.histgms(self._model_fit, l_spos, self.num_part, accep,
                               self.energy_dev_min,cutaccep=False)
        
        hp, hn, idx_model = tup
        
        fig, ax = _plt.subplots(ncols=len(l_spos), nrows=1,figsize=(30,10), sharey=True)
        fig.suptitle('Probability density calculated by Monte-Carlo simulation', fontsize=20)
        
        for index, iten in enumerate(idx_model):
            color = _plt.cm.jet(index/len(idx_model))
            ay = ax[index]
            if not index:
                ay.set_ylabel('PDF', fontsize=25)

            ay.grid(True, alpha=0.5, ls='--', color='k')
            ay.xaxis.grid(False)
            ay.set_xlabel(r'$\delta$ [%]', fontsize=25)
            ay.tick_params(axis='both', labelsize=18)
            ay.hist(hp[index], density=True, bins=200, color=color,
                    label='element:{}, pos:{:.2f} [m]'.format(model[iten].fam_name, s[iten]))
            ay.hist(hn[index], density=True, bins=200, color=color)
            _plt.tight_layout()
            ay.legend()


    def get_track(self,l_scattered_pos, scrap, vchamber):

        all_track = []
        indices = []
        spos = self.spos
        
        self._model.radiation_on = True
        self._model.cavity_on = True
        self._model.vchamber_on = True

        if scrap:
            self.set_vchamber_scraper(vchamber)

        for _, scattered_pos in enumerate(l_scattered_pos):

            index = _np.argmin(_np.abs(scattered_pos-spos))
            indices.append(index)
            res = tousfunc.track_eletrons(self._deltas, self.nturns, index, self._model)
            all_track.append(res)

        hx = self._model_fit[self.scraph_inds[0]].hmax
        hn = self._model_fit[self.scraph_inds[0]].hmin
        vx = self._model_fit[self.scrapv_inds[0]].vmax
        vn = self._model_fit[self.scrapv_inds[0]].vmin
        vchamber = [hx, hn, vx, vn]

        self.set_vchamber_scraper(vchamber)# reseting vchamber height and width (nominal)

        return all_track, indices
    
    def find_data(self, l_scattered_pos, scrap, vchamber):

        all_track, indices = self.get_track(l_scattered_pos, scrap, vchamber)
        spos = self.spos
        
        fact = 0.03 
        tous_rate = self.ltime.touschek_data['rate']
        
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
    
    def get_table(self,l_scattered_pos, scrap, vchamber):
        
        dic_res = {}
        all_lostp, prob, lostp = self.find_data(l_scattered_pos, scrap, vchamber)
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
    
    def get_reordered_dict(self, l_scattered_pos, reording_key, scrap, vchamber): # chatgpt code to reorder the dictionary

        dic = self.get_table(l_scattered_pos, scrap, vchamber)

        zip_tuples = zip(*[dic[chave] for chave in dic])
        new_tuples = sorted(zip_tuples, key=lambda x: x[list(dic.keys()).index(reording_key)])
        zip_ordered = zip(*new_tuples)

        new_dict = {chave: list(valores) for chave, valores in zip(dic.keys(), zip_ordered)}

        return new_dict
    
    def get_lost_profile(self, dic, filename):

        # dic = self.get_reordered_dict(l_scattered_pos, reording_key)
        spos = self.spos

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

        fig, ax = _plt.subplots(figsize=(13,7),gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
        ax.set_title('loss rate integral along the ring', fontsize=16)

        ax.set_xlabel('lost position [m]', fontsize=16)
        ax.set_ylabel('loss rate [1/s]', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)

        ax.plot(list(a.index), summed, color='navy')
        _pyaccel.graphics.draw_lattice(self._model_fit,
                                       offset=-1e-6, height=1e-6, gca=True)
        fig.savefig(filename,dpi=150)

    def get_lost_profilel(self, l_dic):

        # dic = self.get_reordered_dict(l_scattered_pos, reording_key)
        l = []
        for dic in l_dic:    
            s = self.spos

            df = _pd.DataFrame(dic)
            a = df.set_index('lost_positions')

            scat_pos = _np.array(a.columns, dtype=float)
            
            indices = []
            for iten in scat_pos:
                ind =  _np.argmin(_np.abs(s-iten))
                indices.append(ind)

            summed = []
            for idx, iten in a.iterrows():
                sum_row = scyint.trapz(a.loc[idx], s[indices])
                summed.append(sum_row)

            l.append((a.index, summed))
        
        return l

    
    def plot_scat_table(self, l_scattered_pos, new_dic, n_r,filename, n_c=1):
        s = self.spos

        # new_dic = self.get_reordered_dict(l_scattered_pos, 'lost_positions')

        # if len(l_scattered_pos)%2 == 0:
        #     array = _np.arange(l_scattered_pos.size)
        #     j = _np.intp(_np.where(array == int((len(l_scattered_pos)/2 + 1)))[0])
        #     array_j = l_scattered_pos[j]
        #     index = _np.argmin(_np.abs(spos-array_j))

        #     lists = list(find_dict(self._model_fit, 'fam_name').values())

        # else:
        #     array = _np.arange(l_scattered_pos.size)
        #     j = _np.where(array == int((len(l_scattered_pos)+1)/2))[0]
        #     array_j = l_scattered_pos[j]
        #     index = _np.argmin(_np.abs(spos-array_j))

        #     lists = list(find_dict(self._model_fit, 'fam_name').values())

        # for i, l in enumerate(lists):
        #     if _np.isin(index, l).item():
        #         fam_name = list(find_dict(self._model_fit, 'fam_name').keys())[i]

        df = _pd.DataFrame(new_dic)
        df = df.set_index('lost_positions')

        val = df.values.copy()
        idx = val != 0.0
        val[idx] = _np.log10(val[idx])
        val[~idx] = val[idx].min()

        fig, ax = _plt.subplots(figsize=(10,10))

        y = _np.linspace(0,s[-1],df.shape[0]+1)
        x = _np.linspace(0,s[-1],df.shape[1]+1)
        X,Y = _np.meshgrid(x,y)

        heatmp = ax.pcolor(X,Y,val, cmap='jet',shading='flat')

        cbar = _plt.colorbar(heatmp)
        cbar.set_label('Loss rate [1/s] in logarithmic scale', rotation=90)

        ax.set_title('Loss profile', fontsize=16)

        ax.set_xlabel('scattered positions [m]', fontsize=16)
        ax.set_ylabel('lost positions [m]', fontsize=16)

        fig.tight_layout()
        _plt.gca().set_aspect('equal')
        # fig.savefig(filename, dpi=150)
        _plt.show()

