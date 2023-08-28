from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute
import functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt


class Tous_analysis():

    # Defining this class I can call any accelerator I want to realize touschek scattering analysis
    # Keep in mind another terms that could be add in the init of the class

    # the acceptance is calculated utilizing the de accelerator's model so it'isnt necessary a priori 
    # pass the energy acceptance to the class

    # Define the interval to choose the energy deviation to realize the tracking simulation
    # Pass a parameter to define when I desire to use some function 
    # if tracking calculates the simulation utilizing the tracking function that was created 
    # if linear calculates the physical limitants that use a type of model for the calculation

    # I can take the indices a long the ring as parameters to use in the model.

    # eu tenho que pensar, porque se eu ficar fazendo dessa maneira todos os meus atributos serão none 
    # e depois eu vou precisar alterar este atributo manualmente será que é realmente isso que eu desejo ?

    # em primeiro lugar apenas para esclarecimento, eu posso definir um atributo da maneira que eu quiser e depois posso alterá-lo 
    # da forma que eu achar necessário

    # o que é mais vantajoso definir os atributos como None e depois alterar seus valores manualmente ou possuir uma configuração padrão em que
    # esses atributos já são definidos automaticamente?

    # Parece interessante a ideia de que os atributos sejam definidos em uma configuração padrão e sejam alterados conforme eu vá passando 
    # informações para o objeto da classe criada

    # em classes eu posso passar alguns parâmetros para que a classe saiba identificar qual é o modelo necessário que deve ser utilizado
    # ou então como definir os atributos com base nas palavras chave passada nos argumentos

    # vou começar a justificar alguns passos na criação desta classes para posteriormente eu me orientar 
    # e não criar uma classe sem saber interpretar os motivos de estar definindo de uma forma e não de outra
    
    '''Justificativa do por quê definir alguns atributos com valor padrão None'''

    # em primeiro lugar, a partir do que conversei com o fernando no dia 23.08.2023 é uma boa prática de progrmação criar classes que 
    # não gastem muito tempo ao serem iniciadas, ou seja, não é desejado que ao criar um objeto muito tempo seja empregado 
    # na construção do __init__.
    # Pensando nisso, existem atributos que são fundamentais para a classe que está sendo criada e, portanto, já devem ser 
    # definidos no __init__, mas como não desejamos que muito tempo seja empregado na operação de instaciação de um objeto, 
    # definimos alguns objetos com valor padrão None, para posteriormente redefinir essas grandezas provavelmente com a definição de outras 
    # funções que já estão inicialmente implementadas

    # Agora vejamos, eu preciso definir diversos atributos que não necessitam de um tempo de execução muito longo
    # qual a melhor forma de se fazer isso ?

    # eu poderia definir uma função que define mais atributos ao mesmo tempo?
    # isso pode ser util quando feito de maneira clara

    # é interessante que esta classe seja flexível a ponto de permitir que as simulações de tracking sejam realizadas 
    # por quem a esta utilizando 

    # todas as funções que foram definidas até o momento são utilizadas por meio de um loop que é executado em todos os elementos de 
    # uma lista de posições ao longo do feixe, essa lista pode até mesmo ser a lista completa de todas as posições do anel
    # essa lista pode ser de elementos específicos em que se deseje estudar a taxa de espalhamento touschek

    def __init__(self,accelerator):
        self._acc = accelerator
        self.h_pos = get_attribute(self._acc, 'hmax', indices='closed') # getting the vchamber's height
        self.h_neg = get_attribute(self._acc, 'hmin', indices='closed')
        self.ltime = Lifetime(self._acc)
        fam = pymodels.si.get_family_data(self._acc) # Geting the name of the element we desire to study
        self.lname = list(fam.keys()) # geting  
        self.index = tousfunc.el_idx_collector(self._acc, self.lname) # this calculation is relativily fast to execute, 
        
        self._sc_accps = None # check
        self._accep = None # check        
        off_array = _np.linspace(0,0.46, 460) 
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
    
    # Function below calculates the linear model amplitudes for graphic analysis

    # this function defines too the physical limitants calculate by our linear model
    # I have to verify if this function is a good programing practice, I dont know if this is the best aproach


    # This property is defined in this way to calculate the linear model maximum amplitudes
    # and to obtain the indices of the physical limitants

    @property
    def lamppn_idx(self):
        if self._lamppn_idx is None:
            model = pymodels.si.create_accelerator()
            model.cavity_on = False
            model.radiation_on = False
            self.lmd_amp_pos, self.idx_lim_pos = Tous_analysis.calc_amp(model, self.ener_off, self.h_pos, self.h_neg)
            self.lmd_amp_neg, self.idx_lim_neg = Tous_analysis.calc_amp(model, -self.ener_off, self.h_pos, self.h_neg)
        
        return self.lmd_amp_pos, self.idx_lim_pos, self.lmd_amp_neg, self.idx_lim_neg



    # agora é hora de definir outras propriedades que serão utilizadas nesta classe para realizar a contabilização 
    # dos devios de energia mais relevantes para o projeto

    
    # this function will calculate the weight factor to get the most important
    # energy deviation in a touschek scattering process

    # this function must take two functions of my previous code
    # all I need to do is take f_arg_mod and track_electrons 
    # Is important to note that track_electrons is based on the nominal model
    # in this way I must convert the index of the elements from spos calculated by pyaccel
    # and get the index that aproximates scalc to the spos positions

    #o que eu vou precisar para fazer esta função
    #com certeza eu vou precisar tomar a aceitancia da maquina para o modelo passado no 
    #inicio da classe e então selecionar o valor para realizar o corte de desvio de energia 
    # 
    
    def return_tracked(self,s_position, par):
        if 'pos' in par:
            res, ind = tousfunc.trackm_elec(self._acc,self._deltas,self._nturns,s_position)
        elif 'neg' in par:
            res, ind = tousfunc.trackm_elec(self._acc,-self._deltas,self._nturns,s_position)
        
        return res, ind
    
    #this function must pass the s_position for the calculation and the weighting the energy deviations
    # and this function will use the functions that I'm defining in this class to get 
    # the tracked energy deviations already weighted by the touschek scattering piwinski
    # distribution

    #for example:
    #    fast_aquisition(s_position, par) will use return_tracked

    def fast_aquisition(self):
        pass


    
    # tudo bem que desenvolver um código que obtenha diversos trackings para todos os pontos do anel é de fato uma coisa legal
    # mas para fazer isso é necessário empregar muito tempo de simulação
    # dessa forma, é possível definir funções que serão implementadas para realizar simulações rápidas 
    # e é possível definir funções que vão realizar simulações mais demoradas (provavelmente neste caso será realizada uma sondagem ao longo
    # de todo anel de armazenamento)

    # tenho de pensar em como realizar essas simulações rápidas, uma vez que eu já possuo as funções e como 
    # e como podemos saber em quais pontos específicos é bom realizar esta análise

