from pyaccel.lifetime import Lifetime
from pyaccel.lattice import get_attribute
import touschek_pack.functions as tousfunc
import pymodels
import pyaccel.optics as py_op
import numpy as _np
import matplotlib.pyplot as _plt


class tous_analysis():

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
        self.acc = accelerator
        self.h_pos = get_attribute(self.acc, 'hmax', indices='closed') # getting the vchamber's height
        self.h_neg = get_attribute(self.acc, 'hmin', indices='closed')
        self.ltime = Lifetime(self.acc)
        fam = pymodels.si.get_family_data(self.acc) # Geting the name of the element we desire to study
        self.lname = list(fam.keys()) # geting  
        self.index = tousfunc.el_idx_collector() # this calculation is relativily fast to execute, 
        off_array = _np.linspace(0,0.46, 460) 
        self.ener_off = off_array # interval of energy deviation for calculating the amplitudes and idx_limitants from linear model
        
        self.sc_accps = None # check
        self._accep = None # check
        self.nturns = None # check
        self.deltas = None  # check 

        self.linear_model_ampp = None
        self.idx_lim_pos = None
        self.linear_model_ampn = None
        self.idx_lim_neg = None
        


    # Defining the energy acceptance. This method also needs a setter to change the value of the acceptance by a different inserted model
    @property
    def accep(self):
        if self._accep is None:
            self._accep = py_op.calc_touschek_energy_acceptance(self.acc)
        return self._accep

    # Defining the s position, positive and negative accptances along the ring at each 10 meters.
    # the description above is actually the returns of the function
    @property
    def convert_scalc(self):
        if self.sc_accps is None:
            self.sc_accps = tousfunc.get_scaccep(self.acc, self.accep)
        return self.sc_accps

    # Defining the energy array to get the maximum amplitudes and the limitant indices by the linear model
    @property
    def lin_offs_set(self):
        return self.ener_off
    
    @lin_offs_set.setter
    def lin_offs_set(self, new):
        self.ener_off = new
        return self.ener_off
    
    # Defining the electron number of turns to  
    @property
    def nturns_set(self):
        return self.nturns
    
    @nturns_set.setter
    def nturns_set(self, new_turns):
        self.nturns = new_turns
        return self.nturns

    @property
    def deltas_set(self):
        return self.nturns
    
    @deltas_set.setter
    def deltas_set(self, new_deltas):
        self.nturns = new_deltas
        return self.nturns
    

    

    
    # tudo bem que desenvolver um código que obtenha diversos trackings para todos os pontos do anel é de fato uma coisa legal
    # mas para fazer isso é necessário empregar muito tempo de simulação
    # dessa forma, é possível definir funções que serão implementadas para realizar simulações rápidas 
    # e é possível definir funções que vão realizar simulações mais demoradas (provavelmente neste caso será realizada uma sondagem ao longo
    # de todo anel de armazenamento)

    # tenho de pensar em como realizar essas simulações rápidas, uma vez que eu já possuo 
    


    # @property
    # def define_nturns(self):
    #     if 

        



