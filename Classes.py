from pyaccel.lifetime import Lifetime


class tous_analysis():

    # Defining this class I can call any accelerator I want to realize touschek scattering analysis
    # Keep in mind another terms that could be add in the init of the class

    # the acceptance is calculated utilizing the de accelerator's model so it'isnt necessary a priori 
    # pass the energy acceptance to the class

    # Define the interval to choose the energy deviation to realize the tracking simulation
    # Pass a parameter to define when I desire to use some function 
    # if tracking calculates the simulation utilizing the tracking function that was created 
    # if linear calculates the physical limitants that use a type of model for the calculation

    def __init__(self,accelerator):
        self.acc = accelerator
        self.accep = None
        self.ltime = None
        self.nturns = None
        self.deltas = None
        self.param = None


    @property
    def setting_tousdata(self):
        if self.ltime is None:
            self.ltime = Lifetime(self.acc)
            return self.ltime
        
    # @property
    # def define_nturns(self):
    #     if 

        



