import numpy as np
from . import PSO, SimulAnneal, SimpleScan, GA


class Test1(PSO):

    def __init__(self, save=False, niter=0):
        PSO.__init__(self, save=save)
        self.niter = niter

    def initialization(self):
        self.set_limits(upper=np.array([10, 10]), lower=np.array([-10, -10]))

    def calc_obj_fun(self):
        f_out = np.zeros(self._nswarm)

        for i in range(self._nswarm):
            x, y = self.position[i, 0], self.position[i, 1]
            # f_out[i] = (x - 1) ** 2
            f_out[i] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test2(SimulAnneal):

    def __init__(self, save=False, niter=0):
        SimulAnneal.__init__(self, save=save)
        self.niter = niter

    def initialization(self):
        self.set_limits(upper=np.array([10, 10]), lower=np.array([-10, -10]))
        self.set_deltas(dmax=np.array([1, 1]))
        self.temperature = 0
        self.position = np.array([0, 0])

    def calc_obj_fun(self):
        x, y = self.position[0], self.position[1]
        # f_out[i] = (x - 1) ** 2
        f_out = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test3(SimpleScan):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = np.array([-10, -10])
        self._position = np.array([0, 0])

    def calc_merit_function(self):
        f_scan = np.zeros(len(self._delta))
        for j in range(len(self._delta)):
            v = self._position
            v[self._curr_dim] = self._delta[j]
            x = v[0]
            y = v[1]
            # f_scan[j] = (x - 1) ** 2
            f_scan[j] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_scan[j] = 0.26 * (x**2 + y**2) - 0.48 * x * y
            print('Position ' + str(v) + 'Fig. Merit' + str(f_scan[j]))
        f_out = np.min(f_scan)
        v[self._curr_dim] = self._delta[np.argmin(f_scan)]
        print('Best Position ' + str(v) + 'Fig. Merit' + str(f_out))
        return f_out, v[self._curr_dim]


class Test4(GA):

    def __init__(self, npop, nparents, mutrate):
        GA.__init__(self, npop=npop, nparents=nparents, mutrate=mutrate)

    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits

    def calc_merit_function(self):
        f_out = np.zeros(self._npop)

        for i in range(self._npop):
            x, y = self._indiv[i, 0], self._indiv[i, 1]
            # f_out[i] = (x - 1) ** 2
            f_out[i] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out
