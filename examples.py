import numpy as np
from optimization import PSO, SimulAnneal, SimpleScan


class Test1(PSO):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._max_delta = self._upper_limits
        self._min_delta = self._lower_limits
        self._nswarm = 10 + 2 * int(np.sqrt(len(self._upper_limits)))

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)

        for i in range(0, self._nswarm):
            v = self._position[i, :]
            v = self._set_lim(v)
            x = v[0]
            y = v[1]
            # f_out[i] = (x - 1) ** 2
            f_out[i] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test2(SimulAnneal):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._max_delta = self._upper_limits/2
        self._min_delta = self._lower_limits/2
        self._position = np.array([0, 0])
        self._temperature = 0

    def calc_merit_function(self):
        v = self._position
        v = self._set_lim(v)
        x = v[0]
        y = v[1]
        # f_out[i] = (x - 1) ** 2
        f_out = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test3(SimpleScan):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._position = np.array([0, 0])

    def calc_merit_function(self):
        f_scan = np.zeros(len(self._delta))
        for j in range(0, len(self._delta)):
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
