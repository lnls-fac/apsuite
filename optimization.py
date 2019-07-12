#!/usr/bin/env python-sirius

import numpy as np


'''Particle Swarm Optimization Algorithm for Minimization'''


class PSO:

    def __init__(self, nswarm=None):
        self._nswarm = nswarm
        self._c_inertia = 0.7984
        self._c_indiv = 1.49618
        self._c_coll = self.c_indiv
        self._upper_limits = np.array([])
        self._lower_limits = np.array([])
        self.initialization()
        self._ndim = len(self._upper_limits)
        self._check_initialization()
        self._position = np.array([])
        self._velocity = np.array([])
        self._best_particle = np.array([])
        self._best_global = np.array([])

    @property
    def c_inertia(self):
        return self._c_inertia

    @c_inertia.setter
    def c_inertia(self, value):
        self._c_inertia = value

    @property
    def c_indiv(self):
        return self._c_indiv

    @c_indiv.setter
    def c_indiv(self, value):
        self._c_indiv = value

    @property
    def c_coll(self):
        return self._c_coll

    @c_coll.setter
    def c_coll(self, value):
        self._c_coll = value

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = value

    def _create_swarm(self):
        self._best_particle = np.zeros((self._nswarm, self._ndim))
        self._best_global = np.zeros(self._ndim)
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._nswarm, self._ndim)
        self._position = dlim * rarray + self._lower_limits
        self._best_particle = self._position
        self._velocity = np.zeros((self._nswarm, self._ndim))

    def _set_lim(self):
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def initialization(self):
        pass

    def _check_initialization(self):
        if len(self._upper_limits) != len(self._lower_limits):
            print('Warning: Upper and Lower Limits has different lengths')

        if self._ndim != len(self._upper_limits):
            print('Warning: Dimension incompatible with limits!')

        if self._nswarm < int(10 + 2 * np.sqrt(self._ndim)):
            print('Warning: Swarm population lower than recommended!')

    def calc_merit_function(self):
        return np.zeros(self._nswarm)

    def _update_position(self):
        r_indiv = self._c_indiv * np.random.rand()
        r_coll = self._c_coll * np.random.rand()
        self._velocity = self._c_inertia * self._velocity
        self._velocity += r_indiv * (self._best_particle - self._position)
        self._velocity += r_coll * (self._best_global - self._position)
        self._position = self._position + self._velocity
        self._set_lim()

    def _start_optimization(self, niter):
        self._create_swarm()

        f_old = np.zeros(self._nswarm)
        f_new = np.zeros(self._nswarm)

        f_old = self.calc_merit_function()
        self._best_global = self._best_particle[np.argmin(f_old), :]

        k = 0
        while k < niter:
            self._update_position()
            f_new = self.calc_merit_function()
            improve = f_new < f_old
            self._best_particle[improve, :] = self._position[improve, :]
            f_old[improve] = f_new[improve]
            self._best_global = self._best_particle[np.argmin(f_old), :]
            if improve.any():
                print('Global best updated:' + str(self._best_global))
                print('Figure of merit updated:' + str(np.min(f_old)))
            k += 1

        print('Best Position Found:' + str(self._best_global))
        print('Best Figure of Merit Found:' + str(np.min(f_old)))


''' Simulated Annealing Algorithm for Minimization'''


class SimulAnneal():

    def __init__(self):
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self._temperature = 0
        self.initialization()
        self._ndim = len(self._lower_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass

    def _set_lim(self):
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def calc_merit_function(self):
        return 0

    def random_change(self):
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._ndim)
        self._delta = dlim * rarray + self._lower_limits
        self._position = self._position + self._delta
        self._set_lim()

    def _start_optimization(self, niter):
        f_old = self.calc_merit_function()
        best = self._position
        k = 0
        n_acc = 0
        nu = 0

        while k < niter:
            flag_acc = False
            self.random_change()
            f_new = self.calc_merit_function()

            if f_new < f_old:
                flag_acc = True
                best = self._position
                print('Better solution found! ' + str(best))
                nu = 0
            elif f_new > f_old and self._temperature != 0:
                df = f_new - f_old
                if np.random.rand(1) < np.exp(- df / self._temperature):
                    flag_acc = True
                    print('Worse solution accepted! ' + str(self._position))
                else:
                    flag_acc = False
            else:
                flag_acc = False

            if flag_acc:
                f_old = f_new
                n_acc += 1
                print('Number of accepted solutions: ' + str(n_acc))
            else:
                self._position = self._position - self._delta
                nu += 1

            k += 1

            if self._temperature != 0:
                phi = 1 / (1 + 1 / np.sqrt(k * (nu + 1) + nu))
                self._temperature = phi * self._temperature
                print('Temperature is: ' + str(self._temperature))

        print('Best solution is: ' + str(best))
        print('Best figure of merit is: ' + str(f_old))
        print('Number of accepted solutions: ' + str(n_acc))


'''Multidimensional Simple Scan method for Minimization'''


class SimpleScan():

    def __init__(self):
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self._curr_dim = 0
        self.initialization()
        self._ndim = len(self._upper_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass

    def _set_lim(self):
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def calc_merit_function(self):
        return np.zeros(self._ndim), np.zeros(self._ndim)

    def _start_optimization(self, npoints):
        self._delta = np.zeros(npoints)
        f = np.zeros(self._ndim)
        best = np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = np.linspace(
                                self._lower_limits[i],
                                self._upper_limits[i],
                                npoints)
            self._curr_dim = i
            f[i], best[i] = self.calc_merit_function()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(np.min(f)))

''' Powell Conjugated Direction Search Method for Minimization


class Powell():

    GOLDEN = (np.sqrt(5) - 1)/2

    def __init__(self):
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self.initialization()
        self._ndim = len(self._upper_limits)
        self._check_initialization()

    def initialization(self):
        pass

    def _check_initialization(self):
        pass



    def calc_merit_function(self):
        return np.zeros(self._ndim)

    def golden_search(self):
        k = 0
        x = self._position
        x_upper = x[1]
        x_lower = x[0]
        d = GOLDEN * (x_upper - x_lower)
        x1 = x_lower + d
        x2 = x_upper - d
        f1 = self.calc_merit_func(x1)
        f2 = self.calc_merit_func(x2)

        while k < self._nint:
            if f1 > f2:
                x_lower = x2
                x2 = x1
                f2 = f1
                x1 = x_lower + GOLDEN * (x_upper - x_lower)
                f1 = self.calc_merit_func(x1)
            elif f2 > f1:
                x_upper = x1
                x1 = x2
                f1 = f2
                x2 = x_upper - GOLDEN * (x_upper - x_lower)
                f2 = self.calc_merit_func(x2)
            k += 1
        return x_lower, x_upper

    def line_scan(self):
        pass

    def _start_optimization(self, niter):
        pass
        '''
