#!/usr/bin/env python-sirius

from threading import Thread
import numpy as np

'''Particle Swarm Optimization Algorithm for Minimization'''


class PSO:
    """."""

    @property
    def coeff_inertia(self):
        """."""
        return self._coeff_inertia

    @coeff_inertia.setter
    def coeff_inertia(self, value):
        """."""
        self._coeff_inertia = value

    @property
    def coeff_indiv(self):
        """."""
        return self._coeff_indiv

    @coeff_indiv.setter
    def coeff_indiv(self, value):
        """."""
        self._coeff_indiv = value

    @property
    def coeff_coll(self):
        """."""
        return self._coeff_coll

    @coeff_coll.setter
    def coeff_coll(self, value):
        """."""
        self._coeff_coll = value

    @property
    def ndim(self):
        """."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """."""
        self._ndim = value

    @property
    def nswarm(self):
        """."""
        return self._nswarm

    @nswarm.setter
    def nswarm(self, value):
        """."""
        self._nswarm = value

    @property
    def niter(self):
        """."""
        return self._niter

    @niter.setter
    def niter(self, value):
        """."""
        self._niter = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    def __init__(self, save=False):
        """."""
        # Number of particles in the swarm # (Recommended is 10 + 2 * sqrt(d))
        # where d is the dimension of search space
        self._nswarm = 0
        self._niter = 0
        self._coeff_inertia = 0.7984  # Inertia
        self._coeff_indiv = 1.49618  # Best position of individual particle
        self._coeff_coll = self._coeff_indiv  # Best position ever reached by
        # the swarm

        # Boundary limits of problem
        self._upper_limits = np.array([])
        self._lower_limits = np.array([])
        # Elements of PSO
        self._position = np.array([])
        self._velocity = np.array([])
        self._best_indiv = np.array([])
        self._best_global = np.array([])
        self._stop = False
        self._thread = Thread(target=self._optimize, daemon=True)
        self.best_positions_history = np.array([])
        self.best_figures_history = np.array([])

        self._flag_save = save

        self.initialization()
        self._check_initialization()

    def initialization(self):
        """."""
        raise NotImplementedError

    def _check_initialization(self):
        """."""
        if len(self._upper_limits) != len(self._lower_limits):
            raise Exception(
                'Upper and Lower Limits has different lengths!')

        if self._ndim != len(self._upper_limits):
            raise Exception(
                'Dimension incompatible with limits!')


        if self._nswarm < int(10 + 2 * np.sqrt(self._ndim)):
            raise Warning(
                'Swarm population lower than recommended!')

    def _create_swarm(self):
        """."""
        self._best_indiv = np.zeros((self._nswarm, self._ndim))
        self._best_global = np.zeros(self._ndim)
        # Random initialization of swarm position inside the boundary limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._nswarm, self._ndim)
        self._position = dlim * rarray + self._lower_limits
        # Include the zero variation as first particle in the swarm
        self._position[0, :] *= 0
        # The first individual contribution will be zero
        self._best_indiv = self._position
        # Initializing with zero velocity
        self._velocity = np.zeros((self._nswarm, self._ndim))

    def _update_position(self):
        """."""
        r_indiv = self._coeff_indiv * np.random.rand()
        r_coll = self._coeff_coll * np.random.rand()
        # Inertial velocity
        self._velocity = self._coeff_inertia * self._velocity
        # Velocity dependent to distance from best individual position
        self._velocity += r_indiv * (self._best_indiv - self._position)
        # Velocity dependent to distance from best global position
        self._velocity += r_coll * (self._best_global - self._position)
        # Update position and check boundary limits
        self._position = self._position + self._velocity
        self._check_lim()

    def _check_lim(self):
        """."""
        # If particle position exceeds the boundary, set the boundary value
        for i in range(self._upper_limits.size):
            over = self._position[:, i] > self._upper_limits[i]
            under = self._position[:, i] < self._lower_limits[i]
            self._position[over, i] = self._upper_limits[i]
            self._position[under, i] = self._lower_limits[i]

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)
        if not self.nswarm:
            self.nswarm = int(10 + 2 * np.sqrt(self.ndim))

    def _save_data(self, k, f, fbest):
        """."""
        with open('pos_PSO.txt', 'a') as f_pos:
            if k == 0:
                f_pos.write('NEW RUN'.center(50, '='))
            f_pos.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_PSO.txt', 'a') as f_fig:
            if k == 0:
                f_fig.write('NEW RUN'.center(50, '='))
            f_fig.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_fig, f, fmt='%+.8e')
        with open('best_pos_history_PSO.txt', 'a') as f_posh:
            if k == 0:
                f_posh.write('NEW RUN'.center(50, '='))
            f_posh.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_posh, self._best_global, fmt='%+.8e')
        with open('best_fig_history_PSO.txt', 'a') as f_figh:
            if k == 0:
                f_figh.write('NEW RUN'.center(50, '='))
            f_figh.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_figh, np.array([fbest]), fmt='%+.8e')

    def calc_obj_fun(self):
        """Return a vector for every particle evaluation."""
        raise NotImplementedError

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._stop = False
            self._thread = Thread(target=self._optimize, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop = True

    @property
    def isrunning(self):
        return self._thread.is_alive()

    def _optimize(self):
        """."""
        self._create_swarm()

        f_old = np.zeros(self._nswarm)
        f_new = np.zeros(self._nswarm)

        # History of best position and merit function over iteractions
        best_pos_hstry = np.zeros([self.niter, self._ndim])
        best_fig_hstry = np.zeros(self.niter)

        print('>>> Iteraction Number:1')
        f_old = self.calc_obj_fun()
        self._best_global = self._best_indiv[np.argmin(f_old), :]
        best_pos_hstry[0, :] = self._best_global
        best_fig_hstry[0] = np.min(f_old)

        if self._flag_save:
            self._save_data(k=0, f=f_old, fbest=best_fig_hstry[0])
        print('Best particle: ' + str(np.argmin(f_old)+1))
        print('Obj. Func.:' + str(np.min(f_old)))

        k = 1
        while k < self.niter:
            print('>>> Iteraction Number:' + str(k+1))
            self._update_position()
            f_new = self.calc_obj_fun()
            improve = f_new < f_old
            if improve.any():
                # Update best individual position and merit function for
                # comparison only if the merit function is lower
                self._best_indiv[improve, :] = self._position[improve, :]
                if np.min(f_new) < np.min(f_old):
                    self._best_global = self._best_indiv[
                        np.argmin(f_new), :]
                    print('Update global best!')
                    print(
                        'Best particle: ' + str(np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(np.min(f_new)))
                    f_old[improve] = f_new[improve]
                else:
                    print('Best particle: ' + str(np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(np.min(f_new)))

            best_pos_hstry[k, :] = self._best_global
            best_fig_hstry[k] = np.min(f_old)
            if self._flag_save:
                self._save_data(k=k, f=f_new, fbest=best_fig_hstry[k])
            k += 1

        print('Best Position Found:' + str(self._best_global))
        print('Best Obj. Func. Found:' + str(np.min(f_old)))

        self.best_positions_history = best_pos_hstry
        self.best_figures_history = best_fig_hstry
