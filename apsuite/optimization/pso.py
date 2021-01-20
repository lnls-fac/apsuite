#!/usr/bin/env python-sirius

from threading import Thread, Event
import numpy as _np

'''Particle Swarm Optimization Algorithm for Minimization'''


class PSO:
    """."""

    DEFAULT_COEFF_INERTIA = 0.7984
    DEFAULT_COEFF_INDIVIDUAL = 1.49618
    DEFAULT_COEFF_COLLECTIVE = 1.49618

    def __init__(self, save=False):
        """."""
        # Number of particles in the swarm # (Recommended is 10 + 2 * sqrt(d))
        # where d is the dimension of search space
        self._nswarm = 0
        self._niter = 0
        # Inertia
        self._coeff_inertia = PSO.DEFAULT_COEFF_INERTIA
        # Best position of individual particle
        self._coeff_indiv = PSO.DEFAULT_COEFF_INDIVIDUAL
        # Best position ever reached by the swarm
        self._coeff_coll = PSO.DEFAULT_COEFF_COLLECTIVE

        # Boundary limits of problem
        self._pos_lim_upper = _np.array([])
        self._pos_lim_lower = _np.array([])
        # Elements of PSO
        self._initial_position = _np.array([])
        self._position = self._initial_position
        self._velocity = _np.array([])
        self._best_indiv = _np.array([])
        self._best_global = _np.array([])

        self._thread = Thread(target=self._optimize, daemon=True)
        self._stopped = Event()
        self.hist_best_positions = _np.array([])
        self.hist_best_objfun = _np.array([])

        self._flag_save = save

        self.initialization()
        self._check_initialization()

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
        return len(self._initial_position)

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
    def limits_upper(self):
        """."""
        return self._pos_lim_upper

    @limits_upper.setter
    def limits_upper(self, value):
        """."""
        if len(value) != len(self._initial_position):
            raise Exception('Incompatible upper limit!')
        self._pos_lim_upper = _np.array(value)

    @property
    def limits_lower(self):
        """."""
        return self._pos_lim_lower

    @limits_lower.setter
    def limits_lower(self, value):
        """."""
        if len(value) != len(self._initial_position):
            raise Exception('Incompatible lower limit!')
        self._pos_lim_lower = _np.array(value)

    @property
    def initial_position(self):
        """."""
        return self._initial_position

    @initial_position.setter
    def initial_position(self, value):
        """."""
        self._initial_position = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    @property
    def velocity(self):
        """."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """."""
        self._velocity = value

    def initialization(self):
        """."""
        raise NotImplementedError

    def _check_initialization(self):
        """."""
        if len(self._pos_lim_upper) != len(self._pos_lim_lower):
            raise Exception(
                'Upper and Lower Limits has different lengths!')

        if self.ndim != len(self._pos_lim_upper):
            raise Exception(
                'Dimension incompatible with limits!')

        if self.nswarm < round(10 + 2 * _np.sqrt(self.ndim)):
            print(
                'Swarm population lower than recommended!')

    def _create_swarm(self):
        """."""
        self._best_indiv = _np.zeros((self.nswarm, self.ndim))
        self._best_global = _np.zeros(self.ndim)
        # Random initialization of swarm position inside the boundary limits
        dlim = self._pos_lim_upper - self._pos_lim_lower
        rarray = _np.random.rand(self.nswarm, self.ndim)
        self._position = _np.ones((self.nswarm, 1)) * self._initial_position
        self._position += dlim * rarray + self._pos_lim_lower
        # Include the zero variation as first particle in the swarm
        self._position[0, :] *= 0
        self._check_lim()
        # The first individual contribution will be zero
        self._best_indiv = self._position
        # Initializing with zero velocity
        self._velocity = _np.zeros((self.nswarm, self.ndim))

    def _update_position(self):
        """."""
        r_indiv = self._coeff_indiv * _np.random.rand()
        r_coll = self._coeff_coll * _np.random.rand()
        # Inertial velocity
        self._velocity *= self._coeff_inertia
        # Velocity dependent to distance from best individual position
        self._velocity += r_indiv * (self._best_indiv - self._position)
        # Velocity dependent to distance from best global position
        self._velocity += r_coll * (self._best_global - self._position)
        # Update position and check boundary limits
        self._position += self._velocity
        self._check_lim()

    def _check_lim(self):
        """."""
        # If particle position exceeds the boundary, set the boundary value
        for i in range(self._pos_lim_upper.size):
            over = self._position[:, i] > self._pos_lim_upper[i]
            under = self._position[:, i] < self._pos_lim_lower[i]
            self._position[over, i] = self._pos_lim_upper[i]
            self._position[under, i] = self._pos_lim_lower[i]

    def _save_data(self, k, f, fbest):
        """."""
        with open('pos_PSO.txt', 'a') as f_pos:
            if k == 0:
                f_pos.write('NEW RUN'.center(50, '=') + '\n')
            f_pos.write('Step ' + str(k+1) + ' \n')
            _np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_PSO.txt', 'a') as f_fig:
            if k == 0:
                f_fig.write('NEW RUN'.center(50, '=') + '\n')
            f_fig.write('Step ' + str(k+1) + ' \n')
            _np.savetxt(f_fig, f, fmt='%+.8e')
        with open('best_pos_history_PSO.txt', 'a') as f_posh:
            if k == 0:
                f_posh.write('NEW RUN'.center(50, '=') + '\n')
            f_posh.write('Step ' + str(k+1) + ' \n')
            _np.savetxt(f_posh, self._best_global, fmt='%+.8e')
        with open('best_fig_history_PSO.txt', 'a') as f_figh:
            if k == 0:
                f_figh.write('NEW RUN'.center(50, '=') + '\n')
            f_figh.write('Step ' + str(k+1) + ' \n')
            _np.savetxt(f_figh, _np.array([fbest]), fmt='%+.8e')

    def calc_obj_fun(self):
        """Return a vector for every particle evaluation."""
        raise NotImplementedError

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._thread = Thread(target=self._optimize, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        """."""
        self._stopped.set()

    def join(self):
        """."""
        self._thread.join()

    @property
    def isrunning(self):
        """."""
        return self._thread.is_alive()

    def _optimize(self):
        """."""
        self._create_swarm()

        f_old = _np.zeros(self.nswarm)
        f_new = _np.zeros(self.nswarm)

        # History of best position and merit function over iteractions
        best_pos_hstry = _np.zeros((self.niter, self.ndim))
        best_fig_hstry = _np.zeros(self.niter)

        print('>>> Iteraction Number: 1')
        f_old = self.calc_obj_fun()
        self._best_global = self._best_indiv[_np.argmin(f_old), :]
        best_pos_hstry[0, :] = self._best_global
        best_fig_hstry[0] = _np.min(f_old)
        ref0 = self._best_global

        if self._flag_save:
            self._save_data(k=0, f=f_old, fbest=best_fig_hstry[0])
        print('Best particle: ' + str(_np.argmin(f_old)+1))
        print('Obj. Func.:' + str(_np.min(f_old)))

        for niter in range(self.niter):
            print('------------------------------------------------------')
            print('>>> Iteraction Number: ' + str(niter+2))
            self._update_position()
            f_new = self.calc_obj_fun()
            improve = f_new < f_old
            if improve.any():
                # Update best individual position and merit function for
                # comparison only if the merit function is lower
                self._best_indiv[improve, :] = self._position[improve, :]
                if _np.min(f_new) < _np.min(f_old):
                    self._best_global = self._best_indiv[
                        _np.argmin(f_new), :].copy()
                    print('UPDATE GLOBAL BEST!')
                    print(
                        'Best particle: ' + str(_np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(_np.min(f_new)))
                    f_old[improve] = f_new[improve]
                else:
                    print('Best particle: ' + str(_np.argmin(f_new)+1))
                    print('Obj. Func.:' + str(_np.min(f_new)))

            best_pos_hstry[niter, :] = self._best_global
            best_fig_hstry[niter] = _np.min(f_old)
            if self._flag_save:
                self._save_data(k=niter, f=f_new, fbest=best_fig_hstry[niter])
            if self._stopped.is_set():
                print('Stopped!')
                break

        print('Finished!')
        print('Best Position Found:' + str(self._best_global))
        print('Best Obj. Func. Found:' + str(_np.min(f_old)))

        self.hist_best_positions = best_pos_hstry
        self.hist_best_objfun = best_fig_hstry
