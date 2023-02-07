'''Particle Swarm Optimization Algorithm for Minimization'''

from threading import Thread, Event
import logging as _log

import numpy as _np

from .base import Optimize as _Optimize, OptimizeParams as _OptimizeParams


class PSOParams(_OptimizeParams):
    """."""

    DEF_INERTIA = 0.7984
    DEF_INDIVIDUAL = 1.49618
    DEF_COLLECTIVE = 1.49618

    def __init__(self):
        """."""
        super().__init__()
        self.number_of_particles = 13  # (Recommended is 10 + 2 * sqrt(#dim))
        self.initial_swarm_fraction_size = 0.1
        self.coeff_inertia = self.DEF_INERTIA
        self.coeff_individual = self.DEF_INDIVIDUAL
        self.coeff_collective = self.DEF_COLLECTIVE

    def __str__(self):
        """."""
        stg = ''
        stg += self._TMPD.format(
            'number_of_particles', self.number_of_particles, '')
        stg += self._TMPF.format('coeff_inertia', self.coeff_inertia, '')
        stg += self._TMPF.format('coeff_individual', self.coeff_individual, '')
        stg += self._TMPF.format('coeff_collective', self.coeff_collective, '')
        stg += super().__str__()
        return stg


class PSO(_Optimize):
    """."""

    def __init__(self, use_thread=True):
        """."""
        super().__init__(PSOParams(), use_thread=use_thread)
        self.params = PSOParams()

        self._positions = _np.array([], ndmin=2)
        self._velocity = _np.array([], ndmin=2)
        self._best_indiv = _np.array([])
        self._best_global = _np.array([])

    def to_dict(self) -> dict:
        """Dump all relevant object properties to dictionary.

        Returns:
            dict: contains all relevant properties of object.

        """
        dic = super().to_dict()
        dic['positions'] = self._positions
        dic['velocity'] = self._velocity
        dic['best_indiv'] = self._best_indiv
        dic['best_global'] = self._best_global
        return dic

    def from_dict(self, info: dict):
        """Update all relevant info from dictionary.

        Args:
            info (dict): dictionary with all relevant info.

        Returns:
            keys_not_used (set): Set containing keys not used by
                `self.params` object.

        """
        super().from_dict(info)
        self._positions = info['positions']
        self._velocity = info['velocity']
        self._best_indiv = info['best_indiv']
        self._best_global = info['best_global']

    @property
    def positions(self):
        """."""
        return self._positions

    @positions.setter
    def positions(self, value):
        """."""
        self._positions = value

    @property
    def velocity(self):
        """."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """."""
        self._velocity = value

    def _create_swarm(self):
        """."""
        nswarm = self.params.number_of_particles
        ndim = self.params.initial_position.size

        # best_global will be initialized after first evaluation:
        self._best_global = _np.zeros(ndim, dtype=float, ndmin=2)
        # Initializing with zero velocity:
        self._velocity = _np.zeros((nswarm, ndim), dtype=float)

        # Random initialization of swarm position inside the boundary limits
        dlim = self.params.limit_upper - self.params.limit_lower
        dlim *= self.params.initial_swarm_fraction_size
        self._positions = _np.random.rand(nswarm, ndim) - 0.5
        self._positions *= dlim[None, :]
        self._positions[0, :] *= 0  # make sure initial_position is in swarm
        self._positions += self.params.initial_position[None, :]
        self._positions = self.params.check_and_adjust_boundary(
            self._positions)

        # Initialize best individual position:
        self._best_indiv = self._positions.copy()

    def _update_position(self):
        """."""
        r_indiv = self.params.coeff_individual * _np.random.rand()
        r_coll = self.params.coeff_collective * _np.random.rand()

        self._velocity *= self.params.coeff_inertia
        self._velocity += r_indiv * (self._best_indiv - self._positions)
        self._velocity += r_coll * (self._best_global - self._positions)

        # Update position and check boundary limits
        self._positions += self._velocity
        self._positions = self.params.check_and_adjust_boundary(
            self._positions)

    def _optimize(self):
        """."""
        self.params.is_positions_consistent()
        self._num_objective_evals = 0
        niter = self.params.max_number_iters
        nevals = self.params.max_number_evals

        self._create_swarm()
        f_old = _np.zeros(self.params.number_of_particles)
        f_new = _np.zeros(self.params.number_of_particles)

        # History of best position and merit function over iteractions
        best_pos_hstry = []
        best_fig_hstry = []

        _log.info('>>> Iteraction Number: 1')
        f_old = self._objective_func(self._positions)
        idx_min_old = _np.argmin(f_old)

        self._best_global = self._best_indiv[idx_min_old]
        best_pos_hstry.append(self._best_global)
        best_fig_hstry.append(f_old[idx_min_old])

        _log.info('Best particle: ' + str(idx_min_old+1))
        _log.info('Obj. Func.:' + str(f_old[idx_min_old]))

        for iter in range(niter):
            _log.info('------------------------------------------------------')
            _log.info('>>> Iteraction Number: ' + str(iter+2))
            self._update_position()
            f_new = self._objective_func(self._positions)
            improve = f_new < f_old
            if improve.any():
                # Update best individual position and merit function for
                # comparison only if the merit function is lower
                self._best_indiv[improve] = self._positions[improve]

                idx_min_new = _np.argmin(f_new)
                if f_new[idx_min_new] < f_old[idx_min_old]:
                    self._best_global = self._best_indiv[idx_min_new].copy()
                    idx_min_old = idx_min_new
                    _log.info('UPDATE GLOBAL BEST!')

                f_old[improve] = f_new[improve]
                _log.info('Best particle: ' + str(idx_min_new+1))
                _log.info('Obj. Func.:' + str(f_new[idx_min_new]))

            best_pos_hstry.append(self._best_global)
            best_fig_hstry.append(f_old[idx_min_old])

            if self._stopevt.is_set():
                _log.info('User Stopped!')
                break
            elif nevals < self._num_objective_evals:
                _log.info('Maximum Number of objective evaluations reached.')
                break

        _log.info('Finished!')
        _log.info('Best Obj. Func. Found:' + str(f_old[idx_min_old]))

        self.best_positions = _np.array(best_pos_hstry)
        self.best_objfuncs = _np.array(best_fig_hstry)
