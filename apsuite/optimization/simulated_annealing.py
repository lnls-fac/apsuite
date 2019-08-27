#!/usr/bin/env python-sirius

import numpy as np


''' Simulated Annealing Algorithm for Minimization'''


class SimulAnneal:
    """."""

    @property
    def ndim(self):
        """."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """."""
        self._ndim = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    @property
    def niter(self):
        """."""
        return self._niter

    @niter.setter
    def niter(self, value):
        """."""
        self._niter = value

    @property
    def temperature(self):
        """."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """."""
        self._temperature = value

    def __init__(self):
        """."""
        # Boundary Limits
        self._ndim = []
        self._niter = []
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        # Maximum variation to be applied
        self._max_delta = np.array([])
        # Reference configuration
        self._position = np.array([])
        # Variation to be applied
        self._delta = np.array([])
        # Initial temperature of annealing
        self._temperature = 0
        self.initialization()

    def initialization(self):
        """."""
        raise NotImplementedError

    def _check_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        over = self._position > self._upper_limits
        under = self._position < self._lower_limits
        self._position[over] = self._upper_limits[over]
        self._position[under] = self._lower_limits[under]

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)

    def set_deltas(self, dmax=None):
        """."""
        self.ndim = len(dmax)
        self._max_delta = dmax

    def get_change(self):
        """."""
        raise NotImplementedError

    def set_change(self):
        """."""
        raise NotImplementedError

    def calc_merit_function(self):
        """Return a number."""
        raise NotImplementedError

    def _random_change(self):
        # Random change applied in the current position
        dlim = self._max_delta
        rarray = 2 * np.random.rand(self.ndim) - 1  # [-1,1]
        self._delta = dlim * rarray
        self._position = self._position + self._delta
        # self._check_lim()

    def _save_data(self, k, f, acc=False, nacc=None, bp=None, bf=None):
        """."""
        with open('pos_SA.txt', 'a') as f_pos:
            if k == 0:
                f_pos.write('==============NEW RUN==========\n')
            f_pos.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_SA.txt', 'a') as f_fig:
            if k == 0:
                f_fig.write('==============NEW RUN==========\n')
            f_fig.write('Step ' + str(k+1) + ' \n')
            np.savetxt(f_fig, np.array([f]), fmt='%+.8e')
        if acc:
            with open('best_pos_history_SA.txt', 'a') as f_posh:
                if nacc == 1:
                    f_posh.write('==============NEW RUN==========\n')
                f_posh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                np.savetxt(f_posh, bp[nacc, :], fmt='%+.8e')
            with open('best_fig_history_SA.txt', 'a') as f_figh:
                if nacc == 1:
                    f_figh.write('==============NEW RUN==========\n')
                f_figh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                np.savetxt(f_figh, np.array([bf[nacc]]), fmt='%+.8e')

    def start_optimization(self):
        """."""
        bpos_hstry = np.zeros([self.niter, self.ndim])
        bfig_hstry = np.zeros([self.niter])

        f_old = self.calc_merit_function()
        bfig_hstry[0] = f_old
        bpos_hstry[0, :] = self._position
        # Number of accepted solutions
        n_acc = 0
        # Number of iteraction without accepting solutions
        nu = 0

        self._save_data(k=0, f=f_old, acc=False)

        for k in range(self.niter):
            # Flag that a solution was accepted
            flag_acc = False
            self._random_change()
            print('>>> Iteraction Number:' + str(k+1))
            f_new = self.calc_merit_function()

            if f_new < f_old:
                # Accepting solution if it reduces the merit function
                flag_acc = True
                nu = 0
            elif f_new > f_old and self._temperature != 0:
                # If solution increases the merit function there is a chance
                # to accept it
                df = f_new - f_old
                if np.random.rand() < np.exp(- df / self._temperature):
                    flag_acc = True
                    print('Worse solution accepted! ' + str(self._position))
                    print('Temperature is: ' + str(self._temperature))
                else:
                    flag_acc = False
            else:
                # If temperature is zero the algorithm only accepts good
                # solutions
                flag_acc = False

            if flag_acc:
                # Stores the number of accepted solutions
                f_old = f_new
                n_acc += 1
                bpos_hstry[n_acc, :] = self._position
                bfig_hstry[n_acc] = f_old
                print('Better solution found! Obj. Func: {:5f}'.format(f_old))
                print('Number of accepted solutions: ' + str(n_acc))
            else:
                self._position = self._position - self._delta
                nu += 1

            self._save_data(
                k=k+1, f=f_old, acc=flag_acc, nacc=n_acc, bp=bpos_hstry,
                bf=bfig_hstry)

            if self._temperature != 0:
                # Reduces the temperature based on number of iteractions
                # without accepting solutions
                # Ref: An Optimal Cooling Schedule Using a Simulated Annealing
                # Based Approach - A. Peprah, S. Appiah, S. Amponsah
                phi = 1 / (1 + 1 / np.sqrt((k+1) * (nu + 1) + nu))
                self._temperature = phi * self._temperature
        if n_acc:
            bpos_hstry = bpos_hstry[:n_acc, :]
            bfig_hstry = bfig_hstry[:n_acc]

            print('Best solution found: ' + str(bpos_hstry[-1, :]))
            print(
                'Best Obj. Func. found: ' + str(bfig_hstry[-1]))
            print('Number of accepted solutions: ' + str(n_acc))
        else:
            bpos_hstry = bpos_hstry[0, :]
            bfig_hstry = bfig_hstry[0]
            print('It was not possible to find a better solution...')

        return bpos_hstry, bfig_hstry
