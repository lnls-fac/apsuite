"""Simulated Annealing Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event

import numpy as _np


class SimulAnneal:
    """."""

    # NOTE: objects with threading.Event cannot be serialized with pickle

    def __init__(self, save=False, use_thread=True):
        """."""
        self._use_thread = use_thread

        # search space
        self._position = _np.array([])
        self._pos_lim_lower = None
        self._pos_lim_upper = None
        self._pos_delta_max = None

        # search control
        self._niter = 0
        self._temperature = 0
        self._flag_save = save

        if self._use_thread:
            self._thread = None
            self._stopevt = _Event()
        self.hist_best_positions = _np.array([])
        self.hist_best_objfunc = _np.array([])

        # initialization
        self.initialization()
        self._check_initialization()

    @property
    def ndim(self):
        """."""
        return len(self._position)

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

    @property
    def limits_upper(self):
        """."""
        return self._pos_lim_upper

    @limits_upper.setter
    def limits_upper(self, value):
        """."""
        if len(value) != len(self._position):
            raise Exception('Incompatible upper limit!')
        self._pos_lim_upper = _np.array(value)

    @property
    def limits_lower(self):
        """."""
        return self._pos_lim_lower

    @limits_lower.setter
    def limits_lower(self, value):
        """."""
        if len(value) != len(self._position):
            raise Exception('Incompatible lower limit!')
        self._pos_lim_lower = _np.array(value)

    @property
    def deltas(self):
        """."""
        return self._pos_delta_max

    @deltas.setter
    def deltas(self, value):
        """."""
        if len(value) != len(self._position):
            raise Exception('Incompatible deltas!')
        self._pos_delta_max = _np.array(value)

    @temperature.setter
    def temperature(self, value):
        """."""
        self._temperature = value

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return a number."""
        raise NotImplementedError

    def start(self, print_flag=True):
        """."""
        if self._use_thread:
            if not self._thread.is_alive():
                self._thread = _Thread(
                    target=self._optimize, args=(print_flag, ), daemon=True)
                self._stopevt.clear()
                self._thread.start()
        else:
            self._optimize(print_flag=print_flag)

    def stop(self):
        """."""
        if self._use_thread:
            self._stopevt.set()

    def join(self):
        """."""
        if self._use_thread:
            self._thread.join()

    @property
    def isrunning(self):
        """."""
        if self._use_thread:
            return self._thread.is_alive()
        else:
            return False

    def _optimize(self, print_flag=True):
        """."""
        if not self.niter:
            return

        bpos_hstry = _np.zeros([self.niter, self.ndim])
        bfig_hstry = _np.zeros([self.niter])

        # initial position
        p_old, f_old = _np.copy(self._position), self.calc_obj_fun()
        bpos_hstry[0, :] = p_old
        bfig_hstry[0] = f_old

        # Number of accepted and unaccepted solutions.
        nr_acc, nr_unacc = 0, 0

        if self._flag_save:
            self._save_data(kiter=0, func=f_old, acc=False)

        for k in range(self.niter):

            if print_flag:
                print('>>> Iteraction Number:' + str(k+1))

            flag_acc = False  # Flag that a solution was accepted

            self._random_change()
            f_new = self.calc_obj_fun()

            if f_new < f_old:
                # Accepting solution if it improves the merit function
                flag_acc = True
                nr_unacc = 0
            elif f_new > f_old and self._temperature != 0:
                # If solution increases the merit function there is a chance
                # to accept it
                f_dif = f_new - f_old
                if _np.random.rand() < _np.exp(- f_dif / self._temperature):
                    flag_acc = True
                    if print_flag:
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
                nr_acc += 1
                p_old, f_old = _np.copy(self._position), f_new
                bpos_hstry[nr_acc, :] = self._position
                bfig_hstry[nr_acc] = f_old
                if print_flag:
                    print('Better solution found! Obj. Func: '
                          '{:5f}'.format(f_old))
                    print('Number of accepted solutions: ' + str(nr_acc))
            else:
                self._position = p_old
                nr_unacc += 1

            if self._flag_save:
                self._save_data(
                    kiter=k+1, func=f_old, acc=flag_acc,
                    nacc=nr_acc, bpos=bpos_hstry,
                    bfunc=bfig_hstry)

            if self._temperature != 0:
                # Reduces the temperature based on number of iterations
                # without accepting solutions
                # Ref: An Optimal Cooling Schedule Using a Simulated Annealing
                # Based Approach - A. Peprah, S. Appiah, S. Amponsah
                phi = 1 / (1 + 1 / _np.sqrt((k+1) * (nr_unacc + 1) + nr_unacc))
                self._temperature = phi * self._temperature

            if self._use_thread and self._stopevt.is_set():
                if print_flag:
                    print('Stopped!')
                break

        if print_flag:
            print('Finished!')
        if nr_acc:
            bpos_hstry = bpos_hstry[:nr_acc+1, :]
            bfig_hstry = bfig_hstry[:nr_acc+1]
            if print_flag:
                print('Best solution found: ' + str(bpos_hstry[-1, :]))
                print(
                    'Best Obj. Func. found: ' + str(bfig_hstry[-1]))
                print('Number of accepted solutions: ' + str(nr_acc))
        else:
            bpos_hstry = bpos_hstry[0, :]
            bfig_hstry = bfig_hstry[0]
            if print_flag:
                print('It was not possible to find a better solution...')

        self.hist_best_positions = bpos_hstry
        self.hist_best_objfunc = bfig_hstry

    def _check_initialization(self):
        """."""
        if self._pos_lim_upper is None and self._pos_lim_lower is None:
            return

        if len(self._pos_lim_upper) != len(self._pos_lim_lower):
            raise Exception(
                'Upper and Lower Limits has different lengths')

        if self.ndim != len(self._pos_lim_upper):
            raise Exception(
                'Dimension incompatible with limits!')

    def _check_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        if self._pos_lim_upper is not None:
            over = self._position > self._pos_lim_upper
            self._position[over] = self._pos_lim_upper[over]
        if self._pos_lim_lower is not None:
            under = self._position < self._pos_lim_lower
            self._position[under] = self._pos_lim_lower[under]

    def _random_change(self):
        # Random change applied in the current position
        rarray = 2 * _np.random.rand(self.ndim) - 1  # [-1,1]
        delta = self._pos_delta_max * rarray
        self._position = self._position + delta
        self._check_lim()

    def _save_data(self, kiter, func, acc=False, nacc=None,
                   bpos=None, bfunc=None):
        """."""
        with open('pos_SA.txt', 'a') as f_pos:
            if kiter == 0:
                f_pos.write('NEW RUN'.center(50, '=') + '\n')
            f_pos.write('Step ' + str(kiter+1) + ' \n')
            _np.savetxt(f_pos, self._position, fmt='%+.8e')
        with open('fig_SA.txt', 'a') as f_fig:
            if kiter == 0:
                f_fig.write('NEW RUN'.center(50, '=') + '\n')
            f_fig.write('Step ' + str(kiter+1) + ' \n')
            _np.savetxt(f_fig, _np.array([func]), fmt='%+.8e')
        if acc:
            with open('best_pos_history_SA.txt', 'a') as f_posh:
                if nacc == 1:
                    f_posh.write('NEW RUN'.center(50, '=') + '\n')
                f_posh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                _np.savetxt(f_posh, bpos[nacc, :], fmt='%+.8e')
            with open('best_fig_history_SA.txt', 'a') as f_figh:
                if nacc == 1:
                    f_figh.write('NEW RUN'.center(50, '=') + '\n')
                f_figh.write('Accep. Solution ' + str(nacc+1) + ' \n')
                _np.savetxt(f_figh, _np.array([bfunc[nacc]]), fmt='%+.8e')
