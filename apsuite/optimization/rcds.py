"""Robust Conjugated Direction Search Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event

import numpy as _np


class RCDS:
    """The original algorithm was developed by X. Huang, SLAC."""

    # NOTE: objects with threading.Event cannot be serialized with pickle

    GOLDEN_RATIO = (1 + _np.sqrt(5))/2

    def __init__(self, save=False, use_thread=True):
        """."""
        self._use_thread = use_thread

        # search space
        self._position = _np.array([])
        self._pos_lim_lower = None
        self._pos_lim_upper = None

        self.gnoise = None
        self.gcount = None
        self.gdata = None
        self.grange = None
        self.imat = None
        # search control
        self._niter = 0
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
        return None

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

    def bracketing_min(self, pos0, func0, dpos, step):
        """."""
        nevals_func = 0
        if _np.isnan(func0):
            func0 = self.calc_obj_fun(pos0)
            nevals_func += 1

        xflist = _np.array([[0, func0]])
        func_min = func0
        alpha_min = 0
        pos_min = pos0

        step_init = step

        info = dict()
        info['pos0'] = pos0
        info['dpos'] = dpos
        info['step'] = step
        info['nevals_func'] = nevals_func
        info['xflist'] = xflist
        info['alpha_min'] = alpha_min
        info['pos_min'] = pos_min
        info['func_min'] = func_min

        # Looking for first bound
        info = self.search_bound(info, direction='positive')

        alpha2 = info['alpha_bound']
        pos_min, func_min = info['pos_min'], info['func_min']
        alpha_min = info['alpha_min']
        xflist = info['xflist']
        nevals_func = info['nevals_func']
        if func0 > (func_min + 3 * self.gnoise):
            # actually we found the second bound and the first bound was pos0.
            # pos_min is the new reference
            # - first bound:  pos1 = pos0 = pos_min + (-alpha_min)*dpos
            # - second bound: pos2 = pos_min + (alpha2 - alpha_min)*dpos
            alpha1 = -alpha_min
            alpha2 = alpha2 - alpha_min
            xflist[:, 0] -= alpha_min  # subtract alpha_min for all alphas
            return pos_min, func_min, alpha1, alpha2, xflist, nevals_func

        info['step'] = -step_init
        info = self.search_bound(info, direction='negative')
        alpha1 = info['alpha_bound']
        pos_min, func_min = info['pos_min'], info['func_min']
        alpha_min = info['alpha_min']
        xflist = info['xflist']
        nevals_func = info['nevals_func']

        # we found a second bound, just check the order
        if alpha1 > alpha2:
            alpha1, alpha2 = alpha2, alpha1

        # subtract alpha_min since pos_min is the new reference
        alpha1 -= alpha_min
        alpha2 -= alpha_min
        xflist[:, 0] -= alpha_min
        xflist = xflist[_np.argsort(xflist[:, 0])]
        return pos_min, func_min, alpha1, alpha2, xflist, nevals_func

    def search_bound(self, info, direction='positive'):
        """."""
        pos0 = info['pos0']
        dpos = info['dpos']
        step = info['step']
        nevals = info['nevals_func']
        xflist = info['xflist']
        pos_min, func_min = info['pos_min'], info['func_min']

        pos = pos0 + step * dpos
        func = self.calc_obj_fun(pos)
        nevals += 1
        new_xf = _np.array([[step, func]])
        xflist = _np.concatenate((xflist, new_xf), axis=0)

        if func < func_min:
            func_min = func
            alpha_min = step
            pos_min = pos

        while func < (func_min + 3 * self.gnoise):
            step_backup = step
            if abs(step) < 0.1:
                step *= (1 + self.GOLDEN_RATIO)
            else:
                if direction == 'positive':
                    step += 0.1
                elif direction == 'negative':
                    step -= 0.1
                else:
                    raise Exception('invalid direction')

            pos = pos0 + step * dpos
            func = self.calc_obj_fun(pos)
            nevals += 1

            if _np.isnan(func):
                step = step_backup
                break
            else:
                new_xf = _np.array([[step, func]])
                xflist = _np.concatenate((xflist, new_xf), axis=0)
                if func < func_min:
                    func_min = func
                    alpha_min = step
                    pos_min = pos

            info['alpha_bound'] = step
            info['nevals_func'] = nevals
            info['xflist'] = xflist
            info['alpha_min'] = alpha_min
            info['pos_min'], info['func_min'] = pos_min, func_min
            return info
