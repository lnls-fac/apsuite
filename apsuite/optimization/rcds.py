"""Robust Conjugate Direction Search Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event


import numpy as _np
import matplotlib.pyplot as plt


class RCDS:
    """The original algorithm was developed by X. Huang from SLAC."""

    # NOTE: objects with threading.Event cannot be serialized with pickle

    GOLDEN_RATIO = (1 + _np.sqrt(5))/2
    TINY = 1e-25

    def __init__(self, save=False, use_thread=True):
        """."""
        self._use_thread = use_thread

        # search space
        self._position = _np.array([])
        self._pos_lim_lower = None
        self._pos_lim_upper = None
        self._pos_step = None
        self.gnoise = None
        self.gcount = None # not used?
        # search control

        self._niter = 0
        self._flag_save = save
        self._tolerance = 0
        self.dmat0 = None

        if self._use_thread:
            self._thread = _Thread(target=self._optimize, daemon=True)
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
    def tolerance(self):
        """."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        """."""
        self._tolerance = value

    @property
    def limits_upper(self):
        """."""
        return self._pos_lim_upper

    @limits_upper.setter
    def limits_upper(self, value):
        """."""
        #if len(value) != len(self._position): # move these checkings to another place
        #    raise Exception('Incompatible upper limit!') # they impose an order to parameter setting
        self._pos_lim_upper = _np.array(value)

    @property
    def limits_lower(self):
        """."""
        return self._pos_lim_lower

    @limits_lower.setter
    def limits_lower(self, value):
        """."""
        #if len(value) != len(self._position):
        #    raise Exception('Incompatible lower limit!')
        self._pos_lim_lower = _np.array(value)

    @property
    def step(self):
        """."""
        return self._pos_step

    @step.setter
    def step(self, value):
        """."""
        self._pos_step = value

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

    def _check_initialization(self):
        """."""
        if self._pos_lim_upper is None and self._pos_lim_lower is None:
            return

        if len(self._pos_lim_upper) != len(self._pos_lim_lower):
            raise Exception(
                'Upper and Lower Limits have different lengths')

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
    
    def print_initial_settings(self):
        print(f'Parameter space dimension: {self.ndim}')
        print(f'Initial point: {self.position}')
        print(f'Initial step: {self.step}')
        print(f'Tolerance {self.tolerance}')
        print(f'Parameter space span:\n{_np.concatenate((self.limits_lower[:, None],self.limits_upper[:, None]),axis=1)}')
        print(f'Max number of iterations: {self.niter}')
        print(f'Objective function noise-sigma: {self.gnoise}')
        print(f'Initial directions matrix:\n{self.dmat0}')

    def bracketing_min(self, pos0, func0, dv, step):
        """Brackets the minimum

        Args:
            pos0 (n-dimensional np.array): starting point in param space
            func0 (float): objective function at pos0
            dv (n-dimensional np.array): direction vector
            step (float): initial step

        Returns:
            info (dict): dictionary containing the input args as well
            as the number of obj. func evaluations, 'nr_func_evals', a list of such evaluations
            and steps 'xflist', the step for which the minimum is attained 'alpha_min',
            the minimum point 'pos_min' and the obj. func minimum value, 'func_min'.
        """
        nr_func_evals = 0
        if _np.isnan(func0):
            func0 = self.calc_obj_fun(pos0)
            nr_func_evals += 1

        xflist = _np.array([[0, func0]])
        func_min = func0
        alpha_min = 0 
        pos_min = pos0

        step_init = step

        info = dict()
        info['pos0'] = pos0
        info['func0'] = func0
        info['dv'] = dv
        info['step'] = step
        info['nr_func_evals'] = nr_func_evals
        info['xflist'] = xflist
        info['alpha_min'] = alpha_min
        info['pos_min'] = pos_min
        info['func_min'] = func_min

        info = self.search_bound(info, direction='positive') #searches for upper bracket bound, minimum along dv, and constructs xflist: list of steps and function evaluations

        alpha2 = info['alpha_bound'] #setting the upper bound
        pos_min, func_min = info['pos_min'], info['func_min'] # setting the position and value of the minimum
        alpha_min = info['alpha_min'] # setting the step at which the minimum is attained
        xflist = info['xflist'] # steps and function evaluations
        nr_func_evals = info['nr_func_evals'] # number of objective function evaluations
        
        if func0 > (func_min + 3 * self.gnoise): # if true, no need to search bound in the negative direction
            alpha1 = -alpha_min
            alpha2 -= alpha_min
            xflist[:, 0] -= alpha_min  # subtract alpha_min from all alphas to set the minimum at 0
            info['xflist'] = xflist
            info['alpha1'] = alpha1
            info['alpha2'] = alpha2
            # at this point, the bounds are at positions
            # - first bound:  pos1 = pos_min + (-alpha_min)*dv
            # - second bound: pos2 = pos_min + (alpha2 - alpha_min)*dv
            # and the minimum is at 0 of xflist[:,0]
            return info

        # searching for lower bound in the negative direction
        info['step'] = -step_init
        info = self.search_bound(info, direction='negative')
        alpha1 = info['alpha_bound']
        alpha_min = info['alpha_min']
        xflist = info['xflist']

        # upper and lower bounds found. check the order
        if alpha1 > alpha2:
            alpha1, alpha2 = alpha2, alpha1

        # subtract alpha_min since pos_min is the new reference
        alpha1 -= alpha_min
        alpha2 -= alpha_min
        xflist[:, 0] -= alpha_min
        xflist = xflist[_np.argsort(xflist[:, 0])]

        info['xflist'] = xflist
        info['alpha1'] = alpha1
        info['alpha2'] = alpha2
        return info

    def search_bound(self, info, direction='positive'):
        """."""
        pos0 = info['pos0']
        dv = info['dv']
        step = info['step']
        nevals = info['nr_func_evals']
        xflist = info['xflist']
        alpha_min = info['alpha_min']
        pos_min, func_min = info['pos_min'], info['func_min']
        
        pos = pos0 + step * dv
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
                    raise Exception('Invalid Direction')

            pos = pos0 + step * dv
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
        info['nr_func_evals'] = nevals
        info['xflist'] = xflist
        info['alpha_min'] = alpha_min
        info['pos_min'], info['func_min'] = pos_min, func_min
        return info

    def linescan(self, info):
        """."""
        pos0 = info['pos0']
        func0 = info['func0']
        dv = info['dv']
        nr_func_evals = info['nr_func_evals']
        xflist = info['xflist']
        alpha1, alpha2 = info['alpha1'], info['alpha2']

        if _np.isnan(func0):
            func0 = self.calc_obj_fun(pos0)
            nr_func_evals += 1

        if alpha1 >= alpha2:
            print('Upper bound <= Lower bound')
            return pos0, func0, nr_func_evals

        npts = 6
        alphas_step = (alpha2 - alpha1)/(npts - 1)
        alphas = _np.arange(alpha1, alpha2, alphas_step)
        flist = alphas * _np.nan
        
        for idx, alpha in enumerate(xflist[:,0]):
            if alpha >= alpha1 and alpha <= alpha2:
                ik = int((alpha - alpha1) / alphas_step)
                if ik == alphas.size:
                    ik = -1
                alphas[ik] = alpha  
                flist[ik] = xflist[ik,1] 

        mask = [True] * alphas.size
        for idx, alpha in enumerate(alphas):
            if _np.isnan(flist[idx]):
                pos = pos0 + alpha * dv
                flist[idx] = self.calc_obj_fun(pos)
                nr_func_evals += 1
            if _np.isnan(flist[idx]):
                mask[idx] = False

        alphas = alphas[mask]
        flist = flist[mask]

        if alphas.size == 0:
            return pos0, func0, nr_func_evals
        elif alphas.size < npts - 1:
            idx_min = flist.argmin()
            pos_min = pos0 + alphas[idx_min] * dv
            func_min = flist[idx_min]
            return pos_min, func_min, nr_func_evals
        else:
            poly = _np.poly1d(_np.polyfit(alphas, flist, deg=2))
            alpha_v = _np.linspace(alphas[0], alphas[-1], 100)
            func_v = poly(alpha_v)
            #plt.figure() #edit this later
            #plt.plot(alpha_v, func_v)
            #plt.show()
            idx_min = _np.argmin(func_v)
            pos_min = pos0 + alpha_v[idx_min] * dv
            func_min = func_v[idx_min]
            return pos_min, func_min, nr_func_evals

    def _optimize(self, print_flag=True):
        """Powell Direction Search Algorithm with Xiaobiao's bracketing and Linescan"""
        pos0 = self._position
        func0 = self.calc_obj_fun(pos0)
        init_func = func0
        nr_func_evals = 1

        pos_min, func_min = pos0, func0
        dmat = self.dmat0
        step = self.step

        hist_best_pos, hist_best_func = [pos_min], _np.array([func_min])

        for iter in range(self.niter):
            if print_flag:
                print(f'Iteration {iter+1:04d}/{self.niter:04d}\n')
            step /= 1.20 # where does this come from? Not in numerical recipes, check Powell
            dl_ = 0 # largest objective function decrease
            ik_ = 0 # direction index at which the largest decrease happens

            for idx in range(self.ndim): # for each direction in param space
                dv = dmat[:, idx] # choose the corresponding direction vector
                info = self.bracketing_min(pos_min, func_min, dv, step) # bracket the minimum along it

                stg = f'Direction {idx+1:d}. '
                stg += f"Obj. Func. Min: {info['func_min']}"
                if print_flag:
                    print(stg)
                nr_func_evals += info['nr_func_evals']
                info['pos0'], info['func0'] = info['pos_min'], info['func_min'] # minimum found in bracketing is the new starting point
                pos_idx, func_idx, nevals = self.linescan(info) # minimum from parabola within brackets
                nr_func_evals += nevals
                if (func_min - func_idx) > dl_: # if current decrease is the largest decrease
                    dl_ = func_min - func_idx # update largest decrease
                    ik_ = idx # save largest decrease direction index
                    if print_flag:
                        print(f'Largest obj. func delta = {dl_:f}, updated.')
                func_min = func_idx # update obj func min
                pos_min = pos_idx # and position

            pos_e = 2 * pos_min - pos0 # extension point for direction replacement conditions
            print('Evaluating objective function at extension point...')
            func_e = self.calc_obj_fun(pos_e)
            print('Done!\n')
            nr_func_evals += 1

            cond1 = (func0 <= func_e) # Numerical Recipes conditions (same order)
            cond2 = 2 * (func0 - 2 * func_min + func_e) * (func0 - func_min - dl_)**2 >= dl_ * (func_e - func0)**2 
            if cond1 or cond2: 
                if print_flag:
                    print(f'Direction {ik_+1:d} not replaced: Condition 1: {cond1}; Condition 2: {cond2}')
            else:
                diff = pos_min - pos0
                new_dv = diff/_np.linalg.norm(diff) # new conjugate direction
                pos_proj = (new_dv.T * dmat).sum(axis=0) # used for checking orthogonality with other directions
                 
                max_dotp = _np.max(pos_proj)
                if max_dotp < 0.9:
                    if print_flag:
                        print(f'Replacing direction {ik_+1:d}')
                    for idx in range(ik_, self.ndim-1):
                        dmat[:, idx] = dmat[:, idx+1]
                    dmat[:, -1] = new_dv
                    info = self.bracketing_min(pos_min, func_min, new_dv, step)
                    nr_func_evals += info['nr_func_evals']
                    info['pos0'] = info['pos_min']
                    info['func0'] = info['func_min']
                    print(f'Iteration {iter+1:d}, New dir. {ik_+1:d}, Obj. Func. Min {func_min:f}')
                    pos_idx, func_idx, nevals = self.linescan(info)
                    nr_func_evals += nevals
                    func_min = func_idx
                    pos_min = pos_idx
                else:
                    if print_flag:
                        print(f'Direction replacement conditions met.\n')
                        print(f'Skipping new direction {ik_+1:d}: max dot product {max_dotp:f}')

            tol = self.tolerance
            cond = 2 * (func0 - func_min) <= tol * (abs(func0) + abs(func_min))
            # cond = 2 * (func0 - func_min) <= tol * (abs(func0) + abs(func_min)) + self.TINY #Numerical recipes
            if (cond and tol > 0):
            #if abs(func_min) < tol:
                stg = 'Finished! \n'
                stg += f'f_0 = {init_func:4.2e}\n'
                stg += f'f_min = {func_min:4.2e}\n'
                ratio = func_min/init_func
                stg += f'f_min/f0 = {ratio:4.2e}\n'
                if print_flag:
                     print(stg)
                break

            func0 = func_min
            pos0 = pos_min
            hist_best_pos = _np.vstack((hist_best_pos, pos_min))
            hist_best_func = _np.vstack((hist_best_func, func_min))
            print('')

        self.hist_best_positions = hist_best_pos
        self.hist_best_objfunc = hist_best_func
        self.best_direction = dmat
