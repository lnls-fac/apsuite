"""Robust Conjugated Direction Search Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event

import numpy as _np
import matplotlib.pyplot as plt


class RCDS:
    """The original algorithm was developed by X. Huang from SLAC."""

    # NOTE: objects with threading.Event cannot be serialized with pickle

    GOLDEN_RATIO = (1 + _np.sqrt(5))/2

    def __init__(self, save=False, use_thread=True):
        """."""
        self._use_thread = use_thread

        # search space
        self._position = _np.array([])
        self._pos_lim_lower = None
        self._pos_lim_upper = None
        self._pos_step = None

        self.gnoise = None
        self.gcount = None
        self.gdata = None
        self.grange = None
        self.imat = None
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
        info['func0'] = func0
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
            info['xflist'] = xflist
            info['alpha1'] = alpha1
            info['alpha2'] = alpha2
            return info

        info['step'] = -step_init
        info = self.search_bound(info, direction='negative')
        alpha1 = info['alpha_bound']
        alpha_min = info['alpha_min']
        xflist = info['xflist']

        # we found a second bound, just check the order
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
        dpos = info['dpos']
        step = info['step']
        nevals = info['nevals_func']
        xflist = info['xflist']
        alpha_min = info['alpha_min']
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

        # equal comparison is for the noiseless case
        while func <= (func_min + 3 * self.gnoise):
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

    def linescan(self, info):
        """."""
        pos0 = info['pos0']
        func0 = info['func0']
        dpos = info['dpos']
        nevals_func = info['nevals_func']
        xflist = info['xflist']
        alpha1, alpha2 = info['alpha1'], info['alpha2']

        if _np.isnan(func0):
            func0 = self.calc_obj_fun(pos0)
            nevals_func += 1

        if alpha1 >= alpha2:
            print('Upper bound <= Lower bound')
            return pos0, func0, nevals_func

        npts = 6
        alpha_step = (alpha2 - alpha1)/(npts - 1)
        alphas = _np.arange(alpha1, alpha2, alpha_step)

        nlist = _np.shape(xflist)[0]
        flist = alphas * _np.nan
        for idx in range(nlist):
            if xflist[idx, 1] >= alpha1 and xflist[idx, 1] <= alpha2:
                ik = int(round((xflist[idx, 1]-alpha1)/alpha_step))
                if ik == alphas.size:
                    ik -= 1
                alphas[ik] = xflist[idx, 0]
                flist[ik] = xflist[idx, 1]

        mask = [True] * alphas.size
        for idx, alpha in enumerate(alphas):
            if _np.isnan(flist[idx]):
                pos = pos0 + alpha * dpos
                flist[idx] = self.calc_obj_fun(pos)
                nevals_func += 1
                mask[idx] = False

        alphas = alphas[mask]
        flist = flist[mask]

        if alphas.size == 0:
            return pos0, func0, nevals_func
        elif alphas.size < npts - 1:
            idx_min = _np.argmin(flist)
            pos_min = pos0 + alphas[idx_min] * dpos
            func_min = flist[idx_min]
            return pos_min, func_min, nevals_func
        else:
            print('h3')
            poly = _np.poly1d(_np.polyfit(alphas, flist, deg=2))
            mpts = 101
            alpha_v = _np.linspace(alphas[0], alphas[-1], mpts-1)
            func_v = poly(alpha_v)
            plt.figure()
            plt.plot(alpha_v, func_v)
            plt.show()
            idx_min = _np.argmin(func_v)
            pos_min = pos0 + alpha_v[idx_min] * dpos
            func_min = func_v[idx_min]
            return pos_min, func_min, nevals_func

    def _optimize(self, print_flag=True):
        """Powell Direction Search."""
        pos0 = self._position
        func0 = self.calc_obj_fun(pos0)
        init_func = func0
        nevals_func = 1

        pos_min, func_min = pos0, func0
        dmat = self.dmat0
        step = self.step

        hist_best_pos, hist_best_func = [], []
        hist_best_pos.append(pos_min)
        hist_best_func.append(func_min)
        for iter in range(self.niter):
            print(func_min)
            if print_flag:
                print(f'iteration: {iter+1:04d}/{self.niter:04d}')
            step /= 1.20
            ik_ = 1
            dl_ = 0
            for idx in range(self.ndim):
                dpos = dmat[:, idx]
                info = self.bracketing_min(pos_min, func_min, dpos, step)

                stg = f'iter {iter+1:d}, dir {idx:d}: '
                stg += f"begin\t{self.gcount:d}\t{info['func_min']}"
                if print_flag:
                    print(stg)
                nevals_func += info['nevals_func']
                # minimum is the new starting point
                info['pos0'], info['func0'] = info['pos_min'], info['func_min']
                pos_idx, func_idx, nevals = self.linescan(info)
                nevals_func += nevals
                if (func_min - func_idx) > dl_:
                    dl_ = func_min - func_idx
                    ik_ = idx
                    if print_flag:
                        print(f'iteration {iter+1:d}, var {idx:d}: del = {dl_:f} updated\n')
                func_min = func_idx
                pos_min = pos_idx

            pos_e = 2*pos_min - pos0
            print('evaluating self.func_obj')
            func_e = self.calc_obj_fun(pos_e)
            print('done')
            nevals_func += 1

            fat1 = 2*(func0 - 2*func_min + func_e)
            fat2 = (func0 - func_min + dl_)
            fat = fat1 * fat2**2
            cond1 = (func0 <= func_e)
            cond2 = (fat >= dl_*(func_e - func0)**2)
            cond =  cond1 | cond2
            if cond:
                if print_flag:
                    print("   , dir %d not replaced: %d, %d\n" % (ik_,cond1, cond2 ))
            else:
                diff = pos_min - pos0
                new_dpos = diff/_np.linalg.norm(diff)
                pos_proj = _np.zeros([self.ndim])
                for idx in range(self.ndim):
                    pos_proj[idx] = abs(_np.dot(new_dpos.T, dmat[:, idx]))

                if _np.max(pos_proj) < 0.9:
                    if print_flag:
                        print(f'Replacing {ik_:d} direction')
                    for idx in range(ik_, self.ndim-1):
                        dmat[:, idx] = dmat[:, idx+1]
                    dmat[:, -1] = new_dpos
                    info = self.bracketing_min(
                        pos_min, func_min, new_dpos, step)
                    nevals_func += info['nevals_func']
                    info['pos0'] = info['pos_min']
                    info['func0'] = info['func_min']
                    print("iter %d, new dir %d: begin\t%d\t%f " %(iter+1,ik_, self.gcount,info['func_min']))
                    pos_idx, func_idx, nevals = self.linescan(info)
                    nevals_func += nevals
                    func_min = func_idx
                    pos_min = pos_idx
                else:
                    if print_flag:
                        print("    , skipped new direction %d, max dot product %f\n" %(ik_, _np.max(pos_proj)))

            print('g count is ', self.gcount, 'and maxEval is ', 1000)

            tol = self.tolerance
            cond = 2*abs(func0 - func_min) < tol*(abs(func0)+abs(func_min))
            # print(2*abs(func0 - func_min), tol*(abs(func0)+abs(func_min)))
            # if (cond and tol > 0) or func_min > func0:
            if (cond and tol > 0):
            # if abs(func_min) < tol:
                stg = 'Finished! \n'
                stg += f'func0 = {init_func:4.2e} \n'
                stg += f'func_min = {func_min:4.2e} \n'
                ratio = func_min/init_func
                stg += f'func_min/func0 = {ratio:4.2e} \n'
                if print_flag:
                    # print(stg)
                    print("terminated: f0=%4.2e\t, fm=%4.2e, f0-fm=%4.2e\n" %(func0, func_min, func0-func_min))
                break

            func0 = func_min
            pos0 = pos_min
            hist_best_pos.append(pos_min)
            hist_best_func.append(func_min)
            print('')

        self.hist_best_positions = hist_best_pos
        self.hist_best_objfunc = hist_best_func
        self.best_direction = dmat
