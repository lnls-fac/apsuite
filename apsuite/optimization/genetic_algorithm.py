import numpy as np
import matplotlib.pyplot as mplt


class GA:
    """."""

    def __init__(self, npop, nparents, mutrate=0.01):
        """."""
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._indiv = np.array([])
        self.initialization()
        # Dimension of search space is obtained by boundary limits
        self._ndim = len(self._upper_limits)
        # Population size
        self._npop = npop
        # Number of parents to be selected
        self._nparents = nparents
        # Number of offspring generated from parents
        self._nchildren = self._npop - self._nparents
        # Mutation rate (Default = 1%)
        self._mutrate = mutrate

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return array with size equal to the population size."""
        raise NotImplementedError

    def _create_pop(self):
        """."""
        # Random initialization of elements inside the bounday limits
        dlim = self._upper_limits - self._lower_limits
        rarray = np.random.rand(self._npop, self._ndim)
        self._indiv = dlim * rarray + self._lower_limits

    def _select_parents(self, f):
        """."""
        # Select parents based on best ranked ones
        ind_sort = self.nondominated_sort(f)
        return self._indiv[ind_sort[:self._nparents], :]

    @staticmethod
    def nondominated_sort(res, max_rank=None):
        """Perform non-dominated sorting on results.

        I'm still in the search for an algorithm faster than O(MN^2).

        Args:
            res (numpy.ndarray, NxM): values of M objective functions for N
                particles.
            max_rank (int, optional): Maximum rank to sort. Particles with
                rank greater than `max_rank` will not be properly ranked.
                Defaults to `None`, which means all particles will be ranked.

        Returns:
            rank (numpy.ndarray, (N, )): rank of the particles. Pareto front
                has rank 1.

        """
        num_res = res.shape[0]
        rank = np.ones(num_res, dtype=int)
        max_rank = max_rank or num_res
        for i in range(num_res):
            # The 2 lines below are an alternative version of the algorithm:
            # dres = res - res[i]
            # rank[i] += np.sum(np.all(dres < 0, axis=1))

            if rank[i] > max_rank:
                continue
            dres = res[i+1:] - res[i]

            # First update the rank of the current particle finding all
            # other particles that dominate it:
            rank[i] += np.sum(np.all(dres < 0, axis=1))
            # Now update the rank of all particles dominated by this one:
            rank[i+1:][np.all(dres > 0, axis=1)] += 1
        return rank

    @staticmethod
    def plot_rank(res, rank, dirs=(0, 1), max_rank=None):
        """Make figure with particles ranking.

        Args:
            res (numpy.ndarray, (N, M)): values of M objective functions for N
                particles.
            rank (numpy.ndarray, (N, )): rank of the particles.
            dirs (2-tuple, optional): Objective directions to plot.
                Defaults to (0, 1).
            max_rank(int, optional): Maximum rank to plot lines connecting
                particles of the same rank. Defaults to `None`, which means all
                particles with same rank will be connected..

        Returns:
            fig (matplotlib.Figure): Figure created.
            ax (matplotlib.Axes): axes where data was plotted.

        """
        fig, ax = mplt.subplots(1, 1, figsize=(10, 9))
        max_rank = max_rank or rank.max()
        dir1, dir2, *_ = dirs
        for i in range(1, rank.max()+1):
            idx = rank == i
            if not np.any(idx):
                continue
            cor = mplt.cm.jet(i/rank.max())
            res1 = res[idx, dir1]
            res2 = res[idx, dir2]
            ax.plot(res1, res2, 'o', color=cor)
            if i > max_rank:
                continue
            sort = np.argsort(np.arctan2(res2, res1))
            ax.plot(res1[sort], res2[sort], 'o-', color=cor)
        fig.tight_layout()
        fig.show()
        return fig, ax

    def _crossover(self, parents):
        """."""
        child = np.zeros([self._nchildren, self._ndim])
        # Create list of random pairs to produce children
        par_rand = np.random.randint(0, self._nparents, [self._nchildren, 2])
        # Check if occurs that two parents are the same
        equal_par = par_rand[:, 0] == par_rand[:, 1]

        while equal_par.any():
            # While there is two parents that are the same, randomly choose
            # another first parent
            par_rand[equal_par, 0] = np.random.randint(
                0, self._nparents, np.sum(equal_par))
            equal_par = par_rand[:, 0] == par_rand[:, 1]

        for i in range(self._nchildren):
            for j in range(self._ndim):
                # For each child and for each gene, choose which gene will be
                # inherited from parent 1 or parent 2 (each parent has 50% of
                # chance)
                if np.random.rand(1) < 0.5:
                    child[i, j] = parents[par_rand[i, 0], j]
                else:
                    child[i, j] = parents[par_rand[i, 1], j]
        return child

    def _mutation(self, child):
        """."""
        for i in range(self._nchildren):
            # For each child, with MutRate of chance a mutation can occur
            if np.random.rand(1) < self._mutrate:
                # Choose the number of genes to perform mutation (min is 1 and
                # max is the maximum number of genes)
                num_mut = np.random.randint(1, self._ndim)
                # Choose which genes are going to be changed
                gen_mut = np.random.randint(0, self._ndim, num_mut)
                # Mutation occurs as a new random initialization
                dlim = (self._upper_limits - self._lower_limits)[gen_mut]
                rarray = np.random.rand(num_mut)
                change = dlim * rarray + self._lower_limits[gen_mut]
                child[i, gen_mut] = change
        return child

    def start_optimization(self, niter):
        """."""
        self._create_pop()

        for k in range(niter):
            print('Generation number ' + str(k+1))
            fout = self.calc_obj_fun()
            print('Best Figure of Merit: ' + str(np.min(fout)))
            print(
                'Best Configuration: ' + str(self._indiv[np.argmin(fout), :]))
            parents = self._select_parents(fout)
            children = self._crossover(parents)
            children_mut = self._mutation(children)
            self._indiv[:self._nparents, :] = parents
            self._indiv[self._nparents:, :] = children_mut
