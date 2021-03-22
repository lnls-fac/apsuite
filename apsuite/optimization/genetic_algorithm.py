import numpy as np


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
        ind_sort = np.argsort(f)
        return self._indiv[ind_sort[:self._nparents], :]

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
