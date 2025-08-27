"""Script for implementation of the Maximum Entropy (MENT) algorithm.

The algorithm reconstructs 2D-distribution given a set of 1D-projections, using
the method of constrained optimization with Lagrange multipliers.

[1] G. Minerbo, “MENT: A maximum entropy algorithm for reconstructing a source
from projection data”, Computer Graphics and Image Processing, vol. 10, no. 1,
pp. 48-68, May 1979. doi:10.1016/0146-664X(79)90034-0.

[2] O. Silveira and F. de Sá, "MENT algorithm for transverse phase space
reconstruction at SIRIUS", in Proc. IPAC'25, Taipei, Taiwan, Jun. 2025,
pp. 2708-2711. doi:10.18429/JACoW-IPAC25-THPM035.
"""


from copy import deepcopy as _deepcopy

import matplotlib.pyplot as _plt
import numpy as _np
from mathphys.imgproc import Image2D_ROI as _Image2D_ROI
from scipy.interpolate import interp1d as _interp1d
from scipy.spatial import ConvexHull as _ConvexHull, \
    HalfspaceIntersection as _HalfspaceIntersection

from ...utils import DataBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBase


def plot_lines(lines, fig=None, ax=None, **kwargs):
    """Plots a series of lines defined by their coefficients.

    The coefficients are in the form ax + by + c = 0.
    """
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    for a, b, c in lines:
        if _np.isclose(b, 0):
            ax.axvline(-c / a, **kwargs)
        else:
            ax.axline((0, -c / b), slope=-a / b, **kwargs)

    return fig, ax


def plot_convex(
    hull: _ConvexHull,
    fig=None,
    ax=None,
    fill=False,
    fill_kwargs=None,
    **kwargs
):
    """Plots a convex hull."""
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    pts = hull.points[hull.vertices]
    pts = _np.append(pts, [pts[0]], axis=0)

    lin = ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    if fill:
        fill_kwargs = fill_kwargs or dict()
        fill_kwargs['alpha'] = fill_kwargs.get('alpha', 0.5)
        fill_kwargs['color'] = fill_kwargs.get('color', lin.get_color())
        ax.fill(pts[:, 0], pts[:, 1], **fill_kwargs)

    return fig, ax


def paint_convex(hull: _ConvexHull, fig=None, ax=None, **kwargs):
    """Paints the interior of a convex hull."""
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    points = hull.points[hull.vertices]
    ax.fill(points[:, 0], points[:, 1], **kwargs)

    return fig, ax


class ScreenProcess:
    """Class used for processing and analysing screen images.

    This class is used to scale, crop, and get projections of screen images,
    as well as for handling related grid and bin operations.
    """

    def __init__(self, scrn_img, scale_x, scale_y):
        """."""
        self.image = scrn_img
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.shape = scrn_img.shape

        self._roix = None
        self._roiy = None

    @property
    def positions(self):
        """Returns scaled positions of images."""
        sy, sx = self.shape
        x = _np.arange(sx) * self.scale_x
        y = _np.arange(sy) * self.scale_y
        x -= x[sx // 2]
        y -= y[sy // 2]
        return x, y

    @property
    def grids(self):
        """."""
        x, y = self.positions
        return _np.meshgrid(x, y)

    @property
    def roix(self):
        """."""
        if self._roix is None:
            sent = "Image not cropped. Call 'crop_image' first."
            raise ValueError(sent)
        return self._roix

    @property
    def roiy(self):
        """."""
        if self._roiy is None:
            sent = "Image not cropped. Call 'crop_image' first."
            raise ValueError(sent)
        return self._roiy

    def crop_image(self, fwhmx_factor=4, fwhmy_factor=4):
        """Crops image based on fwhm and returns new grids and new image."""
        img2droi = _Image2D_ROI(self.image)
        img2droi.update_roi_with_fwhm(fwhmx_factor, fwhmy_factor)
        self._roix = img2droi.roix
        self._roiy = img2droi.roiy

        img2droi_crop = img2droi.create_trimmed()
        image_crop = img2droi_crop.data
        x, y = self.positions
        new_x = x[slice(*self.roix)]
        new_y = y[slice(*self.roiy)]
        new_gridx, new_gridy = _np.meshgrid(new_x, new_y)
        return new_gridx, new_gridy, image_crop

    def get_projections(self, nr_bins=(50, 50), bin_size=None):
        """Computes the X and Y projections of the image.

        This method computes the sum of pixel intensities along the X and Y
        axes within the defined region of interest for the cropped image. Then,
        it interpolates the projection data into a new array of adjustable
        size. The number of bins for this new array can be specified either by
        the given number of bins or by a fixed bin size.

        Args:
            nr_bins (tuple of int, optional): Number of bins for X and Y axes.
                Default is (50, 50).
            bin_size (tuple of float, optional): Fixed bin size for X and Y
                axes. If not provided, the number of bins (`nr_bins`) will be
                used instead to define the bin sizes.

        Returns:
            tuple: A tuple containing:
                - (projx, projy) (tuple of ndarray): Projections along the X
                and Y axes.
                - (bins_x, bins_y) (tuple of ndarray): Bin edges for the X
                and Y axes.
                - (bin_size_x, bin_size_y) (tuple of float): The calculated bin
                sizes for X and Y axes.
        """
        img = self.image[slice(*self.roiy), slice(*self.roix)]
        projx = _np.sum(img, axis=0)
        projy = _np.sum(img, axis=1)
        x, y = self.positions
        x = x[slice(*self.roix)]
        y = y[slice(*self.roiy)]

        # Interpolates projections
        fx = _interp1d(x, projx)
        fy = _interp1d(y, projy)
        xmin, xmax = x[0], x[-1]
        ymin, ymax = y[0], y[-1]

        if bin_size is None:
            new_x = _np.linspace(xmin, xmax, nr_bins[0])
            new_y = _np.linspace(ymin, ymax, nr_bins[1])
        else:
            bin_size_x, bin_size_y = bin_size
            lengthx = _np.abs(xmax - xmin)
            lengthy = _np.abs(ymax - ymin)
            nr_points_x, excess_x = divmod(lengthx, bin_size_x)
            nr_points_y, excess_y = divmod(lengthy, bin_size_y)
            new_x = self._adjust_limits(xmin, xmax, excess_x, int(nr_points_x))
            new_y = self._adjust_limits(ymin, ymax, excess_y, int(nr_points_y))

        projx = fx(new_x)
        projy = fy(new_y)

        # Create bins
        bins_x = self.position_to_bin(new_x)
        bins_y = self.position_to_bin(new_y)
        bin_size_x = _np.abs(bins_x[1] - bins_x[0])
        bin_size_y = _np.abs(bins_y[1] - bins_y[0])

        return (projx, projy), (bins_x, bins_y), (bin_size_x, bin_size_y)

    @staticmethod
    def plot_image(gridx, gridy, image, fig=None, ax=None, **kwargs):
        """."""
        if fig is None or ax is None:
            fig, ax = _plt.subplots(1, 1)

        ax.pcolormesh(gridx, gridy, image, **kwargs)

        return fig, ax

    @staticmethod
    def position_to_bin(x):
        """Converts a 1D array of positions into bin edges."""
        nr_points = len(x)
        delta = x[1] - x[0]
        xmin = x[0] - delta / 2
        xmax = x[-1] + delta / 2
        return _np.linspace(xmin, xmax, nr_points + 1)

    @staticmethod
    def bin_to_position(x):
        """Converts bin edges into bin center positions."""
        return (x[:-1] + x[1:]) / 2

    def _adjust_limits(self, min_val, max_val, excess, nr_points):
        if min_val < max_val:
            min_val += excess / 2
            max_val -= excess / 2
        else:
            min_val -= excess / 2
            max_val += excess / 2
        return _np.linspace(min_val, max_val, nr_points + 1)


class Params(_ParamsBase):
    """."""

    def __init__(self):
        """."""
        self.nrpts_per_bin = 3
        self.lagmult_fancy_init = False
        self.loop_num_iterations = 100
        self.min_lagmult_change = 1e-5
        self.gain_per_iteration = 1.0
        self.max_lagmult_step = 1.0
        self.min_relative_chi2 = 1e-5
        self.sing_vals_cutoff_ratio = 1e-2

    def __str__(self):
        """."""
        stg = ''
        stg += f'nrpts_per_bin = {self.nrpts_per_bin:d}\n'
        stg += f'lagmult_fancy_init = {str(self.lagmult_fancy_init):s}\n'
        stg += f'loop_num_iterations = {self.loop_num_iterations:d}\n'
        stg += f'min_lagmult_change = {self.min_lagmult_change:.1g}\n'
        stg += f'gain_per_iteration = {self.gain_per_iteration:.2f}\n'
        stg += f'max_lagmult_step = {self.max_lagmult_step:.2f}\n'
        stg += f'min_relative_chi2 = {self.min_relative_chi2:.1g}\n'
        stg += f'sing_vals_cutoff_ratio = {self.sing_vals_cutoff_ratio:.1g}'
        return stg


class DistribReconstruction(_BaseClass):
    """Class for reconstructing 2D-distributions based on a regular grid.

    Each projection is represented by a set of bins (proj_bins) and has a
    matrix associated. For each projection bin, a set of paralel lines that
    separates each bin is generated using the line equation
    ax + by + c = 0. The coefficients 'a', 'b', and 'c' are derived from
    the projection matrix and bin values, where each line is stored as
    [a, b, c].

    Args:
        matrices (list of np.ndarray): A list of 2D arrays where each array
            corresponds to a matrix. Each matrix represents the
            tranformation from a reference point to a point of interest.
        projs (list of np.ndarray): a list of 1D arrays where each array
            contains the projection.
        projs_bins (list of np.ndarray): A list of 1D arrays where each array
            contains the bin edges for the respective projection.
    """

    def __init__(self, matrices=None, projs=None, projs_bins=None):
        """."""
        super().__init__()
        self.params = Params()
        if matrices is None or projs is None or projs_bins is None:
            return

        lines = []
        for i, proj_bins in enumerate(projs_bins):
            matrix = matrices[i]
            nr_bins = len(proj_bins)
            ab_line = _np.tile(matrix[0], (nr_bins, 1))
            lines.append(_np.vstack([ab_line.T, -proj_bins]).T)

        self.data['matrices'] = matrices
        self.data['projs_bins'] = projs_bins
        self.data['lines'] = lines
        self.data['main_convex'] = self._find_main_convex()
        self.data['projs'] = projs
        self.populate_main_convex()

    def __str__(self):
        """."""
        stg = 'Params:\n    '
        stg += '\n    '.join(str(self.params).splitlines())
        stg += '\n\n'
        stg += f'number of projections: {self.nr_projs:d}\n'
        stg += f'number of bins per projection: {self.nr_bins}\n'
        stg += f'gridx # points = {self.model_gridx.shape[1]:d}\n'
        stg += f'gridy # points = {self.model_gridy.shape[0]:d}\n'
        stg += f'total # points = {self.model_gridy.size:d}\n'
        return stg

    @property
    def main_convex(self):
        """Returns the main convex hull."""
        return self.data['main_convex']

    @property
    def lines(self):
        """."""
        return self.data['lines']

    @property
    def nr_projs(self):
        """."""
        return len(self.lines)

    @property
    def nr_bins(self):
        """Return a list with the number of bins in each projection."""
        return [prl.shape[0]-1 for prl in self.lines]

    @property
    def border_lines(self):
        """Calculates the border lines of main convex hull."""
        border_lines = []
        lines = self.lines
        for j in range(self.nr_projs):
            border_lines.append(lines[j][0, :])
            border_lines.append(lines[j][-1, :])
        return _np.array(border_lines)

    @property
    def raveled_lines(self):
        """Unpack the projections bin lines in an array."""
        return _np.vstack(self.lines)

    @property
    def matrices(self):
        """."""
        return self.data['matrices']

    @property
    def projs_meas(self):
        """."""
        return self.data['projs']

    @property
    def projs_bins(self):
        """."""
        return self.data['projs_bins']

    @property
    def init_lagmults(self):
        """Define a initial list for Lagrange multipliers.

        If the projection bin values are non-zero, the Lagrange multiplier is
        set to one. Otherwise, the Lagrange multiplier is set to zero.
        """
        if 'init_lagmults' in self.data:
            return _deepcopy(self.data['init_lagmults'])
        return self.get_lagmults_guess()

    @init_lagmults.setter
    def init_lagmults(self, value):
        """."""
        projs = self.data['projs']
        if len(value) != len(projs):
            raise ValueError(
                'Length of value must match length of projections.'
            )
        for i, (lamb, prj) in enumerate(zip(value, projs)):
            if len(lamb) != len(prj):
                raise ValueError(
                    f'Length of {i}th lambda must match length of {i}th proj.'
                )
        self.data['init_lagmults'] = _deepcopy(value)

    @property
    def lagmults(self):
        """."""
        if 'lagmults' not in self.data:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'reconstruct_distribution' first."
            raise ValueError(sent)
        return self.data['lagmults']

    @property
    def convergence_info(self):
        """."""
        if 'convergence_info' not in self.data:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'reconstruct_distribution' first."
            raise ValueError(sent)
        return self.data['convergence_info']

    @property
    def model_gridx(self):
        """."""
        return self.data['gridx']

    @property
    def model_gridy(self):
        """."""
        return self.data['gridy']

    @property
    def model_points(self):
        """."""
        return _np.array([self.model_gridx.ravel(), self.model_gridy.ravel()])

    @property
    def model_indices(self):
        """."""
        return self.data['indices']

    @property
    def model_grid_area(self):
        """."""
        gdx, gdy = self.model_gridx, self.model_gridy
        return (gdy[1, 0] - gdy[0, 0]) * (gdx[0, 1] - gdx[0, 0])

    def get_interior_grid_points_selection(self):
        """."""
        return self.model_indices[0] != -1

    def get_non_trivial_grid_points_selection(self):
        """Selects convexes whose initial Lagrange multupliers are finite."""
        sel = _np.ones(self.model_gridx.size, dtype=bool)
        if 'init_lagmults' not in self.data:
            return sel

        init_lag = self.init_lagmults
        for j, idx in enumerate(self.model_indices):
            zer = _np.isinf(init_lag[j]).nonzero()[0]
            for m in zer:
                sel &= idx != m
        return sel

    def get_model_projections(self, selection=None, distrib=None):
        """."""
        projs_calc = []
        nr_bins = self.nr_bins
        area = self.model_grid_area
        if distrib is None:
            distrib = self.get_model_distribution(selection=selection)
        for j, idcs in enumerate(self.model_indices):
            p = []
            for m in range(nr_bins[j]):
                idx = idcs == m
                p.append(distrib[idx].sum() * area)
            projs_calc.append(_np.array(p))
        return projs_calc

    def get_lagmults_guess(self):
        """."""
        fancy = self.params.lagmult_fancy_init
        lagmults = []
        for proj in self.projs_meas:
            lambda_proj = _np.zeros_like(proj, dtype=float)
            idx = (proj != 0)
            lambda_proj[~idx] = -_np.inf
            if fancy:
                lambda_proj[idx] = _np.log(proj[idx])

            lagmults.append(lambda_proj)
        return lagmults

    def get_model_distribution(
        self, raveled=True, selection=None, normalize=True
    ):
        """."""
        if 'lagmults' not in self.data:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'recontruct_distribution' first."
            raise ValueError(sent)

        lagmults = self.lagmults
        indices = self.model_indices
        sel = indices[0] != -1
        if selection is not None:
            sel &= selection

        lambs = _np.zeros(sel.sum(), dtype=float)
        for i, idcs in enumerate(indices):
            idcs = idcs[sel]
            lambs += lagmults[i][idcs]
        distrib = _np.zeros(sel.size, dtype=float)
        distrib[sel] = _np.exp(lambs)

        if normalize:
            distrib /= distrib.sum() * self.model_grid_area

        if not raveled:
            return distrib.reshape(self.model_gridx.shape)
        return distrib

    def populate_main_convex(self):
        """Populate the main convex hull with grid points."""
        nrpts = self.params.nrpts_per_bin
        pts = self.main_convex.points
        linx = self._suggest_grid(pts, plane=0, npts=nrpts)
        liny = self._suggest_grid(pts, plane=1, npts=nrpts)

        gridx, gridy = _np.meshgrid(linx, liny, copy=False)
        pts = _np.ones((3, gridx.size), dtype=float)
        pts[0] = gridx.ravel()
        pts[1] = gridy.ravel()

        sel = _np.ones(pts.shape[1], dtype=bool)
        idcs = []
        for lins in self.lines:
            res = lins @ pts
            boo = res * _np.sign(res[0, :]) > 0
            idcs.append(boo.sum(axis=0) - 1)
            sel &= ~boo.all(axis=0)
        idcs = _np.array(idcs)
        idcs[:, ~sel] = -1

        self.data['gridx'] = gridx
        self.data['gridy'] = gridy
        self.data['indices'] = idcs

    def reconstruct_distribution(self):
        """Find values for Lagrange multipliers using Newton's method.

        This algorithm uses the Newton method to find the best Lagrange
        multipliers that explain all projections, using the MENT framework.
        The chi**2 is the sum of the squares of the difference between the
        projections precited by the model and the measured/input/goal
        projections provided at the object initialization.

        At the end of the process, the property lagmult will be updated with
        the last value of the Lagrange multipliers found by this iterative
        process and the related methods of this object will provide the
        predicted distributions and projections.

        This method will make use of the following parameters defined in the
        params attribute:
            loop_num_iterations (int, optional): Maximum number of iterations
                of Newton's iterative method. Defaults to 100.
            min_lagmult_change (float, optional): If the maximum relative
                change of the Lagrange multipliers is lower than this value,
                the iterations are interrupted. Defaults to 1e-5.
            gain_per_iteration (float, optional): Learning rate for the
                Newton's method. Only this fraction of the variations of the
                Lagrange multipliers in each step will be applied.
                Defaults to 1.0.
            max_lagmult_step (float, optional): If even after multiplying the
                variation of the Lagrange multipliers by the gain, the maximum
                variation still exceeds this value, the change vector will be
                re-scaled such that its maximum is equal to this value. This
                is useful for the first iterations of the algorithm, when
                generally the search vector is far from the minimum.
                Defaults to 1.0.
            min_relative_chi2 (float, optional): If the relative chi**2 is
                lower than this value, the iterations are interrupted.
                Defaults to 1e-4.
            sing_vals_cutoff_ratio (float, optional): Condition value used for
                singular value selection of the Jacobian at each step. Please,
                check documentation of numpy.linalg.lstsq for `rcond` for more
                details. Defaults to 1e-3.
            lagmult_fancy_init (bool, optional): If True, the initial guess
                for the Lagrange multipliers will be based on the projections
                values. Otherwise, they will be initialized to one.
                Defaults to False.
        """
        nr_iter = self.params.loop_num_iterations
        tol = self.params.min_lagmult_change
        gain = self.params.gain_per_iteration
        max_step = self.params.max_lagmult_step
        min_rchi2 = self.params.min_relative_chi2
        rcond = self.params.sing_vals_cutoff_ratio

        lagmults = self.get_lagmults_guess()
        self.init_lagmults = lagmults
        self.data['lagmults'] = lagmults

        sel = self.get_non_trivial_grid_points_selection()
        pj_g = _np.hstack(self.projs_meas)
        max_chi2 = _np.sum(pj_g*pj_g)
        nr_bins = self.nr_bins
        tot_bins = _np.r_[0, _np.cumsum(nr_bins)]

        distrib = self.get_model_distribution(selection=sel, normalize=False)
        pjs = _np.hstack(self.get_model_projections(
            selection=sel, distrib=distrib
        ))
        res = pj_g - pjs

        convergence = []
        for itr in range(nr_iter):
            rmat = self._calc_respmat(sel, distrib, tot_bins)
            dlg, _, _, svs = _np.linalg.lstsq(rmat, res, rcond=rcond)
            svs /= svs[0]

            dlg *= gain
            dlg *= max_step / max(max_step, _np.abs(dlg).max())
            lamb = _np.hstack(lagmults) + dlg
            lagmults = _np.split(lamb, tot_bins[1:-1])
            self.data['lagmults'] = lagmults

            distrib = self.get_model_distribution(
                selection=sel, normalize=False
            )
            pjs = _np.hstack(self.get_model_projections(
                selection=sel, distrib=distrib
            ))
            res = pj_g - pjs
            rchi2 = _np.sum(res * res) / max_chi2

            nonzer = _np.abs(lamb) > 0
            max_rchg = _np.abs(dlg[nonzer] / lamb[nonzer]).max()
            converged = max_rchg < tol or rchi2 < min_rchi2
            if not itr % 1 or converged:
                nsv = _np.sum(svs > rcond)
                convergence.append([itr, max_rchg, nsv, rchi2])
                print(
                    f"{itr+1:04d}/{nr_iter:04d} -> "
                    f'max. rel. change={max_rchg:8.2e}, '
                    f"nr_svs={nsv:04d}/{svs.size:04d}, "
                    f'rel. chi_square={rchi2:8.2e}'
                )
            if converged:
                break

        self.data['lagmults'] = lagmults
        self.data['convergence_info'] = convergence
        print('Finished!')

    # ---------------------- methods for visualization ----------------------

    def plot_distribution(
        self,
        fig=None,
        ax=None,
        cmap='jet',
        plot_lines=False,
        lines_kwargs=None
    ):
        """."""
        if ax is None:
            fig, ax = _plt.subplots(1, 1)
        gridx = self.model_gridx
        gridy = self.model_gridy
        distrib = self.get_model_distribution()
        distrib = distrib.reshape(gridx.shape)

        if plot_lines:
            lines_kwargs = lines_kwargs or dict(color='k', lw=0.1, alpha=0.5)
            fig, ax = plot_lines(
                self.raveled_lines, fig=fig, ax=ax, **lines_kwargs
            )

        ax.pcolormesh(gridx, gridy, distrib, cmap=cmap)
        ax.set_title("Recontruction from Projections")
        return fig, ax

    def plot_lagrange_multipliers(self, fig=None, ax=None, plot_init=True):
        """."""
        if ax is None:
            fig, ax = _plt.subplots(1, 1)

        init_lagmults = self.init_lagmults
        lagmults = self.lagmults
        for j in range(self.nr_projs):
            ax.plot(lagmults[j], "-o", label=f"proj. {j}")
            if plot_init:
                ax.plot(init_lagmults[j], "--", color=f'C{j:d}')

        ax.set_title("Lagrange Multipliers")
        ax.grid(ls="--", alpha=0.5)
        ax.legend()
        return fig, ax

    def plot_projections(self, ncols=3, figsize=None):
        """."""
        projs_calc = self.get_model_projections()
        bins = self.projs_bins
        projs_meas = self.projs_meas

        nr_projs = self.nr_projs

        nrows = nr_projs // ncols + int(bool(nr_projs % ncols))

        figsize = figsize or (2.5*ncols, 2*nrows)
        fig, axs = _plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axs_r = _np.array(axs).ravel()

        for i, ax in enumerate(axs_r):
            if i >= nr_projs:
                ax.set_visible(False)
                continue
            x_plot = ScreenProcess.bin_to_position(bins[i])
            ax.plot(x_plot, projs_meas[i], "-o", label="meas.")
            ax.plot(x_plot, projs_calc[i], "-o", label="calc.")
            if not i:
                ax.legend()
            ax.set_title(f"Projection {i}")
            ax.grid(ls="--", alpha=0.5)

        for row in range(nrows):
            axs[row, 0].set_ylabel("Proj.")
        for col in range(ncols):
            row = -1 if col + (nrows - 1) * ncols < nr_projs else -2
            axs[row, col].set_xlabel("x")
        return fig, axs

    # ------------------------- Auxiliary Methods -------------------------

    def _calc_respmat(self, selection, distrib, tot_bins):
        rmat = _np.zeros((tot_bins[-1], tot_bins[-1]), dtype=float)
        idcs = self.model_indices[:, selection]
        dist = distrib[selection]
        area = self.model_grid_area
        for i, idx in enumerate(idcs.T):
            term = dist[i] * area
            for k, l in enumerate(idx):
                id_kl = tot_bins[k] + l
                for r, s in enumerate(idx):
                    rmat[tot_bins[r] + s, id_kl] += term
        return rmat

    def _suggest_grid(self, points, plane=0, npts=3):
        """."""
        # Make sure the thinnest bin has at least npts points in it.
        lines = self.lines
        diff = _np.array([li[1, 2]-li[0, 2] for li in lines])
        coef = _np.array([li[0, plane] for li in lines])
        dis = _np.full(coef.size, _np.inf)
        idc = coef != 0
        dis[idc] = _np.abs(diff[idc] / coef[idc])
        siz = _np.min(dis) / npts

        min_, max_ = points[:, plane].min(), points[:, plane].max()
        npts = int(_np.ceil((max_ - min_) / siz))
        lin = _np.linspace(min_, max_, npts + 1)
        return (lin[1:] + lin[:-1]) / 2

    def _find_main_convex(self, feasible_point=(0.0, 0.0)):
        """Find the convex hull from a feasible point."""
        feasible_point = _np.array(feasible_point)
        sign = _np.sign(self.border_lines @ _np.r_[feasible_point, 1])
        if _np.any(sign == 0):
            raise ValueError("Feasible point belongs to a line.")
        line_eqs = -self.border_lines * sign[:, None]
        hs = _HalfspaceIntersection(line_eqs, feasible_point)
        return _ConvexHull(hs.intersections)
