"""."""

from copy import deepcopy as _deepcopy

import matplotlib.pyplot as _plt
import numpy as _np
from mathphys.imgproc import Image2D_ROI as _Image2D_ROI
from scipy.interpolate import griddata as _griddata, interp1d as _interp1d
from scipy.spatial import ConvexHull as _ConvexHull, \
    HalfspaceIntersection as _HalfspaceIntersection


def plot_lines(lines, fig=None, ax=None, **kwargs):
    """Plots a series of lines defined by their coefficients.

    The coefficients are in the form ax + by + c = 0.
    """
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    for line in lines:
        if _np.isclose(line[1], 0):
            ax.axvline(-line[2] / line[0], **kwargs)
        else:
            a = line[0]
            b = line[1]
            c = line[2]
            ax.axline((0, -c / b), slope=-a / b, **kwargs)

    return fig, ax


def plot_convex(
    hull: _ConvexHull, color="r", marker="o", fig=None, ax=None, **kwargs
):
    """Plots a convex hull."""
    if fig is None or ax is None:
        fig, ax = _plt.subplots(1, 1)

    for simplex in hull.simplices:
        ax.plot(
            hull.points[simplex, 0],
            hull.points[simplex, 1],
            marker=marker,
            color=color,
            **kwargs,
        )

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
        projx = _np.sum(self.image, axis=0)[slice(*self.roix)]
        projy = _np.sum(self.image, axis=1)[slice(*self.roiy)]
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


class ConvexPolygon:
    """Represents a mesh of convexes defined by the intersection of lines.

    The lines are generated from projection bins, with each projection taken in
    a specific direction determined by the provided matrices. This class
    provides methods for finding convexes shapes from these intersections.
    """

    def __init__(self, matrices, projs_bins):
        """Creates a list of lines for each projection bin.

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
        projs_bins (list of np.ndarray): A list of 1D arrays where each array
            contains the bin edges for the respective projection.
        """
        lines = []
        for i, proj_bins in enumerate(projs_bins):
            matrix = matrices[i]
            nr_bins = len(proj_bins)
            ab_line = _np.tile(matrix[0], (nr_bins, 1))
            lines.append(_np.vstack([ab_line.T, -proj_bins]).T)

        self.matrices = matrices
        self.projs_bins = projs_bins
        self.lines = lines
        self.nr_projs = len(lines)
        self._main_convex = None
        self._convexes = dict()

    @property
    def raveled_lines(self):
        """Unpack the projections bin lines in an array."""
        for j, proj_lines in enumerate(self.lines):
            if j == 0:
                r_lines = proj_lines
            else:
                r_lines = _np.vstack([r_lines, proj_lines])
        return r_lines

    @property
    def nr_bins(self):
        """Returns a list with the number of bins in each projection."""
        nr = []
        for proj_lines in self.lines:
            nr.append(proj_lines.shape[0] - 1)
        return nr

    @property
    def convexes(self):
        """Returns all convexes resulted from the bins grid."""
        if not self._convexes:
            sent = "Main convex not splitted. "
            sent += "Please call 'subdivide_main_convex' first."
            raise ValueError(sent)
        return self._convexes

    @property
    def main_convex(self):
        """Returns the main convex hull."""
        if self._main_convex is None:
            sent = "Convex hull not calculated. "
            sent += "Please call 'find_convex_from_borders' first."
            raise ValueError(sent)
        return self._main_convex

    @property
    def border_lines(self):
        """Calculates the border lines of main convex hull."""
        border_lines = []
        for j in range(self.nr_projs):
            border_lines.append(self.lines[j][0, :])
            border_lines.append(self.lines[j][-1, :])
        return _np.array(border_lines)

    def find_convex_from_borders(self, feasible_point):
        """Finds the convex hull from a feasible point."""
        feasible_point = _np.array(feasible_point)
        sign = _np.sign(self.border_lines @ _np.r_[feasible_point, 1])
        if _np.any(sign == 0):
            raise ValueError("Feasoble point belongs to a line.")
        line_eqs = -self.border_lines * sign[:, None]
        hs = _HalfspaceIntersection(line_eqs, feasible_point)
        hull = _ConvexHull(hs.intersections)
        self._main_convex = hull

        return hull

    def subdivide_main_convex(
        self, convex_list: list, lines=None, level=0, indices=None
    ):
        """Recursively subdivides the main convex hull."""
        if self._main_convex is None:
            sent = "Convex hull not calculated. "
            sent += "Please call 'find_convex_from_borders' first."
            raise ValueError(sent)

        if indices is None:
            indices = [[] for _ in convex_list]

        if lines is None:
            lines = self.lines

        if level >= len(lines):
            self._convexes = {
                tuple(idx): c for c, idx in zip(convex_list, indices)
            }
            return convex_list, indices

        new_list = []
        new_idcs = []

        for elem, index in zip(convex_list, indices):
            subdivided, sub_indices = self._subdivide_convex(
                elem, lines[level][1:-1, :]
            )
            new_list.extend(subdivided)
            new_idcs.extend([index + [sub_idx] for sub_idx in sub_indices])

        return self.subdivide_main_convex(new_list, lines, level + 1, new_idcs)

    def get_convexes_propty(self, propty_name='area', convexes=None):
        """."""
        propty_name = propty_name.lower()
        if propty_name not in {'area', 'vertices', 'centroid'}:
            raise ValueError('propty_name not valid.')

        convexes = convexes or self.convexes
        propties = {}
        for idcs, convex in convexes.items():
            if propty_name.startswith('vertices'):
                points = convex.points
                ppt = points[convex.vertices]
            elif propty_name.startswith('centroid'):
                points = convex.points
                ppt = _np.sum(points, axis=0) / points.shape[0]
            elif propty_name.startswith('area'):
                ppt = convex.volume
            propties[tuple(idcs)] = ppt
        return propties

    def _subdivide_convex(self, hull, line_eqs):
        convex_list = [hull]
        idx_list = [None]
        for m, line in enumerate(line_eqs):
            cut_result, pol_changed = self._cut_convex(convex_list[-1], line)
            convex_list[-1:] = cut_result
            if len(cut_result) == 2:
                idx_list[-1:] = [m, m + 1]
            if len(cut_result) == 1 and idx_list[-1] is None:
                if pol_changed:
                    idx_list[-1] = m
                    break
                elif m == line_eqs.shape[0] - 1:
                    idx_list[-1] = m + 1
                    break

        return convex_list, idx_list

    def _cut_convex(self, hull: _ConvexHull, line_eq):
        # find the intersection points
        intersec_points = self._intersection_points(line_eq, hull.equations)

        # verify if intersection points are in polygon
        cut_points = self._points_in_cvx(intersec_points, hull.equations)

        all_points = hull.points
        if len(cut_points) > 1:
            all_points = _np.vstack([hull.points, cut_points])
        all_points = self._unique_points(all_points)
        ones_column = _np.ones((all_points.shape[0], 1))
        test_points = line_eq @ _np.hstack([all_points, ones_column]).T

        on_line_mask = _np.isclose(test_points, 0)
        neg_side_mask = (test_points < 0) & ~on_line_mask
        pos_side_mask = (test_points > 0) & ~on_line_mask

        mid_points = all_points[on_line_mask]
        neg_points = all_points[neg_side_mask]
        pos_points = all_points[pos_side_mask]

        if len(pos_points) == 0:
            # second output tells if there was polarity change:
            return [hull], True
        if len(neg_points) == 0:
            return [hull], False

        convex_neg = _ConvexHull(_np.vstack([neg_points, mid_points]))
        convex_pos = _ConvexHull(_np.vstack([pos_points, mid_points]))

        return [convex_neg, convex_pos], None

    def _intersection_points(self, line_eq, cvx_eqs):
        matrices = _np.zeros((cvx_eqs.shape[0], 2, 2))
        matrices[:, 0, :] = cvx_eqs[:, :2]
        matrices[:, 1, :] = line_eq[None, :2]
        idcs = ~_np.isclose(_np.linalg.det(matrices), 0)
        matrices = matrices[idcs]
        c = _np.zeros((matrices.shape[0], 2))
        c[:, 0] -= cvx_eqs[idcs, 2]
        c[:, 1] -= line_eq[2]
        return _np.linalg.solve(matrices, c)

    def _points_in_cvx(self, points, cvx_eqs):
        eq_points = _np.hstack([points, _np.ones((points.shape[0], 1))])
        fxy = eq_points @ cvx_eqs.T
        idcs = _np.all(_np.isclose(fxy, 0) | (fxy < 0), axis=1)
        return self._unique_points(points[idcs])

    def _unique_points(self, points, rtol=1e-12, atol=1e-12):
        uniques = []
        for point in points:
            if not any(
                _np.all(_np.isclose(point, p, rtol=rtol, atol=atol))
                for p in uniques
            ):
                uniques.append(point)

        return _np.array(uniques)


class DistribReconstruction(ConvexPolygon):
    """Class for reconstructing 2D-distributions based on a convex mesh."""

    def __init__(self, matrices, projs, projs_bins):
        """."""
        super().__init__(matrices, projs_bins)
        self._projs = projs
        self._lagmults = None
        self._convergence_info = None
        self._init_lagmults = None

    @property
    def init_lagmults(self):
        """Define a initial list for Lagrange multipliers.

        If the projection bin values are non-zero, the Lagrange multiplier is
        set to one. Otherwise, the Lagrange multiplier is set to zero.
        """
        if self._init_lagmults is not None:
            return _deepcopy(self._init_lagmults)
        return self.get_lagmults_guess(fancy=False)

    @init_lagmults.setter
    def init_lagmults(self, value):
        """."""
        if len(value) != len(self._projs):
            raise ValueError(
                'Length of value must match length of projections.'
            )
        for i, (lamb, prj) in enumerate(zip(value, self._projs)):
            if len(lamb) != len(prj):
                raise ValueError(
                    f'Length of {i}th lambda must match length of {i}th proj.'
                )
        self._init_lagmults = _deepcopy(value)

    @property
    def lagmults(self):
        """."""
        if self._lagmults is None:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'recontruct_distribution' first."
            raise ValueError(sent)
        return self._lagmults

    @property
    def convergence_info(self):
        """."""
        if self._convergence_info is None:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'recontruct_distribution' first."
            raise ValueError(sent)
        return self._convergence_info

    def select_non_trivial_convexes(self):
        """Selects convexes whose initial Lagrange multupliers are non-zero."""
        if self._init_lagmults is None:
            return self.convexes
        selected = dict()
        for idx, convex in self.convexes.items():
            for j, m in enumerate(idx):
                if not bool(self._init_lagmults[j][m]):
                    break
            else:
                selected[idx] = convex
        return selected

    def get_model_projections(self):
        """."""
        if self._lagmults is None:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'recontruct_distribution' first."
            raise ValueError(sent)
        convexes = self.convexes
        lagmults = self.lagmults
        projs_calc = []
        nr_bins = self.nr_bins

        for j in range(self.nr_projs):
            p = []
            for m in range(nr_bins[j]):
                p.append(
                    self._get_projection_at_bin(j, m, lagmults, convexes)
                )
            projs_calc.append(_np.array(p))
        return projs_calc

    def get_lagmults_guess(self, fancy=False):
        """."""
        lagmults = []
        for proj in self._projs:
            lambda_proj = _np.ones_like(proj, dtype="float")
            idx = proj == 0
            lambda_proj[idx] = -_np.inf
            if fancy:
                lambda_proj[~idx] = _np.log(proj[~idx])

            lagmults.append(lambda_proj)
        return lagmults

    def get_convexes_propty(self, propty_name='area', convexes=None, **kwargs):
        """."""
        propty_name = propty_name.lower()
        if propty_name not in {'area', 'vertices', 'centroid', 'distrib'}:
            raise ValueError('propty_name not valid.')
        elif propty_name.startswith('distrib') and self._lagmults is None:
            raise ValueError(
                'Cannot return distribution. '
                "Please call 'recontruct_distribution' first."
            )

        convexes = convexes or self.convexes

        if propty_name in {'area', 'vertices', 'centroid'}:
            return super().get_convexes_propty(
                propty_name=propty_name, convexes=convexes
            )

        lagmults = self.lagmults
        distrib = []
        for idx in convexes.keys():
            distrib.append(self._get_distrib_at_bin(idx, lagmults))
        distrib = _np.array(distrib)
        areas = _np.array(list(kwargs.get('areas', self.get_convexes_propty(
            propty_name='area', convexes=convexes
        )).values()))
        distrib /= _np.sum(distrib * areas)

        propties = {idx: val for idx, val in zip(convexes.keys(), distrib)}
        return propties

    def get_model_distrib_in_grid(self, gridx, gridy):
        """Interpolates distribution in a regular lattice."""
        if self._lagmults is None:
            sent = "Distribution not reconstructed. "
            sent += "Please call 'recontruct_distribution' first."
            raise ValueError(sent)

        centroids = _np.array(list(self.get_convexes_propty(
            propty_name='centroid'
        ).values()))
        distrib = _np.array(list(self.get_convexes_propty(
            propty_name='distrib'
        ).values()))
        points = _np.vstack([gridx.ravel(), gridy.ravel()]).T
        deltax = gridx[0, 1] - gridx[0, 0]
        deltay = gridy[1, 0] - gridy[0, 0]
        grid_area = deltax * deltay
        distrib_itp = _griddata(centroids, distrib, points, fill_value=0)
        distrib_itp = distrib_itp.reshape(gridx.shape)
        return distrib_itp / _np.sum(distrib_itp) / grid_area

    def reconstruct_distribution_original(self, nr_iter=1000, tol=0.001):
        """Finds values for Lagrange multipliers using a interative process.

        This method updates the Lagrange multipliers initial values
        (`lagmults`) based on the projection values. It employs an iterative
        algorithm that adjusts the distribution until convergence is reached or
        the maximum number of iterations is completed. In each iteration "i",
        the value of "lambda" for the j-th projection and m-th bin is updated
        based on the measured projection value and the calculated projection
        value. The update follows the formula:
        lambda_(i + 1) = ln(G_meas / G_calc) + lambda_(i).
        G_calc is obtained from lambda_(i) values.

        Args:
        nr_iter (int, optional): The maximum number of iterations to perform.
            Default is 1000.
        tol (float, optional): The tolerance level for convergence.
            The iteration stops when the projection relative value
            (G_meas / G_calc) is below this value. Default is 0.001.
        weight (float, optional): A weighting factor that influences the
            adjustment of the distribution values based on the difference
            between measured and calculated projections. Default is 1.

        Returns:
            None: The method updates the internal attributes
                `_lagmults` and `_convergence_info`.
        """
        lagmults = self.init_lagmults
        selecteds_in_distrib = self.select_non_trivial_convexes()
        convergence = []

        for i in range(nr_iter):
            max_frac = 0
            for j in range(len(lagmults)):
                idcs_m = _np.unique([
                    k[j] for k in selecteds_in_distrib.keys()
                ])
                for m in idcs_m:
                    true_proj = self._projs[j][m]
                    new_proj = self._get_projection_at_bin(
                        j, m, lagmults, selecteds_in_distrib
                    )
                    diff_proj = _np.log(true_proj) - _np.log(new_proj)
                    lagmults[j][m] = diff_proj + lagmults[j][m]
                    frac = true_proj / new_proj
                    max_frac = max(abs(frac - 1), max_frac)
            converged = max_frac < tol
            if not i % 10 or converged:
                convergence.append([i, max_frac])
                print(f"{i:04d}/{nr_iter:04d} -> {max_frac:.4f}")
            if converged:
                break

        self._lagmults = lagmults
        self._convergence_info = convergence
        sent = "Distribution rescontructed. "
        sent += "Call 'convexes_properties' "
        sent += "to get distribution values."
        print(sent)

    def reconstruct_distribution(
        self,
        nr_iter=1000,
        tol=0.001,
        gain=0.05,
        max_change=1,
        min_rchi2=1e-4,
        rcond=None,
        fancy_init=False,
    ):
        """Finds values for Lagrange multipliers using a interative process.

        This method updates the Lagrange multipliers initial values
        (`lagmults`) based on the projection values. It employs an iterative
        algorithm that adjusts the distribution until convergence is reached or
        the maximum number of iterations is completed. In each iteration "i",
        the value of "lambda" for the j-th projection and m-th bin is updated
        based on the measured projection value and the calculated projection
        value. The update follows the formula:
        lambda_(i + 1) = ln(G_meas / G_calc) + lambda_(i).
        G_calc is obtained from lambda_(i) values.

        Args:
        nr_iter (int, optional): The maximum number of iterations to perform.
            Default is 1000.
        tol (float, optional): The tolerance level for convergence.
            The iteration stops when the projection relative value
            (G_meas / G_calc) is below this value. Default is 0.001.
        weight (float, optional): A weighting factor that influences the
            adjustment of the distribution values based on the difference
            between measured and calculated projections. Default is 1.

        Returns:
            None: The method updates the internal attributes
                `_lagmults` and `_convergence_info`.
        """
        self._lagmults = self.get_lagmults_guess(fancy=fancy_init)

        convexes_sel = self.select_non_trivial_convexes()
        idcs = list(convexes_sel.keys())
        pj_g = _np.hstack(self._projs)
        max_chi2 = _np.sum(pj_g*pj_g)
        nr_bins = self.nr_bins
        tot_bins = _np.r_[0, _np.cumsum(nr_bins)]

        areas = self.get_convexes_propty(
            propty_name='area', convexes=convexes_sel
        )
        distrib = self.get_convexes_propty(
            propty_name='distrib', convexes=convexes_sel, areas=areas
        )
        pjs = _np.hstack(self.get_model_projections())
        res = pj_g - pjs

        convergence = []
        for itr in range(nr_iter):
            rmat = self._calc_respmat(idcs, distrib, areas, tot_bins, nr_bins)
            dlg, _, _, svs = _np.linalg.lstsq(rmat, res, rcond=rcond)
            svs /= svs[0]

            dlg *= gain
            dlg *= max_change / max(max_change, _np.abs(dlg).max())
            lamb = _np.hstack(self._lagmults) + dlg
            self._lagmults = _np.split(lamb, tot_bins[1:-1])

            distrib = self.get_convexes_propty(
                propty_name='distrib', convexes=convexes_sel, areas=areas
            )
            pjs = _np.hstack(self.get_model_projections())
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
                    f'max. rel. change={max_rchg:9.2e}, '
                    f"nr_svs={nsv:04d}/{svs.size:04d}, "
                    f'rel. chi_square={rchi2:9.2e}'
                )
            if converged:
                break

        self._convergence_info = convergence
        sent = "Distribution rescontructed. "
        sent += "Call 'convexes_properties' "
        sent += "to get distribution values."
        print(sent)

    def _calc_respmat(self, idcs, distrib, areas, tot_bins, nr_bins):
        rmat = _np.zeros((tot_bins[-1], tot_bins[-1]), dtype=float)
        for k, n in enumerate(nr_bins):
            for l in range(n):
                idc_fil = [idx for idx in idcs if idx[k] == l]
                for idx in idc_fil:
                    idx = tuple(idx)
                    term = distrib[idx] * areas[idx]
                    for r, s in enumerate(idx):
                        rmat[tot_bins[r] + s, tot_bins[k] + l] += term
        return rmat

    def _get_projection_at_bin(self, nr_proj, nr_bin, lagmults, convexes):
        """Calculates projections values from Lagrange multipliers."""
        cvx_sel = {k: v for k, v in convexes.items() if k[nr_proj] == nr_bin}
        if not cvx_sel:
            print(nr_proj, nr_bin)
            return 1
        proj_value = 0
        for idcs, convex in cvx_sel.items():
            proj_value += convex.volume * self._get_distrib_at_bin(
                idcs, lagmults
            )
        return proj_value

    def _get_distrib_at_bin(self, convex_idcs, lagmults):
        """Calculates projections values from Lagrange multipliers."""
        value = -1
        for j, m in enumerate(convex_idcs):
            value += lagmults[j][m]
        return _np.exp(value)
