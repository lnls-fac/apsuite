"""."""

import time as _time
from multiprocessing import Pool as _Pool

import matplotlib.pyplot as _plt
import numpy as _np
from mathphys.imgproc import Image2D_Fit as _Image2D_Fit
from scipy.integrate import trapz as _trapezoid
from scipy.interpolate import interp1d as _interp1d
from scipy.ndimage import median_filter as _median_filter
from siriuspy.devices import CurrInfoLinear, EVG, PowerSupply, Screen, Trigger
from siriuspy.magnet.factory import NormalizerFactory as _Normalizer
from skimage.morphology import reconstruction as _reconstruction

from apsuite.commisslib.tomography_linac import trans_matrices as _ment_mats

from ...utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class Params(_ParamsBaseClass):
    """."""

    DEFAULT_NR_POINTS = 16
    DEFAULT_QUAD_CURR_MIN = -4.5  # [A]
    DEFAULT_QUAD_CURR_MAX = 3.5  # [A]

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = self.DEFAULT_NR_POINTS  # Number of measurement points
        self.nr_repeat = 2  # Number of repetitions per point
        self.quad_curr_min = self.DEFAULT_QUAD_CURR_MIN
        self.quad_curr_max = self.DEFAULT_QUAD_CURR_MAX
        self.wait_quad = 2  # [s]
        self.wait_repeat = 1  # [s]
        self.bo_dip_energy = 0.144  # [GeV]

    def __str__(self):
        """."""
        st = ''
        st += f'nr_points = {self.nr_points:d}\n'
        st += f'nr_repeat = {self.nr_repeat:d}\n'
        st += f'quad_curr_min = {self.quad_curr_min:.2f}\n'
        st += f'quad_curr_max = {self.quad_curr_max:.2f}\n'
        st += f'wait_quad = {self.wait_quad:d}\n'
        st += f'wait_repeat = {self.wait_repeat:d}\n'
        st += f'bo_dip_energy = {self.bo_dip_energy:.3f}\n'
        return st


class MeasTomography(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(params=Params(), isonline=isonline)
        self._curr_range = None
        if isonline:
            self.devices['qf'] = PowerSupply('LI-01:PS-QF3')
            self.devices['scrn'] = Screen(Screen.DEVICES.LI_5)
            self.devices['li_curr'] = CurrInfoLinear(CurrInfoLinear.DEVICES.LI)
            self.devices['li_trig'] = Trigger('LI-Fam:TI-Scrn')
            self.devices['evg'] = EVG()

            self.normalizer = _Normalizer.create(
                self.devices['qf'].devname.replace('PS', 'MA')
            )

    @property
    def curr_range(self):
        """Return QF current range (A). Custom if set, else linear."""
        if self._curr_range is not None:
            return self._curr_range

        cmin = self.params.quad_curr_min
        cmax = self.params.quad_curr_max
        nr_points = self.params.nr_points
        return _np.linspace(cmin, cmax, nr_points)

    @curr_range.setter
    def curr_range(self, values):
        """Set a custom quadrupole current range (A)."""
        arr = _np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError('curr_range must not be empty.')
        self._curr_range = arr
        # keep nr_points consistent with custom range
        self.params.nr_points = arr.size
        self.params.quad_curr_min = _np.min(arr)
        self.params.quad_curr_max = _np.max(arr)

    def reset_curr_range(self):
        """Reset QF current range to the default linear range from Params."""
        self._curr_range = None
        self.params.nr_points = self.params.DEFAULT_NR_POINTS
        self.params.quad_curr_min = self.params.DEFAULT_QUAD_CURR_MIN
        self.params.quad_curr_max = self.params.DEFAULT_QUAD_CURR_MAX

    @property
    def kl_range(self):
        """."""
        if not self.isonline:
            raise RuntimeError('Normalizer not available in offline mode.')

        return self.normalizer.conv_current_2_strength(
            self.curr_range, strengths_dipole=self.params.bo_dip_energy
        )

    def make_measurement(self):
        """Run quadrupole scan and acquire raw data.

        Sets the QF current to each value in 'curr_range' and, for each point,
        triggers a configurable number of shots. For every shot the method
        reads beam current, quadrupole force, screen image and timing
        information and stores them in 'self.data' as nested lists
        [nr_points][nr_repeat]. The loop can be interrupted interactively via
        the EVG pulse menu.
        """
        qf = self.devices['qf']
        curr0 = qf.current

        keys = [
            'timestamp',
            'quad_curr_rb',
            'quad_curr_mon',
            'quad_kl_rb',
            'quad_kl_mon',
            'ict1_curr',
            'ict2_curr',
            'image_raw',
            'image_exposure',
            'image_gain',
            'scalex',
            'scaley',
            'trigger_delay_raw',
        ]

        # Creates nested lists fill with None to store measured data
        for key in keys:
            self.data[key] = [
                [None for _ in range(self.params.nr_repeat)]
                for _ in range(self.params.nr_points)
            ]

        doexit = False
        for idx1, curr in enumerate(self.curr_range):
            qf.set_current(
                value=curr,
                tol=0.01,
                timeout=self.params.wait_quad,
                wait_mon=True,
            )

            print(f'Measuring {idx1 + 1:02d}/{self.params.nr_points:02d}')
            print(f'QF current: {curr:.4f} A')

            for idx2 in range(self.params.nr_repeat):
                doexit = self._pulse_evg()

                if not idx2 and doexit:
                    print('Exiting...')
                    break

                print(f'    repeat {idx2 + 1}/{self.params.nr_repeat}')
                data = self._get_single_data()

                for key in keys:
                    self.data[key][idx1][idx2] = data[key]

                _time.sleep(self.params.wait_repeat)

            print('..finished point\n')

            # Meas loop is not broken during repetitions of same point.
            # It measures all repetitions and then breaks.
            if doexit:
                break

        qf.set_current(
            value=curr0, tol=0.01, timeout=self.params.wait_quad, wait_mon=True
        )
        print('Finished measurement!')

    def process_data(
        self,
        nr_bins=(50, 50),
        nr_fwhms=(3, 3),
        filter_background=True,
        nr_workers=2,
    ):
        """Process raw images in self.data and fill self.analysis.

        Optionally removes background, applies median filter, defines a 2D ROI
        using FWHM in X and Y, recenters beam at (0, 0), builds the position
        grid and computes normalized X/Y projections and their bins.

        Args:
            nr_bins (tuple of int, optional): Number of bins for the X and Y
                projections. Default is (50, 50).
            nr_fwhms (tuple of float, optional): Number of FWHM in X and Y used
                to define ROI size around the beam center. Default is (2, 2).
            filter_background (bool, optional): If True, applies background
                removal to each image before computing projections. Default is
                True.
            nr_workers (int, optional): Number of worker processes used in
                parallel to process the images. Default is 2.

        Returns:
            None: The processed data are stored in "self.analysis" dictionary.
        """
        if self.data is None:
            st = 'No data available in self.data.'
            st += ' Run measurement or load data first.'
            raise RuntimeError(st)

        nr_p = self.params.nr_points
        nr_r = self.params.nr_repeat

        # Collect only images that were actually measured (skip None entries).
        # This is important if the measurement loop was interrupted.
        images = []
        scalesx = []
        scalesy = []
        for p in range(nr_p):
            for r in range(nr_r):
                img = self.data['image_raw'][p][r]
                if img is None:
                    continue
                images.append(img)
                scalesx.append(self.data['scalex'][p][r])
                scalesy.append(self.data['scaley'][p][r])

        # Parallel processing of each valid image.
        args_list = [
            (img, sx, sy, nr_bins, nr_fwhms, filter_background)
            for img, sx, sy in zip(images, scalesx, scalesy)
        ]

        with _Pool(processes=nr_workers) as pool:
            # map sends each args tuple to _process_single_wrapper
            results = pool.map(
                MeasTomography._process_single_wrapper, args_list
            )

        # Recompute nr_p: only points that were fully measured are kept.
        # Measurement logic guarantees len(images) is a multiple of nr_r.
        nr_p = len(images) // nr_r
        keys = [
            'image_cropped',
            'grid_x',
            'grid_y',
            'proj_x',
            'proj_y',
            'bins_x',
            'bins_y',
            'bin_size_x',
            'bin_size_y',
        ]

        # Reshape flat results into nested list [nr_points][nr_repeat]
        for k_idx, key in enumerate(keys):
            flat = [item[k_idx] for item in results]
            nested = [flat[p * nr_r : (p + 1) * nr_r] for p in range(nr_p)]
            self.analysis[key] = nested

        # Find transfer matrices
        def _get_mat(kl):
            return _ment_mats.matrix_from_qf3_to_scrn(kl)

        kl = self.data['quad_kl_rb']
        mx = [[_get_mat(kl[p][r]) for r in range(nr_r)] for p in range(nr_p)]
        my = [[_get_mat(-kl[p][r]) for r in range(nr_r)] for p in range(nr_p)]

        self.analysis['matrices_x'] = mx
        self.analysis['matrices_y'] = my

    @staticmethod
    def _process_single_image(
        image, scalex, scaley, nr_bins, nr_fwhms, filter_background
    ):
        """."""
        img = _np.copy(image)

        if filter_background:
            img = MeasTomography.remove_background(img)

        # Crop image
        img = _median_filter(img, size=3)
        img2dfit = _Image2D_Fit(data=img)
        img2dfit.update_roi_with_fwhm(nr_fwhms[0], nr_fwhms[1])
        roix, roiy = img2dfit.roix, img2dfit.roiy
        img_crop = MeasTomography.crop_image(img, roix, roiy)

        # Determine positions and center beam in (0, 0)
        posx, posy = MeasTomography.get_positions(img, -scalex, -scaley)
        x0 = posx[img2dfit.fitx.roi_center]
        y0 = posy[img2dfit.fity.roi_center]
        posx = posx - x0
        posy = posy - y0
        posx = posx[slice(*roix)]
        posy = posy[slice(*roiy)]
        gridx, gridy = _np.meshgrid(posx, posy)

        projx, projy, bins_x, bins_y, bin_size_x, bin_size_y = (
            MeasTomography.get_projections_bins(
                image=img_crop, posx=posx, posy=posy, nr_bins=nr_bins
            )
        )

        results = (
            img_crop,
            gridx,
            gridy,
            projx,
            projy,
            bins_x,
            bins_y,
            bin_size_x,
            bin_size_y,
        )

        return results

    @staticmethod
    def _process_single_wrapper(args):
        """Wrapper for multiprocessing.Pool.

        Args:
            args (tuple): (self_obj, image, scalex, scaley, nr_bins, nr_fwhms,
            filter_background)

        Returns:
            tuple: Result of MeasTomography._process_single_image(...)
        """
        (image, scalex, scaley, nr_bins, nr_fwhms, filter_background) = args
        return MeasTomography._process_single_image(
            image,
            scalex,
            scaley,
            nr_bins=nr_bins,
            nr_fwhms=nr_fwhms,
            filter_background=filter_background,
        )

    @staticmethod
    def get_projections_bins(image, posx, posy, nr_bins=(50, 50)):
        """Compute normalized X and Y projections and their bins.

        This method computes the sum of pixel intensities along the X and Y
        axes of the given image, interpolates the projections onto a uniform
        grid with a specified number of bins, and returns the corresponding bin
        edges and bin sizes. Projections are normalized such that their
        integrals is equals one.

        Args:
            image (ndarray): 2D image array.
            posx (ndarray): X-axis positions corresponding to image columns.
            posy (ndarray): Y-axis positions corresponding to image rows.
            nr_bins (tuple of int, optional): Number of bins for X and Y axes.
                Default is (50, 50).

        Returns:
            tuple: (projx, projy, bins_x, bins_y, bin_size_x, bin_size_y):
                - projx, projy (ndarray): Normalized projections along X and Y.
                - bins_x, bins_y (ndarray): Bin edges for X and Y axes.
                - bin_size_x, bin_size_y (float): Bin sizes for X and Y axes.
        """
        # Interpolates projections
        projx = _np.sum(image, axis=0)
        projy = _np.sum(image, axis=1)
        fx = _interp1d(posx, projx)
        fy = _interp1d(posy, projy)
        xmin, xmax = posx[0], posx[-1]
        ymin, ymax = posy[0], posy[-1]

        new_x = _np.linspace(xmin, xmax, nr_bins[0])
        new_y = _np.linspace(ymin, ymax, nr_bins[1])

        projx = fx(new_x)
        projy = fy(new_y)

        # Create bins
        bins_x = MeasTomography.position_to_bin(new_x)
        bins_y = MeasTomography.position_to_bin(new_y)
        bin_size_x = _np.abs(bins_x[1] - bins_x[0])
        bin_size_y = _np.abs(bins_y[1] - bins_y[0])

        # Normalize Projection
        projx = MeasTomography.normalize_projection(projx, bin_size_x)
        projy = MeasTomography.normalize_projection(projy, bin_size_y)

        return projx, projy, bins_x, bins_y, bin_size_x, bin_size_y

    @staticmethod
    def remove_background(image):
        """Remove background from image using morphological reconstruction."""
        # image = img_as_float(image)
        seed = _np.copy(image)
        seed[1:-1, 1:-1] = _np.min(image)
        mask = image

        dilated = _reconstruction(seed, mask, method='dilation')
        image = image - dilated
        mask = _np.log10(image) <= 0
        image[mask] = 0

        return image

    @staticmethod
    def normalize_projection(proj, bin_size):
        """."""
        proj -= _np.min(proj)
        const = _trapezoid(proj, dx=bin_size)
        proj /= const
        return proj * bin_size

    @staticmethod
    def crop_image(image, roix, roiy):
        """."""
        return image[slice(*roiy), slice(*roix)]

    @staticmethod
    def get_positions(image, scalex, scaley):
        """Returns scaled positions of images."""
        sy, sx = image.shape
        x = _np.arange(sx) * scalex
        y = _np.arange(sy) * scaley
        x -= x[sx // 2]
        y -= y[sy // 2]
        return x, y

    @staticmethod
    def position_to_bin(x):
        """Converts a 1D array of positions into bin edges."""
        nr_points = len(x)
        delta = x[1] - x[0]
        xmin = x[0] - delta / 2
        xmax = x[-1] + delta / 2
        return _np.linspace(xmin, xmax, nr_points + 1)

    def _pulse_evg(self):
        while True:
            self.devices['evg'].cmd_turn_on_injection()
            print('Injection pulse triggered.')
            st = '[Y] pulse again  |'
            st += '  [N] continue measurement  |'
            st += '  [E] exit measurement'
            print(st)
            ret = input('Choose an option [Y/N/E]: ').strip().lower()

            if ret in ('e', 'exit'):
                # abort entire scan
                return True

            if ret in ('n', 'no'):
                # continue measurement
                return False

            if ret in ('y', 'yes', ''):
                # pulse again
                continue

            print('Invalid option. Please type Y, N or E.')

    def _get_single_data(self):
        data = dict()
        data['timestamp'] = _time.time()

        qf = self.devices['qf']
        curr = qf.current
        curr_mon = qf.current_mon
        norm = self.normalizer
        data['quad_curr_rb'] = curr
        data['quad_curr_mon'] = curr_mon
        data['quad_kl_rb'] = norm.conv_current_2_strength(
            curr, strengths_dipole=self.params.bo_dip_energy
        )
        data['quad_kl_mon'] = norm.conv_current_2_strength(
            curr_mon, strengths_dipole=self.params.bo_dip_energy
        )

        data['ict1_curr'] = self.devices['li_curr'].charge_ict1
        data['ict2_curr'] = self.devices['li_curr'].charge_ict2

        data['image_raw'] = self.devices['scrn'].image
        data['image_exposure'] = self.devices['scrn'].cam_exposure
        data['image_gain'] = self.devices['scrn'].cam_gain
        data['scalex'] = self.devices['scrn'].scale_factor_x
        data['scaley'] = self.devices['scrn'].scale_factor_y
        data['trigger_delay_raw'] = self.devices['li_trig'].delay_raw

        return data

    # ----------Visualization Methods----------

    def plot_raw_images(
        self,
        points=None,
        repeat=None,
        max_per_fig=9,
        cmap='jet',
        figsize=(9, 9),
        suptitle=None,
    ):
        """Plot raw images for selected measurement points."""
        if self.data is None or 'image_raw' not in self.data:
            raise RuntimeError('No raw images available in self.data.')

        nr_p = self.params.nr_points
        nr_r = self.params.nr_repeat

        # Select valid data points
        repeat = self._resolve_repeat(repeat, nr_r)
        indices = self._resolve_points(points, repeat, nr_p)

        # Pagineted plot
        self._plot_raw_images_paginated(
            indices=indices,
            repeat=repeat,
            max_per_fig=max_per_fig,
            cmap=cmap,
            figsize=figsize,
            suptitle=suptitle,
        )

    def plot_cropped_images(
        self,
        points=None,
        repeat=None,
        max_per_fig=9,
        cmap='jet',
        figsize=(9, 9),
        suptitle=None,
    ):
        """Plot cropped images from self.analysis with their position grids."""
        if self.analysis is None or 'image_cropped' not in self.analysis:
            raise RuntimeError(
                'No processed images available in self.analysis.'
            )

        nr_p = len(self.analysis['image_cropped'])
        nr_r = self.params.nr_repeat

        # resolve repeat e pontos válidos
        repeat = self._resolve_repeat(repeat, nr_r)
        indices = self._resolve_points_from_analysis(points, repeat, nr_p)

        # paginação e plot
        self._plot_cropped_images_paginated(
            indices=indices,
            repeat=repeat,
            max_per_fig=max_per_fig,
            cmap=cmap,
            figsize=figsize,
            suptitle=suptitle,
        )

    def plot_projections(
        self,
        points=None,
        repeat=None,
        plane='x',  # 'x' ou 'y' (removido 'both')
        max_per_fig=9,
        figsize=(12, 8),
        suptitle=None,
    ):
        """Plot binned projections from self.analysis."""
        if self.analysis is None or 'proj_x' not in self.analysis:
            raise RuntimeError(
                'No processed projections available in self.analysis.'
            )

        nr_p = len(self.analysis['image_cropped'])
        nr_r = self.params.nr_repeat

        if plane not in ('x', 'y'):
            raise ValueError("plane must be 'x' or 'y'.")

        repeat = self._resolve_repeat(repeat, nr_r)
        indices = self._resolve_points_from_analysis(points, repeat, nr_p)

        self._plot_projections_paginated(
            indices=indices,
            repeat=repeat,
            plane=plane,
            max_per_fig=max_per_fig,
            figsize=figsize,
            suptitle=suptitle,
        )

    def _resolve_repeat(self, repeat, nr_r):
        """Return a valid repeat index."""
        if repeat is None:
            repeat = 0
        if not (0 <= repeat < nr_r):
            raise ValueError(
                f'repeat must be in [0, {nr_r - 1}], got {repeat}.'
            )
        return repeat

    def _resolve_points(self, points, repeat, nr_p):
        """Return list of point indices with valid images for given repeat."""
        if points is None:
            points = list(range(nr_p))
        else:
            points = [int(p) for p in points if 0 <= int(p) < nr_p]

        indices = [
            p for p in points if self.data['image_raw'][p][repeat] is not None
        ]
        return indices

    def _resolve_points_from_analysis(self, points, repeat, nr_p):
        """Return list of point indices with valid cropped images."""
        if points is None:
            points = list(range(nr_p))
        else:
            points = [int(p) for p in points if 0 <= int(p) < nr_p]

        indices = [
            p
            for p in points
            if self.analysis['image_cropped'][p][repeat] is not None
        ]
        return indices

    def _plot_raw_images_paginated(
        self, indices, repeat, max_per_fig, cmap, figsize, suptitle
    ):
        """Iterate over pages and plot subsets of images."""
        total = len(indices)
        start = 0
        fig_idx = 1

        while start < total:
            end = min(start + max_per_fig, total)
            batch = indices[start:end]

            self._plot_raw_images_page(
                batch_indices=batch,
                repeat=repeat,
                cmap=cmap,
                figsize=figsize,
                page_idx=fig_idx,
                suptitle=suptitle,
            )

            fig_idx += 1
            start = end

    def _plot_raw_images_page(
        self, batch_indices, repeat, cmap, figsize, page_idx, suptitle
    ):
        """Plot a single page (figure) of raw images."""
        n_imgs = len(batch_indices)
        ncols = min(3, n_imgs)
        nrows = (n_imgs + ncols - 1) // ncols

        fig, axs = _plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=True
        )
        axs = _np.array(axs).ravel()

        for ax, p in zip(axs, batch_indices):
            curr = self.data['quad_curr_rb'][p][repeat]
            img = self.data['image_raw'][p][repeat]
            ax.imshow(img, cmap=cmap)
            ax.text(
                0.5,
                0.92,
                f'QF3: {curr:.2f} A  (p={p}, r={repeat})',
                color='white',
                fontsize=9,
                ha='center',
                va='top',
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused axes
        for ax in axs[n_imgs:]:
            ax.axis('off')

        if suptitle is not None:
            title = suptitle
        else:
            title = f'Raw images (repeat={repeat}, page {page_idx})'
        fig.suptitle(title)
        fig.tight_layout()
        _plt.show()

    def _plot_cropped_images_paginated(
        self, indices, repeat, max_per_fig, cmap, figsize, suptitle
    ):
        """Iterate over pages and plot subsets of cropped images."""
        total = len(indices)
        start = 0
        fig_idx = 1

        while start < total:
            end = min(start + max_per_fig, total)
            batch = indices[start:end]

            self._plot_cropped_images_page(
                batch_indices=batch,
                repeat=repeat,
                cmap=cmap,
                figsize=figsize,
                page_idx=fig_idx,
                suptitle=suptitle,
            )

            fig_idx += 1
            start = end

    def _plot_cropped_images_page(
        self, batch_indices, repeat, cmap, figsize, page_idx, suptitle
    ):
        """Plot a single page of cropped images."""
        n_imgs = len(batch_indices)
        ncols = min(3, n_imgs)
        nrows = (n_imgs + ncols - 1) // ncols

        fig, axs = _plt.subplots(
            nrows, ncols, figsize=figsize, sharex=False, sharey=False
        )
        axs = _np.array(axs).ravel()

        for ax, p in zip(axs, batch_indices):
            curr = self.data['quad_curr_rb'][p][repeat]
            img = self.analysis['image_cropped'][p][repeat]
            gridx = self.analysis['grid_x'][p][repeat]
            gridy = self.analysis['grid_y'][p][repeat]

            ax.pcolormesh(gridx, gridy, img, cmap=cmap)
            ax.text(
                0.5,
                0.92,
                f'QF3: {curr:.2f} A  (p={p}, r={repeat})',
                color='white',
                fontsize=9,
                ha='center',
                va='top',
                transform=ax.transAxes,
            )
            ax.set_aspect('equal')

        # Hide unused axes
        for ax in axs[n_imgs:]:
            ax.axis('off')

        if suptitle is not None:
            title = suptitle
        else:
            title = f'Cropped images (repeat={repeat}, page {page_idx})'
        fig.suptitle(title)
        fig.tight_layout()
        _plt.show()

    def _plot_projections_paginated(
        self, indices, repeat, plane, max_per_fig, figsize, suptitle
    ):
        """Iterate over pages and plot subsets of projections."""
        total = len(indices)
        start = 0
        fig_idx = 1

        global_xmin, global_xmax = self._get_global_xlim(
            indices, repeat, plane
        )

        while start < total:
            end = min(start + max_per_fig, total)
            batch = indices[start:end]

            self._plot_projections_page(
                batch_indices=batch,
                repeat=repeat,
                plane=plane,
                figsize=figsize,
                page_idx=fig_idx,
                suptitle=suptitle,
                global_xlim=(global_xmin, global_xmax),
            )

            fig_idx += 1
            start = end

    def _get_global_xlim(self, indices, repeat, plane):
        all_bins = []
        for p in indices:
            bins = self.analysis[f'bins_{plane}'][p][repeat]
            all_bins.append(bins)

        xmin = _np.min(all_bins)
        xmax = _np.max(all_bins)
        delta = xmax - xmin
        lim_min = xmin - 0.05 * delta
        lim_max = xmax + 0.05 * delta
        return lim_min, lim_max

    def _plot_projections_page(
        self,
        batch_indices,
        repeat,
        plane,
        figsize,
        page_idx,
        suptitle,
        global_xlim,
    ):
        """Plot a single page of projections (plane='x' or 'y')."""
        n_imgs = len(batch_indices)
        ncols = min(3, n_imgs)
        nrows = (n_imgs + ncols - 1) // ncols

        fig, axs = _plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=False
        )
        axs = _np.array(axs).ravel()

        color = 'C0' if plane == 'x' else 'C1'
        for ax, p in zip(axs, batch_indices):
            kl = self.data['quad_kl_rb'][p][repeat]

            proj = self.analysis[f'proj_{plane}'][p][repeat]
            bins_ = self.analysis[f'bins_{plane}'][p][repeat]
            binsize = self.analysis[f'bin_size_{plane}'][p][repeat]

            ax.bar(
                bins_[:-1],
                proj,
                width=binsize,
                align='edge',
                color=color,
                alpha=0.8,
            )
            ax.set_title(f'{plane.upper()} (KL={kl:.2f} 1/m)')
            ax.set_ylabel('Intensity')
            ax.set_xlim(global_xlim)

        # Hide unused axes
        for ax in axs[n_imgs:]:
            ax.axis('off')

        if suptitle is not None:
            title = suptitle
        else:
            title = f'{plane.upper()} projections '
            title += f'(repeat={repeat}, page {page_idx})'
        fig.suptitle(title)
        fig.tight_layout()
        _plt.show()
