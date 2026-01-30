"""."""

import time as _time
from multiprocessing import Pool as _Pool

import numpy as _np

from scipy.integrate import trapezoid as _trapezoid
from scipy.interpolate import interp1d as _interp1d
from scipy.ndimage import median_filter as _median_filter
from skimage.morphology import reconstruction as _reconstruction

from siriuspy.devices import CurrInfoLinear, EVG, PowerSupply, Screen, Trigger
from siriuspy.magnet.factory import NormalizerFactory as _Normalizer
from mathphys.imgproc import Image2D_Fit as _Image2D_Fit
from apsuite.commisslib.tomography_linac import trans_matrices as _ment_mats

from ...utils import (
    MeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)


def _process_single_wrapper(args):
    """Wrapper for multiprocessing.Pool.

    Args:
        args (tuple): (self_obj, image, scalex, scaley, nr_bins, nr_fwhms,
        filter_background)

    Returns:
        tuple: Result of MeasTomography._process_single_image(...)
    """
    (self_obj, image, scalex, scaley, nr_bins, nr_fwhms, filter_background) = (
        args
    )
    return self_obj._process_single_image(
        image,
        scalex,
        scaley,
        nr_bins=nr_bins,
        nr_fwhms=nr_fwhms,
        filter_background=filter_background,
    )


class Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = 16  # Number of measurement points
        self.nr_repeat = 2  # Number of repetitions per point
        self.quad_curr_min = -4.5  # [A]
        self.quad_curr_max = 3.5  # [A]
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
        """."""
        cmin = self.params.quad_curr_min
        cmax = self.params.quad_curr_max
        nr_points = self.params.nr_points
        return _np.linspace(cmin, cmax, nr_points)

    @property
    def kl_range(self):
        """."""
        if not hasattr(self, 'normalizer'):
            raise RuntimeError('Normalizer not available in offline mode.')

        return self.normalizer.conv_current_2_strength(
            self.curr_range, strengths_dipole=self.params.bo_dip_energy
        )

    def make_measurement(self):
        """."""
        curr0 = self.devices['qf'].current

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
            self._set_current(curr, timeout=self.params.wait_quad)

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

        self._set_current(curr0, timeout=self.params.wait_quad)
        print('Finished measurement!')

    def process_data(
        self,
        nr_bins=(50, 50),
        nr_fwhms=(3, 3),
        filter_background=True,
        nr_workers=2,
    ):
        """Process raw images in self.data and fill self.analysis.

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
            (self, img, sx, sy, nr_bins, nr_fwhms, filter_background)
            for img, sx, sy in zip(images, scalesx, scalesy)
        ]

        with _Pool(processes=nr_workers) as pool:
            # map sends each args tuple to _process_single_wrapper
            results = pool.map(_process_single_wrapper, args_list)

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

    def _process_single_image(
        self, image, scalex, scaley, nr_bins, nr_fwhms, filter_background
    ):
        """."""
        img = _np.copy(image)

        if filter_background:
            img = self.remove_background(img)

        # Crop image
        img = _median_filter(img, size=3)
        img2dfit = _Image2D_Fit(data=img)
        img2dfit.update_roi_with_fwhm(nr_fwhms[0], nr_fwhms[1])
        roix, roiy = img2dfit.roix, img2dfit.roiy
        img_crop = self.crop_image(img, roix, roiy)

        # Determine positions and center beam in (0, 0)
        posx, posy = self.get_positions(img, -scalex, -scaley)
        x0 = posx[img2dfit.fitx.roi_center]
        y0 = posy[img2dfit.fity.roi_center]
        posx = posx - x0
        posy = posy - y0
        posx = posx[slice(*roix)]
        posy = posy[slice(*roiy)]
        gridx, gridy = _np.meshgrid(posx, posy)

        projx, projy, bins_x, bins_y, bin_size_x, bin_size_y = (
            self.get_projections_bins(
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

    def get_projections_bins(self, image, posx, posy, nr_bins=(50, 50)):
        """."""
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
        bins_x = self.position_to_bin(new_x)
        bins_y = self.position_to_bin(new_y)
        bin_size_x = _np.abs(bins_x[1] - bins_x[0])
        bin_size_y = _np.abs(bins_y[1] - bins_y[0])

        # Normalize Projection
        projx = self.normalize_projection(projx, bin_size_x)
        projy = self.normalize_projection(projy, bin_size_y)

        return projx, projy, bins_x, bins_y, bin_size_x, bin_size_y

    @staticmethod
    def remove_background(image):
        """."""
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

    def _set_current(self, value, timeout=10):
        qf = self.devices['qf']
        qf.current = value
        return qf._wait_float(
            'Current-Mon', value, abs_tol=0.01, timeout=timeout
        )

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
        data['quad_curr_rb'] = curr
        data['quad_curr_mon'] = curr_mon
        data['quad_kl_rb'] = qf.conv_current_2_strength(
            curr, strengths_dipole=self.params.bo_dip_energy
        )
        data['quad_kl_mon'] = qf.conv_current_2_strength(
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
