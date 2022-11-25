"""Image Fitting Classes."""



import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches

from scipy.optimize import curve_fit as _curve_fit


class Image:
    """."""

    def __init__(self, data=None, roix=None, roiy=None):
        """."""
        self._data = None
        self._roix = None
        self._roiy = None
        self._indcx = None
        self._indcy = None
        self._projx = None
        self._projy = None
        self._meanx = None
        self._meany = None
        self._sigmax = None
        self._sigmay = None
        self._update_image_roi(data, roix, roiy)
        
    @property
    def data(self):
        """."""
        return self._data

    @property
    def shape(self):
        """."""
        if self.data is not None:
            return self.data.shape
        else:
            return None

    @property
    def sizex(self):
        """."""
        return None if self.shape is None else self.shape[1]
        
    @property
    def sizey(self):
        """."""
        return None if self.shape is None else self.shape[0]

    @data.setter
    def image(self, value):
        """."""
        self._update_image_roi(value, self._roix, self._roiy)

    @property
    def roix(self):
        """."""
        return self._roix

    @roix.setter
    def roix(self, value):
        """."""
        self._update_image_roi(self._data, value, self._roiy)

    @property
    def roiy(self):
        """."""
        return self._roiy

    @roiy.setter
    def roiy(self, value):
        """."""
        self._update_image_roi(self._data, self._roix, value)

    @property
    def indcx(self):
        """."""
        return self._indcx

    @property
    def indcy(self):
        """."""
        return self._indcy

    @property
    def projx(self):
        """."""
        return self._projx

    @property
    def projy(self):
        """."""
        return self._projy

    @property
    def meanx(self):
        """."""
        return self._meanx

    @property
    def meany(self):
        """."""
        return self._meany

    @property
    def sigmax(self):
        """."""
        return self._sigmax

    @property
    def sigmay(self):
        """."""
        return self._sigmay

    def imshow(self, fig=None, axis=None):
        """."""
        if None in (fig, axis):
            fig, axis = _plt.subplots()

        color = 'tab:red'
        axis.imshow(self.image)
        if None not in (self.meanx, self.meany):
            axis.plot(self.meanx, self.meany, 'o', ms=2, color=color)
        if None not in (self.sigmax, self.sigmay):
            # intersecting elipse at half maximum
            width = self.sigmax * _np.sqrt(-2*_np.log(0.5))
            height = self.sigmay * _np.sqrt(-2*_np.log(0.5))
            ellipse = _patches.Ellipse(
                xy=(self.meanx, self.meany),
                width=width, height=height, angle=0,
                linewidth=1, edgecolor=color, fill='false', facecolor='none')
            axis.add_patch(ellipse)
        if None not in (self.roix, self.roiy):
            roix1, roix2 = self.roix
            roiy1, roiy2 = self.roiy
            w, h = _np.abs(roix2-roix1), _np.abs(roiy2-roiy1)
            rect = _patches.Rectangle(
                (roix1, roiy1), w, h, linewidth=1, edgecolor=color,
                fill='False',
                facecolor='none')
            axis.add_patch(rect)

        return fig, axis

    def trim(self):
        """."""
        data = Image._trim_image(self.data, self.roix, self.roiy)
        return Image(data=data, roix=self.roix, roiy=self.roiy)

    def _update_image_roi(self, image, roix, roiy):
        """."""
        self._data = _np.asarray(image)
        self._roix = roix or [0, self._data.shape[1]]
        self._roiy = roiy or [0, self._data.shape[0]]

        # update indc
        self._indcx = None
        self._indcy = None
        if self._roix is not None:
            self._indcx = Image._get_indc(self._data, self._roix, axis=1)
        if self._roiy is not None:
            self._indcy = Image._get_indc(self._data, self._roiy, axis=0)

        # update proj, mean, sigma
        self._projx, self._meanx, self._sigmax = [None] * 3
        self._projy, self._meany, self._sigmay = [None] * 3
        if self._indcx is not None and self._indcy is not None:
            image = Image._trim_image(self._data, self._roix, self._roiy)
            self._projx = proj = _np.sum(image, axis=0)
            self._meanx = _np.sum(self._indcx * proj) / _np.sum(proj)
            self._projy = proj = _np.sum(image, axis=1)
            self._meany = _np.sum(self._indcy * proj) / _np.sum(proj)
            indc, mean, proj = self._indcx, self._meanx, self._projx
            projsum = _np.sum(proj)
            self._sigmax = _np.sqrt(_np.sum((indc - mean)**2 * proj)/projsum)
            indc, mean, proj = self._indcy, self._meany, self._projy
            projsum = _np.sum(proj)
            self._sigmay = _np.sqrt(_np.sum((indc - mean)**2 * proj)/projsum)
        
    @staticmethod 
    def _trim_image(image, roix, roiy):
        roix1, roix2 = roix
        roiy1, roiy2 = roiy
        return image[roiy1:roiy2, roix1:roix2]

    @staticmethod
    def _get_indc(image, roi, axis):
        if roi[1] <= image.shape[axis]:
            return _np.arange(image.shape[axis])[roi[0]:roi[1]]
        else:
            return None


class ImageMoments:
    """."""
    def __init__(self, imagex, imagey, arrayx, arrayy):
        """."""
        self._imagex = imagex
        self._imagey = imagey
        self._arrayx = arrayx
        self._arrayy = arrayy

    def calculate(self):
        """."""

    def calc_moments(self):
        """."""
        m1y, m2y, projy = ImageMoments._calc_moments(self.imagey, self.arrayy, axis=0)
        m1x, m2x, projx = ImageMoments._calc_moments(self.imagex, self.arrayx, axis=1)
        return m1x, m1y, m2x, m2y, projx, projy

    @staticmethod
    def _calc_moments(image, arr, axis):
        """."""
        # project intensities
        proj = _np.sum(image, axis=axis)
        proj = _np.array(proj)
        
        # calc first and second image projection moments witin roi
        moment1 = _np.sum(arr * proj)
        moment2 = _np.sqrt(_np.sum((arr - moment1)**2 * proj))
        
        return moment1, moment2, proj


class ImageFitting:
    """."""

    def __init__(self, image=None):
        self._image = image
        self._func_fit = None

    @property
    def image(self):
        """Return image."""
        return self._image

    @image.setter
    def image(self, value):
        """Set image."""
        self._image = value

    @property
    def function_proj(self):
        """Return fitting projection function."""
        return self._func_fit

    @function_proj.setter
    def function_proj(self, value):
        """Set fitting projection function.
        
        function arguments:
            first: independent variable
            second: centroid (first moment)
            third: sigma (second moment)
            fourth: amplitude
            fith and up: other function parameters
        """
        self._func_fit = value

    def fit_image_roi(
            self, roix, roiy, window):
        """Fit image.

        Parameters
        ----------
        roiy : list
            Vertical region of interest ( image index 0)
        roix : list
            Horizontal region of interest (image index 1)
        window : int
            Width of the lines that will be taken to generate the projected
            distribution where the sizes are computed, by default 4

        Returns
        -------
        win : tupple
        roi : tupple
        
        """
        image = self.image
        func_proj = self.function_proj

        # build roi x and y arrays and images
        arrayy = _np.arange(image.shape[0])[roiy]
        arrayx = _np.arange(image.shape[1])[roix]
        imagey = image[roiy, roix]
        imagex = image[roiy, roix]

        # projections fitting for roi
        roi = ImageFitting.fit_image_projections(
            imagex, imagey, arrayx, arrayy, func_proj)
        
        # build win x and y arrays and images
        roi_pfitx, roi_pfity = roi[0], roi[1]
        roi_m1y_idx = _np.argmin(_np.abs(arrayy - roi_pfity[1]))
        roi_m1x_idx = _np.argmin(_np.abs(arrayx - roi_pfitx[1]))
        vsel1 = max(int(roi_m1y_idx - window/2), 0)
        vsel2 = min(int(roi_m1y_idx + window/2), image.shape[0])
        hsel1 = max(int(roi_m1x_idx - window/2), 0)
        hsel2 = min(int(roi_m1x_idx + window/2), image.shape[1])
        imagey = image[:, hsel1:hsel2]
        imagex = image[vsel1:vsel2, :]
        
        # projections fitting for win
        win = ImageFitting.fit_image_projections(
            imagex, imagey, arrayx, arrayy, func_proj)
        
        return win, roi


    @staticmethod
    def fit_image_projections(image, func):
        """."""
        # moments as initial fitting parameters
        p0x = [image.meanx, image.sigmax, _np.max(image.projx), 0]
        p0y = [image.meany, image.sigmay, _np.max(image.projy), 0]
        pfity, pcovy = _curve_fit(func, image.indcy, image.projy, p0=p0y)
        pfitx, pcovx = _curve_fit(func, image.indcx, image.projx, p0=p0x)

        # calculate moments uncertaities
        pfitx_err = _np.sqrt(_np.diag(pcovx))
        pfity_err = _np.sqrt(_np.diag(pcovy))

        # return parameters
        res = (
            pfitx, pfity,
            pfitx_err, pfity_err,
            )

        return res
 
    @staticmethod
    def fit_parameter_centroid(fit_parameters):
        """Return image centroid, first fitted parameter."""
        return fit_parameters[0]

    @staticmethod
    def fit_parameter_sigma(fit_parameters):
        """Return image sigma, second fitted parameter."""
        return fit_parameters[1]

    @staticmethod
    def fit_parameter_amplitude(fit_parameters):
        """Return image amplitude, third fitted parameter."""
        return fit_parameters[2]

    @staticmethod
    def fit_parameter_centroid_err(fit_parameters_err):
        """Return image centroid error, first fitted parameter."""
        return fit_parameters_err[0]

    @staticmethod
    def fit_parameter_sigma_err(fit_parameters_err):
        """Return image sigma error, second fitted parameter."""
        return fit_parameters_err[1]

    @staticmethod
    def fit_parameter_amplitude_err(fit_parameters_err):
        """Return image amplitude error, third fitted parameter."""
        return fit_parameters_err[2]

    @staticmethod
    def fitfunc_gauss(u, u0, sigma, amp, off):
        """."""
        return amp*_np.exp(- (u-u0)**2 / (2 * sigma**2)) + off

