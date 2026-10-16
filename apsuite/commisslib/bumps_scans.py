"""."""

import time as _time
import numpy as _np
from siriuspy.devices import SOFB, HLFOFB, CurrInfoSI
from apsuite.orbcorr.si_bumps import SiCalcBumps

from ..utils import (
    ThreadedMeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)


class BumpParams(_ParamsBaseClass):
    """."""
    GRID_ORDER = ('psx', 'psy', 'agx', 'agy')
    _FMT = 'pts_{}'.format

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        self.pts_psx = _np.array([])  # 1d numpy array [um]
        self.pts_psy = _np.array([])  # 1d numpy array [um]
        self.pts_agx = _np.array([])  # 1d numpy array [urad]
        self.pts_agy = _np.array([])  # 1d numpy array [urad]
        self.subsec = '01SA'

        self.bump_n_bpms_outx = 3
        self.bump_n_bpms_outy = 3
        self.bump_minsingval = 0.2
        self.bump_residue = 3  # [um]
        self.fofb_min_kick = 2  # [urad]

        self.buffer_sloworb = 20
        self.wait_meas = 2  # [s]
        self.orbcorr_nr_iters = 5
        self.orbcorr_residue = 5  # [um]
        self.nr_orbit_verification_closed_loop = 5  # [um]

        self.sofb_closed_loop = False
        self.fofb_closed_loop = False
        self.timeout_fofb_ramp = 15.0  # [s]
        self.sleep_time = 0.5  # [s]

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:9.2f} - {2:9.2f} {3:s} with {4:1.0f} pts\n'.format
        stg = '{0:20s} = {1:>9s}\n'.format('subsec', self.subsec)

        stg += dtmp('bump_n_bpms_outx', self.bump_n_bpms_outx, '')
        stg += dtmp('bump_n_bpms_outy', self.bump_n_bpms_outy, '')
        stg += ftmp('bump_minsingval', self.bump_minsingval, '')
        stg += ftmp('bump_residue', self.bump_residue, '[um]')
        stg += ftmp('fofb_min_kick', self.fofb_min_kick, '[urad]')

        stg += dtmp('buffer_sloworb', self.buffer_sloworb, '')
        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += dtmp('orbcorr_nr_iters', self.orbcorr_nr_iters)
        stg += ftmp('orbcorr_residue', self.orbcorr_residue, '[um]')

        stg += dtmp('sofb_closed_loop', self.sofb_closed_loop)
        stg += dtmp('fofb_closed_loop', self.fofb_closed_loop)
        stg += ftmp('timeout_fofb_ramp', self.timeout_fofb_ramp, '')
        stg += ftmp('sleep_time', self.sleep_time, '')

        if not self.is_standard_grid:
            stg += '\n\n' + '#' * 70 + '\n'
            stn = ' Grid not standardized! Call `standardize_grid` to fix '
            stg += stn.center(70, '#')
            stg += '\n' + '#' * 70 + '\n'

        stg += '\nGrid Variables:\n'
        for grf in self.GRID_ORDER:
            var = getattr(self, self._FMT(grf))
            stg += f'    {self._FMT(grf):20s} = '
            if var is None or not len(var):
                stg += 'empty\n'
                continue
            stg += f'{var.min():9.2f} - {var.max():9.2f} '
            stg += 'um' if 'ps' in grf else 'urad'
            stg += f' with {var.size:4d} pts\n'
        return stg

    @property
    def is_standard_grid(self):
        """Check whether all grid variables have the same length.

        The grid variables are: pts_psx, pts_psy, pts_agx, pts_agy.
        Se docstring of `standardize_grid` for more details.

        Returns:
            bool: True if all grid variables are properly set.

        """
        grid = self._get_grid()
        grlen = [len(spn) for spn in grid if spn is not None]
        return len(grlen) == len(grid) and len(set(grlen)) == 1

    def standardize_grid(self):
        """Adjust grid variables to have the same length.

        Only the variable with larger length will not be changed.
        Shorter grid variables will be filled with their last value.
        Grid variables that are empty or `None` will be set to zero.

        Example:
            >>> obj = BumpParams()
            >>> obj.pts_psx = np.arange(4)
            >>> obj.pts_psy = np.arange(2)
            >>> obj.pts_agx = None
            >>> obj.pts_psx
            array([0, 1, 2, 3])
            >>> obj.pts_psy
            array([0, 1])
            >>> obj.pts_agx
            >>> obj.pts_agy
            array([])
            >>> obj.is_standard_grid
            False
            >>> obj.standardize_grid()
            >>> obj.pts_psx
            array([0, 1, 2, 3])
            >>> obj.pts_psy
            array([0, 1, 1, 1])
            >>> obj.pts_agx
            array([0, 0, 0, 0])
            >>> obj.pts_agy
            array([0, 0, 0, 0])
            >>> obj.is_standard_grid
            True

        """
        grid = self._get_grid()
        siz = _np.max([len(spn) for spn in grid if spn is not None])
        for ogd, spn in zip(self.GRID_ORDER, grid):
            arr = _np.zeros(siz, dtype=float)
            lng = 0 if spn is None else len(spn)
            if lng:
                arr[:lng] = spn
                arr[lng:] = spn[-1]
            setattr(self, self._FMT(ogd), arr)

    @staticmethod
    def create_meshgrid_zigzag(*arrays):
        """Create N-D Zig-Zag mesh grid from arrays.

        The code ensures that the trajectory followed is continuous, in the
        sense that if we want to map all the points of an hyper-cube in N-D
        space:
            >>> x = [1, 2, 3]
            >>> N = 5
            >>> arrays = [x, ] * N
            >>> points = np.array(create_meshgrid_zigzag(*arrays))
            >>> print(points.shape == (N, len(x) ** N)
            True
            >>> np.linalg.norm(np.diff(points, axis=1), axis=0) == 1
            True

        In the simple case of a 2-D grid, we have:
            >>> x = [1, 2, 3]
            >>> y = [1, 2, 3, 4]
            >>> X, Y = create_meshgrid_zigzag(x, y)
            >>> points = np.array([X, Y])
            >>> points
            array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]])

        Note that X varies slowly, while Y is fast.

        Args:
            *arrays: sequence of 1D arrays to create meshgrid.

        Returns:
            grids: list of numpy.ndarray with grid coordinates. Each array is
                1D, with size equal to the product of the sizes of the input
                arrays.

        """
        # NOTE: This code was done by Google AI in 2026/10/15. It takes some
        # time to digest, but if you go through the details, it makes sense.
        # Besides, I tested it using the example of the docstring and others
        # and it gives consistent results when checked for the distance
        # between successive points shown in the docstring. If you want to
        # learn more about this topic, search for "Boustrophedon trajectory".
        N = len(arrays)
        if N == 0:
            return ()

        arrays = [_np.asarray(arr) for arr in arrays]
        shapes = [arr.size for arr in arrays]

        # 1. Start from the last dimension of the list (last array)
        curr_indices = _np.arange(shapes[-1])[:, None]

        # 2. Build tree indices from bottom-up (from last to first dimension)
        for i in reversed(range(N - 1)):
            S_i = shapes[i]
            M, D = curr_indices.shape

            # Get exact memory block for repetitions
            repeated_sub = _np.empty((S_i, M, D), dtype=int)

            # Vectorized trick: even slices keep order, odd ones are inverted
            repeated_sub[0::2] = curr_indices
            # Invert whole block from first axis (M)
            repeated_sub[1::2] = curr_indices[::-1]

            # Combine slices from 3D block into 2D array
            repeated_sub = repeated_sub.reshape(-1, D)

            # Create and repeat indices corresponding to current dimension
            curr_dim_idx = _np.repeat(_np.arange(S_i), M)[:, None]

            # Stack current dimension to the left of sub-dims. already created.
            curr_indices = _np.hstack([curr_dim_idx, repeated_sub])

        # 3. Map matrix of indices back to the real values of the input arrays
        return tuple(arrays[k][curr_indices[:, k]] for k in range(N))

    def _get_grid(self):
        return [getattr(self, self._FMT(o)) for o in self.GRID_ORDER]


class Bump(_BaseClass):
    """."""

    def __init__(
        self,
        meas_func=None,
        meas_func_args=None,
        meas_func_kwargs=None,
        isonline=True
    ):
        """."""
        _BaseClass.__init__(
            self, params=BumpParams(), target=self._do_scan, isonline=isonline,
        )
        self.data = dict()
        self.bumptools = SiCalcBumps()
        self.meas_func = meas_func
        self.meas_func_args = meas_func_args or ()
        self.meas_func_kwargs = meas_func_kwargs or dict()
        if self.isonline:
            self._create_devices()

    def get_state(self):
        """Get initial state of SOFB and FOFB."""
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        init_stt = {}
        init_stt['refx'] = sofb.refx
        init_stt['refy'] = sofb.refy
        init_stt['bpmxenbl'] = sofb.bpmxenbl
        init_stt['bpmyenbl'] = sofb.bpmyenbl
        init_stt['fofb_bpmxenbl'] = fofb.bpmxenbl
        init_stt['fofb_bpmyenbl'] = fofb.bpmyenbl
        return init_stt

    def set_state(self, state):
        """Set SOFB and FOFB to a given state.

        Args:
            state (dict): dictionary with keys:
                'refx', 'refy',
                'bpmxenbl', 'bpmxenbl',
                'fofb_bpmxenbl', 'fofb_bpmxenbl'
                All keys are optional.

        """
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        if 'refx' in state:
            sofb.refx = state['refx']
        if 'refy' in state:
            sofb.refy = state['refy']
        if 'bpmxenbl' in state:
            sofb.bpmxenbl = state['bpmxenbl']
        if 'bpmyenbl' in state:
            sofb.bpmyenbl = state['bpmyenbl']

        if not self.params.fofb_closed_loop:
            return

        if 'fofb_bpmxenbl' in state:
            fofb.bpmxenbl = state['fofb_bpmxenbl']
        if 'fofb_bpmyenbl' in state:
            fofb.bpmyenbl = state['fofb_bpmyenbl']

    def update_initial_state(self):
        """Update self.data with initial state of SOFB and FOFB."""
        self.data['initial_state'] = self.get_state()

    def restore_initial_state(self):
        """."""
        init_state = self.data.get('initial_state', None)
        if init_state is None:
            print('Initial state not found.')
            return
        self.set_state(init_state)

    def config_sofb(self):
        """."""
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        if sofb.opmode != sofb.data.SOFBMode.SlowOrb:
            raise ValueError('ERR: SOFB is not in SlowOrb mode.')

        print('Configuring sofb...')
        sofb.nr_points = self.params.buffer_sloworb
        sofb.cmd_reset()
        sofb.wait_buffer(timeout=sofb.nr_points / sofb.data.BPMsFreq * 1.2)
        print('Done!')
        if self.params.sofb_closed_loop:
            sofb.cmd_turn_on_autocorr()
        else:
            sofb.cmd_turn_off_autocorr()

        if self.params.fofb_closed_loop:
            fofb.cmd_turn_on_loop_state()
        else:
            fofb.cmd_turn_off_loop_state()

    def remove_bpms(self):
        """Remove BPMs from correction system."""
        subsec = self.params.subsec
        n_bpms_outx = self.params.bump_n_bpms_outx
        n_bpms_outy = self.params.bump_n_bpms_outy
        section_type, section_nr = self._subsec_2_sectype_nr(subsec)

        sofb = self.devices['sofb']
        idcs_out = self.bumptools.get_closest_bpms_indices(
            section_type=section_type,
            sidx=section_nr - 1,
            n_bpms_outx=n_bpms_outx,
            n_bpms_outy=n_bpms_outy,
        )
        enblx = self._bpmxenbl
        enbly = self._bpmyenbl
        enblx, enbly = self._generate_bpm_enbl(
            n_bpms_outx, n_bpms_outy, enblx, enbly, idcs_out
        )
        sofb.bpmxenbl = enblx
        sofb.bpmyenbl = enbly

        if self.params.closed_loops:
            fofb = self.devices['fofb']
            enblx = self._fofb_bpmxenbl
            enbly = self._fofb_bpmyenbl
            enblx, enbly = self._generate_bpm_enbl(
                n_bpms_outx, n_bpms_outy, enblx, enbly, idcs_out
            )
            fofb.bpmxenbl = enblx
            fofb.bpmyenbl = enbly
        _time.sleep(
            self.params.sleep_time
        )  # NOTE: For some reason We have to wait here.

    def set_reforb(self, orbx, orby):
        """Update orbit of corr. sytems.

        Args:
            orbx (numpy.ndarray, (N, )): Horizontal orbit
            orby (numpy.ndarray, (N, )): Vertical orbit

        """
        if self.params.fofb_closed_loop:
            fofb = self.devices['fofb']
            fofb.refx = orbx
            fofb.refy = orby
        sofb = self.devices['sofb']
        sofb.refx = orbx
        sofb.refy = orby

    def implement_bump(self, agx=0, agy=0, psx=0, psy=0, subsec=None):
        """."""
        sofb = self.devices['sofb']
        refx0 = self.reforbx
        refy0 = self.reforby
        subsec = subsec or self.params.subsec
        n_bpms_outx = self.params.bump_n_bpms_outx
        n_bpms_outy = self.params.bump_n_bpms_outy
        minsingval = self.params.bump_minsingval

        refx, refy = sofb.si_calculate_bumps(
            refx0,
            refy0,
            subsec,
            agx=agx,
            agy=agy,
            psx=psx,
            psy=psy,
            n_bpms_outx=n_bpms_outx,
            n_bpms_outy=n_bpms_outy,
            minsingval=minsingval,
        )

        self.set_reforb(refx, refy)

        try:
            return self._apply_bump_and_wait_convergence(subsec)
        except ValueError:
            return False

    # ################### Auxiliary Methods ####################

    def _create_devices(self):
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['fofb'] = HLFOFB(HLFOFB.DEVICES.SI)
        self.devices['currinfo'] = CurrInfoSI()

    def _is_beam_alive(self):
        if self.devices['currinfo'].storedbeam:
            return True
        print('Beam was dumped!')

    def _generate_bpm_enbl(
        self, n_bpms_outx, n_bpms_outy, enblx, enbly, idcs_out
    ):
        if n_bpms_outx != 0:
            enblx[idcs_out[: n_bpms_outx * 2]] = False
        if n_bpms_outy != 0:
            enbly[idcs_out[n_bpms_outy * 2:] - 160] = False
        return enblx, enbly

    def _do_scan(self):
        """Start bumps scan."""
        prms = self.params

        grid = self._prepare_grid()


        data = list()
        for i in range(x_span.size):
            if not self._is_beam_alive():
                self.restore_initial_state()
                break
            x = x_span[i]
            y = y_span[i]

            if prms.do_angular_bumps:
                self.implement_bump(agx=x, agy=y)
                unit = 'urad'
            else:
                self.implement_bump(psx=x, psy=y)
                unit = 'um'
            _time.sleep(prms.wait_meas)
            data.append(self.get_measurement_data())
            fstr = f'(x, y) = ({x:6.1f}, {y:6.1f}) ' + unit
            print(fstr)
            data[-1]['bump'] = (x, y)
            data[-1]['orbx'] = self.devices['sofb'].orbx
            data[-1]['orby'] = self.devices['sofb'].orby
            self.data['bumps_data'] = data
            if self._stopevt.is_set():
                print('Stopping...')
                break

        print('Finished!')

    def _apply_bump_and_wait_convergence(self, subsec):
        bump_residue = self.params.bump_residue
        fofb_min_kick = self.params.fofb_min_kick
        nr_iters = self.params.orbcorr_nr_iters
        residue = self.params.orbcorr_residue
        nr_orbit_verification_closed_loop = (
            self.params.nr_orbit_verification_closed_loop
        )

        section_type, section_nr = self._subsec_2_sectype_nr(subsec)
        idcs_bpm = self.bumptools.get_bpm_indices(
            section_type=section_type, sidx=section_nr - 1
        )

        # Verify orbit correction
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']

        print('Waiting orbit...')
        fofb_ok = True
        if self.params.sofb_closed_loop:
            for _ in _np.arange(nr_orbit_verification_closed_loop):
                _time.sleep(self.params.buffer_sloworb / sofb.data.BPMsFreq)
                rms_residue = self._get_orbrms(idcs_bpm)

                fofb_ok = True
                if self.params.fofb_closed_loop:
                    kick = _np.max((
                        _np.abs(fofb.kickch_acc),
                        _np.abs(fofb.kickcv_acc),
                    ))
                    fofb_ok = kick <= fofb_min_kick
                if rms_residue <= bump_residue and fofb_ok:
                    break
        else:
            _ = sofb.correct_orbit_manually(
                nr_iters=nr_iters, residue=residue
            )
            sofb.cmd_reset()
            sofb.wait_buffer()
            rms_residue = self._get_orbrms(idcs_bpm)

        if rms_residue <= bump_residue and fofb_ok:
            print('Done!')
            return True
        print('WARN: Bump Implementation did not converge!')
        return False

    def _get_orbrms(self, idx):
        """Calculate rms of orbit distortion.

        Args:
            idx (numpy.ndarray, (2*N, )): Indices of BPMs used in bump.

        Return:
            float: RMS of orbit distortion at indices defined by `idx`.

        """
        sofb = self.devices['sofb']
        ref = _np.r_[sofb.refx, sofb.refy]
        orb = _np.r_[sofb.orbx, sofb.orby]
        dorb = (orb - ref)[idx] ** 2
        return _np.sqrt(dorb.sum())

    def _get_measurement_data(self):
        """."""
        currinfo = self.devices['currinfo']
        data = {'timestamp': _time.time(), 'stored_current': currinfo.current}
        data['measurements:'] = self.do_measurement()
        return data

    def _do_measurement(self):
        """Measurement function."""
        if self.meas_func is None:
            print('Not a measurement....')
        return self.meas_func(
            *self.meas_func_args, **self.meas_func_kwargs
        )

    @staticmethod
    def _subsec_2_sectype_nr(subsec):
        """Convert subsec string to subsec type and number.

        Args:
            subsec (str): Subsection name. example "02C1"

        Raises:
            ValueError: Must be a valid section

        Returns:
            str, int: section type and section number

        """
        section_nr = int(subsec[:2])
        if not 1 <= section_nr <= 20:
            raise ValueError('Section must be between 01..20.')
        section_type = subsec[2:]
        return section_type, section_nr
