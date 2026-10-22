"""."""

import time as _time
import numpy as _np
from siriuspy.devices import SOFB, HLFOFB, CurrInfoSI
from apsuite.orbcorr.si_bumps import SICalcBumps

from ..utils import (
    ThreadedMeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)


class BumpScanParams(_ParamsBaseClass):
    """."""
    GRID_ORDER = ('psx', 'psy', 'agx', 'agy')
    _FMT = 'pts_{}'.format

    def __init__(self):
        """."""
        super().__init__()
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
        self.orbcorr_nr_iters = 10

        self.sofb_closed_loop = False
        self.fofb_closed_loop = False
        self.timeout_fofb_ramp = 15.0  # [s]
        self.sleep_time = 0.5  # [s]

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stg = '{0:20s} = {1:>9s}\n'.format('subsec', self.subsec)

        stg += dtmp('bump_n_bpms_outx', self.bump_n_bpms_outx, '')
        stg += dtmp('bump_n_bpms_outy', self.bump_n_bpms_outy, '')
        stg += ftmp('bump_minsingval', self.bump_minsingval, '')
        stg += ftmp('bump_residue', self.bump_residue, '[um]')
        stg += ftmp('fofb_min_kick', self.fofb_min_kick, '[urad]')

        stg += dtmp('buffer_sloworb', self.buffer_sloworb, '')
        stg += ftmp('wait_meas', self.wait_meas, '[s]')
        stg += dtmp('orbcorr_nr_iters', self.orbcorr_nr_iters)

        stg += dtmp('sofb_closed_loop', self.sofb_closed_loop)
        stg += dtmp('fofb_closed_loop', self.fofb_closed_loop)
        stg += ftmp('timeout_fofb_ramp', self.timeout_fofb_ramp, '')
        stg += ftmp('sleep_time', self.sleep_time, '')

        if not self.is_standard_grid:
            stg += '\n\n' + '#' * 70 + '\n'
            stn = ' Grid not standardized! Call `standardize_grid()` to fix '
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
            >>> obj = BumpScanParams()
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
        ndims = len(arrays)
        if ndims == 0:
            return ()

        arrays = [_np.asarray(arr) for arr in arrays]
        shapes = [arr.size for arr in arrays]

        # 1. Start from the last dimension of the list (last array)
        curr_indices = _np.arange(shapes[-1])[:, None]

        # 2. Build tree indices from bottom-up (from last to first dimension)
        for i in reversed(range(ndims - 1)):
            shape_i = shapes[i]
            prev_size, cum_dims = curr_indices.shape

            # Get exact memory block for repetitions
            cum_indices = _np.empty((shape_i, prev_size, cum_dims), dtype=int)

            # Vectorized trick: even slices keep order, odd ones are inverted
            cum_indices[0::2] = curr_indices
            # Invert whole block from first axis (prev_size)
            cum_indices[1::2] = curr_indices[::-1]

            # Combine slices from 3D block into 2D array
            cum_indices = cum_indices.reshape(-1, cum_dims)

            # Create and repeat indices corresponding to current dimension
            curr_dim_indices = _np.repeat(_np.arange(shape_i), prev_size)

            # Stack current dimension to the left of sub-dims. already created.
            curr_indices = _np.hstack([curr_dim_indices[:, None], cum_indices])

        # 3. Map matrix of indices back to the real values of the input arrays
        return tuple(arrays[k][curr_indices[:, k]] for k in range(ndims))

    def _get_grid(self):
        return [getattr(self, self._FMT(o)) for o in self.GRID_ORDER]


class BumpScan(_BaseClass):
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
            self,
            params=BumpScanParams(),
            target=self._do_scan,
            isonline=isonline,
        )
        self.data = dict()
        self.bumptools = SICalcBumps()
        self.meas_func = meas_func
        self.meas_func_args = meas_func_args or ()
        self.meas_func_kwargs = meas_func_kwargs or dict()
        if self.isonline:
            self._create_devices()

    def get_state(self):
        """Get initial state of SOFB and FOFB."""
        sofb = self.devices['sofb']
        init_stt = {}
        init_stt['sofb_nr_points'] = sofb.nr_points
        init_stt['sofb_refx'] = sofb.refx
        init_stt['sofb_refy'] = sofb.refy
        init_stt['sofb_bpmxenbl'] = sofb.bpmxenbl
        init_stt['sofb_bpmyenbl'] = sofb.bpmyenbl
        init_stt['sofb_loop_state'] = sofb.autocorrsts

        if not self.params.fofb_closed_loop:
            return init_stt
        fofb = self.devices['fofb']
        init_stt['fofb_refx'] = fofb.refx
        init_stt['fofb_refy'] = fofb.refy
        init_stt['fofb_bpmxenbl'] = fofb.bpmxenbl
        init_stt['fofb_bpmyenbl'] = fofb.bpmyenbl
        init_stt['fofb_loop_state'] = fofb.loop_state
        return init_stt

    def set_state(self, state):
        """Set SOFB and FOFB to a given state.

        Args:
            state (dict): dictionary with keys:
                    'sofb_nr_points',
                    'sofb_refx', 'sofb_refy',
                    'sofb_bpmxenbl', 'sofb_bpmyenbl',
                    'sofb_loop_state',
                    'fofb_refx', 'fofb_refy',
                    'fofb_bpmxenbl', 'fofb_bpmyenbl',
                    'fofb_loop_state'
                All keys are optional.

        """
        # First turn off loops to set parameters
        if 'fofb_loop_state' in state:
            if not self._handle_fofb_loop_state(False):
                raise RuntimeError('FOFB loop state could not be turned off.')
        if 'sofb_loop_state' in state:
            if not self._handle_sofb_loop_state(False):
                raise RuntimeError('SOFB loop state could not be turned off.')

        prps = ['refx', 'refy', 'bpmxenbl', 'bpmyenbl', 'nr_points']
        # Set SOFB parameters
        sofb = self.devices['sofb']
        for prp in prps:
            key = f'sofb_{prp}'
            if key in state:
                setattr(sofb, prp, state[key])

        # Set FOFB parameters
        fofb = self.devices['fofb']
        for prp in prps[:-1]:  # FOFB does not have nr_points
            key = f'fofb_{prp}'
            if key in state:
                setattr(fofb, prp, state[key])

        # Configure loop states.
        if 'sofb_loop_state' in state:
            if not self._handle_sofb_loop_state(state['sofb_loop_state']):
                raise RuntimeError('SOFB loop state could not be set.')
        if 'fofb_loop_state' in state:
            if not self._handle_fofb_loop_state(state['fofb_loop_state']):
                raise RuntimeError('FOFB loop state could not be set.')

    # ################### Auxiliary Methods ####################

    def _do_scan(self):
        """Start bumps scan."""
        prms = self.params

        if not prms.is_standard_grid:
            raise ValueError('Grid is not configured properly.')

        print('Getting initial state')
        state = self.get_state()
        self.data['initial_state'] = state
        refx = state['sofb_refx']
        refy = state['sofb_refy']

        print('Configuring SOFB...')
        sofb = self.devices['sofb']
        if sofb.opmode != sofb.data.SOFBMode.SlowOrb:
            raise ValueError('ERR: SOFB is not in SlowOrb mode.')

        enbls = self._calc_bpmenbl_list(
            sofb_enblx=sofb.bpmxenbl, sofb_enbly=sofb.bpmyenbl,
            fofb_enblx=self.devices['fofb'].bpmxenbl,
            fofb_enbly=self.devices['fofb'].bpmyenbl
        )
        sofb_enblx, sofb_enbly, fofb_enblx, fofb_enbly = enbls
        self.set_state({
            'sofb_bpmxenbl': sofb_enblx, 'sofb_bpmyenbl': sofb_enbly,
            'fofb_bpmxenbl': fofb_enblx, 'fofb_bpmyenbl': fofb_enbly,
            'sofb_nr_points': prms.buffer_sloworb,
            'sofb_loop_state': prms.sofb_closed_loop,
            'fofb_loop_state': prms.fofb_closed_loop,
        })

        print('Starting bumps scan...')
        data = list()
        for i in range(prms.pts_psx.size):
            if not self._is_beam_alive():
                self.restore_initial_state()
                break
            psx = prms.pts_psx[i]
            psy = prms.pts_psy[i]
            agx = prms.pts_agx[i]
            agy = prms.pts_agy[i]
            self._implement_bump(
                refx=refx, refy=refy, psx=psx, psy=psy, agx=agx, agy=agy
            )

            _time.sleep(prms.wait_meas)
            data.append(self._call_meas_func())
            fstr = f'(x, y) = ({psx:6.1f}, {psy:6.1f}) um  and  '
            fstr += f"(x', y') = ({agx:6.1f}, {agy:6.1f}) urad"
            print(fstr)
            data[-1]['bump'] = (psx, psy, agx, agy)
            data[-1]['refx'] = self.devices['sofb'].refx
            data[-1]['refy'] = self.devices['sofb'].refy
            data[-1]['orbx'] = self.devices['sofb'].orbx
            data[-1]['orby'] = self.devices['sofb'].orby
            self.data['bumps_data'] = data
            if self._stopevt.is_set():
                print('Stopping...')
                break

        self.set_state(state)
        print('Finished!')

    def _implement_bump(
        self, refx, refy, psx=0, psy=0, agx=0, agy=0, subsec=None
    ):
        """."""
        sofb = self.devices['sofb']
        subsec = subsec or self.params.subsec

        refx, refy = sofb.si_calculate_bumps(
            refx,
            refy,
            subsec,
            agx=agx,
            agy=agy,
            psx=psx,
            psy=psy,
            n_bpms_outx=self.params.bump_n_bpms_outx,
            n_bpms_outy=self.params.bump_n_bpms_outy,
            minsingval=self.params.bump_minsingval,
        )

        state = {
            'sofb_refx': refx, 'sofb_refy': refy,
            'fofb_refx': refx, 'fofb_refy': refy,
            'fofb_loop_state': self.params.fofb_closed_loop,
        }
        self.set_state(state)

        return self._apply_bump_and_wait_convergence(subsec)

    def _apply_bump_and_wait_convergence(self, subsec):
        bump_residue = self.params.bump_residue
        fofb_min_kick = self.params.fofb_min_kick
        nr_iters = self.params.orbcorr_nr_iters

        section_type, section_nr = self._subsec_2_sectype_nr(subsec)
        idcs_bpm = self.bumptools.get_bpm_indices(
            section_type=section_type, sidx=section_nr - 1
        )

        # Verify orbit correction
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']

        print('Waiting orbit...')
        for _ in _np.arange(nr_iters):
            if self.params.sofb_closed_loop:
                _time.sleep(1 / sofb.loop_freq * 1.1)
            else:
                sofb.correct_orbit_manually(nr_iters=1, residue=bump_residue)
                sofb.wait_buffer()
            rms_residue = self._get_orbrms(idcs_bpm)

            fofb_ok = True
            if self.params.fofb_closed_loop:
                kick = _np.max((
                    _np.abs(fofb.kickch_acc),
                    _np.abs(fofb.kickcv_acc),
                ))
                fofb_ok = kick <= fofb_min_kick
            if rms_residue <= bump_residue and fofb_ok:
                print('Done!')
                return True

        print('WARN: Bump Implementation did not converge!')
        return False

    def _calc_bpmenbl_list(
        self,
        sofb_enblx=None,
        sofb_enbly=None,
        fofb_enblx=None,
        fofb_enbly=None
    ):
        """Remove BPMs from correction system."""
        subsec = self.params.subsec
        n_bpms_outx = self.params.bump_n_bpms_outx
        n_bpms_outy = self.params.bump_n_bpms_outy
        section_type, section_nr = self._subsec_2_sectype_nr(subsec)

        idcs_out = self.bumptools.get_bpms_indices_to_remove(
            section_type=section_type,
            sidx=section_nr - 1,
            n_bpms_outx=n_bpms_outx,
            n_bpms_outy=n_bpms_outy,
        )
        sofb = self.devices['sofb']
        fofb = self.devices['fofb']
        sofb_enblx = sofb.bpmxenbl if sofb_enblx is None else sofb_enblx
        sofb_enbly = sofb.bpmyenbl if sofb_enbly is None else sofb_enbly
        sofb_enblx[idcs_out[: n_bpms_outx * 2]] = False
        sofb_enbly[idcs_out[n_bpms_outx * 2:] - 160] = False

        fofb_enblx = fofb.bpmxenbl if fofb_enblx is None else fofb_enblx
        fofb_enbly = fofb.bpmyenbl if fofb_enbly is None else fofb_enbly
        fofb_enblx[idcs_out[: n_bpms_outx * 2]] = False
        fofb_enbly[idcs_out[n_bpms_outx * 2:] - 160] = False
        return sofb_enblx, sofb_enbly, fofb_enblx, fofb_enbly

    def _handle_fofb_loop_state(self, state):
        state = bool(state)
        fofb = self.devices['fofb']
        if state == bool(fofb.loop_state):
            return True
        if state:
            return fofb.cmd_turn_on_loop_state()
        if not fofb.cmd_turn_off_loop_state():
            return False
        if not fofb.cmd_corr_accclear():
            return False
        if not fofb.cmd_corr_set_current_zero():
            return False
        return True

    def _handle_sofb_loop_state(self, state):
        state = bool(state)
        sofb = self.devices['sofb']
        if state == bool(sofb.autocorrsts):
            return True
        if state:
            return sofb.cmd_turn_on_loop_state()
        return sofb.cmd_turn_off_loop_state()

    def _create_devices(self):
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['fofb'] = HLFOFB(HLFOFB.DEVICES.SI)
        self.devices['currinfo'] = CurrInfoSI()

    def _is_beam_alive(self):
        if self.devices['currinfo'].storedbeam:
            return True
        print('Beam was dumped!')

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
        return _np.sqrt(dorb.sum() / dorb.size)

    def _call_meas_func(self):
        """."""
        if self.meas_func is None or not callable(self.meas_func):
            print('Measurement function not defined. Skipping ...')
            return
        return {
            'timestamp': _time.time(),
            'stored_current': self.devices['currinfo'].current,
            'measurements': self.meas_func(
                *self.meas_func_args, **self.meas_func_kwargs
            ),
        }

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
