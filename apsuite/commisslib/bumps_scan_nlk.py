"""."""
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, BunchbyBunch, PowerSupplyPU, BPM
from siriuspy.search import BPMSearch as _BPMSearch
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
from siriuspy.sofb.utils import si_calculate_bump

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BumpNLKParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        self.posx_min = 0  # [um]
        self.posx_max = 0  # [um]
        self.nr_stepsx = 1
        self.posy_min = 0  # [um]
        self.posy_max = 0  # [um]
        self.nr_stepsy = 1
        self.nlk_kick = 0  # [urad]
        self.acq_nrsamples_pre = 0
        self.acq_nrsamples_post = 2000
        self.filename = ''
        self.save_data = True

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:}\n'.format
        stg = ftmp('posx_min', self.posx_min, '[um]')
        stg += ftmp('posx_max', self.posx_max, '[um]')
        stg += dtmp('nr_stepsx', self.nr_stepsx)
        stg += ftmp('posy_min', self.posy_min, '[um]')
        stg += ftmp('posy_max', self.posy_max, '[um]')
        stg += dtmp('nr_stepsy', self.nr_stepsy)
        stg += ftmp('nlk_kick', self.nlk_kick, '[urad]')
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre)
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post)
        stg += stmp('filename', self.filename)
        stg += dtmp('save_data', self.save_data)
        return stg


class BumpNLK(_BaseClass):
    """."""

    DEFAULT_SS = '01SA'

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            params=BumpNLKParams(), target=self._do_measure, isonline=isonline)
        self.data = dict(measure=dict(), analysis=dict())
        if self.isonline:
            self._bpmsnames = _BPMSearch.get_names({'sec': 'SI', 'dev': 'BPM'})
            self._create_bpms()
            self._csbpm = self.devices[self._bpmsnames[0]].csdata
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
            self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
            self.devices['nlk'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
            self._configdb = _ConfigDBClient(config_type='si_orbit')
            self.reforb = self._configdb.get_config_value('ref_orb')
            self.reforbx = _np.array(self.reforb['x'])
            self.reforby = _np.array(self.reforb['y'])

    def _create_bpms(self):
        """."""
        properties = BPM(self._bpmsnames[0]).auto_monitor_status()
        self._bpms = dict()
        for name in self._bpmsnames:
            bpm = BPM(name)
            for ppt in properties:
                bpm.set_auto_monitor(ppt, False)
            self._bpms[name] = bpm
        self.devices.update(self._bpms)

    def set_nr_samples_pre(self, value):
        """."""
        value = int(value)
        for bpm in self._bpmsnames:
            self.devices[bpm].acq_nrsamples_pre = value

    def set_nr_samples_post(self, value):
        """."""
        value = int(value)
        for bpm in self._bpmsnames:
            self.devices[bpm].acq_nrsamples_post = value

    def get_trajectory_bpms(self):
        """."""
        bpms = self._bpmsnames
        nrpts = len(self.devices[bpms[0]].mt_posx)
        trajx = _np.zeros((len(bpms), nrpts))
        trajy = _np.zeros((len(bpms), nrpts))
        for num, bpm in enumerate(bpms):
            trajx[num, :] = self.devices[bpm].mt_posx
            trajy[num, :] = self.devices[bpm].mt_posy
        return trajx, trajy

    def get_measurement_data(self):
        """."""
        bbbh, bbbv = self.devices['bbbh'], self.devices['bbbv']
        trajx, trajy = self.get_trajectory_bpms()
        data = {
            'timestamp': _time.time(),
            'stored_current': bbbh.dcct.current,
            'spec_magh': bbbh.sram.spec_mag,
            'spec_datah': bbbh.sram.spec_data,
            'spec_freqh': bbbh.sram.spec_freq,
            'spec_mark1h': bbbh.sram.spec_marker1_mag,
            'trajx': trajx,

            'spec_magv': bbbv.sram.spec_mag,
            'spec_datav': bbbv.sram.spec_data,
            'spec_freqv': bbbv.sram.spec_freq,
            'spec_mark1v': bbbv.sram.spec_marker1_mag,
            'trajy': trajy,
            }
        return data

    def implement_bump(
            self, refx0=None, refy0=None,
            agx=0, agy=0, psx=0, psy=0, nr_iters=5, residue=1):
        """."""
        sofb = self.devices['sofb']
        refx0 = refx0 or self.reforbx
        refy0 = refy0 or self.reforby
        nrefx, nrefy = si_calculate_bump(
                refx0, refy0, BumpNLK.DEFAULT_SS,
                agx=agx, agy=agy, psx=psx, psy=psy)
        sofb.refx, sofb.refy = nrefx, nrefy
        sofb.correct_orbit_manually(nr_iters=nr_iters, residue=residue)

    def _do_measure(self):
        prms = self.params
        posx_span = _np.linspace(prms.posx_min, prms.posx_max, prms.nr_stepsx)
        posy_span = _np.linspace(prms.posy_min, prms.posy_max, prms.nr_stepsy)
        idy, idx = _np.meshgrid(range(prms.nr_pointsy), range(prms.nr_pointsx))
        idy[1::2] = _np.flip(idy[1::2])
        idx, idy = idx.ravel(), idy.ravel()

        self.set_nr_samples_pre(value=prms.acq_nrsamples_pre)
        self.set_nr_samples_post(value=prms.acq_nrsamples_post)

        nlk = self.devices['nlk']
        nlk.cmd_turn_on()
        nlk.strength = prms.nlk_kick
        nlk.cmd_turn_on_pulse()
        # use wait on strength
        _time.sleep(5)

        # go to initial bump configuration
        self.implement_bump(psx=posx_span[idx[0]], psy=posy_span[idy[0]])
        data = list()
        for iter in range(idx.size):
            posx = posx_span[idx[iter]]
            posy = posy_span[idy[iter]]
            self.implement_bump(psx=posx, psy=posy, nr_iters=3)
            fstr = 'posx = {posx:6.1f} um, posy = {posy:6.1f} um'
            print(fstr)
            data.append(self.get_measurement_data())
            data[-1]['bump'] = (posx, posy)
            self.data = data
            if prms.save_data:
                self.save_data(fname=prms.filename, overwrite=True)
                print('Data saved!')

        # return to initial ref_orbit
        self.implement_bump(psx=0, psy=0, nr_iters=10)
