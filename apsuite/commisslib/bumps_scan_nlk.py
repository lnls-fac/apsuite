"""."""
import time as _time
import numpy as _np
from siriuspy.devices import SOFB, BunchbyBunch, PowerSupplyPU, BPM
from siriuspy.search import BPMSearch
from siriuspy.clientconfigdb import ConfigDBClient
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
        self.nr_steps_x = 1
        self.posy_min = 0  # [um]
        self.posy_max = 0  # [um]
        self.nr_steps_y = 1
        self.nlk_kick = 0  # [urad]
        self.filename = ''

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:}\n'.format
        stg = ftmp('posx_min', self.posx_min, '[um]')
        stg += ftmp('posx_max', self.posx_max, '[um]')
        stg += dtmp('nr_steps_x', self.nr_steps_x)
        stg += ftmp('posy_min', self.posy_min, '[um]')
        stg += ftmp('posy_max', self.posy_max, '[um]')
        stg += dtmp('nr_steps_y', self.nr_steps_y)
        stg += ftmp('nlk_kick', self.nlk_kick, '[urad]')
        stg += stmp('filename', self.filename)
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
            self._create_bpms()
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
            self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
            self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
            self.devices['nlk'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
            self.configdb = ConfigDBClient(config_type='si_orbit')
            self.reforb = self.configdb.get_config_value('ref_orb')

    def _create_bpms(self):
        """."""
        bpmsnames = BPMSearch.get_names({'sec': 'SI', 'dev': 'BPM'})
        properties = BPM(bpmsnames[0]).auto_monitor_status()
        for name in bpmsnames:
            bpm = BPM(name)
            for ppt in properties:
                bpm.set_auto_monitor(ppt, False)
            self._bpms[name] = bpm
        self.devices.update(self._bpms)

    def get_measurement_data(self, plane):
        """."""
        bbbtype = 'bbbh' if plane.upper() == 'H' else 'bbbv'
        bbb = self.devices[bbbtype]
        data = {
            'timestamp': _time.time(),
            'stored_current': bbb.dcct.current,
            'spec_mag': bbb.sram.spec_mag,
            'spec_data': bbb.sram.spec_data,
            'spec_freq': bbb.sram.spec_freq,
            'spec_mark1': bbb.sram.spec_marker1_mag,
            }
        return data

    def implement_bump(
            self, refx0=None, refy0=None,
            agx=0, agy=0, psx=0, psy=0, nr_iters=5, residue=1):
        """."""
        sofb = self.devices['sofb']
        refx0 = refx0 or _np.array(self.reforb['x'])
        refy0 = refy0 or _np.array(self.reforb['y'])
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

        nlk = self.devices['nlk']
        nlk.cmd_turn_on()
        nlk.strength = prms.nlk_kick
        _time.sleep(5)
        nlk.cmd_turn_on_pulse()

        # go to initial bump configuration
        self.implement_bump(psx=posx_span[idx[0]], psy=posy_span[idy[0]])
        data, datah, datav = dict(), list(), list()
        for iter in range(idx.size):
            px_ = posx_span[idx[iter]]
            py_ = posy_span[idy[iter]]
            self.implement_bump(psx=px_, psy=py_, nr_iters=3)
            datah.append(self.get_measurement_data(plane='H'))
            datav.append(self.get_measurement_data(plane='V'))
            data['horizontal'] = datah
            data['vertical'] = datav
            self.save_data(fname=prms.filename, overwrite=True)
            print('Data saved!')

        self.implement_bump(psx=0, psy=0)
