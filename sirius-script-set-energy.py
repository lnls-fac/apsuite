#!/usr/bin/env python-sirius
"""."""

import epics
# from siriuspy.search.ps_search import PSSearch as _PSSearch
from siriuspy.search.ma_search import MASearch as _MASearch
import siriuspy.magnet.util as _mutil
import siriuspy.magnet.data as _mdata
from siriuspy.namesys import SiriusPVName as _SiriusPVName


def magfunc(madata):
    """."""
    psnames = madata.psnames
    return madata.magfunc(psnames[0])


def get_manames_dict(accelerator):
    """."""
    manames = _MASearch.get_manames()
    magnets = []
    for maname in manames:
        maname = _SiriusPVName(maname)
        if maname.sec != accelerator:
            continue
        if maname.dev == 'FC':
            continue
        if maname.sub != 'Fam' and ('SD' in maname.dev or 'SF' in maname.dev):
            continue
        if 'EjeKckr' in maname.dev:
            continue
        dipname = _mutil.get_section_dipole_name(maname)
        madata = _mdata.MAData(maname=maname)
        mfunc = magfunc(madata)
        if mfunc == 'dipole':
            continue
        dstruc = {
            'maname': maname,
            'madata': madata,
        }
        magnets.append(dstruc)
    dipole = {
        'maname': dipname,
        'madata': _mdata.MAData(maname=dipname),
    }
    return dipole, magnets


def create_pvs(dipole, magnets):
    """."""
    maname = dipole['maname']
    dipole['pv_sp'] = epics.PV(maname + ':Energy-SP')
    for d in magnets:
        maname = d['maname']
        madata = d['madata']
        magf = magfunc(madata)
        if 'corrector' in magf:
            d['pv_sp'] = epics.PV(maname + ':Kick-SP')
        elif 'quadrupole' in magf:
            d['pv_sp'] = epics.PV(maname + ':KL-SP')
        elif 'sextupole' in magf:
            d['pv_sp'] = epics.PV(maname + ':SL-SP')


def get_sp(dipole, magnets):
    """."""
    print('{}: {}'.format(dipole['pv_sp'].pvname, dipole['pv_sp'].value))
    for m in magnets:
        if m['pv_sp'].connected:
            m['sp_value'] = m['pv_sp'].value
        else:
            print('get_sp: {} disconnected'.format(m['pv_sp'].pvname))
            m['sp_value'] = None


def put_sp(dipole, magnets):
    """."""
    get_sp(dipole, magnets)
    for m in magnets:
        value = m['sp_value']
        if value is not None and m['pv_sp'].connected:
            m['pv_sp'].value = value
        else:
            print('put_sp: {} disconnected'.format(m['pv_sp'].pvname))
        print('{}: {}'.format(m['pv_sp'].pvname, m['pv_sp'].value))

def set_energy(energy, dipole, magnets):
    """."""
    read_sp(dipole, magnets)

    print('{}: {}'.format(dipole['pv_sp'].pvname, dipole['pv_sp'].value))
    for m in magnets:
        print('{}: {}'.format(m['pv_sp'].pvname, m['pv_sp'].value))

bo_dipole, bo_magnets = get_manames_dict('BO')
create_pvs(bo_dipole, bo_magnets)
read_sp(bo_dipole, bo_magnets)
