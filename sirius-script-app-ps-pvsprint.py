#!/usr/bin/env python-sirius
"""."""

import sys
from siriuspy.search.ps_search import PSSearch
from siriuspy.namesys import SiriusPVName
from siriuspy.pwrsupply.data import PSData


def get_all_psnames():
    """."""
    pss = PSSearch()
    # psn = pss.get_psnames() + li_psnames
    psn = pss.get_psnames()
    psnames = [SiriusPVName(psname) for psname in psn]
    return psnames


def select_psnames(psgroup):
    """."""
    psnames = []
    allps = get_all_psnames()
    if psgroup in allps:
        psnames.append(psgroup)
    elif psgroup.lower() == 'bo':
        for ps in allps:
            if ps.sec == 'BO':
                psnames.append(ps)
    if psgroup.lower() == 'bo-correctors':
        for ps in allps:
            if ps.sec == 'BO' and (ps.dev == 'CH' or ps.dev == 'CV'):
                psnames.append(ps)
    elif psgroup.lower() == 'li':
        for ps in allps:
            if ps.sec == 'LA' and ps.sub == 'CN':
                psnames.append(ps)
    elif psgroup.lower() in ('li-dipole', 'li-spectrometer'):
        for ps in allps:
            if ps.sec == 'LA' and ps.sub == 'CN' and ps.dis == 'H1DPPS':
                psnames.append(ps)
    elif psgroup.lower() in ('li-quadrupoles',):
        for ps in allps:
            if ps.sec == 'LA' and ps.sub == 'CN' and 'QPS' in ps.dis:
                psnames.append(ps)
    elif psgroup.lower() in ('li-correctors',):
        for ps in allps:
            if ps.sec == 'LA' and ps.sub == 'CN' and 'CPS' in ps.dis:
                psnames.append(ps)
    return sorted(psnames)


def print_pvs(psnames):
    """."""
    for ps in psnames:
        psdata = PSData(ps)
        db = psdata.propty_database
        for prop in db:
            print(ps + ':' + prop)


def run():
    """."""
    args = sys.argv[1:]
    if len(args) == 0:
        args = ['all']
    psnames = []
    for arg in args:
        psnames += select_psnames(arg)
    psnames = sorted(psnames)
    print_pvs(psnames)


if __name__ == '__main__':
    run()
