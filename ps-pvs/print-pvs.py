#!/usr/bin/env python-sirius
"""."""

from siriuspy.search.ps_search import PSSearch
from siriuspy.namesys import SiriusPVName
from siriuspy.pwrsupply.data import PSData


def get_all_psnames():
    """."""
    pss = PSSearch()
    psnames = [SiriusPVName(psname) for psname in pss.get_psnames()]
    return psnames


def select_psnames(psgroup):
    """."""
    psnames = []
    allps = get_all_psnames()
    if psgroup.lower() == 'bo-correctors':
        for ps in allps:
            if ps.sec == 'BO' and (ps.dev == 'CH' or ps.dev == 'CV'):
                psnames.append(ps)
    return sorted(psnames)


def print_pvs(psgroup):
    """."""
    psnames = select_psnames(psgroup)
    for ps in psnames:
        psdata = PSData(ps)
        db = psdata.propty_database
        for prop in db:
            print(ps + ':' + prop)


print_pvs('bo-correctors')
