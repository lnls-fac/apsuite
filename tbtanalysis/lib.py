import numpy as _np
from mathphys.functions import load_pickle
from apsuite.tbt_analysis import TbTAnalysis

def multibunch_kick_spread(kick):
    """."""
    rev = 1.7e-6
    kickx_width = 2 * rev
    kicky_width = 3 * rev
    bunch_half_duration = 50e-9 / 2
    time = bunch_half_duration
    percentx = 100*(_np.cos((_np.pi/2)*time/(kickx_width/2)) - 1)
    percenty = 100*(_np.cos((_np.pi/2)*time/(kicky_width/2)) - 1)
    print('kickx spread: {} %'.format(percentx))
    print('kicky spread: {} %'.format(percenty))
    print('kickx spread: {} urad'.format(kick * percentx / 100))
    print('kicky spread: {} urad'.format(kick * percenty / 100))

def convert_data(fname, correct_chrom=True):
    data = load_pickle(fname)
    if 'trajx' in data:
        if isinstance(data['chromx_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromx_err'])
        if isinstance(data['chromy_err'], (list, tuple)):
            data['chromy_err'] = max(data['chromy_err'])
        data['kicktype'] = 'CHROMX' if data['kicktype'] == 'X' else data['kicktype']
        data['kicktype'] = 'CHROMY' if data['kicktype'] == 'Y' else data['kicktype']
        ndata = data
    else:
        ndata = dict()
        ndata['trajx'] = data['sofb_tbt']['x'].reshape((1, -1, 160))
        ndata['trajy'] = data['sofb_tbt']['y'].reshape((1, -1, 160))
        ndata['trajsum'] = data['sofb_tbt']['sum'].reshape((1, -1, 160))
        ndata['kicks'] = [data['kick']]
        ndata['tunex'] = data['tune']['x']
        ndata['tuney'] = data['tune']['y']
    if correct_chrom:
        # discutir!
        tbt = TbTAnalysis
        if 'chromx' in ndata:
            ndata['chromx'] = ndata['chromx'] - tbt.NOM_HARMONIC_NR * tbt.NOM_ALPHA
            ndata['chromy'] = ndata['chromy'] - tbt.NOM_HARMONIC_NR * tbt.NOM_ALPHA
    return ndata

def calc_param_stats(param, cutoff):

    param = _np.array(param)
    stdval = _np.std(param)
    meanval = _np.median(param)
    filtered = (abs(param - meanval) <= cutoff*stdval)
    filtered_out = (abs(param - meanval) > cutoff*stdval)
    param_mean = _np.mean(param[filtered])
    param_std = _np.std(param[filtered])

    return filtered, filtered_out, param_mean, param_std


def create_tbt(fname, kicktype=None, correct_chrom=True):
    newdata = convert_data(fname, correct_chrom)
    tbt = TbTAnalysis(data_fname=fname, data=newdata, kicktype=kicktype)
    print('meas. chromx : {:+.4f} ± {:.4f}'.format(tbt.chromx, tbt.chromx_err))
    print('meas. chromy : {:+.4f} ± {:.4f}'.format(tbt.chromy, tbt.chromy_err))
    if 'tunex' in tbt.data:
        print('meas. tunex  : {:+.6f} ± {:.6f}'.format(tbt.data['tunex'], 0.0))
    if 'tuney' in tbt.data:
        print('meas. tuney  : {:+.6f} ± {:.6f}'.format(tbt.data['tuney'], 0.0))
    if 'tunex_excitation_sts' in tbt.data:
        print('exc. tunex   : {}'.format(tbt.data['tunex_excitation_sts']))
    if 'tuney_excitation_sts' in tbt.data:
        print('exc. tuney   : {}'.format(tbt.data['tuney_excitation_sts']))

    print()
    return tbt
    



