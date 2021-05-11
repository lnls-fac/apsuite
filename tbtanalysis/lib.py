import numpy as _np
from mathphys.functions import load_pickle
from apsuite.tbt_analysis import TbTAnalysis

def convert_data(fname):
    data = load_pickle(fname)
    if 'trajx' in data:
        if isinstance(data['chromx_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromx_err'])
        if isinstance(data['chromy_err'], (list, tuple)):
            data['chromx_err'] = max(data['chromy_err'])
        data['kicktype'] = 'CHROMX' if data['kicktype'] == 'X' else data['kicktype']
        data['kicktype'] = 'CHROMY' if data['kicktype'] == 'Y' else data['kicktype']
        return data
    ndata = dict()
    ndata['trajx'] = data['sofb_tbt']['x'].reshape((1, -1, 160))
    ndata['trajy'] = data['sofb_tbt']['y'].reshape((1, -1, 160))
    ndata['trajsum'] = data['sofb_tbt']['sum'].reshape((1, -1, 160))
    ndata['kicks'] = [data['kick']]
    ndata['tunex'] = data['tune']['x']
    ndata['tuney'] = data['tune']['y']
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


def create_tbt(fname, kicktype=None):
    newdata = convert_data(fname)
    tbt = TbTAnalysis(data=newdata, kicktype=kicktype)
    print('meas. chromx: {:+.4f} ± {:.4f}'.format(tbt.chromx, tbt.chromx_err))
    print('meas. chromy: {:+.4f} ± {:.4f}'.format(tbt.chromy, tbt.chromy_err))
    print()
    return tbt
    



