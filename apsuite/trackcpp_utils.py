import subprocess as _subprocess
import numpy as _np
import os as _os

_full = lambda x: _os.path.sep.join(x)
default_track_version = 'trackcpp'
_commom_keys = ['flat_filename','energy','harmonic_number',
                'cavity_state','radiation_state','vchamber_state']


def _prepare_args(dynap_type, mand_keys, **kwargs):

    args = [kwargs.pop('track_version',default_track_version),dynap_type]
    for key in mand_keys:
        args.append(str(kwargs.pop(key)))
    if kwargs:
        print('Keys : '+', '.join(sorted(kwargs.keys()))+ ' were not used.')
    return args

# -- dynap_xy --
def load_dynap_xy(path, var_plane='x'):
    # Carrego os dados:
    nr_header_lines = 13
    fname = _full([path, 'dynap_xy_out.txt'])
    turn,plane,x,y = _np.loadtxt(fname,skiprows=nr_header_lines,usecols=(1,3,5,6),unpack=True)

    # Identifico quantos x e y existem:
    nx = len(_np.unique(x))
    ny = x.shape[0]//nx

    # Redimensiono para que todos os x iguais fiquem na mesma coluna:
    # o flipud é usado porque y é decrescente:
    fun = lambda x: _np.flipud(x.reshape((nx,ny)).T)
    turn, plane, x, y = fun(turn), fun(plane), fun(x), fun(y)
    dados = dict(x=x,y=y,plane=plane,turn=turn)

    # E identifico a borda da DA:
    if var_plane =='y':
        lost = plane != 0
        ind = lost.argmax(axis=0)
        # Caso a abertura vertical seja maior que o espaço calculado:
        anyloss = lost.any(axis=0)
        ind = ind*anyloss + (~anyloss)*(y.shape[0]-1)

        # por fim, defino a DA:
        h = x[0]
        v = y[:,0][ind]
        aper = _np.vstack([h,v])
        area = _np.trapz(v,x=h)
    else:
        idx  = x > 0
        # para x negativo:
        x_mi     = _np.fliplr(x[~idx].reshape((ny,-1)))
        plane_mi = _np.fliplr(plane[~idx].reshape((ny,-1)))
        lost  = plane_mi != 0
        ind_neg = lost.argmax(axis=1)
        # Caso a abertura horizontal seja maior que o espaço calculado:
        anyloss = lost.any(axis=1)
        ind_neg = ind_neg*anyloss + (~anyloss)*(x_mi.shape[1]-1)

        h_neg = x_mi[0][ind_neg]
        v_neg = y[:,0]
        aper_neg = _np.vstack([h_neg,v_neg])
        area_neg = _np.trapz(h_neg,x=v_neg)

        #para x positivo
        x_ma = x[idx].reshape((ny,-1))
        plane_ma = plane[idx].reshape((ny,-1))
        lost    = plane_ma != 0
        ind_pos = lost.argmax(axis=1)
        # Caso a abertura horizontal seja maior que o espaço calculado:
        anyloss = lost.any(axis=1)
        ind_pos = ind_pos*anyloss + (~anyloss)*(x_ma.shape[1]-1)

        # por fim, defino a DA em x positivo:
        h_pos = x_ma[0][ind_pos]
        v_pos = y[:,0]
        aper_pos = _np.fliplr(_np.vstack([h_pos,v_pos]))
        area_pos = _np.trapz(h_pos,x=v_pos)

        aper = _np.hstack([aper_neg,aper_pos])
        area = -_np.trapz(aper[0],x=aper[1])

    return aper, area, dados
def dynap_xy(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['de', 'nr_turns', 'x_nrpts', 'x_min', 'x_max',
                      'y_nrpts', 'y_min', 'y_max', 'nr_threads'])
    dynap = 'dynap_xy'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    _subprocess.call(args)

# -- dynap_ex --
def load_dynap_ex(path):
    # Carrego os dados:
    nr_header_lines = 13
    fname = _full([path, 'dynap_ex_out.txt'])
    turn,plane,x,en = _np.loadtxt(fname,skiprows=nr_header_lines,usecols=(1,3,5,7),unpack=True)

    # Identifico quantos x e y existem:
    ne = len(_np.unique(x))
    nx = x.shape[0]//ne

    # Redimensiono para que todos os x iguais fiquem na mesma linha:
    fun = lambda x: x.reshape((nx,ne)).T
    turn, plane, x, en = fun(turn), fun(plane), fun(x), fun(en)
    dados = dict(x=x,en=en,plane=plane,turn=turn)

    lost = plane != 0
    ind = lost.argmax(axis=0)
    # Caso a abertura horizontal seja maior que o espaço calculado:
    anyloss = lost.any(axis=0)
    ind = ind*anyloss + (~anyloss)*(x.shape[0]-1)

    # por fim, defino a DA:
    h = en[0]
    v = x[:,0][ind]
    aper = _np.vstack([h,v])

    return aper, dados
def dynap_ex(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['y','nr_turns','e_nrpts','e_min','e_max',
                      'x_nrpts','x_min','x_max','nr_threads'])
    dynap = 'dynap_ex'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    _subprocess.call(args)

# -- dynap_ma --
def load_dynap_ma(path):

    # Carrego os dados:
    nr_header_lines = 13
    fname = _full([path, 'dynap_ma_out.txt'])
    turn,el,pos,en = _np.loadtxt(fname,skiprows=nr_header_lines,usecols=(1,2,4,7),unpack=True)

    pos  = pos[::2]
    # the -abs is for cases where the momentum aperture is less than the tolerance
    accep = _np.vstack([ en[1::2], -_np.abs(en[0::2]) ])
    nLost = _np.vstack([turn[1::2],turn[0::2]])
    eLost = _np.vstack([el[1::2],  el[0::2]])

    return pos, accep, nLost, eLost
def dynap_ma(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['nr_turns','y0','e_init','e_delta','nr_steps_back',
        'rescale','nr_iterations','s_min','s_max','nr_threads'])
    dynap = 'dynap_ma'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    for famname in kwargs['fam_names']:
        args.append(famname)
    _subprocess.call(args)

# -- dynap_pxa --
def dynap_pxa(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['nr_turns','y0','e_init','e_delta','nr_steps_back',
        'rescale','nr_iterations','s_min','s_max','nr_threads'])
    dynap = 'dynap_pxa'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    for famname in fam_names:
        args.append(famname)
    _subprocess.call(args)

# -- dynap_pya --
def dynap_pya(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['nr_turns','y0','e_init','e_delta','nr_steps_back',
            'rescale','nr_iterationss_min','s_max','nr_threads'])
    dynap = 'dynap_pya'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    for famname in fam_names:
        args.append(famname)
    _subprocess.call(args)

# -- dynap_xyfmap --
def dynap_xyfmap(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['de','nr_turns','x_nrpts','x_min','x_max',
                      'y_nrpts','y_min','y_max','nr_threads'])
    dynap = 'dynap_xyfmap'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    _subprocess.call(args)

# -- dynap_exfmap --
def dynap_exfmap(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['y','nr_turns','e_nrpts','e_min','e_max',
                      'x_nrpts','x_min','x_max','nr_threads'])
    dynap = 'dynap_exfmap'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    _subprocess.call(args)

# -- track_linepass --
def track_linepass(**kwargs):
    mand_keys = _commom_keys.copy()
    mand_keys.extend(['start_element','rx0','px0','ry0','py0','de0','dl0'])
    dynap = 'track_linepass'

    args = _prepare_args(dynap, mand_keys,**kwargs)
    _subprocess.call(args)
