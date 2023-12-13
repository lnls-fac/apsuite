"""Module 'buttons' for the class Object Button"""
from apsuite.pynel.std_si_data import MODEL_BASE, STD_ERROR_DELTAS, STD_TYPES, SI_FAMDATA, STD_ELEMS, SI_SPOS
from apsuite.orbcorr import OrbitCorr as _OrbitCorr
import numpy as _np
from copy import deepcopy as _deepcopy
from apsuite.pynel.functions import calc_vdisp as _calc_vdisp, _SET_FUNCS, rmk_correct_orbit

_OC_MODEL = MODEL_BASE()
_OC = _OrbitCorr(_OC_MODEL, 'SI')
_OC.params.maxnriters = 30
_OC.params.convergencetol = 1e-9
_OC.params.use6dorb = True
_INIT_KICKS = _OC.get_kicks()
_JAC = _OC.get_jacobian_matrix()
_DELTAS = STD_ERROR_DELTAS()
_STD_TYPES = STD_TYPES()
_fam = SI_FAMDATA()
_STD_ELEMS = STD_ELEMS()
_sects_dict = {fam_name:[int(s[:2]) for i,s in enumerate(_fam[fam_name]['subsection'])] for fam_name in _STD_ELEMS}
_SI_SPOS = SI_SPOS()

class Button:
    """Button object for storing a magnet, it's sector (sect), it's indices 
       and it's vertical dispersion signature due to an misaligment (dtype)"""
    def __init__(self, dtype=None, **kwargs):
        self._elem = None
        self._sect = None
        self._indices = None

        if dtype in _STD_TYPES:
            self._dtype = dtype
        else: raise ValueError('Invalid dtype')

        if 'func' not in kwargs:
            self._func = 'vertical_disp'
        elif 'func' in kwargs and kwargs['func'] in ['vertical_disp', 'testfunc']:
            self._func = kwargs['func']
        else: 
            raise ValueError('Invalid func')

        if 'indices' in kwargs:
            if any([k in kwargs for k in ('elem', 'sect')]) and any(kwargs[k] is not None for k in ('elem', 'sect')):
                raise ValueError('Too much args')
            else:
                self._indices = kwargs['indices']

        elif all([k in kwargs for k in ('elem', 'sect')]):
            self._sect = kwargs['sect']
            self._elem = kwargs['elem']
        else: raise ValueError('Missing input args')

        self.__force_init__()
            
        self._signature = self.__calc_signature()

    def __str__(self) -> str:
        return '('+str(self._sect)+','+str(self._dtype)+','+str(self._fantasy_name)+')'

    def __repr__(self) -> str:
        return '('+str(self._sect)+','+str(self._dtype)+','+str(self._fantasy_name)+')'

    def __eq__(self, other):
        try:
            if (self._dtype == other.dtype) and (self._indices == other.indices) and (self._fantasy_name == other.fantasy_name):
                return True
            return False
        except:
            return False

    def __calc_signature(self):
        if self._func == 'testfunc':
            if isinstance(self._fantasy_name, list):
                return [_np.zeros(160) for i in self._fantasy_name]
            else:
                return _np.zeros(160) 
        func = _SET_FUNCS[self._dtype]
        if all(isinstance(i, list) for i in self._indices): # list of list of ints
            indices = self._indices
        elif all(isinstance(i, (int, _np.integer)) for i in self._indices): # list of ints
            indices = [self._indices]  
        else:
            raise ValueError('Indices with format problem')
        # the calculation:
        disp = []
        delta = _DELTAS[self._dtype][self._elem[0]]
        for ind in indices:
            disp_0 = _calc_vdisp(_OC_MODEL)
            func(_OC_MODEL, indices=ind, values=delta) # applying (SETTING) positive delta
            rmk_correct_orbit(_OC, _JAC)
            disp_p = _calc_vdisp(_OC_MODEL)
            disp.append(((disp_p-disp_0)/delta).ravel())
            func(_OC_MODEL, indices=ind, values=0.0)
            _OC.set_kicks(_INIT_KICKS)
        return disp        

    def __force_init__(self):
        elem = self._elem
        fixpos = -1
        if elem is not None:
            if isinstance(elem, str):
                elem = elem.rsplit('_')
            if len(elem) == 1:
                elem = elem[0]
            else:
                elem, fixpos = elem[0], int(elem[1])
        
        sect = self._sect
        if sect is not None:
            if not isinstance(sect, (_np.integer, int)) or sect < 1 or sect > 20:
                raise ValueError('problem with sect')
            
        indices = self._indices
        split_flag = False
        if indices is not None:
            if isinstance(indices, (int, _np.integer)):
                indices = [indices]
            elif isinstance(indices, (_np.ndarray, list, tuple)) and all(isinstance(i, (_np.integer, int)) for i in indices):
                pass
            else:
                ValueError('indices passed in wrong format')
            found_elems = [fname for fname in list(set([_OC_MODEL[int(idx)].fam_name for idx in indices])) if fname in _STD_ELEMS]
            if len(found_elems) != 1:
                raise ValueError('invalid indices')
            elem = found_elems.pop()
            indices = [f for f in _fam[elem]['index'] if indices[0] in f]
            if len(indices) == 1:
                indices = indices[0]
        if indices is None:
            indices = [_fam[elem]['index'][i] for i,s in enumerate(_sects_dict[elem]) if s == sect]
            if len(indices) == 1:
                if isinstance(indices[0], (list, tuple, _np.ndarray)) and len(indices[0]) > 1:
                    indices = indices[0]
            else:
                split_flag = True
        
        spos = [_SI_SPOS[i[0]] for i in indices] if split_flag else _SI_SPOS[indices[0]]
        if len(set(_sects_dict[elem])) != len(_sects_dict[elem]) and split_flag == False:
            pos = 0
            for i, s in enumerate(_sects_dict[elem]):
                if s == sect:
                    pos += 1
                    if indices[0] in _fam[elem]['index'][i]:
                        break
            fantasy_name = elem+'_'+str(pos)
        elif len(set(_sects_dict[elem])) != len(_sects_dict[elem]) and split_flag == True:
            fantasy_name = [elem+'_'+str(i+1) for i in range(len(indices))]
        else:
            fantasy_name = elem
        if fixpos == -1:
            self._fantasy_name = fantasy_name
            self._sect = sect
            self._spos = spos
            self._indices = indices
        else:
            self._fantasy_name = fantasy_name[fixpos-1]
            self._sect = sect
            self._spos = spos[fixpos-1]
            self._indices = indices[fixpos-1]

    @property
    def func(self):
        return self._func
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def elem(self):
        return self._elem
    
    @property
    def sect(self):
        return self._sect
    
    @property
    def indices(self):
        return self._indices
    
    @property
    def signature(self):
        return self._signature
    
    @property
    def spos(self):
        return self._spos
    
    @property
    def fantasy_name(self):
        return self._fantasy_name

    def flatten(self):
        if not isinstance(self, Button):
            print('arg is not a Button object')
            return
        if isinstance(self.fantasy_name, list):
            buttons = []
            for i in range(len(self.fantasy_name)):
                print(f'spliting {i+1} {self}')
                b = _deepcopy(self)
                b._signature = self.signature[i]
                b._fantasy_name = self.fantasy_name[i]
                b._indices = self.indices[i]
                buttons.append(b)
            return buttons
        else: 
            return self