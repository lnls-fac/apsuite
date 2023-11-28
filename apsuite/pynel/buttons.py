"""Module 'buttons' for the class Object Button"""

from .std_si_data import MODEL_BASE, SI_SPOS, SI_SECT_SPOS, \
    STD_SECTS, STD_TYPES, STD_ELEMS_HALB, STD_ELEMS_LBLP, \
    SI_SECTOR_TYPES, STD_ERROR_DELTAS, STD_ELEMS, \
    SI_SECT_INDICES, ELEMS_ALL_INDICES
from apsuite.orbcorr import OrbitCorr as _OrbitCorr
import numpy as _np
from copy import deepcopy as _deepcopy
from .functions import calc_vdisp as _calc_vdisp, _SET_FUNCS, rmk_correct_orbit

_SI_SPOS          = SI_SPOS()
_SI_SECT_SPOS     = SI_SECT_SPOS()
_STD_SECTS        = STD_SECTS()       
_STD_TYPES        = STD_TYPES()       
_STD_ELEMS        = STD_ELEMS()
_STD_ELEMS_HALB   = STD_ELEMS_HALB() 
_STD_ELEMS_LBLP   = STD_ELEMS_LBLP() 
_OC_MODEL = MODEL_BASE()
_OC = _OrbitCorr(_OC_MODEL, 'SI')
_OC.params.maxnriters = 30
_OC.params.convergencetol = 1e-9
_OC.params.use6dorb = True
_INIT_KICKS = _OC.get_kicks()
_JAC = _OC.get_jacobian_matrix()
_STD_SECT_TYPES = SI_SECTOR_TYPES()
_SI_SECT_INDICES = SI_SECT_INDICES()
_DELTAS = STD_ERROR_DELTAS()
_FAMS_INDICES = ELEMS_ALL_INDICES()

class Button:
    """Button object for storing a magnet, it's sector (sect), it's indices 
       and it's vertical dispersion signature due to an misaligment (dtype)"""
    def __init__(self, sect=None, dtype=None, elem=None, func='testfunc', indices='auto', default_valids='std'):
        self.func = func
        default_valids = self.__process_valids(default_valids)
        if indices == 'auto':
            if (sect is None) and (elem is None) and (dtype is None):
                raise ValueError('Some parameters are missing "sect, elem, dtype" (OR) "indices"')
            else:
                self.__init_by_default(sect=sect, dtype=dtype, elem=elem, func=func, default_valids=default_valids)
        elif isinstance(indices, (list, _np.ndarray)):
            if all(isinstance(i, (int, _np.integer)) for i in indices):
                self.__init_by_indices(indices=indices, dtype=dtype, func=func, default_valids=default_valids)
            else:
                raise ValueError('indices with invalid values')
        elif isinstance(indices, (int, _np.integer)):
            self.__init_by_indices(indices=[indices], dtype=dtype, func=func, default_valids=default_valids)
        else:
            raise ValueError('Indices passed in wrong format')

    def __init_by_indices(self, indices, dtype, func, default_valids):
        self.indices = indices
        elem = _OC_MODEL[indices[0]].elem
        sect = list({j + 1 for i in indices for j in range(20) if (_SI_SECT_SPOS[j] < _SI_SPOS[i]) and (_SI_SPOS[i] < _SI_SECT_SPOS[j + 1])})
        if len(sect) > 1:
            raise ValueError(f'some elements passed are from different sectors: {sect}')
        self.sect = sect[-1]
        self.elem = elem
        self.fantasy_name = elem
        self.dtype = dtype
        self.sectype = self.__sector_type()

        if default_valids[0] == 'off':
            self.__validsects = 'off'
        elif default_valids[0] == 'std':
            self.__validsects = _STD_SECTS 

        if default_valids[1] == 'off':
            self.__validtypes = 'off'
        elif default_valids[1] == 'std':
            self.__validtypes = _STD_TYPES

        if default_valids[2] == 'off':
            self.__validnames = 'off'
        elif default_valids[2] == 'std':
            self.__validnames = _STD_ELEMS

        self.signature = _np.array([0.0 for _ in range(160)])
        if self.check_isvalid():
            if func == 'vertical_disp':
                temp = self.__calc_vertical_dispersion_signature()
            elif func == 'testfunc':
                temp = self.__calc_test_func_signature()
            elif func == 'twiss':
                temp = self.__calc_twiss_signatures()
            else:
                raise KeyError('Invalid signature function. Try "vertical_disp", "testfunc" or "twiss"')
            if isinstance(temp, list) and len(temp) == 1:
                if len(temp[0]) == 160:
                    self.signature = temp[0]
                else:
                    raise ValueError('Error calculating signature')
            else:
                self.signature = temp

    def __init_by_default(self, sect, dtype, elem, func, default_valids):
        #print('init by default', default_valids)
        self.elem = elem.rsplit('_')[0]
        self.fantasy_name = elem
        self.dtype = dtype
        self.sect = sect
        self.sectype = self.__sector_type()

        if default_valids[0] == 'std':
            self.__validsects = _STD_SECTS
        if default_valids[0] == 'off':
            self.__validsects = 'off'

        if default_valids[1] == 'std':
            self.__validtypes = _STD_TYPES
        if default_valids[1] == 'off':
            self.__validtypes = 'off'

        if default_valids[2] == 'std':
            if self.sectype in ['HighBetaA -> LowBetaB', 'LowBetaB -> HighBetaA']:
                self.__validnames = _STD_ELEMS_HALB
            elif self.sectype in ['LowBetaP -> LowBetaB', 'LowBetaB -> LowBetaP']:
                self.__validnames = _STD_ELEMS_LBLP
            else:
                self.__validnames = []
        if default_valids[2] == 'off':
            self.__validnames = 'off'

        self.indices = []
        self.signature = _np.zeros(160)
        temp = [0.0 for _ in range(160)]

        if self.check_isvalid():
            self.indices = self.__find_indices()

            if func == 'vertical_disp':
                temp = self.__calc_vertical_dispersion_signature()
            elif func == 'testfunc':
                temp = self.__calc_test_func_signature()
            elif func == 'twiss':
                temp = self.__calc_twiss_signatures()
            else:
                raise KeyError('Invalid signature function. Try "vertical_disp", "testfunc" or "twiss"')
            
        if all(isinstance(l, list) for l in self.indices):
            if len(self.indices) == 1:
                if all(isinstance(i, (int, _np.integer)) for i in self.indices[0]):
                    self.indices = self.indices[0]
                    self.signature = temp[0]
            elif all(all(isinstance(i, (int, _np.integer)) for i in self.indices[k]) for k in range(len(self.indices))):
                self.signature = temp
            else:
                raise ValueError('indices has lists, but not lists of ints')

        elif all(isinstance(i, (int, _np.integer)) for i in self.indices):
            if len(temp) == 1:
                if len(temp[0]) == 160:
                    self.signature = temp[0]
                else:
                    raise ValueError('error with signature')
                
            elif len(temp) == 160:
                self.signature = temp
            else:
                raise ValueError('error with signature')
        else:
            raise ValueError('indices error')

    def check_isflat(self):
        if self.indices == []:
            return True
        elif isinstance(self.indices, list):
            if all(isinstance(idx, (int, _np.integer)) for idx in self.indices):
                return True
            elif all(isinstance(idx, list) for idx in self.indices):
                return False
        else:
            raise ValueError('flat error: problem with indices')

    def __str__(self) -> str:
        return '('+str(self.sect)+','+str(self.dtype)+','+str(self.fantasy_name)+')'

    def __repr__(self) -> str:
        return '('+str(self.sect)+','+str(self.dtype)+','+str(self.fantasy_name)+')'

    def __eq__(self, other):
        try: # cant use "isinstance" because "full_buttons" pickle was generated with package "pynel", not "apsuite.pynel" ! 
            if (self.dtype == other.dtype) and (self.indices == other.indices) and (self.fantasy_name == other.fantasy_name):
                return True
            return False
        except:
            return False

    def check_isvalid(self):
        validify = [False, False, False]
        if self.__validsects == 'off':
            validify[0] = True
        elif (self.sect in self.__validsects):
            validify[0] = True
        if self.__validtypes == 'off':
            validify[1] = True
        elif (self.dtype in self.__validtypes):
            validify[1] = True
        if self.__validnames == 'off':
            validify[2] = True
        elif (self.elem in self.__validnames):
            validify[2] = True
        return all(validify)
    
    def __check_isvalid_for_printing(self):
        validify = [False, False, False]
        if self.__validsects == 'off':
            validify[0] = True
        elif (self.sect in self.__validsects):
            validify[0] = True
        if self.__validtypes == 'off':
            validify[1] = True
        elif (self.dtype in self.__validtypes):
            validify[1] = True
        if self.__validnames == 'off':
            validify[2] = True
        elif (self.elem in self.__validnames):
            validify[2] = True
        return validify
        
    def show_invalid_parameters(self):
        valids = self.__check_isvalid_for_printing()
        invalid = []
        strings = ['sector', 'dtype', 'elem']
        for i, v in enumerate(valids):
            if not v:
                invalid.append(strings[i])
        if len(invalid) == 0:
            print('(%d, %s, %s) ---> valid button' % (self.sect,self.dtype, self.elem))
        if len(invalid) == 1:
            print('(%d, %s, %s) ---> invalid %s' % (self.sect, self.dtype, self.elem,  invalid[0]))
        if len(invalid) == 2:
            print('(%d, %s, %s) ---> invalid %s & %s' % (self.sect, self.dtype, self.elem, invalid[0], invalid[1]))
        if len(invalid) == 3:
            print('(%d, %s, %s) ---> completely invalid' % (self.sect, self.dtype, self.elem))

    def __find_indices(self):
        famidx = _FAMS_INDICES[self.elem]
        idx = _np.where((famidx > _SI_SECT_INDICES[self.sect-1]) & (famidx < _SI_SECT_INDICES[self.sect]))
        idx = list(famidx[idx])
        if '_' in self.fantasy_name:
            number = int(self.fantasy_name.rsplit('_')[-1])
            if number == 1:
                return idx[:int(len(idx)/2)]
            return idx[int(len(idx)/2):]
        elif self.elem in ['Q1', 'Q2', 'Q3', 'Q4']:
            return [[idx[0]], [idx[1]]]
        elif self.elem in ['B2', 'B1']:
            return [idx[:int(len(idx)/2)], idx[int(len(idx)/2):]]
        return [idx]

    def __sector_type(self):
        if self.sect in [2, 6, 10, 14, 18]:
            return _STD_SECT_TYPES[1]
        elif self.sect in [3, 7, 11, 15, 19]:
            return _STD_SECT_TYPES[2]
        elif self.sect in [4, 8, 12, 16, 20]:
            return _STD_SECT_TYPES[3]
        elif self.sect in [1, 5, 9, 13, 17]:
            return _STD_SECT_TYPES[0]
        else:
            return 'Not_Sirius_Sector'

    def __calc_test_func_signature(self):
        disp = []
        if all(isinstance(i, list) for i in self.indices):
            #print('list of lists')
            for ind in self.indices:
                _disp = _np.array([i for i in range(160)])
                disp.append(_disp.ravel())
        elif all(isinstance(i, (int, _np.integer)) for i in self.indices):
            #print('list of ints')
            _disp = _np.array([i for i in range(160)])
            disp.append(_disp.ravel())
        else:
            raise ValueError('Indices with format problem')
        return disp
    
    def __calc_vertical_dispersion_signature(self):
        func = _SET_FUNCS[self.dtype]
        if all(isinstance(i, list) for i in self.indices): # list of list of ints
            indices = self.indices
        elif all(isinstance(i, (int, _np.integer)) for i in self.indices): # list of ints
            indices = [self.indices]  
        else:
            raise ValueError('Indices with format problem')
        # the calculation:
        disp = []
        delta = _DELTAS[self.dtype][self.elem[0]]
        for ind in indices:
            disp_0 = _calc_vdisp(_OC_MODEL)
            func(_OC_MODEL, indices=ind, values=delta) # applying (SETTING) positive delta
            rmk_correct_orbit(_OC, _JAC)
            disp_p = _calc_vdisp(_OC_MODEL)
            disp.append(((disp_p-disp_0)/delta).ravel())
            func(_OC_MODEL, indices=ind, values=0.0)
            _OC.set_kicks(_INIT_KICKS)
        return disp

    def flatten(self):
        """Split the button if its contains two or more magnets"""
        if not isinstance(self.indices, list):
            raise ValueError('indices error')
        elif self.indices != []:
            if isinstance(self.indices[0], list):
                # Split the button into multiple buttons
                buttons = []
                for i in range(len(self.indices)):
                    sub_button = _deepcopy(self)
                    sub_button.indices = self.indices[i]
                    sub_button.signature = self.signature[i]
                    sub_button.fantasy_name = self.fantasy_name+'_'+str(i+1)
                    buttons.append(sub_button)
                return buttons
            else:
                # Return the button as a single-item list
                return [self]
        else:
            return [self] # flat button, but propably is invalid
        
    def __process_valids(self, arg):
        #print('arg = ', arg)
        if arg == 'std':
            return ['std', 'std', 'std']
        elif arg == 'off':
            return ['off', 'off', 'off']
        elif isinstance(arg, (tuple, list)):
            if len(arg) == 3:
                if any((d not in ['off', 'std']) for d in arg):
                    raise ValueError('the "default_valids should contain only "std" of "off" strings"')
                else:
                    return arg
            else:
                raise ValueError('"default_valids" parameter should be a list of 3 strings: "off" and/or "std"')
        else:
            raise ValueError('"default_valids" parameter should be "off", "std" or a list/tuple')
        