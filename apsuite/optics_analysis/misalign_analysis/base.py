"""Module 'base' for the class object 'Base': a collection of 'Button'(s)"""
from .si_data import si_elems, std_misaligment_types, si_sectors,\
    si_bpmidx, si_dipoles, si_sextupoles, si_quadrupoles
from .buttons import Button as _Button
import numpy as _np
from copy import deepcopy as _dpcopy
from time import time as _time
from mathphys.functions import load_pickle, save_pickle
_STD_ELEMS          = si_elems()
_STD_TYPES          = std_misaligment_types()
_STD_SECTS          = si_sectors()
_STD_SEXTUPOLES     = si_sextupoles()
_STD_QUADRUPOLES    = si_quadrupoles()
_STD_DIPOLES        = si_dipoles()

path = '/'.join(__file__.split('/')[:-1])

DEFAULT_BUTTONS = None
def load_default_base_button():
    globals()['DEFAULT_BUTTONS'] = load_pickle(path+'/Default_Pynel_Base_Buttonsaaaaaaaaaaaa.pickle')
try:
    load_default_base_button()
except FileNotFoundError:
    DEFAULT_BUTTONS = []


class Base:
    """I'll rewrite this docstring. Be patience.
    """
    def __init__(self, **kwargs):

        # reading the func arg >>> default = "vertical_disp"
        self._func = kwargs.get('func', 'vertical_disp')
        if self._func not in ('testfunc', 'vertical_disp'):
            raise ValueError('invalid arg: func')

        self._use_root = kwargs.get('use_root_Buttons', True)
        if self._use_root not in (True, False):
            raise ValueError('invalid arg: use_root_Buttons')

        if ('buttons' in kwargs) and any(i in kwargs for i in ('sects', 'elems')):
            raise ValueError('too much args: (buttons) and (elems or sects)')

        if 'buttons' not in kwargs:
            # reading dtypes arg
            self._dtypes = kwargs.get('dtypes', sorted(list(set(_STD_TYPES)),
                                                       reverse=True))
            if self._dtypes is not None:
                if isinstance(self._dtypes, (list, tuple)) and \
                    all(i in _STD_TYPES for i in self._dtypes):
                    self._dtypes = sorted(list(set(self._dtypes)),
                                          key=lambda x: _STD_TYPES.index(x))
                elif self._dtypes in _STD_TYPES:
                    self._dtypes = [self._dtypes]
            else:
                raise ValueError('invalid arg: dtypes')

            # reading sects
            self._sects = kwargs.get('sects', _STD_SECTS)
            if (isinstance(self._sects, (list, tuple, _np.ndarray)) and \
                all(isinstance(i, (int, _np.integer)) for i in self._sects)\
                    and all(0<i<=20 for i in self._sects)) \
                    or (isinstance(self._sects, (int, _np.integer)) \
                        and 0 < self._sects <= 20):
                self._sects = sorted(list(set(self._sects)))
            else:
                raise ValueError('invalid arg: sects')

            # reading elems
            self._elems = kwargs.get('elements', _STD_ELEMS)
            if isinstance(self._elems, (list, tuple)) and \
                all(i in _STD_ELEMS for i in self._elems):
                self._elems = sorted(list(set(self._elems)),
                                     key=lambda x:_STD_ELEMS.index(x))
            elif self._elems in _STD_ELEMS:
                self._elems = [self._elems]
            elif self._elems == 'sextupoles':
                self._elems = _STD_SEXTUPOLES
            elif self._elems == 'quadrupoles':
                self._elems = _STD_QUADRUPOLES
            elif self._elems == 'dipoles':
                self._elems = _STD_DIPOLES
            else:
                raise ValueError('invalid arg: elems')

            # gen buttons
            self._buttons = self.__generate_buttons()

        else:
            # reading buttons
            self._buttons = kwargs.get('buttons', None)
            if self._buttons is not None:
                if isinstance(self._buttons, (list, tuple)) and \
                    all(isinstance(i, _Button) for i in self._buttons):
                    self._buttons = self._buttons
                elif isinstance(self._buttons, _Button):
                    self._buttons = [self._buttons]
                else:
                    raise ValueError('invalid arg: buttons')

            self._sects = []
            self._elems = []
            self._dtypes = []
            for button in self._buttons:
                self._sects.append(button.sect)
                self._elems.append(button.elem)
                self._dtypes.append(button.dtype)
            self._sects = sorted(list(set(self._sects)))
            self._elems = sorted(list(set(self._elems)),
                                 key=lambda x: _STD_ELEMS.index(x))
            self._dtypes = sorted(list(set(self._dtypes)),
                                  key=lambda x: _STD_TYPES.index(x))

    def __generate_buttons(self):
        all_buttons = []
        for dtype in self._TYPES:
            for sect in self._SECTS:
                for elem in self._ELEMS:
                    sig_flag = 0
                    for bt in DEFAULT_BUTTONS:
                        if (sect,dtype,elem) == (bt.sect,bt.dtype,bt.elem):
                            sig = bt.signature
                            sig_flag = 1
                    if sig_flag == 0:
                        temp_Button = _Button(elem=elem, dtype=dtype,
                                              sect=sect, func='vertical_disp')
                    else:
                        temp_Button = _Button(elem=elem, dtype=dtype,
                                              sect=sect, func='testfunc')
                        temp_Button.signature = _dpcopy(sig)
                    all_buttons.append(temp_Button)
        flat_buttons = []
        for button in all_buttons:
            b = button.flatten()
            flat_buttons += b if isinstance(b, (list, tuple, _np.ndarray))\
                else [b]
        return flat_buttons

    @property
    def buttons(self):
        """Returns the Base buttons list"""
        return self.__buttons_list

    @property
    def sectors(self):
        """Returns the sectors presents in the Base"""
        return self._SECTS

    @property
    def magnets(self):
        """Returns the magnets (elements) presents in the Base"""
        return self._ELEMS

    @property
    def named_magnets(self):
        _SPLIT_ELEMS = []
        for b in self.buttons:
            if b.fantasy_name not in _SPLIT_ELEMS:
                _SPLIT_ELEMS.append(b.fantasy_name)
        return _SPLIT_ELEMS

    @property
    def dtypes(self):
        """Returns the modification types used to construct the Base"""
        return self._TYPES

    @property
    def sector_types(self):
        """Returns the sector-types presents in the Base"""
        return self._SECT_TYPES

    @property
    def resp_mat(self):
        """Returns the Base Response Matrix"""
        return self.__matrix

    def __len__(self):
        return len(self.__buttons_list)

    def __eq__(self, other) -> bool:
        if isinstance(other, Base):
            for b in other.buttons():
                if b not in self.buttons():
                    return False
            return True
        return False

    def set_default_base_buttons(self):
        if self.__func == 'vertical_disp':
            save_pickle(self.buttons, path+'/Default_Pynel_Base_Buttons.pickle', overwrite=True)
            load_default_base_button()
        else:
            raise ValueError('Only "vertical dispersion" Bases can be set as default Pynel Base and Buttons')

__all__ = ("Base")
