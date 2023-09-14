"""Module 'base' for the class object 'Base': a collection of 'Button'(s)"""

from .std_si_data import STD_ELEMS, STD_SECTS, \
    STD_TYPES, BPMIDX,  SI_SECTOR_TYPES, \
    COMPLETE_BUTTONS_VERTICAL_DISPERSION
from .buttons import Button as _Button
import numpy as _np
from copy import deepcopy as _dpcopy
from time import time as _time

_STD_ELEMS          = STD_ELEMS()
_STD_SECTS          = STD_SECTS()
_STD_TYPES          = STD_TYPES()
_STD_SECT_TYPES     = SI_SECTOR_TYPES()
_bpmidx             = BPMIDX()
_FULL_VERTC_BUTTONS = COMPLETE_BUTTONS_VERTICAL_DISPERSION()

class Base:
    """
    Object Base: a collection of buttons (Button Object)
    About:
    ---> The Base object was implemented to group Button objects and perform analisys on how these buttons can modify the optics in SIRIUS ring.

    Creation:
    ---> Creating a Base can be performed in two basic ways: passing specified elements and sectors or passing girder indices:

    > Creating by default requires passing three args: 
    >'sects' (integers), 'elements' (name strings of the magnets) and 'dtypes' (variations between 'dx', 'dy', 'dr')

    > Creating by buttons requires passing only one arg: 
    >'buttons' (a list of buttons or a single one)

    > Creating by girders indices requires passing only two args: 
    > 'girders' (the indices of a single girder or more girders) and 'dtypes' (variations between 'dx', 'dy', 'dr')

    *kwargs:
    auto_refine: default=True ---> automatically refines the Base by removing invalid buttons and flatten the valids
    exclude: default=None ---> create the base without a group of unwanted elements, sects or dtypes
    valids_cond: default=False ---> reset the 'valid' condition for buttons if it is not a SIRIUS standart valid button ("Sandbox buttons")
    func: default='vertical_disp'/'testfunc' ---> set the default signature function of the buttons
    """
    def __init__(self, sects='all', elements='all', dtypes='all', auto_refine=True, exclude=None, valids_cond=['std', 'std', 'std'], func='vertical_disp', buttons=None, force_rebuild=False):
        self.rebuild = force_rebuild
        self.__func = func
        self.__init_flag = None
        self.bpmidx = _bpmidx
        if func == 'twiss':
            print('The TWISS Base is deactivated...')
        if buttons == None:
            self.__init_by_default(sects=sects, elements=elements, dtypes=dtypes, exclude=exclude, valids_cond=valids_cond, func=func)

        elif buttons != None and sects == 'all' and dtypes == 'all' and elements == 'all':
            if isinstance(buttons, list) and all(isinstance(i, _Button) for i in buttons):
                self.__init_by_buttons(buttons=buttons)
            elif isinstance(buttons, _Button):
                self.__init_by_buttons(buttons=[buttons])
            else:
                raise ValueError('parameter "buttons" passed with wrong format')
            
        else:
            raise ValueError('conflict when passing "buttons"')

        self._SECT_TYPES = self.__find_sector_types()
        self.__is_flat = self.__check_isflat()
        self.__is_updated = False

        if auto_refine:
            self.refine_base(update_buttons=True, flatten=True, return_removed=False, show_invalids=False)

        if self.rebuild == False and self.__init_flag == 'by_default':
            temp_buttons = []
            for button in self.__buttons_list:
                for buttonV in _FULL_VERTC_BUTTONS:
                    if button == buttonV:
                        temp_buttons.append(buttonV)

            self.__buttons_list = temp_buttons

        self.__matrix = self.__make_matrix()
        _t = _time()
        self.id = str(int((_t-int(_t))*1e6))
        return

    def __init_by_buttons(self, buttons):
        #print('starting by buttons')
        __stdfunc = 'None'
        _SECTS =[]
        _ELEMS =[]
        _TYPES =[]

        for button in buttons:
            if button.sect not in _SECTS: 
                _SECTS.append(button.sect) 

            if button.bname not in _ELEMS: 
                _ELEMS.append(button.bname) 

            if button.dtype not in _TYPES: 
                _TYPES.append(button.dtype)

        self._SECTS, self._ELEMS, self._TYPES, self.__buttons_list = _SECTS, _ELEMS, _TYPES, buttons
        self.__init_flag = 'by_buttons'

    def __check_isflat(self):
        for b in self.__buttons_list:
            if isinstance(b.indices, (list, tuple, _np.ndarray)) and b.indices == []:
                return True
            elif isinstance(b.indices, (list, tuple, _np.ndarray)) and b.indices != []:
                if isinstance(b.indices[0], (list, tuple, _np.ndarray)):
                    return False
                elif all(isinstance(idx, (int, _np.integer)) for idx in b.indices):
                    return True
                else:
                    raise ValueError('list of indices with problem')
        return False

    def __init_by_default(self, sects, elements, dtypes, exclude, valids_cond, func):
        #print('starting by default')
        if sects == 'all':
            _SECTS = _STD_SECTS
        else:
            if isinstance(sects, list):
                _SECTS = sects
            elif isinstance(sects, int):
                _SECTS = [sects]
            else:
                raise TypeError('sects parameter not in correct format')
        if elements == 'all':
            _ELEMS = _STD_ELEMS
        else:
            if isinstance(elements, list):
                _ELEMS = elements
            elif isinstance(elements, str):
                _ELEMS = [elements]
            else:
                raise TypeError('elements parameter not in correct format')

        if dtypes == 'all':
            _TYPES = _STD_TYPES
        else:
            if isinstance(dtypes, list):
                _TYPES = dtypes
            elif isinstance(dtypes, str):
                _TYPES = [dtypes]
            else:
                raise TypeError('dtypes parameter not in correct format')

        __default_valids = valids_cond

        self._SECTS, self._ELEMS, self._TYPES, = _SECTS, _ELEMS, _TYPES
        self.__buttons_list = self.__generate_buttons(exclude, stdfunc=func, default_valids=__default_valids)
        self.__init_flag = 'by_default'

    def __find_sector_types(self):
        sectypes = []
        for sect in self._SECTS:
            if sect in [2, 6, 10, 14, 18]:
                sectypes.append((sect, _STD_SECT_TYPES[1]))
            elif sect in [3, 7, 11, 15, 19]:
                sectypes.append((sect, _STD_SECT_TYPES[2]))
            elif sect in [4, 8, 12, 16, 20]:
                sectypes.append((sect, _STD_SECT_TYPES[3]))
            elif sect in [1, 5, 9, 13, 17]:
                sectypes.append((sect, _STD_SECT_TYPES[0]))
            else:
                sectypes.append((sect, 'Not_Sirius_Sector'))
        return dict(sectypes)

    def __generate_buttons(self, exclude=None, stdfunc='vertical_disp', default_valids=['std', 'std', 'std']):
        to_exclude = []
        if exclude == None:
            exclude = set()
        elif isinstance(exclude, (str, int)):
            exclude = set([exclude])
        elif isinstance(exclude, (list, tuple)):
            exclude = set(exclude)
        else:
            raise TypeError("Exclude parameters not in format!")

        for e in exclude:
            if isinstance(e, (str, int)):
                to_exclude.extend(self.__exclude_buttons(e))
            elif isinstance(e, (tuple, list)):
                to_exclude.extend(self.__exclude_buttons(*e))
            else:
                raise TypeError("Exclude parameters not in format!")
        
        if to_exclude == []:
            to_exclude = [_Button(sect=-1, name='FalseButton', dtype='dF')]
        exparams=[]
        for exbutton in to_exclude:
            exparams.append((exbutton.sect, exbutton.dtype, exbutton.bname))

        if self.rebuild == True:
            all_buttons = []
            for dtype in self._TYPES:
                for sect in self._SECTS:
                    for elem in self._ELEMS:
                        if (sect, dtype, elem) not in exparams:
                            temp_Button = _Button(name=elem, dtype=dtype, sect=sect, default_valids=default_valids, func=stdfunc)
                            all_buttons.append(temp_Button)
        elif self.rebuild == False:
            all_buttons = []
            for dtype in self._TYPES:
                for sect in self._SECTS:
                    for elem in self._ELEMS:
                        if (sect, dtype, elem) not in exparams:
                            temp_Button = _Button(name=elem, dtype=dtype, sect=sect, default_valids=default_valids, func='testfunc')
                            all_buttons.append(temp_Button)
        return all_buttons

    def __exclude_buttons(self, par1, par2=None, par3=None):
        if par2 == None and par3 == None:
            if isinstance(par1, int):
                exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                             for sect in self._SECTS
                             for dtype in self._TYPES
                             for elem in self._ELEMS
                             if sect == par1]
            elif isinstance(par1, str):
                if par1[0] == 'd':
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if dtype == par1]
                else:
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if elem == par1]
        elif par3 == None:
            if isinstance(par1, int):  # par1 = sect
                if par2.startswith('d'):  # par1 = sect, par2 = dtype #### (sect, dtype)
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (dtype == par2 and sect == par1)]
                # par1 = sect, par2 = elem                     #### (sect, elem)
                else:
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (elem == par2 and sect == par1)]
            elif isinstance(par2, int):  # par2 = sect
                if par1.startswith('d'):  # par1 = dtype, par2 = sect #### (dtype, sect)
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (dtype == par1 and sect == par2)]
                # par1 = elem, par2 = sect                     #### (elem, sect)
                else:
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (elem == par1 and sect == par2)]
            else:  # par1, par2 = elem or dtype:
                if par1.startswith('d'):  # par1 = dtype, par2 = elem #### (dtype, elem)
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (dtype == par1 and elem == par2)]
                # par1 = elem, par2 = dtype                   #### (elem, dtype)
                else:
                    exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )
                                 for sect in self._SECTS
                                 for dtype in self._TYPES
                                 for elem in self._ELEMS
                                 if (elem == par1 and dtype == par2)]
        else:
            for el in (par1, par2, par3):
                if isinstance(el, int):
                    sect = el
                if isinstance(el, str) and el[0] == 'd':
                    dtype = el
                if isinstance(el, str) and el[0] != 'd':
                    elem = el
            exbuttons = [_Button(name=elem, dtype=dtype, sect=sect, func='testfunc' )]
        return exbuttons

    def refine_base(self, update_buttons=True, flatten=True, return_removed=False, show_invalids=False):  
        """Function that refines the Base:
        update_buttons: default=True --> the refining will find and remove invalid buttons
        return_removed: default=False --> return a list of the invalid buttons (removed or set to remove)
        show_invalids: default=False --> print the invalid-buttons invalid parameters
        flatten: default=True --> split not-flat buttons
        """
        if flatten:
            flat = []
            for b in self.__buttons_list:
                for new_b in b.flatten():
                    flat.append(new_b)
            self.__buttons_list = flat
            self.__is_flat = self.__check_isflat()

        to_remove = []
        for b in self.__buttons_list:
            if not b.check_isvalid():
                to_remove.append(b)

        if update_buttons:
            
            self._SECTS = []
            self._ELEMS = []
            self._TYPES = []
            old_buttons = _dpcopy(self.__buttons_list)

            self.__buttons_list = []
            for button in old_buttons:
                if button not in to_remove: 
                    self.__buttons_list.append(button)
                    if button.sect not in self._SECTS: 
                        self._SECTS.append(button.sect) 
                    if button.bname not in self._ELEMS: 
                        self._ELEMS.append(button.bname) 
                    if button.dtype not in self._TYPES: 
                        self._TYPES.append(button.dtype) 

            self.__is_updated = True

        if show_invalids:
            for b in to_remove:
                b.show_invalid_parameters()

        if return_removed:
            return to_remove
        
        self.__matrix = self.__make_matrix()

    def __make_matrix(self):
        if self.__func == 'twiss':
            return 0
        if len(self.__buttons_list) <= 0:
            print('Zero buttons, matrix not generated')
        elif self.__is_flat and self.__is_updated:
            M = _np.zeros((160, len(self.__buttons_list)))
            for i, b in enumerate(self.__buttons_list):
                M[:, i] = _np.array(b.signature).ravel()
            return M
        elif self.__is_flat == True and self.__is_updated == False:
            print('Base flat, but not updated please refine (update)')
            return 0
        elif self.__is_flat == False and self.__is_updated == True:
            print('Base not flat, please refine (flatten)')
            return 0
        else:
            print('Please refine Base (update & flatten)')
            return 0

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

    def is_flat(self):
        """Verifies if the Base is flat
        -> (verifies if the buttons in the Base are flatten)"""
        return self.__check_isflat()

    def is_updated(self):
        """Verifies if the Base is up-to-date"""
        return self.__is_updated
    
    def __len__(self):
        return len(self.__buttons_list)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Base):
            for b in other.buttons():
                if b not in self.buttons():
                    return False
            return True
        return False
    
__all__ = ("Base")