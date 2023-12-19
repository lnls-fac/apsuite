"""Module 'base' for the class object 'Base': a collection of 'Buttons'."""

import os as _os
from copy import deepcopy as _dpcopy

import numpy as _np
from mathphys.functions import load_pickle, save_pickle

from . import buttons
from .si_data import si_elems, si_sectors, std_misaligment_types

_STD_ELEMS = si_elems()
_STD_TYPES = std_misaligment_types()
_STD_SECTS = si_sectors()

_D_BUTTONS_FILE = _os.path.join(
    _os.path.dirname(__file__), "Default_Buttons.pickle"
)


def set_model(model=None):
    """."""
    # print("entered base set model")
    buttons.buttons_set_model(model)


def __load_default_buttons():
    globals()["_DEFAULT_BUTTONS"] = load_pickle(_D_BUTTONS_FILE)


def update_default_base():
    """Update the Default Base."""
    __load_default_buttons()


try:
    update_default_base()
except FileNotFoundError:
    _DEFAULT_BUTTONS = []


class Base:
    """Base object for misalign analysis.

    Args:
            elems (list[str], str): List of magnets families names.\
            The valid options are the SIRIUS's dipoles, quadrupoles and\
            sextupoles.

            sects (list[int], int): List of sectors. Defaults to "all": \
            list of 1 to 20.

            dtypes (list[str], str): List of misalignmente types. Defaults to \
            "all": ['dx', 'dy', 'dr', 'drp', 'dry'].

            buttons (list[Button], optional): List of Button objects.

            func (str): The analysis function. Defaults to "vertical_disp". \
            The valid options are: 'vertical_disp' and 'testfunc'.

            use_root_buttons (bool, optional): Use default pre-saved buttons.

    About:
            Passing args 'elems' + 'sects' require the arg 'buttons' be None.

            The arg 'elems' can be (str) 'all', that contains all the magnets.

            The arg 'func' can be (str) 'testfunc': the button signature will\
            be all zero arrays.

            The 'use_root_buttons' arg sets if the Base creation will or not \
            use pre saved buttons and its signatures.
    """

    def __init__(
        self,
        elems="all",
        sects="all",
        dtypes="all",
        buttons=None,
        func="vertical_disp",
        use_root_buttons=True,
    ):
        """."""
        self._func = None
        self._use_root = None

        self._func, self._use_root = self.__handle_input(
            elems, sects, buttons, func, use_root_buttons
        )

        if buttons is None:
            self.__force_init(elems, sects, dtypes)

        else:
            self.__handle_buttons(buttons)

        self._matrix = self.__make_matrix()

    def __handle_input(self, elems, sects, buttons, func, use_root_buttons):
        """."""
        if func not in ("testfunc", "vertical_disp"):
            raise ValueError("invalid arg: func")

        if use_root_buttons not in (True, False):
            raise ValueError("invalid arg: use_root_Buttons")

        if any(f is not None for f in [elems, sects]) and buttons is not None:
            raise ValueError("too much args: (buttons) and (elems or sects)")

        return func, use_root_buttons

    def __force_init(self, elems, sects, dtypes):
        """."""
        # reading dtypes
        self._dtypes = dtypes
        if isinstance(dtypes, (list, tuple)) and all(
            i in _STD_TYPES for i in self._dtypes
        ):
            self._dtypes = sorted(
                list(set(self._dtypes)), key=lambda x: _STD_TYPES.index(x)
            )
        elif self._dtypes in _STD_TYPES:
            self._dtypes = [self._dtypes]
        elif self._dtypes == "all":
            self._dtypes = _STD_TYPES
        else:
            raise ValueError("invalid arg: dtypes")

        # reading sects
        self._sects = sects
        if (
            isinstance(self._sects, (list, tuple, _np.ndarray))
            and all(isinstance(i, (int, _np.integer)) for i in self._sects)
            and all(0 < i <= 20 for i in self._sects)
        ) or (
            isinstance(self._sects, (int, _np.integer))
            and 0 < self._sects <= 20
        ):
            self._sects = sorted(list(set(self._sects)))
        elif self._sects == "all":
            self._sects = _STD_SECTS
        else:
            raise ValueError("invalid arg: sects")

        # reading elems
        self._elems = elems
        if isinstance(self._elems, (list, tuple)) and all(
            i in _STD_ELEMS for i in self._elems
        ):
            self._elems = sorted(
                list(set(self._elems)), key=lambda x: _STD_ELEMS.index(x)
            )
        elif self._elems in _STD_ELEMS:
            self._elems = [self._elems]
        elif self._elems == "all":
            self._elems = _STD_ELEMS
        else:
            raise ValueError("invalid arg: elems")

        # gen buttons
        self._buttons = self.__generate_buttons()

    def __generate_buttons(self):
        all_buttons = []
        for dtype in self._dtypes:
            for sect in self._sects:
                for elem in self._elems:
                    if self._use_root and self._func == "vertical_disp":
                        temp_button = buttons.Button(
                            elem=elem, dtype=dtype, sect=sect, func="testfunc"
                        ).flatten()
                        for tb in temp_button:
                            if tb in _DEFAULT_BUTTONS:
                                sig = _DEFAULT_BUTTONS[
                                    _DEFAULT_BUTTONS.index(tb)
                                ].signature
                                tb._signature = _dpcopy(sig)
                                all_buttons += [tb]
                            else:
                                tb_new = buttons.Button(
                                    indices=tb.indices,
                                    dtype=tb.dtype,
                                    func="vertical_disp",
                                )
                                all_buttons += [tb_new]
                    else:
                        temp_button = buttons.Button(
                            elem=elem, dtype=dtype, sect=sect, func=self._func
                        ).flatten()
                        all_buttons += temp_button

        return all_buttons

    def __handle_buttons(self, buttons):
        # reading buttons
        self._buttons = buttons

        if isinstance(self._buttons, (list, tuple)) and all(
            isinstance(i, buttons.Button) for i in self._buttons
        ):
            self._buttons = self._buttons
        elif isinstance(self._buttons, buttons.Button):
            self._buttons = [self._buttons]
        else:
            raise ValueError("invalid arg: buttons")

        self._sects = []
        self._elems = []
        self._dtypes = []
        for button in self._buttons:
            self._sects.append(button.sect)
            self._elems.append(button.elem)
            self._dtypes.append(button.dtype)
        self._sects = sorted(list(set(self._sects)))
        self._elems = sorted(
            list(set(self._elems)), key=lambda x: _STD_ELEMS.index(x)
        )
        self._dtypes = sorted(
            list(set(self._dtypes)), key=lambda x: _STD_TYPES.index(x)
        )

    def __make_matrix(self):
        matrix = _np.array([b.signature for b in self._buttons]).T
        return matrix

    @property
    def buttons(self):
        """Returns the Base buttons list."""
        return self._buttons

    @property
    def resp_mat(self):
        """Returns the Base response matrix."""
        return self._matrix

    @property
    def sectors(self):
        """Returns the sectors presents in the Base."""
        return self._sects

    @property
    def magnets(self):
        """Returns the magnets (elements) presents in the Base."""
        return self._elems

    @property
    def dtypes(self):
        """Returns the misalignment types presents in the Base."""
        return self._dtypes

    def __len__(self):
        """Base length (number of Buttons)."""
        return len(self._buttons)

    def __eq__(self, other) -> bool:
        """Comparison method."""
        if isinstance(other, Base):
            for b in other.buttons():
                if b not in self.buttons():
                    return False
            return True
        return False


def save_default_base(base):
    """Save the input Base and its Buttons.

    Args:
        base (Base): Base object to be saved as Default Base.

    About:
        The save process only saves the Buttons of the input Base. Only \
        new and unsaved Buttons will be saved. Only Buttons with vertical_disp\
        signatures will be saved.
    """
    if base._func == "vertical_disp":
        for b in base.buttons:
            if b not in _DEFAULT_BUTTONS:
                _DEFAULT_BUTTONS.append(b)

        save_pickle(_DEFAULT_BUTTONS, _D_BUTTONS_FILE, overwrite=True)
        update_default_base()
        print("Saved/Updated default Base and Buttons!")
    else:
        print("Nothing saved.")
        pass


def delete_default_base():
    """Restore the Default Base to an empty list."""
    save_pickle([], _D_BUTTONS_FILE, overwrite=True)
    update_default_base()
    print("Default Base/Buttons deleted!")


__all__ = (
    "Base",
    "save_default_base",
    "delete_default_base",
    "update_default_base",
)
