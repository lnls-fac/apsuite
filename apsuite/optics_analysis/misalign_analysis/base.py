"""Module 'base' for the class object 'Base': a collection of 'Button'(s)."""

import os as _os
from copy import deepcopy as _dpcopy

import numpy as _np
from mathphys.functions import load_pickle, save_pickle

from .buttons import Button as _Button
from .si_data import si_elems, si_sectors, std_misaligment_types

_STD_ELEMS = si_elems()
_STD_TYPES = std_misaligment_types()
_STD_SECTS = si_sectors()

_D_BUTTONS_FILE = _os.path.join(
    _os.path.dirname(__file__), "Default_Buttons.pickle"
)


def load_default_base_button():
    globals()["_DEFAULT_BUTTONS"] = load_pickle(_D_BUTTONS_FILE)


try:
    load_default_base_button()
except FileNotFoundError:
    _DEFAULT_BUTTONS = []


class Base:
    """I'll rewrite this docstring. Be patience."""

    def __init__(
        self,
        elems="all",
        sects="all",
        dtypes="all",
        buttons=None,
        func="vertical_disp",
        use_root_buttons=True,
    ):
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
                    temp_button = _Button(
                        elem=elem, dtype=dtype, sect=sect, func=self._func
                    ).flatten()
                    all_buttons += (
                        temp_button
                        if isinstance(temp_button, (list, tuple, _np.ndarray))
                        else [temp_button]
                    )
        return all_buttons

    def __handle_buttons(self, buttons):
        # reading buttons
        self._buttons = buttons

        if isinstance(self._buttons, (list, tuple)) and all(
            isinstance(i, _Button) for i in self._buttons
        ):
            self._buttons = self._buttons
        elif isinstance(self._buttons, _Button):
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
        matrix = _np.zeros(shape=(160, self.__len__()))
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
        """Returns the modification types used to construct the Base."""
        return self._dtypes

    def __len__(self):
        return len(self._buttons)

    def __eq__(self, other) -> bool:
        if isinstance(other, Base):
            for b in other.buttons():
                if b not in self.buttons():
                    return False
            return True
        return False

    def set_default_base_buttons(self):
        if self._func == "vertical_disp":
            for b in self._buttons:
                if b not in _DEFAULT_BUTTONS:
                    _DEFAULT_BUTTONS.append(b)

            save_pickle(_DEFAULT_BUTTONS, _D_BUTTONS_FILE, overwrite=True)
            load_default_base_button()
            print("Saved Base/Buttons!")
        else:
            print("Base not saved.")
            pass


def delete_default_base_buttons():
    save_pickle([], _D_BUTTONS_FILE, overwrite=True)
    load_default_base_button()


__all__ = "Base"
