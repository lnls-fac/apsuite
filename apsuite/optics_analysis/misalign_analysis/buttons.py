"""Module 'buttons' for the class Object Button."""
from copy import deepcopy as _deepcopy

import numpy as _np

from apsuite.orbcorr import OrbitCorr as _OrbitCorr

from .functions import _SET_FUNCS, calc_vdisp as _vdisp, rmk_orbit_corr
from .si_data import (
    model_base,
    si_elems,
    si_famdata,
    si_spos,
    std_misaligment_tolerance,
    std_misaligment_types,
)


_OC_MODEL = model_base()
_OC = _OrbitCorr(_OC_MODEL, "SI")
_OC.params.maxnriters = 30
_OC.params.convergencetol = 1e-9
_OC.params.use6dorb = True
_INIT_KICKS = _OC.get_kicks()
_JAC = _OC.get_jacobian_matrix()
_DELTAS = std_misaligment_tolerance()
_STD_TYPES = std_misaligment_types()
_fam = si_famdata()
_STD_ELEMS = si_elems()
_sects_dict = {
    fam_name: [int(s[:2]) for i, s in enumerate(_fam[fam_name]["subsection"])]
    for fam_name in _STD_ELEMS
}
_SI_SPOS = si_spos()
_anly_funcs = ["vertical_disp", "testfunc"]


class Button:
    """Button object for misaligment analysis of a single magnet."""

    def __init__(self, dtype=None, **kwargs):
        """."""
        self._elem = None
        self._sect = None
        self._indices = None

        if dtype in _STD_TYPES:
            self._dtype = dtype
        else:
            raise ValueError("Invalid dtype")

        if "func" not in kwargs:
            self._func = "vertical_disp"
        elif "func" in kwargs and kwargs["func"] in _anly_funcs:
            self._func = kwargs["func"]
        else:
            raise ValueError("Invalid func")

        if "indices" in kwargs:
            if any([k in kwargs for k in ("elem", "sect")]) and any(
                kwargs[k] is not None for k in ("elem", "sect")
            ):
                raise ValueError("Too much args")
            else:
                self._indices = kwargs["indices"]

        elif all([k in kwargs for k in ("elem", "sect")]):
            self._sect = kwargs["sect"]
            self._elem = kwargs["elem"]
        else:
            raise ValueError("Missing input args")

        self.__force_init__()

        self._isvalid = True
        if self.fantasy_name == []:
            self._isvalid = False

        self._signature = self.__calc_signature()

    def __str__(self) -> str:
        """."""
        return f"({self._sect}, {self._dtype}, {self._fantasy_name})"

    def __repr__(self) -> str:
        """."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """."""
        try:
            if (
                (self._dtype == other.dtype)
                and (self._indices == other.indices)
                and (self._fantasy_name == other.fantasy_name)
            ):
                return True
            return False
        except Exception:
            return False

    def __calc_signature(self):
        if not self._isvalid:
            return _np.zeros(160)
        if self._func == "testfunc":
            if isinstance(self._fantasy_name, list):
                return [_np.zeros(160) for i in self._fantasy_name]
            else:
                return _np.zeros(160)
        func = _SET_FUNCS[self._dtype]
        if all(isinstance(i, list) for i in self._indices):
            indices = self._indices
        elif all(isinstance(i, (int, _np.integer)) for i in self._indices):
            indices = [self._indices]
        else:
            raise ValueError("Indices with format problem")
        # the calculation:
        disp = []
        delta = _DELTAS[self._dtype][self._elem[0]]
        for ind in indices:
            disp_0 = _vdisp(_OC_MODEL)
            func(_OC_MODEL, indices=ind, values=delta)
            rmk_orbit_corr(_OC, _JAC)
            disp_p = _vdisp(_OC_MODEL)
            disp.append(((disp_p - disp_0) / delta).ravel())
            func(_OC_MODEL, indices=ind, values=0.0)
            _OC.set_kicks(_INIT_KICKS)
        return disp

    def __force_init_old__(self):
        """."""
        elem = self._elem
        fixpos = -1
        if elem is not None:
            if isinstance(elem, str):
                elem = elem.rsplit("_")
            if len(elem) == 1:
                elem = elem[0]
            else:
                elem, fixpos = elem[0], int(elem[1])

        sect = self._sect

        if sect is not None:
            if (
                not isinstance(sect, (_np.integer, int))
                or sect < 1
                or sect > 20
            ):
                raise ValueError("problem with sect")

        indices = self._indices
        # print(elem, fixpos, sect, indices)
        split_flag = False
        if indices is not None:
            if isinstance(indices, (int, _np.integer)):
                indices = [indices]
            elif isinstance(indices, (_np.ndarray, list, tuple)) and all(
                isinstance(i, (_np.integer, int)) for i in indices
            ):
                pass
            else:
                ValueError("indices passed in wrong format")

            found_elems = [
                fname
                for fname in list(
                    set([_OC_MODEL[int(idx)].fam_name for idx in indices])
                )
                if fname in _STD_ELEMS
            ]
            if len(found_elems) != 1:
                raise ValueError("invalid indices")
            elem = found_elems.pop()
            indices = [f for f in _fam[elem]["index"] if indices[0] in f]
            if len(indices) == 1:
                indices = indices[0]
        if indices is None:
            indices = [
                _fam[elem]["index"][i]
                for i, s in enumerate(_sects_dict[elem])
                if s == sect
            ]
            if len(indices) == 1:
                if (
                    isinstance(indices[0], (list, tuple, _np.ndarray))
                    and len(indices[0]) > 1
                ):
                    indices = indices[0]
            else:
                split_flag = True

        spos = (
            [_SI_SPOS[i[0]] for i in indices]
            if split_flag
            else _SI_SPOS[indices[0]]
        )
        if (
            len(set(_sects_dict[elem])) != len(_sects_dict[elem])
            and split_flag is False
        ):
            pos = 0
            for i, s in enumerate(_sects_dict[elem]):
                if s == sect:
                    pos += 1
                    if indices[0] in _fam[elem]["index"][i]:
                        break
            fantasy_name = elem + "_" + str(pos)
        elif (
            len(set(_sects_dict[elem])) != len(_sects_dict[elem])
            and split_flag is True
        ):
            fantasy_name = [
                elem + "_" + str(i + 1) for i in range(len(indices))
            ]
        else:
            fantasy_name = elem
        if fixpos == -1:
            self._fantasy_name = fantasy_name
            self._sect = sect
            self._spos = spos
            self._indices = indices
        else:
            self._fantasy_name = fantasy_name[fixpos - 1]
            self._sect = sect
            self._spos = spos[fixpos - 1]
            self._indices = indices[fixpos - 1]

    @property
    def func(self):
        """."""
        return self._func

    @property
    def is_valid(self):
        """."""
        return self._isvalid

    @property
    def dtype(self):
        """."""
        return self._dtype

    @property
    def elem(self):
        """."""
        return self._elem

    @property
    def sect(self):
        """."""
        return self._sect

    @property
    def indices(self):
        """."""
        return self._indices

    @property
    def signature(self):
        """."""
        return self._signature

    @property
    def spos(self):
        """."""
        return self._spos

    @property
    def fantasy_name(self):
        """."""
        return self._fantasy_name

    def flatten(self):
        """Split button.

        Returns:
            soon.
        """
        if not isinstance(self, Button):
            print("arg is not a Button object")
            return
        if isinstance(self.fantasy_name, list):
            buttons = []
            # print(f'spliting: {self}', end=' ')
            for i in range(len(self.fantasy_name)):
                b = _deepcopy(self)
                b._signature = self.signature[i]
                b._fantasy_name = self.fantasy_name[i]
                b._indices = self.indices[i]
                buttons.append(b)
            # print(buttons)
            return buttons
        else:
            # print('already individual:', self)
            return self

    def __force_init__(self):
        """."""
        # Extract elements
        elem, fixpos, sect, indices = (
            self._elem,
            -1,
            self._sect,
            self._indices,
        )

        # Handle sector
        sect = self._handle_sector(sect)

        # Handle indices
        elem, sect, indices, split_flag = self._handle_indices(
            elem, sect, indices
        )

        # Handle fantasy name
        fantasy_name = self._handle_fantasy_name(
            elem, sect, indices, split_flag, fixpos
        )

        # Update attributes
        self._update_attributes(
            fantasy_name, sect, indices, split_flag, fixpos
        )

    def _handle_sector(self, sect) -> int:
        """."""
        if sect is not None:
            if (
                not isinstance(sect, (_np.integer, int))
                or sect < 1
                or sect > 20
            ):
                raise ValueError("problem with sect")
            else:
                return int(sect)
        else:
            return int(sect)

    def _handle_indices(self, elem, sect, indices):
        """Handle indices logic."""
        split_flag = False

        if indices is None:
            indices = [
                _fam[elem]["index"][i]
                for i, s in enumerate(_sects_dict[elem])
                if s == sect
            ]
            if len(indices) == 1:
                if (
                    isinstance(indices[0], (list, tuple, _np.ndarray))
                    and len(indices[0]) > 1
                ):
                    indices = indices[0]
            else:
                split_flag = True

        else:
            if isinstance(indices, (int, _np.integer)):
                indices = [indices]
            elif isinstance(indices, (_np.ndarray, list, tuple)) and all(
                isinstance(i, (_np.integer, int)) for i in indices
            ):
                pass
            else:
                raise ValueError("indices passed in wrong format")

            # Logic for handling indices
            elem, indices, split_flag = self._process_indices(elem, indices)

        return elem, sect, indices, split_flag

    def _process_indices(self, elem, indices):
        """Process indices logic."""
        found_elems = [
            fname
            for fname in list(
                set([_OC_MODEL[int(idx)].fam_name for idx in indices])
            )
            if fname in _STD_ELEMS
        ]
        if len(found_elems) != 1:
            raise ValueError("invalid indices")
        elem = found_elems.pop()
        indices = [
            ind
            for ind in _fam[elem]["index"]
            if all(i in ind for i in indices)
        ]
        if len(indices) == 1:
            indices = indices[0]

        return elem, indices, True if len(indices) != 1 else False

    def _handle_fantasy_name(self, elem, sect, indices, split_flag, fixpos):
        """Handle fantasy name logic."""
        # Logic for generating fantasy name
        fantasy_name = self._generate_fantasy_name(
            elem, sect, indices, split_flag
        )

        # Logic for fixing position if necessary
        if fixpos != -1:
            fantasy_name = (
                fantasy_name[fixpos - 1] if split_flag else fantasy_name
            )
        return fantasy_name

    def _generate_fantasy_name(self, elem, sect, indices, split_flag):
        """Generate fantasy name logic."""
        if (
            len(set(_sects_dict[elem])) != len(_sects_dict[elem])
            and split_flag is False
        ):
            pos = 0
            for i, s in enumerate(_sects_dict[elem]):
                if s == sect:
                    pos += 1
                    if indices[0] in _fam[elem]["index"][i]:
                        break
            fantasy_name = elem + "_" + str(pos)
        elif (
            len(set(_sects_dict[elem])) != len(_sects_dict[elem])
            and split_flag is True
        ):
            fantasy_name = [
                elem + "_" + str(i + 1) for i in range(len(indices))
            ]
        else:
            fantasy_name = elem
        return fantasy_name

    def _update_attributes(
        self, fantasy_name, sect, indices, split_flag, fixpos
    ):
        """Update instance attributes."""
        if fixpos == -1:
            self._fantasy_name = fantasy_name
            self._sect = sect
            self._spos = (
                [_SI_SPOS[i[0]] for i in indices]
                if split_flag
                else _SI_SPOS[indices[0]]
            )
            self._indices = indices
        else:
            self._fantasy_name = fantasy_name
            self._sect = sect
            self._spos = [_SI_SPOS[i[0]] for i in [indices[fixpos - 1]]]
            self._indices = [indices[fixpos - 1]]
