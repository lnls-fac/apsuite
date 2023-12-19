"""Module 'buttons' for the class Object Button."""

from copy import deepcopy as _deepcopy
from importlib import reload

import numpy as _np

from apsuite.orbcorr import OrbitCorr as _OrbitCorr

from . import functions, si_data

_OC_MODEL = None
_OC = None
_INIT_KICKS = None
_JAC = None
_SI_SPOS = None
_fam = None
_sects_dict = None

_anly_funcs = ["vertical_disp", "testfunc"]
_DELTAS = si_data.std_misaligment_tolerance()
_STD_TYPES = si_data.std_misaligment_types()
_STD_ELEMS = si_data.si_elems()


def buttons_set_model(model=None):
    """."""
    # print("entered buttons set model")
    si_data.__set_model(model)
    reload(functions)
    # print("buttons -> updating model")
    mod = si_data.get_model()
    # print("buttons -> model is", type(mod))
    globals()["_OC_MODEL"] = mod
    globals()["_OC"] = _OrbitCorr(globals()["_OC_MODEL"], "SI")
    globals()["_OC"].params.maxnriters = 30
    globals()["_OC"].params.convergencetol = 1e-9
    globals()["_OC"].params.use6dorb = True
    globals()["_JAC"] = globals()["_OC"].get_jacobian_matrix()
    functions.rmk_orbit_corr(
        globals()["_OC"], jacobian_matrix=globals()["_JAC"]
    )
    globals()["_INIT_KICKS"] = globals()["_OC"].get_kicks()
    globals()["_SI_SPOS"] = si_data.si_spos()
    globals()["_fam"] = si_data.si_famdata()
    globals()["_sects_dict"] = {
        fam_name: [
            int(s[:2])
            for i, s in enumerate(globals()["_fam"][fam_name]["subsection"])
        ]
        for fam_name in _STD_ELEMS
    }
    # print("sects_dict is", type(_sects_dict))
    # print("buttons -> model updated")


class Button:
    """Button object for misalignment analysis.

    Args:
            elem (str): Magnet's family name.\
            The valid options are the SIRIUS's dipoles, quadrupoles and \
            sextupoles family names.\
            The names can be followed by postfixes like: '_1' or '_2' to \
            specify the magnet.

            sect (int): The magnet sector location.\
            The valid options are the 1 to 20 sectors of SIRIUS storage ring.

            dtype (str): The misalignmente type.\
            Valid options are: 'dx'(horizontal), 'dy' (vertical), 'dr' \
            (rotation roll), 'drp' (rotation pitch) and 'dry' (totation yaw).

            func (str): The analysis function. Defaults to "vertical_disp".\
            The valid options are: 'vertical_disp' and 'testfunc'.

            indices ((int, list[int])): The index or indices of the magnet in\
            the SIRIUS pymodels's model. Its important to have the latest\
            model always up-to-date.

    """

    def __init__(
        self,
        elem=None,
        sect=None,
        dtype=None,
        func="vertical_disp",
        indices=None,
    ):
        """."""
        if _OC_MODEL is None:
            fstr = "Creating new model."
            fstr += " To switch model, use function:"
            fstr += " 'apsuite.optics_analysis.misalign_analysis.set_model()'"
            print(fstr)
            buttons_set_model()

        self._elem = elem
        self._sect = sect
        self._indices = indices

        if dtype in _STD_TYPES:
            self._dtype = dtype
        else:
            raise ValueError("Invalid dtype")

        if func not in _anly_funcs:
            raise ValueError("Invalid func")
        self._func = func

        if indices is not None:
            if any(f is not None for f in [elem, sect]):
                raise ValueError("Too much args")
            else:
                self._indices = indices
        elif indices is None and all(f is not None for f in [elem, sect]):
            self._sect = sect
            self._elem = elem
        else:
            raise ValueError("Missing input args")

        self.__force_init__()

        self._is_valid = True  # if self.fantasy_name == [] else False

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
        if self._func == "testfunc":
            if isinstance(self._fantasy_name, list):
                return [_np.zeros(160) for i in self._fantasy_name]
            else:
                return _np.zeros(160)

        else:
            func = functions._SET_FUNCS[self._dtype]
            disp = []
            delta = _DELTAS[self._dtype][self._elem[0]]
            if isinstance(self._fantasy_name, list):
                loop = self.indices
                flag = -1
            else:
                loop = [self.indices]
                flag = 1

            for ind in loop:
                disp_0 = functions.calc_vdisp(_OC_MODEL)
                func(_OC_MODEL, indices=ind, values=delta)
                functions.rmk_orbit_corr(_OC, _JAC)
                disp_p = functions.calc_vdisp(_OC_MODEL)
                disp.append(((disp_p - disp_0) / delta).ravel())
                func(_OC_MODEL, indices=ind, values=0.0)
                _OC.set_kicks(_INIT_KICKS)

            if flag == 1:
                return disp[0]
            else:
                return disp

    @property
    def func(self):
        """."""
        return self._func

    @property
    def is_valid(self):
        """."""
        return self._is_valid

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
        """."""
        if not isinstance(self, Button):
            print("arg is not a Button object")
            return
        if isinstance(self.fantasy_name, list):
            buttons = []
            for i in range(len(self.fantasy_name)):
                b = _deepcopy(self)
                b._signature = self.signature[i]
                b._fantasy_name = self.fantasy_name[i]
                b._indices = self.indices[i]
                buttons.append(b)
            return buttons
        else:
            return [self]

    def __force_init__(self):
        """."""
        # Extract elements
        elem, sect, indices = (self._elem, self._sect, self._indices)

        # Handle sector
        sect = self._handle_sector(sect)

        # Handle elem
        elem, fixpos = (
            self._handle_elem(elem, sect) if elem is not None else (None, -1)
        )

        # Handle indices
        elem, sect, indices, split_flag = self._handle_indices(
            elem, sect, indices
        )

        # Handle fantasy name
        fantasy_name = self._handle_fantasy_name(elem, indices, split_flag)

        # Update attributes
        self._update_attributes(elem, sect, fantasy_name, indices, fixpos)

    def _handle_elem(self, elem, sect):
        """."""
        fixpos = -1
        elem = elem.split("_")
        if len(elem) == 1:
            elem = elem[0]
            fixpos = -1
        else:
            fixpos = int(elem[1])
            elem = elem[0]
            if fixpos > _sects_dict[elem].count(sect) or fixpos <= 0:
                raise ValueError("invalid postfix number")

        return elem, fixpos

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
            return sect

    def _handle_indices(self, elem, sect, indices):
        """."""
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

            elem, indices, split_flag = self._process_indices(elem, indices)

        return elem, sect, indices, split_flag

    def _process_indices(self, elem, indices):
        """."""
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

    def _handle_fantasy_name(self, elem, indices, split_flag):
        """Handle fantasy name logic."""
        if split_flag is True:
            fantasy_name = [
                elem + "_" + str(i + 1) for i in range(len(indices))
            ]
        else:
            fantasy_name = elem
        return fantasy_name

    def _update_attributes(self, elem, sect, fantasy_name, indices, fixpos):
        """Update instance attributes."""
        self._elem = elem
        self._sect = sect
        self._fantasy_name = fantasy_name
        self._indices = indices
        self._spos = [_SI_SPOS[i] for i in indices]

        if fixpos != -1:
            self._fantasy_name = self._fantasy_name[fixpos - 1]
            self._indices = self._indices[fixpos - 1]
            self._spos = self._spos[fixpos - 1]
