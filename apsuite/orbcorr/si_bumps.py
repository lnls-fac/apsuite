"""."""

import numpy as np
import matplotlib.pyplot as mplt

from pymodels import si
import pyaccel
from . import OrbitCorr


class SiCalcBumps:
    """Class to calculate bumps and related tools."""

    SS_LENGTHS = {
        "SA": 7.0358,
        "SB": 6.1758,
        "SP": 6.1758,
    }

    MARKER_NAMES = {
        "C1": "B1_SRC",
        "C2": "B2_SRC",
        "BC": "mc",
        "SA": "mia",
        "SB": "mib",
        "SP": "mip",
    }

    BPM_SEC_IDCS = {
        "C1": [0, 1],
        "C2": [2, 3],
        "BC": [3, 4],
        "SA": [-1, 0],
        "SB": [-1, 0],
        "SP": [-1, 0],
    }

    SS_NUMBERS = {
        "SA": np.arange(1, 20, 4),
        "SB": np.arange(2, 22, 2),
        "SP": np.arange(3, 20, 4),
    }

    def __init__(
        self, model=None, section_type=None, section_nr=None, n_bpms_out=None
    ):
        """."""
        self._section_type = section_type
        self._section_nr = section_nr
        self._n_bpms_out = n_bpms_out
        self._model = model
        self._mat_i2s = None
        self._mat_i2r = None
        self._mat_s2r = None

    @property
    def section_type(self):
        """Type of section where the bump will be applied.

        Returns:
            str: C1, C2, BC, SA, SB, SP
        """
        return self._section_type

    @section_type.setter
    def section_type(self, value):
        self.section_type_ = value

    @property
    def section_nr(self):
        """Number of the section where the bump will be applied.

        Returns:
            int: Section number
        """
        return self._section_nr

    @section_nr.setter
    def section_nr(self, value):
        self._section_nr = value

    @property
    def n_bpms_out(self):
        """Number of BPMs to remove from each side.

        Returns:
           int: Number of BPMs to be removed from each side
             of the BPMs used in the bump.
        """
        return self._n_bpms_out

    @n_bpms_out.setter
    def n_bpms_out(self, value):
        self._n_bpms_out = value

    @property
    def model(self):
        """SIRIUS pymodels object.

        Returns:
            SI pymodels object: Sirius storage ring model
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def mat_i2s(self):
        """Matrix relating ideal orbit @ BPMs and orbit at source.

        Returns:
            2d numpy array: Square matrix relating ideal orbit
                @ BPMs used in the bump and at source.
        """
        return self._mat_i2s

    @mat_i2s.setter
    def mat_i2s(self, value):
        self._mat_i2s = value

    @property
    def mat_i2r(self):
        """Matrix relating ideal and real orbit @ BPMS.

        Returns:
            2d numpy array: Square matrix relating ideal and real orbit
                @ BPMs used in the bump.
        """
        return self._mat_i2r

    @mat_i2r.setter
    def mat_i2r(self, value):
        self._mat_i2r = value

    @property
    def mat_s2r(self):
        """Matrix relating orbit @ source and real orbit @ BPMs.

        Returns:
            2d numpy array: Square matrix relating orbit at source
              and real orbit at the BPMs used in the bump.
        """
        return self._mat_s2r

    @mat_s2r.setter
    def mat_s2r(self, value):
        self._mat_s2r = value

    @staticmethod
    def _get_matrix_ss_section(leng):
        return np.array(
            [
                [1, -leng / 2, 0, 0],
                [1, leng / 2, 0, 0],
                [0, 0, 1, -leng / 2],
                [0, 0, 1, leng / 2],
            ]
        )

    def _get_sec_bpm_indices(self, section_type=None):
        if section_type is None:
            section_type = self.section_type
        bdict = self.BPM_SEC_IDCS
        return bdict[section_type][0], bdict[section_type][1]

    def print_ss_section(self):
        """Print straight section numbers."""
        print("Straight section numbers: ")
        print("SA:", end=" ")
        print(self.SS_NUMBERS["SA"])
        print("SB:", end=" ")
        print(self.SS_NUMBERS["SB"])
        print("SP:", end=" ")
        print(self.SS_NUMBERS["SP"])

    def get_bpm_indices(self, section_type=None, sidx=None):
        """Get BPMs indices to set orbit for bump.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            sidx (int): Section indice. Defaults to None.

        Returns:
            1d numpy array: Indices of the BPMs where orbit bump will
              be applied.
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        bpm1_sec_index, bpm2_sec_index = self._get_sec_bpm_indices(
            section_type
        )
        idlist = np.arange(0, 160, 1)
        idcs = np.zeros(2, dtype=int)
        idcs[0] = idlist[bpm1_sec_index + 8 * sidx]
        idcs[1] = idlist[bpm2_sec_index + 8 * sidx]
        idcs = np.array(idcs)
        idcs = np.tile(idcs, 2)
        idcs[2:] += 160
        return idcs

    def get_closest_bpms_indices(
        self, section_type=None, sidx=None, n_bpms_out=None
    ):
        """Get closest BPMs indices to remove from orbit correction.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to 'C1'.
            sidx (int): Section indice. Defaults to None.
            n_bpms_out (int): Nr of BPMs to remove from each side.
              Defaults to None.

        Returns:
            1d numpy array: Indices of the closest BPMS.
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        if n_bpms_out is None:
            n_bpms_out = self.n_bpms_out
        bpm1_sec_index, bpm2_sec_index = self._get_sec_bpm_indices(
            section_type
        )
        idlist = np.arange(0, 160, 1)
        idcs_ignore = list()
        for i in np.arange(n_bpms_out):
            idcs_ignore.append(idlist[bpm1_sec_index + 8 * sidx - (i + 1)])
            idcs_ignore.append(idlist[bpm2_sec_index + 8 * sidx + (i + 1)])
        idcs_ignore = np.array(idcs_ignore)
        idcs_ignore = np.tile(idcs_ignore, 2)
        idcs_ignore[n_bpms_out * 2 :] += 160
        return idcs_ignore

    def get_btwbpm_corrs_indices(self, section_type=None, sidx=None):
        """Get correctors indices between BPMs and source.

        Args:
        section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
            Defaults to 'C1'.
        sidx (int): Section indice. Defaults to 0.

        Returns:
            1d numpy array: Indices of the corrector to be removed from
              correction.
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        if "S" in section_type:
            return [], []

        ch_idcs = dict()
        cv_idcs = dict()
        ch_idcs["C1"] = [sidx * 6 + 0, sidx * 6 + 1]
        cv_idcs["C1"] = [sidx * 8 + 0, sidx * 8 + 1]

        ch_idcs["C2"] = [sidx * 6 + 2]
        cv_idcs["C2"] = [sidx * 8 + 2, sidx * 8 + 3]

        ch_idcs["BC"] = []
        cv_idcs["BC"] = []

        return ch_idcs[section_type], cv_idcs[section_type]

    def get_source_marker_idx(self, model=None, section_type=None, sidx=None):
        """Get source marker index in the model.

        Args:
            model (Pymodels object): SIRIUS model
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to 'C1'.
            sidx (int): Section indice. Defaults to 0.

        Returns:
            1d numpy array: Indice of source marker in the model.
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        if model is None:
            model = self.model
        if "S" not in section_type:
            nr = 1 if (section_type == "BC") else 2
            marker = pyaccel.lattice.find_indices(
                model, "fam_name", self.MARKER_NAMES[section_type]
            )[nr * sidx]
            return marker

        mia = pyaccel.lattice.find_indices(model, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(model, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(model, "fam_name", "mip")
        idcs = np.sort(mia + mib + mip)

        if sidx % 4 == 0:
            if section_type != "SA":
                raise ValueError(
                    "section {:.0f} is a SA section!".format(sidx + 1)
                )

        elif ((sidx - 1) % 4 == 0) or ((sidx - 3) % 4 == 0):
            if section_type != "SB":
                raise ValueError(
                    "section {:.0f} is a SB section!".format(sidx + 1)
                )

        elif (sidx - 2) % 4 == 0:
            if section_type != "SP":
                raise ValueError(
                    "section {:.0f} is a SP section!".format(sidx + 1)
                )

        marker = idcs[sidx]
        return marker

    def remove_corrs_btwbpm(self, orbcorr, section_type=None, sidx=None):
        """Remove correctors between BPMs.

        Args:
            orbcorr (OrbCorr object): OrbCorr object
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            sidx (int): Section indice. Defaults to None.

        Returns:
            orbcorr: OrbCorr object
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        ch_idcs, cv_idcs = self.get_btwbpm_corrs_indices(section_type, sidx)
        if len(ch_idcs) != 0:
            orbcorr.params.enbllistch[ch_idcs] = False
        if len(cv_idcs) != 0:
            orbcorr.params.enbllistcv[cv_idcs] = False
        return orbcorr

    def remove_closest_bpms(
        self, orbcorr, section_type=None, sidx=None, n_bpms_out=None
    ):
        """Remove closest BPMs form orbit correction.

        Args:
            orbcorr (OrbCorr object): OrbCorr object
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            sidx (int): Section indice. Defaults to None.
            n_bpms_out (int): Nr of BPMs to remove from each side.
              Defaults to None.

        Returns:
            orbcorr: OrbCorr object
        """
        if section_type is None:
            section_type = self.section_type
        if sidx is None:
            sidx = self.section_nr - 1
        if n_bpms_out is None:
            n_bpms_out = self.n_bpms_out
        idcs_ignore = self.get_closest_bpms_indices(
            section_type, sidx, n_bpms_out
        )
        if idcs_ignore.size != 0:
            orbcorr.params.enbllistbpm[idcs_ignore] = False
        return orbcorr

    def calc_matrices(
        self,
        section_type=None,
        section_nr=None,
        n_bpms_out=None,
        use_ss_tfm=False,
        minsingval=0.2,
        deltax=10e-6,
    ):
        """Calculate bump matrices.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            section_nr (int): Section number. Defaults to None.
            n_bpms_out (int): Nr of BPMs to remove from each side.
              Defaults to None.
            use_ss_tfm (bool, optional): Use straight section transfer matrix.
              Defaults to False.
            minsingval (float, optional): Minimum singular value.
                Defaults to 0.2.
            deltax (float, optional): delta kick to calculate jacobian.
              Defaults to 10e-6.

        Raises:
            ValueError: The cavity mus be turned ON in the model.

        Returns:
            2d numpy arrays: Bump matrices
        """
        # NOTE: valid only for bendings in subsectors C1, C2 or BC

        if section_type is None:
            section_type = self.section_type
        if section_nr is None:
            section_nr = self.section_nr
        if n_bpms_out is None:
            n_bpms_out = self.n_bpms_out

        if "S" in section_type and use_ss_tfm:
            length = self.SS_LENGTHS[section_type]
            mat_s2r = self._get_matrix_ss_section(length)
            self.mat_s2r = mat_s2r
            return None, None, mat_s2r

        # Create accelerator and orbcorr
        if self.model is None:
            mod = si.create_accelerator()
            mod.cavity_on = True
        elif self.model.cavity_on is False:
            raise ValueError("Model cavity must be turned on!")
        orbcorr = OrbitCorr(mod, "SI", use6dorb=True)
        orbcorr.params.enblrf = True
        orbcorr.params.tolerance = 1e-9
        orbcorr.params.minsingval = minsingval

        # Get source marker idx
        sidx = max(min(section_nr, 20), 1)
        sidx -= 1
        marker = self.get_source_marker_idx(
            orbcorr.respm.model, section_type, sidx
        )

        orb0 = orbcorr.get_orbit()
        kicks0 = orbcorr.get_kicks()

        # Get BPM indices
        idcs = self.get_bpm_indices(section_type, sidx)

        # remove corrs between BPMs
        orbcorr = self.remove_corrs_btwbpm(orbcorr, section_type, sidx)

        # remove closest BPMS
        orbcorr = self.remove_closest_bpms(
            orbcorr, section_type, sidx, n_bpms_out
        )

        mat_i2s = np.zeros((4, 4), dtype=float)
        mat_i2r = np.zeros((4, 4), dtype=float)
        for i, idx in enumerate(idcs):
            gorb = orb0.copy()
            orbcorr.set_kicks(kicks0)

            gorb[idx] += deltax / 2
            orbcorr.correct_orbit(goal_orbit=gorb)
            orbp = orbcorr.get_orbit()[idcs]
            b2p = pyaccel.tracking.find_orbit(
                orbcorr.respm.model, indices="open"
            )
            b2p = b2p[0:4, marker]

            gorb[idx] -= deltax
            orbcorr.correct_orbit(goal_orbit=gorb)
            orbn = orbcorr.get_orbit()[idcs]
            b2n = pyaccel.tracking.find_orbit(
                orbcorr.respm.model, indices="open"
            )
            b2n = b2n[0:4, marker]

            mat_i2s[:, i] = (b2p - b2n) / deltax
            mat_i2r[:, i] = (orbp - orbn) / deltax

        mat_s2r = np.linalg.solve(mat_i2s.T, mat_i2r.T).T
        self.mat_i2s = mat_i2s
        self.mat_i2r = mat_i2r
        self.mat_s2r = mat_s2r
        return mat_i2s, mat_i2r, mat_s2r

    def test_matrices(
        self,
        section_type=None,
        section_nr=None,
        flag_n_bpms=True,
        flag_singvals=True,
    ):
        """Test bump matrices with diferent parameters.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            section_nr (int): Section number. Defaults to None.
            flag_n_bpms (bool, optional): Test with different bpms number.
                Defaults to True.
            flag_singvals (bool, optional): Test with different singular
                values. Defaults to True.

        Returns:
            matplotlib figure: Matplotlib figure object and axis
        """
        ms_i2s = []
        ms_i2r = []
        ms_s2r = []
        cases = []
        svals = 0.2
        if flag_n_bpms:
            for n_bpms in [0, 1, 2]:
                cases.append((n_bpms, svals))
                m_i2s, m_i2r, m_s2r = self.calc_matrices(
                    minsingval=svals,
                    section_nr=section_nr,
                    section_type=section_type,
                    n_bpms_out=n_bpms,
                )
                ms_i2s.append(m_i2s)
                ms_i2r.append(m_i2r)
                ms_s2r.append(m_s2r)

        n_bpms = 0
        if flag_singvals:
            for svals in [0.2, 2, 20]:
                cases.append((n_bpms, svals))
                m_i2s, m_i2r, m_s2r = self.calc_matrices(
                    minsingval=svals,
                    section_nr=section_nr,
                    section_type=section_type,
                    n_bpms_out=n_bpms,
                )
                ms_i2s.append(m_i2s)
                ms_i2r.append(m_i2r)
                ms_s2r.append(m_s2r)

        fig, (a_i2s, a_i2r, a_s2r) = mplt.subplots(3, 1, figsize=(6, 9))

        for m_i2s, m_i2r, m_s2r, case in zip(ms_i2s, ms_i2r, ms_s2r, cases):
            lab = f"nbpm={case[0]:d} svals={case[1]:.2f}"
            a_i2s.plot(m_i2s.ravel(), "-o", label=lab)
            a_i2r.plot(m_i2r.ravel(), "-o", label=lab)
            a_s2r.plot(m_s2r.ravel(), "-o", label=lab)

        a_i2r.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
        fig.tight_layout()
        return fig, (a_i2s, a_i2r, a_s2r)

    def calculate_bumps(
        self,
        section_type=None,
        section_nr=None,
        n_bpms_out=None,
        m_s2r=None,
        use_ss_tfm=False,
        posx=0,
        angx=0,
        posy=0,
        angy=0,
    ):
        """Calculate bumps.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            section_nr (int): Section number. Defaults to None.
            n_bpms_out (int): Nr of BPMs to remove from each side.
              Defaults to None.
            m_s2r (_type_, optional): Matrix source to real orbit @ BPM.
              Defaults to None.
            use_ss_tfm (bool, optional): Use straight section transfer matrix.
              Defaults to False.
            posx (float): Horizontal position [um]. Defaults to 0.
            angx (float): Horizontal angle [urad]. Defaults to 0.
            posy (float): Vertical position [um]. Defaults to 0.
            angy (float): Vertical angle [urad]. Defaults to 0.

        Raises:
            ValueError: The cavity mus be turned ON in the model.

        Returns:
            1d numpy array: Goal orbit with bump
        """
        if section_type is None:
            section_type = self.section_type
        if section_nr is None:
            section_nr = self.section_nr
        if n_bpms_out is None:
            n_bpms_out = self.n_bpms_out

        # Get bump matrices
        if m_s2r is None:
            _, _, m_s2r = self.calc_matrices(
                section_type=section_type,
                minsingval=0.2,
                section_nr=section_nr,
                n_bpms_out=n_bpms_out,
                use_ss_tfm=use_ss_tfm,
            )
        sidx = max(min(section_nr, 20), 1)
        sidx -= 1
        vec = np.array([posx, angx, posy, angy])

        # Create accelerator and orbcorr
        if self.model is None:
            mod = si.create_accelerator()
            mod.cavity_on = True
        elif self.model.cavity_on is False:
            raise ValueError("Model cavity must be turned on!")
        orbcorr = OrbitCorr(mod, "SI", use6dorb=True)
        orbcorr.params.enblrf = True
        orbcorr.params.tolerance = 1e-9
        orbcorr.params.minsingval = 0.2

        # Get BPM indices
        idcs = self.get_bpm_indices(section_type, sidx)

        # remove corrs between BPMs
        orbcorr = self.remove_corrs_btwbpm(orbcorr, section_type, sidx)

        # remove closest BPMS
        orbcorr = self.remove_closest_bpms(
            orbcorr, section_type, sidx, n_bpms_out
        )

        gorb = orbcorr.get_orbit()

        x = np.dot(m_s2r, vec)
        gorb[idcs] = x
        return gorb, orbcorr

    def test_bumps(
        self,
        section_type=None,
        section_nr=None,
        n_bpms_out=None,
        m_s2r=None,
        use_ss_tfm=False,
        plot_results=True,
        posx=100e-6,
        angx=50e-6,
        posy=100e-6,
        angy=50e-6,
    ):
        """Compare desired bump with bump to be applied.

        Args:
            section_type (str): Bump section (C1, C2, BC, SA, SB, SP).
                Defaults to None.
            section_nr (int): Section number. Defaults to None.
            n_bpms_out (int): Nr of BPMs to remove from each side.
              Defaults to None.
            m_s2r (_type_, optional): Matrix source to real orbit @ BPM.
              Defaults to None.
            use_ss_tfm (bool, optional): Use straight section transfer matrix.
              Defaults to False.
            plot_results (bool, optional): _description_. Defaults to True.
            posx (float): Horizontal position [um]. Defaults to 100e-6.
            angx (float): Horizontal angle [urad]. Defaults to 50e-6.
            posy (float): Vertical position [um]. Defaults to 100e-6.
            angy (float): Vertical angle [urad]. Defaults to 50e-6.

        Returns:
            tuple 1d numpy array: bump at source, kicks necessary to
            implement bump, orbit in the whole ring
        """
        if section_type is None:
            section_type = self.section_type
        if section_nr is None:
            section_nr = self.section_nr
        if n_bpms_out is None:
            n_bpms_out = self.n_bpms_out

        sidx = max(min(section_nr, 20), 1)
        sidx -= 1

        gorb, orbcorr = self.calculate_bumps(
            section_type,
            section_nr,
            n_bpms_out,
            m_s2r,
            use_ss_tfm,
            posx,
            angx,
            posy,
            angy,
        )

        vec = np.array([posx, angx, posy, angy])
        orbcorr.correct_orbit(goal_orbit=gorb)
        marker = self.get_source_marker_idx(
            orbcorr.respm.model, section_type, sidx
        )
        xres = pyaccel.tracking.find_orbit(
            orbcorr.respm.model, indices="open"
        )[0:4, marker]

        kicks = orbcorr.get_kicks()[:-1] * 1e6
        orbit = orbcorr.get_orbit() * 1e6
        if plot_results:
            fig, (ax, ay, az) = mplt.subplots(3, 1, figsize=(6, 9))

            ax.plot(
                1e6 * vec, "-o", label="Input bump (posx, angx, posy, angy)"
            )
            ax.plot(1e6 * xres, "-o", label="Resultant bump ")
            ax.legend()

            ay.plot(kicks)
            ay.set_ylabel("Corr. kicks [urad]")
            ay.set_xlabel("Corr idx")

            az.plot(orbit)
            az.set_ylabel("Orbit [urad]")
            az.set_xlabel("BPMS idx")

            fig.tight_layout()
            mplt.show()
        return xres, kicks, orbit
