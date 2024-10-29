import warnings
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import dfttools.utils.file_utils as fu
from dfttools.base_parser import BaseParser
from dfttools.geometry import AimsGeometry


class Output(BaseParser):
    """
    Base class for parsing output files from electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new supported file
    type to _supported_files, call the super().__init__ method, include the new file
    type as a kwarg in the super().__init__ call. Optionally include the self.lines line
    in examples.

    ...

    Attributes
    ----------
    supported_files
    lines

    Examples
    --------
    class AimsOutput(Output):
        def __init__(self, aims_out: str = "aims.out"):
            super().__init__(aims_out=aims_out)
            self.lines = self._file_contents["aims_out"]
    """

    # FHI-aims, ...
    _supported_files = ["aims_out", "elsi_out"]

    def __init__(self, **kwargs: str):
        super().__init__(self._supported_files, **kwargs)

        for val in kwargs.keys():
            fu.check_required_files(self._supported_files, val)

    @property
    def supported_files(self):
        return self._supported_files

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value


class AimsOutput(Output):
    """
    FHI-aims output file parser.

    ...

    Attributes
    ----------
    lines
    aims_out : str
        The path to the aims.out file
    """

    def __init__(self, aims_out: str = "aims.out"):
        super().__init__(aims_out=aims_out)
        self.lines = self.file_contents["aims_out"]

        # Check if the aims.out file was provided
        fu.check_required_files(self._supported_files, "aims_out")

        # Enforce the the first file parsed in self.file_paths is the aims.out file
        if list(self.file_paths.keys())[0] != "aims_out":
            raise ValueError(
                "The aims.out file must be the first file provided."
            )

        self.aims_out_path = self.file_paths["aims_out"]

    def get_number_of_atoms(self) -> int:
        """
        Return number of atoms in unit cell

        Returns
        -------
        int
            Number of atoms in the unit cell
        """
        n_atoms = None

        for l in self.lines:
            if "| Number of atoms" in l:
                n_atoms = int(l.strip().split()[5])

        if n_atoms is None:
            raise ValueError("Number of atoms not found in aims.out file")

        return n_atoms

    def get_geometry(self):
        """
        Extract the geometry file from the aims output and return it as a
        Geometry object

        Returns
        -------
        geometry
            Geometry object
        """
        geometry_lines = []
        geometry_file_reached = False
        for l in self.lines:
            if (
                "Parsing geometry.in (first pass over file, find array dimensions only)."
                in l
            ):
                geometry_file_reached = True
            if geometry_file_reached:
                geometry_lines.append(l)
            if "Completed first pass over input file geometry.in ." in l:
                break
        geometry_text = "\n".join(geometry_lines[6:-3])
        geometry = AimsGeometry()
        geometry.parse(geometry_text)

        return geometry

    def get_geometry_steps_of_optimisation(
        self, nr_of_occurrence=None
    ) -> list:
        """
        Get a list of all geometry steps performed.

        Parameters
        ----------
        n_occurrence : int or None
            If there are multiple energies in a file (e.g. during a geometry
            optimization) this parameters allows to select which energy is
            returned. If set to -1 the last one is returned (e.g. result of a
            geometry optimization), if set to None, all values will be returned
            as a numpy array.

        Returns
        -------
        geometry_files : list
            List of geometry objects.

        """
        geometry_files = [self.get_geometry()]  # append initial geometry

        state = 0
        # 0... before geometry file,
        # 1... between start of geometry file and lattice section
        # 2... in lattice section of geometry file
        # 3... in atoms section of geometry file

        for l in self.lines:
            if (
                "Updated atomic structure:" in l
                or "Atomic structure that was used in the preceding time step of the wrapper"
                in l
            ):
                state = 1
                geometry_lines = []

            if state > 0 and "atom " in l:
                state = 3
            if state == 1 and "lattice_vector  " in l:
                state = 2

            if state > 0:
                geometry_lines.append(l)

            if state == 3 and not "atom " in l:
                state = 0
                geometry_text = "".join(geometry_lines[2:-1])
                g = AimsGeometry()
                g.parseTextAIMS(geometry_text)
                geometry_files.append(g)

        return geometry_files

    def get_parameters(self) -> str:
        """
        Extract the control file from the aims output and return it as a
        parameters object

        Returns
        -------
        str
            The control file as a string
        """
        control_lines = []
        control_file_reached = False
        for l in self.lines:
            if (
                "Parsing control.in (first pass over file, find array dimensions only)."
                in l
            ):
                control_file_reached = True
            if control_file_reached:
                control_lines.append(l)
            if "Completed first pass over input file control.in ." in l:
                break

        control_text = "\n".join(control_lines[6:-3])

        return control_text

    def check_exit_normal(self) -> bool:
        """
        Check if the FHI-aims calculation exited normally.

        Returns
        -------
        bool
            whether the calculation exited normally or not
        """

        if "Have a nice day." == self.lines[-2].strip():
            exit_normal = True
        else:
            exit_normal = False

        return exit_normal

    def get_time_per_scf(self) -> npt.NDArray[np.float64]:
        """
        Calculate the average time taken per SCF iteration.

        Returns
        -------
        npt.NDArray[np.float64]
            The average time taken per SCF iteration.
        """

        # Get the number of SCF iterations
        n_scf_iters = self.get_n_scf_iters()
        scf_iter_times = np.zeros(n_scf_iters)

        # Get the time taken for each SCF iteration
        iter_num = 0
        for line in self.lines:
            if "Time for this iteration" in line:
                scf_iter_times[iter_num] = float(line.split()[-4])
                iter_num += 1

        return scf_iter_times

    ###############################################################################
    #                                   Energies                                  #
    ###############################################################################
    def _get_energy(
        self,
        n_occurrence,
        search_string,
        token_nr=None,
        energy_invalid_indicator=None,
        energy_valid_indicator=None,
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Generalized energy parser

        Parameters
        ----------
        n_occurrence : int or None
            If there are multiple energies in a file (e.g. during a geometry optimization)
            this parameters allows to select which energy is returned.
            If set to -1 the last one is returned (e.g. result of a geometry optimization),
            if set to None, all values will be returned as a numpy array.
        search_string : str
            string to be searched in the output file
        token_nr : int
            take n-th element of found line
        energy_invalid_indicator
            In some cases an energy value can be found in the output file although it is invalid -> ignore this value
            example: a line having 'restarting mixer to attempt better convergence'
                        indicates that this scf-cycle leads to invalid energies
        param energy_valid_indicator
            In some cases the value is only valid after a certain phrase is used -> ignore all values before
            example: The post-SCF vdW energy correction is 0.00 until the SCF is converged.

        Returns
        -------
        energies : Union[float, npt.NDArray[np.float64]]
            Energies that have been grepped

        """
        skip_next_energy = (
            False  # only relevant if energy_invalid_indicator is not None
        )
        use_next_energy = (
            False  # only relevant if energy_valid_indicator is not None
        )

        assert not (skip_next_energy and use_next_energy), (
            "AIMSOutput._get_energy: usage of skip_next_energy and "
            "use_next_energy at the same function call is undefined!"
        )
        # energy (in)valid indicator allows now for multiple values, if a list is provided. Otherwise, verything works out as before.
        if energy_valid_indicator is not None and not isinstance(
            energy_valid_indicator, list
        ):
            energy_valid_indicator = [
                energy_valid_indicator,
            ]

        if energy_invalid_indicator is not None and not isinstance(
            energy_invalid_indicator, list
        ):
            energy_invalid_indicator = [
                energy_invalid_indicator,
            ]

        energies = []

        for line_text in self.lines:
            # check for energy_invalid_indicator:
            if energy_invalid_indicator is not None:
                for ind in energy_invalid_indicator:
                    if ind in line_text:
                        skip_next_energy = True

            if energy_valid_indicator is not None:
                for ind in energy_valid_indicator:
                    if ind in line_text:
                        use_next_energy = True
            else:
                use_next_energy = True

            if search_string in line_text:
                if skip_next_energy is True:
                    skip_next_energy = False  # reset this 'counter'
                elif use_next_energy:
                    if token_nr is None:
                        token_nr = len(search_string.split()) + 3
                    energies.append(float(line_text.strip().split()[token_nr]))
                    use_next_energy = False
                else:
                    pass

        if len(energies) == 0:
            raise ValueError(
                f"Energy not found in aims.out file for {search_string}"
            )

        energies = np.array(energies)

        if n_occurrence is None:
            return energies
        else:
            return energies[n_occurrence]

    def get_change_of_total_energy(
        self, n_occurrence=-1, energy_invalid_indicator=None
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Change of total energy",
            token_nr=6,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_change_of_forces(
        self, n_occurrence=-1, energy_invalid_indicator=None
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Change of forces",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_change_of_sum_of_eigenvalues(
        self, n_occurrence=-1, energy_invalid_indicator=None
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Change of sum of eigenvalues",
            token_nr=7,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_maximum_force(
        self, n_occurrence=-1, energy_invalid_indicator=None
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Maximum force component",
            token_nr=4,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_energy_corrected(
        self,
        n_occurrence: Union[int, None] = -1,
        skip_E_after_mixer: bool = True,
        all_scfs: bool = False,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Return the total corrected energy.
        example:
            >> "  | Total energy corrected        :         -0.183419000541342E+07 eV  <-- do not rely
                        on this value for anything but (periodic) metals"
            -> search for "| Total energy corrected" and take value on index (0-based) nr. 5

        Parameters
        ----------
        n_occurrence : Union[int, None]
            If there are multiple energies in a file (e.g. during a geometry optimization)
            this parameters allows to select which energy is returned.
            If set to -1 the last one is returned (e.g. result of a geometry optimization),
            if set to None, all values will be returned as a numpy array.

        skip_E_after_mixer : boolean
            If the scf cycles of one geometry optimisation step didn't converge,
            aims will restart the mixer and this optimisation step.
            However, it still prints out the total energy, which can be totally nonsense.
            => if skip_E_after_mixer==True:
                    ignore first total energy after 'restarting mixer to attempt better convergence'
        """
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        if all_scfs:
            return self.get_total_energy_T0(n_occurrence, skip_E_after_mixer)
        else:
            return self._get_energy(
                n_occurrence,
                search_string="| Total energy corrected",
                energy_invalid_indicator=energy_invalid_indicator,
                token_nr=5,
            )

    def get_total_energy_T0(
        self,
        n_occurrence: Union[None, int] = -1,
        skip_E_after_mixer: bool = True,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        return self._get_energy(
            n_occurrence,
            search_string="| Total energy, T -> 0",
            energy_invalid_indicator=energy_invalid_indicator,
            token_nr=9,
        )

    def get_energy_uncorrected(
        self,
        n_occurrence: Union[None, int] = -1,
        skip_E_after_mixer: bool = True,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Return uncorrected (without smearing correction) energy

        Parameters
        ----------
        n_occurrence : Union[int, None]
            see getEnergyCorrected()

        skip_E_after_mixer : bool
            If the scf cycles of one geometry optimisation step didn't converge,
            aims will restart the mixer and this optimisation step.
            However, it still prints out the total energy, which can be totally nonsense.
            => if skip_E_after_mixer==True:
                    ignore first total energy after 'restarting mixer to attempt better convergence'
        example:
            >> "  | Total energy uncorrected      :         -0.183419000513561E+07 eV"
            -> search for "| Total energy uncorrected" and take value on index (0-based) nr. 5

        Returns
        -------
        Union[float, npt.NDArray[np.float64]]
            Uncorrected energy
        """
        if skip_E_after_mixer:
            energy_invalid_indicator += [
                "restarting mixer to attempt better convergence"
            ]

        return self._get_energy(
            n_occurrence,
            search_string="| Total energy uncorrected",
            energy_invalid_indicator=energy_invalid_indicator,
            token_nr=5,
        )

    def get_energy_without_vdw(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        energy = self.get_energy_corrected(
            n_occurrence=n_occurrence,
            energy_invalid_indicator=energy_invalid_indicator,
        )

        energy_vdw = self.get_vdw_energy(
            n_occurrence=n_occurrence,
            energy_invalid_indicator=energy_invalid_indicator,
        )

        energy_without_vdw = energy - energy_vdw

        return energy_without_vdw

    def get_HOMO_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Highest occupied state",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_LUMO_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "Lowest unoccupied state",
            token_nr=5,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_vdw_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:

        search_keyword = "| vdW energy correction"
        token_nr = None

        result = self._get_energy(
            n_occurrence,
            search_keyword,
            token_nr=token_nr,
            energy_invalid_indicator=energy_invalid_indicator,
            energy_valid_indicator="Self-consistency cycle converged",
        )
        return result

    def get_exchange_correlation_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| XC energy correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_electrostatic_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Electrostatic energy ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_kinetic_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Kinetic energy ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_sum_of_eigenvalues(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Sum of eigenvalues  ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_cx_potential_correction(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| XC potential correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_free_atom_electrostatic_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Free-atom electrostatic energy:",
            token_nr=6,
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_entropy_correction(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Entropy correction ",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_hartree_energy_correction(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        return self._get_energy(
            n_occurrence,
            "| Hartree energy correction",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_ionic_embedding_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """The energy of the nuclei in the potential of the external electric
        field."""
        return self._get_energy(
            n_occurrence,
            "| Ionic    embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_density_embedding_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """The energy of the electrons (electron density) in the potential of
        the external electric field."""
        return self._get_energy(
            n_occurrence,
            "| Density  embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_nonlocal_embedding_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        Energy of non local electron interaction (i think?) in the potential
        of the electric field.

        """
        return self._get_energy(
            n_occurrence,
            "| Nonlocal embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_external_embedding_energy(
        self,
        n_occurrence: Union[None, int] = -1,
        energy_invalid_indicator: list[str] = [],
    ) -> Union[float, npt.NDArray[np.float64]]:
        """
        This is the sum of all the embedding energies.
        I.e. ionic + (electronic) density + nonlocal.

        """
        return self._get_energy(
            n_occurrence,
            "| External embedding energy",
            energy_invalid_indicator=energy_invalid_indicator,
        )

    def get_forces(
        self, n_occurrence: Union[None, int] = -1
    ) -> npt.NDArray[np.float64]:
        """
        Return forces on all atoms

        """
        natoms = self.get_number_of_atoms()
        all_force_values = []

        for j, l in enumerate(self.lines):
            if "Total atomic forces" in l:
                force_values = np.ones([natoms, 3]) * np.nan
                for i in range(natoms):
                    force_values[i, :] = [
                        float(x) for x in self.lines[j + i + 1].split()[2:5]
                    ]
                all_force_values.append(np.array(force_values))

        if len(all_force_values) == 0:
            raise ValueError(f"Forces not found in {self.aims_out_path} file")

        if n_occurrence is None:
            return np.array(all_force_values)
        else:
            return all_force_values[n_occurrence]

    def check_spin_polarised(self) -> bool:
        """
        Check if the FHI-aims calculation was spin polarised.

        Returns
        -------
        bool
            Whether the calculation was spin polarised or not
        """

        spin_polarised = False

        for line in self.lines:
            spl = line.split()
            if len(spl) == 2:
                # Don't break the loop if spin polarised calculation is found as if the
                # keyword is specified again, it is the last one that is used
                if spl[0] == "spin" and spl[1] == "collinear":
                    spin_polarised = True

                if spl[0] == "spin" and spl[1] == "none":
                    spin_polarised = False

        return spin_polarised

    def get_conv_params(self) -> dict:
        """
        Get the convergence parameters from the aims.out file.

        Returns
        -------
        dict
            The convergence parameters from the aims.out file
        """

        # Setup dictionary to store convergence parameters
        self.convergence_params = {
            "charge_density": 0.0,
            "sum_eigenvalues": 0.0,
            "total_energy": 0.0,
            "total_force": 0.0,
        }

        for line in self.lines:
            spl = line.split()
            if len(spl) > 1:
                if "accuracy" == spl[1] and "charge density" in line:
                    self.convergence_params["charge_density"] = float(spl[-1])
                if "accuracy" == spl[1] and "sum of eigenvalues" in line:
                    self.convergence_params["sum_eigenvalues"] = float(spl[-1])
                if "accuracy" == spl[1] and "total energy" in line:
                    self.convergence_params["total_energy"] = float(spl[-1])
                if "accuracy" == spl[1] and "forces:" == spl[3]:
                    self.convergence_params["total_force"] = float(spl[-1])

                # No more values to get after SCF starts
                if "Begin self-consistency loop" in line:
                    break

        return self.convergence_params

    def get_final_energy(self) -> Union[float, None]:
        """
        Get the final energy from a FHI-aims calculation.

        Returns
        -------
        Union[float, None]
            The final energy of the calculation
        """

        for line in self.lines:
            if "s.c.f. calculation      :" in line:
                return float(line.split()[-2])

    def get_n_relaxation_steps(self) -> int:
        """
        Get the number of relaxation steps from the aims.out file.

        Returns
        -------
        int
            the number of relaxation steps
        """

        n_relax_steps = 0
        for line in reversed(self.lines):
            if "Number of relaxation steps" in line:
                return int(line.split()[-1])

            # If the calculation did not finish normally, the number of relaxation steps
            # will not be printed. In this case, count each relaxation step as they were
            # calculated by checking when the SCF cycle converged.
            if "Self-consistency cycle converged." == line.strip():
                n_relax_steps += 1

        return n_relax_steps

    def get_n_scf_iters(self) -> int:
        """
        Get the number of SCF iterations from the aims.out file.

        Returns
        -------
        int
            The number of scf iterations
        """

        n_scf_iters = 0
        for line in reversed(self.lines):
            if "Number of self-consistency cycles" in line:
                return int(line.split()[-1])

            # If the calculation did not finish normally, the number of SCF iterations
            # will not be printed. In this case, count each SCF iteration as they were
            # calculated
            if "Begin self-consistency iteration #" in line:
                n_scf_iters += 1

        return n_scf_iters

    def get_i_scf_conv_acc(self) -> dict:
        """
        Get SCF convergence accuracy values from the aims.out file.

        Returns
        -------
        dict
            The scf convergence accuracy values from the aims.out file
        """

        # Read the total number of SCF iterations
        n_scf_iters = self.get_n_scf_iters()
        n_relax_steps = self.get_n_relaxation_steps() + 1

        # Check that the calculation finished normally otherwise number of SCF
        # iterations is not known
        self.scf_conv_acc_params = {
            "scf_iter": np.zeros(n_scf_iters),
            "change_of_charge": np.zeros(n_scf_iters),
            "change_of_charge_spin_density": np.zeros(n_scf_iters),
            "change_of_sum_eigenvalues": np.zeros(n_scf_iters),
            "change_of_total_energy": np.zeros(n_scf_iters),
            # "change_of_forces": np.zeros(n_relax_steps),
            "forces_on_atoms": np.zeros(n_relax_steps),
        }

        current_scf_iter = 0
        current_relax_step = 0
        # new_scf_iter = True

        for line in self.lines:
            spl = line.split()
            if len(spl) > 1:
                if "Begin self-consistency iteration #" in line:
                    # save the scf iteration number
                    self.scf_conv_acc_params["scf_iter"][current_scf_iter] = (
                        int(spl[-1])
                    )
                    # use a counter rather than reading the SCF iteration number as it
                    # resets upon re-initialisation and for each geometry opt step
                    current_scf_iter += 1

                # Use spin density if spin polarised calculation
                if "Change of charge/spin density" in line:

                    self.scf_conv_acc_params["change_of_charge"][
                        current_scf_iter - 1
                    ] = float(spl[-2])
                    self.scf_conv_acc_params["change_of_charge_spin_density"][
                        current_scf_iter - 1
                    ] = float(spl[-1])

                # Otherwise just use change of charge
                elif "Change of charge" in line:
                    self.scf_conv_acc_params["change_of_charge"][
                        current_scf_iter - 1
                    ] = float(spl[-1])

                if "Change of sum of eigenvalues" in line:
                    self.scf_conv_acc_params["change_of_sum_eigenvalues"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                if "Change of total energy" in line:
                    self.scf_conv_acc_params["change_of_total_energy"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                # NOTE
                # In the current aims compilation I'm using to test this, there is
                # something wrong with printing the change of forces. It happens
                # multiple times per relaxation and is clearly wrong so I am removing
                # this functionality for now

                # if "Change of forces" in line:
                #     # Only save the smallest change of forces for each geometry
                #     # relaxation step. I have no idea why it prints multiple times but
                #     # I assume it's a data race of some sort
                #     if new_scf_iter:
                #         self.scf_conv_acc_params["change_of_forces"][
                #             current_relax_step - 1
                #         ] = float(spl[-2])

                #         new_scf_iter = False

                #     elif (
                #         float(spl[-2])
                #         < self.scf_conv_acc_params["change_of_forces"][-1]
                #     ):
                #         self.scf_conv_acc_params["change_of_forces"][
                #             current_relax_step - 1
                #         ] = float(spl[-2])

                if "Forces on atoms" in line:
                    self.scf_conv_acc_params["forces_on_atoms"][
                        current_relax_step - 1
                    ] = float(spl[-2])

                if line.strip() == "Self-consistency cycle converged.":
                    # new_scf_iter = True
                    current_relax_step += 1

        return self.scf_conv_acc_params

    def get_n_initial_ks_states(self, include_spin_polarised=True) -> int:
        """
        Get the number of Kohn-Sham states from the first SCF step.

        Parameters
        ----------
        include_spin_polarised : bool, optional
            Whether to include the spin-down states in the count if the calculation is
            spin polarised (the default is True).

        Returns
        -------
        int
            The number of kohn-sham states
        """

        target_line = (
            "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"
        )

        init_ev_start = 0
        n_ks_states = 0

        # Find the first time the KS states are printed
        for init_ev_start, line in enumerate(self.lines):
            if target_line == line.strip():
                init_ev_start += 1
                break

        # Then count the number of lines until the next empty line
        for init_ev_end, line in enumerate(self.lines[init_ev_start:]):
            if len(line) > 1:
                n_ks_states += 1
            else:
                break

        # Count the spin-down eigenvalues if the calculation is spin polarised
        if include_spin_polarised:
            init_ev_end = init_ev_start + n_ks_states
            if target_line == self.lines[init_ev_end + 3].strip():
                init_ev_end += 4
                for line in self.lines[init_ev_end:]:
                    if len(line) > 1:
                        n_ks_states += 1
                    else:
                        break

            else:  # If SD states are not found 4 lines below end of SU states
                warnings.warn(
                    "A spin polarised calculation was expected but not found."
                )

        return n_ks_states

    def _get_ks_states(self, ev_start, eigenvalues, scf_iter, n_ks_states):
        """
        Get any set of KS states, occupations, and eigenvalues.

        Parameters
        ----------
        ev_start : int
            The line number where the KS states start.
        eigenvalues : dict
            The dictionary to store the KS states, occupations, and eigenvalues.
        scf_iter : int
            The current SCF iteration.
        n_ks_states : int
            The number of KS states to save.
        """

        for i, line in enumerate(
            self.lines[ev_start : ev_start + n_ks_states]
        ):
            values = line.split()
            eigenvalues["state"][scf_iter][i] = int(values[0])
            eigenvalues["occupation"][scf_iter][i] = float(values[1])
            eigenvalues["eigenvalue_eV"][scf_iter][i] = float(values[3])

        # return eigenvalues

    def get_all_ks_eigenvalues(self) -> Union[dict, Tuple[dict, dict]]:
        """Get all Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        Union[dict, Tuple[dict, dict]]
            dict
                the kohn-sham eigenvalues
            Tuple[dict, dict]
                dict
                    the spin-up kohn-sham eigenvalues
                dict
                    the spin-down kohn-sham eigenvalues

        Raises
        ------
        ValueError
            the calculation was not spin polarised
        """

        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Get the number of KS states and scf iterations
        # Add 2 to SCF iters as if output_level full is specified, FHI-aims prints the
        # KS states once before the SCF starts and once after it finishes
        n_scf_iters = self.get_n_scf_iters() + 2
        n_ks_states = self.get_n_initial_ks_states(
            include_spin_polarised=False
        )

        # Parse line to find the start of the KS eigenvalues
        target_line = (
            "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"
        )

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
                "eigenvalue_eV": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
            }

            n = 0  # Count the current SCF iteration
            for i, line in enumerate(self.lines):
                if target_line in line:
                    n += 1
                    # Get the KS states from this line until the next empty line
                    self._get_ks_states(i, eigenvalues, n, n_ks_states)

            return eigenvalues

        elif spin_polarised:
            su_eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
                "eigenvalue_eV": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
            }
            sd_eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
                "eigenvalue_eV": np.zeros(
                    (n_scf_iters, n_ks_states), dtype=float
                ),
            }

            # Count the number of SCF iterations for each spin channel
            up_n = 0
            down_n = 0
            for i, line in enumerate(self.lines):

                # Printing of KS states is weird in aims.out. Ensure that we don't add
                # more KS states than the array is long
                if up_n == n_scf_iters and down_n == n_scf_iters:
                    break

                if target_line in line:
                    # The spin-up line is two lines above the target line
                    if self.lines[i - 2].strip() == "Spin-up eigenvalues:":
                        # Get the KS states from this line until the next empty line
                        self._get_ks_states(
                            i + 1, su_eigenvalues, up_n, n_ks_states
                        )
                        up_n += 1

                    # The spin-down line is two lines above the target line
                    if self.lines[i - 2].strip() == "Spin-down eigenvalues:":
                        # Get the KS states from this line until the next empty line
                        self._get_ks_states(
                            i + 1, sd_eigenvalues, down_n, n_ks_states
                        )
                        down_n += 1

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError(
                "Could not determine if calculation was spin polarised."
            )

    def get_final_ks_eigenvalues(self) -> Union[dict, Tuple[dict, dict]]:
        """Get the final Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        Union[dict, Tuple[dict, dict]]
            dict
                the final kohn-sham eigenvalues
            Tuple[dict, dict]
                dict
                    the spin-up kohn-sham eigenvalues
                dict
                    the spin-down kohn-sham eigenvalues

        Raises
        ------
        ValueError
            the calculation was not spin polarised
        """

        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Get the number of KS states
        n_ks_states = self.get_n_initial_ks_states(
            include_spin_polarised=False
        )

        # Parse line to find the start of the KS eigenvalues
        target_line = (
            "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"
        )

        # Iterate backwards from end of aims.out to find the final KS eigenvalues
        for i, line in enumerate(reversed(self.lines)):
            if target_line == line.strip():
                final_ev_start = -i
                break

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((1, n_ks_states), dtype=int),
                "occupation": np.zeros((1, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((1, n_ks_states), dtype=float),
            }
            # Get the KS states from this line until the next empty line
            self._get_ks_states(final_ev_start, eigenvalues, 0, n_ks_states)

            return eigenvalues

        elif spin_polarised:
            su_eigenvalues = {
                "state": np.zeros((1, n_ks_states), dtype=int),
                "occupation": np.zeros((1, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((1, n_ks_states), dtype=float),
            }
            sd_eigenvalues = su_eigenvalues.copy()

            # The spin-down states start from here
            self._get_ks_states(final_ev_start, sd_eigenvalues, 0, n_ks_states)

            # Go back one more target line to get the spin-up states
            for i, line in enumerate(
                reversed(self.lines[: final_ev_start - 1])
            ):
                if target_line == line.strip():
                    final_ev_start += -i - 1
                    break

            self._get_ks_states(final_ev_start, su_eigenvalues, 0, n_ks_states)

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError(
                "Could not determine if calculation was spin polarised."
            )

    def get_pert_soc_ks_eigenvalues(self) -> dict:
        """
        Get the perturbative SOC Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        dict
            The perturbative SOC kohn-sham eigenvalues
        """

        # Get the number of KS states
        n_ks_states = self.get_n_initial_ks_states()

        target_line = (
            "State    Occupation    Unperturbed Eigenvalue [eV]"
            "    Eigenvalue [eV]    Level Spacing [eV]"
        )

        print(target_line)

        # Iterate backwards from end of aims.out to find the perturbative SOC
        # eigenvalues
        for i, line in enumerate(reversed(self.lines)):
            if target_line == line.strip():
                final_ev_start = -i
                break

        eigenvalues = {
            "state": np.zeros(n_ks_states, dtype=int),
            "occupation": np.zeros(n_ks_states, dtype=float),
            "unperturbed_eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            "eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            "level_spacing_eV": np.zeros(n_ks_states, dtype=float),
        }

        for i, line in enumerate(
            self.lines[final_ev_start : final_ev_start + n_ks_states]
        ):
            spl = line.split()
            eigenvalues["state"][i] = int(spl[0])
            eigenvalues["occupation"][i] = float(spl[1])
            eigenvalues["unperturbed_eigenvalue_eV"][i] = float(spl[2])
            eigenvalues["eigenvalue_eV"][i] = float(spl[3])
            eigenvalues["level_spacing_eV"][i] = float(spl[4])

        return eigenvalues


class ELSIOutput(Output):
    """
    Parse matrix output written in a binary csc format from ELSI.

    ...

    Attributes
    ----------
    lines :
        Contents of ELSI output file.
    n_basis : int
        Number of basis functions
    n_non_zero : int
        Number of non-zero elements in the matrix
    """

    def __init__(self, elsi_out: str):
        super().__init__(elsi_out=elsi_out)
        self.lines = self.file_contents["elsi_out"]

    def get_elsi_csc_header(self) -> npt.NDArray[np.int64]:
        """
        Get the contents of the ELSI file header

        Returns
        -------
        tuple
            The contents of the ELSI csc file header
        """

        return np.frombuffer(self.lines[0:128], dtype=np.int64)

    @property
    def n_basis(self) -> int:
        return self.get_elsi_csc_header()[3]

    @property
    def n_non_zero(self) -> int:
        return self.get_elsi_csc_header()[5]

    def read_elsi_as_csc(
        self, csc_format: bool = False
    ) -> Union[sp.csc_matrix, npt.NDArray[np.float64]]:
        """
        Get a CSC matrix from an ELSI output file

        Parameters
        ----------
        csc_format : bool, default=True
            Whether to return the matrix in CSC format or a standard numpy array

        Returns
        -------
        Tuple[sp.csc_matrix, np.ndarray]
            The CSC matrix or numpy array
        """

        header = self.get_elsi_csc_header()

        # Get the column pointer
        end = 128 + self.n_basis * 8
        col_ptr = np.frombuffer(self.lines[128:end], dtype=np.int64)
        col_ptr = np.append(col_ptr, self.n_non_zero + 1)
        col_ptr -= 1

        # Get the row index
        start = end + self.n_non_zero * 4
        row_idx = np.array(
            np.frombuffer(self.lines[end:start], dtype=np.int32)
        )
        row_idx -= 1

        if header[2] == 0:  # real
            nnz_val = np.frombuffer(
                self.lines[start : start + self.n_non_zero * 8],
                dtype=np.float64,
            )

        else:  # complex
            nnz_val = np.frombuffer(
                self.lines[start : start + self.n_non_zero * 16],
                dtype=np.complex128,
            )

        if csc_format:
            return sp.csc_matrix(
                (nnz_val, row_idx, col_ptr), shape=(self.n_basis, self.n_basis)
            )

        else:
            return sp.csc_matrix(
                (nnz_val, row_idx, col_ptr), shape=(self.n_basis, self.n_basis)
            ).toarray()
