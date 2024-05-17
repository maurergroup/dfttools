import glob
import struct
import warnings
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

import dfttools.utils.file_utils as fu
from dfttools.base_parser import BaseParser


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

    # Add new supported files to this list
    # FHI-aims, ELSI, ...
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
        self.lines = self._file_contents["aims_out"]

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
                    self.scf_conv_acc_params["scf_iter"][current_scf_iter] = int(
                        spl[-1]
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

    def get_n_initial_ks_states(self, include_spin_polarised: bool = True) -> int:
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

        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

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

        for i, line in enumerate(self.lines[ev_start : ev_start + n_ks_states]):
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
        n_ks_states = self.get_n_initial_ks_states(include_spin_polarised=False)

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
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
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
            }
            sd_eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
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
                        self._get_ks_states(i + 1, su_eigenvalues, up_n, n_ks_states)
                        up_n += 1

                    # The spin-down line is two lines above the target line
                    if self.lines[i - 2].strip() == "Spin-down eigenvalues:":
                        # Get the KS states from this line until the next empty line
                        self._get_ks_states(i + 1, sd_eigenvalues, down_n, n_ks_states)
                        down_n += 1

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError("Could not determine if calculation was spin polarised.")

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
        n_ks_states = self.get_n_initial_ks_states(include_spin_polarised=False)

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

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
            for i, line in enumerate(reversed(self.lines[: final_ev_start - 1])):
                if target_line == line.strip():
                    final_ev_start += -i - 1
                    break

            self._get_ks_states(final_ev_start, su_eigenvalues, 0, n_ks_states)

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError("Could not determine if calculation was spin polarised.")

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
    Parse matrix output written in a binary format from ELSI.

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

    def __init__(self, elsi_out: str = "D_spin_01_kpt_000001.csc"):
        super().__init__(elsi_out=elsi_out)
        self.lines = self._file_contents["elsi_out"]

    def get_elsi_csc_header(self):  # -> Tuple(str):
        """
        Get the contents of the ELSI file header

        Returns
        -------
        FIXME: Add return type
        """

        return struct.unpack("l" * 16, self.lines[0:128])

    @property
    def n_basis(self) -> int:
        return self.get_elsi_csc_header()[3]

    @property
    def n_non_zero(self) -> int:
        return self.get_elsi_csc_header()[5]

    def _get_column_pointer_new(self):  # -> npt.NDArray[np.int64]:
        """
        Get the column pointer from the ELSI file.

        Returns
        -------
        npt.NDArray[np.int64]
            The column pointer
        """

        col_ptr = np.frombuffer(
            self.lines[128 : 128 + self.n_basis * 8], dtype=np.int64
        )
        return np.append(col_ptr, self.n_non_zero + 1)

    def read_elsi_csc(self):
        pass
