from typing import Tuple, Union

import numpy as np

import dfttools.utils as utils
from dfttools.base_parser import BaseParser


class Output(BaseParser):
    """Base class for parsing output files from electronic structure calculations.

    ...

    Attributes
    ----------
    _supported_files : list
        The supported file types that can be parsed.
    file_paths : dict
        The paths to the files to be parsed.
    file_contents : dict
        The contents of the files to be parsed.
    """

    # FHI-aims, ...
    _supported_files = ["aims_out"]

    def __init__(self, **kwargs):
        super().__init__(self._supported_files, **kwargs)

    @property
    def supported_files(self):
        return self._supported_files


class AimsOutput(Output):
    """FHI-aims output file parser.

    ...

    Attributes
    ----------
    aims_out : str
        The path to the aims.out file.
    """

    def __init__(self, aims_out="aims.out"):
        super().__init__(aims_out=aims_out)
        # Check if the aims.out file was provided
        utils.check_required_files(self._supported_files, "aims_out")

    def check_exit_normal(self) -> bool:
        """Check if the FHI-aims calculation exited normally.

        Returns
        -------
        bool
            Whether the calculation exited normally or not.
        """

        if "Have a nice day." == self.file_contents["aims_out"][-2].strip():
            exit_normal = True
        else:
            exit_normal = False

        return exit_normal

    def check_spin_polarised(self) -> bool:
        """Check if the FHI-aims calculation was spin polarised.

        Returns
        -------
        bool
            Whether the calculation was spin polarised or not.
        """

        spin_polarised = False

        for line in self.file_contents["aims_out"]:
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
        """Get the convergence parameters from the aims.out file.

        Returns
        -------
        dict
            The convergence parameters from the aims.out file.
        """

        # Setup dictionary to store convergence parameters
        self.convergence_params = {
            "charge_density": 0.0,
            "sum_eigenvalues": 0.0,
            "total_energy": 0.0,
            "total_force": None,
        }

        for line in self.file_contents["aims_out"]:
            spl = line.split()
            if len(spl) > 1:
                if "accuracy" in spl and "charge density" in spl:
                    self.convergence_params["charge_density"] = float(spl[-1])
                if "accuracy" in spl and "sum of eigenvalues" in spl:
                    self.convergence_params["sum_eigenvalues"] = float(spl[-1])
                if "accuracy" in spl and "total energy" in spl:
                    self.convergence_params["total_energy"] = float(spl[-1])
                if "accuracy" in spl and "total force" in spl:
                    self.convergence_params["total_force"] = float(spl[-1])
                if "Defaulting to 'sc_accuracy_forces not checked'." in line:
                    self.convergence_params["total_force"] = None

                # No more values to get after SCF starts
                if "Begin self-consistency loop" in line:
                    break

        return self.convergence_params

    def get_final_energy(self) -> Union[float, None]:
        """Get the final energy from a FHI-aims calculation.

        Returns
        -------
        Union[float, None]
            The final energy of the calculation.
        """

        for line in self.file_contents["aims_out"]:
            if "s.c.f. calculation      :" in line:
                return float(line.split()[-2])

    def get_all_ks_eigenvalues(self) -> Union[dict, Tuple[dict, dict]]:
        """Get all Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        Union[dict, Tuple[dict, dict]]
            dict
                The Kohn-Sham eigenvalues
            Tuple[dict, dict]
                dict
                    The spin-up Kohn-Sham eigenvalues
                dict
                    The spin-down Kohn-Sham eigenvalues
        """

        aims_out = self.file_contents["aims_out"]

        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Get the number of KS states and scf iterations
        n_ks_states = self.get_n_initial_ks_states()
        n_scf_iters = self.get_n_scf_iters()

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros((n_scf_iters, n_ks_states), dtype=int),
                "occupation": np.zeros((n_scf_iters, n_ks_states), dtype=float),
                "eigenvalue_eV": np.zeros((n_scf_iters, n_ks_states), dtype=float),
            }

            n = 0  # Count the current SCF iteration
            for i, line in enumerate(aims_out):
                if target_line in line:
                    n += 1
                    # Get the KS states from this line until the next empty line
                    for j, line in enumerate(aims_out[i:]):
                        if len(line) > 1:
                            values = line.split()
                            eigenvalues["state"][n][j] = int(values[0])
                            eigenvalues["occupation"][n][j] = float(values[1])
                            eigenvalues["eigenvalue_eV"][n][j] = float(values[3])
                        else:
                            break

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

            # Count the number of SCF iterations for each spin channels
            up_n = 0
            down_n = 0
            for i, line in enumerate(aims_out):
                if target_line in line:
                    if aims_out[i - 2] == "Spin-up eigenvalues:":
                        up_n += 1
                        # Get the KS states from this line until the next empty line
                        for j, line in enumerate(aims_out[i:]):
                            if len(line) > 1:
                                values = line.split()
                                su_eigenvalues["state"][up_n][i] = int(values[0])
                                su_eigenvalues["occupation"][up_n][i] = float(values[1])
                                su_eigenvalues["eigenvalue_eV"][up_n][i] = float(
                                    values[3]
                                )
                            else:
                                break

                    if aims_out[i - 2] == "Spin-down eigenvalues:":
                        down_n += 1
                        # Get the KS states from this line until the next empty line
                        for j, line in enumerate(aims_out[i:]):
                            if len(line) > 1:
                                values = line.split()
                                sd_eigenvalues["state"][down_n][i] = int(values[0])
                                sd_eigenvalues["occupation"][down_n][i] = float(
                                    values[1]
                                )
                                sd_eigenvalues["eigenvalue_eV"][down_n][i] = float(
                                    values[3]
                                )
                            else:
                                break

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError("Could not determine if calculation was spin polarised.")

    def get_final_ks_eigenvalues(self) -> Union[dict, Tuple[dict, dict]]:
        """Get the final Kohn-Sham eigenvalues from a calculation.

        Returns
        -------
        Union[dict, Tuple[dict, dict]]
            dict
                The final Kohn-Sham eigenvalues
            Tuple[dict, dict]
                dict
                    The spin-up Kohn-Sham eigenvalues
                dict
                    The spin-down Kohn-Sham eigenvalues
        """

        aims_out = self.file_contents["aims_out"]

        # Check if the calculation was spin polarised
        spin_polarised = self.check_spin_polarised()

        # Get the number of KS states
        n_ks_states = self.get_n_initial_ks_states()

        # Parse line to find the start of the KS eigenvalues
        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        # Iterate backwards from end of aims.out to find the final KS eigenvalues
        final_ev_start = -1
        while target_line not in aims_out[final_ev_start]:
            final_ev_start -= 1

        if not spin_polarised:
            eigenvalues = {
                "state": np.zeros(n_ks_states, dtype=int),
                "occupation": np.zeros(n_ks_states, dtype=float),
                "eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            }
            # Get the KS states from this line until the next empty line
            for i, line in enumerate(aims_out[final_ev_start:]):
                if len(line) > 1:
                    values = line.split()
                    eigenvalues["state"][i] = int(values[0])
                    eigenvalues["occupation"][i] = float(values[1])
                    eigenvalues["eigenvalue_eV"][i] = float(values[3])
                else:
                    break

            return eigenvalues

        elif spin_polarised:
            su_eigenvalues = {
                "state": np.zeros(n_ks_states, dtype=int),
                "occupation": np.zeros(n_ks_states, dtype=float),
                "eigenvalue_eV": np.zeros(n_ks_states, dtype=float),
            }
            sd_eigenvalues = su_eigenvalues.copy()

            # The spin-down states start from here
            for i, line in enumerate(aims_out[final_ev_start + 1 :]):
                if len(line) > 1:
                    values = line.split()
                    sd_eigenvalues["state"][i] = int(values[0])
                    sd_eigenvalues["occupation"][i] = float(values[1])
                    sd_eigenvalues["eigenvalue_eV"][i] = float(values[3])
                else:
                    break

            # Go back one more target line to get the spin-up states
            while target_line not in aims_out[final_ev_start]:
                final_ev_start -= 1

            for i, line in enumerate(aims_out[final_ev_start + 1 :]):
                if len(line) > 1:
                    values = line.split()
                    su_eigenvalues["state"][i] = int(values[0])
                    su_eigenvalues["occupation"][i] = float(values[1])
                    su_eigenvalues["eigenvalue_eV"][i] = float(values[3])
                else:
                    break

            return su_eigenvalues, sd_eigenvalues

        else:
            raise ValueError("Could not determine if calculation was spin polarised.")

    def get_i_scf_conv_acc(self) -> dict:
        """Get SCF convergence accuracy values from the aims.out file.

        Returns
        -------
        dict
            The SCF convergence accuracy values from the aims.out file.
        """

        # Read the total number of SCF iterations
        n_scf_iters = self.get_n_scf_iters()

        # Check that the calculation finished normally otherwise number of SCF
        # iterations is not known
        self.scf_conv_acc_params = {
            "scf_iter": np.zeros(n_scf_iters),
            "change_of_charge": np.zeros(n_scf_iters),
            "change_of_charge_spin_density": np.zeros(n_scf_iters),
            "change_of_sum_eigenvalues": np.zeros(n_scf_iters),
            "change_of_total_energy": np.zeros(n_scf_iters),
            "change_of_forces": np.zeros(n_scf_iters),
            "forces_on_atoms": np.zeros(n_scf_iters),
        }

        current_scf_iter = 0

        for line in self.file_contents["aims_out"]:
            spl = line.split()
            if len(spl) > 1:
                if "begin self-consistency iteration #" in line:
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

                if "change of total energy" in line:
                    self.scf_conv_acc_params["change_of_total_energy"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                if "Change of forces" in line:
                    self.scf_conv_acc_params["change_of_forces"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

                if "Forces on atoms" in line:
                    self.scf_conv_acc_params["forces_on_atoms"][
                        current_scf_iter - 1
                    ] = float(spl[-2])

        return self.scf_conv_acc_params

    def get_n_initial_ks_states(self) -> int:
        """Get the number of Kohn-Sham states from the first SCF step.

        Returns
        -------
        int
            The number of Kohn-Sham states.
        """

        target_line = "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

        init_ev_start = 0
        n_ks_states = 0
        while target_line not in self.file_contents["aims_out"][0]:
            init_ev_start += 1

        else:
            while len(self.file_contents["aims_out"][init_ev_start]) > 1:
                n_ks_states += 1

        return n_ks_states

    def get_n_scf_iters(self) -> int:
        """Get the number of SCF iterations from the aims.out file.

        Returns
        -------
        int
            The number of SCF iterations.
        """

        n_scf_iters = 0
        for line in reversed(self.file_contents["aims_out"]):
            if "Number of self-consistency cycles" in line:
                return int(line.split()[-1])

            # If the calculation did not finish normally, the number of SCF iterations
            # will not be printed. In this case, count each SCF iteration as they were
            # calulated
            if "Begin self-consistency iteration #" in line:
                n_scf_iters += 1

        return n_scf_iters
