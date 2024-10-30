import os
import subprocess

import numpy as np
import pytest
import yaml
from dfttools.output import AimsOutput
from dfttools.utils.file_utils import aims_bin_path_prompt


class TestAimsOutput:

    @pytest.fixture(params=range(1, 11), autouse=True)
    def aims_out(self, request, run_aims):

        cwd = os.path.dirname(os.path.realpath(__file__))

        # Run the FHI-aims calculations if the run_aims option is specified but not if
        # they already exist.
        # Force it to run if the run_aims option is "change_bin"
        # run_aims fixture is defined in conftest.py
        if request.param == 1 and run_aims is not False:
            binary = aims_bin_path_prompt(run_aims, cwd)
            subprocess.run(["bash", f"{cwd}/run_aims.sh", binary, str(run_aims)])
            aims_out_dir = "custom_bin_aims_calcs"
        elif run_aims is not False:
            aims_out_dir = "custom_bin_aims_calcs"
        else:
            aims_out_dir = "default_aims_calcs"

        self.ao = AimsOutput(
            aims_out=f"{cwd}/fixtures/{aims_out_dir}/{str(request.param)}/aims.out"
        )

        with open(f"{cwd}/test_references.yaml", "r") as references:
            self.ref_data = yaml.safe_load(references)

        # Set class attribute to check in xfail tests
        # self._run_aims = False if run_aims is False else True

    @property
    def _aims_fixture_no(self) -> int:
        return int(self.ao.aims_out_path.split("/")[-2])

    def test_get_number_of_atoms(self):
        if self._aims_fixture_no in [4, 6, 8, 10]:
            assert self.ao.get_number_of_atoms() == 2
        else:
            assert self.ao.get_number_of_atoms() == 3

    # TODO
    # def test_get_geometry(self):

    # TODO
    # def test_get_parameters(self):

    def test_check_exit_normal(self):
        if self._aims_fixture_no in [7, 8]:
            assert self.ao.check_exit_normal() is False
        else:
            assert self.ao.check_exit_normal() is True

    def test_get_time_per_scf(self):

        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 2e-3
        assert np.allclose(
            self.ao.get_time_per_scf(),
            self.ref_data["timings"][self._aims_fixture_no - 1],
            atol=2e-3,
        )

    def test_get_change_of_total_energy_1(self):
        """Using default args (final energy change)"""

        final_energies = np.array(
            [
                1.599e-08,
                1.611e-09,
                1.611e-09,
                -1.492e-07,
                -5.833e-09,
                3.703e-09,
                1.509e-05,
                -0.0001144,
                6.018e-06,
                7.119e-06,
            ]
        )

        assert (
            abs(
                self.ao.get_change_of_total_energy()
                - final_energies[self._aims_fixture_no - 1]
            )
            < 1e-8
        )

    def test_get_change_of_total_energy_2(self):
        """Get every energy change"""

        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 1e-10
        assert np.allclose(
            self.ao.get_change_of_total_energy(n_occurrence=None),
            self.ref_data["energy_diffs"][self._aims_fixture_no - 1],
            atol=1e-8,
        )

    def test_get_change_of_total_energy_3(self):
        """Get the 1st energy change"""

        first_energies = [
            1.408,
            -0.1508,
            -0.1508,
            0.871,
            1.277,
            0.1063,
            1.19,
            0.871,
            -5.561,
            -0.07087,
        ]

        assert (
            abs(
                self.ao.get_change_of_total_energy(n_occurrence=1)
                - first_energies[self._aims_fixture_no - 1]
            )
            < 1e-8
        )

    # TODO: Lukas can you check that this is correct
    # TODO Use yaml files for these expected values
    # def test_get_change_of_total_energy_4(self):
    #     """
    #     Use an energy invalid indicator
    #     """

    #     assert np.allclose(
    #         self.ao.get_change_of_total_energy(n_occurrence=1),
    #         self.ref_data['all_energies'][self.aims_fixture_no(self.ao) - 1],
    #         atol=1e-10,
    #     )

    # Not necessary to include every possible function argument in the next tests as all
    # of the following functions wrap around _get_energy(), which have all been tested in
    # the previous 4 tests

    def test_get_change_of_forces(self):
        forces = [0.4728, 8.772e-09, 6.684e-12]

        if self._aims_fixture_no in [5, 6, 7]:
            assert (
                abs(self.ao.get_change_of_forces() - forces[self._aims_fixture_no - 5])
                < 1e-8
            )

        else:
            with pytest.raises(ValueError):
                self.ao.get_change_of_forces()

    def get_change_of_sum_of_eigenvalues(self):
        pass

    # TODO: currently a palceholder
    # def test_all_output_functions(self_9):, #     aims = aims_out_9

    #     aims.get_geometry()
    #     aims.get_parameters()
    #     aims.check_exit_normal()
    #     aims.get_change_of_total_energy()
    #     # aims.get_change_of_forces()
    #     aims.get_change_of_sum_of_eigenvalues()
    #     # aims.get_maximum_force()
    #     aims.get_final_energy()
    #     aims.get_energy_corrected()
    #     aims.get_total_energy_T0()
    #     aims.get_energy_uncorrected()
    #     # aims.get_energy_without_vdw()
    #     aims.get_HOMO_energy()
    #     aims.get_LUMO_energy()
    #     # aims.get_vdw_energy()
    #     aims.get_exchange_correlation_energy()
    #     aims.get_electrostatic_energy()
    #     aims.get_kinetic_energy()
    #     aims.get_sum_of_eigenvalues()
    #     aims.get_cx_potential_correction()
    #     aims.get_free_atom_electrostatic_energy()
    #     aims.get_entropy_correction()
    #     aims.get_hartree_energy_correction()
    #     # aims.get_ionic_embedding_energy()
    #     # aims.get_density_embedding_energy()
    #     # aims.get_nonlocal_embedding_energy()
    #     # aims.get_external_embedding_energy()
    #     # aims.get_forces()
    #     aims.check_spin_polarised()
    #     aims.get_conv_params()
    #     aims.get_n_relaxation_steps()
    #     aims.get_n_scf_iters()
    #     aims.get_i_scf_conv_acc()
    #     aims.get_n_initial_ks_states()
    #     # aims.get_all_ks_eigenvalues()# -> functionality does not work
    #     aims.get_final_ks_eigenvalues()
    #     # aims.get_pert_soc_ks_eigenvalues()# -> not great but may work if that output is there

    def test_check_spin_polarised(self):
        if self._aims_fixture_no in [2, 3]:
            assert self.ao.check_spin_polarised() is True
        else:
            assert self.ao.check_spin_polarised() is False

    def test_get_conv_params(self):
        if self._aims_fixture_no in [7, 8]:
            assert self.ao.get_conv_params() == self.ref_data["conv_params"][1]
        else:
            assert self.ao.get_conv_params() == self.ref_data["conv_params"][0]

    def test_get_final_energy(self):
        final_energies = [
            -2080.832254505,
            -2080.832254498,
            -2080.832254498,
            -15785.832821011,
            -2080.832254506,
            -15802.654211961,
            None,
            None,
            -2081.000809207,
            -15804.824029071,
        ]

        final_energy = self.ao.get_final_energy()

        if self._aims_fixture_no in [7, 8]:
            assert final_energy is None

        else:
            assert abs(final_energy - final_energies[self._aims_fixture_no - 1]) < 1e-8

    def get_n_relaxation_steps_test(self):
        n_relaxation_steps = [1, 1, 1, 1, 4, 2, 3, 0, 1, 1]
        assert (
            self.ao.get_n_relaxation_steps()
            == n_relaxation_steps[self._aims_fixture_no - 1]
        )

    def test_get_n_scf_iters(self):
        n_scf_iters = [12, 13, 13, 10, 42, 27, 56, 8, 14, 11]
        assert self.ao.get_n_scf_iters() == n_scf_iters[self._aims_fixture_no - 1]

    # TODO
    # def get_i_scf_conv_acc_test(self):

    def test_get_n_initial_ks_states(self):
        n_initial_ks_states = [11, 22, 48, 20, 11, 20, 11, 20, 11, 20]

        def compare_n_initial_ks_states():
            assert (
                self.ao.get_n_initial_ks_states()
                == n_initial_ks_states[self._aims_fixture_no - 1]
            )

        if self._aims_fixture_no in [2, 3]:
            compare_n_initial_ks_states()
        else:
            with pytest.warns(UserWarning):
                compare_n_initial_ks_states()

    # TODO Setup YAML files for storing the expected values for the following tests
    # def test_get_all_ks_eigenvalues(self):

    # def get_final_ks_eigenvalues_test(self):

    # def get_pert_soc_ks_eigenvalues_test(self):
