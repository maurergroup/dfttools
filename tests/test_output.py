import os
import subprocess

import numpy as np
import pytest
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

        # Set class attribute to check in xfail tests
        self._run_aims = False if run_aims is False else True

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

    # TODO Use yaml files for these expected values
    def test_get_time_per_scf(self):
        # fmt: off
        timings = [
            np.array([0.042, 0.045, 0.038, 0.037, 0.036, 0.038, 0.033, 0.036, 0.032, 0.032, 0.031, 0.023]),
            np.array([0.033, 0.033, 0.033, 0.034, 0.035, 0.035, 0.036, 0.036, 0.036, 0.036, 0.036, 0.036, 0.026]),
            np.array([0.033, 0.033, 0.034, 0.034, 0.034, 0.035, 0.036, 0.036, 0.036, 0.036, 0.036, 0.037, 0.027]),
            np.array([0.649, 0.666, 0.667, 0.666, 0.667, 0.667, 0.667, 0.665, 0.666, 0.326]),
            np.array([0.028, 0.028, 0.029, 0.029, 0.029, 0.029, 0.029, 0.029, 0.03, 0.03, 0.072, 0.028, 0.028, 0.028, 0.028, 0.029, 0.029, 0.03, 0.03, 0.073, 0.028, 0.028, 0.028, 0.028, 0.028, 0.029, 0.029, 0.03, 0.073, 0.028, 0.028, 0.029, 0.028, 0.029, 0.029, 0.072, 0.028, 0.028, 0.028, 0.029, 0.028, 0.072]),
            np.array([0.785, 0.801, 0.803, 0.808, 0.802, 0.803, 0.802, 0.802, 0.801, 0.803, 2.276, 9.297, 0.797, 0.79, 0.793, 0.792, 0.794, 2.248, 9.114, 0.797, 0.794, 0.793, 0.793, 0.793, 0.793, 2.248, 9.248]),
            np.array([0.028, 0.028, 0.029, 0.029, 0.029, 0.029, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.073, 0.073, 0.028, 0.028, 0.029, 0.029, 0.029, 0.029, 0.029, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.073, 0.073, 0.028, 0.028, 0.028, 0.029, 0.029, 0.029, 0.029, 0.029, 0.031, 0.03, 0.03, 0.03, 0.03, 0.03, 0.073, 0.073, 0.028, 0.028, 0.028, 0.029, 0.0]),
            np.array([0.636, 0.653, 0.652, 0.652, 0.657, 0.65, 0.65, 0.0]),
            np.array([0.048, 0.046, 0.046, 0.046, 0.049, 0.047, 0.048, 0.047, 0.047, 0.047, 0.047, 0.047, 0.047, 0.039]),
            np.array([34.786, 33.136, 33.081, 33.97, 33.873, 33.9, 33.933, 33.921, 33.945, 33.936, 33.575]),
        ]
        # fmt: on

        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 2e-3
        assert np.allclose(
            self.ao.get_time_per_scf(),
            timings[self._aims_fixture_no - 1],
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

    # TODO Use yaml files for these expected values
    def test_get_change_of_total_energy_2(self):
        """Get every energy change"""

        # fmt: off
        energy_diffs = [
            np.array([0.4663, 1.408, -0.4107, 0.5095, 0.004077, 0.01574, 0.003194, 2.061e-05, 7.458e-06, -3.721e-07, 3.822e-07, 1.599e-08]),
            np.array([0.2611, -0.1508, -0.8718, 0.2303, 0.05782, -0.001582, 0.0006921, -0.0002017, 8.818e-05, -7.301e-06, -1.198e-06, -5.717e-09, 1.611e-09]),
            np.array([0.2611, -0.1508, -0.8718, 0.2303, 0.05782, -0.001582, 0.0006921, -0.0002017, 8.818e-05, -7.301e-06, -1.198e-06, -5.717e-09, 1.611e-09]),
            np.array([0.1202, 0.871, 0.02343, 0.005611, -0.0003911, 0.0002121, -0.0001144, 1.971e-05, -3.046e-06, -1.492e-07]),
            np.array([0.4426, 1.277, -0.3396, 0.453, -0.001744, 0.01792, 0.0002052, 8.435e-06, 1.013e-06, -2.988e-08, 1.168e-07, -2081.0, 0.002236, 0.0004654, 1.558e-05, 5.188e-09, -8.163e-07, -1.626e-07, -6.659e-09, -3.078e-10, -2081.0, 0.00177, 0.0003977, 2.452e-05, 3.896e-07, -7.184e-07, -3.316e-07, -9.563e-09, 3.674e-11, -2081.0, 3.323e-05, 9.511e-06, -1.416e-06, 1.403e-07, -3.84e-07, -1.012e-09, -2081.0, 6.632e-07, 2.993e-07, -2.037e-07, 1.171e-08, -5.833e-09]),
            np.array([0.01644, 0.1063, 0.006359, 0.002794, 0.0001406, 2.873e-05, 5.028e-06, 4.511e-06, -8.154e-07, -3.882e-08, -2.815e-10, 6.156e-10, -15800.0, -5.771e-06, -2.262e-05, -6.403e-06, -1.75e-07, -2.497e-08, -4.832e-09, -15800.0, -9.886e-06, -4.329e-05, -1.378e-05, -1.58e-07, -1.163e-07, -4.241e-08, 3.703e-09]),
            np.array([0.4295, 1.19, -0.3077, 0.4194, 0.004835, 0.01161, 0.001319, -2.423e-05, 2.34e-05, -7.693e-09, 1.223e-07, -8.323e-09, 4.424e-10, 2.738e-10, 1.856e-11, 2.32e-12, 7.734e-13, 0.0, 0.0, -2081.0, 0.00942, 0.002121, 0.0002073, 1.587e-05, -7.203e-07, -1.062e-06, 1.538e-08, 1.902e-09, -1.001e-09, -1.868e-10, 8.082e-11, -3.867e-12, 0.0, 0.0, -3.867e-13, -2081.0, 0.004698, 0.001127, 8.39e-05, 7.677e-06, -7.438e-07, -8.215e-07, 9.776e-09, -2.514e-11, -2.224e-10, -8.314e-11, 2.243e-11, -3.48e-12, -3.867e-13, 3.867e-13, -3.867e-13, -2081.0, 0.0001081, 2.93e-05, 1.509e-05]),
            np.array([0.1202, 0.871, 0.02343, 0.005611, -0.0003911, 0.0002121, -0.0001144]),
            np.array([-0.03058, -5.561, 6.612, 0.601, 0.1137, 0.0394, 0.03515, 0.006006, -0.0006064, 0.01499, -0.002865, 0.0002206, 7.186e-05, 6.018e-06]),
            np.array([-2.158, -0.07087, 0.04834, 0.1184, -0.003943, 0.0103, -0.001032, 0.004801, 0.002519, 0.001141, 7.119e-06]),
        ]
        # fmt: on

        # Fail if the absolute tolerance between any values in test vs. reference array is
        # greater than 1e-10
        assert np.allclose(
            self.ao.get_change_of_total_energy(n_occurrence=None),
            energy_diffs[self._aims_fixture_no - 1],
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

    #     # fmt: off
    #     all_energies = [
    #         np.array([0.4663, 1.408, -0.4107, 0.5095, 0.004077, 0.01574, 0.003194, 2.061e-05, 7.458e-06, -3.721e-07, 3.822e-07, 1.599e-08]),
    #         np.array([0.2611, -0.1508, -0.8718, 0.2303, 0.05782, -0.001582, 0.0006921, -0.0002017, 8.818e-05, -7.301e-06, -1.198e-06, -5.717e-09, 1.611e-09]),
    #         np.array([0.2611, -0.1508, -0.8718, 0.2303, 0.05782, -0.001582, 0.0006921, -0.0002017, 8.818e-05, -7.301e-06, -1.198e-06, -5.717e-09, 1.611e-09]),
    #         np.array([0.1202, 0.871, 0.02343, 0.005611, -0.0003911, 0.0002121, -0.0001144, 1.971e-05, -3.046e-06, -1.492e-07]),
    #         np.array([0.4426, 1.277, -0.3396, 0.453, -0.001744, 0.01792, 0.0002052, 8.435e-06, 1.013e-06, -2.988e-08, 1.168e-07, 0.002236, 0.0004654, 1.558e-05, 5.188e-09, -8.163e-07, -1.626e-07, -6.659e-09, -3.078e-10, 0.00177, 0.0003977, 2.452e-05, 3.896e-07, -7.184e-07, -3.316e-07, -9.563e-09, 3.674e-11, 3.323e-05, 9.511e-06, -1.416e-06, 1.403e-07, -3.84e-07, -1.012e-09, 6.632e-07, 2.993e-07,-2.037e-07, 1.171e-08, -5.833e-09]),
    #         np.array([0.01644, 0.1063, 0.006359, 0.002794, 0.0001406, 2.873e-05, 5.028e-06, 4.511e-06, -8.154e-07, -3.882e-08, -2.815e-10, 6.156e-10, -5.771e-06, -2.262e-05, -6.403e-06, -1.75e-07, -2.497e-08, -4.832e-09, -9.886e-06, -4.329e-05, -1.378e-05, -1.58e-07, -1.163e-07, -4.241e-08, 3.703e-09]),
    #         np.array([0.4295, 1.19, -0.3077, 0.4194, 0.004835, 0.01161, 0.001319, -2.423e-05, 2.34e-05, -7.693e-09, 1.223e-07, -8.323e-09, 4.424e-10, 2.738e-10, 1.856e-11, 2.32e-12, 7.734e-13, 0.0, 0.0, 0.00942, 0.002121, 0.0002073, 1.587e-05, -7.203e-07, -1.062e-06, 1.538e-08, 1.902e-09, -1.001e-09, -1.868e-10, 8.082e-11, -3.867e-12, 0.0, 0.0, -3.867e-13, 0.004698, 0.001127, 8.39e-05, 7.677e-06, -7.438e-07, -8.215e-07, 9.776e-09, -2.514e-11, -2.224e-10, -8.314e-11, 2.243e-11, -3.48e-12, -3.867e-13, 3.867e-13, -3.867e-13, 0.0001081, 2.93e-05, 1.509e-05]),
    #         np.array([0.1202, 0.871, 0.02343, 0.005611, -0.0003911, 0.0002121, -0.0001144]),
    #         np.array([-0.03058, -5.561, 6.612, 0.601, 0.1137, 0.0394, 0.03515, 0.006006, -0.0006064, 0.01499, -0.002865, 0.0002206, 7.186e-05, 6.018e-06]),
    #         np.array([-2.158, -0.07087, 0.04834, 0.1184, -0.003943, 0.0103, -0.001032, 0.004801, 0.002519, 0.001141, 7.119e-06]),
    #     ]
    #     # fmt: on

    #     assert np.allclose(
    #         self.ao.get_change_of_total_energy(n_occurrence=1),
    #         all_energies[self.aims_fixture_no(self.ao) - 1],
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

    # TODO Use yaml files for these expected values
    # @pytest.mark.parametrize(
    #     "aims_out, expected",
    #     [
    #         (
    #             aims_out_1,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_2,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_3,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_4,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_5,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_6,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_7,
    #             {
    #                 "charge_density": 1e-10,
    #                 "sum_eigenvalues": 1e-06,
    #                 "total_energy": 1e-12,
    #                 "total_force": 1e-08,
    #             },
    #         ),
    #         (
    #             aims_out_8,
    #             {
    #                 "charge_density": 1e-10,
    #                 "sum_eigenvalues": 1e-06,
    #                 "total_energy": 1e-12,
    #                 "total_force": 1e-08,
    #             },
    #         ),
    #         (
    #             aims_out_9,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #         (
    #             aims_out_10,
    #             {
    #                 "charge_density": 0.0,
    #                 "sum_eigenvalues": 0.0,
    #                 "total_energy": 0.0,
    #                 "total_force": 0.0,
    #             },
    #         ),
    #     ],
    # )
    # def test_get_conv_params(aims_out, expected):
    #     assert self.ao.get_conv_params() == expected

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
