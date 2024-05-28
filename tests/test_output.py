import pathlib

import pytest
from dfttools.output import AimsOutput


@pytest.fixture
def aims_out_1():
    """
    Completed, cluster, w/o spin

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=pbe
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/1/aims.out")


@pytest.fixture
def aims_out_2():
    """
    Completed, cluster, w/ spin

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=pbe
    spin=collinear
    default_initial_moment=1
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/2/aims.out")


@pytest.fixture
def aims_out_3():
    """
    Completed, cluster, w/ spin, w/ spin-orbit

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=pbe
    spin=collinear
    default_initial_moment=1
    include_spin_orbit=non_self_consistent
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/3/aims.out")


@pytest.fixture
def aims_out_4():
    """
    Completed, PBC, gamma point

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=pbe
    k_grid=(1, 1, 1)
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/4/aims.out")


@pytest.fixture
def aims_out_5():
    """
    Completed, cluster, w/ geometry relaxation

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=pbe
    relax_geometry=bfgs 5e-3
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/5/aims.out")


@pytest.fixture
def aims_out_6():
    """
    Completed, PBC, w/ geometry relaxation, 8x8x8

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    relax_geometry=bfgs 5e-3
    relax_unit_cell=full
    k_grid=(8, 8, 8)
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/6/aims.out")


@pytest.fixture
def aims_out_7():
    """
    Failed, cluster

    Stats
    -----
    exit_normal=False

    Parameters
    ----------
    sc_iter_limit=10
    sc_accuracy_rho=1e-10
    sc_accuracy_eev=1e-6
    sc_accuracy_etot=1e-12
    sc_accuracy_forces=1e-8
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/7/aims.out")


@pytest.fixture
def aims_out_8():
    """
    Failed, PBC, gamma point

    Stats
    -----
    exit_normal=False

    Parameters
    ----------
    k_grid=(1, 1, 1)
    sc_iter_limit=10
    sc_accuracy_rho=1e-10
    sc_accuracy_eev=1e-6
    sc_accuracy_etot=1e-12
    sc_accuracy_forces=1e-8
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/8/aims.out")


@pytest.fixture
def aims_out_9():
    """
    Completed, cluster, hybrid functional

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=hse06
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/aims_calculations/9/aims.out")


@pytest.fixture
def aims_out_10():
    """
    Completed, PBC, hybrid functional, 8x8x8

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=hse06 0.11
    k_grid=(8, 8, 8)
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    ao = AimsOutput(
        f"{pathlib.Path(__file__).parent.resolve()}/fixtures/aims_calculations/10/"
        "aims.out"
    )
    return ao.check_exit_normal()


def test_check_exit_normal(aims_out_10):
    assert aims_out_10 is True
    

def test_get_number_of_atoms(aims_out_9):
    aims = aims_out_9
    
    n_atoms = aims.get_number_of_atoms()
    
    assert n_atoms == 3


# TODO: currently a palceholder
def test_all_output_functions(aims_out_9):
    aims = aims_out_9
    
    aims.get_geometry()
    aims.get_parameters()
    aims.check_exit_normal() 
    aims.get_change_of_total_energy()
    #aims.get_change_of_forces()
    aims.get_change_of_sum_of_eigenvalues()
    #aims.get_maximum_force()
    aims.get_final_energy()
    aims.get_energy_corrected()
    aims.get_total_energy_T0()
    aims.get_energy_uncorrected()
    #aims.get_energy_without_vdw()
    aims.get_HOMO_energy()
    aims.get_LUMO_energy()
    #aims.get_vdw_energy()
    aims.get_exchange_correlation_energy()
    aims.get_electrostatic_energy()
    aims.get_kinetic_energy()
    aims.get_sum_of_eigenvalues()
    aims.get_cx_potential_correction()
    aims.get_free_atom_electrostatic_energy()
    aims.get_entropy_correction()
    aims.get_hartree_energy_correction()
    #aims.get_ionic_embedding_energy()
    #aims.get_density_embedding_energy()
    #aims.get_nonlocal_embedding_energy()
    #aims.get_external_embedding_energy()
    #aims.get_forces()
    aims.check_spin_polarised()
    aims.get_conv_params()
    aims.get_n_relaxation_steps()
    aims.get_n_scf_iters()
    aims.get_i_scf_conv_acc()
    aims.get_n_initial_ks_states()
    #aims.get_all_ks_eigenvalues()# -> functionality does not work 
    aims.get_final_ks_eigenvalues()
    #aims.get_pert_soc_ks_eigenvalues()# -> not great but may work if that output is there
        
    



# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, True),
#         (aims_out_2, True),
#         (aims_out_3, True),
#         (aims_out_4, True),
#         (aims_out_5, True),
#         (aims_out_6, True),
#         (aims_out_7, False),
#         (aims_out_8, False),
#         (aims_out_9, True),
#         (aims_out_10, True),
#     ],
# )
# def test_check_exit_normal(aims_out, expected):
#     assert aims_out is expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, False),
#         (aims_out_2, True),
#         (aims_out_3, True),
#         (aims_out_4, False),
#         (aims_out_5, False),
#         (aims_out_6, False),
#         (aims_out_7, False),
#         (aims_out_8, False),
#         (aims_out_9, False),
#         (aims_out_10, False),
#     ],
# )
# def test_check_spin_polarised(aims_out, expected):
#     assert aims_out.check_spin_polarised() == expected


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
#     assert aims_out.get_conv_params() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, -2080.832254505),
#         (aims_out_2, -2080.832254498),
#         (aims_out_3, -2080.832254498),
#         (aims_out_4, -15785.832821011),
#         (aims_out_5, -2080.832254506),
#         (aims_out_6, -15802.654211961),
#         (aims_out_7, None),
#         (aims_out_8, None),
#         (aims_out_9, -2081.000809207),
#         (aims_out_10, -15804.824029071),
#     ],
# )
# def test_get_final_energy(aims_out, expected):
#     assert aims_out.get_final_energy() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, 1),
#         (aims_out_2, 1),
#         (aims_out_3, 1),
#         (aims_out_4, 1),
#         (aims_out_5, 4),
#         (aims_out_6, 2),
#         (aims_out_7, 3),
#         (aims_out_8, 0),
#         (aims_out_9, 1),
#         (aims_out_10, 1),
#     ],
# )
# def get_n_relaxation_steps_test(aims_out, expected):
#     assert aims_out.get_n_relaxation_steps == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, 12),
#         (aims_out_2, 13),
#         (aims_out_3, 13),
#         (aims_out_4, 10),
#         (aims_out_5, 42),
#         (aims_out_6, 27),
#         (aims_out_7, 56),
#         (aims_out_8, 8),
#         (aims_out_9, 14),
#         (aims_out_10, 11),
#     ],
# )
# def test_get_n_scf_iters(aims_out, expected):
#     assert aims_out.get_n_scf_iters() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, None),
#         (aims_out_2, None),
#         (aims_out_3, None),
#         (aims_out_4, None),
#         (aims_out_5, None),
#         (aims_out_6, None),
#         (aims_out_7, None),
#         (aims_out_8, None),
#         (aims_out_9, None),
#         (aims_out_10, None),
#     ],
# )
# def get_i_scf_conv_acc_test(aims_out, expected):
#     assert aims_out.get_i_scf_conv_acc() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, None),
#         (aims_out_2, None),
#         (aims_out_3, None),
#         (aims_out_4, None),
#         (aims_out_7, None),
#     ],
# )
# def get_n_initial_ks_states_test(aims_out, expected):
#     assert aims_out.get_n_initial_ks_states() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, None),
#         (aims_out_2, None),
#         (aims_out_3, None),
#         (aims_out_4, None),
#         (aims_out_7, None),
#     ],
# )
# def get_all_ks_eigenvalues_test(aims_out, expected):
#     assert aims_out.get_all_ks_eigenvalues() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, None),
#         (aims_out_2, None),
#         (aims_out_3, None),
#         (aims_out_4, None),
#         (aims_out_7, None),
#     ],
# )
# def get_final_ks_eigenvalues_test(aims_out, expected):
#     assert aims_out.get_final_ks_eigenvalues() == expected


# @pytest.mark.parametrize(
#     "aims_out, expected",
#     [
#         (aims_out_1, None),
#         (aims_out_2, None),
#         (aims_out_3, None),
#         (aims_out_4, None),
#         (aims_out_7, None),
#     ],
# )
# def get_pert_soc_ks_eigenvalues_test(aims_out, expected):
#     assert aims_out.get_pert_soc_ks_eigenvalues() == expected
