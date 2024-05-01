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
    sc_iter_limit=200
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/1_aims.out")


@pytest.fixture
def aims_out_2():
    """
    Completed, cluster, w/ spin

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    spin=collinear
    sc_iter_limit=200
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/2_aims.out")


@pytest.fixture
def aims_out_3():
    """
    Completed, cluster, w/ spin, w/ spin-orbit

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    spin=collinear
    include_spin_orbit=non_self_consistent
    sc_iter_limit=200
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/3_aims.out")


@pytest.fixture
def aims_out_4():
    """
    Completed, PBC, gamma point

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    k_grid=(1, 1, 1)
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/4_aims.out")


@pytest.fixture
def aims_out_5():
    """
    Completed, cluster, w/ geometry relaxation

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    relax_geometry=bfgs 5e-3
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/5_aims.out")


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
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/6_aims.out")


@pytest.fixture
def aims_out_7():
    """
    Failed, cluster

    Stats
    -----
    exit_normal=False

    Parameters
    ----------
    sc_iter_limit=100
    sc_accuracy_rho=1e-10
    sc_accuracy_eev=1e-6
    sc_accuracy_etot=1e-12
    sc_accuracy_forces=1e-8
    """

    return AimsOutput(aims_out="fixtures/7_aims.out")


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
    sc_iter_limit=100
    sc_accuracy_rho=1e-10
    sc_accuracy_eev=1e-6
    sc_accuracy_etot=1e-12
    sc_accuracy_forces=1e-8
    """

    return AimsOutput(aims_out="fixtures/8_aims.out")


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

    return AimsOutput(aims_out="fixtures/9_aims.out")


@pytest.fixture
def aims_out_10():
    """
    Completed, PBC, hybrid functional, 8x8x8

    Stats
    -----
    exit_normal=True

    Parameters
    ----------
    xc=hse06
    k_grid=(8, 8, 8)
    sc_accuracy_rho=1e-5
    sc_accuracy_eev=1e-3
    sc_accuracy_etot=1e-6
    sc_accuracy_forces=1e-4
    """

    return AimsOutput(aims_out="fixtures/10_aims.out")


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, True),
        (aims_out_2, True),
        (aims_out_3, True),
        (aims_out_4, True),
        (aims_out_5, True),
        (aims_out_6, True),
        (aims_out_7, False),
        (aims_out_8, False),
        (aims_out_9, True),
        (aims_out_10, True),
    ],
)
def check_exit_normal_test(aims_out, expected):
    assert aims_out.check_exit_normal() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, False),
        (aims_out_2, True),
        (aims_out_3, False),
        (aims_out_4, True),
        (aims_out_5, False),
        (aims_out_6, False),
        (aims_out_7, False),
        (aims_out_8, False),
        (aims_out_9, False),
        (aims_out_10, False),
    ],
)
def check_spin_polarised_test(aims_out, expected):
    assert aims_out.check_spin_polarised() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_5, None),
    ],
)
def get_conv_params_test(aims_out, expected):
    assert aims_out.get_conv_params() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_4, None),
        (aims_out_5, None),
    ],
)
def get_final_energy_test(aims_out, expected):
    assert aims_out.get_final_energy() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_5, None),
    ],
)
def get_n_relaxation_steps_test(aims_out, expected):
    assert aims_out.get_n_relaxation_steps == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_5, None),
        (aims_out_6, None),
        (aims_out_7, None),
        (aims_out_8, None),
        (aims_out_9, None),
        (aims_out_10, None),
    ],
)
def get_i_scf_conv_acc_test(aims_out, expected):
    assert aims_out.get_i_scf_conv_acc() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_7, None),
    ],
)
def get_n_initial_ks_states_test(aims_out, expected):
    assert aims_out.get_n_initial_ks_states() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_7, None),
    ],
)
def get_all_ks_eigenvalues_test(aims_out, expected):
    assert aims_out.get_all_ks_eigenvalues() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_7, None),
    ],
)
def get_final_ks_eigenvalues_test(aims_out, expected):
    assert aims_out.get_final_ks_eigenvalues() == expected


@pytest.mark.parametrize(
    "aims_out, expected",
    [
        (aims_out_1, None),
        (aims_out_2, None),
        (aims_out_3, None),
        (aims_out_4, None),
        (aims_out_7, None),
    ],
)
def get_pert_soc_ks_eigenvalues_test(aims_out, expected):
    assert aims_out.get_pert_soc_ks_eigenvalues() == expected
