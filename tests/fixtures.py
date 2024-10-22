# @pytest.fixture
# class AimsFixtures:
#     FIXTURE_DIR = "tests/fixtures/aims_calculations/"

#     @classmethod
#     def aims_out_1(cls):
#         """
#         Completed, cluster, w/o spin

#         Stats
#         -----
#         exit_normal=True

#         Parameters
#         ----------
#         xc=pbe
#         """

#         return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/1/aims.out")

#     @classmethod
#     def aims_out_2(cls):
#         """
#         Completed, cluster, w/ spin

#         Stats
#         -----
#         exit_normal=True

#         Parameters
#         ----------
#         xc=pbe
#         spin=collinear
#         default_initial_moment=1
#         """

#         return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/2/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_3(cls):
#     """
#     Completed, cluster, w/ spin, w/ spin-orbit

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     xc=pbe
#     spin=collinear
#     default_initial_moment=1
#     include_spin_orbit=non_self_consistent
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/3/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_4(cls):
#     """
#     Completed, PBC, gamma point

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     xc=pbe
#     k_grid=(1, 1, 1)
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/4/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_5(cls):
#     """
#     Completed, cluster, w/ geometry relaxation

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     xc=pbe
#     relax_geometry=bfgs 5e-3
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/5/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_6(cls):
#     """
#     Completed, PBC, w/ geometry relaxation, 8x8x8

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     relax_geometry=bfgs 5e-3
#     relax_unit_cell=full
#     k_grid=(8, 8, 8)
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/6/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_7(cls):
#     """
#     Failed, cluster

#     Stats
#     -----
#     exit_normal=False

#     Parameters
#     ----------
#     sc_iter_limit=10
#     sc_accuracy_rho=1e-10
#     sc_accuracy_eev=1e-6
#     sc_accuracy_etot=1e-12
#     sc_accuracy_forces=1e-8
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/7/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_8(cls):
#     """
#     Failed, PBC, gamma point

#     Stats
#     -----
#     exit_normal=False

#     Parameters
#     ----------
#     k_grid=(1, 1, 1)
#     sc_iter_limit=10
#     sc_accuracy_rho=1e-10
#     sc_accuracy_eev=1e-6
#     sc_accuracy_etot=1e-12
#     sc_accuracy_forces=1e-8
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/8/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_9(cls):
#     """
#     Completed, cluster, hybrid functional

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     xc=hse06
#     sc_accuracy_rho=1e-5
#     sc_accuracy_eev=1e-3
#     sc_accuracy_etot=1e-6
#     sc_accuracy_forces=1e-4
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/9/aims.out")

# @classmethod
# @pytest.fixture
# def aims_out_10(cls):
#     """
#     Completed, PBC, hybrid functional, 8x8x8

#     Stats
#     -----
#     exit_normal=True

#     Parameters
#     ----------
#     xc=hse06 0.11
#     k_grid=(8, 8, 8)
#     sc_accuracy_rho=1e-5
#     sc_accuracy_eev=1e-3
#     sc_accuracy_etot=1e-6
#     sc_accuracy_forces=1e-4
#     """

#     return AimsOutput(aims_out=f"{cls.FIXTURE_DIR}/10/aims.out")
