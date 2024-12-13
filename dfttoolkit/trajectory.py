import numpy as np
from ase import units
from ase.io.trajectory import Trajectory
from dfttoolkit.geometry import Geometry
import dfttoolkit.utils.vibrations_utils as vu
import copy
import numpy.typing as npt


class MDTrajectory:
    def __init__(self, filename: str, cutoff_end: int = 0):
        traj_0 = Trajectory(filename)
        self.traj = []
        for ind in range(len(traj_0) - cutoff_end):
            self.traj.append(traj_0[ind])

        self.geom = Geometry()
        self.geom.get_from_ase_atoms_object(self.traj[0])

    def __add__(self, other_traj):
        traj = copy.deepcopy(self)
        traj += other_traj
        return traj

        self.traj = self.traj + other_traj.traj

    def __iadd__(self, other_traj):
        self.traj = self.traj + other_traj.traj
        return self

    def __len__(self):
        return len(self.traj)

    def get_velocities(
        self, steps: int = 1, cutoff_start: int = 0, cutoff_end: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Get atomic velocities along an MD trajectory.

        Parameters
        ----------
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.

        Returns
        -------
        velocities : npt.NDArray[np.float64]

        """
        velocities = []

        for ind in range(cutoff_start, len(self.traj) - cutoff_end):
            if ind % steps == 0.0:
                velocities_new = self.traj[ind].get_velocities()
                velocities.append(velocities_new)

        return np.array(velocities, dtype=np.float64)

    def get_velocities_mass_weighted(
        self, steps: int = 1, cutoff_start: int = 0, cutoff_end: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Weighs velocities by atomic masses.

        Parameters
        ----------
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.

        Returns
        -------
        velocities_mass_weighted : np.array
            Velocities weighted by atomic masses.

        """
        velocities = self.get_velocities(
            steps=steps, cutoff_start=cutoff_start, cutoff_end=cutoff_end
        )

        velocities_mass_weighted = np.zeros_like(velocities)
        atomic_masses = self.geom.get_atomic_masses()

        for i in range(velocities.shape[1]):
            velocities_mass_weighted[:, i, :] = velocities[:, i, :] * np.sqrt(
                atomic_masses[i]
            )

        return velocities_mass_weighted

    def get_velocities_projected(
        self,
        projection_vectors: npt.NDArray[np.float64],
        mass_weighted: bool = True,
        steps: int = 1,
        cutoff_start: int = 0,
        cutoff_end: int = 0,
        use_numba: bool = True,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the normal-mode-decomposition of the velocities. This is done
        by projecting the atomic velocities onto the vibrational eigenvectors.
        See equation 10 in: https://doi.org/10.1016/j.cpc.2017.08.017

        Parameters
        ----------
        projection_vectors : npt.NDArray[np.float64]
            Array containing the vectors onto which the velocities should be
            projected. Normally you will want this to be the eigenvectors
            return by dfttoolkit.vibration.get_eigenvalues_and_eigenvectors.
        mass_weighted : bool
            Wheter the velocities should be mass weighted. If you do
            normal-mode-decomposition with eigenvectors than this should be
            True. The default is True.
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.
        use_numba : bool
            Use numba for projection. Usually faster, but if your numbe
            installation is wonky, it may produce falty results.

        Returns
        -------
        velocities_projected : npt.NDArray[np.float64]
            Velocities projected onto the eigenvectors structured as follows:
            [number of time steps, number of frequencies]

        """
        if mass_weighted:
            velocities = self.get_velocities_mass_weighted(
                steps=steps, cutoff_start=cutoff_start, cutoff_end=cutoff_end
            )
        else:
            velocities = self.get_velocities(
                steps=steps, cutoff_start=cutoff_start, cutoff_end=cutoff_end
            )

        velocities_projected = vu.get_normal_mode_decomposition(
            velocities,
            projection_vectors,
            use_numba=use_numba,
        )

        return velocities_projected

    def get_temperature(
        self, steps: int = 1, cutoff_start: int = 0, cutoff_end: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Get atomic temperature along an MD trajectory.

        Parameters
        ----------
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.

        Returns
        -------
        temperature : npt.NDArray[np.float64]

        """
        temperature = []

        for ind in range(cutoff_start, len(self.traj) - cutoff_end):
            if ind % steps == 0.0:
                atoms = self.traj[ind]
                unconstrained_atoms = len(atoms) - len(
                    atoms.constraints[0].index
                )

                ekin = atoms.get_kinetic_energy() / unconstrained_atoms
                temperature.append(ekin / (1.5 * units.kB))

        return np.array(temperature, dtype=np.float64)

    def get_total_energy(
        self, steps: int = 1, cutoff_start: int = 0, cutoff_end: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Get atomic total energy along an MD trajectory.

        Parameters
        ----------
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.

        Returns
        -------
        total_energy : npt.NDArray[np.float64]

        """
        total_energy = []

        for ind in range(cutoff_start, len(self.traj) - cutoff_end):
            if ind % steps == 0.0:
                atoms = self.traj[ind]
                total_energy.append(atoms.get_total_energy())

        return np.array(total_energy, dtype=np.float64)

    def get_coords(self, atoms):
        unconstrained_atoms = atoms.constraints[0].index

        coords = []
        for ind, atom in enumerate(atoms):
            if not ind in unconstrained_atoms:
                coords.append(atom.position)

        return np.array(coords)

    def get_atomic_displacements(
        self,
        coords_0=None,
        steps: int = 1,
        cutoff_start: int = 0,
        cutoff_end: int = 0,
    ) -> npt.NDArray[np.float64]:
        """
        Get atomic atomic displacements with respect to the first time step
        along an MD trajectory.

        Parameters
        ----------
        steps : int, optional
            Read every nth step. The default is 1 -> all steps are read. If for
            instance steps=5 every 5th step is read.
        cutoff_start : int, optional
            Cutoff n stept at the beginning of the trajectory. The default is 0.
        cutoff_end : int, optional
            Cutoff n stept at the end of the trajectory. The default is 0.

        Returns
        -------
        atomic_displacements : npt.NDArray[np.float64]

        """
        atomic_displacements = []

        if coords_0 is None:
            coords_0 = self.get_coords(self.traj[cutoff_start])

        for ind in range(cutoff_start, len(self.traj) - cutoff_end):
            if ind % steps == 0.0:
                coords = self.get_coords(self.traj[ind])

                disp = np.linalg.norm(coords - coords_0, axis=1)
                atomic_displacements.append(disp)

        return np.array(atomic_displacements, dtype=np.float64)
