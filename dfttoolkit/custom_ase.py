import os.path

import numpy as np
from ase.calculators.aims import Aims
from ase.calculators.calculator import FileIOCalculator
from ase.io import aims


class CustomAims(Aims):
    """
    A custom FHI-aims calculator which addresses bugs in the default ASE
    implementation.

    Please refer to the ASE calculator for the documentation:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/FHI-aims.html#module-ase.calculators.aims
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_input(
        self,
        atoms,
        properties=None,
        system_changes=None,
        ghosts=None,
        geo_constrain=None,
        scaled=False,
        velocities=None,
    ):

        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        if scaled is None:
            scaled = np.all(atoms.get_pbc())
        if velocities is None:
            velocities = atoms.has("momenta")

        if geo_constrain is None:
            geo_constrain = scaled and "relax_geometry" in self.parameters

        have_lattice_vectors = atoms.pbc.any()
        have_k_grid = "k_grid" in self.parameters or "kpts" in self.parameters

        if have_lattice_vectors and not have_k_grid:
            raise RuntimeError("Found lattice vectors but no k-grid!")
        if not have_lattice_vectors and have_k_grid:
            raise RuntimeError("Found k-grid but no lattice vectors!")

        aims.write_aims(
            os.path.join(self.directory, "geometry.in"),
            atoms,
            scaled,
            geo_constrain,
            velocities=velocities,
            ghosts=ghosts,
        )

        self.write_control(atoms, os.path.join(self.directory, "control.in"))
        self.write_species(atoms, os.path.join(self.directory, "control.in"))
        self.parameters.write(os.path.join(self.directory, "parameters.ase"))
