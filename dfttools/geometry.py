import ast
import colorsys
import copy
import itertools
import os
import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from fractions import Fraction
from typing import Tuple, Union

import ase
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance
import spglib
from ase.constraints import FixAtoms
from scipy.spatial import distance_matrix

import dfttools.utils.math_utils as utils
import dfttools.utils.units as units
from dfttools.utils.periodic_table import PeriodicTable


class Geometry:
    """
    This class represents a geometry file for (in principle) any DFT code.
    In practice it has only been fully implemented for FHI-aims geometry.in
    files.

    """

    def __init__(self, filename: Union[str, None] = None):
        """
        filename : str
            Path to text file of geometry.

        center: dict
            atom indices and linear combination of them to define the center of a molecule.
            Used only for mapping to first unit cell.
            Example: if the center should be the middle between the first three atoms,
            center would be {1:1/3,2:1/3,3:1/3}
        """
        self.species = []
        self.lattice_vectors = np.zeros([3, 3])
        self.comment_lines = []
        self.constrain_relax = np.zeros([0, 3], bool)
        self.external_force = np.zeros([0, 3], np.float64)
        self.calculate_friction = np.zeros([0], np.float64)
        self.initial_moment = []
        self.initial_charge = []
        self.name = filename
        self.center = None
        self.energy = None
        self.forces = None
        self.hessian = None
        self.geometry_parts = []  # list of lists: indices of each geometry part
        self.geometry_part_descriptions = (
            []
        )  # list of strings:  name of each geometry part
        self.symmetry_axes = None
        self.inversion_index = None
        self.vacuum_level = None
        self.multipoles = []
        self.homogeneous_field = None
        self.read_as_fractional_coords = False
        self.symmetry_params = None
        self.n_symmetry_params = None
        self.symmetry_LVs = None
        self.symmetry_frac_coords = None  # symmetry_frac_coords should have str values, not float, to include the parameters
        self.original_lattice_vectors = None  # Save the original lattice vectors if the geometry is periodically replicated

        if filename is None:
            self.n_atoms = 0
            self.coords = np.zeros([self.n_atoms, 3])
        else:
            self.read_from_file(filename)

        self.periodic_table = PeriodicTable()

    def __eq__(self, other):
        if len(self) != len(other):
            equal = False
        else:
            equal = np.allclose(self.coords, other.coords)
            equal = equal and np.allclose(self.lattice_vectors, other.lattice_vectors)
            equal = equal and self.species == other.species
        return equal

    def __len__(self):
        return self.n_atoms

    def __add__(self, other_geometry):
        geom = copy.deepcopy(self)
        geom += other_geometry
        return geom

    def __iadd__(self, other_geometry):
        self.add_geometry(other_geometry)
        return self

    def get_instance_of_other_type(self, geometry_type):
        if geometry_type == "aims":
            new_geometry = AimsGeometry()
        elif geometry_type == "vasp":
            new_geometry = VaspGeometry()
        elif geometry_type == "xyz":
            new_geometry = XYZGeometry()
        elif geometry_type == "molden":
            new_geometry = MoldenGeometry()
        elif geometry_type == "xsf":
            new_geometry = XSFGeometry()
        else:
            raise ValueError(f'Type "{geometry_type}" is not availlable.')

        new_geometry.__dict__ = self.__dict__
        return new_geometry

    ###############################################################################
    #                             INPUT PARSER                                    #
    ###############################################################################
    def read_from_file(self, filename):
        """
        Parses a geometry file

        """
        with open(filename) as f:
            text = f.read()

        self.parse(text)

    def parse(self, text):
        raise NotImplementedError

    def add_top_comment(self, comment_string):
        """
        Adds comments that are saved at the top of the geometry file.

        """
        lines = comment_string.split("\n")
        for l in lines:
            if not l.startswith("#"):
                l = "# " + l
            self.comment_lines.append(l)

    ###############################################################################
    #                             OUTPUT PARSER                                   #
    ###############################################################################
    def save_to_file(self, filename, **kwargs):

        geometry_type = get_file_format_from_ending(filename)

        new_geometry = self.get_instance_of_other_type(geometry_type)

        text = new_geometry.get_text(**kwargs)

        # Enforce linux file ending, even if running on windows machine by
        # using binary mode
        with open(filename, "w", newline="\n") as f:
            f.write(text)

    def get_text(self, **kwargs):
        raise NotImplementedError

    ###############################################################################
    #                            Data exchange with ASE                           #
    ###############################################################################
    def get_from_ase_atoms_object(self, atoms):
        """Reads an ASE.Atoms object. Taken from ase.io.aims and adapted. Only basic features are implemented.
        Args:
            atoms: ase.atoms.Atoms
                structure to output to the file
            scaled: bool
                If True use fractional coordinates instead of Cartesian coordinates
            info_str: str
                A string to be added to the header of the file
            wrap: bool
                Wrap atom positions to cell before writing

        """

        if isinstance(atoms, (list, tuple)):
            if len(atoms) > 1:
                raise RuntimeError(
                    "Don't know how to save more than " "one image to FHI-aims input"
                )
            else:
                atoms = atoms[0]  # pyright:ignore

        if atoms.get_pbc().any():
            self.lattice_vectors = np.array(atoms.get_cell())

        fix_cart = np.zeros([len(atoms), 3])
        if atoms.constraints:
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    fix_cart[constr.index] = [True, True, True]
        constrain_relax = fix_cart

        coords = []
        species_list = []
        for i, atom in enumerate(atoms):
            species = atom.symbol
            if isinstance(species, int):
                species = self.periodic_table.get_chemical_symbol(species)
            species_list.append(species)
            coords.append(atom.position)
        coords = np.array(coords)
        self.add_atoms(coords, species_list, constrain_relax)

    def get_as_ase(self):
        """
        Convert geometry file to ASE object

        """
        # atoms_string = ""
        atom_coords = []
        atom_numbers = []
        atom_constraints = []
        for i in range(self.n_atoms):
            # Do not export 'emptium" atoms
            if self.species[i] != "Em":
                atom_coords.append(self.coords[i, :])
                atom_numbers.append(
                    self.periodic_table.get_atomic_number(self.species[i])
                )

                if np.any(self.constrain_relax[i]):
                    atom_constraints.append(i)

        ase_system = ase.Atoms(numbers=atom_numbers, positions=atom_coords)
        ase_system.cell = self.lattice_vectors

        c = FixAtoms(indices=atom_constraints)
        ase_system.set_constraint(c)

        if not np.sum(self.lattice_vectors) == 0.0:
            ase_system.pbc = [1, 1, 1]
            
        if self.energy is not None:
            ase_system.info['energy'] = self.energy
            
        if self.forces is not None:
            ase_system.arrays['forces'] = self.forces
                
        return ase_system

    ###############################################################################
    #                    Adding and removing atoms (in place)                     #
    ###############################################################################
    def add_atoms(
        self,
        cartesian_coords,
        species,
        constrain_relax=None,
        initial_moment=None,
        initial_charge=None,
        external_force=None,
        calculate_friction=None,
    ) -> None:
        """
        Add additional atoms to the current geometry file.

        Parameters
        ----------
        cartesion_coords : List of numpy arrays of shape [nx3]
            coordinates of new atoms
        species : list of strings
            element symbol for each atom
        constrain_relax : list of lists of bools (optional)
            [bool,bool,bool] (for [x,y,z] axis) for all atoms that should be constrained during a geometry relaxation

        Retruns
        -------
        None

        """
        if constrain_relax is None or len(constrain_relax) == 0:
            constrain_relax = np.zeros([len(species), 3], bool)
        if external_force is None or len(external_force) == 0:
            external_force = np.zeros([len(species), 3], np.float64)
        if calculate_friction is None:
            calculate_friction = np.array([False] * len(species))
        if initial_moment is None:
            initial_moment = [0.0] * len(species)
        if initial_charge is None:
            initial_charge = [0.0] * len(species)
        # TODO: this should not be necessary as self.coords should always be a np.array
        if not hasattr(self, "coords") or self.coords is None:
            assert isinstance(cartesian_coords, np.ndarray)
            self.coords = cartesian_coords
        else:
            self.coords = np.concatenate((self.coords, cartesian_coords), axis=0)
        self.species += species
        self.n_atoms = self.coords.shape[0]

        self.constrain_relax = np.concatenate(
            (self.constrain_relax, constrain_relax), axis=0
        )
        self.external_force = np.concatenate(
            (self.external_force, external_force), axis=0
        )
        self.calculate_friction = np.concatenate(
            (self.calculate_friction, calculate_friction)
        )
        self.initial_moment += initial_moment
        self.initial_charge += initial_charge

    def add_geometry(self, geometry):
        """Adds full geometry to initial GeometryFile."""

        # check parts: (needs to be done before adding atoms to self)
        if hasattr(self, "geometry_parts") and hasattr(geometry, "geometry_parts"):
            for part, name in zip(
                geometry.geometry_parts, geometry.geometry_part_descriptions
            ):
                if len(part) > 0:
                    self.geometry_parts.append([i + self.n_atoms for i in part])
                    self.geometry_part_descriptions.append(name)

        # some lines of code in order to preserve backwards compatibility
        if not hasattr(geometry, "external_force"):
            geometry.external_force = np.zeros([0, 3], np.float64)
        self.add_atoms(
            geometry.coords,
            geometry.species,
            constrain_relax=geometry.constrain_relax,
            initial_moment=geometry.initial_moment,
            initial_charge=geometry.initial_charge,
            external_force=geometry.external_force,
            calculate_friction=geometry.calculate_friction,
        )

        # check lattice vectors:
        # g has lattice and self not:
        if not np.any(self.lattice_vectors) and np.any(geometry.lattice_vectors):
            self.lattice_vectors = np.copy(geometry.lattice_vectors)

        # both have lattice vectors:
        elif np.any(self.lattice_vectors) and np.any(geometry.lattice_vectors):
            warnings.warn(
                "Caution: The lattice vectors of the first file will be used!"
            )

        # add multipoles
        self.add_multipoles(geometry.multipoles)

        # check center:
        # g has center and self not:
        if hasattr(self, "center") and hasattr(geometry, "center"):
            if self.center is None and geometry.center is not None:
                self.center = geometry.center.copy()
            # both have a center:
            elif self.center is not None and geometry.center is not None:
                warnings.warn("Caution: The center of the first file will be used!")

    def add_multipoles(self, multipoles):
        """
        Adds multipoles to the the geometry.
        Each multipole is defined as a list: [x, y, z, order, charge]
        With: x,y,z: cartesian coordinates
              order: 0 for monopoles, 1 for dipoles
              charge: charge
        :param multipoles: list of float, or list of lists
        :return:
        """
        # if multiple multipoles are given: indented lists
        if len(multipoles) == 0:
            return
        if isinstance(multipoles[0], list):
            for x in multipoles:
                self.multipoles.append(x)
        # else: one list
        else:
            self.multipoles.append(multipoles)

    def remove_atoms(self, atom_inds: npt.NDArray[np.int64]) -> None:
        """
        Remove atoms with indices atom_inds. If no indices are specified, all
        atoms are removed.

        Parameters
        ----------
        atom_inds : np.array
            Indices of atoms to be removed.

        Returns
        -------
        None

        """
        if hasattr(self, "geometry_parts") and len(self.geometry_parts) > 0:
            # (AE): added "len(self.geometry_parts) > 0" to suppress this frequent warning when it is supposely not relevant (?)
            warnings.warn(
                "CAUTION: geometry_parts indices are not updated after atom deletion!!\n \
                           You are welcome to implement this!!"
            )
        mask = np.ones(len(self.species), dtype=bool)
        mask[atom_inds] = False

        self.species = list(np.array(self.species)[mask])
        self.constrain_relax = self.constrain_relax[mask, :]
        self.external_force = self.external_force[mask, :]
        self.calculate_friction = self.calculate_friction[mask]
        self.coords = self.coords[mask, :]
        self.n_atoms = len(self.constrain_relax)

        if hasattr(self, "hessian") and self.hessian is not None:
            flat_mask = np.kron(mask, np.ones(3, dtype=bool))
            new_dim = np.sum(flat_mask)
            a, b = np.meshgrid(flat_mask, flat_mask)
            hess_mask = np.logical_and(a, b)
            new_hessian = self.hessian[hess_mask].reshape(new_dim, new_dim)
            self.hessian = new_hessian

    def remove_atoms_by_species(self, species: str) -> None:
        """
        Removes all atoms of a given species.

        Parameters
        ----------
        species : str
            Atom species to be removed.

        Returns
        -------
        None

        """
        L = np.array(self.species) == species
        atom_inds = np.where(L)[0]
        self.remove_atoms(atom_inds)

    def remove_constrained_atoms(self) -> None:
        """
        Remove all atoms where all coordinates are constrained.

        Returns
        -------
        None

        """
        remove_inds = self.get_constrained_atoms()
        self.remove_atoms(remove_inds)

    def remove_unconstrained_atoms(self):
        """
        Remove all atoms where all coordinates are constrained.

        Returns
        -------
        None

        """
        remove_inds = self.get_unconstrained_atoms()
        self.remove_atoms(remove_inds)

    def truncate(self, n_atoms: int) -> None:
        """
        Keep only the first n_atoms atoms

        Parameters
        ----------
        n_atoms : int
            Number of atoms to be kept.

        Returns
        -------
        None

        """
        self.species = self.species[:n_atoms]
        self.constrain_relax = self.constrain_relax[:n_atoms]
        self.external_force = self.external_force[:n_atoms]
        self.calculate_friction = self.calculate_friction[:n_atoms]
        self.coords = self.coords[:n_atoms, :]
        self.n_atoms = n_atoms

    def crop(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        auto_margin=False,
    ) -> None:
        """
        Removes all atoms that are outside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders

        """
        indices_to_remove = self.get_cropping_indices(xlim, ylim, zlim, auto_margin)
        self.remove_atoms(indices_to_remove)

    def crop_inverse(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        auto_margin=False,
    ) -> None:
        """Removes all atoms that are inside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders"""
        indices_to_keep = self.get_cropping_indices(xlim, ylim, zlim, auto_margin)
        indices_to_remove = np.array(
            [i for i in range(self.n_atoms) if i not in indices_to_keep]
        )
        self.remove_atoms(indices_to_remove)

    def crop_to_unit_cell(self, lattice=None, frac_coord_factors=[0, 1]):
        """Removes all atoms that are outside the given unit cell. Similar to self.crop() but allows for arbitrary unit cells"""
        # Atoms that have fractional coordinates outside the defined frac_coord_factors are removed. Per default frac_coord_factors=[0,1]

        if lattice is None:
            lattice = self.lattice_vectors
        frac_coords = utils.get_fractional_coords(self.coords, lattice)

        remove_inds = []
        remove_inds += list(np.where(frac_coords[:, 0] >= frac_coord_factors[1])[0])
        remove_inds += list(np.where(frac_coords[:, 1] >= frac_coord_factors[1])[0])
        remove_inds += list(np.where(frac_coords[:, 0] < frac_coord_factors[0])[0])
        remove_inds += list(np.where(frac_coords[:, 1] < frac_coord_factors[0])[0])
        remove_inds += list(np.where(frac_coords[:, 2] > frac_coord_factors[1])[0])
        remove_inds += list(np.where(frac_coords[:, 2] < frac_coord_factors[0])[0])

        remove_inds = np.array(set(remove_inds))

        self.remove_atoms(remove_inds)
        self.lattice_vectors = lattice

        # In the following all redundant atoms, i.e. atoms that are multiplied at the same position when the unitcell is repeated periodically, are removed from the new unit cell
        epsilon = 0.1
        # Distance in Angstrom for which two atoms are assumed to be in the same position
        init_geom = self
        allcoords = init_geom.coords

        # generate all possible translation vectors that could map an atom of the unit cell into itsself
        prim_lat_vec = []
        for i in range(3):
            prim_lat_vec.append(
                [init_geom.lattice_vectors[i], -init_geom.lattice_vectors[i]]
            )
        self_mapping_translation_vectors = []

        for i in prim_lat_vec:
            for sign in range(2):
                self_mapping_translation_vectors.append(i[sign])

        for i in range(3):
            for sign0 in range(2):
                for k in range(3):
                    for sign1 in range(2):
                        if i != k:
                            # print(f'i {i} k {k} sign0 {sign0} sign1 {sign1}')
                            single_addition_vector = (
                                prim_lat_vec[i][sign0] + prim_lat_vec[k][sign1]
                            )
                            self_mapping_translation_vectors.append(
                                single_addition_vector
                            )

        for i in range(3):
            for sign0 in range(2):
                for k in range(3):
                    for sign1 in range(2):
                        for l in range(3):
                            for sign2 in range(2):
                                if i != k and i != l and k != l:
                                    single_addition_vector = (
                                        prim_lat_vec[i][sign0]
                                        + prim_lat_vec[k][sign1]
                                        + prim_lat_vec[l][sign2]
                                    )
                                    self_mapping_translation_vectors.append(
                                        single_addition_vector
                                    )

        ## Find the indices of those atoms that are equivalent, i.e. atoms that are doubled when the unit cell is repeated periodically

        doubleindices = []  # list of pairs of atom indices that are equivalent
        for i, coords_i in enumerate(allcoords):
            for trans_l in self_mapping_translation_vectors:
                coords_i_shift_l = copy.deepcopy(coords_i)
                coords_i_shift_l += trans_l
                for j, coords_j in enumerate(allcoords):
                    if j != i:
                        distance_i_shift_l_j = np.linalg.norm(
                            coords_i_shift_l - coords_j
                        )
                        if distance_i_shift_l_j < epsilon:
                            doubleindices.append([i, j])

        for i in range(len(doubleindices)):
            doubleindices[i].sort()

        ###################################################################
        ##Create a list of redundant atoms according to the atoms that are equivalent
        # according to all the pairs in doubleindices

        liste = doubleindices
        to_be_killed = []  # List of all atom indicess that are redundant
        for i, liste_i in enumerate(liste):
            replacer = liste_i[0]
            to_be_replaced = liste_i[1]
            to_be_killed.append(to_be_replaced)
            for j, liste_j in enumerate(liste):
                for k in range(2):
                    if liste_j[k] == to_be_replaced:
                        liste[j][k] = replacer
        remainers = [j[0] for j in liste]
        for r in remainers:
            for k in to_be_killed:
                if k == r:
                    to_be_killed.remove(k)

        self.remove_atoms(np.array(to_be_killed))

    def remove_metal_atoms(self) -> None:
        "Removes all atoms with atomic number > 18 and atomic numbers 3,4,11,12,13,14"
        metal_atoms = self.get_indices_of_metal()
        self.remove_atoms(metal_atoms)

    def remove_non_metallic_atoms(self) -> None:
        "Removes all atoms that are not metal"
        mol_inds = self.get_indices_of_molecules()
        self.remove_atoms(mol_inds)

    def remove_substrate(self, primitive_substrate) -> None:
        """
        Removes all substrate atoms given the primitive substrate by
        identifying species and height

        Parameters
        ----------
        primitive_substrate: Geometry
            primitive substrate file of system

        dimension: int
            dimension to use as z-axis

        threshold: float
            height threshold in A
        """
        substrate_indices = self.get_substrate_indices(
            primitive_substrate=primitive_substrate
        )
        self.remove_atoms(substrate_indices)

    def remove_adsorbates(self, primitive_substrate=None) -> None:
        """
        Removes all atoms that are not part of the substrate

        """
        adsorbate_indices = self.get_adsorbate_indices(
            primitive_substrate=primitive_substrate
        )
        self.remove_atoms(adsorbate_indices)

    def remove_collisions(self, keep_latest: Union[bool, slice] = True) -> None:
        """Removes all atoms that are in a collision group as given by GeometryFile.getCollidingGroups.

        Args:
            keep_latest (Union[bool, slice], optional): Whether to keep the earliest or latest added.
                If a slice object is given, the selection is used to determine which atoms to keep.
                Defaults to True.

        Raises:
            ValueError: Raised when keep_latest is neither a bool nor a slice object.
        """
        indices = []
        if isinstance(keep_latest, bool):
            if keep_latest:
                selection = slice(None, -1, None)
            else:
                selection = slice(1, None, None)
        elif isinstance(keep_latest, slice):
            selection = keep_latest
        collisions = self.get_colliding_groups()
        for group in collisions:
            indices += group[selection]
        self.remove_atoms(np.array(indices))

    ###############################################################################
    #                         Transformations (in place)                          #
    ###############################################################################
    def map_to_first_unit_cell(
        self, lattice_vectors=None, dimensions=np.array(range(3))
    ):
        """
        Maps the coordinate of a geometry in multiples of the substrate lattice
        vectors to a point that is closest to the origin

        lattice_vectors : float-array
            lattice vectors of the substrate

        dimensions : float-array
            dimensions (x, y, z) where the mapping should be done

        """
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors

        assert not np.allclose(
            lattice_vectors, np.zeros([3, 3])
        ), "Lattice vector must be defined in Geometry or given as function parameter"

        frac_coords = utils.get_fractional_coords(self.coords, lattice_vectors)
        # modulo 1 maps all coordinates to first unit cell
        frac_coords[:, dimensions] = frac_coords[:, dimensions] % 1
        new_coords = utils.get_cartesian_coords(frac_coords, lattice_vectors)
        self.coords = new_coords

    def map_center_of_atoms_to_first_unit_cell(self, lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors

        assert not np.allclose(
            lattice_vectors, np.zeros([3, 3])
        ), "Lattice vector must be defined in Geometry or given as function parameter"

        offset = self.get_geometric_center()
        frac_offset = utils.get_fractional_coords(offset, lattice_vectors)
        frac_offset = np.floor(frac_offset)
        self.move_all_atoms_by_fractional_coords(-frac_offset, lattice_vectors)

    def center_coordinates(
        self, ignore_center_attribute: bool = False, dimensions=np.array(range(3))
    ):
        """
        Shift the coordinates of a geometry such that the "center of mass" or
        specified center lies at (0,0,0)

        Parameters
        ----------
        ignore_center_attribute : bool
            Switch usage of *center* attribute off/on. The default is False.

        dimensions: np.array
            Dimensions that should be cnetered. The default is False [0, 1, 2].

        Returns
        -------
        None.

        """
        offset = self.get_geometric_center(
            ignore_center_attribute=ignore_center_attribute
        )[dimensions]
        self.coords[:, dimensions] -= offset
        return offset

    def move_all_atoms(self, shift):
        """
        Translates the whole geometry by vector 'shift'
        """
        self.coords += shift

    def move_all_atoms_by_fractional_coords(self, frac_shift, lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors

        self.coords += utils.get_cartesian_coords(frac_shift, lattice_vectors)

    def move_adsorbates(self, shift, primitive_substrate=None):
        """
        shifts the adsorbates in Cartesian coordinates
        """
        adsorbates = self.get_adsorbates(primitive_substrate=primitive_substrate)
        adsorbates.coords += shift

        self.remove_adsorbates(primitive_substrate=primitive_substrate)
        self += adsorbates

    def rotate_lattice_around_axis(
        self,
        angle_in_degree: float,
        axis: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
    ) -> None:
        """
        Rotates lattice around a given axis.

        Parameters
        ----------
        angle_in_degree : float
            angle of rotation

        axis: np.array
            Axis around which to rotate. The default is

        Returns
        -------
        None.

        """
        R = utils.get_rotation_matrix_around_axis(axis, angle_in_degree * np.pi / 180)
        self.lattice_vectors = np.dot(self.lattice_vectors, R)

    def rotate_coords_around_axis(
        self,
        angle_in_degree,
        axis: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
        center=None,
        indices=None,
    ) -> None:
        """Rotates structure COUNTERCLOCKWISE around a point defined by <center>.

        If center == None, the geometric center of the structure (as defined by
        self.get_geometric_center())
        is used as the pivot for the rotation.
        """
        # ??? <aj, 25.2.2020> not sure if center should be the geometric center or [0,0,0], both
        # have their pros and cons, former version was [0,0,0], new version is geometric center
        if indices is None:
            indices = np.arange(self.n_atoms)
        if center is None:
            center = self.get_geometric_center(indices=indices)

        R = utils.get_rotation_matrix_around_axis(axis, angle_in_degree * np.pi / 180)
        temp_coords = copy.deepcopy(self.coords[indices])
        temp_coords -= center
        temp_coords = np.dot(temp_coords, R)
        temp_coords += center
        self.coords[indices] = temp_coords

    def mirror_through_plane(self, normal_vector: npt.NDArray[np.float64]) -> None:
        """
        Mirrors the geometry through the plane defined by the normal vector.
        """
        mirror_matrix = utils.get_mirror_matrix(normal_vector=normal_vector)
        self.transform(mirror_matrix)

    def align_into_xy_plane(self, atom_indices):
        """Rotates a planar molecule (defined by 3 atom indices) into the XY plane.
        Double check results, use with caution"""
        p1 = self.coords[atom_indices[0]]
        p2 = self.coords[atom_indices[1]]
        p3 = self.coords[atom_indices[2]]

        X = np.zeros([3, 3])
        X[0, :] = p2 - p1
        X[1, :] = p3 - p1
        X[2, :] = np.cross(X[0], X[1])
        for i in range(3):
            X[i] /= np.linalg.norm(X[i])
        X[1, :] = np.cross(X[2], X[0])

        U = np.linalg.inv(X)
        self.transform(U)

    def align_with_z_vector(self, new_z_vec: npt.NDArray[np.float64]) -> None:
        """
        Transforms the coordinate system of the geometry file to a new z-vector
        calculates rotation martrix for coordinate transformation to new z-vector
        and uses it to transform coordinates of geometry object

        Parameters
        ----------
        new_z_vec : npt.NDArray[np.float64]
            The vector to align with the z-axis.

        Returns
        -------
        None

        """
        # get old_positions
        old_positions = self.coords

        # normalize new_z_vec
        new_z_vec = new_z_vec / np.linalg.norm(new_z_vec)

        # Check if the desired vector is antiparallel to the z-axis
        if np.allclose(new_z_vec, -np.array([0, 0, 1])):
            rotation_matrix = np.diag([-1, -1, 1])  # Antiparallel case
        else:
            # Calculate the rotation matrix
            z_axis = np.array([0, 0, 1])
            cross_product = np.cross(new_z_vec, z_axis)
            dot_product = np.dot(new_z_vec, z_axis)
            skew_symmetric_matrix = np.array(
                [
                    [0, -cross_product[2], cross_product[1]],
                    [cross_product[2], 0, -cross_product[0]],
                    [-cross_product[1], cross_product[0], 0],
                ]
            )
            rotation_matrix = (
                np.eye(3)
                + skew_symmetric_matrix
                + np.dot(skew_symmetric_matrix, skew_symmetric_matrix)
                * (1 - dot_product)
                / (np.linalg.norm(cross_product) ** 2)
            )

        # Apply the rotation to all positions
        rotated_positions = np.dot(old_positions, rotation_matrix.T)

        self.coords = rotated_positions

    def align_vector_to_vector(
        self, vector: npt.NDArray[np.float64], vector_to_align: npt.NDArray[np.float64]
    ):
        """
        Aligns a vector and the atomic coordiantes to a given vector.

        vector : npt.NDArray[np.float64]
            vector for alignment

        vector_to_align : npt.NDArray[np.float64]
            index of the lattice vector that should be aligned

        """
        vector_to_align_normed = vector_to_align / np.linalg.norm(vector_to_align)

        vector_normed = vector / np.linalg.norm(vector)

        R = utils.get_rotation_matrix(vector_normed, vector_to_align_normed)

        self.lattice_vectors = np.dot(self.lattice_vectors, R)
        self.coords = np.dot(self.coords, R)

    def align_lattice_vector_to_vector(self, vector, lattice_vector_index):
        """
        Aligns a lattice vector and the atomic coordiantes to a given axis.

        vector : array
            vector for alignment

        lattice_vector_index : int
            index of the lattice vector that should be aligned

        """
        lattice_vector_normed = self.lattice_vectors[
            lattice_vector_index
        ] / np.linalg.norm(self.lattice_vectors[lattice_vector_index])

        self.align_vector_to_vector(vector, lattice_vector_normed)

    def align_cartiesian_axis_to_vector(self, vector, axis_index):
        """
        Aligns a lattice vector and the atomic coordiantes to a given axis.

        vector : array
            vector for alignment

        axis_index : int
            index of the axis that should be aligned

        """
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = 1.0

        self.align_vector_to_vector(vector, axis)

    def align_with_view_direction(
        self, view_direction: npt.NDArray[np.float64]
    ) -> None:

        view_direction /= np.linalg.norm(view_direction)

        vec_z = np.array([0.0, 0.0, -1.0])

        view_direction_y = np.cross(vec_z, view_direction)
        norm_y = np.linalg.norm(view_direction_y)

        if norm_y == 0.0:
            sign = np.dot(vec_z, view_direction)
            self.lattice_vectors[2] *= sign
            self.coords[:, 2] *= sign

        else:
            # Orient z-axis in view direction
            self.align_cartiesian_axis_to_vector(-view_direction, 2)

    def align_main_axis_along_xyz(self) -> None:
        """
        Align coordinates of rodlike molecules along specified axis

        """
        vals, vecs = self.get_main_axes()
        R = np.linalg.inv(vecs.T)
        self.coords = np.dot(self.coords, R)

    def transform(
        self,
        R: npt.NDArray[np.float64],
        t: npt.NDArray[np.float64] = np.array([0, 0, 0]),
        rotation_center: Union[npt.NDArray[np.float64], None] = None,
        atom_indices: Union[npt.NDArray[np.int64], None] = None,
    ) -> None:
        """
        Performs a symmetry transformation of the coordinates by rotation and
        translation. The transformation is applied as
        x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]

        Parameters
        ----------
        R : np.array
            Rotation matrix in Catesian coordinates.
        t : np.array, optional
            Translation vector in Catesian coordinates. The default is np.array([0,0,0]).
        rotation_center : np.array | None, optional
            Centre of rotation. The default is None.
        atom_indices : np.array | None, optional
            List of indexes of atoms that should be transformed. The default is
            None.

        Returns
        -------
        None

        """
        if atom_indices is None:
            atom_indices = np.arange(self.n_atoms)
        if rotation_center is None:
            temp_coords = np.dot(self.coords[atom_indices, :], R) + t
            self.coords[atom_indices, :] = temp_coords
        else:
            temp_coords = copy.deepcopy(self.coords[atom_indices, :])
            temp_coords -= rotation_center
            temp_coords = np.dot(temp_coords, R) + t
            temp_coords += rotation_center
            self.coords[atom_indices, :] = temp_coords

    def transform_fractional(
        self, R: npt.NDArray[np.float64], t: npt.NDArray[np.float64], lattice=None
    ):
        """
        Transforms the coordinates by rotation and translation, where R,t are
        given in fractional coordinates
        The transformation is applied as c_new[3x1] = R[3x3] * c_old[3x1] + t[3x1]

        """
        if lattice is None:
            lattice = self.lattice_vectors
        coords_frac = utils.get_fractional_coords(self.coords, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1, 3])
        self.coords = utils.get_cartesian_coords(coords_frac, lattice)

    def transform_lattice(
        self,
        R: npt.NDArray[np.float64],
        t: npt.NDArray[np.float64] = np.array([0, 0, 0]),
    ) -> None:
        """
        Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]
        Notice that this works in cartesian coordinates.
        Use transform_fractional if you got your R and t from get_symmetries

        Parameters
        ----------
        R : np.array
            Rotation matrix in Catesian coordinates.
        t : np.array, optional
            Translation vector in Catesian coordinates. The default is np.array([0,0,0]).

        Returns
        -------
        None

        """
        new_lattice_vectors = np.dot(self.lattice_vectors, R) + t
        self.lattice_vectors = new_lattice_vectors

    def transform_lattice_fractional(self, R, t, lattice):
        """Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]"""
        coords_frac = utils.get_fractional_coords(self.lattice_vectors, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1, 3])
        self.lattice_vectors = utils.get_cartesian_coords(coords_frac, lattice)

    def swap_lattice_vectors(self, axis_1=0, axis_2=1):
        """
        Can be used to interchange two lattice vectors
        Attention! Other values - for instance k_grid - will stay unchanged!!
        :param axis_1 integer [0,1,2]
        :param axis_2 integer [0,1,2]     axis_1 !=axis_2
        :return:

        """
        self.lattice_vectors[[axis_1, axis_2], :] = self.lattice_vectors[
            [axis_2, axis_1], :
        ]
        self.coords[[axis_1, axis_2], :] = self.coords[[axis_2, axis_1], :]

    def symmetrize(self, symmetry_operations, center=None):
        """
        Symmetrizes Geometry with given list of symmetry operation matrices
        after transferring it to the origin.
        Do not include the unity matrix for symmetrizing, as it is already the first geometry!
        ATTENTION: use symmetrize_periodic to reliably symmetrize periodic structures

        """
        if center is not None:
            offset = center
            self.coords -= center
        else:
            offset = np.mean(self.coords, axis=0)
            self.center_coordinates()
        temp_coords = copy.deepcopy(
            self.coords
        )  # this corresponds to the unity matrix symmetry operation
        for R in symmetry_operations:
            new_geom = copy.deepcopy(self)
            new_geom.transform(R)
            new_geom.reorder_atoms(self.get_transformation_indices(new_geom))
            temp_coords += new_geom.coords
        self.coords = temp_coords / (len(symmetry_operations) + 1) + offset

    def average_with(self, other_geometries) -> None:
        """
        Average self.coords with those of other_geometries and apply on self
        ATTENTION: this can change bond lengths etc.!Ok n

        Parameters
        ----------
        other_geometries: List of Geometrys ... might be nice to accept list of coords too

        Returns
        -------
        None

        """
        if len(other_geometries) > 0:
            offset = (
                self.get_geometric_center()
            )  # Attribute center should be used if it exists
            self.coords -= offset

            for other_geom in other_geometries:
                geom = copy.deepcopy(other_geom)
                # center all other geometries to remove offset
                geom.center_coordinates()
                # all geometries have to be ordered like first geometry in order to sum them
                geom.reorder_atoms(self.get_transformation_indices(geom))
                self.coords += geom.coords
            self.coords /= len(other_geometries) + 1  # +1 for this geometry itself
            self.coords += offset

    def reorder_atoms(self, inds: npt.NDArray[np.int64]) -> None:
        """
        eorders Atoms with index list

        Parameters
        ----------
        inds : npt.NDArray[np.int64]
            Array of indices with new order of atoms.

        Returns
        -------
        None

        """
        self.coords = self.coords[inds, :]
        self.species = [self.species[i] for i in inds]
        self.constrain_relax = self.constrain_relax[inds, :]
        self.initial_charge = [self.initial_charge[i] for i in inds]
        self.initial_moment = [self.initial_moment[i] for i in inds]

        inds_hessian = []
        for ind in inds:
            inds_new = np.array([0, 1, 2]) + 3 * ind
            inds_hessian.append(inds_new)

        inds_hessian = np.array(inds_hessian).flatten()

        if self.hessian is not None:
            self.hessian = self.hessian[inds_hessian, :]
            self.hessian = self.hessian[:, inds_hessian]

    def shift_to_bottom(self):
        """
        Shifts the coordiantes such that the coordinate with the samllest
        z-value ists at (x, y, 0).

        Returns
        -------
        None.

        """
        min_z = np.min(self.coords[:, -1])
        self.coords[:, -1] -= min_z

    def displace_atoms(
        self, displacement_strength: float, displacement_indices: npt.NDArray[np.int64]
    ) -> None:
        """
        Displaces atoms randomly.

        Parameters
        ----------
        displacement_strength : float
            Scaling factor for the strenght of the dispacements.
        displacement_indices : npt.NDArray[np.int64]
            Indices where atoms should be dispaced.

        Returns
        -------
        None

        """
        displacements = np.random.rand(len(displacement_indices), 3) - 0.5
        displacements *= displacement_strength

        self.coords[displacement_indices, :] += displacements

    def move_multipoles(self, shift: npt.NDArray[np.float64]) -> None:
        """
        Moves all the multipoles by a shift vector
        :param shift: list or array, len==3
        :return:
        """
        assert len(shift) == 3
        for m in self.multipoles:
            m[0] += shift[0]
            m[1] += shift[1]
            m[2] += shift[2]

    ###############################################################################
    #                      Set Properties of the Geometry                         #
    ###############################################################################
    def set_vacuum_height(self, vac_height, bool_shift_to_bottom=False) -> None:
        if bool_shift_to_bottom:
            self.shift_to_bottom()
        min_z = np.min(self.coords[:, -1])
        max_z = np.max(self.coords[:, -1])
        self.lattice_vectors[-1, -1] = max_z + vac_height - min_z

        if vac_height < min_z:
            raise Exception(
                """set_vacuum_height: the defined vacuum height is smaller than
                height of the lowest atom. Shift unit cell either manually or by
                the keyword bool_shift_to_bottom towards the bottom
                of the unit cell."""
            )
        self.lattice_vectors[-1, -1] = max_z + vac_height - min_z

    def set_vacuum_level(self, vacuum_level: float) -> None:
        """
        Sets vacuum level of geometry calculation

        Parameters
        ----------
        vacuum_level : float
            Height of the vacuum level.

        Returns
        -------
        None

        """
        self.vacuum_level = vacuum_level

    def set_multipoles_charge(self, charge: npt.NDArray[np.float64]) -> None:
        """
        Sets the charge of all multipoles

        Parameters
        ----------
        charge : list or float or int

        Returns
        -------
        None

        """
        if isinstance(charge, list):
            assert len(charge) == len(self.multipoles)
            for i, m in enumerate(self.multipoles):
                m[4] = charge[i]
        else:
            for i, m in enumerate(self.multipoles):
                m[4] = charge

    def set_constraints(
        self,
        indices_of_atoms_to_constrain: list,
        constrain_dim_flags=None,
    ):
        """
        Sets a constraint for a few atoms in the system (identified by
        'indices_of_atoms_to_constrain') for a geometry relaxation.
        Since the relaxation constraint can be in any and/or all dimensions
        the second parameter, 'constraint_dim_flags', makes it possible to
        set which dimension(s) should be constrained for which molecule.
        By default all dimensions are to be constrained for all atoms are
        constrained. If the dimension to constrain should be set individually
        for different atoms, you need to provide a list of booleans of the shape
        len(indices_of_atoms_to_constrain) x 3, which contains the constrain
        flags for each dimension for each atom.

        Parameters
        ----------
        indices_of_atoms_to_constrain : list
            List of atoms to constrain.
        constrain_dim_flags : list[boolean]
            The default is: [True, True, True].

        Returns
        -------
        None

        """
        if constrain_dim_flags is None:
            constrain_dim_flags = [True, True, True]

        self.constrain_relax[indices_of_atoms_to_constrain, :] = constrain_dim_flags

    def set_constraints_based_on_space(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        constrain_dim_flags=None,
    ):
        """
        Constrain all atoms that are within a cuboid (defined by
        limits in all dimensions: xlim, etc.) for a geometry relaxation.

        It is possible to define which dimension will be constrained, but since
        the number of atoms in the cuboid is only calculated at runtime
        the dimensions may only be set for all atoms at once. If you need to
        set them individually please use set_constraints.

        Parameters
        ----------
        zlim
        xlim
        ylim
        constrain_dim_flags : list[boolean]
            The default is: [True, True, True].

        Returns
        -------
        None

        """
        if constrain_dim_flags is None:
            constrain_dim_flags = [True, True, True]

        # --- get indices of all atoms outside the required interval ---
        indices_outside = self.get_cropping_indices(
            xlim=xlim, ylim=ylim, zlim=zlim, auto_margin=False
        )
        # ---

        # --- Filter all that are outside ---
        # The indices of the atoms of relevance to us are all that are NOT
        # outside of the cuboid
        indices_inside = [i for i in range(len(self)) if i not in indices_outside]
        # ---

        self.set_constraints(indices_inside, constrain_dim_flags)

    def free_all_constraints(self) -> None:
        """
        Frees all constraints.

        Returns
        -------
        None

        """
        self.constrain_relax = np.zeros([len(self.species), 3], bool)

    def set_calculate_friction(
        self, indices_of_atoms: list, calculate_friction: bool = True
    ) -> None:
        """
        Sets to calculate electronic friction for atoms specified by the given
        list of indices.

        Parameters
        ----------
        indices_of_atoms : list
            List of atoms from which electronic friction should be calculated.
        calculate_friction : bool, optional
            Calculate friction for these atims. The default is True.

        Returns
        -------
        None

        """
        self.calculate_friction[indices_of_atoms] = calculate_friction

    def free_all_calculate_friction(self) -> None:
        """
        Set calculate electronic friction to false on all atoms.

        Returns
        -------
        None

        """
        self.calculate_friction = np.array([False] * len(self))

    def set_external_forces(
        self,
        indices_of_atoms: Union[int, npt.NDArray[np.int64]],
        external_force: npt.NDArray[np.float64],
    ) -> None:
        """
        Sets a constraint for a few atoms in the system (identified by
        'indices_of_atoms_to_constrain') for a geometry relaxation.
        Since the relaxation constraint can be in any and/or all dimensions
        the second parameter, 'constraint_dim_flags', makes it possible to
        set which dimension(s) should be constrained for which molecule.
        By default all dimensions are to be constrained for all atoms are
        constrained. If the dimension to constrain should be set individually
        for different atoms, you need to provide a list of booleans of the shape
        len(indices_of_atoms_to_constrain) x 3, which contains the constrain
        flags for each dimension for each atom.

        Parameters
        ----------
        indices_of_atoms_to_constrain : int or npt.NDArray[np.int64]
            Index of atoms to which atoms should should be applied.
        constrain_dim_flags : npt.NDArray[np.float64]
            Force that should act on a given atom.
        """

        self.external_force[indices_of_atoms, :] = external_force

    def free_all_external_forces(self) -> None:
        """
        Set calculate electronic friction to false on all atoms.

        Returns
        -------
        None

        """
        self.external_force = np.zeros((len(self), 3))

    def remove_periodicity(self):
        """
        Makes geometry non-periodic by setting its lattice vectors to zero

        """
        self.lattice_vectors = np.zeros((3, 3), dtype=float)

    def set_homogeneous_field(self, E):
        """Field should be a numpy array (Ex, Ey, Ez) with the Field in V/A"""
        assert len(E) == 3, "Expected E-field components [Ex, Ey, Ez], but got " + str(
            E
        )
        self.homogeneous_field = np.asarray(E)

    def free_homogeneous_field(self):
        self.homogeneous_field = np.array([0.0, 0.0, 0.0])

    ###############################################################################
    #                      Get Properties of the Geometry                         #
    ###############################################################################
    def get_is_periodic(self) -> bool:
        """
        Checks if the geometry is periodic.

        Returns
        -------
        bool
            Ture if geometry is periodic.

        """
        return not np.allclose(self.lattice_vectors, np.zeros([3, 3]))

    def get_reassembled_molecule(self, threshold: float = 2.0):

        geom_replica = self.get_periodic_replica(
            (1, 1, 1), explicit_replications=[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        )

        tree = scipy.spatial.KDTree(geom_replica.coords)
        pairs = tree.query_pairs(threshold)

        new_cluster = True

        while new_cluster:

            clusters = []
            new_cluster = False

            for pair in pairs:

                in_culster = False
                for ind, indices in enumerate(clusters):

                    for p in pair:
                        if p in indices:
                            clusters[ind] = set(list(indices) + list(pair))
                            new_cluster = True
                            in_culster = True
                            break

                if not in_culster:
                    clusters.append(set(pair))

            pairs = copy.deepcopy(clusters)

        for index_array in pairs:
            if len(index_array) == len(self):
                final_geom = geom_replica.get_atoms_by_indices(
                    np.sort(np.array(list(index_array), dtype=np.int32))
                )
                final_geom.lattice_vectors = self.lattice_vectors
                final_geom.map_center_of_atoms_to_first_unit_cell()

                return final_geom

        warnings.warn(
            "Geometry.getReassembledMolecule could not reassemble \
                      molecule. Returning original Geometry."
        )
        return self

    def get_scaled_copy(self, scaling_factor):
        """
        Returns a copy of the geometry, scaled by scaling_factor.
        Both the coordinates of the atoms and the length of the lattice vectors are affected
        :param scaling_factor: int or iterable of length 3
                if int: the VOLUME will be scaled accordingly
                if iterable: the LENGTH of the lattice vectors will be scaled accordingly
        :return: Geometry
        """

        assert hasattr(
            self, "lattice_vectors"
        ), "This function only works for geometries with a Unit Cell"

        if isinstance(scaling_factor, float):
            scaling_factors = [
                scaling_factor ** (1 / 3),
            ] * 3
        else:
            assert len(scaling_factor) == 3
            scaling_factors = scaling_factor
        scaled_geom = deepcopy(self)
        lattice_vectors = deepcopy(self.lattice_vectors)
        lattice_vectors[0] *= scaling_factors[0]
        lattice_vectors[1] *= scaling_factors[1]
        lattice_vectors[2] *= scaling_factors[2]

        new_coords = utils.get_cartesian_coords(
            self.get_fractional_coords(), lattice_vectors
        )
        scaled_geom.lattice_vectors = lattice_vectors
        scaled_geom.coords = new_coords

        return scaled_geom
    
    def get_displaced_atoms(
        self,
        displacement_strength: float,
        displace_only_unconstrained: bool = True,
    ):
        """
        Returns a copy of the geometry, where the atoms have been displaced
        randomly.

        Parameters
        ----------
        displacement_strength : float
            Scaling factor for the strenght of the dispacements.
        displace_only_unconstrained : bool
            Indices where atoms should be dispaced. The default is True.

        Returns
        -------
        geometry.

        """
        if displace_only_unconstrained:
            displacement_indices = self.get_unconstrained_atoms()
        else:
            displacement_indices = np.array(range(len(self)))

        new_geometry = deepcopy(self)
        new_geometry.displace_atoms(displacement_strength, displacement_indices)

        return new_geometry

    def get_fractional_coords(self, lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors

        assert not np.allclose(
            lattice_vectors, np.zeros([3, 3])
        ), "Lattice vector must be defined in Geometry or given as function parameter"

        fractional_coords = np.linalg.solve(lattice_vectors.T, self.coords.T)
        return fractional_coords.T

    def get_fractional_lattice_vectors(self, lattice_vectors=None):
        """
        Calculate the fractional representation of lattice vectors of the
        geometry file in another basis.
        Useful to calculate epitaxy matrices

        """
        fractional_coords = np.linalg.solve(lattice_vectors.T, self.lattice_vectors.T)
        return fractional_coords.T

    def get_reciprocal_lattice_vectors(self) -> npt.NDArray[np.float64]:
        """
        Calculate the reciprocal lattice of the Geometry lattice_vectors in standard form
        For convention see en.wikipedia.org/wiki/Reciprocal_lattice

        Returns
        -------
        recip_lattice : npt.NDArray[np.float64]
            Row-wise reciprocal lattice vectors (3x3)

        """
        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]

        volume = np.cross(a1, a2).dot(a3)

        b1 = np.cross(a2, a3)
        b2 = np.cross(a3, a1)
        b3 = np.cross(a1, a2)

        recip_lattice = np.array([b1, b2, b3]) * 2 * np.pi / volume
        return recip_lattice

    def get_main_axes(
        self, weights: Union[str, npt.NDArray[np.float64]] = "unity"
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get main axes and eigenvalues of a molecule
        https://de.wikipedia.org/wiki/Tr%C3%A4gheitstensor

        Parameters
        ----------
        weights : Union[str, npt.NDArray[np.float64]] = "unity"
            Specifies how the atoms are weighted.
            "unity": all same weight
            "Z": weighted by atomic number.
            The default is 'unity'.

        Returns
        -------
        vals : npt.NDArray[np.float64]
            TODO DESCRIPTION.
        vecs : npt.NDArray[np.float64]
            TODO DESCRIPTION.

        """
        if weights == "unity":
            weights = np.ones(self.n_atoms)
        elif weights == "Z":
            weights = np.array(
                [self.periodic_table.get_atomic_mass(s) for s in self.species]
            )

        coords = self.coords - np.mean(self.coords, axis=0)
        diag_entry = np.sum(np.sum(coords**2, axis=1) * weights)
        I = np.eye(3) * diag_entry
        for i in range(self.n_atoms):
            I -= weights[i] * np.outer(coords[i, :], coords[i, :])

        vals, vecs = scipy.linalg.eigh(I)
        sort_ind = np.argsort(vals)

        return vals[sort_ind], vecs[:, sort_ind]

    def get_distance_between_all_atoms(self) -> npt.NDArray[np.float64]:
        """
        Get the distance between all atoms in the current Geometry
        object. Gives an symmetric array where distances between atom i and j
        are denoted in the array elements (ij) and (ji).

        """
        distances = scipy.spatial.distance.cdist(self.coords, self.coords)
        return distances

    def get_closest_atoms(self, indices, species=None, n_closest=1):
        """
        Get the indices of the closest atom(s) for the given index or list of
        indices

        Parameters
        ----------
        index: int or iterable
            atoms for which the closest indices are  to be found
        species: None or list of species identifiers
            species to consider for closest atoms. This allows to get only the
            closest atoms of the same or another species
        n_closest: int
            number of closest atoms to return

        Returns
        -------
        closest_atoms_list: list or list of lists
            closest atoms for each entry in index
        """

        all_distances = self.get_distance_between_all_atoms()

        if species is None:
            species_to_consider = list(set(self.species))
        else:
            assert isinstance(
                species, list
            ), "species must be a list of species identifiers or None if all atoms should be probed"
            species_to_consider = species

        return_single_list = False
        if not isinstance(indices, Iterable):
            return_single_list = True
            indices = [indices]

        indices_to_consider = []
        for i, s in enumerate(self.species):
            if (s in species_to_consider) and (i not in indices):
                indices_to_consider.append(i)
        indices_to_consider = np.array(indices_to_consider)

        closest_atoms_list = []
        for index in indices:
            distances = all_distances[index, indices_to_consider]
            distance_indices = np.argsort(distances)
            closest_atoms = indices_to_consider[distance_indices]
            if len(closest_atoms) > n_closest:
                closest_atoms = closest_atoms[:n_closest]

            closest_atoms_list.append(closest_atoms.tolist())

        if return_single_list:  # can only be true if only a single index was specified
            return closest_atoms_list[0]
        else:
            return closest_atoms_list

    def get_distance_between_two_atoms(self, atom_indices: list) -> np.float64:
        """Get the distance between two atoms in the current Geometry
        object."""
        atom1 = self.coords[atom_indices[0], :]
        atom2 = self.coords[atom_indices[1], :]
        vec = atom2 - atom1
        dist = np.linalg.norm(vec)

        return dist

    def get_volume_of_unit_cell(self) -> np.float64:
        """
        Calcualtes the volume of the unit cell.

        Returns
        -------
        volume : float
            Volume of the unit cell.

        """
        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]
        volume = np.cross(a1, a2).dot(a3)
        return volume

    def get_geometric_center(
        self, ignore_center_attribute=False, indices=None
    ) -> npt.NDArray[np.float64]:
        """
        Returns the center of the geometry. If the attribute *center* is set,
        it is used as the definition for the center of the geometry.

        Parameters
        ----------
        ignore_center_attribute : Bool
            If True, the attribute self.center is used
            Otherwise, the function returns the geometric center of the structure,
            i.e. the average over the position of all atoms.

        indices: iterable of indices or None
            indices of all atoms to consider when calculating the center. Useful to calculate centers of adsorbates only
            if None, all atoms are used

        Returns
        --------
        center : npt.NDArray[np.float64]
            Center of the geometry
        """

        if (
            not hasattr(self, "center")
            or self.center is None
            or ignore_center_attribute
            or indices is not None
        ):
            if indices is None:
                indices = np.arange(self.n_atoms)
            center = np.mean(self.coords[indices], axis=0)
        else:
            center = np.zeros([3])
            for i, weight in self.center.items():
                center += self.coords[i, :] * weight
        return center

    def get_center_of_mass(self) -> npt.NDArray[np.float64]:
        """
        Mind the difference to self.get_geometric_center

        Returns
        -------
        center_of_mass: npt.NDArray[np.float64]
            The 3D-coordinate of the center of mass

        """
        species_helper = []
        for si in self.species:
            si_new = si.split("_")[0]
            species_helper.append(si_new)

        masses_np = np.array(
            [self.periodic_table.get_atomic_mass(s) for s in species_helper],
            dtype=np.float64,
        )
        center_of_mass = self.coords.T.dot(masses_np) / masses_np.sum()
        return center_of_mass

    def get_symmetries(self, save_directory=None, symmetry_precision=1e-05):
        """
        Returns symmetries (rotation and translation matrices) from spglig.
        works only for unitcell and supercell geometries (lattice vecotrs must not be 0)

        Beware: The returned symmetry matrices are given with respect to
        fractional coordinates, not Cartesian ones!

        See https://atztogo.github.io/spglib/python-spglib.html#get-symmetry for details

        Parameters:
        -----------
        save_directory : str
            save directory in string format, file will be name symmetry.pickle
            (default = None --> symmetry is not saved)
        """

        if np.count_nonzero(self.lattice_vectors) == 0:
            print(
                "Lattice vectors must not be 0! getSymmetry requires a unitcell-like geometry file!"
            )
            raise ValueError(self.lattice_vectors)

        unit_cell = self.get_spglib_cell()
        symmetry = spglib.get_symmetry(unit_cell, symprec=symmetry_precision)

        if not save_directory == None:
            if not os.path.exists(save_directory):
                print("symmetry not saved; save_directory does not exist")
            else:
                save_directory = os.path.join(save_directory, "symmetry.pickle")
                pickle.dump(
                    symmetry,
                    open(save_directory, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        return symmetry

    def get_atomic_numbers_of_atoms(self) -> npt.NDArray[np.float64]:
        """Get the atomic numbers of all atoms in the geometry file"""
        species = [self.periodic_table.get_atomic_number(s) for s in self.species]
        return np.array(species)

    def get_number_of_electrons(self) -> float:
        """
        Determines the number of electrons.

        Returns
        -------
        float
            Number of electrons.

        """
        electrons = []
        for s in self.species:
            try:
                if "_" in s:
                    curr_species = s.split("_")[0]
                else:
                    curr_species = s
                electrons.append(self.periodic_table.get_atomic_number(curr_species))

            except KeyError:
                KeyError("Species {} is not known".format(s))
        return np.sum(electrons)

    def get_atomic_masses(self) -> npt.NDArray[np.float64]:
        """
        Determines the atomic mass for all atoms.

        Returns
        -------
        masses : np.array
            List of atomic masses for all atoms in the same order as
            the atoms.

        """
        masses = []
        for s in self.species:
            try:
                if "_" in s:
                    curr_species = s.split("_")[0]
                else:
                    curr_species = s

                mass = self.periodic_table.get_atomic_mass(curr_species)
                masses.append(mass)

            except KeyError:
                KeyError("Atomic mass for species {} is not known".format(s))
        return np.array(masses)

    def get_total_mass(self) -> float:
        """
        Determines the atomic mass of the entrie geometry.

        Returns
        -------
        atomic_mass : float
            Atomic mass of the entrie geometry.

        """
        atomic_mass = 0

        for s in self.species:
            atomic_mass += self.periodic_table.get_atomic_mass(s)

        return atomic_mass

    def get_largest_atom_distance(self, dims_to_consider=(0, 1, 2)) -> float:
        """
        Find largest distance between atoms in geometry.
        #search tags; molecule length, maximum size

        Parameters
        ----------
        dims_to_consider : list
            Dimensions along which largest distance should be calculated.

        Returns
        -------
        geometry_size : float
            Largest distance between two atoms in the unit cell.

        """
        mask = np.array([i in dims_to_consider for i in range(3)], dtype=bool)
        geometry_size = 0.0
        for ind1 in range(self.n_atoms):
            for ind2 in range(ind1, self.n_atoms):
                geometry_size_test = np.linalg.norm(
                    self.coords[ind1][mask] - self.coords[ind2][mask]
                )

                if geometry_size_test > geometry_size:
                    geometry_size = float(geometry_size_test)

        return geometry_size

    def get_area(self) -> np.float64:
        """
        Returns the area of the surface described by lattice_vectors 0 and 1 of
        the geometry, assuming that the lattice_vector 2 is orthogonal to both.

        Returns
        -------
        area : float
            Area of the unit cell.

        """
        a = deepcopy(self.lattice_vectors[0, :-1])
        b = deepcopy(self.lattice_vectors[1, :-1])
        area = np.abs(np.cross(a, b)[-1])

        return area

    def get_area_in_atom_numbers(
        self,
        substrate_indices: Union[npt.NDArray[np.float64], None] = None,
        substrate=None,
    ) -> float:
        """
        Returns the area of the unit cell in terms of substrate atoms in the
        topmost substrate layer. The substrate is determined by
        default by the function self.getSubstrate(). For avoiding a faulty
        substrate determination it can be either given through indices or
        through the substrate geometry itself

        Parameters
        ----------
        substrate_indices : Union[np.array, None], optional
            List of indices of substrate atoms. The default is None.
        substrate : TYPE, optional
            Geometry of substrate. The default is None.

        Returns
        -------
        float
            Area of the unit cell in units of the area of the substrate.

        """
        topmost_sub_layer = self.get_substrate_layers(
            layer_indices=[0],
            substrate_indices=substrate_indices,
            primitive_substrate=substrate,
        )
        return topmost_sub_layer.n_atoms

    def get_bond_lengths(self, bond_factor: float = 1.5) -> npt.NDArray[np.float64]:
        """
        Parameters
        ----------
        Parameter for bond detection based on atom-distance : float, optional
            DESCRIPTION. The default is 1.5.

        Returns
        -------
        bond_lengths : npt.NDArray[np.float64]
            List of bond lengths for neighbouring atoms.

        """

        raise NotImplementedError

        # TODO write the below function
        neighbouring_atoms = self.get_all_neighbouring_atoms(bond_factor=bond_factor)

        bond_lengths = []
        for v in neighbouring_atoms.values():
            bond_lengths.append(v[1])

        bond_lengths = np.array(bond_lengths)
        return bond_lengths

    def get_number_of_atom_layers(self, threshold: float = 1e-2) -> Tuple[dict, float]:
        """
        Return the number of atom layers.

        Parameters
        ----------
        threshold : float, optional
            Threshold to determine the number of layers. The default is 1e-2.

        Returns
        -------
        dict
            Number of layers per atom species.
        float
            Total number of layers.

        """
        layers = self.get_atom_layers_indices(threshold=threshold)

        total_number_layers = 0
        for atom_species in layers:
            layers[atom_species] = len(layers[atom_species])
            total_number_layers += layers[atom_species]

        return layers, total_number_layers

    def get_unit_cell_parameters(
        self,
    ) -> Tuple[np.float64, np.float64, np.float64, float, float, float]:
        """
        Determines unit cell parameters.

        Returns
        -------
        a : float
            Length of lattice vector 1.
        b : float
            Length of lattice vector 2.
        c : float
            Length of lattice vector 3.
        alpha : float
            Angle between lattice vectors 1 and 2.
        beta : float
            Angle between lattice vectors 2 and 3.
        gamma : float
            Angle between lattice vectors 1 and 3.

        """
        cell = self.lattice_vectors

        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        c = np.linalg.norm(cell[2])

        alpha = np.arccos(np.dot(cell[1], cell[2]) / (c * b))
        gamma = np.arccos(np.dot(cell[1], cell[0]) / (a * b))
        beta = np.arccos(np.dot(cell[2], cell[0]) / (a * c))

        return a, b, c, alpha, beta, gamma

    def get_spglib_cell(self):
        """
        Returns the unit cell in a format that can be used in spglib (to find symmetries)

        return : tuple
            (lattice vectors, frac coordinates of atoms, atomic numbers)
        """
        coordinates = utils.get_fractional_coords(self.coords, self.lattice_vectors)

        atom_number = []
        for atom_name in self.species:
            atom_number.append(self.periodic_table.get_atomic_number(atom_name))

        return (self.lattice_vectors, coordinates, atom_number)

    def get_orientation_of_main_axis(self) -> float:
        """
        Get the orientation of the main axis relative to the x axis
        the main axis is transformed such that it always points in the upper half of cartesian space

        Returns:
        float
            angle between main axis and x axis in degree
        """
        main_ax = self.get_main_axes()[1][:, 0]
        if main_ax[1] < 0:
            main_ax *= -1
        return np.arctan2(main_ax[1], main_ax[0]) * 180 / np.pi

    def get_constrained_atoms(self) -> npt.NDArray[np.int64]:
        """
        Returns indice of constrained atoms.

        Returns
        -------
        inds : npt.NDArray[np.int64]
            Indice of constrained atoms.

        """
        constrain = np.any(self.constrain_relax, axis=1)
        inds = [i for i, c in enumerate(constrain) if c]
        inds = np.array(inds)
        return inds

    def get_unconstrained_atoms(self) -> npt.NDArray[np.int64]:
        """
        Returns indice of unconstrained atoms.

        Returns
        -------
        inds : npt.NDArray[np.int64]
            Indice of unconstrained atoms.

        """
        all_inds = list(range(len(self)))
        keep_inds = self.get_constrained_atoms()
        inds = np.array(list(set(all_inds) - set(keep_inds)))
        return inds

    def get_cropping_indices(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        auto_margin=False,
        inverse=False,
    ) -> npt.NDArray[np.int64]:
        """
        Gets indices of all atoms that are outside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders
        :param inverse : if True, gets indices of all atoms INSIDE specified bounds

        """
        assert len(xlim) == 2, "xlim must have a lower and an upper bound"
        assert len(ylim) == 2, "ylim must have a lower and an upper bound"
        assert len(zlim) == 2, "zlim must have a lower and an upper bound"
        if auto_margin:
            margin = max(
                [self.periodic_table.get_covalent_radius(s) for s in self.species]
            )
            xlim = xlim[0] - margin, xlim[1] + margin
            ylim = ylim[0] - margin, ylim[1] + margin
            zlim = zlim[0] - margin, zlim[1] + margin

        remove = np.zeros(len(self), bool)
        remove = remove | (self.coords[:, 0] < xlim[0])
        remove = remove | (self.coords[:, 0] > xlim[1])
        remove = remove | (self.coords[:, 1] < ylim[0])
        remove = remove | (self.coords[:, 1] > ylim[1])
        remove = remove | (self.coords[:, 2] < zlim[0])
        remove = remove | (self.coords[:, 2] > zlim[1])
        indices_to_remove = np.arange(len(self))[remove]
        if inverse:
            indices_to_remove = [
                i for i in range(self.n_atoms) if i not in indices_to_remove
            ]

        indices_to_remove = np.array(indices_to_remove)

        return indices_to_remove

    def get_substrate_indices_from_parts(
        self, do_warn=True
    ) -> Union[None, npt.NDArray[np.int64]]:
        """
        Returns indices of those atoms that a part of the substrate. The
        definition of the substrate does NOT rely of it being a metal or the
        height or something like that. Instead a geometry part named
        'substrate' must be defined.

        Parameters
        ----------
        do_warn : boolean
            Can be set to False to suppress warnings

        Returns
        -------
        substrate_indices : npt.NDArray[np.int64]

        """
        substrate_key = "substrate"  # should probably moved to a class variable, but first finished folder-read-write

        if not hasattr(self, "geometry_parts") or not hasattr(
            self, "geometry_part_descriptions"
        ):
            if do_warn:
                print("getSubstrate: geometry parts not defined")
            return None

        if substrate_key not in self.geometry_part_descriptions:
            if do_warn:
                print(
                    "getSubstrate: geometry parts are defined, "
                    "but part '{}' not found".format(substrate_key)
                )
            return None

        index_of_geometry_parts_substrate = self.geometry_part_descriptions.index(
            substrate_key
        )
        substrate_indices = self.geometry_parts[index_of_geometry_parts_substrate]
        substrate_indices = np.array(substrate_indices)
        return substrate_indices

    def get_indices_of_metal(self) -> npt.NDArray[np.int64]:
        """
        Gets indices of all atoms with atomic number > 18 and atomic numbers
        3,4,11,12,13,14

        Returns
        -------
        metal_atoms : npt.NDArray[np.int64]
            Inidices of metallic atoms.

        """
        atom_inds = [self.periodic_table.get_atomic_number(s) for s in self.species]
        metal_atoms = []
        for i, ind in enumerate(atom_inds):
            if (ind > 18) or (ind in [3, 4, 11, 12, 13, 14]):
                metal_atoms.append(i)

        metal_atoms = np.array(metal_atoms)
        return metal_atoms

    def get_indices_of_molecules(self, substrate_species=None) -> npt.NDArray[np.int64]:
        """
        Fetches the indices of the substrate atoms, but it defaults to
        just returning all non-metal atom's indices!

        If substrate_species is given indices of all atoms that are not of the
        substrate species are returned.

        """
        if substrate_species:
            substrate_indices = self.get_indices_of_species(substrate_species)
        else:
            substrate_indices = self.get_indices_of_metal()

        molecules = [i for i in range(self.n_atoms) if i not in substrate_indices]
        molecules = np.array(molecules)
        return molecules

    def get_indices_of_species(self, species) -> npt.NDArray[np.int64]:
        """
        Returns all indices of atoms the are of species defined in the input.
        species can be a string of a list

        """
        # make sure species is a list
        if isinstance(species, str):
            species = [species]

        species_indices = [i for i in range(self.n_atoms) if self.species[i] in species]
        species_indices = np.array(species_indices)
        return species_indices

    def get_substrate_indices(
        self, primitive_substrate=None, dimension=2, threshold=0.3
    ) -> npt.NDArray[np.int64]:
        """
        This method returns the indices of all atoms that are part of the substrate.
        Often these are simply all metal atoms of the geometry.
        But the substrate can also be organic in which case it can't be picked by being a metal.
        And there might be multiple adsorbate layers in which case only the molecules of the highest layer
        shall be counted as adsorbates and the others are part of the substrate.

        dimension: int      only used if primitive_substrate is not None
        threshold: float    only used if primitive_substrate is not None

        Returns
        -------
        indices of all substrate atoms: npt.NDArray[np.int64]

        """
        # case 1: if a primitive_substrate was passed, use that one for the decision
        # (copied from self.removeSubstrate)

        substrate_indices = []

        # case 2: if no substrate was passed but a geometry_parts "substrate" is defined in geometry_parts, use that one
        substrate_indices_from_parts = self.get_substrate_indices_from_parts(
            do_warn=False
        )

        if primitive_substrate is not None:
            substrate_species = set(primitive_substrate.species)
            substrate_heights = primitive_substrate.coords[:, dimension]
            substrate_candidate_indices = [
                i for i, s in enumerate(self.species) if s in substrate_species
            ]
            for c in substrate_candidate_indices:
                if np.any(
                    np.absolute(substrate_heights - self.coords[c, dimension])
                    < threshold
                ):
                    substrate_indices.append(c)

        elif substrate_indices_from_parts is not None:
            substrate_indices = substrate_indices_from_parts

        else:
            # case 3: if neither a substrate was passed, nor a geometry_parts "substrate" is defined,
            #            use a fallback solution: assume that the substrate (and nothing else) is a metal
            warnings.warn(
                "Geometry.getIndicesOfAdsorbates: Substrate is not explicitly defined. "
                "Using fallback solution of counting all metal atoms as substrate."
            )
            substrate_indices = self.get_indices_of_metal()

        return np.array(substrate_indices)

    def get_adsorbate_indices(self, primitive_substrate=None) -> npt.NDArray[np.int64]:
        """
        This method returns the indices of all atoms that are NOT part of the substrate.
        In a classical organic monolayer on a metal substrate these are simply all molecules.
        But the substrate can also be organic in which case it can't be picked by being a metal.
        And there might be multiple adsorbate layers in which case only the molecules of the highest layer
        shall be counted as adsorbates.

        Returns
        -------
        indices of all adsorbate atoms: npt.NDArray[np.int64]
        """

        substrate_indices = self.get_substrate_indices(
            primitive_substrate=primitive_substrate
        )
        # invert:
        return np.array(
            [i for i in self.get_indices_of_all_atoms() if i not in substrate_indices]
        )

    def get_indices_of_all_atoms(self, species=None):
        if species is None:
            return [i for i in range(self.n_atoms)]
        else:
            return [i for i in range(self.n_atoms) if self.species[i] == species]

    def get_atom_layers_indices(self, threshold: float = 1e-2) -> dict:
        """
        Returns a dict of the following form.

        Parameters
        ----------
        threshold : float, optional
            Treshold within which atoms are considered to be in the same layer.
            The default is 1e-2.

        Returns
        -------
        layers : dict
            {<Element symbol>: {height: [indices of atoms of element at height]}}.

        """
        layers = {}

        for ind, atom_coord in enumerate(self.coords):
            atom_species = self.species[ind]
            if atom_species not in layers:
                layers[atom_species] = {}

            add_new_z_coord = True
            for z_coord in layers[atom_species].keys():
                if abs(atom_coord[2] - z_coord) < threshold:
                    layers[atom_species][z_coord].append(ind)
                    add_new_z_coord = False

            if add_new_z_coord:
                layers[atom_species][atom_coord[2]] = [ind]

        return layers

    def get_atom_layers_indices_by_height(self, threshold: float = 1e-2) -> dict:
        """
        Similarly to get_atom_layers_indices this function returns a dict
        continaing info about height and the indices of atoms at that height.

        Parameters
        ----------
        threshold : float, optional
            reshold within which atoms are considered to be in the same layer.
            The default is 1e-2.

        Returns
        -------
        layers_by_height : dict
            {height: [indices of atoms at that height]}.

        """
        layers_by_species = self.get_atom_layers_indices(threshold=threshold)

        layers_by_height = defaultdict(list)

        # --- merge height-indices dicts ---
        for data in layers_by_species.values():

            for height, indices in data.items():
                new = True
                for new_height in layers_by_height.keys():
                    if abs(height - new_height) < threshold:
                        layers_by_height[new_height] += indices
                        new = False
                if new:
                    layers_by_height[height] += indices

        # sort dictionary by descending height
        layers_by_height = dict(sorted(layers_by_height.items(), reverse=True))
        return layers_by_height

    def get_list_of_neighbouring_atoms(self, bond_factor: float = 1.5) -> dict:
        """
        Get a dictionary of neighbouring atoms.

        Parameters
        ----------
        bond_factor : float, optional
            Multiply for covelent radius. If the distance between two atoms is
            smaller thand (r_0+r_1)*bond_factor, the atoms are considered
            neighbours. The default is 1.5.

        Returns
        -------
        neighbouring_atoms : dict
            {(index_0, index_1) : [(element_0, element_1), atom distance]}.

        """
        coords = self.coords
        species = self.species

        all_species = set(species)
        all_species_pairs = itertools.product(all_species, repeat=2)

        bond_thresholds = {}
        for pair in all_species_pairs:
            bond_thresholds[pair] = (
                self.periodic_table.get_covalent_radius(pair[0])
                + self.periodic_table.get_covalent_radius(pair[1])
            ) * bond_factor

        neighbouring_atoms = {}

        for i, coord in enumerate(coords):
            for j, coord_test in enumerate(coords):
                if i >= j:
                    continue

                pair_index = (i, j)

                if pair_index not in neighbouring_atoms:
                    dist = np.linalg.norm(coord - coord_test)

                    pair_species = (species[i], species[j])

                    if dist < bond_thresholds[pair_species]:
                        neighbouring_atoms[pair_index] = [pair_species, dist]

        return neighbouring_atoms

    def get_principal_moments_of_inertia(self) -> npt.NDArray[np.float64]:
        """
        Calculates the eigenvalues of the moments of inertia matrix

        Returns
        -------
        moments : np.array, shape=(3,), dtype=np.float64
            principal moments of inertia in kg * m**2

        """
        masses_kg = [
            units.ATOMIC_MASS_IN_KG * self.periodic_table.get_atomic_mass(s)
            for s in self.species
        ]

        center_of_mass = self.get_center_of_mass()
        r_to_center_in_m = units.ANGSTROM_IN_METER * (self.coords - center_of_mass)

        ###########
        # begin: code based on ase/atoms.py: get_moments_of_inertia
        # (GNU Lesser General Public License)
        # Initialize elements of the inertial tensor
        I11 = I22 = I33 = I12 = I13 = I23 = 0.0
        for i in range(len(self)):
            x, y, z = r_to_center_in_m[i]
            m = masses_kg[i]

            I11 += m * (y**2 + z**2)
            I22 += m * (x**2 + z**2)
            I33 += m * (x**2 + y**2)
            I12 += -m * x * y
            I13 += -m * x * z
            I23 += -m * y * z

        I = np.array(
            [[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]], dtype=np.float64
        )

        moments, _ = np.linalg.eigh(I)
        return moments

    def get_homogeneous_field(self):
        """Field is a numpy array (Ex, Ey, Ez) with the Field in V/A"""
        if not hasattr(self, "_homogeneous_field"):
            self.homogeneous_field = None

        return self.homogeneous_field

    ###############################################################################
    #               Get Properties in comparison to other Geometry                #
    ###############################################################################
    def get_distance_to_equivalent_atoms(self, other_geometry) -> np.float64:
        """
        Calculate the maximum distance that atoms of geom would have to be moved,
        to coincide with the atoms of self.

        """
        _, dist = self.get_transformation_indices(other_geometry, get_distances=True)
        return np.max(dist)

    def get_transformation_indices(
        self,
        other_geometry,
        get_distances=False,
        periodic_2D=False,
    ):
        """
        Associates every atom in self to the closest atom of the same specie in
        other_geometry.

        If self should be orderd like other_geometry then this is done in the
        following way:
        >>> transformation_indices = other_geometry.get_transformation_indices(self)
        >>> self.reorder_atoms(transformation_indices)

        Parameters
        ----------
        other_geometry: geometry
        norm_threshold : float
        get_distances : bool
            The default is False.
        periodic_2D : bool
            The default is False.

        Returns
        -------
        transformation_indices : np.array.
            The positions on the array correspond to the atoms in self;
            the values of the array correspond to the atoms in other_geometry

        """
        assert len(self) == len(
            other_geometry
        ), "Geometries have different number of atoms {0} != {1}".format(
            len(self), len(other_geometry)
        )

        # Replicate other geometry to also search in neighbouring cells
        if periodic_2D:
            other_geometry = other_geometry.get_periodic_replica((3, 3, 1))
            other_geometry.move_all_atoms_by_fractional_coords([-1 / 3.0, -1 / 3.0, 0])

        # Get the atomic numbers of each geometry file: Later only compare matching atom types
        Z_values_1 = np.array(
            [self.periodic_table.get_atomic_number(s) for s in self.species], np.int64
        )
        Z_values_2 = np.array(
            [self.periodic_table.get_atomic_number(s) for s in other_geometry.species],
            np.int64,
        )
        unique_Z = set(Z_values_1)

        # Allocate output arrays
        transformation_indices = np.zeros(len(self), np.int64)
        distances = np.zeros(len(self))
        atom2_indices = np.arange(len(other_geometry)) % len(self)

        # Loop over all types of atoms
        for Z in unique_Z:
            # Select all the coordinates that belong to that species
            select1 = Z_values_1 == Z
            select2 = Z_values_2 == Z
            # Calculate all distances between the geometries
            dist_matrix = scipy.spatial.distance_matrix(
                self.coords[select1, :], other_geometry.coords[select2, :]
            )
            # For each row (= atom in self with species Z) find the index of the other_geometry (with species Z) that is closest
            index_of_smallest_mismatch = np.argmin(dist_matrix, axis=1)
            transformation_indices[select1] = atom2_indices[select2][
                index_of_smallest_mismatch
            ]
            if get_distances:
                distances[select1] = [
                    dist_matrix[i, index_of_smallest_mismatch[i]]
                    for i in range(len(dist_matrix))
                ]

        if get_distances:
            return transformation_indices, distances
        else:
            return transformation_indices

    def is_equivalent(
        self, geom, tolerance=0.01, check_neightbouring_cells=False
    ) -> bool:
        """
        Check if this geometry is equivalent to another given geometry.
        The function checks that the same atoms sit on the same positions
        (but possibly in some permutation)

        :param check_neightbouring_cells: for periodic structures, recognizes two structures as equivalent, even if one\
         of them has its atoms distributed in different unit cells compared to the other. More complete, but slower.  """

        # Check that both geometries have same number of atoms
        # If not, they cannot be equivalent
        if geom.n_atoms != self.n_atoms:
            return False

        # check in neighbouring cells, to account for geometries 'broken' around the cell border
        if check_neightbouring_cells:
            if self.lattice_vectors is not None:
                geom = geom.get_periodic_replica(
                    (3, 3), explicit_replications=[[-1, 0, 1], [-1, 0, 1]]
                )
            else:
                print("Non periodic structure. Ignoring check_neighbouring_cells")

        n_atoms = self.n_atoms
        n_atoms_geom = geom.n_atoms
        # Check for each atom in coords1 that is has a matching atom in coords2
        for n1 in range(n_atoms):
            is_ok = False
            for n2 in range(n_atoms_geom):
                if self.species[n1] == geom.species[n2]:
                    d = np.linalg.norm(self.coords[n1, :] - geom.coords[n2, :])
                    if d < tolerance:
                        # Same atom and same position
                        is_ok = True
                        break

            if not is_ok:
                return False
        return True

    def is_equivalent_up_to_translation(
        self,
        geom,
        get_translation=False,
        tolerance=0.01,
        check_neighbouring_cells=False,
    ):
        """
            returns True if self can be transformed into geom by a translation
                (= without changing the geometry itself).

        Parameters
        ----------
        geom
        get_translation: additionally return the found translation
        tolerance
        check_neightbouring_cells: for periodic structures, recognizes two structures as equivalent, even if one\
         of them has its atoms distributed in different unit cells compared to the other. More complete, but slower.
        Returns
        -------

        """
        # shift both geometries to origin, get their relative translation.
        # Ignore center attribute (GeometryFile.center), if defined
        meanA = self.get_geometric_center(ignore_center_attribute=True)
        meanB = geom.get_geometric_center(ignore_center_attribute=True)
        translation = meanA - meanB
        self.center_coordinates(ignore_center_attribute=True)
        geom.center_coordinates(ignore_center_attribute=True)

        # check if they are equivalent (up to permutation)
        is_equivalent = self.is_equivalent(
            geom, tolerance, check_neightbouring_cells=check_neighbouring_cells
        )

        # undo shifting to origin
        self.coords += meanA
        geom.coords += meanB

        if get_translation:
            return is_equivalent, translation
        else:
            return is_equivalent

    ###############################################################################
    #                          Get Part of a Geometry                             #
    ###############################################################################
    def get_atoms_by_indices(self, atom_indices: npt.NDArray[np.int64]) -> "Geometry":
        """
        Return a geometry instance with the atoms listed in atom_indices

        Parameters
        ----------
        atom_indices : Union[int, np.array]
            List of integers, indices of those atoms which should be copied to
            new geometry

        Returns
        -------
        new_geom : Geometry

        """
        new_geom = self.__class__()
        new_geom.add_atoms(
            self.coords[atom_indices, :],
            [self.species[i] for i in atom_indices],
            constrain_relax=self.constrain_relax[atom_indices],
            initial_moment=[self.initial_moment[i] for i in atom_indices],
            initial_charge=[self.initial_charge[i] for i in atom_indices],
        )
        new_geom.lattice_vectors = self.lattice_vectors
        return new_geom

    def get_atoms_by_species(self, species) -> "Geometry":
        """
        get new geometry file with specific atom species
        """

        L = np.array(self.species) == species
        atom_indices = np.where(L)[0]
        return self.get_atoms_by_indices(atom_indices)

    def get_primitive_slab(self, surface, threshold=1e-6):
        """
        Generates a primitive slab unit cell with the z-direction perpendicular
        to the surface.

        Arguments:
        ----------
        surface : array_like
            miller indices, eg. (1,1,1)

        threshold : float
            numerical threshold for symmetry operations

        Returns:
        --------
        primitive_slab : Geometry
        """

        lattice, scaled_positions, atomic_numbers = spglib.standardize_cell(
            self.get_spglib_cell()
        )

        surface_vector = (
            surface[0] * lattice[0, :]
            + surface[1] * lattice[1, :]
            + surface[2] * lattice[2, :]
        )

        # TODO: this way of building lattice vectors parallel to the surface
        # is not ideal for certain surfaces
        dot_0 = surface_vector.dot(surface_vector)
        dot_1 = surface_vector.dot(lattice[0, :])
        dot_2 = surface_vector.dot(lattice[1, :])
        dot_3 = surface_vector.dot(lattice[2, :])

        if abs(dot_1) > threshold:
            frac = Fraction(dot_0 / dot_1).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v1 = m * surface_vector - n * lattice[0, :]
        else:
            v1 = lattice[0, :]

        if abs(dot_2) > threshold:
            frac = Fraction(dot_0 / dot_2).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v2 = m * surface_vector - n * lattice[1, :]
        else:
            v2 = lattice[1, :]

        if abs(dot_3) > threshold:
            frac = Fraction(dot_0 / dot_3).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v3 = m * surface_vector - n * lattice[2, :]
        else:
            v3 = lattice[2, :]

        surface_lattice = np.zeros((3, 3))
        surface_lattice[0, :] = surface_vector

        ind = 1
        for v in [v1, v2, v3]:
            if not np.linalg.norm(v) == 0:
                surface_lattice[ind, :] = v
                rank = np.linalg.matrix_rank(surface_lattice)

                if rank == ind + 1:
                    ind += 1
                    if ind == 3:
                        break

        # flip surface lattice such that surface normal becomes the z-axis
        surface_lattice = np.flip(surface_lattice, 0)

        slab = self.__class__()
        slab.lattice_vectors = surface_lattice

        # shellsize 100 such that the code does not run infinitely
        shellsize = 100
        for shell in range(0, shellsize):
            add_next_shell = False
            for h in range(-shell, shell + 1):
                for k in range(-shell, shell + 1):
                    for l in range(-shell, shell + 1):

                        if (abs(h) < shell) and (abs(k) < shell) and (abs(l) < shell):
                            continue

                        for new_species, coord in zip(atomic_numbers, scaled_positions):

                            new_coord = coord.dot(lattice) + np.array([h, k, l]).dot(
                                lattice
                            )
                            frac_new_coord = utils.get_fractional_coords(
                                new_coord, surface_lattice
                            )

                            L1 = np.sum(frac_new_coord >= 1 - threshold)
                            L2 = np.sum(frac_new_coord < -threshold)

                            if not L1 and not L2:
                                slab.add_atoms(
                                    [new_coord],
                                    [
                                        self.periodic_table.get_chemical_symbol(
                                            new_species
                                        )
                                    ],
                                )
                                add_next_shell = True

            if not shell == 0 and not add_next_shell:
                break

            if shell == 100:
                warnings.warn(
                    "<Geometry.get_primitive_slab> could not build a correct slab."
                )

        slab.align_lattice_vector_to_vector(np.array([0, 0, 1]), 2)
        slab.align_lattice_vector_to_vector(np.array([1, 0, 0]), 0)

        scaled_slab_lattice = np.array(slab.lattice_vectors)
        # break symmetry in z-direction
        scaled_slab_lattice[2, :] *= 2
        frac_coords = utils.get_fractional_coords(slab.coords, scaled_slab_lattice)
        species = [self.periodic_table.get_chemical_symbol(s) for s in slab.species]

        (
            primitive_slab_lattice,
            primitive_slab_scaled_positions,
            primitive_slab_atomic_numbers,
        ) = spglib.find_primitive(
            (scaled_slab_lattice, frac_coords, species), symprec=1e-5
        )

        primitive_slab_species = [
            self.periodic_table.get_chemical_symbol(s)
            for s in primitive_slab_atomic_numbers
        ]
        primitive_slab_coords = primitive_slab_scaled_positions.dot(
            primitive_slab_lattice
        )
        # replace lattice vector in z-direction
        primitive_slab_lattice[2, :] = slab.lattice_vectors[2, :]

        primitive_slab = self.__class__()
        primitive_slab.lattice_vectors = primitive_slab_lattice
        primitive_slab.add_atoms(primitive_slab_coords, primitive_slab_species)
        primitive_slab.map_to_first_unit_cell()

        # Sanity check: primitive_slab must be reducable to the standard unit cell
        check_lattice, _, _ = spglib.standardize_cell(primitive_slab.get_spglib_cell())

        assert np.allclose(
            check_lattice, lattice
        ), "<Geometry.get_primitive_slab> the slab that was constructed \
        could not be reduced to the original bulk unit cell. Something \
        must have gone wrong."

        return primitive_slab

    def get_slab(
        self,
        layers: int,
        surface: Union[npt.NDArray[np.int64], None] = None,
        threshold: float = 1e-6,
        surface_replica: Tuple[int, int] = (1, 1),
        vacuum_height: Union[float, None] = None,
        bool_shift_slab_to_bottom: bool = False,
    ) -> "Geometry":
        """
        Generates a slab.

        Parameters
        ----------
        layers : int
            Number of layers of the slab.
        surface : npt.NDArray[np.int64] | None, optional
            miller indices, eg. (1,1,1)
        threshold : float, optional
            numerical threshold for symmetry operations
        surface_replica : Tuple[int, int], optional
            Replications of surface. The default is (1,1).
        vacuum_height : float | None, optional
            DESCRIPTION. The default is None.
        bool_shift_slab_to_bottom : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        slab_new : Geometry
            New Geometry

        """
        if surface is not None:
            primitive_slab = self.get_primitive_slab(surface, threshold=threshold)
        else:
            primitive_slab = self

        slab_layers = primitive_slab.get_number_of_atom_layers()[1]

        replica = np.array([1, 1, int(np.ceil(layers / slab_layers))], dtype=np.int32)
        replica[:2] = surface_replica
        slab_new = primitive_slab.get_periodic_replica(tuple(replica))

        slab_new_layers = slab_new.get_atom_layers_indices()

        for atom_species in slab_new_layers:
            z_coords = list(slab_new_layers[atom_species])
            z_coords = sorted(z_coords)

            n_layers_to_remove = len(z_coords) - layers

            atom_indices_to_remove = []
            for ind in range(n_layers_to_remove):
                atom_indices_to_remove += slab_new_layers[atom_species][z_coords[ind]]

            slab_new.remove_atoms(np.array(atom_indices_to_remove, dtype=np.int32))

            if vacuum_height is not None:
                slab_new.set_vacuum_height(
                    vac_height=vacuum_height,
                    bool_shift_to_bottom=bool_shift_slab_to_bottom,
                )
            else:
                if bool_shift_slab_to_bottom:
                    self.shift_to_bottom()

        return slab_new

    def get_colliding_groups(self, distance_threshold=1e-2, check_3D=False):
        """
        Remove atoms that are too close too each other from the geometry file.
        This approach is useful if one maps back atoms into a different cell 
        and then needs to get rid of overlapping atoms

        Parameters
        ----------
        distance_threshold: float
            maximum distance between atoms below which they are counted as duplicates

        Returns
        -------
        """
        import networkx as nx
        
        # get all distances between all atoms

        z_period = [-1, 0, 1] if check_3D else [0]
        index_tuples = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in z_period:
                    curr_shift = (
                        i * self.lattice_vectors[0, :]
                        + j * self.lattice_vectors[1, :]
                        + k * self.lattice_vectors[2, :]
                    )

                    atom_distances = scipy.spatial.distance.cdist(
                        self.coords, self.coords + curr_shift
                    )
                    index_tuples += self._get_collision_indices(
                        atom_distances, distance_threshold
                    )
        if len(index_tuples) > 0:
            G = nx.Graph()
            G = nx.from_edgelist(
                itertools.chain.from_iterable(
                    itertools.pairwise(e) for e in index_tuples
                )
            )
            G.add_nodes_from(set.union(*map(set, index_tuples)))  # adding single items
            atoms_to_remove = list(nx.connected_components(G))
            return [sorted(list(s)) for s in atoms_to_remove]
        else:
            return []

    def get_cropped_geometry(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        auto_margin=False,
    ):
        """
        Returns a copy of the object to which self.crop has been applied

        """
        newgeom = deepcopy(self)
        newgeom.crop(xlim=xlim, ylim=ylim, zlim=zlim, auto_margin=auto_margin)
        return newgeom

    def get_substrate(self, primitive_substrate=None):
        substrate_indices = self.get_substrate_indices(
            primitive_substrate=primitive_substrate
        )
        return self.get_atoms_by_indices(substrate_indices)

    def get_adsorbates(self, primitive_substrate=None):
        adsorbate_indices = self.get_adsorbate_indices(
            primitive_substrate=primitive_substrate
        )
        return self.get_atoms_by_indices(adsorbate_indices)

    def get_periodic_replica(
        self,
        replications: tuple,
        lattice: Union[npt.NDArray[np.float64], None] = None,
        explicit_replications: Union[list, None] = None,
    ):
        """
        Return a new geometry file that is a periodic replica of the original file.
        repeats the geometry N-1 times in all given directions:
        (1,1,1) returns the original file

        Parameters
        ----------
        TODO Fix types
        replications : tuple or list
            number of replications for each dimension
        lattice : numpy array of shape [3x3]
            super-lattice vectors to use for periodic replication
            if lattice is None (default) the lattice vectors from the current
            geometry file are used.
        explicit_replications : iterable of iterables
             a way to explicitly define which replicas should be made.
             example: [[-1, 0, 1], [0, 1, 2, 3], [0]] will repeat 3 times in x (centered) and 4 times in y (not centered)

        Returns
        -------
        New geometry
        """
        # TODO implement geometry_parts the right way (whatever this is)
        if lattice is None:
            lattice = np.array(self.lattice_vectors)

        if explicit_replications:
            rep = explicit_replications
            lattice_multipliers = [np.max(t) - np.min(t) for t in explicit_replications]
        else:
            rep = [list(range(r)) for r in replications]
            lattice_multipliers = replications

        # old: n_replicas = np.abs(np.prod(replications))
        n_replicas = np.prod([len(i) for i in rep])
        n_atoms_new = n_replicas * self.n_atoms
        new_coords = np.zeros([n_atoms_new, 3])
        new_species = list(self.species) * n_replicas
        new_constrain = list(self.constrain_relax) * n_replicas

        insert_pos = 0
        # itertools.product = nested for loop
        for frac_offset in itertools.product(*rep):
            frac_shift = np.zeros([1, 3])
            frac_shift[0, : len(frac_offset)] = frac_offset
            offset = utils.get_cartesian_coords(frac_shift, lattice)
            new_coords[insert_pos : insert_pos + self.n_atoms, :] = self.coords + offset
            insert_pos += self.n_atoms

        new_geom = self.__class__()

        new_geom.add_atoms(new_coords, new_species, new_constrain)
        new_geom.lattice_vectors = lattice

        # save original lattice vector for visualization
        if hasattr(self, "original_lattice_vectors"):
            new_geom.original_lattice_vectors = copy.deepcopy(
                self.original_lattice_vectors
            )
        else:
            new_geom.original_lattice_vectors = copy.deepcopy(self.lattice_vectors)

        for i, r in enumerate(lattice_multipliers):
            new_geom.lattice_vectors[i, :] *= r
        return new_geom

    def get_split_into_molecules(self, threshold) -> list:
        """
        Splits a structure into individual molecules. Two distinct molecules
        A and B are defined as two sets of atoms, such that no atom in A is
        closer than the selected thresold to any atom of B

        """

        coords = deepcopy(self.coords)
        distances = distance_matrix(coords, coords)
        distances[distances <= threshold] = 1
        distances[distances > threshold] = 0

        def scan_line(line_index, matrix, already_scanned_lines_indices):
            already_scanned_lines_indices.append(line_index)
            line = matrix[line_index]
            links = np.nonzero(line)[0]
            links = [l for l in links if l not in already_scanned_lines_indices]
            return links, already_scanned_lines_indices

        molecules_indices_sets = []
        scanned_lines_indices = []
        indices_set = []
        # scan lines one by one, but skips those that have already been examined
        for i in range(len(distances)):
            if i in scanned_lines_indices:
                continue
            # add line to the present set
            indices_set.append(i)
            # get indices of the lines connected to the examined one
            links, scanned_lines_indices = scan_line(
                i, distances, scanned_lines_indices
            )
            indices_set += links
            # as long as new links are found, adds the new lines to the present set
            while len(links) > 0:
                new_links = []
                for l in links:
                    if l not in scanned_lines_indices:
                        new_links_part, scanned_lines_indices = scan_line(
                            l, distances, scanned_lines_indices
                        )
                        new_links += new_links_part
                links = set(new_links)
                indices_set += links
            # once no more links are found, stores the present set and starts a new one
            molecules_indices_sets.append(indices_set)
            indices_set = []

        molecules = []
        for molecule_indices in molecules_indices_sets:
            complementary_indices = [
                x for x in self.get_indices_of_all_atoms() if x not in molecule_indices
            ]
            g = deepcopy(self)
            g.remove_atoms(np.array(complementary_indices))
            molecules.append(g)

        return molecules

    def get_layers(self, layer_indices: list, threshold: float = 1e-2):
        """
        Get substrate layer by indices. The substrate is determined by
        default by the function self.get_substrate. For avoiding a faulty
        substrate determination it can be either given through indices or
        through the substrate geometry itself.

        Parameters
        ----------
        layer_indices : list
            List of indices of layers that shall be returned.
        substrate_indices : Union[None, list], optional
            List of indices of substrate atoms. The default is None.
        substrate : Union[None, Geometry()], optional
            Geometry of substrate. The default is None.
        threshold : float, optional
            DESCRIPTION. The default is 1e-2.
        primitive_substrate : Geometry, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        geometry_of_layer : Geometry
            Geometry of layer.

        """
        layers = self.get_atom_layers_indices_by_height(threshold=threshold)

        heights = list(layers.keys())
        heights = np.sort(heights)
        heights = heights[::-1]

        # if layer_indices is an empty list, keeps the substrate as a whole
        if not layer_indices:
            geometry_of_layer = self
        else:
            geometry_of_layer = self.__class__()
            geometry_of_layer.lattice_vectors = self.lattice_vectors
            for layer_ind in layer_indices:
                geometry_of_layer += self.get_atoms_by_indices(
                    layers[heights[layer_ind]]
                )

        return geometry_of_layer

    def get_substrate_layers(
        self,
        layer_indices: list,
        threshold: float = 1e-2,
        primitive_substrate=None,
        substrate_indices: Union[None, npt.NDArray[np.int64]] = None,
    ):
        """
        Get substrate layer by indices. The substrate is determined by
        default by the function self.get_substrate. For avoiding a faulty
        substrate determination it can be either given through indices or
        through the substrate geometry itself.

        Parameters
        ----------
        layer_indices : list
            List of indices of layers that shall be returned.
        threshold : float, optional
            DESCRIPTION. The default is 1e-2.
        primitive_substrate : TYPE, optional
            DESCRIPTION. The default is None.
        substrate_indices : Union[None, list], optional
            List of indices of substrate atoms. The default is None.

        Returns
        -------
        geometry_of_layer : TYPE
            Geometry of substrate layer.

        """
        if substrate_indices is not None:
            sub = self.get_atoms_by_indices(substrate_indices)
        else:
            sub = self.get_substrate(primitive_substrate=primitive_substrate)

        return sub.get_layers(layer_indices=layer_indices, threshold=threshold)

    ###############################################################################
    #                           Evaluation Functions                              #
    ###############################################################################
    def check_symmetry(
        self,
        tolerance: float,
        R: npt.NDArray[np.float64],
        t: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
        return_symmetrical: bool = False,
    ):
        """
        Returns True if the geometry is symmetric with respect to the
        transformation, and False if it is not. If the geometry is periodic,
        transformation can be tuple (rotation, translation) or np.array
        (only rotation), otherwise it can only be np.array

        Parameters
        ----------
        tolerance ; float
            Tolerance for checking symmetry.

        R : npt.NDArray[np.float64]
            Symmetry transformation agaist which geometry should be checked.

        t: npt.NDArray[np.float64]
            Translation vector

        return_symmetrical : bool
            Return the corresponding transformed geometry together with the
            result.

        Returns
        -------
        is_symmetric : bool
        symm_geometry : Geometry

        or

        is_symmetric : bool

        """
        # original structure with all atoms in the first unit cell
        self_1UC = copy.deepcopy(self)
        self_1UC.map_to_first_unit_cell()

        # original structure centered for reordering
        centered_geometry = copy.deepcopy(self)
        centered_geometry.center_coordinates()

        # apply transformation
        symm_geometry = copy.deepcopy(self)
        symm_geometry.transform_fractional(R, np.array([0, 0, 0]), self.lattice_vectors)
        symm_geometry.move_all_atoms_by_fractional_coords(t)

        # prevent problems if the center is very close to the edge
        center = utils.get_fractional_coords(
            symm_geometry.get_geometric_center(), symm_geometry.lattice_vectors
        )
        center[:2] %= 1.0
        if 1 - center[0] < 0.001:
            adjust = -(center[0] - 0.0001)
            symm_geometry.move_all_atoms_by_fractional_coords([adjust, 0, 0])
        if 1 - center[1] < 0.001:
            adjust = -(center[1] - 0.0001)
            symm_geometry.move_all_atoms_by_fractional_coords([0, adjust, 0])

        symm_geometry.map_center_of_atoms_to_first_unit_cell(
            lattice_vectors=self.lattice_vectors
        )

        # reorder atoms
        offset_symm = np.mean(symm_geometry.coords, axis=0)
        symm_geometry.center_coordinates()
        indices = centered_geometry.get_transformation_indices(symm_geometry)
        symm_geometry.reorder_atoms(np.array(indices))
        symm_geometry.move_all_atoms(offset_symm)

        # compare in first unit cell
        symm_geometry_1UC = copy.deepcopy(symm_geometry)
        symm_geometry_1UC.map_to_first_unit_cell()
        is_symmetric = symm_geometry_1UC.is_equivalent(
            self_1UC, tolerance=tolerance, check_neightbouring_cells=True
        )

        if return_symmetrical:
            return is_symmetric, symm_geometry
        else:
            return is_symmetric

    ###############################################################################
    #                                 Visualisation                               #
    ###############################################################################
    def visualise(
        self,
        axes=[0, 1],
        min_zorder=0,
        value_list=None,
        maxvalue=None,
        minvalue=None,
        cbar_label="",
        hide_axes=False,
        axis_labels=True,
        auto_limits=True,
        crop_ratio=None,
        brightness_modifier=None,
        print_lattice_vectors=False,
        alpha=1.0,
        linewidth=1,
        lattice_linewidth=None,
        lattice_color="k",
        atom_scale=1,
        highlight_inds=[],
        highlight_color="C2",
        color_list=None,
        cmap=None,
        ax=None,
        xlim=None,
        ylim=None,
        zlim=None,
        plot_method="circles",
        invert_colormap=False,
        edge_color=None,
        show_colorbar=True,
        reverse_sort_inds=False,
        axis_labels_format="/",
    ) -> None:
        """
        Generates at plt-plot of the current geometry. This function has a
        large number of options. In most cases the following examples will
        work well:

        - Just look at the geometry:
            geometry.visualise()

        - Turn aff axis:
            geometry.visualise(hide_axes=True)

        - Turn off axis and set limits:
            geometry.visualise(hide_axes=True,
                               xlim=(-10, 10))

        - If you want to look at the geoemtry in the xz-plane:
            geometry.visualise(axes=[0,2],
                               hide_axes=True,
                               xlim=(-10, 10))

        Visualise is one of the most useful things about geometry. Reading
        through this code you may think that it is very ugly and on to of that
        if has it's own imports. Still it is a great function and it must be
        part of geometry. If you think otherwise you are wrong.



        Parameter:
        ----------
        axes : list of 2 int elements
            axis that should be visualized, x=0, y=1, z=2
            By default, we look at the geometry from:
            the "top" (our viewpoint is at z = +infinity) when we visualize the xy plane;
            the "right" (our viewpoint is at x = +infinity) when we visualize the yz plane;
            the "front" (our viewpoint is at y = -infinity) when we visualize the xz plane.
            In order to visualize the geometry from the opposite viewpoints, one needs to use the reverse_sort_inds flag,
            and invert the axis when necessary (= set axis limits so that the first value is larger than the second value)

        min_zorder : int
            plotting layer

        value_list : None or list of length nr. atoms

        maxvalue : None

        cbar_label : str

        hide_axes : bool
            hide axis

        axis_labels : bool
            generates automatic axis labels

        auto_limits : bool
            set xlim, ylim automatically

        crop_ratio: float
            defines the ratio between xlim and ylim if auto_limits is enabled

        brightness_modifier : float or list/array with length equal to the number of atoms
            modifies the brightness of selected atoms. If brightness_modifier is a list/array, then
            brightness_modifier[i] sets the brightness for atom i, otherwise all atoms are set to the same brightness value.
            This is done by tweaking the 'lightness' value of said atoms' color in the HSL (hue-saturation-lightness) colorspace.
            Effect of brightness_modifier in detail:
              -1.0 <= brightness_modifier < 0.0  : darker color
              brightness_modifier == 0.0 or None : original color
              0.0 < brightness_modifier <= 1.0   :  brighter color

        print_lattice_vectors : bool
            display lattice vectors

        print_unit_cell : bool
            display original unit cell

        alpha : float between 0 and 1

        color_list : list or string
            choose colors for visualizing each atom. If only one color is passed, all atoms will have that color.

        plot_method: str
            circles: show filled circles for each atom
            wireframe: show molecular wireframe, standard settings: don't show H,

        reverse_sort_inds: bool
            if set to True, inverts the order at which atoms are visualized, allowing to visualize the geometry from the "bottom", from the "left" or from the "back".
            Example: if one wants to visualize the geometry from the "left" (= viewpoint at x=-infinity), atoms at lower x values should be visualized after atoms at high x values, and hide them.
            This is the opposite of the default behavior of this function, and can be achieved with reverse_sort_inds=True
            NOTE: in order to correctly visualize the structure from these non-default points of view, setting this flag to True is not sufficient: one must also invert the XY axes of the plot where needed.
            Example: when visualizing from the "left", atoms with negative y values should appear on the right side of the plot, and atoms with positive y values should appear on the left side of the plot.
            But if one simply sets reverse_sort_inds=True, atoms with negative y values will appear on the left side of the plot (because the x axis of the plot, the horizontal axis, goes from left to right!) and viceversa.
            This is equivalent to visualizing a mirrored image of the structure.
            To visualize the structure correctly, one should then set the x_limits of the plot with a first value smaller than the second value, so the x axis is inverted, and shows y-negative values on the left and viceversa.

        Returns
        -------
        None.

        """

        # default for lattice_linewidth (which is used to draw the lattice)
        if lattice_linewidth is None:
            lattice_linewidth = 2 * linewidth

        orig_inds = np.arange(self.n_atoms)
        remove_inds = []
        if xlim is not None:
            remove_x = self.get_cropping_indices(xlim=xlim, auto_margin=True)
            remove_inds += list(remove_x)
        if ylim is not None:
            remove_y = self.get_cropping_indices(ylim=ylim, auto_margin=True)
            remove_inds += list(remove_y)
        if zlim is not None:
            remove_z = self.get_cropping_indices(zlim=zlim, auto_margin=True)
            remove_inds += list(remove_z)

        crop_inds = list(set(remove_inds))

        if len(crop_inds) > 0:
            orig_inds = [orig_inds[i] for i in orig_inds if i not in crop_inds]
            cropped_geom = copy.deepcopy(self)
            cropped_geom.remove_atoms(np.array(crop_inds))
        else:
            cropped_geom = self

        if ax is None:
            ax = plt.gca()

        axnames = ["x", "y", "z"]
        orig_coords = cropped_geom.coords
        orig_species = cropped_geom.species
        #        orig_constrain = cropped_geom.constrain_relax

        # sorting along projecting dimension.
        # If sort_ind == 1, which means that we look at XZ, along the Y axis, in order to enforce our default behaviour
        # of looking at the XZ from "under" (== from the negative side of the Y axis), we need to flip the order
        # at which we see atoms, so we reverse the order of sort inds.
        # If the flat reverse_sort_inds is set to True, the order will be flipped again, to bring us out of our default.
        for i in range(3):
            if i not in axes:
                sort_ind = i

        inds = np.argsort(orig_coords[:, sort_ind])

        if sort_ind == 1:
            inds = inds[::-1]
        if reverse_sort_inds:
            inds = inds[::-1]

        orig_inds = [orig_inds[i] for i in inds]
        coords = orig_coords[inds]
        #        constrain = orig_constrain[inds]
        species = [orig_species[i] for i in inds]
        n_atoms = len(species)
        circlesize = [
            self.periodic_table.get_covalent_radius(s) * atom_scale for s in species
        ]

        # Specify atom colors by value list or default atom colors
        if value_list is None and color_list is None:
            colors = [self.periodic_table.get_species_colors(s) for s in species]
            colors = np.array(colors)
        elif color_list is not None:
            if len(color_list) == 1:
                colors = list(color_list) * len(self.species)
                colors = [mpl.colors.to_rgb(colors[i]) for i in inds]
            else:
                assert len(species) == len(color_list), (
                    "Color must be specified for all atoms or none!"
                    + f" Expected {len(species)}, but got {len(color_list)} values"
                )
                colors = [
                    mpl.colors.to_rgb(color_list[i]) for i in inds
                ]  # converting all types of color inputs to rgba here
            colors = np.array(colors)
        elif value_list is not None:
            assert (
                len(value_list) == self.n_atoms
            ), "Number of Values does not match number of atoms in geometry"
            values = [value_list[i] for i in orig_inds]

            if minvalue is not None:
                assert (
                    maxvalue is not None
                ), "Error! If minvalue is defined also maxvalue must be defined"

            if maxvalue is None and minvalue is None:
                maxvalue = np.max(np.abs(value_list))
                minvalue = -maxvalue

                if maxvalue < 1e-5:
                    maxvalue = 1e-5
                    print(
                        "Maxvalue for colormap not specified and smaller 1E-5, \nsetting it automatically to: ",
                        maxvalue,
                    )
                else:
                    print(
                        "Maxvalue for colormap not specified, \nsetting it automatically to: ",
                        maxvalue,
                    )

            if maxvalue is not None and minvalue is None:
                minvalue = -maxvalue

            if cmap is None:
                if invert_colormap:
                    cw = plt.get_cmap("coolwarm_r")
                else:
                    cw = plt.get_cmap("coolwarm")
            else:
                cw = plt.get_cmap(cmap)

            cNorm = matplotlib.colors.Normalize(vmin=minvalue, vmax=maxvalue)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cw)

            a = np.array([[minvalue, maxvalue]])
            img = plt.imshow(a, cmap=cw)
            img.set_visible(False)
            colors = []
            for v in values:
                colors.append(scalarMap.to_rgba(v))

        # make specified atoms brighter by adding color_offset to all rgb values

        if brightness_modifier is not None:

            # Check if brightness modifier is flat (i.e. a single value) or per atom (list of length n_atoms)
            if isinstance(brightness_modifier, float) or isinstance(
                brightness_modifier, int
            ):
                brightness_modifier = brightness_modifier * np.ones(n_atoms)

            else:
                # Sort list according to orig_inds (which is already cropped if necessary!)
                assert len(brightness_modifier) == self.n_atoms, (
                    "Argument 'brightness_modifier' must either be a "
                    "scalar (float or int) or a list with length equal "
                    "to the number of atoms"
                )
                brightness_modifier = [brightness_modifier[i] for i in orig_inds]

            assert (
                len(brightness_modifier) == n_atoms
            ), "Something went wrong while reformatting brightness_modifier!"
            for i in range(n_atoms):
                # TODO fix the pyright errors
                hls_color = np.array(
                    colorsys.rgb_to_hls(*colors[i, :])  # pyright:ignore
                )
                hls_color[1] += brightness_modifier[i] * (1 - hls_color[1])
                hls_color = np.clip(hls_color, 0, 1)
                colors[i, :] = colorsys.hls_to_rgb(*hls_color)  # pyright:ignore
        else:
            brightness_modifier = np.zeros(n_atoms)

        zorder = min_zorder

        if plot_method == "circles":
            for i in range(len(species)):
                if plot_method == "circles":
                    x1 = coords[i, axes[0]]
                    x2 = coords[i, axes[1]]
                    if orig_inds[i] not in highlight_inds:
                        if edge_color is None:
                            curr_edge_color = (
                                np.zeros(3) + brightness_modifier[i]
                                if brightness_modifier[i] > 0
                                else np.zeros(3)
                            )
                        else:
                            curr_edge_color = edge_color

                        ax.add_artist(
                            mpl.patches.Circle(
                                (x1, x2),
                                circlesize[i],
                                color=colors[i],
                                zorder=zorder,
                                linewidth=linewidth,
                                alpha=alpha,
                                ec=curr_edge_color,
                            )
                        )
                    else:
                        if edge_color is None:
                            curr_edge_color = highlight_color
                        else:
                            curr_edge_color = edge_color
                        ax.add_artist(
                            mpl.patches.Circle(
                                [x1, x2],
                                circlesize[i],
                                color=colors[i],
                                zorder=zorder,
                                linewidth=linewidth,
                                alpha=alpha,
                                ec=curr_edge_color,
                            )
                        )
                    zorder += 2

        elif plot_method == "wireframe":
            raise NotImplementedError("self.visualize_wireframe is not implemented")
            # self.visualizeWireframe(coords=coords, species=species,
            #                         linewidth=linewidth, min_zorder=min_zorder,
            #                         axes=axes, alpha=alpha, **kwargs)

        if print_lattice_vectors:
            ax.add_artist(
                plt.arrow(
                    0,
                    0,
                    *cropped_geom.lattice_vectors[0, axes],
                    zorder=zorder,
                    fc=lattice_color,
                    ec=lattice_color,
                    head_width=0.5,
                    head_length=1,
                )
            )
            ax.add_artist(
                plt.arrow(
                    0,
                    0,
                    *cropped_geom.lattice_vectors[1, axes],
                    zorder=zorder,
                    fc=lattice_color,
                    ec=lattice_color,
                    head_width=0.5,
                    head_length=1,
                )
            )

        # scale:
        xmax = np.max(coords[:, axes[0]]) + 2
        xmin = np.min(coords[:, axes[0]]) - 2
        ymax = np.max(coords[:, axes[1]]) + 2
        ymin = np.min(coords[:, axes[1]]) - 2

        if auto_limits:
            if print_lattice_vectors:
                xmin_lattice = np.min(cropped_geom.lattice_vectors[:, axes[0]]) - 1
                xmax_lattice = np.max(cropped_geom.lattice_vectors[:, axes[0]]) + 1
                ymin_lattice = np.min(cropped_geom.lattice_vectors[:, axes[1]]) - 1
                ymax_lattice = np.max(cropped_geom.lattice_vectors[:, axes[1]]) + 1

                ax_xmin = min(xmin, xmin_lattice)
                ax_xmax = max(xmax, xmax_lattice)
                ax_ymin = min(ymin, ymin_lattice)
                ax_ymax = max(ymax, ymax_lattice)

            else:
                ax_xmin, ax_xmax, ax_ymin, ax_ymax = xmin, xmax, ymin, ymax
                # allow for a fixed ratio when defining the limits
                # For this calculate the lengths and make the smaller limit longer so that the ratio fits

            if crop_ratio is not None:

                len_xlim = ax_xmax - ax_xmin
                len_ylim = ax_ymax - ax_ymin
                curr_crop_ratio = len_xlim / len_ylim

                if curr_crop_ratio > crop_ratio:
                    # make y limits larger
                    y_padding_fac = len_xlim / (crop_ratio * len_ylim)
                    y_padding = len_ylim * (y_padding_fac - 1)
                    ax_ymin -= y_padding / 2
                    ax_ymax += y_padding / 2

                else:
                    # make x limits larger
                    x_padding_fac = (crop_ratio * len_ylim) / len_xlim
                    x_padding = len_xlim * (x_padding_fac - 1)
                    ax_xmin -= x_padding / 2
                    ax_xmax += x_padding / 2

            # TODO: fix pyright linting errors
            ax.set_xlim([ax_xmin, ax_xmax])  # pyright:ignore
            ax.set_ylim([ax_ymin, ax_ymax])  # pyright:ignore

        # If limits are given, set them
        limits = [xlim, ylim, zlim]
        x1lim = limits[axes[0]]
        x2lim = limits[axes[1]]
        if x1lim is not None:
            ax.set_xlim(x1lim)
        if x2lim is not None:
            ax.set_ylim(x2lim)

        if axis_labels:
            if axis_labels_format == "/":
                ax.set_xlabel(r"{} / $\AA$".format(axnames[axes[0]]))
                ax.set_ylabel(r"{} / $\AA$".format(axnames[axes[1]]))
            elif axis_labels_format == "[]":
                ax.set_xlabel(r"{} [$\AA$]".format(axnames[axes[0]]))
                ax.set_ylabel(r"{} [$\AA$]".format(axnames[axes[1]]))

        if show_colorbar and (value_list is not None):
            cbar = plt.colorbar(ax=ax)
            cbar.ax.set_ylabel(cbar_label)

        ax.set_aspect("equal")
        plt.grid(False)
        if hide_axes:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    ###############################################################################
    #                            Protected Functions                              #
    ###############################################################################
    def _get_collision_indices(self, atom_distances, distance_threshold=1e-2):
        """helper function for removeDuplicateAtoms

        Parameters
        ----------
        distance_threshold: float
            maximum distance between atoms below which they are counted as duplicates

        Returns
        -------

        atoms_to_remove : list
            indices of atoms that can be removed due to collision
        """

        # get all distances between all atoms
        is_collision = atom_distances < distance_threshold

        colliding_atoms_dict = {}
        colliding_atoms_list = []

        # loop over all atoms
        for i in range(self.n_atoms):

            # evaluate only if atom is not already on the black list
            if i not in colliding_atoms_list:
                colliding_atoms_dict[i] = []
                # loop over all distances to other atoms, neglecting the diagonal (thus i+1)
                for j in range(i + 1, self.n_atoms):
                    if is_collision[i, j]:
                        colliding_atoms_dict[i].append(j)
                        colliding_atoms_list.append(j)

        return [
            (k, ind) for k, value in colliding_atoms_dict.items() for ind in list(value)
        ]


class AimsGeometry(Geometry):
    def parse(self, text):
        """
        Parses text from AIMS geometry file and sets all necessary parameters
        in AimsGeometry.

        Parameters
        ----------
        text : str
            line wise text file of AIMS geometry.
        """
        atom_lines = []
        is_fractional = False
        is_own_hessian = False
        self.trust_radius = False
        self.vacuum_level = None
        self.constrain_relax = []
        self.external_force = []
        self.calculate_friction = []
        self.multipoles = []
        self.homogeneous_field = None
        self.symmetry_params = None
        self.n_symmetry_params = None
        self.symmetry_LVs = None  # symmetry_LVs should have str values, not float, to allow for the inclusion of the parameters
        symmetry_LVs_lines = []
        self.symmetry_frac_coords = None  # symmetry_frac_coords should have str values, not float, to allow for the inclusion of the parameters
        symmetry_frac_lines = []
        lattice_vector_lines = []
        atom_line_ind = []
        hessian_lines = []
        text_lines = text.split("\n")

        for ind_line, line in enumerate(text_lines):
            line = line.strip()  # Remove leading and trailing space in line
            # Comment in input file
            if line.startswith("#"):
                if "DFT_ENERGY " in line:
                    self.DFT_energy = float(line.split()[2])
                elif "ADSORPTION_ENERGY " in line:
                    self.E_ads = float(line.split()[2])
                elif "ADSORPTION_ENERGY_UNRELAXED " in line:
                    self.E_ads_sp = float(line.split()[2])
                elif "CENTER" in line:
                    self.center = ast.literal_eval(" ".join(line.split()[2:]))
                # check if it is an own Hessian and not from a geometry optimization
                elif "own_hessian" in line:
                    is_own_hessian = True

                # PARTS defines parts of the geometry that can later on be treated separately.
                # intended for distinction between different molecules and substrate
                elif "PARTS" in line:
                    part_definition = ast.literal_eval(" ".join(line.split()[2:]))
                    if isinstance(part_definition, dict):
                        for k, v in part_definition.items():
                            self.geometry_part_descriptions.append(k)
                            self.geometry_parts.append(v)
                    elif isinstance(part_definition, list):
                        if isinstance(part_definition[0], list):
                            for part in part_definition:
                                self.geometry_part_descriptions.append("")
                                self.geometry_parts.append(part)
                        else:
                            self.geometry_parts.append(part)
                            self.geometry_part_descriptions.append("")

                else:
                    # Remove '#' at beginning of line, then remove any leading whitespace
                    line_comment = line[1:].lstrip()
                    # Finally add line comment to self.comment_lines
                    self.comment_lines.append(line_comment)

            else:
                # Extract all lines that define atoms, lattice vectors, multipoles or the Hessian matrix
                if "atom" in line:
                    atom_lines.append(line)
                    atom_line_ind.append(ind_line)
                if "lattice_vector" in line:
                    lattice_vector_lines.append(line)
                # c Check for fractional coordinates
                if "_frac" in line:
                    is_fractional = True
                if "hessian_block" in line:
                    hessian_lines.append(line)
                if "trust_radius" in line:
                    self.trust_radius = float(line.split()[-1])
                if "set_vacuum_level" in line:
                    self.vacuum_level = float(line.split()[1])
                if "multipole" in line:
                    multipole = [float(x) for x in list(line.split())[1:]]
                    assert len(multipole) == 5
                    self.multipoles.append(multipole)
                # extract lines concerning symmetry params
                if "symmetry_n_params" in line:
                    self.n_symmetry_params = [int(x) for x in list(line.split())[1:]]
                if "symmetry_params" in line:
                    self.symmetry_params = list(line.split())[1:]
                if "symmetry_lv" in line:
                    symmetry_LVs_lines.append(line)
                if "symmetry_frac" in line:
                    symmetry_frac_lines.append(line)
                if "homogeneous_field" in line:
                    self.homogeneous_field = np.asarray(
                        list(map(float, line.split()[1:4]))
                    )

        # c Read all constraints/ moments and spins
        for i, l in enumerate(atom_line_ind):
            constraints = [False, False, False]
            external_force = np.zeros(3)
            calculate_friction = False
            charge = 0.0
            moment = 0.0
            if i < len(atom_line_ind) - 1:
                last_line = atom_line_ind[i + 1]
            else:
                last_line = len(text_lines)
            for j in range(l, last_line):
                line = text_lines[j]
                if not line.startswith("#"):
                    if "initial_moment" in line:
                        moment = float(line.split()[1])
                    elif "initial_charge" in line:
                        charge = float(line.split()[1])
                    elif "constrain_relaxation" in line:
                        directions = line.split("constrain_relaxation")[1].lower()
                        if ".true." in directions:
                            constraints = [True, True, True]
                        if "x" in directions:
                            constraints[0] = True
                        if "y" in directions:
                            constraints[1] = True
                        if "z" in directions:
                            constraints[2] = True
                    elif "external_force" in line:
                        external_force[0] = float(line.split()[1])
                        external_force[1] = float(line.split()[2])
                        external_force[2] = float(line.split()[3])
                    elif "calculate_friction" in line:
                        if ".true." in line:
                            calculate_friction = True

            self.constrain_relax.append(constraints)
            self.external_force.append(external_force)
            self.calculate_friction.append(calculate_friction)
            self.initial_charge.append(charge)
            self.initial_moment.append(moment)

        # read the atom species and coordinates
        self.n_atoms = len(atom_lines)
        self.coords = np.zeros([self.n_atoms, 3])
        for i, l in enumerate(atom_lines):
            tokens = l.split()
            # self.species.append(tokens[-1])
            self.species.append(tokens[4])
            self.coords[i, :] = [float(x) for x in tokens[1:4]]

        # store symmetry_lv and symmetry_frac
        if len(symmetry_LVs_lines) != 0:
            self.symmetry_LVs = []
            if len(symmetry_LVs_lines) != 3:
                print(
                    "Warning: Number of symmetry_LVs is: "
                    + str(len(symmetry_LVs_lines))
                )
            for i, l in enumerate(symmetry_LVs_lines):
                l = l[11:]
                terms = [t.strip() for t in l.split(",")]
                self.symmetry_LVs.append(terms)
        if len(symmetry_frac_lines) != 0:
            self.symmetry_frac_coords = []
            for i, l in enumerate(symmetry_frac_lines):
                l = l[13:]
                terms = [t.strip() for t in l.split(",")]
                # self.species.append(tokens[-1])
                self.symmetry_frac_coords.append(terms)

        # read the hessian matrix if it is an own Hessian
        if is_own_hessian:
            # hessian has three coordinates for every atom
            self.hessian = np.zeros([self.n_atoms * 3, self.n_atoms * 3])
            for i, l in enumerate(hessian_lines):
                tokens = l.split()
                ind_1 = int(tokens[1])
                ind_2 = int(tokens[2])
                value_line = np.array([float(x) for x in tokens[3:12]])
                self.hessian[
                    (ind_1 - 1) * 3 : ind_1 * 3, (ind_2 - 1) * 3 : ind_2 * 3
                ] = value_line.reshape((3, 3))
            # self.hessian += np.tril(self.hessian.T, -1) # make symmetric hessian matrix

        if len(lattice_vector_lines) != 3 and len(lattice_vector_lines) != 0:
            print(
                "Warning: Number of lattice vectors is: "
                + str(len(lattice_vector_lines))
            )
        for i, l in enumerate(lattice_vector_lines):
            tokens = l.split()
            self.lattice_vectors[i, :] = [float(x) for x in tokens[1:4]]

        # convert to cartesian coordinates
        if is_fractional:
            self.coords = utils.get_cartesian_coords(self.coords, self.lattice_vectors)
            self.read_as_fractional_coords = True

        self.constrain_relax = np.array(self.constrain_relax)
        self.external_force = np.array(self.external_force)
        self.calculate_friction = np.array(self.calculate_friction)

        # update Part list and add all atoms that are not yet in the list
        if len(self.geometry_parts) > 0:
            already_indexed = list(itertools.chain.from_iterable(self.geometry_parts))
            if len(already_indexed) < self.n_atoms:
                additional_indices = [
                    i for i in range(self.n_atoms) if i not in already_indexed
                ]
                self.geometry_parts.append(additional_indices)
                self.geometry_part_descriptions.append("rest")

    def get_text(self, is_fractional=None):
        """
        If symmetry_params are to be used, the coordinates need to be fractional.
        So, if symmetry_params are found, is_fractional is overridden to true.
        """
        if is_fractional is None:

            if hasattr(self, "symmetry_params") and self.symmetry_params is not None:
                is_fractional = True
            else:
                is_fractional = False
        elif is_fractional is False:
            if hasattr(self, "symmetry_params") and self.symmetry_params is not None:
                warnings.warn(
                    "The symmetry parameters of your geometry will be lost. "
                    "To keep them set is_fractional to True"
                )

        text = ""
        for l in self.comment_lines:
            if l.startswith("#"):
                text += l + "\n"
            else:
                text += (
                    "# " + l.lstrip() + "\n"
                )  # str.lstrip() removes leading whitespace in comment line 'l'

        # If set, write 'center' dict ( see docstring of Geometry.__init__ ) to file
        if hasattr(self, "center") and isinstance(self.center, dict):
            center_string = "# CENTER " + str(self.center)
            text += center_string + "\n"

        if hasattr(self, "geometry_parts") and (len(self.geometry_parts) > 0):
            part_string = "# PARTS "
            part_dict = {}
            for part, name in zip(self.geometry_parts, self.geometry_part_descriptions):
                if not name == "rest":
                    if name not in part_dict:
                        part_dict[name] = part
                    else:
                        warnings.warn(
                            "Multiple equally named parts in file, renaming automatically!"
                        )
                        part_dict[name + "_1"] = part
            part_string += str(part_dict) + "\n"
            text += part_string

        if hasattr(self, "vacuum_level") and (self.vacuum_level is not None):
            text += "set_vacuum_level {: 15.10f}".format(self.vacuum_level) + "\n"

        # Lattice vector relaxation constraints
        constrain_vectors = np.zeros([3, 3], dtype=bool)
        # if is_2D:
        #    constrain_vectors[0, 2], constrain_vectors[1, 2], constrain_vectors[2] = True, True, 3*[True]

        # TODO: Some sort of custom lattice vector relaxation constraints parser

        if (self.lattice_vectors != 0).any():
            for i in range(3):
                line = "lattice_vector"
                for j in range(3):
                    line += "     {:.8f}".format(self.lattice_vectors[i, j])
                text += line + "\n"
                cr = "\tconstrain_relaxation "
                if constrain_vectors.any():
                    if constrain_vectors[i].all():
                        text += f"{cr}.true.\n"
                    else:
                        if constrain_vectors[i, 0]:
                            text += f"{cr}x\n"
                        if constrain_vectors[i, 1]:
                            text += f"{cr}y\n"
                        if constrain_vectors[i, 2]:
                            text += f"{cr}z\n"

        # write down the homogeneous field if any is present
        if self.homogeneous_field is not None:
            text += "homogeneous_field {} {} {}\n".format(*self.homogeneous_field)

        if is_fractional:
            coords = utils.get_fractional_coords(self.coords, self.lattice_vectors)
            line_start = "atom_frac"
        else:
            coords = self.coords
            line_start = "atom"

        for n in range(self.n_atoms):
            if self.species[n] == "Em":  # do not save "Emptium" atoms
                warnings.warn("Emptium atom was removed!!")
                continue
            line = line_start
            for j in range(3):
                line += "     {:.8f}".format(coords[n, j])
            line += " " + self.species[n]
            text += line + "\n"
            # backwards compatibilty for old-style constrain_relax
            if isinstance(self.constrain_relax[0], bool):
                if self.constrain_relax[n]:
                    text += "constrain_relaxation .true.\n"
            else:
                if all(self.constrain_relax[n]):
                    text += "constrain_relaxation .true.\n"
                else:
                    if self.constrain_relax[n][0]:
                        text += "constrain_relaxation x\n"
                    if self.constrain_relax[n][1]:
                        text += "constrain_relaxation y\n"
                    if self.constrain_relax[n][2]:
                        text += "constrain_relaxation z\n"
            if not len(self.initial_charge) == 0 and self.initial_charge[n] != 0.0:
                text += "initial_charge {: .6f}\n".format(self.initial_charge[n])
            if not len(self.initial_moment) == 0 and self.initial_moment[n] != 0.0:
                text += "initial_moment {: .6f}\n".format(self.initial_moment[n])
            if (
                hasattr(self, "external_force")
                and np.linalg.norm(self.external_force[n]) != 0.0
            ):
                text += "external_force {: .6f} {: .6f} {: .6f}\n".format(
                    self.external_force[n][0],
                    self.external_force[n][1],
                    self.external_force[n][2],
                )
            if hasattr(self, "calculate_friction") and self.calculate_friction[n]:
                text += "calculate_friction .true.\n"

        if hasattr(self, "hessian") and self.hessian is not None:
            text += "# own_hessian\n# This is a self calculated Hessian, not from a geometry optimization!\n"
            for i in range(self.n_atoms):
                for j in range(self.n_atoms):
                    s = "hessian_block  {} {}".format(i + 1, j + 1)
                    H_block = self.hessian[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)]
                    # max_diff = np.max(np.abs(H_block-H_block.T))
                    # print("Max diff in H: {:.3f}".format(max_diff))
                    for h in H_block.flatten():
                        s += "  {:.6f}".format(h)
                    text += s + "\n"

        # write down symmetry_params and related data
        if is_fractional:
            if self.symmetry_params is not None:
                l = "symmetry_params "
                for p in self.symmetry_params:
                    l += "{} ".format(p)
                l += "\n"
                text += "\n" + l
            if self.n_symmetry_params is not None:
                l = "symmetry_n_params "
                for n in self.n_symmetry_params:
                    l += "{} ".format(n)
                text += l + "\n"
                text += "\n"
            if self.symmetry_LVs is not None:
                for i in range(3):
                    line = "symmetry_lv     {}  ,  {}  ,  {}".format(
                        *self.symmetry_LVs[i]
                    )
                    text += line + "\n"
                text += "\n"
            if self.symmetry_frac_coords is not None:
                for c in self.symmetry_frac_coords:
                    line = "symmetry_frac     {}  ,  {}  ,  {}".format(*c)
                    text += line + "\n"
                text += "\n"

        # write down multipoles
        for m in self.multipoles:
            text += "multipole {}   {}   {}   {}   {}\n".format(*m)
        return text


class VaspGeometry(Geometry):
    def parse(self, text):
        """Read the VASP structure definition in the typical POSCAR format
            (also used by CONTCAR files, for example) from the file with the given filename.

        Return a dict containing the following information:
        systemname
            The name of the system as given in the first line of the POSCAR file.
        vecs
            The unit cell vector as a 3x3 numpy.array. vecs[0,:] is the first unit
            cell vector, vecs[:,0] are the x-coordinates of the three unit cell cevtors.
        scaling
            The scaling factor of the POSCAR as given in the second line. However, this
            information is not processed, it is up to the user to use this information
            to scale whatever needs to be scaled.
        coordinates
            The coordinates of all the atoms. Q[k,:] are the coordinates of the k-th atom
            (the index starts with 0, as usual). Q[:,0] are the x-coordinates of all the atoms.
            These coordinates are always given in Cartesian coordinates.
        elementtypes
            A list of as many entries as there are atoms. Gives the type specification for every
            atom (typically the atom name). elementtypes[k] is the species of the k-th atom.
        typenames
            The names of all the species. This list contains as many elements as there are species.
        numberofelements
            Gives the number of atoms per species. This list contains as many elements as there are species.
        elementid
            Gives the index (from 0 to the number of atoms-1) of the first atom of a certain
            species. This list contains as many elements as there are species.
        cartesian
            A logical value whether the coordinates were given in Cartesian form (True) or as direct
            coordinates (False).
        originalcoordinates
            The original coordinates as read from the POSCAR file. It has the same format as coordinates.
            For Cartesian coordinates (cartesian == True) this is identical to coordinates, for direct
            coordinates (cartesian == False) this contains the direct coordinates.
        selective
            True or False: whether selective dynamics is on.
        selectivevals
            Consists of as many rows as there are atoms, three colums: True if selective dynamics is on
            for this coordinate for the atom, else False. Only if selective is True.
        """
        lino = 0
        vecs = []
        scaling = 1.0
        typenames = []
        nelements = []
        cartesian = False
        selective = False
        selectivevals = []
        P = []
        fi = text.split("\n")

        for line in fi:
            lino += 1
            line = line.strip()

            if lino == 1:
                self.add_top_comment(line)
            if lino == 2:
                scaling = float(line)
                # RB: now the scaling should be taken account for below when the lattice vectors and coordinates
                # if scaling != 1.0:
                #    print("WARNING (readin_struct): universal scaling factor is not one. This is ignored.")

            if lino in (3, 4, 5):
                vecs.append(list(map(float, line.split())))
            if lino == 6:
                if line[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    lino += 1
                else:
                    typenames = line.split()
            if lino == 7:
                splitline = line.split()
                nelements = list(map(int, splitline))
                elementid = np.cumsum(np.array(nelements))
                self.n_atoms = elementid[-1]
            if lino == 8:
                if line[0] in ("S", "s"):
                    selective = True
                else:
                    lino += 1
            if lino == 9:
                if line[0] in ("K", "k", "C", "c"):  # cartesian coordinates
                    cartesian = True
            if lino >= 10:
                if lino >= 10 + self.n_atoms:
                    break
                P.append(list(map(float, line.split()[0:3])))

                if selective:
                    # TODO: experimental...
                    constraints = list(
                        map(lambda x: x in ("F", "f"), line.split()[3:6])
                    )
                    if len(constraints) != 3:
                        self.constrain_relax.append([False, False, False])
                    else:
                        self.constrain_relax.append(constraints)
                    selectivevals.append(constraints)
                else:
                    self.constrain_relax.append([False, False, False])
                    # TODO: write true value
                    self.initial_charge.append(0)
                    self.initial_moment.append(0)

                self.external_force = np.append(
                    self.external_force, np.atleast_2d(np.zeros(3)), axis=0
                )
                self.calculate_friction = np.append(
                    self.calculate_friction, np.array([False])
                )

        vecs = np.array(vecs)
        P = np.array(P)
        if not cartesian:
            Q = np.dot(P, vecs)
        else:
            Q = P
        if len(typenames) > 0:
            for k in range(Q.shape[0]):
                self.species.append(typenames[np.min(np.where(elementid > k)[0])])

        self.lattice_vectors = vecs
        self.coords = Q
        self.constrain_relax = np.array(self.constrain_relax)

        # RB: include the scaling. should work for both direct and cartesian settings
        self.lattice_vectors = vecs * scaling
        self.coords = Q * scaling

    def get_text(self, comment="POSCAR file written by Geometry.py"):
        comment = comment.replace("\n", " ")
        text = comment + "\n"
        text += "1\n"
        if (self.lattice_vectors != 0).any():
            for i in range(3):
                line = ""
                for j in range(3):
                    line += "     {:-4.8f}".format(self.lattice_vectors[i, j])
                text += line.strip() + "\n"

        all_species = sorted(
            list(set(self.species))
        )  # get unique species and sort alphabetically
        text += " ".join(all_species) + "\n"
        species_coords = {}
        n_of_species = {}
        # R.B. relax constraints
        relax_constraints = {}
        ## R.B. relax constraints end

        for species in all_species:
            is_right_species = np.array(
                [s == species for s in self.species], dtype=bool
            )
            curr_species_coords = self.coords[is_right_species, :]
            species_coords[species] = curr_species_coords
            n_of_species[species] = curr_species_coords.shape[0]

            # R.B. relax constraints
            curr_species_constrain_relax = self.constrain_relax[is_right_species, :]
            relax_constraints[species] = curr_species_constrain_relax
            ## R.B. relax constraints end

        # add number of atoms per species
        text += " ".join([str(n_of_species[s]) for s in all_species]) + "\n"

        # R.B. Write out selective dynamics so that the relaxation constraints are read
        text += "Selective dynamics" + "\n"

        text += "Cartesian" + "\n"

        for species in all_species:
            curr_coords = species_coords[species]
            n_atoms = n_of_species[species]

            ## R.B. relax constraints
            curr_relax_constr = relax_constraints[species]
            ## R.B. relax constraints end

            for n in range(n_atoms):
                line = ""
                for j in range(3):
                    if j > 0:
                        line += "    "
                    line += "{: 2.8f}".format(curr_coords[n, j])

                ## R.B. relax constraints
                for j in range(3):
                    if curr_relax_constr[n, j] is True:
                        line += "  " + "F"
                    elif curr_relax_constr[n, j] is False:
                        line += "  " + "T"
                ## R.B. relax constraints end

                text += line + "\n"

        return text


class XYZGeometry(Geometry):
    def parse(self, text):
        """
        Reads a .xyz file. Designed to work with .xyz files produced by Avogadro

        """

        # to use add_atoms we need to initialize coords the same as for Geometry
        self.n_atoms = 0
        self.coords = np.zeros([self.n_atoms, 3])

        read_natoms = None
        count_natoms = 0
        coords = []
        forces = []
        species = []
        fi = text.split("\n")

        # parse will assume first few lines are comments
        started_parsing_atoms = False

        for ind, line in enumerate(fi):
            if ind == 0:
                if len(line.split()) == 1:
                    read_natoms = int(line.split()[0])
                    continue

            # look for lattice vectors
            if 'Lattice' in line:
                #split_line = line.split('Lattice')[1]
                split_line = line.split('"')[1]
                
                lattice_parameters = re.findall("\d+\.\d+", split_line)

                if len(lattice_parameters) == 9:
                    lattice_parameters = np.array(lattice_parameters, dtype=np.float64)
                    self.lattice_vectors = np.reshape(lattice_parameters, (3,3))
            
            if 'energy' in line:
                split_line = line.split('energy')[1]
                
                energy = re.findall('-?[\d.]+(?:e-?\d+)?', split_line)
                
                if len(energy) > 0:
                    self.energy = np.float64(energy[0])
            
            # words = re.findall('[a-zA-Z]+', line)
            # numbers = re.findall('-?[\d.]+(?:e-?\d+)?', line)
            
            # n_words = len( words )
            # n_floats = len( numbers )
            
            split_line = line.split()
            
            n_words = 0
            n_floats = 0
            
            for l in split_line:
                n_words_new = len( re.findall('[a-zA-Z]+', l) )
                n_floats_new = len( re.findall('-?[\d.]+(?:e-?\d+)?', l) )
                
                if n_words_new == 1 and n_floats_new == 1:
                    n_floats += 1
                else:
                    n_words += n_words_new
                    n_floats += n_floats_new
            
            # first few lines may be comments or properties
            if not started_parsing_atoms:
                if n_words == 1 and (n_floats == 3 or n_floats == 6):
                    continue
                else:
                    started_parsing_atoms = True

            else:
                if split_line == []:
                    break
                else:
                    assert n_words == 1 and (n_floats == 3 or n_floats == 6), \
                        "Bad atoms specification: " + str(split_line) + f'{n_words} {n_floats}'
                
                # write atoms
                species.append(str(split_line[0]))
                coords.append( np.array(split_line[1:4], dtype=np.float64) )
                
                if n_floats == 6:
                    forces.append( np.array(split_line[4:], dtype=np.float64) )
                
                count_natoms += 1

        if not started_parsing_atoms:
            raise RuntimeError("Not atoms found in xyz file!")

        if read_natoms is not None:
            assert read_natoms == count_natoms, "Not all atoms found!"

        coords = np.asarray(coords)
        self.add_atoms(coords,species)
        
        if forces:
            forces = np.asarray(forces)
            self.forces = forces

    def get_text(self, comment="XYZ file written by Geometry.py"):
        text = str(self.n_atoms) + "\n"
        comment = comment.replace("\n", " ")
        text += comment + "\n"
        for index in range(self.n_atoms):
            element = self.species[index]
            x, y, z = self.coords[index]
            text += (
                "{}    {:-4.8f}    {:-4.8f}    {:-4.8f}".format(element, x, y, z) + "\n"
            )
        return text


class XSFGeometry(Geometry):
    def get_text(self):
        text = ""
        text += "CRYSTAL\n"
        text += "PRIMVEC\n"
        for i in range(3):
            line = ""
            for j in range(3):
                line += "    {:.8f}".format(self.lattice_vectors[i, j])
            text += line + "\n"
        text += "PRIMCOORD\n"
        # the 1 is mysterious but is needed for primcoord according to XSF docu
        text += str(self.n_atoms) + " 1\n"
        for i in range(self.n_atoms):
            if self.constrain_relax[i]:
                raise NotImplementedError(
                    "Constrained relaxation not supported for XSF output file"
                )
            line = str(self.periodic_table.get_atomic_number(self.species[i]))
            for j in range(3):
                line += "    {:.8f}".format(self.coords[i, j])
            text += line + "\n"
        return text


###############################################################################
#                            Auxiliary Functions                              #
###############################################################################
def get_file_format_from_ending(filename):
    if filename.endswith(".in"):
        return "aims"
    elif filename.endswith(".next_step"):
        return "aims"
    elif filename.endswith(".xsf"):
        return "xsf"
    elif filename.endswith(".molden"):
        return "molden"
    elif filename.endswith("POSCAR"):
        return "vasp"
    elif filename.endswith("CONTCAR"):
        return "vasp"
    elif filename.endswith(".xyz"):
        return "xyz"
    return None
