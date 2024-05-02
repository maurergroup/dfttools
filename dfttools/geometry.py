import itertools
from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from typing import Union
import copy
import warnings
import ast
import re

import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance
import networkx as nx

import os.path

import dfttools.utils.math_utils as utils
from dfttools.utils.periodic_table import PeriodicTable



def get_file_format_from_ending(filename):
    if filename.endswith('.in'):
        return 'aims'
    elif filename.endswith('.next_step'):
        return 'aims'
    elif filename.endswith('.xsf'):
        return 'xsf'
    elif filename.endswith('.molden'):
        return 'molden'
    elif filename.endswith('POSCAR'):
        return 'vasp'
    elif filename.endswith('CONTCAR'):
        return 'vasp'
    elif filename.endswith('.xyz'):
        return 'xyz'
    return None


def handle_unknown_file_format(filename, file_format=None):
    unknown_format = ''
    if file_format is None:
    # Determine file format from filename (extension for filename itsself)
        basename, ext = os.path.splitext(os.path.basename(filename))
        if ext == '':
            unknown_format = basename
        else:
            unknown_format = ext

    else:
        unknown_format = file_format

    raise NotImplementedError("Unknown file format {} for reading.".format(unknown_format))


class Geometry:
    """
    This class represents a geometry file for (in principle) any DFT code.
    In practice it has only been fully implemented for FHI-aims geometry.in files.
    """

    def __init__(self, filename=None, file_format=None):
        """
            
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
        self.geometry_parts = []                # list of lists: indices of each geometry part
        self.geometry_part_descriptions = []    # list of strings:  name of each geometry part
        self.symmetry_axes = None
        self.inversion_index = None
        self.vacuum_level = None
        self.multipoles = []
        self._homogeneous_field = None
        self.readAsFractionalCoords = False
        self.symmetry_params = None
        self.n_symmetry_params = None
        self.symmetry_LVs = None
        self.symmetry_frac_coords = None  # symmetry_frac_coords should have str values, not float, to include the parameters
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
    
    
    def convert_to_other_tpye(self, geometry_type):
        if geometry_type == 'aims':
            new_geometry = AimsGeometry()
        if geometry_type == 'vasp':
            new_geometry = VaspGeometry()
        if geometry_type == 'xyz':
            new_geometry = XYZGeometry()
        
        new_geometry.__dict__ = self.__dict__
        return new_geometry
    

    @property
    def homogenous_field(self):
        """Field is a numpy array (Ex, Ey, Ez) with the Field in V/A"""
        if not hasattr(self, "_homogeneous_field"):
            self._homogeneous_field = None
            
        return self._homogeneous_field


    @homogenous_field.setter
    def homogenous_field(self, E):
        """Field should be a numpy array (Ex, Ey, Ez) with the Field in V/A"""
        assert len(E) == 3, "Expected E-field components [Ex, Ey, Ez], but got " + str(E)
        self._homogeneous_field = np.asarray(E)


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
        if len(multipoles)==0:
            return
        if isinstance(multipoles[0],list):
            for x in multipoles:
                self.multipoles.append(x)
        # else: one list
        else:
            self.multipoles.append(multipoles)


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
        lines = comment_string.split('\n')
        for l in lines:
            if not l.startswith('#'):
                l = '# ' + l
            self.comment_lines.append(l)


###############################################################################
#                             OUTPUT PARSER                                   #
###############################################################################
    def save_to_file(self, filename, **kwargs):
        # TODO: add file format handeling
        text = self.get_text(**kwargs)

        # Enforce linux file ending, even if running on windows machine by
        # using binary mode
        with open(filename, 'w', newline='\n') as f:
            f.write(text)


    def get_text(self, **kwargs):
        raise NotImplementedError


###############################################################################
#                         Data exchange with ASE                              #
###############################################################################
    def get_from_ase_atoms_object(self,
                                  atoms,
                                  scaled=False,
                                  info_str=None,
                                  wrap=False):
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
        from ase.constraints import FixAtoms
        if isinstance(atoms, (list, tuple)):
            if len(atoms) > 1:
                raise RuntimeError(
                    "Don't know how to save more than "
                    "one image to FHI-aims input"
                )
            else:
                atoms = atoms[0]
    
        if atoms.get_pbc().any():
            self.lattice_vectors = np.array(atoms.get_cell())
    
        fix_cart = np.zeros([len(atoms), 3])
        if atoms.constraints:
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    fix_cart[constr.index] = [True, True, True]
        constrain_relax=fix_cart
    
        coords=[]
        species_list=[]
        for i, atom in enumerate(atoms):
            species = atom.symbol
            if isinstance(species, int):
                species = PERIODIC_TABLE[species]
            species_list.append(species)
            coords.append(atom.position)
        coords = np.array(coords)
        self.add_atoms(coords, species_list, constrain_relax)
    
    
    def get_as_ase(self):
        import ase
        """
        Convert geometry file to ASE object
        
        """
        #atoms_string = ""
        atom_coords = []
        atom_numbers = []
        for i in range(self.n_atoms):
            # Do not export 'emptium" atoms
            if self.species[i] != 'Em':
                atom_coords.append(self.coords[i,:])
                atom_numbers.append(self.periodic_table.get_atomic_number(self.species[i]))
                
        ase_system = ase.Atoms(numbers=atom_numbers, positions=atom_coords)
        ase_system.cell = self.lattice_vectors
        
        if not np.sum(self.lattice_vectors) == 0.0:
            ase_system.pbc = [1, 1, 1]
                
        return ase_system


###############################################################################
#                    Adding and removing atoms (in place)                     #
###############################################################################
    def add_atoms(self,
                  cartesian_coords,
                  species,
                  constrain_relax=None,
                  initial_moment=None,
                  initial_charge=None,
                  external_force=None,
                  calculate_friction=None) -> None:
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
            constrain_relax = np.zeros([len(species),3], bool)
        if external_force is None or len(external_force) == 0:
            external_force = np.zeros([len(species),3], np.float64)
        if calculate_friction is None:
            calculate_friction = np.array([False]*len(species))
        if initial_moment is None:
            initial_moment = [0.0]*len(species)
        if initial_charge is None:
            initial_charge = [0.0]*len(species)
        # TODO: this should not be necessary as self.coords should always be a np.array
        if not hasattr(self, 'coords') or self.coords is None:
            assert isinstance(cartesian_coords, np.ndarray)
            self.coords = cartesian_coords
        else:
            self.coords = np.concatenate((self.coords, cartesian_coords), axis=0)
        self.species += species
        self.n_atoms = self.coords.shape[0]
        
        self.constrain_relax = np.concatenate((self.constrain_relax, constrain_relax), axis=0)
        self.external_force = np.concatenate((self.external_force, external_force), axis=0)
        self.calculate_friction = np.concatenate((self.calculate_friction, calculate_friction))
        self.initial_moment  += initial_moment
        self.initial_charge  += initial_charge
        

    def add_geometry(self, geometry):
        """Adds full geometry to initial GeometryFile."""
        
        #check parts: (needs to be done before adding atoms to self)
        if hasattr(self,'geometry_parts') and hasattr(geometry,'geometry_parts'):
            for part,name in zip(geometry.geometry_parts, geometry.geometry_part_descriptions):
                if len(part) > 0:
                    self.geometry_parts.append([i + self.n_atoms for i in part])
                    self.geometry_part_descriptions.append(name)
                    
        # some lines of code in order to preserve backwards compatibility
        if not hasattr(geometry, "external_force"):
            geometry.external_force = np.zeros([0, 3], np.float64)
        self.add_atoms(geometry.coords,
                      geometry.species,
                      constrain_relax=geometry.constrain_relax,
                      initial_moment=geometry.initial_moment,
                      initial_charge=geometry.initial_charge,
                      external_force=geometry.external_force,
                      calculate_friction=geometry.calculate_friction)
        
        #check lattice vectors:
        # g has lattice and self not:
        if not np.any(self.lattice_vectors) and np.any(geometry.lattice_vectors):
            self.lattice_vectors = np.copy(geometry.lattice_vectors)
            
        # both have lattice vectors:
        elif np.any(self.lattice_vectors) and np.any(geometry.lattice_vectors):
            warnings.warn('Caution: The lattice vectors of the first file will be used!')
    
        #add multipoles
        self.add_multipoles(geometry.multipoles)
        
        #check center:
        # g has center and self not:
        if hasattr(self, 'center') and hasattr(geometry, 'center'):
            if self.center is None and geometry.center is not None:
                self.center = geometry.center.copy()
            # both have a center:
            elif self.center is not None and geometry.center is not None:
                warnings.warn('Caution: The center of the first file will be used!')


    def remove_atoms(self, atom_inds: np.array) -> None:
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
        if hasattr(self,'geometry_parts') and len(self.geometry_parts) > 0:
            # (AE): added "len(self.geometry_parts) > 0" to suppress this frequent warning when it is supposely not relevant (?)
            warnings.warn('CAUTION: geometry_parts indices are not updated after atom deletion!!\n \
                           You are welcome to implement this!!')
        if atom_inds is None:
            atom_inds = range(len(self))
        mask = np.ones(len(self.species), dtype=bool)
        mask[atom_inds] = False
        
        self.species = list(np.array(self.species)[mask])
        self.constrain_relax = self.constrain_relax[mask,:]
        self.external_force = self.external_force[mask,:]
        self.calculate_friction = self.calculate_friction[mask]
        self.coords = self.coords[mask,:]
        self.n_atoms = len(self.constrain_relax)


        if hasattr(self, 'hessian') and self.hessian is not None:
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
        self.removeAtoms(atom_inds)
        

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
        self.calculate_friction= self.calculate_friction[:n_atoms]
        self.coords = self.coords[:n_atoms,:]
        self.n_atoms = n_atoms


###############################################################################
#                         Transformations (in place)                          #
###############################################################################
    def map_to_first_unit_cell(self, lattice_vectors=None, dimensions=np.array(range(3))):
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
        
        assert not np.allclose(lattice_vectors, np.zeros([3,3])), \
            'Lattice vector must be defined in Geometry or given as function parameter'
        
        frac_coords = utils.get_fractional_coords(self.coords, lattice_vectors)
        frac_coords[:,dimensions] = frac_coords[:,dimensions] % 1 # modulo 1 maps all coordinates to first unit cell
        new_coords = utils.get_cartesian_coords(frac_coords, lattice_vectors)
        self.coords = new_coords


    def map_center_of_atoms_to_first_unit_cell(self, lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors
        
        assert not np.allclose(lattice_vectors, np.zeros([3,3])), \
            'Lattice vector must be defined in Geometry or given as function parameter'
        
        offset = self.getGeometricCenter()
        frac_offset = utils.get_fractional_coords(offset, lattice_vectors)
        frac_offset = np.floor(frac_offset)
        self.moveByFractionalCoords(-frac_offset, lattice_vectors)
    
    
    def center_coordinates(self, ignore_center_attribute=False, dimensions=np.array(range(3))):
        """
        Shift the coordinates of a geometry such that the "center of mass" or specified center lies at (0,0,0)
        :param bool ignore_center_attribute: Switch usage of *center* attribute off/on.
        :returns: Offset by which this geometry was shifted
        """
        offset = self.getGeometricCenter(ignore_center_attribute=ignore_center_attribute)[dimensions]
        self.coords[:,dimensions] -= offset
        return offset
    

    def move_all_atoms(self, shift):
        """
        Translates the whole geometry by vector 'shift'
        """
        self.coords += shift
        

    def move_all_atoms_by_fractional_coords(self,
                                            frac_shift,
                                            lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors
            
        self.coords += utils.get_cartesian_coords(frac_shift, lattice_vectors)
    
    
    def move_adsorbates(self, shift, primitive_substrate=None):
        """
        shifts the adsorbates in Cartesian coordinates
        """
        adsorbates = self.getAdsorbates(primitive_substrate=primitive_substrate)
        adsorbates.coords += shift
        
        self.removeAdsorbates(primitive_substrate=primitive_substrate)
        self += adsorbates
        
    
    def rotate_lattice_around_z_axis(self, angle_in_degree):
        """
        Rotates lattice around z axis

        Parameters
        ----------
        angle_in_degree : float
            angle of rotation

        Returns
        -------
        None.
        """
        R = utils.get_rotation_matrix_around_z_axis(angle_in_degree * np.pi / 180)
        self.lattice_vectors = np.dot(self.lattice_vectors, R)

    
    def rotate_around_z_axis(self, angle_in_degree, center=None, indices=None):
        """Rotates structure COUNTERCLOCKWISE around a point defined by <center>.

        If center == None, the geometric center of the structure (as defined by self.getGeometricCenter())
        is used as the pivot for the rotation.
        """
        # ??? <aj, 25.2.2020> not sure if center should be the geometric center or [0,0,0], both
        # have their pros and cons, former version was [0,0,0], new version is geometric center
        if indices is None:
            indices = np.arange(self.n_atoms)
        if center is None:
            center = self.getGeometricCenter(indices=indices)

        R = utils.get_rotation_matrix_around_z_axis(angle_in_degree * np.pi / 180, get_3x3_matrix=True)
        temp_coords = copy.deepcopy(self.coords[indices])
        temp_coords -= center
        temp_coords = np.dot(temp_coords,R)
        temp_coords += center
        self.coords[indices] = temp_coords


    def mirror_through_plane(self, normal_vector: np.array) -> None:
        """
        Mirrors the geometry through the plane defined by the normal vector.
        """
        mirror_matrix = utils.get_mirror_matrix(normal_vector=normal_vector)
        self.transform(mirror_matrix)
        

    def align_into_XY_plane(self, atom_indices):
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
    
    
    def align_with_Z_vector(self, new_z_vec: np.array) -> None:
        """
        Transforms the coordinate system of the geometry file to a new z-vector
        calculates rotation martrix for coordinate transformation to new z-vector
        and uses it to transform coordinates of geometry object 
        
        Parameters
        ----------
        new_z_vec : np.array
            The vector to align with the z-axis.

        Returns
        -------
        None

        """
        #get old_positions
        old_positions= self.coords
        
        # normalize new_z_vec
        new_z_vec= new_z_vec/np.linalg.norm(new_z_vec)
        
        
        # Check if the desired vector is antiparallel to the z-axis
        if np.allclose(new_z_vec, -np.array([0, 0, 1])):
            rotation_matrix = np.diag([-1, -1, 1])  # Antiparallel case
        else:
            # Calculate the rotation matrix
            z_axis = np.array([0, 0, 1])
            cross_product = np.cross(new_z_vec, z_axis)
            dot_product = np.dot(new_z_vec, z_axis)
            skew_symmetric_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                            [cross_product[2], 0, -cross_product[0]],
                                            [-cross_product[1], cross_product[0], 0]])
            rotation_matrix = np.eye(3) + skew_symmetric_matrix + np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * (1 - dot_product) / (np.linalg.norm(cross_product) ** 2)

        # Apply the rotation to all positions
        rotated_positions = np.dot(old_positions, rotation_matrix.T)
        
        self.coords= rotated_positions
        
        
    def align_lattice_vector_to_vector(self, vector, lattice_vector_index):
        """
        Align a lattice vector to a given axis
        
        vector : array
            vector for alignment
        
        lattice_vector_index : int
            index of the lattice vector that should be aligned
            
        """
        lattice_vector_normed = \
            self.lattice_vectors[lattice_vector_index] / \
                np.linalg.norm(self.lattice_vectors[lattice_vector_index])

        vector_normed = vector / np.linalg.norm(vector)
                
        R = utils.get_rotation_matrix(vector_normed, lattice_vector_normed)

        self.lattice_vectors = np.dot(self.lattice_vectors, R)
        self.coords = np.dot(self.coords, R)
        
    
    def align_main_axis_along_XYZ(self) -> None:
        """
        Align coordinates of rodlike molecules along specified axis
        
        """
        vals, vecs = self.get_main_axes()
        R = np.linalg.inv(vecs.T)
        self.coords = np.dot(self.coords,R)


    def transform(self,
                  R: np.array,
                  t: np.array=np.array([0,0,0]),
                  rotation_center: Union[np.array, None]=None,
                  atom_indices: Union[np.array, None]=None) -> None:
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
            temp_coords = np.dot(self.coords[atom_indices,:], R) + t
            self.coords[atom_indices, :] = temp_coords
        else:
            temp_coords = copy.deepcopy(self.coords[atom_indices,:])
            temp_coords -= rotation_center
            temp_coords = np.dot(temp_coords,R) + t
            temp_coords += rotation_center
            self.coords[atom_indices,:] = temp_coords


    def transform_lattice(self, R: np.array, t: np.array=np.array([0,0,0])) -> None:
        """
        Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]
        Notice that this works in cartesian coordinates.
        Use transformFractional if you got your R and t from getSymmetries

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


    def transform_lattice_fractional(self,R,t,lattice):
        """Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]"""
        coords_frac = utils.get_fractional_coords(self.lattice_vectors, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1,3])
        self.lattice_vectors = utils.get_cartesian_coords(coords_frac, lattice)


    def swap_lattice_vectors(self, axis_1=0, axis_2=1):
        """
        Can be used to interchange two lattice vectors
        Attention! Other values - for instance k_grid - will stay unchanged!!
        :param axis_1 integer [0,1,2]
        :param axis_2 integer [0,1,2]     axis_1 !=axis_2
        :return:
        
        """
        self.lattice_vectors[[axis_1, axis_2], :] = self.lattice_vectors[[axis_2, axis_1], :]
        self.coords[[axis_1, axis_2], :] = self.coords[[axis_2, axis_1], :]


    def transform_fractional(self, R, t, lattice=None):
        """
        Transforms the coordinates by rotation and translation, where R,t are
        given in fractional coordinates
        The transformation is applied as c_new[3x1] = R[3x3] * c_old[3x1] + t[3x1]
        
        """
        if lattice is None:
            lattice=self.lattice_vectors
        coords_frac = utils.get_fractional_coords(self.coords, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1,3])
        self.coords = utils.get_cartesian_coords(coords_frac, lattice)
    
    
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
            self.centerCoordinates()
        temp_coords = copy.deepcopy(self.coords) # this corresponds to the unity matrix symmetry operation
        for R in symmetry_operations:
            new_geom = copy.deepcopy(self)
            new_geom.transform(R)
            new_geom.reorder_atoms(self.getTransformationIndices(new_geom))
            temp_coords += new_geom.coords
        self.coords = temp_coords / (len(symmetry_operations) + 1) + offset
        

    def symmetrize_periodic(self, symmetries):
        """
        Reliably symmetrizes a periodic structure on a set of symmetries, as received from getSymmetries()
        Differently from symmetrize(), symmetries MUST include the identity too
        NOTE: I have not tested this function thoroughly; use it with caution. (fc 13.05.2020)
        
        """
        Rs = symmetries['rotations']
        ts = symmetries['translations']
        transformed_geoms = []

        # bring all atoms to first UC. Provides a univocal distribution of the atoms in the UCs.
        self.moveToFirstUnitCell(coords=np.array([0,1])) # only move to 1UC on XY

        for i,R in enumerate(Rs):
            t = ts[i]
            new_geom = copy.deepcopy(self)
            # store centered reference which will be used to reorder atoms later
            centered = copy.deepcopy(new_geom)
            centered.centerXYCoordinates()
            # rotate
            new_geom.transformFractional(R,np.array([0,0,0]),self.lattice_vectors)
            # translate
            new_geom.moveByFractionalCoords(t)
            # bring all atoms to first UC. Provides a univocal distribution of the atoms in the UCs.
            new_geom.moveToFirstUnitCell(lattice=self.lattice_vectors,coords=np.array([0,1]))
            # store offset and center
            offset2 = np.mean(new_geom.coords, axis=0)
            new_geom.centerXYCoordinates()
            # reoreder atoms with the centered reference
            indices = centered.getTransformationIndices(new_geom,periodic_2D=True)
            new_geom.reorder_atoms(indices)
            # move back to pre-centered position
            new_geom.move(offset2)
            transformed_geoms.append(new_geom)
        # we have all the structures, including the original, in transformed geoms. It can be nice for visualization.
        # average the coordinates
        transformed_coords = np.stack([g.coords for g in transformed_geoms])
        symm_coords = np.mean(transformed_coords,axis=0)
        self.coords = symm_coords
        

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
            offset = self.getGeometricCenter() # Attribute center should be used if it exists
            self.coords -= offset

            for other_geom in other_geometries:
                geom = copy.deepcopy(other_geom)
                # center all other geometries to remove offset
                geom.centerCoordinates()
                # all geometries have to be ordered like first geometry in order to sum them
                geom.reorder_atoms(self.getTransformationIndices(geom))
                self.coords += geom.coords
            self.coords /= (len(other_geometries) + 1)    # +1 for this geometry itself
            self.coords += offset


    def reorder_atoms(self, inds: np.array) -> None:
        """
        eorders Atoms with index list

        Parameters
        ----------
        inds : np.array
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


    def set_multipoles_charge(self, charge: np.array) -> None:
        """
        Sets the charge of all multipoles
        :param charge: list or float or int
        :return:
        """
        if isinstance(charge,list):
            assert len(charge) == len(self.multipoles)
            for i, m in enumerate(self.multipoles):
                m[4] = charge[i]
        else:
            for i, m in enumerate(self.multipoles):
                m[4] = charge


    def free_all_constraints(self) -> None:
        """
        Frees all constraints.

        Returns
        -------
        None

        """
        self.constrain_relax=np.zeros([len(self.species), 3], bool)
        

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


    def get_reassembled_molecule(self, threshold: float=2.0):
        
        geom_replica = self.getPeriodicReplica((1,1,1), explicit_replications=([-1,0,1],[-1,0,1],[-1,0,1]))
        
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
                            clusters[ind] = set( list(indices) + list(pair) )
                            new_cluster = True
                            in_culster = True
                            break
                    
                if not in_culster:
                    clusters.append( set(pair) )
            
            pairs = copy.deepcopy(clusters)
                
        for index_array in pairs:
            if len(index_array) == len(self):
                final_geom = geom_replica.getAtomsByIndices( np.sort( np.array(list(index_array), dtype=np.int32) ) )
                final_geom.lattice_vectors = self.lattice_vectors
                final_geom.mapCenterOfAtomsClosestToOrigin()
                    
                return final_geom
        
        warnings.warn('Geometry.getReassembledMolecule could not reassemble molecule. Returning original Geometry.')
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

        assert hasattr(self,'lattice_vectors'), "This function only works for geometries with a Unit Cell"

        if isinstance(scaling_factor,float):
            scaling_factors = [scaling_factor**(1/3),]*3
        else:
            assert len(scaling_factor)==3
            scaling_factors = scaling_factor
        scaled_geom = deepcopy(self)
        lattice_vectors = deepcopy(self.lattice_vectors)
        lattice_vectors[0]*=scaling_factors[0]
        lattice_vectors[1]*=scaling_factors[1]
        lattice_vectors[2]*=scaling_factors[2]

        new_coords = utils.get_cartesian_coords(self.get_fractional_coords(), lattice_vectors)
        scaled_geom.lattice_vectors = lattice_vectors
        scaled_geom.coords = new_coords

        return scaled_geom

    def get_fractional_coords(self, lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors
            
        assert not np.allclose(lattice_vectors, np.zeros([3,3])), \
            'Lattice vector must be defined in Geometry or given as function parameter'
        
        fractional_coords = np.linalg.solve(lattice_vectors.T, self.coords.T)
        return fractional_coords.T


    def get_fractional_lattice_vectors(self, lattice_vectors=None):
        """
        calculate the fractional representation of lattice vectors of the
        geometry file in another basis.
        Useful to calculate epitaxy matrices
        """

        fractional_coords = np.linalg.solve(lattice_vectors.T, self.lattice_vectors.T)
        return fractional_coords.T
    
    
    def get_reciprocal_lattice(self) -> np.array:
        """
        Calculate the reciprocal lattice of the Geometry lattice_vectors in standard form
        For convention see en.wikipedia.org/wiki/Reciprocal_lattice
        Returns: 
            recip_lattice: np.array(3x3)
                rowwise reciprocal lattice vectors
        """

        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]

        volume = np.cross(a1, a2).dot(a3)

        b1 = np.cross(a2, a3)
        b2 = np.cross(a3, a1)
        b3 = np.cross(a1, a2)

        recip_lattice = np.array([b1, b2, b3])*2*np.pi/volume
        return recip_lattice
    

    def get_main_axes(self, weights: str='unity') -> (np.array, np.array):
        """
        Get main axes and eigenvalues of a molecule
        https://de.wikipedia.org/wiki/Tr%C3%A4gheitstensor


        weights: 

        Parameters
        ----------
        weights : str, optional
            Specifies how the atoms are weighted.
            "unity": all same weight
            "Z": weighted by atomic number.
            The default is 'unity'.

        Returns
        -------
        vals : np.array
            DESCRIPTION.
        vecs : np.array
            DESCRIPTION.

        """
        if weights == 'unity':
            weights = np.ones(self.n_atoms)
        elif weights == 'Z':
            weights = np.array([ut.PERIODIC_TABLE[s] for s in self.species])

        coords = self.coords - np.mean(self.coords,axis=0)
        diag_entry = np.sum(np.sum(coords**2, axis=1) * weights)
        I = np.eye(3) * diag_entry
        for i in range(self.n_atoms):
            I -= weights[i] * np.outer(coords[i,:],coords[i,:])
        
        vals,vecs = scipy.linalg.eigh(I)
        sort_ind = np.argsort(vals)
        
        return vals[sort_ind], vecs[:,sort_ind]


    def get_distance_between_all_atoms(self) -> np.array:
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
            assert isinstance(species, list), 'species must be a list of species identifiers or None if all atoms should be probed'
            species_to_consider=species

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
            distances = all_distances[index,indices_to_consider]
            distance_indices = np.argsort(distances)
            closest_atoms = indices_to_consider[distance_indices]
            if len(closest_atoms) > n_closest:
                closest_atoms = closest_atoms[:n_closest]

            closest_atoms_list.append(closest_atoms.tolist())

        if return_single_list: #can only be true if only a single index was specified
            return closest_atoms_list[0]
        else:
            return closest_atoms_list


    def get_distance_between_two_atoms(self, atom_indices: list) -> float:
        """Get the distance between two atoms in the current Geometry
        object."""
        atom1 = self.coords[atom_indices[0],:]
        atom2 = self.coords[atom_indices[1],:]
        vec = atom2 - atom1
        dist = np.linalg.norm(vec)

        return dist


###############################################################################
#               Get Properties in comparison to other Geometry                #
###############################################################################
    def get_distance_to_equivalent_atoms(self, other_geometry) -> float:
        """
        Calculate the maximum distance that atoms of geom would have to be moved,
        to coincide with the atoms of self.
        
        """
        trans,dist = self.getTransformationIndices(other_geometry,
                                                   get_distances=True)
        return np.max(dist)
    
    
    def getTransformationIndices(self, other_geometry, norm_threshold=0.5, get_distances=False, periodic_2D=False):
        """Associates every atom in self to the closest atom of the same specie in other_geometry,
        :returns transformation_indices : np.array. The positions on the array correspond to the atoms in self;
            the values of the array correspond to the atoms in other_geometry"""
            
        assert len(self) == len(other_geometry), \
               "Geometries have different number of atoms {0} != {1}".format(len(self), len(other_geometry))

        # Replicate other geometry to also search in neighbouring cells
        if periodic_2D:
            other_geometry = other_geometry.getPeriodicReplica((3,3,1))
            other_geometry.moveByFractionalCoords([-1/3.0, -1/3.0, 0])

        # Get the atomic numbers of each geometry file: Later only compare matching atom types
        Z_values_1 = np.array([ut.PERIODIC_TABLE[s] for s in self.species], np.int64)
        Z_values_2 = np.array([ut.PERIODIC_TABLE[s] for s in other_geometry.species], np.int64)
        unique_Z = set(Z_values_1)

        # Allocate output arrays
        transformation_indices = np.zeros(len(self), np.int64)
        distances = np.zeros(len(self))
        atom2_indices = np.arange(len(other_geometry)) % len(self)

        # Loop over all types of atoms
        for Z in unique_Z:
            # Select all the coordinates that belong to that species
            select1 = (Z_values_1 == Z)
            select2 = (Z_values_2 == Z)
            # Calculate all distances between the geometries
            dist_matrix = scipy.spatial.distance_matrix(self.coords[select1, :], other_geometry.coords[select2, :])
            # For each row (= atom in self with species Z) find the index of the other_geometry (with species Z) that is closest
            index_of_smallest_mismatch = np.argmin(dist_matrix, axis=1)
            transformation_indices[select1] = atom2_indices[select2][index_of_smallest_mismatch]
            if get_distances:
                distances[select1] = [dist_matrix[i,index_of_smallest_mismatch[i]] for i in range(len(dist_matrix))]

        if get_distances:
            return transformation_indices, distances
        else:
            return transformation_indices


    def get_volume_of_unit_cell(self) -> float:
        """
        Calcualtes the volume of the unit cell.
    
        Returns
        -------
        float
            Volume of the unit cell.
    
        """
        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]
        volume = np.cross(a1, a2).dot(a3)
        return volume
    
    
    def get_geometric_center(self, ignore_center_attribute=False, indices=None) -> np.array:
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
        center : np.array
            Center of the geometry
        """

        if not hasattr(self, 'center') or self.center is None or ignore_center_attribute or indices is not None:
            if indices is None:
                indices = np.arange(self.n_atoms)
            center = np.mean(self.coords[indices],axis=0)
        else:
            center = np.zeros([3])
            for i, weight in self.center.items():
                center += self.coords[i,:] * weight
        return center


    def get_center_of_mass(self) -> np.array:
        """
        Mind the difference to self.getGeometricCenter

        Returns
        -------
        center_of_mass: np.array
            The 3D-coordinate of the center of mass
            
        """
        species_helper = []
        for si in self.species:
            si_new = si.split('_')[0]
            species_helper.append(si_new)
        
        masses_np = np.array([ATOMIC_MASSES[PERIODIC_TABLE[s]] for s in species_helper], dtype=np.float64)       
        center_of_mass = self.coords.T.dot(masses_np) / masses_np.sum()
        return center_of_mass
    
    
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
                if '_' in s:
                    curr_species = s.split('_')[0]
                else:
                    curr_species = s
                electrons.append(PERIODIC_TABLE[curr_species])

            except KeyError:
                KeyError('Species {} is not known'.format(s))
        return np.sum(electrons)
    
    
    def get_mass_of_all_atoms(self) -> list:
        """
        Determines the atomic mass for all atoms.

        Returns
        -------
        list
            List of atomic masses for all atoms in the same order as
            the atoms.

        """
        masses = []
        for s in self.species:
            try:
                if '_' in s:
                    curr_species = s.split('_')[0]
                else:
                    curr_species = s
                masses.append(ATOMIC_MASSES[PERIODIC_TABLE[curr_species]])

            except KeyError:
                KeyError('Atomic mass for species {} is not known'.format(s))
        return masses


    def get_atomic_mass(self) -> float:
        """
        Determines the atomic mass of the entrie geometry.

        Returns
        -------
        float
            Atomic mass of the entrie geometry.

        """
        atomic_mass = 0
        
        for s in self.species:
            atomic_mass += ATOMIC_MASSES[PERIODIC_TABLE[s]]
        
        return atomic_mass


    def get_max_size(self) -> float:
        """
        Determine the maximum distance between two atoms of the geometry.

        Returns
        -------
        max_dist : float
            Maximum distance between atoms.

        """
        coords = copy.deepcopy(self.coords)
        coords[:, :2] -= np.mean(coords[:, :2], axis=0)
        # self.centerXYCoordinates()
        distances=[]
        for atom in coords:
            distances += [np.linalg.norm(atom[:2])]
        max_dist = max(distances)
        return max_dist
    
    
    def get_area(self) -> float:
        """
        Returns the area of the surface described by lattice_vectors 0 and 1 of
        the geometry, assuming that the lattice_vector 2 is orthogonal to both.

        Returns
        -------
        float
            Area of the unit cell.

        """
        a=deepcopy(self.lattice_vectors[0,:-1])
        b=deepcopy(self.lattice_vectors[1,:-1])
        area = np.abs(np.cross(a,b))
        return area


    def get_area_in_atom_numbers(self,
                                 substrate_indices: Union[np.array, None]=None,
                                 substrate=None) -> float:
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
        topmost_sub_layer = self.getSubstrateLayer(
            layer_indices=[0],
            substrate_indices=substrate_indices,
            substrate=substrate
        )
        return topmost_sub_layer.n_atoms
    
    
    def get_bond_lengths(self, bond_factor: float=1.5) -> np.array:
        """
        Parameters
        ----------
        Parameter for bond detection based on atom-distance : float, optional
            DESCRIPTION. The default is 1.5.

        Returns
        -------
        np.array
            List of bond lengths for neighbouring atoms.

        """
        neighbouring_atoms = self.getAllNeighbouringAtoms(bond_factor=bond_factor)
        
        bond_lengths = []
        for v in neighbouring_atoms.values():
            bond_lengths.append(v[1])
            
        return np.array(bond_lengths)


    def get_number_of_atom_layers(self, threshold: float=1e-2) -> Union[dict, float]:
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
        layers = self.getAtomLayers(threshold=threshold)
        
        total_number_layers = 0
        for atom_species in layers:
            layers[atom_species] = len(layers[atom_species])
            total_number_layers += layers[atom_species]
    
        return layers, total_number_layers


    def get_unit_cell_parameters(self) -> Union[float, float, float, float, float, float]:
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

        alpha = np.arccos(np.dot(cell[1], cell[2])/(c*b))
        gamma = np.arccos(np.dot(cell[1], cell[0])/(a*b))
        beta = np.arccos(np.dot(cell[2], cell[0])/(a*c))

        return a, b, c, alpha, beta, gamma


    def get_lammps_unit_cell(self) -> Union[float, float, float, float, float, float]:
        """
        

        Returns
        -------
        Union[float, float, float, float, float, float]
            DESCRIPTION.

        """
        #cell = structure.get_cell(supercell=supercell)
        cell = self.lattice_vectors
        
        # generate lammps oriented lattice vectors
        ax = np.linalg.norm(cell[0])
        bx = np.dot(cell[1], cell[0]/np.linalg.norm(cell[0]))
        by = np.linalg.norm(np.cross(cell[0]/np.linalg.norm(cell[0]), cell[1]))
        cx = np.dot(cell[2], cell[0]/np.linalg.norm(cell[0]))
        cy = (np.dot(cell[1], cell[2])- bx*cx)/by
        cz = np.sqrt(np.dot(cell[2],cell[2])-pow(cx,2)-pow(cy,2))
        
        # get rotation matrix (poscar->lammps) from lattice vectors matrix
        lammps_cell = np.array([[ax, bx, cx],[0, by, cy],[0,0,cz]])
        # trans_cell = np.dot(np.linalg.inv(cell), lammps_cell.T)
        
        # rotate positions to lammps orientation
        #positions = np.dot(positions, trans_cell)
        
        # build lammps lattice vectors
        a, b, c, alpha, beta, gamma = self.get_cell_parameters()
        
        xhi = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        yhi = np.sqrt(pow(b,2)-pow(xy,2))
        yz = (b*c*np.cos(alpha)-xy * xz)/yhi
        zhi = np.sqrt(pow(c,2)-pow(xz,2)-pow(yz,2))
        
        return xhi, yhi, zhi, xy, xz, yz


    def get_orientation_of_main_axis(self) -> float:
        """
        Get the orientation of the main axis relative to the x axis
        the main axis is transformed such that it always points in the upper half of cartesian space
        
        Returns:
        float
            angle between main axis and x axis in degree
        """
        main_ax = self.get_main_axes()[1][:,0]
        if main_ax[1]<0:
            main_ax *= -1
        return (np.arctan2(main_ax[1], main_ax[0])*180/np.pi)
    

    def get_constrained_atoms(self) -> np.array:
        """
        Returns indice of constrained atoms.

        Returns
        -------
        inds : np.array
            Indice of constrained atoms.

        """
        constrain = np.any(self.constrain_relax,axis=1)
        inds = [i for i, c in enumerate(constrain) if c]
        return inds
    
    def get_unconstrained_atoms(self) -> np.array:
        """
        Returns indice of unconstrained atoms.

        Returns
        -------
        inds : np.array
            Indice of unconstrained atoms.

        """
        all_inds = list(range(len(self)))
        keep_inds = self.get_constrained_atoms()
        inds = list(set(all_inds) - set(keep_inds))
        return inds
    

###############################################################################
#                          Get Part of a Geometry                             #
###############################################################################
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
        import spglib
        lattice, scaled_positions, atomic_numbers = spglib.standardize_cell(self.getSPGlibCell())
        
        surface_vector = surface[0]*lattice[0,:] + surface[1]*lattice[1,:] + surface[2]*lattice[2,:]
        
        # TODO: this way of building lattice vectors parallel to the surface
        # is not ideal for certain surfaces
        dot_0 = surface_vector.dot(surface_vector)
        dot_1 = surface_vector.dot(lattice[0,:])
        dot_2 = surface_vector.dot(lattice[1,:])
        dot_3 = surface_vector.dot(lattice[2,:])
        
        if abs(dot_1) > threshold:
            frac = Fraction(dot_0 / dot_1).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v1 = m*surface_vector - n*lattice[0,:]
        else:
            v1 = lattice[0,:]
        
        if abs(dot_2) > threshold:
            frac = Fraction(dot_0 / dot_2).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v2 = m*surface_vector - n*lattice[1,:]
        else:
            v2 = lattice[1,:]
        
        if abs(dot_3) > threshold:
            frac = Fraction(dot_0 / dot_3).limit_denominator(1000)
            n, m = frac.numerator, frac.denominator
            v3 = m*surface_vector - n*lattice[2,:]
        else:
            v3 = lattice[2,:]
        
        surface_lattice = np.zeros((3,3))
        surface_lattice[0,:] = surface_vector
        
        ind = 1
        for v in [v1, v2, v3]:
            if not np.linalg.norm(v) == 0:
                surface_lattice[ind,:] = v
                rank = np.linalg.matrix_rank( surface_lattice )
    
                if rank == ind+1:
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
            for h in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    for l in range(-shell, shell+1):
                        
                        if (abs(h) < shell) and (abs(k) < shell) and (abs(l) < shell):
                            continue
                        
                        for new_species, coord in zip(atomic_numbers, scaled_positions):
                            
                            new_coord = coord.dot(lattice) + np.array([h,k,l]).dot(lattice)
                            frac_new_coord = utils.get_fractional_coords(new_coord, surface_lattice)
                        
                            L1 = np.sum( frac_new_coord >= 1-threshold )
                            L2 = np.sum( frac_new_coord < -threshold )
                            
                            if not L1 and not L2:
                                slab.add_atoms([new_coord], [PERIODIC_TABLE[new_species]])
                                add_next_shell = True
                        
                        
            if not shell == 0 and not add_next_shell:
                break
            
            if shell == 100:
                warnings.warn('<Geometry.get_primitive_slab> could not build a correct slab.')
        
        slab.alignLatticeVectorToVector(np.array([0,0,1]),2)
        slab.alignLatticeVectorToVector(np.array([1,0,0]),0)
        
        scaled_slab_lattice = np.array(slab.lattice_vectors)
        # break symmetry in z-direction
        scaled_slab_lattice[2,:] *= 2
        frac_coords = utils.get_fractional_coords(slab.coords, scaled_slab_lattice)
        species = [PERIODIC_TABLE[s] for s in slab.species]
        
        (primitive_slab_lattice, primitive_slab_scaled_positions, primitive_slab_atomic_numbers) \
        = spglib.find_primitive((scaled_slab_lattice, frac_coords, species), symprec=1e-5)
        
        primitive_slab_species = [PERIODIC_TABLE[s] for s in primitive_slab_atomic_numbers]
        primitive_slab_coords = primitive_slab_scaled_positions.dot(primitive_slab_lattice)
        # replace lattice vector in z-direction
        primitive_slab_lattice[2,:] = slab.lattice_vectors[2,:]
        
        primitive_slab = self.__class__()
        primitive_slab.lattice_vectors = primitive_slab_lattice
        primitive_slab.add_atoms(primitive_slab_coords, primitive_slab_species)
        primitive_slab.moveToFirstUnitCell()
        
        # Sanity check: primitive_slab must be reducable to the standard unit cell 
        check_lattice, _, _ = spglib.standardize_cell(primitive_slab.getSPGlibCell())
        
        assert np.allclose( check_lattice, lattice ), \
        '<Geometry.get_primitive_slab> the slab that was constructed \
        could not be reduced to the original bulk unit cell. Something \
        must have gone wrong.'
        
        return primitive_slab

    
    def get_slab(self,
                 layers: int,
                 surface: Union[np.array, None]=None,
                 threshold: float=1e-6,
                 surface_replica: (int, int)=(1,1),
                 vacuum_height: Union[float, None]=None, 
                 bool_shift_slab_to_bottom: bool=False):
        """
        Generates a slab.
        
        Returns:
        --------
        primitive_slab : Geometry

        Parameters
        ----------
        layers : int
            Number of layers of the slab.
        surface : np.array | None, optional
            miller indices, eg. (1,1,1)
        threshold : float, optional
            numerical threshold for symmetry operations
        surface_replica : (int, int), optional
            DESCRIPTION. The default is (1,1).
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
        
        slab_layers = primitive_slab.getNumberOfAtomLayers()[1]

        replica = np.array([1,1, int(np.ceil(layers / slab_layers))], dtype=np.int32)
        replica[:2] = surface_replica
        slab_new = primitive_slab.getPeriodicReplica(replica)
        
        slab_new_layers = slab_new.getAtomLayers()
        
        for atom_species in slab_new_layers:
            z_coords = list(slab_new_layers[atom_species])
            z_coords = sorted(z_coords)
            
            n_layers_to_remove = len(z_coords) - layers
            
            atom_indices_to_remove = []
            for ind in range(n_layers_to_remove):
                atom_indices_to_remove += slab_new_layers[atom_species][z_coords[ind]]
            
            slab_new.removeAtoms( np.array(atom_indices_to_remove, dtype=np.int32) )

            if vacuum_height is not None:
                slab_new.setVacuumheight(
                    vac_height=vacuum_height,
                    bool_shift_to_bottom=bool_shift_slab_to_bottom
                )
            else:
                if bool_shift_slab_to_bottom:
                    self.shift_to_bottom()

        return slab_new


###############################################################################
#                           Evaluation Functions                              #
###############################################################################
    def check_symmetry(self, transformation, tolerance, return_symmetrical=False):
        """Returns True if the geometry is symmetric with respect to the transformation, and False if it is not.
        If the geometry is periodic, transformation can be tuple (rotation, translation) or np.array (only rotation), \
        otherwise it can only be np.array
        :param return_symmetrical: return the corresponding transformed geometry together with the result"""
        if isinstance(transformation,np.ndarray):
            R = transformation
            t=np.array([0,0,0])
        elif isinstance(transformation,tuple):
            if self.lattice_vectors is None:
                raise AttributeError('Can not check translational symmetries in a non periodic structure')
            else:
                R,t=transformation

        # original structure with all atoms in the first unit cell
        self_1UC = copy.deepcopy(self)
        self_1UC.moveToFirstUnitCell()
        # original structure centered for reordering
        centered_geometry = copy.deepcopy(self)
        centered_geometry.centerCoordinates()
        #apply transformation
        symm_geometry = copy.deepcopy(self)
        symm_geometry.transformFractional(R,np.array([0,0,0]),self.lattice_vectors)
        symm_geometry.moveByFractionalCoords(t)
        #prevent problems if the center is very close to the edge
        center = utils.get_fractional_coords(symm_geometry.getGeometricCenter(),symm_geometry.lattice_vectors)
        center[:2] %= 1.0
        if 1-center[0]<0.001:
            adjust = -(center[0]-0.0001)
            symm_geometry.moveByFractionalCoords([adjust,0,0])
        if 1-center[1]<0.001:
            adjust = -(center[1]-0.0001)
            symm_geometry.moveByFractionalCoords([0,adjust,0])

        symm_geometry.moveCenterOfAtomsToFirstUnitCell(lattice=self.lattice_vectors)

        #reorder atoms
        offset_symm = np.mean(symm_geometry.coords,axis=0)
        symm_geometry.centerCoordinates()
        indices = centered_geometry.getTransformationIndices(symm_geometry)
        symm_geometry.reorder_atoms(indices)
        symm_geometry.move(offset_symm)
        # compare in first unit cell
        symm_geometry_1UC = copy.deepcopy(symm_geometry)
        symm_geometry_1UC.moveToFirstUnitCell()
        is_symmetric = symm_geometry_1UC.isEquivalent(self_1UC,tolerance=tolerance,check_neightbouring_cells=True)
        # the simultaneous use of: 1)The moveToFirstUnitCell() function
        #                          2)The check_neighbouring_cells flag
        # is somehow redundant, but it works correctly.

        if return_symmetrical:
            return is_symmetric,symm_geometry
        else:
            return is_symmetric

    

###############################################################################
#                             VARIOUS                                         #
#                          not yet sorted                                     #
###############################################################################
    def move_multipoles(self, shift: np.array) -> None:
        """
        Moves all the multipoles by a shift vector
        :param shift: list or array, len==3
        :return:
        """
        assert len(shift)==3
        for m in self.multipoles:
            m[0]+=shift[0]
            m[1]+=shift[1]
            m[2]+=shift[2]
        
    
    def getCollidingGroups(self, distance_threshold=1E-2, check_3D = False):
        """
        Remove atoms that are too close too each other from the geometry file.
        This approach is useful if one maps back atoms into a different cell and then needs to get rid
        of overlapping atoms

        Parameters
        ----------
        distance_threshold: float
            maximum distance between atoms below which they are counted as duplicates

        Returns
        -------
        """

        # get all distances between all atoms

        z_period = [-1,0,1] if check_3D else [0]
        index_tuples = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in z_period:
                    curr_shift = i*self.lattice_vectors[0,:] + j*self.lattice_vectors[1,:] + k*self.lattice_vectors[2,:]

                    atom_distances = scipy.spatial.distance.cdist(self.coords, self.coords+curr_shift)
                    index_tuples += self._getCollisionIndices(atom_distances, distance_threshold)
        if len(index_tuples) > 0:
            G = nx.Graph()
            G = nx.from_edgelist(itertools.chain.from_iterable(itertools.pairwise(e) for e in index_tuples))
            G.add_nodes_from(set.union(*map(set, index_tuples))) # adding single items
            atoms_to_remove = list(nx.connected_components(G))
            return [sorted(list(s)) for s in atoms_to_remove]
        else:
            return []

    
    def _getCollisionIndices(self, atom_distances, distance_threshold=1E-2):
        """ helper function for removeDuplicateAtoms
        
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
                        
        return [(k, ind) for k, value in colliding_atoms_dict.items() for ind in list(value)]


    def constrainAtomsBasedOnIndex(
        self, 
        indices_of_atoms_to_constrain,
        constrain_dim_flags=None,
    ):
        """Sets a constraint for a few atoms in the system (identified by
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
        indices_of_atoms_to_constrain
        constrain_dim_flags                 list[boolean]   default: [True, True, True]
        """
        if constrain_dim_flags is None:
            constrain_dim_flags = [True, True, True]

        self.constrain_relax[indices_of_atoms_to_constrain,:] = constrain_dim_flags


    def constrainAtomsBasedOnSpaceInterval(
        self,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        zlim=(-np.inf, np.inf),
        constrain_dim_flags=None
    ):
        """
        Constrain all atoms that are within a cuboid (defined by
        limits in all dimensions: xlim, etc.) for a geometry relaxation.

        It is possible to define which dimension will be constrained, but since
        the number of atoms in the cuboid is only calculated at runtime
        the dimensions may only be set for all atoms at once. If you need to
        set them individually please use constrainAtomsBasedOnIndex.

        Parameters
        ----------
        zlim
        xlim
        ylim
        constrain_dim_flags                 list[boolean]   default: [True, True, True]
        """
        if constrain_dim_flags is None:
            constrain_dim_flags = [True, True, True]

        #--- get indices of all atoms outside the required interval ---
        indices_outside = self.getCroppingIndices(
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            auto_margin=False
        )
        #---

        #--- Filter all that are outside ---
        # The indices of the atoms of relevance to us are all that are NOT
        # outside of the cuboid
        indices_inside = [
            i for i in range(len(self)) if i not in indices_outside
        ]
        #---

        self.constrainAtomsBasedOnIndex(indices_inside, constrain_dim_flags)


    def setExternalForceBasedOnIndex(
        self, 
        indices_of_atoms_to_constrain,
        external_force=np.zeros(3),
    ):
        """Sets a constraint for a few atoms in the system (identified by
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
        indices_of_atoms_to_constrain
        constrain_dim_flags                 list[boolean]   default: [True, True, True]
        """
        
        self.external_force[indices_of_atoms_to_constrain,:] = external_force
        
    
    def setCalculateFrictionBasedOnIndex(
            self, 
            indices_of_atoms,
            calculate_friction=True,
            ):
        
        self.calculate_friction[indices_of_atoms] = calculate_friction


    def cropToUnitCell(self, lattice=None, frac_coord_factors=[0,1]):
        ''' Removes all atoms that are outside the given unit cell. Similar to self.crop() but allows for arbitrary unit cells'''
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

        remove_inds = list(set(remove_inds))

        self.removeAtoms(remove_inds)
        self.lattice_vectors = lattice
        
        #In the following all redundant atoms, i.e. atoms that are multiplied at the same position when the unitcell is repeated periodically, are removed from the new unit cell
        epsilon = 0.1 # Distance in Angstrom for which two atoms are assumed to be in the same position
        init_geom = self
        allcoords = init_geom.coords
        allindices = init_geom.getIndicesOfAllAtoms()


        # generate all possible translation vectors that could map an atom of the unit cell into itsself 
        prim_lat_vec = []
        for i in range(3):
            prim_lat_vec.append([init_geom.lattice_vectors[i], -init_geom.lattice_vectors[i]])
        self_mapping_translation_vectors = []

        for i in prim_lat_vec:
            for sign in range(2):
                self_mapping_translation_vectors.append(i[sign])

        for i in range(3):
            for sign0 in range(2):
                for k in range(3):
                    for sign1 in range(2):
                        if i != k:
                            #print(f'i {i} k {k} sign0 {sign0} sign1 {sign1}')
                            single_addition_vector = prim_lat_vec[i][sign0]+prim_lat_vec[k][sign1]
                            self_mapping_translation_vectors.append(single_addition_vector)
                            
        for i in range(3):
            for sign0 in range(2):
                for k in range(3):
                    for sign1 in range(2):
                                for l in range(3):
                                    for sign2 in range(2):
                                        if i != k and i != l and k != l :
                                            single_addition_vector = prim_lat_vec[i][sign0]+prim_lat_vec[k][sign1]+prim_lat_vec[l][sign2]
                                            self_mapping_translation_vectors.append(single_addition_vector)
                        



        ## Find the indices of those atoms that are equivalent, i.e. atoms that are doubled when the unit cell is repeated periodically
        
        doubleindices = [] # list of pairs of atom indices that are equivalent
        for i, coords_i in enumerate(allcoords):
            for trans_l in self_mapping_translation_vectors:
                coords_i_shift_l = copy.deepcopy(coords_i)
                coords_i_shift_l += trans_l
                for j, coords_j in enumerate(allcoords):
                    if j != i:
                        distance_i_shift_l_j = np.linalg.norm(coords_i_shift_l - coords_j)
                        if distance_i_shift_l_j < epsilon:
                            doubleindices.append([i,j])

        for i in range(len(doubleindices)):
               doubleindices[i].sort()

        ###################################################################
        ##Create a list of redundant atoms according to the atoms that are equivalent 
        # according to all the pairs in doubleindices

        liste = doubleindices 
        to_be_killed = [] # List of all atom indicess that are redundant
        for i, liste_i in enumerate(liste):
            replacer = liste_i[0]
            to_be_replaced = liste_i[1]  
            to_be_killed.append(to_be_replaced)
            for j, liste_j in enumerate(liste):
                for k in range(2):
                    if liste_j[k] == to_be_replaced :
                        liste[j][k] = replacer
        remainers = [j[0] for j in liste]                
        for r in remainers:
            for k in to_be_killed:
                if k == r :
                    to_be_killed.remove(k)

        self.removeAtoms(to_be_killed)




    def crop(self, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), zlim=(-np.inf, np.inf), auto_margin=False):
        """Removes all atoms that are outside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders"""
        indices_to_remove = self.getCroppingIndices(xlim, ylim, zlim, auto_margin)
        self.removeAtoms(indices_to_remove)

    def getCropped(self, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), zlim=(-np.inf, np.inf), auto_margin=False):
        """Returns a copy of the object to which self.crop has been applied"""
        newgeom = deepcopy(self)
        newgeom.crop(xlim=xlim,ylim=ylim,zlim=zlim,auto_margin=auto_margin)
        return newgeom

    def cropInverse(self, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), zlim=(-np.inf, np.inf), auto_margin=False):
        """Removes all atoms that are inside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders"""
        indices_to_keep = self.getCroppingIndices(xlim, ylim, zlim, auto_margin)
        indices_to_remove = [i for i in range(self.n_atoms) if i not in indices_to_keep]
        self.removeAtoms(indices_to_remove)
    
    def getCroppingIndices(self,
                           xlim=(-np.inf, np.inf),
                           ylim=(-np.inf, np.inf),
                           zlim=(-np.inf, np.inf),
                           auto_margin=False,
                           inverse=False
                           ):
        """Gets indices of all atoms that are outside specified bounds.
        If auto_margin == True then an additional margin of the maximum covalent radius
        is added to all borders
        :param inverse : if True, gets indices of all atoms INSIDE specified bounds"""

        assert len(xlim) == 2, "xlim must have a lower and an upper bound"
        assert len(ylim) == 2, "ylim must have a lower and an upper bound"
        assert len(zlim) == 2, "zlim must have a lower and an upper bound"
        if auto_margin:
            margin = max([ut.getCovalentRadius(s) for s in self.species])
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
            indices_to_remove = [i for i in range(self.n_atoms) if i not in indices_to_remove]

        return indices_to_remove


    def getSubstrateIndicesFromParts(self, do_warn=True):
        """
        returns indices of those atoms that a part of the substrate.
        The definition of the substrate does NOT rely of it being a metal or the height or something like that.
        Instead a geometry part named 'substrate' must be defined.

        do_warn: boolean    can be set to False to suppress warnings

        Returns
        -------
        substrate_indices   list[int]
        """

        substrate_key = 'substrate'     # should probably moved to a class variable, but first finished folder-read-write

        if not hasattr(self, 'geometry_parts') or not hasattr(self, 'geometry_part_descriptions'):
            if do_warn:
                print("getSubstrate: geometry parts not defined")
            return None

        if substrate_key not in self.geometry_part_descriptions:
            if do_warn:
                print("getSubstrate: geometry parts are defined, "
                      "but part '{}' not found" .format(substrate_key))
            return None

        index_of_geometry_parts_substrate = self.geometry_part_descriptions.index(substrate_key)
        substrate_indices = self.geometry_parts[index_of_geometry_parts_substrate]
        return substrate_indices

    def getIndicesOfMetal(self):
        "Gets indices of all atoms with atomic number > 18 and atomic numbers 3,4,11,12,13,14"
        atom_inds = [ut.getAtomicNumber(s) for s in self.species]
        metal_atoms = []
        for i,ind in enumerate(atom_inds):
            if (ind> 18) or (ind in [3,4,11,12,13,14]):
                metal_atoms.append(i)
        return metal_atoms

    def getIndicesOfSubsection(self,
                               xlim=(-np.inf, np.inf),
                               ylim=(-np.inf, np.inf),
                               zlim=(-np.inf, np.inf),
                               ):
        "Gets indices of all atoms contained in the indicated volume of space"
        indices = self.getCroppingIndices(xlim=xlim,
                                          ylim=ylim,
                                          zlim=zlim,
                                          inverse=True)
        return indices


    
    def getIndicesOfMolecules(self, substrate_species=None):
        """WARNING: do not use this function, it is deprecated!
        It fetches the indices of the substrate atoms, but it defaults to 
        just returning all non-metal atom's indices!
        
        If substrate_species is given indices of all atoms that are not of the 
        substrate species are returned.
        """
        
        if substrate_species:
            substrate_indices = self.getIndicesOfSpecies(substrate_species)
        else:
            substrate_indices = self.getIndicesOfMetal()
        
        return [i for i in range(self.n_atoms) if i not in substrate_indices]


    def getIndicesOfSpecies(self, species):
        """Returns all indices of atoms the are of species defined in the input.
        species can be a string of a list
        """
        # make sure species is a list
        if isinstance(species, str):
            species = [species]
        
        return [i for i in range(self.n_atoms) if self.species[i] in species]


    def getSubstrateIndices(self, primitive_substrate=None, dimension = 2,threshold = .3):
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
        indices of all substrate atoms: list[int]
        """

        # case 1: if a primitive_substrate was passed, use that one for the decision
        # (copied from self.removeSubstrate)
        if primitive_substrate is not None:
            substrate_species = set(primitive_substrate.species)
            substrate_heights = primitive_substrate.coords[:,dimension]
            substrate_candidate_indices = [i for i,s in enumerate(self.species) if s in substrate_species]
            substrate_indices = []
            for c in substrate_candidate_indices:
                if np.any(np.absolute(substrate_heights - self.coords[c,dimension]) < threshold):
                    substrate_indices.append(c)
            return substrate_indices

        # case 2: if no substrate was passed but a geometry_parts "substrate" is defined in geometry_parts, use that one
        substrate_indices_from_parts = self.getSubstrateIndicesFromParts(do_warn=False)
        if substrate_indices_from_parts is not None:
            return substrate_indices_from_parts
            # adsorbate_indices = [i for i in self.getIndicesOfAllAtoms() if i not in substrate_indices_from_parts]

        # case 3: if neither a substrate was passed, nor a geometry_parts "substrate" is defined,
        #            use a fallback solution: assume that the substrate (and nothing else) is a metal
        warnings.warn("Geometry.getIndicesOfAdsorbates: Substrate is not explicitly defined. "
                      "Using fallback solution of counting all metal atoms as substrate.")
        substrate_indices_from_metal = self.getIndicesOfMetal()
        return substrate_indices_from_metal
        # adsorbate_indices = [i for i in self.getIndicesOfAllAtoms() if i not in substrate_indices_from_metal]

    def getSubstrate(self, primitive_substrate=None):
        substrate_indices = self.getSubstrateIndices(primitive_substrate=primitive_substrate)
        return self.getAtomsByIndices(substrate_indices)

    def getAdsorbateIndices(self, primitive_substrate=None):
        """
        This method returns the indices of all atoms that are NOT part of the substrate.
        In a classical organic monolayer on a metal substrate these are simply all molecules.
        But the substrate can also be organic in which case it can't be picked by being a metal.
        And there might be multiple adsorbate layers in which case only the molecules of the highest layer
        shall be counted as adsorbates.

        Returns
        -------
        indices of all adsorbate atoms: list[int]
        """

        substrate_indices = self.getSubstrateIndices(primitive_substrate=primitive_substrate)
        # invert:
        return [i for i in self.getIndicesOfAllAtoms() if i not in substrate_indices]

    def getAdsorbates(self, primitive_substrate=None):
        adsorbate_indices = self.getAdsorbateIndices(primitive_substrate=primitive_substrate)
        return self.getAtomsByIndices(adsorbate_indices)
    
    def getIndicesOfAllAtoms(self, species=None):
        if species is None:
            return [i for i in range(self.n_atoms)]
        else:
            return [i for i in range(self.n_atoms) if self.species[i] == species]

    #TODO: Rename this function to removeMetalAtoms to avoid confusion / unexpected results!
    def removeMetalSubstrate(self):
        "Removes all atoms with atomic number > 18 and atomic numbers 3,4,11,12,13,14"
        metal_atoms = self.getIndicesOfMetal()
        self.removeAtoms(metal_atoms)

    def removeSubstrate(self, primitive_substrate, dimension=2, threshold=0.3):
        '''Removes all substrate atoms given the primitive substrate by identifying 
           species and height
           
           Parameters
           ----------
           primitive_substrate: Geometry
               primitive substrate file of system
               
           dimension: int
               dimension to use as z-axis
           
           threshold: float
               height threshold in A
        '''
        # TODO create species dependent heights and speedup calculations

        # old:
        # substrate_species = set(primitive_substrate.species)
        # substrate_heights = primitive_substrate.coords[:,dimension]
        # candidates = [i for i,s in enumerate(self.species) if s in substrate_species]
        # remove_atoms = []
        # for c in candidates:
        #     if np.any(np.absolute(substrate_heights - self.coords[c,dimension])<threshold):
        #         remove_atoms.append(c)
        # self.removeAtoms(remove_atoms)

        substrate_indices = self.getSubstrateIndices(primitive_substrate=primitive_substrate)
        self.removeAtoms(substrate_indices)

    
    def removeMolecules(self):
        "Removes all atoms that are not metal"
        mol_inds = self.getIndicesOfMolecules()
        self.removeAtoms(mol_inds)

    def removeAdsorbates(self, primitive_substrate=None):
        """
        Removes all atoms that are not part of the substrate (in place!)
        """
        adsorbate_indices = self.getAdsorbateIndices(primitive_substrate=primitive_substrate)
        self.removeAtoms(adsorbate_indices)

    def removePeriodicity(self):
        """
        Makes geometry non-periodic by setting its lattice vectors to zero
        """
        self.lattice_vectors = np.zeros((3,3), dtype=float)

    def getAtomsByIndices(self, atom_indices):
        """
        Return a geometry instance with the atoms listed in atom_indices
        Parameters
        ----------
        atom_indices    list of integers, indices of those atoms which should be copied to new geometry

        Returns
        -------
        Geometry
        """
        
        new_geom = self.__class__()
        new_geom.add_atoms(self.coords[atom_indices, :], [self.species[i] for i in atom_indices],
                     constrain_relax=self.constrain_relax[atom_indices],
                     initial_moment=[self.initial_moment[i] for i in atom_indices],
                     initial_charge=[self.initial_charge[i] for i in atom_indices])
        new_geom.lattice_vectors = self.lattice_vectors
        return new_geom
    
    def getAtomsBySpecies(self, species):
        """
        get new geometry file with specific atom species
        """
        
        L = np.array(self.species) == species
        atom_indices = np.where(L)[0]
        return self.getAtomsByIndices(atom_indices)
    
    
    def getMolecules(self):
        mol_inds = self.getIndicesOfMolecules()
        return self.getAtomsByIndices(mol_inds)
    
    def getMetalSubstrate(self):
        """
        Returns all atoms with atomic number > 18 and atomic numbers 3,4,11,12,13,14
        """
        metal_atoms = self.getIndicesOfMetal()
        return self.getAtomsByIndices(metal_atoms)
    
    def getFirstMetalSubstrateLayer(self):
        sub = self.getMetalSubstrate()
        max_z = max( sub.coords[:,2] )
        L1 = sub.coords[:,2] > max_z-1
        atom_indices = np.nonzero(L1)
        return sub.getAtomsByIndices(atom_indices[0])

    def getFirstSubstrateLayer(self, primitive_substrate=None, layer_separation = 1.0, only_contains_substrate=False):
        """
        Returns a new GeometryFile that contains the top-most layer of the substrate. If the substrate is not purely
        metallic, the primitive unit cell of the slab ('primitive_substrate') is needed in order to discriminate between
        substrate and adsorbate molecules.
        """
        if only_contains_substrate:
            sub = self
        else:
            sub = self.getSubstrate(primitive_substrate)

        max_z = max( sub.coords[:,2] )
        L1 = sub.coords[:,2] > max_z - layer_separation
        atom_indices = np.nonzero(L1)
        return sub.getAtomsByIndices(atom_indices[0])
    
    def getAtomLayers(self, threshold=1e-2):
        """Returns a dict of the following form:
        {<Element symbol>: {height: [indices of atoms of element at height]}}
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

    def getAtomLayersByHeight(self, threshold=1e-2):
        """Similarly to the above method this function returns a dict continaing
        info about height and the indices of atoms at that height. The
        dict structure is thus:

        {height: [indices of atoms at that height]}

        The difference to above: in the above case there is another dictionary
        level, which divides by atomic species.
        """

        layers_by_species = self.getAtomLayers(threshold=threshold)

        layers_by_height = defaultdict(list)

        #--- merge height-indices dicts ---
        for data in layers_by_species.values():

            for height, indices in data.items():
                new=True
                for new_height in layers_by_height.keys():
                    if abs(height-new_height) < threshold:
                        layers_by_height[new_height] += indices
                        new=False
                if new:
                    layers_by_height[height] += indices

        # sort dictionary by descending height
        layers_by_height = dict(sorted(layers_by_height.items(),reverse=True))
        return layers_by_height


    def getSubstrateLayer(self, layer_indices, substrate_indices=None,
                          substrate=None, threshold=1e-2,primitive_substrate=None):
        """
        get substrate layer by indices. The substrate is determined by
        default by the function self.getSubstrate(). For avoiding a faulty
        substrate determination it can be either given through indices or
        through the substrate geometry itself
        :param layer_indices: list of indices of layers that shall be returned
        :param substrate_indices: list of indices of substrate atoms
        :param substrate: geometry file of substrate
        :param substrate: geometry file of a primitive unit cell of substrate
        """

        if substrate is not None:
            sub = substrate
        elif substrate_indices is not None:
            sub = self.getAtomsByIndices(substrate_indices)
        else:
            sub = self.getSubstrate(primitive_substrate=primitive_substrate)
        layers = sub.getAtomLayersByHeight(threshold=threshold)
        
        heights = list(layers.keys())
        heights = np.sort(heights)
        heights = heights[::-1]
        # if layer_indices is an empty list, keeps the substrate as a whole
        if not layer_indices:
            sub_new = sub
        else:
            sub_new = self.__class__()
            sub_new.lattice_vectors = sub.lattice_vectors
            for layer_ind in layer_indices:
                sub_new += sub.getAtomsByIndices(layers[heights[layer_ind]])
        
        return sub_new

    
    def getAllNeighbouringAtoms(self, bond_factor=1.5):
        coords = self.coords
        species = self.species

        all_species = set(species)
        cov_radii = ut.COVALENT_RADII
        all_species_pairs = itertools.product(all_species, repeat=2)
        
        bond_thresholds = {}
        for pair in all_species_pairs:
            bond_thresholds[pair] = (cov_radii[pair[0]]+cov_radii[pair[1]])*bond_factor

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
    

    def removeCollisions(self, keep_latest: Union[bool, slice] = True):
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
        else:
            raise ValueError("keep_latest must be a bool or a slice object")
        collisions = self.getCollidingGroups()
        for group in collisions:
            indices += group[selection]
        self.removeAtoms(indices)


    def removeRemovium(self):
        indices = []
        rm_indices = [i for i, s in enumerate(self.species) if s == "Rm"]
        collisions = self.getCollidingGroups()
        for group in collisions:
            if any([i in rm_indices for i in group]):
                indices += group
        self.removeAtoms(indices)

    def add_CPs(self, CP_coords):
        """ Adds CPs for CREST calculations"""
        self.add_atoms(CP_coords,
                      ['CP' for n in range(len(CP_coords))],
                      constrain_relax=[[True,True,True] for n in range(len(CP_coords))])


    def remove_CP(self):
        """ Removes CPs to turn the CREST setting of the calculation off"""
        CPs = np.array(self.species)=='CP'
        CP_inds = np.arange(0,len(self.species))[CPs]
        self.removeAtoms(CP_inds)

    def getSpeciesAtomicNumber(self):
        """Get the atomic numbers of all atoms in the geometry file"""
        species = [PERIODIC_TABLE[s] for s in self.species]
        return species


    def get_periodic_replica(self,
                             replications: tuple,
                             lattice: Union[np.array, None]=None,
                             explicit_replications: Union[list, None]=None):
        """
        Return a new geometry file that is a periodic replica of the original file.
        repeats the geometry N-1 times in all given directions:
        (1,1,1) returns the original file

        Parameters
        ----------
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
        #TODO implement geometry_parts the right way (whatever this is)
        if lattice is None:
            lattice = np.array(self.lattice_vectors)

        if explicit_replications:
            rep = explicit_replications
            lattice_multipliers = [np.max(t)-np.min(t) for t in explicit_replications]
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
            frac_shift = np.zeros([1,3])
            frac_shift[0,:len(frac_offset)] = frac_offset
            offset = utils.get_cartesian_coords(frac_shift, lattice)
            new_coords[insert_pos:insert_pos+self.n_atoms, :] = self.coords + offset
            insert_pos += self.n_atoms
            
        new_geom = self.__class__()
        
        new_geom.add_atoms(new_coords, new_species, new_constrain)
        new_geom.lattice_vectors = lattice

        # save original lattice vector for visualization
        if hasattr(self, 'original_lattice_vectors'):
            new_geom.original_lattice_vectors = copy.deepcopy(self.original_lattice_vectors)
        else:
            new_geom.original_lattice_vectors = copy.deepcopy(self.lattice_vectors)

        for i,r in enumerate(lattice_multipliers):
            new_geom.lattice_vectors[i,:] *= r
        return new_geom
    

    def getPeriodicReplicaToFillBox(self,
                                    box_limits
                                    ):
        """
        Returns a XY periodic replica of the geometry file of exactly the necessary dimensions to fill the space indicated by box_limits
        :param box_limits: iterable(min_x, max_x, min_y, max_y)
        :return: GeometryFile()
        """
        from sample.helpers.trigonometry import getPointsOnSquare

        def any_in_box(coords_2D_array,box):
            # checks if any of the XY coordiantes in coords_2D_array is included within the boundaries of box
            atoms_in = (coords_2D_array[:,0]>=box[0]) & (coords_2D_array[:,0]<=box[1]) & (coords_2D_array[:,1]>=box[2]) & (coords_2D_array[:,1]<=box[3])
            any_in = np.any(atoms_in)
            return any_in

        original_geom = deepcopy(self)
        new_geom = deepcopy(self)
        layer_index = 1
        while True:
            # proceeds along squares of increasing size
            shifts_copies_layer = getPointsOnSquare(layer_index)
            any_in_radius = False
            for shift in shifts_copies_layer:
                shifted_geom = deepcopy(original_geom)
                shifted_geom.moveByFractionalCoords(np.array([shift[0],shift[1],0]),shifted_geom.lattice_vectors)
                # if the shifted geometry has an atom within the box, includes it in the box
                if any_in_box(shifted_geom.coords[:,:-1],box_limits):
                    new_geom += shifted_geom
                    any_in_radius = True
            # if a square layer is completely outside of the box, stops
            if not any_in_radius:
                break
            layer_index += 1
        return new_geom
    

    def splitIntoMolecules(self,threshold):
        """Splits a structure into individual molecules. Two distinct molecules A and B are defined as two sets of atoms,
        such that no atom in A is closer than the selected thresold to any atom of B"""



        from scipy.spatial import distance_matrix
        coords = deepcopy(self.coords)
        distances = distance_matrix(coords,coords)
        distances[distances<=threshold]=1
        distances[distances>threshold]=0

        def scan_line(line_index,matrix,already_scanned_lines_indices):
            already_scanned_lines_indices.append(line_index)
            line = matrix[line_index]
            links = np.nonzero(line)[0]
            links = [l for l in links if l not in already_scanned_lines_indices]
            return links, already_scanned_lines_indices


        molecules_indices_sets = []
        scanned_lines_indices = []
        indices_set = []
        # scan lines one by one, but skips those that have already been examined
        for i,line in enumerate(distances):
            if i in scanned_lines_indices:
                continue
            # add line to the present set
            indices_set.append(i)
            # get indices of the lines connected to the examined one
            links, scanned_lines_indices = scan_line(i,distances,scanned_lines_indices)
            indices_set += links
            # as long as new links are found, adds the new lines to the present set
            while len(links)>0:
                new_links = []
                for l in links:
                    if l not in scanned_lines_indices:
                        new_links_part, scanned_lines_indices = scan_line(l,distances,scanned_lines_indices)
                        new_links += new_links_part
                links=set(new_links)
                indices_set += links
            # once no more links are found, stores the present set and starts a new one
            molecules_indices_sets.append(indices_set)
            indices_set=[]

        molecules = []
        for molecule_indices in molecules_indices_sets:
            complementary_indices = [x for x in self.getIndicesOfAllAtoms() if x not in molecule_indices]
            g=deepcopy(self)
            g.removeAtoms(complementary_indices)
            molecules.append(g)

        return molecules


    def isEquivalent(self, geom, tolerance=0.01,check_neightbouring_cells=False):
        """Check if this geometry is equivalent to another given geometry.
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
                geom = geom.getPeriodicReplica((3,3),explicit_replications=[[-1,0,1],[-1,0,1]])
            else:
                print('Non periodic structure. Ignoring check_neighbouring_cells')


        n_atoms = self.n_atoms
        n_atoms_geom = geom.n_atoms
        # Check for each atom in coords1 that is has a matching atom in coords2
        for n1 in range(n_atoms):
            is_ok = False
            for n2 in range(n_atoms_geom):
                if self.species[n1] == geom.species[n2]:
                    d = np.linalg.norm(self.coords[n1,:] - geom.coords[n2,:])
                    if d < tolerance:
                        # Same atom and same position
                        is_ok = True
                        break
                    
            if not is_ok:
                return False
        return True
        
    def isEquivalentUpToTranslation(self, geom, get_translation=False, tolerance=0.01, check_neighbouring_cells=False):
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
        meanA = self.getGeometricCenter(ignore_center_attribute=True)
        meanB = geom.getGeometricCenter(ignore_center_attribute=True)
        translation = meanA - meanB
        self.centerCoordinates(ignore_center_attribute=True)
        geom.centerCoordinates(ignore_center_attribute=True)

        # check if they are equivalent (up to permutation)
        is_equivalent = self.isEquivalent(geom, tolerance, check_neightbouring_cells=check_neighbouring_cells)
        
        # undo shifting to origin
        self.coords += meanA
        geom.coords += meanB
        
        if get_translation:
            return is_equivalent, translation
        else:
            return is_equivalent        




    def getSPGlibCell(self):
        """
        Returns the unit cell in a format that can be used in spglib (to find symmetries)
        
        return : tuple
            (lattice vectors, frac coordinates of atoms, atomic numbers)
        """
        coordinates = utils.get_fractional_coords(self.coords, self.lattice_vectors)
        
        atom_number = []
        for atom_name in self.species:
            atom_number.append( PERIODIC_TABLE[atom_name] )
        
        return (self.lattice_vectors, coordinates, atom_number)
    
    def getSymmetries(self, save_directory=None,symmetry_precision=1e-05):
        """
        Returns symmetries (rotation and translation matrices) from spglig.
        works only for unitcell and supercell geometries (lattice vecotrs must not be 0)

        Beware: The returned symmetry matrices are given with respect to fractional coordinates, not Cartesian ones!

        See https://atztogo.github.io/spglib/python-spglib.html#get-symmetry for details

        Parameters:
        -----------
        save_directory : str
            save directory in string format, file will be name symmetry.pickle (default = None --> symmetry is not saved)
        """
        import spglib
        import pickle
        import os
        
        if np.count_nonzero(self.lattice_vectors) == 0:
            print('Lattice vectors must not be 0! getSymmetry requires a unitcell-like geometry file!')
            raise ValueError(self.lattice_vectors)
        
        unit_cell = self.getSPGlibCell()
        symmetry = spglib.get_symmetry(unit_cell, symprec=symmetry_precision)
        
        if not save_directory == None:
            if not os.path.exists(save_directory):
                print('symmetry not saved; save_directory does not exist')
            else:
                save_directory = os.path.join(save_directory, 'symmetry.pickle')
                pickle.dump( symmetry, open( save_directory, "wb" ),  protocol=pickle.HIGHEST_PROTOCOL)
        
        return symmetry
    
    def getLargestAtomDistance(self,dims_to_consider = (0,1,2)):
        """
        find largest distance between atoms in geometry
        
        dims_to_consider: dimensions along which largest distance should be calculated

        #search tags; molecule length, maximum size
        """
        mask = np.array([i in dims_to_consider for i in range(3)],dtype = bool)
        geometry_size = 0
        for ind1 in range(self.n_atoms):
            for ind2 in range(ind1, self.n_atoms):
                geometry_size_test = np.linalg.norm(self.coords[ind1][mask] - self.coords[ind2][mask])
                
                if geometry_size_test > geometry_size:
                    geometry_size = geometry_size_test
                    
        return geometry_size

    def getDistanceList(self):
        """
        purpose: Return list of distances between atoms
        including distances to adjacent unit cells
        currently only returns distances between carbon atoms (denoted C, C1, or C2 in geometry.in)
        usage: 
              MyDistances=[]
              MyDistances=gf.getDistanceList()
        """
        
        #Note: In principle, it would be possible to exclude the interaction of an moleucle with itself
        #(via its species designation). However, that would require assumptions about the entry 
        # (e.g. that the molecule is actually connected, even if it goes beyond the borders of the unit cell)
        #These cannot be easily asserted in this function, therefore I opted not to implement this
        # functionality. Feel free to change. 
        
        
        AllowedSpecies=('C','C1','C2') #only compute distances between the listed species        
        DistanceCutOff=25 #Same unit as input. Do not list distances which are further than that. 
        debug_info=False
        
        DistanceSet=[]


        for OriginAtom in range (0,self.n_atoms):   #Loop over all atoms
            if (self.species[OriginAtom] not in AllowedSpecies  ): continue
            OriginCoord=self.coords[OriginAtom]
            
            for TargetAtom in range(OriginAtom,self.n_atoms): #Loop over all other atoms in the list.
                if (self.species[TargetAtom] not in AllowedSpecies  ): continue
                
                for Lat1 in range(-5,6): #Loop over all unit cells defined by lattice 1
                    for Lat2 in range(-5,6): #Loop over all unit cells defined by lattice 2
                        if (OriginAtom==TargetAtom and Lat1==0 and Lat2==0): continue #No distance between itself
                        
                        TargetCoord=self.coords[TargetAtom]+Lat1*self.lattice_vectors[0,:]+Lat2*self.lattice_vectors[1,:]
                        ThisDistance = ( ((OriginCoord[0]-TargetCoord[0])**2)+((OriginCoord[1]-TargetCoord[1])**2)+((OriginCoord[2]-TargetCoord[2])**2) )**0.5
                        if (ThisDistance <= DistanceCutOff):
                            DistanceSet.append(ThisDistance)

                        if (debug_info):
                            print("Origin: ", OriginAtom, OriginCoord)
                            print("Target: ", TargetAtom, TargetCoord)
                            print ("Supercell ", Lat1, Lat2)
                            print("Distance: ", ThisDistance)
                            
        print ("Loop finished")
        print (len(DistanceSet), " entries found")
        return DistanceSet


    def getPrincipalMomentsOfInertia(self):
        """
        Calculates the eigenvalues of the moments of inertia matrix

        Returns
        -------
        principal moments of inertia in kg * m**2:   numpy.ndarray, shape=(3,), dtype=np.float64
        """
        masses_kg = [Units.ATOMIC_MASS_IN_KG * ATOMIC_MASSES[PERIODIC_TABLE[s]] for s in self.species]

        center_of_mass = self.getCenterOfMass()
        r_to_center_in_m = Units.ANGSTROM_IN_METER * (self.coords - center_of_mass)

        ###########
        # begin: code based on ase/atoms.py: get_moments_of_inertia
        # (GNU Lesser General Public License)
        # Initialize elements of the inertial tensor
        I11 = I22 = I33 = I12 = I13 = I23 = 0.0
        for i in range(len(self)):
            x, y, z = r_to_center_in_m[i]
            m = masses_kg[i]

            I11 += m * (y ** 2 + z ** 2)
            I22 += m * (x ** 2 + z ** 2)
            I33 += m * (x ** 2 + y ** 2)
            I12 += -m * x * y
            I13 += -m * x * z
            I23 += -m * y * z

        I = np.array([[I11, I12, I13],
                      [I12, I22, I23],
                      [I13, I23, I33]], dtype=np.float64)

        evals, evecs = np.linalg.eigh(I)
        return evals


###############################################################################
#                        ControlFile Helpers                                  #
###############################################################################
    def getKPoints(self, k_per_atom, slab_geom,get_as_array=False):
        """ Calculate K-Point density for a supercell, given the 
        primitive unit cell (slab_geom) and the K-Point density per primitive unit cell
        (k_per_atom).

        Args:
        -----
        k_per_atom: np.array
            k-grid densities for primitive UC (for x,y and z dimension)
        slab_geom: GeometryFile
            geometry file of the primitve slab.
        get_as_array: bool
            Flag whether to return the result as np.array or list


        Returns:
        --------
        list or np.array (shape=(3,)) containing the kgrid density for x,y and z 
        dimension.
         """
        len_vprim_1 = np.linalg.norm(slab_geom.lattice_vectors[0, :])
        len_vprim_2 = np.linalg.norm(slab_geom.lattice_vectors[1, :])
        len_vprim_3 = np.linalg.norm(slab_geom.lattice_vectors[2, :])
        if (len_vprim_1 - len_vprim_2) > 0.1:
            Warning('k-points per Atom might be a unsuitable unit because of ill-shaped primitive unit cell')

        len_prim = np.min([len_vprim_1, len_vprim_2, len_vprim_3])
        k_per_ang = k_per_atom * len_prim  # number of k-points per angstrom?

        k_1 = np.ceil(k_per_ang / np.linalg.norm(self.lattice_vectors[0, :]))
        k_2 = np.ceil(k_per_ang / np.linalg.norm(self.lattice_vectors[1, :]))
        k_3 = np.ceil(k_per_ang / np.linalg.norm(self.lattice_vectors[2, :]))
        k_str = '{} {} {}'.format(int(k_1), int(k_2), int(k_3))
        if get_as_array:
            return np.array([k_1, k_2, k_3])
        else:
            return k_str


    def splitForAtomProjectedDOSByIndex(self, groups):
        """
            split species of this geometry file according to passed list of integers

        :param groups:  iterable of integers:
                same length as self.species!
                if there are atoms of the same species in different groups, they will be distinguished by a counter
                groups are defined by identical integers in this list
                0 is used as wildcard, i.e. name of these species won't be changed!
        :return:
            species_name_dict       dict(species_old:[list of new species names]), to be passed to control file
        """
        assert len(groups) == len(self.species), "groups must have same length as self.species"
        assert max(groups) <= 1000, "Implement longer zero padding in current_species_new"

        species_name_dict = {s: [] for s in set(self.species)}
        new_species = []
        for current_species, group in zip(self.species, groups):
            if group == 0:
                # use 0 as wildcard -> don't change those species names where group == 0
                current_species_new = current_species
            else:
                current_species_new = "{}_{:03}".format(current_species, group)
                species_name_dict[current_species].append(current_species_new)
            new_species.append(current_species_new)
        self.species = new_species

        species_name_dict = {s_old: sorted(list(set(s_new))) for s_old, s_new in species_name_dict.items()}
        return species_name_dict
    
        
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
        self._homogeneous_field = None
        self.symmetry_params = None
        self.n_symmetry_params = None
        self.symmetry_LVs = None # symmetry_LVs should have str values, not float, to allow for the inclusion of the parameters
        symmetry_LVs_lines = []
        self.symmetry_frac_coords = None  # symmetry_frac_coords should have str values, not float, to allow for the inclusion of the parameters
        symmetry_frac_lines = []
        lattice_vector_lines = []
        atom_line_ind = []
        hessian_lines = []
        text_lines = text.split('\n')

        for ind_line, line in enumerate(text_lines):
            line = line.strip() # Remove leading and trailing space in line
            # Comment in input file
            if line.startswith('#'):
                if 'DFT_ENERGY ' in line:
                    self.DFT_energy = float(line.split()[2])
                elif 'ADSORPTION_ENERGY ' in line:
                    self.E_ads = float(line.split()[2])
                elif 'ADSORPTION_ENERGY_UNRELAXED ' in line:
                    self.E_ads_sp = float(line.split()[2])
                elif 'CENTER' in line:
                    self.center = ast.literal_eval(' '.join(line.split()[2:]))
                # check if it is an own Hessian and not from a geometry optimization
                elif 'own_hessian' in line:
                    is_own_hessian = True
                
                # PARTS defines parts of the geometry that can later on be treated separately.
                # intended for distinction between different molecules and substrate
                elif 'PARTS' in line:
                    part_definition = ast.literal_eval(' '.join(line.split()[2:]))
                    if isinstance(part_definition,dict):
                        for k,v in part_definition.items():
                            self.geometry_part_descriptions.append(k)
                            self.geometry_parts.append(v)
                    elif isinstance(part_definition,list):
                        if isinstance(part_definition[0],list):
                            for part in part_definition:
                                self.geometry_part_descriptions.append('')
                                self.geometry_parts.append(part)
                        else:
                            self.geometry_parts.append(part)
                            self.geometry_part_descriptions.append('')

                else:
                    # Remove '#' at beginning of line, then remove any leading whitespace
                    line_comment = line[1:].lstrip()
                    # Finally add line comment to self.comment_lines
                    self.comment_lines.append(line_comment)

            else:
                # Extract all lines that define atoms, lattice vectors, multipoles or the Hessian matrix
                if 'atom' in line:
                    atom_lines.append(line)
                    atom_line_ind.append(ind_line)
                if 'lattice_vector' in line:
                    lattice_vector_lines.append(line)
                # c Check for fractional coordinates
                if '_frac' in line:
                    is_fractional = True
                if 'hessian_block' in line:
                    hessian_lines.append(line)
                if 'trust_radius' in line:
                    self.trust_radius = float(line.split()[-1])
                if 'set_vacuum_level' in line:
                    self.vacuum_level = float(line.split()[1])
                if 'multipole' in line:
                    multipole = [float(x) for x in list(line.split())[1:]]
                    assert len(multipole) == 5
                    self.multipoles.append(multipole)
                # extract lines concerning symmetry params
                if 'symmetry_n_params' in line:
                    self.n_symmetry_params = [int(x) for x in list(line.split())[1:]]
                if 'symmetry_params' in line:
                    self.symmetry_params = list(line.split())[1:]
                if 'symmetry_lv' in line:
                    symmetry_LVs_lines.append(line)
                if 'symmetry_frac' in line:
                    symmetry_frac_lines.append(line)
                if 'homogeneous_field' in line:
                    self._homogeneous_field = \
                        np.asarray(list(map(float, line.split()[1:4])))
                    
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
                if not line.startswith('#'):
                    if 'initial_moment' in line:
                        moment = float(line.split()[1])
                    elif 'initial_charge' in line:
                        charge = float(line.split()[1])
                    elif 'constrain_relaxation' in line:
                        directions = line.split('constrain_relaxation')[1].lower()
                        if '.true.' in directions:
                            constraints = [True, True, True]
                        if 'x' in directions:
                            constraints[0] = True
                        if 'y' in directions:
                            constraints[1] = True
                        if 'z' in directions:
                            constraints[2] = True
                    elif 'external_force' in line:
                        external_force[0] = float(line.split()[1])
                        external_force[1] = float(line.split()[2])
                        external_force[2] = float(line.split()[3])
                    elif 'calculate_friction' in line:
                        if '.true.' in line:
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

        #store symmetry_lv and symmetry_frac
        if len(symmetry_LVs_lines) != 0:
            self.symmetry_LVs=[]
            if len(symmetry_LVs_lines) != 3:
                print("Warning: Number of symmetry_LVs is: " + str(len(symmetry_LVs_lines)))
            for i, l in enumerate(symmetry_LVs_lines):
                l = l[11:]
                terms = [t.strip() for t in l.split(',')]
                self.symmetry_LVs.append(terms)
        if len(symmetry_frac_lines) != 0:
            self.symmetry_frac_coords=[]
            for i, l in enumerate(symmetry_frac_lines):
                l = l[13:]
                terms = [t.strip() for t in l.split(',')]
                # self.species.append(tokens[-1])
                self.symmetry_frac_coords.append(terms)

        #read the hessian matrix if it is an own Hessian
        if is_own_hessian:
            # hessian has three coordinates for every atom
            self.hessian = np.zeros([self.n_atoms*3,self.n_atoms*3])
            for i,l in enumerate(hessian_lines):
                tokens = l.split()
                ind_1 = int(tokens[1])
                ind_2 = int(tokens[2])
                value_line = np.array([float(x) for x in tokens[3:12]])
                self.hessian[(ind_1-1)*3:ind_1*3, (ind_2-1)*3:ind_2*3] = value_line.reshape((3,3))
            #self.hessian += np.tril(self.hessian.T, -1) # make symmetric hessian matrix
            
        if len(lattice_vector_lines) != 3 and len(lattice_vector_lines) != 0:
            print("Warning: Number of lattice vectors is: " + str(len(lattice_vector_lines)))
        for i, l in enumerate(lattice_vector_lines):
            tokens = l.split()
            self.lattice_vectors[i, :] = [float(x) for x in tokens[1:4]]

        # convert to cartesian coordinates
        if is_fractional:
            self.coords = utils.get_cartesian_coords(self.coords, self.lattice_vectors)
            self.readAsFractionalCoords=True

        self.constrain_relax = np.array(self.constrain_relax)
        self.external_force = np.array(self.external_force)
        self.calculate_friction = np.array(self.calculate_friction)


        # update Part list and add all atoms that are not yet in the list
        if len(self.geometry_parts)>0:
            already_indexed =list(itertools.chain.from_iterable(self.geometry_parts))
            if len(already_indexed) < self.n_atoms:
                additional_indices = [i for i in range(self.n_atoms) if i not in already_indexed]
                self.geometry_parts.append(additional_indices)
                self.geometry_part_descriptions.append('rest')
    
    
    def get_text(self, is_fractional=None):
        """
        If symmetry_params are to be used, the coordinates need to be fractional.
        So, if symmetry_params are found, is_fractional is overridden to true.
        """
        if is_fractional is None:

            if hasattr(self,"symmetry_params") and self.symmetry_params is not None:
                is_fractional = True
            else:
                is_fractional = False
        elif is_fractional == False:
            if hasattr(self,"symmetry_params") and self.symmetry_params is not None:
                warnings.warn("The symmetry parameters of your geometry will be lost. "
                                "To keep them set is_fractional to True")


        text = ""
        for l in self.comment_lines:
            if l.startswith('#'):
                text += l + '\n'
            else:
                text += '# ' + l.lstrip() + '\n' # str.lstrip() removes leading whitespace in comment line 'l'

        # If set, write 'center' dict ( see docstring of Geometry.__init__ ) to file
        if hasattr(self, 'center') and isinstance(self.center, dict):
            center_string = "# CENTER " + str(self.center)
            text += center_string + '\n'

        if hasattr(self,'geometry_parts') and (len(self.geometry_parts)>0):
            part_string = '# PARTS '
            part_dict = {}
            for part,name in zip(self.geometry_parts,self.geometry_part_descriptions):
                if not name == 'rest':
                    if name not in part_dict:
                        part_dict[name] = part
                    else:
                        warnings.warn('Multiple equally named parts in file, renaming automatically!')
                        part_dict[name+'_1'] = part
            part_string += str(part_dict) +'\n'
            text += part_string
                    
        if hasattr(self,'vacuum_level') and (self.vacuum_level is not None):
            text += 'set_vacuum_level {: 15.10f}'.format(self.vacuum_level) + '\n'
        
        # Lattice vector relaxation constraints
        constrain_vectors = np.zeros([3, 3], dtype=bool)
        #if is_2D:
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
        if self.homogenous_field is not None:
            text += "homogeneous_field {} {} {}\n".format(*self.homogenous_field)

        if is_fractional:
            coords = utils.get_fractional_coords(self.coords, self.lattice_vectors)
            line_start = "atom_frac"
        else:
            coords = self.coords
            line_start = "atom"

        for n in range(self.n_atoms):
            if self.species[n] == 'Em':  # do not save "Emptium" atoms
                warnings.warn("Emptium atom was removed!!")
                continue
            line = line_start
            for j in range(3):
                line += "     {:.8f}".format(coords[n, j])
            line += " " + self.species[n]
            text += line + "\n"
            # backwards compatibilty for old-style constrain_relax
            if type(self.constrain_relax[n]) == bool:
                if self.constrain_relax[n]:
                    text += 'constrain_relaxation .true.\n'
            else:
                if all(self.constrain_relax[n]):
                    text += 'constrain_relaxation .true.\n'
                else:
                    if self.constrain_relax[n][0]:
                        text += 'constrain_relaxation x\n'
                    if self.constrain_relax[n][1]:
                        text += 'constrain_relaxation y\n'
                    if self.constrain_relax[n][2]:
                        text += 'constrain_relaxation z\n'
            if self.initial_charge[n] != 0.0:
                text += 'initial_charge {: .6f}\n'.format(self.initial_charge[n])
            if self.initial_moment[n] != 0.0:
                text += 'initial_moment {: .6f}\n'.format(self.initial_moment[n])
            if hasattr(self,'external_force') and np.linalg.norm(self.external_force[n]) != 0.0:
                text += 'external_force {: .6f} {: .6f} {: .6f}\n'.format(self.external_force[n][0],
                                                                          self.external_force[n][1],
                                                                          self.external_force[n][2])
            if hasattr(self, 'calculate_friction') and self.calculate_friction[n]:
                text += 'calculate_friction .true.\n'
                
        if hasattr(self, 'hessian') and self.hessian is not None:
            text += '# own_hessian\n# This is a self calculated Hessian, not from a geometry optimization!\n'
            for i in range(self.n_atoms):
                for j in range(self.n_atoms):
                    s = "hessian_block  {} {}".format(i+1, j+1)
                    H_block = self.hessian[3*i:3*(i+1), 3*j:3*(j+1)]
                    # H_block = H_block.T #TODO: yes/no/maybe? tested: does not seem to make a large difference^^
                    # max_diff = np.max(np.abs(H_block-H_block.T))
                    # print("Max diff in H: {:.3f}".format(max_diff))
                    for h in H_block.flatten():
                        s += "  {:.6f}".format(h)
                    text += s + "\n"

        # write down symmetry_params and related data
        if is_fractional:
            if self.symmetry_params is not None:
                l = 'symmetry_params '
                for p in self.symmetry_params:
                    l += '{} '.format(p)
                l += '\n'
                text +='\n' + l
            if self.n_symmetry_params is not None:
                l = 'symmetry_n_params '
                for n in self.n_symmetry_params:
                    l += '{} '.format(n)
                text += l + '\n'
                text += '\n'
            if self.symmetry_LVs is not None:
                for i in range(3):
                    line = "symmetry_lv     {}  ,  {}  ,  {}".format(*self.symmetry_LVs[i])
                    text += line + "\n"
                text+="\n"
            if self.symmetry_frac_coords is not None:
                for c in self.symmetry_frac_coords:
                    line = "symmetry_frac     {}  ,  {}  ,  {}".format(*c)
                    text += line +"\n"
                text += '\n'

        # write down multipoles
        for m in self.multipoles:
            text+='multipole {}   {}   {}   {}   {}\n'.format(*m)
        return text
    
    
class VaspGeometry(Geometry):
    def parse(self, text):
        """ Read the VASP structure definition in the typical POSCAR format 
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
        self.constrain_relax = []
        lino = 0
        vecs = []
        scaling = 1.0
        typenames = []
        nelements = []
        cartesian = False
        selective = False
        selectivevals = []
        P = []
        fi = text.split('\n')
        
        for line in fi:
            lino += 1
            line = line.strip()

            if lino == 1:
                self.add_top_comment(line)
            if lino == 2:
                scaling = float(line)
                # RB: now the scaling should be taken account for below when the lattice vectors and coordinates
                #if scaling != 1.0:
                #    print("WARNING (readin_struct): universal scaling factor is not one. This is ignored.")
                
            if lino in (3, 4, 5):
                vecs.append(list(map(float, line.split())))
            if lino == 6:
                if line[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    lino += 1
                else:
                    typenames = line.split()
            if lino == 7:
                splitline = line.split()
                nelements = list(map(int, splitline))
                elementid = np.cumsum(np.array(nelements))
                self.n_atoms = elementid[-1]
            if lino == 8:
                if line[0] in ('S', 's'):
                    selective = True
                else:
                    lino += 1
            if lino == 9:
                if line[0] in ('K', 'k', 'C', 'c'):  # cartesian coordinates
                    cartesian = True
            if lino >= 10:
                if lino >= 10 + self.n_atoms:
                    break
                P.append(list(map(float, line.split()[0:3])))

                if selective:
                    # TODO: experimental...
                    constraints = list(map(lambda x: x in ('F', 'f'), line.split()[3:6]))
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
                
                self.external_force = np.append(self.external_force, np.atleast_2d(np.zeros(3)), axis=0)
                self.calculate_friction = np.append(self.calculate_friction, np.array([False]))
                    
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


    def get_text(self, comment='POSCAR file written by Geometry.py'):
        comment = comment.replace('\n', ' ')
        text = comment + '\n'
        text += '1\n'
        if (self.lattice_vectors != 0).any():
            for i in range(3):
                line = ""
                for j in range(3):
                    line += "     {:-4.8f}".format(self.lattice_vectors[i, j])
                text += line.strip() + "\n"
    
        all_species = sorted(list(set(self.species))) # get unique species and sort alphabetically
        text += ' '.join(all_species) + '\n'
        species_coords = {}
        n_of_species = {}
        # R.B. relax constraints
        relax_constraints = {}
        ## R.B. relax constraints end
                    
        for species in all_species:
            is_right_species = np.array([s == species for s in self.species],dtype=bool)
            curr_species_coords = self.coords[is_right_species, :]
            species_coords[species] = curr_species_coords
            n_of_species[species] = curr_species_coords.shape[0]
            
            # R.B. relax constraints
            curr_species_constrain_relax = self.constrain_relax[is_right_species, :]
            relax_constraints[species] = curr_species_constrain_relax
            ## R.B. relax constraints end
            
    
        # add number of atoms per species
        text += ' '.join([str(n_of_species[s]) for s in all_species]) + '\n'
    
        # R.B. Write out selective dynamics so that the relaxation constraints are read
        text += 'Selective dynamics' + '\n'
        
        text += 'Cartesian' + '\n'
        
    
        for species in all_species:
            curr_coords = species_coords[species]
            n_atoms = n_of_species[species]
            
            ## R.B. relax constraints
            curr_relax_constr = relax_constraints[species]
            ## R.B. relax constraints end
            
            for n in range(n_atoms):
                line = ""
                for j in range(3):
                    if j>0:
                        line += '    '
                    line += "{: 2.8f}".format(curr_coords[n, j])
                    
                    
             ## R.B. relax constraints
                for j in range(3):
                    if curr_relax_constr[n,j] == True:
                        line += '  ' + 'F'
                    elif curr_relax_constr[n,j] == False:
                        line += '  ' + 'T'
             ## R.B. relax constraints end
    
                text += line+ '\n'
    
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
        species = []
        fi = text.split('\n')

        # parse will assume first few lines are comments
        started_parsing_atoms = False

        for ind, line in enumerate(fi):
            if ind == 0:
                if len(line.split())==1:
                    read_natoms = int(line.split()[0])
                    continue
                
            # look for lattice vectors
            if 'Lattice' in line:
                split_line = line.split('Lattice')[1]
                
                lattice_parameters = re.findall("\d+\.\d+", split_line)
                
                if len(lattice_parameters) == 9:
                    lattice_parameters = np.array(lattice_parameters, dtype=np.float64)
                    self.lattice_vectors = np.reshape(lattice_parameters, (3,3))
            
            n_words = len( re.findall('[a-zA-Z]+', line) )
            n_floats = len( re.findall('\d+\.\d+', line) )
            split_line = line.split()
            
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
                        "Bad atoms specification: " + str(split_line)
                
                # write atoms
                coords.append( np.array(split_line[1:4], dtype=np.float64) )
                species.append(str(split_line[0]))
                count_natoms += 1

        if not started_parsing_atoms:
            raise RuntimeError("Not atoms found in xyz file!")     

        if read_natoms != None:
            assert read_natoms == count_natoms, "Not all atoms found!"

        
        coords = np.asarray(coords)
        self.add_atoms(coords,species)


    def get_text(self, comment='XYZ file written by Geometry.py'):
        text = str(self.n_atoms) + '\n'
        comment = comment.replace('\n', ' ')
        text += comment + '\n'
        for index in range(self.n_atoms):
            element = self.species[index]
            x,y,z = self.coords[index]
            text += "{}    {:-4.8f}    {:-4.8f}    {:-4.8f}".format(element, x, y, z) + "\n"
        return text


class MoldenGeometry(Geometry):
    def get_text(self, title=''):
        # Conversion from Angstrom to Bohr, as all lenghts should be in Bohr for Molden
        length_conversion = 1 / Units.BOHR_IN_ANGSTROM

        # First line
        text = '[Molden Format]\n'

        # geometry coordinates in xyz format
        text_xyz = '[GEOMETRIES] XYZ\n'
        text_xyz += '{0:6}\n'.format(self.n_atoms)
        text_xyz += title + '\n'
        for i_atom, species in enumerate(self.species):
            text_xyz += '{0:6}'.format(species)
            for i_coord in range(3):
                text_xyz += '{0:10.4f}'.format(self.coords[i_atom, i_coord]*length_conversion)
                text_xyz += '\n'
        text += text_xyz + '\n'

        # add eigenmodes and frequencies if they exist
        if hasattr(self, 'hessian') and self.hessian is not None:
            frequencies, displacement_coords = self.getEigenvaluesAndEigenvectors(
                bool_symmetrize_hessian=True,
                bool_only_real=False
            )
            print("INFO: Eigenfrequencies and -modes are calculated after "
                  "symmetrizing the hessian")

            # first add frequencies
            text_freq = '[FREQ]\n'
            for freq in frequencies:
                text_freq += '{0:10.3f}\n'.format(freq)
            text_freq += '\n'

            # next come the infrared intensities
            # TODO: we cant calculate infrared intensities at the moment
            text_freq += '[INT]\n'
            for i in range(len(frequencies)):
                text_freq += '{0:17.6e}\n'.format(1)
            text_freq += '\n'

            # then again coordinates for all atoms
            text_freq += '[FR-COORD]\n'
            for i_atom, species in enumerate(self.species):
                text_freq += '{0:6}'.format(species)
                for i_coord in range(3):
                    text_freq += '{0:10.4f}'.format(self.coords[i_atom, i_coord]*length_conversion)
                text_freq += '\n'
            text += text_freq

            # finally add displacements for all atoms for each vibration
            text_dist = '[FR-NORM-COORD]\n'
            for i, (freq, displacements) in enumerate(zip(frequencies, displacement_coords)):
                text_dist += 'vibration {0:6}\n'.format(i + 1)
                for l in range(len(displacements)):
                    for d in range(3):
                        text_dist += '{0:10.4f}'.format(np.real(displacements[l, d])*length_conversion)
                    text_dist += '\n'
            text += text_dist
        return text


class XSFGeometry(Geometry):
    def get_text(self):
        text = ""
        text += 'CRYSTAL\n'
        text += 'PRIMVEC\n'
        for i in range(3):
            line = ""
            for j in range(3):
                line += "    {:.8f}".format(self.lattice_vectors[i, j])
            text += line + "\n"
        text += 'PRIMCOORD\n'
        # the 1 is mysterious but is needed for primcoord according to XSF docu
        text += str(self.n_atoms) + ' 1\n'
        for i in range(self.n_atoms):
            if self.constrain_relax[i]:
                raise NotImplementedError('Constrained relaxation not supported for XSF output file')
            line = str(PERIODIC_TABLE[self.species[i]])
            for j in range(3):
                line += '    {:.8f}'.format(self.coords[i, j])
            text += line + '\n'
        return text
                             

if __name__ == '__main__':
    pass

