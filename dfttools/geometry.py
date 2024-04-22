import itertools
from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from typing import Union
import copy
import warnings
import ast

import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance
import networkx as nx
#import spglib # import in corresponding functions instead as spglib can cause trouble sometimes

import os.path

from aimstools.GaussianInput import GaussianInput

from aimstools.Utilities import PERIODIC_TABLE, getCartesianCoords, \
        getFractionalCoords, SLATER_EFFECTIVE_CHARGES, ATOMIC_MASSES, \
        COVALENT_RADII, getSpeciesColor, C6_COEFFICIENTS, R0_COEFFICIENTS, \
        getCovalentRadius, getAtomicNumber
from aimstools import Units
from aimstools import Utilities as ut
from aimstools.ControlFile import CubeFileSettings
import aimstools.ZMatrixUtils as ZMatrixUtils



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
    """This class represents a geometry file for (in principle) any DFT code.
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
        self.hessian = None
        self.geometry_parts = []                # list of lists: indices of each geometry part
        self.geometry_part_descriptions = []    # list of strings:  name of each geometry part
        self.symmetry_axes = None
        self.inversion_index = None
        self.vacuum_level = None
        self._multipoles = []
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
            self.readFromFile(filename, file_format)

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

    def __add__(self, other):
        geom = copy.deepcopy(self)
        geom += other
        return geom

    def __iadd__(self, other):
        self.add_geometry(other)
        return self

    @property
    def multipoles(self):
        if not hasattr(self, "_multipoles"):
            self.multipoles = []
        return self._multipoles

    @multipoles.setter
    def multipoles(self, new_multipoles):
        self._multipoles = new_multipoles

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


    def is_periodic(self):
        return not np.allclose(self.lattice_vectors, np.zeros([3, 3]))


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


    def add_multipoles(self,multipoles):
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


    def getFromASEAtomsObject(self,
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
        species=[]
        for i, atom in enumerate(atoms):
            specie = atom.symbol
            if isinstance(specie,int):
                specie = PERIODIC_TABLE[specie]
            species.append(specie)
            coords.append(atom.position)
        coords = np.array(coords)
        self.add_atoms(coords,species,constrain_relax)

    ###############################################################################
    #                             INPUT PARSER                                    #
    ###############################################################################
    def parse_geometry(self, text):
        raise NotImplementedError


    


    def parseTextVASP(self, text):
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
                self.addTopComment(line)
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

    def readFromXYZFile(self, file_name):
        """Read from .xyz file."""
        with open(file_name, "r") as f:
            self.parseTextXYZ(f.read())

    def parseTextXYZ(self, text):
        """Reads a .xyz file. Designed to work with .xyz files produced by Avogadro"""

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

        for ind,line in enumerate(fi):
            if ind == 0:
                if len(line.split())==1:
                    read_natoms = int(line.split()[0])
                    continue
            split_line = line.split()

            # first few lines may be comments or properties
            if not started_parsing_atoms:
                
                if len(split_line) != 4:
                    continue
                else:
                    started_parsing_atoms = True
                           
            else:
                
                if split_line == []:
                    # finished
                    break
                else:

                    # now all lines must have 4 entries
                    assert len(split_line) == 4, "Bad atoms specification: " + str(split_line)


            
            #--- parse atom ---
            specie,x,y,z = split_line
            coords.append([float(x),float(y),float(z)])
            species.append(str(specie))
            count_natoms += 1
            #--

        if not started_parsing_atoms:
            raise RuntimeError("Not atoms found in xyz file!")     

        if read_natoms != None:
            assert read_natoms == count_natoms, "Not all atoms found!"

        
        coords = np.asarray(coords)
        self.add_atoms(coords,species)

    def addTopComment(self, comment_string):
        """Adds comments that are saved at the top of the geometry file."""
        lines = comment_string.split('\n')
        for l in lines:
            if not l.startswith('#'):
                l = '# ' + l
            self.comment_lines.append(l)

###############################################################################
#                             OUTPUT PARSER                                   #
###############################################################################
    def saveToFile(self, filename, file_format=None, isFractional=None, constrain_2D = False):
        if file_format is None:
            file_format = getFileFormatFromEnding(filename) # Returns None if it encounters an unknown file format and a string otherwise

        if isinstance(file_format, str):
            file_format = file_format.lower()

        if file_format == 'aims':
            text = self.getTextAIMS(isFractional)
        elif file_format == 'xsf':
            text = self.getTextXSF()
        elif file_format == 'gaussian':
            text = self.getTextGaussian()
        elif file_format == 'molden':
            text = self.getTextMolden()
        elif file_format == 'vasp':
            assert self.isPeriodic(), 'vasp files must be periodic. The present Geometry is not periodic.'
            text = self.getTextVASP()
        elif file_format == 'xyz':
            text = self.getTextXYZ()
        elif file_format == 'z_matrix':
            text = self.getTextZMatrix()
        else:
            handleUnknownFileFormat(filename, file_format) # raises NotImplementedError

        self.is_2D = constrain_2D

        # Enforce linux file ending, even if running on windows machine by using binary mode
        with open(filename, 'w', newline='\n') as f:
            f.write(text)

    def getTextZMatrix(self, distance_variables=False, angle_variables=False, dihedral_variables=False):
        """
        get ZMatrix representation of current geometry.

        Parameters:
        distance_variables: bool
            If true, define variables for the atom distances
        angle_variables : bool
            If true, define variables for the angles
        dihedral_variables:
            If true, define variables for the dihedral angles
        """

        return ZMatrixUtils.getTextZMatrix(self.coords, self.species, rvar=distance_variables,
                                                  avar=angle_variables, dvar=dihedral_variables)

    def getTextXSF(self):
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

    def getTextMolden(self, title=''):
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


    def getTextAIMS(self, isFractional=None, is_2D=False):
        # if symmetry_params are to be used, the coordinates need to be fractional. So, if symmetry_params are found, isFractional is overridden to true.
        if isFractional is None:

            if hasattr(self,"symmetry_params") and self.symmetry_params is not None:
                isFractional = True
            else:
                isFractional = False
        elif isFractional == False:
            if hasattr(self,"symmetry_params") and self.symmetry_params is not None:
                warnings.warn("The symmetry parameters of your geometry will be lost. "
                                "To keep them set isFractional to True")


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
        if is_2D:
            constrain_vectors[0, 2], constrain_vectors[1, 2], constrain_vectors[2] = True, True, 3*[True]

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
        if not self.homogenousField is None:
            text += "homogeneous_field {} {} {}\n".format(*self.homogenousField)

        if isFractional:
            coords = getFractionalCoords(self.coords, self.lattice_vectors)
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
        if isFractional:
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

    def getTextVASP(self, comment='POSCAR file written by Geometry.py'):
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



    def getTextGaussian(self,route='', link0='%nproc=1',title='', charge=0, multiplicity=1):
        """Creates Gaussian input for coordinates
            Settings input via string will be added at the top of the document"""

        # there also exists a gaussian input class which could be used
        text = GaussianInput(
            geometry=self,
            route=route,
            link0=link0,
            title=title,
            charge=charge,
            multiplicity=multiplicity
        ).getTextGaussian()

        return text
        

    def getTextXYZ(self, comment='XYZ file written by Geometry.py'):
       text = str(self.n_atoms) + '\n'
       comment = comment.replace('\n', ' ')
       text += comment + '\n'
       for index in range(self.n_atoms):
           element = self.species[index]
           x,y,z = self.coords[index]
           text += "{}    {:-4.8f}    {:-4.8f}    {:-4.8f}".format(element, x, y, z) + "\n"
       return text


###############################################################################
#                             Transformation                                  #
###############################################################################
    def moveToFirstUnitCell(self, lattice=None, coords=np.array(range(3))):
        ''' maps all atoms into the first unit cell'''
        if lattice is None:
            lattice = self.lattice_vectors
        frac_coords = ut.getFractionalCoords(self.coords, lattice)
        frac_coords[:,coords] = frac_coords[:,coords] % 1 # modulo 1 maps all coordinates to first unit cell
        new_coords = ut.getCartesianCoords(frac_coords, lattice)
        self.coords = new_coords



    def moveCenterOfAtomsToFirstUnitCell(self, lattice=None):
        """Shift the center of the structure to the first unit cell"""
        if lattice is None:
            lattice = self.lattice_vectors
        offset = self.getGeometricCenter()
        frac_offset = getFractionalCoords(offset, lattice)
        frac_offset = np.floor(frac_offset)
        self.moveByFractionalCoords(-frac_offset, lattice)

    def mapToOrigin(self, lattice):
        """
        maps the x and y coordinate of a geometry in multiples ot the substrate lattice vectors
        to a point that is closest to the origin
        
        lattice : float-array
            lattice vectors of the substrate
        """
        centre = self.getGeometricCenter() # Geometry.center attribute should be used, if defined
        frac_centre = getFractionalCoords(centre, lattice)
        frac_centre[:2] = frac_centre[:2] % 1.0
        frac_candidates = frac_centre + np.array([[0,0,0],
                                                  [-1,0,0],
                                                  [0,-1,0],
                                                  [-1,-1,0]])
        cartesian_candidates = getCartesianCoords(frac_candidates, lattice)
        distance_from_origin = np.linalg.norm(cartesian_candidates,axis=1)
        ind_best = np.argmin(distance_from_origin)
        new_center_coords = cartesian_candidates[ind_best,:]
        
        self.coords -= centre
        self.coords += new_center_coords

    def mapToFirstUnitCell(self, lattice_vectors=None, clean_borders=False):
        """
        Maps the coordinates of the atoms into the first unit cell"""
        
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors
        
        new_coords = ut.mapToFirstUnitCell(self.coords, lattice_vectors)
        self.coords = new_coords
    
    def mapAtomsAroundCenter(self):
        """
        This function, for now, is a very dirty fix for VASPs
        mapping of atoms to the other side of the unit cell
        """
        
        frac_coords = ut.getFractionalCoords(self.coords, self.lattice_vectors)
        frac_coords = frac_coords % 1 # modulo 1 maps all to first unit cell
        L = frac_coords > 0.5
        frac_coords[L] -= 1.0
        self.coords = ut.getCartesianCoords(frac_coords, self.lattice_vectors)
    
    def mapCenterOfAtomsClosestToOrigin(self, lattice=None):
        """Shift the center of the structure to the first unit cell"""
        if lattice is None:
            lattice = self.lattice_vectors
            
        offset = self.getGeometricCenter()
        frac_offset = getFractionalCoords(offset, lattice)
        
        frac_offset_floor = np.floor(frac_offset)
        L = (frac_offset - frac_offset_floor) > 0.5
        
        frac_offset[L] += 1.0
        
        frac_offset = np.floor(frac_offset)
        self.moveByFractionalCoords(-frac_offset, lattice)
    
    def getReassembledMolecule(self, threshold=2.0):
        
        geom_replica = self.getPeriodicReplica((1,1,1), explicit_replications=([-1,0,1],[-1,0,1],[-1,0,1]))
        #geom_replica = self.getPeriodicReplica((1,1,1), explicit_replications=([-1,0],[-1,0],[0]))
        
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

    def getScaledCopy(self,scaling_factor):
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

        new_coords = getCartesianCoords(self.getFractionalCoords(),lattice_vectors)
        scaled_geom.lattice_vectors = lattice_vectors
        scaled_geom.coords = new_coords

        return scaled_geom

    def getFractionalCoords(self,lattice_vectors=None):
        if lattice_vectors is None:
            lattice_vectors = self.lattice_vectors
        assert not np.allclose(lattice_vectors,np.zeros([3,3])), 'Lattice vector must be defined in Geometry or given as function parameter'
        
        fractional_coords = np.linalg.solve(lattice_vectors.T, self.coords.T)
        return fractional_coords.T

    def getFractionalLatticeVectors(self, lattice_vectors=None):
        ''' calculate the fractional representation of lattice vectors of the geometry file in another basis.
        Useful to calculate epitaxy matrices'''

        fractional_coords = np.linalg.solve(lattice_vectors.T, self.lattice_vectors.T)
        return fractional_coords.T

    def moveToCenterOfCell(self, lattice=None, molecule_only=False, primitive_substrate=None):
        """
        Shift the center of the structure to the center of the first unit cell plane

        Parameters
        ----------
        lattice
        molecule_only           move only adsorbate molecule
        primitive_substrate     Geometry; only relevant if molecule_only is True;
                                used to distinguish between molecule and substrate
        """
        if lattice is None:
            lattice = self.lattice_vectors
        center_of_cell = (self.lattice_vectors[0, :] + self.lattice_vectors[1, :]) / 2
        if molecule_only:
            mol = self.getAdsorbates(primitive_substrate=primitive_substrate)
            offset = mol.getGeometricCenter()
        else:
            offset = self.getGeometricCenter()

        frac_offset_mol = np.floor(getFractionalCoords(offset, lattice))
        frac_offset_cell = np.round(getFractionalCoords(center_of_cell, lattice))
        # print(frac_offset_mol,frac_offset_cell)
        print_cell = getFractionalCoords(center_of_cell, lattice)
        # print(print_cell[0],print_cell[1],print_cell[2])
        frac_offset = frac_offset_cell - frac_offset_mol
        if molecule_only:
            mol.moveByFractionalCoords(frac_offset, lattice)
            mol_inds = self.getAdsorbateIndices(primitive_substrate=primitive_substrate)
            self.coords[mol_inds, :] = mol.coords
        else:
            self.moveByFractionalCoords(-frac_offset, lattice)

    def moveByFractionalCoords(self, frac_shift, lattice=None):
        if lattice is None:
            lattice = self.lattice_vectors
        self.coords += getCartesianCoords(frac_shift, lattice)

    def move(self, shift):
        """ Translates the whole geometry by vector 'shift'"""
        self.coords += shift
    
    def moveAdsorbates(self, shift, primitive_substrate=None):
        """
        shifts the adsorbates in Cartesian coordinates
        """
        adsorbates = self.getAdsorbates(primitive_substrate=primitive_substrate)
        adsorbates.coords += shift
        
        self.removeAdsorbates(primitive_substrate=primitive_substrate)
        self += adsorbates
    
    def transform_coord_sys_to_new_z_vec(self, new_z_vec, inplace=False):
        """
        transforms the coordinate system of the geometry file to a new z-vector
        calculates rotation martrix for coordinate transformation to new z-vector
        and uses it to transform coordinates of geometry object 
        
        Parameters:
        new_z_vec (numpy.ndarray): The vector to align with the z-axis.
        inplace: BOOLEAN: if TRUE, the geometry file is transformed in place, if FALSE, only the new coords are returned

        Returns:
        new positions (numpy.ndarray): The positions in the new coordinate system.
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
        
        if inplace == True:
            self.coords= rotated_positions

        return rotated_positions
        

    def rotateLatticeAroundZAxis(self, angle_in_degree):
        """Rotates Lattice around z axis"""
        R = ut.getCartesianRotationMatrix(angle_in_degree * np.pi / 180, get_3x3_matrix=True)
        self.lattice_vectors = np.dot(self.lattice_vectors, R)

    def rotateAroundZAxis(self, angle_in_degree, center=None, indices=None):
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

        R = ut.getCartesianRotationMatrix(angle_in_degree * np.pi / 180, get_3x3_matrix=True)
        temp_coords = copy.deepcopy(self.coords[indices])
        temp_coords -= center
        temp_coords = np.dot(temp_coords,R)
        temp_coords += center
        self.coords[indices] = temp_coords

    def tilt(self, theta_deg, tilt_axis, atom_lock=False):
        """ RS
        Rotates the molecule along the axis towards the z-axis, out of the xy-plane.
        Input
            theta_deg: angle to tilt in degrees (0 = flat, 90 = upstanding)
            tilt_axis: [x, y, z] array that defines the axis to tilt (beware of previous rotations!)
            atom_lock: to lock the lowest atoms to the surface or not
        Output
            none: function transforms the Geometry object
        """
        # define the flipping axis and resulting rotation matrix
        theta_rad = theta_deg/180 * np.pi
        flip_axis = -np.array([-tilt_axis[1], tilt_axis[0], 0])
        rot_mat = self.getRotationMatrixAroundAxis(axis=flip_axis, phi=theta_rad)

        # Fix the lowest atoms to the surface
        if atom_lock is True:
            shift = np.array([0, 0, 0])  # TODO: atom locking
            print('Atom locking not available!')
        else:
            shift = np.array([0, 0, 0])

        # Do the transformation
        self.transform(R=rot_mat, t=shift)

    def mirrorThroughPlane(self,normal_vector):
        """
        Mirrors the geometry through the plane defined by the normal vector.
        """
        mirror_matrix = ut.getMirrorMatrix(normal_vector=normal_vector)
        self.transform(mirror_matrix)

    def alignIntoXYPlane(self, atom_indices):
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
        return U

    def centerCoordinates(self, ignore_center_attribute=False):
        """
        Shift the coordinates of a geometry such that the "center of mass" or specified center lies at (0,0,0)
        :param bool ignore_center_attribute: Switch usage of *center* attribute off/on.
        :returns: Offset by which this geometry was shifted
        """
        offset = self.getGeometricCenter(ignore_center_attribute=ignore_center_attribute)
        self.coords -= offset
        return offset

    def centerXYCoordinates(self, ignore_center_attribute=False):
        """
        shift the x and y coordinates of a geometry such that the "surface centre of mass" lies at (0,0)
        !Use centerOnLatticePlane for lattice vectors not in the xy plane!
        :param bool ignore_center_attribute: Switch usage of *center* attribute off/on.
        """
        offset = self.getGeometricCenter(ignore_center_attribute=ignore_center_attribute)[:2]
        self.coords[:,[0,1]] -= offset

    def centerOnLatticePlane(self, ignore_center_attribute=True):
        """
        shift the coordinates of a geometry such that the "surface centre of mass" 
        lies at (0,0) in lattice vector coordinates
        :param bool ignore_center_attribute: Switch usage of *center* attribute off/on.
        """
        frac_coords = self.getFractionalCoords()
        offset = self.getGeometricCenter(ignore_center_attribute=ignore_center_attribute)
        offset[2] = 0
        self.moveByFractionalCoords(-offset)

    def getMainAxes(self, weights='unity'):
        """
        Get main axes and eigenvalues of a molecule
        https://de.wikipedia.org/wiki/Tr%C3%A4gheitstensor


        weights: Specifies how the atoms are weighted
            "unity": all same weight
            "Z": weighted by atomic number
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
        
        return vals[sort_ind],vecs[:,sort_ind]
    
    @ut.deprecated #<jc: there is a duplicate function in aimstools.utilities. Also this has nothing to do with GeomtryFile.
    def getRotationMatrix(self, ax_vec, initial_vec):
        """ Calculates rotation matrix from initial_vec to ax_vec
        """
        initial_vec_normed = initial_vec / scipy.linalg.norm(initial_vec)
        ax_vec_normed = ax_vec / scipy.linalg.norm(ax_vec)
        # v = np.outer(ax_vec, initial_vec_normed)
        v = np.cross(ax_vec_normed, initial_vec_normed)
        c = np.dot(ax_vec_normed, initial_vec_normed)
        vx = np.array([[0,       -v[2],  v[1]],
                      [v[2],    0,      -v[0]],
                      [-v[1],   v[0],      0]])
        
        if (c == -1) or (np.abs((c+1)) < 1e-6):
            R = np.diag([-1, -1, 1])
        elif ( c== 1) or (np.abs(c-1) < 1e-6):
            R = np.eye(3)
        else:
            R = np.eye(3) + vx + vx.dot(vx) /(1+c)
            # If c == -1 (divide by 0 error ) the coordinate system simply has 
            # to be flipped totally, resulting in R == np.diag([-1,-1,-1])  
            
        return R

    @ut.deprecated #<LH: there is a duplicate function in aimstools.utilities. Also this has nothing to do with GeomtryFile.
    def getRotationMatrixAroundAxis(self, axis, phi):
        """

        Parameters
        ----------
        axis 3D-axis
        phi     angle of rotation around axis
        """

        axis_vec = np.array(axis, dtype=np.float64)
        axis_vec /= np.linalg.norm(axis_vec)

        eye = np.eye(3, dtype=np.float64)
        ddt = np.outer(axis_vec, axis_vec)
        skew = np.array([[0,            axis_vec[2],    -axis_vec[1]],
                         [-axis_vec[2], 0,              axis_vec[0]],
                         [axis_vec[1],  -axis_vec[0],   0]],
                        dtype=np.float64)

        R = ddt + np.cos(phi) * (eye - ddt) + np.sin(phi) * skew
        return R
    
    def alignLatticeVectorToVector(self, vector, lattice_vector_index):
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
                
        R = ut.getRotationMatrix(
            vector_normed,
            lattice_vector_normed
        )

        self.lattice_vectors = np.dot(self.lattice_vectors, R)
        self.coords = np.dot(self.coords, R)
    
    def alignMainAxisAlongXYZ(self):
        """
        align coordinates of rodlike molecules along specified axis
        """
        vals, vecs = self.getMainAxes()
        R = np.linalg.inv(vecs.T)
        # print("align")
        # print(np.dot(R, vecs.T))
        self.coords = np.dot(self.coords,R)


    def transform(self, R, t=np.array([0,0,0]),rotation_center = None, atom_indices=None):
        """Transforms the coordinates by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]"""
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
        return self

    def transformLattice(self,R,t=np.array([0,0,0])):
        """Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]
        Notice that this works in cartesian coordinates.
        Use transformFractional if you got your R and t from getSymmetries"""
        new_lattice_vectors = np.dot(self.lattice_vectors, R) + t
        self.lattice_vectors = new_lattice_vectors
        return self

    def transformLatticeFractional(self,R,t,lattice):
        """Transforms the lattice vectors by rotation and translation.
        The transformation is applied as x_new[3x1] = x_old[3x1] x R[3x3] + t[3x1]"""
        coords_frac = getFractionalCoords(self.lattice_vectors, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1,3])
        self.lattice_vectors = getCartesianCoords(coords_frac, lattice)
        return self

    def swapLatticeVectors(self, axis_1=0, axis_2=1):
        """
            Can be used to interchange two lattice vectors
            Attention! Other values - for instance k_grid - will stay unchanged!!
        :param axis_1 integer [0,1,2]
        :param axis_2 integer [0,1,2]     axis_1 !=axis_2
        :return:
        """
        self.lattice_vectors[[axis_1, axis_2], :] = self.lattice_vectors[[axis_2, axis_1], :]
        self.coords[[axis_1, axis_2], :] = self.coords[[axis_2, axis_1], :]
        return self

    def transformFractional(self, R, t, lattice=None):
        """Transforms the coordinates by rotation and translation, where R,t are
        given in fractional coordinates
        The transformation is applied as c_new[3x1] = R[3x3] * c_old[3x1] + t[3x1]"""
        if lattice is None:
            lattice=self.lattice_vectors
        coords_frac = getFractionalCoords(self.coords, lattice)
        coords_frac = np.dot(coords_frac, R.T) + t.reshape([1,3])
        self.coords = getCartesianCoords(coords_frac, lattice)
        return self
    
    def getDistanceToEquivalentAtoms(self, geom):
        """Calculate the maximum distance that atoms of geom would have to be moved,
           to coincide with the atoms of self.
        """
        trans,dist = self.getTransformationIndices(geom, get_distances=True)
        return np.max(dist)

    def getDistanceBetweenAllAtoms(self):
        """Get the distance between all atoms in the current Geometry
        object. Gives an symmetric array where distances between atom i and j
        are denoted in the array elements (ij) and (ji)."""

        ## old version (changed by aj on 2020.12.19)
        # distances = np.zeros((self.n_atoms, self.n_atoms))
        # for i in np.arange(self.n_atoms):
        #     for j in np.arange(i + 1, self.n_atoms):
        #         atom1 = self.coords[i, :]
        #         atom2 = self.coords[j, :]
        #         vec = atom2 - atom1
        #         dist = np.linalg.norm(vec)
        #         distances[i, j] = dist
        #         distances[j, i] = dist

        distances = scipy.spatial.distance.cdist(self.coords, self.coords)
        return distances

    def getClosestAtoms(self, indices, species=None, n_closest=1):
        """
        get the indices of the closest atom(s) for the given index or list of indices

        Parameters
        ----------
        index: int or iterable
            atoms for which the closest indices are  to be found
        species: None or list of species identifiers
            species to consider for closest atoms. This allows to get only the closest atoms of the same or another species
        n_closest: int
            number of closest atoms to return

        Returns
        -------
        closest_atoms_list: list or list of lists
            closest atoms for each entry in index
        """

        all_distances = self.getDistanceBetweenAllAtoms()

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


    def getDistanceBetweenTwoAtoms(self, atom_indices):
        """Get the distance between two atoms in the current Geometry
        object."""
        atom1 = self.coords[atom_indices[0],:]
        atom2 = self.coords[atom_indices[1],:]
        vec = atom2 - atom1
        dist = np.linalg.norm(vec)

        return dist

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

    def symmetrize(self, symmetry_operations, center=None):
        """symmetrizes Geometry with given list of symmetry operation matrices
         after transferring it to the origin.
         Do not include the unity matrix for symmetrizing, as it is already the first geometry!
         ATTENTION: use symmetrize_periodic to reliably symmetrize periodic structures"""
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
            new_geom.reorderAtoms(self.getTransformationIndices(new_geom))
            temp_coords += new_geom.coords
        self.coords = temp_coords / (len(symmetry_operations) + 1) + offset

    def symmetrize_periodic(self,symmetries):
        """reliably symmetrizes a periodic structure on a set of symmetries, as received from getSymmetries()
        Differently from symmetrize(), symmetries MUST include the identity too
        NOTE: I have not tested this function thoroughly; use it with caution. (fc 13.05.2020)"""

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
            new_geom.reorderAtoms(indices)
            # move back to pre-centered position
            new_geom.move(offset2)
            transformed_geoms.append(new_geom)
        # we have all the structures, including the original, in transformed geoms. It can be nice for visualization.
        # average the coordinates
        transformed_coords = np.stack([g.coords for g in transformed_geoms])
        symm_coords = np.mean(transformed_coords,axis=0)
        self.coords = symm_coords

    def average_with(self, other_geometries):
        """
            average self.coords with those of other_geometries and apply on self
            ATTENTION: this can change bond lengths etc.!Ok n
        Parameters
        ----------
        other_geometries: List of Geometrys ... might be nice to accept list of coords too

        Returns
        -------
        works in place (on self)
        """
        if len(other_geometries) > 0:
            offset = self.getGeometricCenter() # Attribute center should be used if it exists
            self.coords -= offset

            for other_geom in other_geometries:
                geom = copy.deepcopy(other_geom)
                # center all other geometries to remove offset
                geom.centerCoordinates()
                # all geometries have to be ordered like first geometry in order to sum them
                geom.reorderAtoms(self.getTransformationIndices(geom))
                self.coords += geom.coords
            self.coords /= (len(other_geometries) + 1)    # +1 for this geometry itself
            self.coords += offset


    def getPrimitiveSlab(self, surface, threshold=1e-6):
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
        
        frac_surface_lattice = getFractionalCoords( surface_lattice, lattice )
        #print(frac_surface_lattice)
        
        slab = Geometry()
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
                            frac_new_coord = getFractionalCoords(new_coord, surface_lattice)
                        
                            L1 = np.sum( frac_new_coord >= 1-threshold )
                            L2 = np.sum( frac_new_coord < -threshold )
                            
                            if not L1 and not L2:
                                slab.add_atoms([new_coord], [PERIODIC_TABLE[new_species]])
                                add_next_shell = True
                        
                        
            if not shell == 0 and not add_next_shell:
                break
            
            if shell == 100:
                warnings.warn('<Geometry.getPrimitiveSlab> could not build a correct slab.')
        
        slab.alignLatticeVectorToVector(np.array([0,0,1]),2)
        slab.alignLatticeVectorToVector(np.array([1,0,0]),0)
        
        scaled_slab_lattice = np.array(slab.lattice_vectors)
        # break symmetry in z-direction
        scaled_slab_lattice[2,:] *= 2
        frac_coords = getFractionalCoords(slab.coords, scaled_slab_lattice)
        species = [PERIODIC_TABLE[s] for s in slab.species]
        
        (primitive_slab_lattice, primitive_slab_scaled_positions, primitive_slab_atomic_numbers) \
        = spglib.find_primitive((scaled_slab_lattice, frac_coords, species), symprec=1e-5)
        
        primitive_slab_species = [PERIODIC_TABLE[s] for s in primitive_slab_atomic_numbers]
        primitive_slab_coords = primitive_slab_scaled_positions.dot(primitive_slab_lattice)
        # replace lattice vector in z-direction
        primitive_slab_lattice[2,:] = slab.lattice_vectors[2,:]
        
        primitive_slab = Geometry()
        primitive_slab.lattice_vectors = primitive_slab_lattice
        primitive_slab.add_atoms(primitive_slab_coords, primitive_slab_species)
        primitive_slab.moveToFirstUnitCell()
        
        # Sanity check: primitive_slab must be reducable to the standard unit cell 
        check_lattice, _, _ = spglib.standardize_cell(primitive_slab.getSPGlibCell())
        
        assert np.allclose( check_lattice, lattice ), \
        '<Geometry.getPrimitiveSlab> the slab that was constructed \
        could not be reduced to the original bulk unit cell. Something \
        must have gone wrong.'
        
        return primitive_slab

    def shiftSlabToBottom(self):
        min_z = np.min(self.coords[:, -1])
        self.coords[:, -1] -= min_z
    def setVacuumheight(self,vac_height, bool_shift_to_bottom=False):
        if bool_shift_to_bottom:
            self.shiftSlabToBottom()
        min_z = np.min(self.coords[:, -1])
        max_z = np.max(self.coords[:, -1])
        self.lattice_vectors[-1, -1] = max_z + vac_height - min_z

        if vac_height < min_z:
            raise Exception(
                """setVacuumheight: the defined vacuum height is smaller than 
                height of the lowest atom. Shift unit cell either manually or by
                the keyword bool_shift_to_bottom towards the bottom
                of the unit cell."""
            )
        self.lattice_vectors[-1, -1] = max_z + vac_height - min_z
    
    def getSlab(self, layers, surface=None, threshold=1e-6, surface_replica=(
            1,1),vacuum_height=None, bool_shift_slab_to_bottom=False):
        
        if surface is not None:
            primitive_slab = self.getPrimitiveSlab(surface, threshold=threshold)
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
                    self.shiftSlabToBottom()

        return slab_new

    def getMaxSize(self):
        coords = copy.deepcopy(self.coords)
        coords[:, :2] -= np.mean(coords[:, :2], axis=0)
        # self.centerXYCoordinates()
        distances=[]
        for atom in coords:
            distances += [np.linalg.norm(atom[:2])]
        max_dist = max(distances)
        return max_dist

    def getSurroundingBox(self, on_atoms=False, on_both=False, correct_ratio=True, max_atom_radius=2.5, fixed_ratio=None):
        """ returns coordinates [min_x,max_x,min_y,max_y] such that the unit cell is well surrounded
            on_atoms: surrounds the atoms and not the unit cell
            on_both: surrounds atoms and unit cell"""
        vecs = self.lattice_vectors
        if on_both:
            x_candidates = np.array([0,vecs[0][0],vecs[1][0],vecs[0][0]+vecs[1][0],np.min(self.coords[:,0]),np.max(self.coords[:,0])])
            y_candidates = np.array([0, vecs[0][1], vecs[1][1], vecs[0][1] + vecs[1][1],np.min(self.coords[:,1]),np.max(self.coords[:,1])])
        else:
            if on_atoms:
                x_candidates = np.array([np.min(self.coords[:,0]),np.max(self.coords[:,0])])
                y_candidates = np.array([np.min(self.coords[:,1]),np.max(self.coords[:,1])])

            else:
                x_candidates = np.array([0,vecs[0][0],vecs[1][0],vecs[0][0]+vecs[1][0]])
                y_candidates = np.array([0, vecs[0][1], vecs[1][1], vecs[0][1] + vecs[1][1]])

        x_min = np.min(x_candidates)-max_atom_radius
        x_max = np.max(x_candidates)+max_atom_radius
        y_min = np.min(y_candidates)-max_atom_radius
        y_max = np.max(y_candidates)+max_atom_radius
        x_range = x_max-x_min
        y_range = y_max-y_min
        ratio = y_range/x_range
        if correct_ratio and fixed_ratio is None:
            if ratio > 1.7:
                mod = (ratio-1.7)+1
                x_max += mod*x_range/2
                x_min -= mod*x_range/2
            elif ratio < 0.58:
                mod = (ratio - 0.58)+1
                y_max += mod*y_range/2
                y_min -= mod*y_range/2
        if fixed_ratio is not None:
            assert isinstance(fixed_ratio,float) or isinstance(fixed_ratio,int)
            if ratio > fixed_ratio:
                # ratio is defined as y/x
                # y is bigger than it should be, x is smaller than it should be
                # we don't want to crop, only expand: x must grow
                new_x_range = y_range/fixed_ratio
                range_increase = new_x_range - x_range
                x_max += range_increase/2
                x_min -= range_increase/2
            elif ratio <= fixed_ratio:
                # ratio is defined as y/x
                # y is smaller than it should be, x is bigger than it should be
                # we have to increase y
                new_y_range = x_range*fixed_ratio
                range_increase = new_y_range - y_range
                y_max += range_increase/2
                y_min -= range_increase/2

        return x_min,x_max,y_min,y_max

    def reflectOnAxis(self, angle):
        """
        reflects the geometry with respect to an axis centered on the origin, with inclination = angle (degrees)
        """
        #conversion from angle
        angle_radians = np.deg2rad(angle)
        m = np.tan(angle_radians)
        new_geom = copy.deepcopy(self)
        for i, coord in enumerate(new_geom.coords):
            x = coord[0]
            y = coord[1]
            z = coord[2]
            x_new = ((1-(m**2))*x + 2*m*y)/(m**2 + 1)
            y_new = (((m**2)-1)*y + 2*m*x)/(m**2 + 1)
            new_coord = [x_new, y_new, z]
            self.coords[i] = new_coord

    def replaceSpeciesByLayers(self,layers,threshold=1):
        """Modifies the specie of the substrate layers individually
            :param new_layers = list [layer on top, second layer from top... last layer on bottom]
            Including the adsorbate!!!
            each layer can be: str() if all atoms have to transform to the same specie
                               list(), to set every individual atom specie
                               None, if the layer species should not change
            :param threshold: see getAtomLayersByHeight()"""

        old_layers_dict = self.getAtomLayersByHeight(threshold=threshold)
        old_layers_heights = [x for x in old_layers_dict.keys()]
        old_layers_heights.sort(reverse=True) #sort top to bottom
        old_layers_inds = [old_layers_dict[k] for k in old_layers_heights]

        assert len(layers) == len(old_layers_inds),\
            'The number of provided layers is different from the number of' \
            ' layers in the geometry'
        old_species = np.array(self.species)
        for i,layer in enumerate(layers):
            if layer is None:
                continue
            elif isinstance(layer,list):
                assert len(layer) == len(old_layers_inds[i]),\
                'Layer n.{} has a different number of atoms from the provided new layer'.format(i)
                new_layer = layer
            elif isinstance(layer,str):
                new_layer = [layer for n in range(len(old_layers_inds[i]))]
            else:
                raise TypeError('Wrong layer format. Each layer must be None, str or list.')
            old_species[old_layers_inds[i]] = new_layer

        new_species = list(old_species)
        self.species = new_species





###############################################################################
#                           Evaluation Functions                              #
###############################################################################

    def getEigenvaluesAndEigenvectors(
            self,
            bool_only_real=True,
            bool_symmetrize_hessian=False,
            bool_omega2=False
    ):
        """
        This function is supposed to return all eigenvalues and eigenvectors of the matrix self.hessian

        Parameters
        ----------
        geometries: List of Geometrys ... might be nice to accept list of coords too
        bool_only_real: returns only real valued eigenfrequencies + eigenmodes (ATTENTION: if you want to also include
        instable modes, you have to symmetrize the hessian as provided below)
        bool_symmetrize_hessian: symmetrized the hessian only for this function (no global change)
        bool_omega2: returns omega2 as third argument

        Returns
        -------
        Eigenfrequencies: numpy array of the eigenfrequencies in cm^(-1)
        Eigenvectors: list of numpy arrays, where each array is a normalized
            displacement for the corresponding eigenfrequency, such that
            new_coords = coords + displacement * amplitude.
            the ith row in an array are the x,y,z displacements for the ith
            atom
        Omega2: (only if bool_omega2!) direct eigenvalues as squared angular frequencies instead of inverse wavelengths
        """

        assert hasattr(self,'hessian') and self.hessian is not None, \
            'Hessian must be given to calculate the Eigenvalues!'
        try:
            masses = [ut.ATOMIC_MASSES[ut.PERIODIC_TABLE[s]] for s in self.species]
        except KeyError:
            print('getEigenValuesAndEigenvectors: Some Species were not known, used version without _ suffix')
            masses = [ut.ATOMIC_MASSES[ut.PERIODIC_TABLE[s.split('_')[0]]] for s in self.species]

        masses = np.repeat(masses, 3)
        M = np.diag(1.0 / masses)

        hessian = copy.deepcopy(self.hessian)
        if bool_symmetrize_hessian:
            hessian = (hessian + np.transpose(hessian)) / 2

        omega2, X = np.linalg.eig(M.dot(hessian))
        
        if bool_only_real:
            # only real valued eigen modes
            real_mask = np.isreal(omega2)
            min_omega2 = 1e-3
            min_mask = omega2 >= min_omega2
            mask = np.logical_and(real_mask, min_mask)

            omega2 = np.real(omega2[mask])
            X = np.real(X[:, mask])
            omega = np.sqrt(omega2)
        else:
            # all eigen modes
            omega = np.sign(omega2) * np.sqrt(np.abs(omega2))

        conversion = np.sqrt((Units.EV_IN_JOULE) / (Units.ATOMIC_MASS_IN_KG * Units.ANGSTROM_IN_METER ** 2))
        omega_SI = omega * conversion
        f_inv_cm = omega_SI * Units.INVERSE_CM_IN_HZ / (2 * np.pi)
        
        eigenvectors = [column.reshape(-1, 3) for column in X.T]

        # sort modes by energies (ascending)
        ind_sort = np.argsort(f_inv_cm)
        eigenvectors = list(np.array(eigenvectors)[ind_sort, :, :])
        f_inv_cm = f_inv_cm[ind_sort]
        omega2 = omega2[ind_sort]

        if bool_omega2:
            return f_inv_cm, eigenvectors, omega2
        else:
            return f_inv_cm, eigenvectors

    def symmetrizeHessian(self,hessian=None):
        h = copy.deepcopy(self.hessian)
        self.hessian = (h+np.transpose(h))/2

    def getAtomFormFactor(self, recip_vector_length):
        species = self.species
        atom_factor = {}
        for s in set(species):
            atom_factor[s] = ut.getAtomicFormFactor(s, recip_vector_length)

        return np.array([atom_factor[s] for s in species])
        
    def getDiffractionIntensities(self, hkl_index_dict = None, q_minmax_dict = None, density_matrix=None, matrix_lattice=None, scaleToUnity = False, debug = False, use_atom_factors=True):
        """Calculates a mock x-ray spectrum of the geometry, as intensities at reciprocal lattice points 
        I.e., for each reciprocal lattice point k, we calculate
        sum_atoms exp(-ikr), where r is the real-space position of each atom in the geometry file.
        Note that EITHER hkl_index_dict or q_minmax_dict should be provided, but never both
        Arguments: 
                - hkl_index_dict type(dicdicttionry): Should contain the minimimum and maximum values of the hkl
                                                  indices, using the keys h_min, h_max, k_min, k_max, l_min, l_max.
                                                  Missing indices are interpreted as zero.

                - q_minmax_dict, type (dict): Contains the minimum and maximum reciprocal lattice vector to be probed. 
                                              Keys are q_min and q_max.

                - density_matrix: np.array(dim=3)
                    threedimensional numpy array with electron density in e/A³

                - matrix_lattice: np.array(3x3)
                    lattice corresponding to the density matrix grid, i.e. the voxel lattice
                    if density_matrix is specified, matrix_lattice must be specified as well

                - scaleToUnity: Normally, intensities scale with the number of electrons in the basis
                                (which is equaivalent to the intensity of the 0,0,0 peak).
                                with scaleToUnity, the intensities are divided by the electron number

                - debug: bool
                    internal debug variable

        returns: dict with key = h/k/l, and value as tuple with (reciprocal lattice point , |S|^2)
        Recommended usage: Remove substrate first!
        """
        #TODO: Parallelize

        #0) Preparation: 
        recip_lattice = self.getReciprocalLattice()
        b1 = recip_lattice[0]
        b2 = recip_lattice[1]
        b3 = recip_lattice[2]
        ikr_array = np.zeros(self.n_atoms, dtype='complex128')
        structure_factor_array = np.zeros(self.n_atoms, dtype='complex128')
        xrayspectrum = {}
        species_numbers = self.getSpeciesAtomicNumber()
        n_electrons = np.sum(species_numbers)

        if density_matrix is not None:
            assert matrix_lattice is not None, 'If density_matrix is given, matrix_lattice must be specified as well!'
        
        #1) Assert that the function was called correctly
        assert not ((hkl_index_dict is None) and (q_minmax_dict is None)), 'Assertion Error in getDiffractionIntensities. Please provide either hkl_index_dict or q_minmax_dict.'
        assert not ((hkl_index_dict is not None) and (q_minmax_dict is not None)), 'Assertion Error in getDiffractionIntensities. Please provide either hkl_index_dict or q_minmax_dict, not both'
        assert hkl_index_dict is None or type(hkl_index_dict) == type(dict()), 'Assertion Error in getDiffractionIntensities. hkl_index_dict must be a dictionary (or None)'
        assert q_minmax_dict is None or type(q_minmax_dict) == type(dict()), 'Assertion Error in getDiffractionIntensities. q_minmax_dict must be a dictionary (or None)'

        #2a Try reading the min/max values from the dictory, if it exists 
        if hkl_index_dict is not None:
            try:
                h_min = hkl_index_dict['h_min']
            except KeyError:
                h_min = 0

            try:
                h_max = hkl_index_dict['h_max']
            except KeyError:
                h_max = 0

            try:
                k_min = hkl_index_dict['k_min']
            except KeyError:
                k_min = 0

            try:
                k_max = hkl_index_dict['k_max']
            except KeyError:
                k_max = 0
            
            try:
                l_min = hkl_index_dict['l_min']
            except KeyError:
                l_min = 0
            
            try:
                l_max = hkl_index_dict['l_max']
            except KeyError:
                l_max = 0
                
            assert not all([v==0 for v in hkl_index_dict.values()]), 'At least one h k or l value must be different to 0.'


        #2b: Try to estimate upper boundaries hkl from q_max
        if q_minmax_dict is not None:
            try:
                q_min = q_minmax_dict['q_min']
            except:
                q_min = 0
            try:
                q_max = q_minmax_dict['q_max']
            except:
                q_max = 0

            #The following originates from a general solution of (hx1+kx2)^2 + (hy1+ky2)^2 < q_max^2
            p_k = 2*(b1[0]*b2[0]+b1[1]*b2[1])/(b2[0]**2+b2[1]**2)
            k_max = int(np.ceil(-p_k/2 + np.sqrt(p_k**2/4+q_max**2)))
            
            p_h = 2*(b1[0]*b2[0]+b1[1]*b2[1])/(b1[0]**2+b1[1]**2)
            h_max = int(np.ceil(-p_h/2 + np.sqrt(p_k**2/4+q_max**2)))
#            print("OTH debug information: h_max, k_max", h_max, k_max)
            #TODO: Implement minima. For now, I don't think this will save a lot of time (not more than I need to calculate the values anyways)
            h_min = 0
            k_min = 0
            l_min = 0
            l_max = 0
            if (debug):
                k_max = max(h_max, k_max)
                h_max = max(h_max, k_max)
                k_max = k_max*5
                h_max = h_max*5
#                h_min = -h_max
#                k_min = -k_max


        #3) Here starts the actual work: calculate the results
        for h in range(h_min,h_max+1):
            for k in range(k_min, k_max+1):
                for l in range(l_min, l_max+1):
                    recip_vector = h*b1 + k*b2 + l*b3
                    recip_coordinate = np.sqrt(recip_vector[0]**2+recip_vector[1]**2+recip_vector[2]**2)

                    if q_minmax_dict is not None:
                        if not (q_min < recip_coordinate < q_max): 
                            continue
    
                    #write more pyhtonic
                    for i_atom in range(self.n_atoms):
                         my_coord = self.coords[i_atom]
                         ikr_array[i_atom] = np.dot(my_coord, recip_vector)*1j #calculate ikr

                    eikr_array = np.exp(ikr_array) # calculate e^ikr for each element

                    #Multiply number of electrons in for each element!
                    if use_atom_factors:
                        structure_factors = self.getAtomFormFactor(recip_coordinate)
                    else:
                        structure_factors = species_numbers
                    structure_factor_array = structure_factors*eikr_array

                    Structure_Factor = np.sum(structure_factor_array) # sum over all elements
                    Abs_Structure_Factor = np.abs(Structure_Factor) # take S absolute

                    if scaleToUnity:
                        Abs_Structure_Factor = Abs_Structure_Factor/n_electrons

                    xrayspectrum[(h,k,l)] = (recip_coordinate, Abs_Structure_Factor)

        return xrayspectrum

    # not sure if this belongs to Geometry
    def getVanDerWaalsEnergy(self,geometry2 = None, s6 = 0.94, alpha = 23):
        ''' Calculates the Van der Waals Interaction Energy between two molecules 
            s6 is the global scaling factor for Grimme VdW
            s6 for PBE in aims is 0.94
            alpha is the damping function factor'''
            
        if geometry2 is None:
            raise NotImplementedError('Vdw Self Interaction Energy is not yet implemented')
        
        else:
            not_avail_self = [i for i in range(self.n_atoms) if self.species[i] not in C6_COEFFICIENTS]
            not_avail_g2 = [i for i in range(geometry2.n_atoms) if geometry2.species[i] not in C6_COEFFICIENTS]
            
            
            if len(not_avail_self) > 0 or len(not_avail_g2)>0:
                warnings.warn('Could not find VdW Coefficients for {}'.format( set([self.species[i] for i in not_avail_self]+ [geometry2.species[j] for j in not_avail_g2])))
            
            g1 = copy.deepcopy(self)
            g1.removeAtoms(not_avail_self)
            g2 = copy.deepcopy(geometry2)
            g2.removeAtoms(not_avail_g2)
            
            
            R_ij = scipy.spatial.distance.cdist(g1.coords, g2.coords)
            C_i = np.array([C6_COEFFICIENTS[s] for s in g1.species])
            C_j = np.array([C6_COEFFICIENTS[s] for s in g2.species])
            C_i,C_j = np.meshgrid(C_i,C_j,indexing='ij')
            C_ij = 2 * C_i * C_j / (C_i + C_j)
            
            R0_i = np.array([R0_COEFFICIENTS[s] for s in g1.species])
            R0_j = np.array([R0_COEFFICIENTS[s] for s in g2.species])
            R0_i, R0_j = np.meshgrid(R0_i, R0_j,indexing='ij')
            R0_ij = R0_i + R0_j
            
            f_damp = 1 / (1 + np.exp(-alpha*(R_ij/R0_ij -1)))
            
            E_disp = - s6 * np.sum(C_ij/R_ij**6 *f_damp)
            
            return E_disp


    def getCoulombMatrix(self):
        """
        Calculate the CouloumbMatrix of the geometry.
        This can be used as a feature vector for machine learning.
        See: R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, J. Chem. Theor. Comput. 2015, 11, 2087
        """        
        M = np.zeros([self.n_atoms, self.n_atoms])
        Z = [PERIODIC_TABLE[s] for s in self.species]
        Z_eff = [SLATER_EFFECTIVE_CHARGES[z] for z in Z]
        #diagonal
        for i in range(self.n_atoms):
            M[i,i] = 0.5 * Z_eff[i]**2.4
        # off diagonal; symmetric  
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                M[i,j] = Z_eff[i]*Z_eff[j]/np.linalg.norm(self.coords[i,:]-self.coords[j,:])
                M[j,i] = M[i,j]
        return M


    def getD3DispersionCorrection(self):

        return self.getD3DispersionObject().getEnergyAndForces()

    def getD3DispersionObject(self):
        from aimstools.D3DispersionCorrection import D3DispersionCorrection
        return D3DispersionCorrection(self)

    def checkSymmetry(self,transformation,tolerance,return_symmetrical=False):
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
        center = getFractionalCoords(symm_geometry.getGeometricCenter(),symm_geometry.lattice_vectors)
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
        symm_geometry.reorderAtoms(indices)
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

    def getVolumeOfUnitCell(self):
        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]
        volume = np.cross(a1, a2).dot(a3)
        return volume

###############################################################################
#                             VARIOUS                                         #
#                          not yet sorted                                     #
###############################################################################
    def doesASmallerUnitCellExist(self):
        """Purpose: Returns true is a smaller unit cell can be generated
        Functionality: If a smaller unit cell exists, the 1,0,0 or the 0,1,0 peak 
        of the X Ray Spectrum have no intensity. To find out what the smallest periodicity is
        one wouuld have to find the first non-zero-intensity peak"""

        hkl_minmax_dict = {"h_min":0, "h_max":1, "k_min": 0, "k_max":1}
        my_spectrum = self.getDiffractionIntensities(hkl_index_dict = hkl_minmax_dict)
        if np.isclose(my_spectrum[(1,0,0)],0) or np.isclose(my_spectrum[(0,1,0)],0):
            return True
        else:
            return False


    def getOrientationOfMainAxis(self):
        """
        Get the orientation of the main axis relative to the x axis
        the main axis is transformed such that it always points in the upper half of cartesian space
        
        Returns:
            angle between main axis and x axis in degree
        """
        main_ax = self.getMainAxes()[1][:,0]
        if main_ax[1]<0:
            main_ax *= -1
        return (np.arctan2(main_ax[1], main_ax[0])*180/np.pi)


    def getReciprocalLattice(self):
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


    def setVacuumLevel(self,vacuum_level):
        ''' sets vacuum level of geometry calculation '''
        
        self.vacuum_level = vacuum_level


    def setMultipolesCharge(self,charge):
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


    def moveMultipoles(self,shift):
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


    def truncate(self, n_atoms):
        """Keep only the first n_atoms atoms"""
        self.species = self.species[:n_atoms]
        self.constrain_relax = self.constrain_relax[:n_atoms]
        self.external_force = self.external_force[:n_atoms]
        self.calculate_friction= self.calculate_friction[:n_atoms]
        self.coords = self.coords[:n_atoms,:]
        self.n_atoms = n_atoms

    def removeAtoms(self,atom_inds):
        """remove atoms with indices atom_inds.
        If no indices are specified, all atoms are removed"""
        if hasattr(self,'geometry_parts') and len(self.geometry_parts) > 0:
            # (AE): added "len(self.geometry_parts) > 0" to suppress this frequent warning when it is supposely not relevant (?)
            warnings.warn('CAUTION: geometry_parts indices are not updated after atom deletion!!\n \
                           You are welcome to implement this!!')
        if atom_inds is None:
            atom_inds = range(len(self))
        mask = np.ones(len(self.species), dtype=bool)
        mask[atom_inds] = False

#        self.species = [i for j,i in enumerate(self.species) if j not in atom_inds]
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
    
    def removeAtomsBySpecies(self, species):
        """
        removes specific atom species
        """
        
        L = np.array(self.species) == species
        atom_inds = np.where(L)[0]
        self.removeAtoms(atom_inds)

    def removeAllConstraints(self):
        self.constrain_relax=np.zeros([len(self.species), 3], bool)

    def removeConstrainedAtoms(self):
        """
        remove all atoms where all coordinates are constrained
        """
        remove_inds = self.getConstrainededAtoms()
        self.removeAtoms(remove_inds)
        
    def removeUnconstrainedAtoms(self):
        """
        remove all atoms where all coordinates are constrained
        """
        remove_inds = self.getUnconstrainededAtoms()
        self.removeAtoms(remove_inds)
        
    def getConstrainededAtoms(self):
        constrain = np.any(self.constrain_relax,axis=1)
        inds = [i for i, c in enumerate(constrain) if c]
        return inds
    
    def getUnconstrainededAtoms(self):
        all_inds = list(range(len(self)))
        keep_inds = self.getConstrainededAtoms()
        inds = list(set(all_inds) - set(keep_inds))
        return inds
    
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
            i for i in range(len(self)) if not i in indices_outside
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
        frac_coords = ut.getFractionalCoords(self.coords, lattice)
        

        
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
        epsilon = 0.1; # Distance in Angstrom for which two atoms are assumed to be in the same position
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

    def removeEmptium(self):
        empt_inds = []
        for i,s in enumerate(self.species):
            if 'Em' in s:
                empt_inds.append(i)
        self.removeAtoms(empt_inds)


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


    @ut.deprecated      # this implicitly assumes a metal substrate, use getAdsorbateIndices instead
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

    @ut.deprecated      # use removeAdsorbates instead
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
        
        new_geom = Geometry()
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
    
    @ut.deprecated      # use getAdsorbates (which is more general) instead
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
            if not atom_species in layers:
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

    def getDirtyZMappingFixForVASP(self):
        """
        This function, for now, is a very dirty fix for VASPs
        mapping of atoms to the other side of the unit cell
        """
        vec_z = self.lattice_vectors[2]
        mag_z = np.linalg.norm(vec_z)
        
        for coord in self.coords:
            if coord[2] > mag_z*0.75:
                coord[2] -= mag_z

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
            sub_new = GeometryFile()
            sub_new.lattice_vectors = sub.lattice_vectors
            for layer_ind in layer_indices:
                sub_new += sub.getAtomsByIndices(layers[heights[layer_ind]])
        
        return sub_new


    def getAreaInNm2(self):
        """Returns the area of the surface described by lattice_vectors 0 and 1 of the geometry, assuming that the lattice_vector 2 is orthogonal to both"""
        a=deepcopy(self.lattice_vectors[0,:-1])
        b=deepcopy(self.lattice_vectors[1,:-1])
        a/=10
        b/=10
        area = np.abs(np.cross(a,b))
        return area


    def getAreaInAtomNumbers(self,substrate_indices=None,substrate=None):
        """
        Returns the area of the unit cell in terms of substrate atoms in the
        topmost substrate layer. The substrate is determined by
        default by the function self.getSubstrate(). For avoiding a faulty
        substrate determination it can be either given through indices or
        through the substrate geometry itself
        :param substrate_indices: list of indices of substrate atoms
        :param substrate: geometry file of substrate
        """
        topmost_sub_layer = self.getSubstrateLayer(
            layer_indices=[0],
            substrate_indices=substrate_indices,
            substrate=substrate
        )
        return topmost_sub_layer.n_atoms

    def getNumberOfAtomLayers(self, threshold=1e-2):
        layers = self.getAtomLayers(threshold=threshold)
        
        total_number_layers = 0
        for atom_species in layers:
            layers[atom_species] = len(layers[atom_species])
            total_number_layers += layers[atom_species]
    
        return layers, total_number_layers

    # IDEA: It would be wise to create a new function getCenter
    def getGeometricCenter(self, ignore_center_attribute=False, indices=None):
        """
        Returns the center of the geometry. If the attribute *center* is set, it is used as the definition for the center of the geometry, provided that

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

    def getCenterOfMass(self):
        """
        Mind the difference to self.getGeometricCenter

        Returns
        -------
        center_of_mass: well, the 3D-coordinate of the center of mass
        """
        
        # R.B. debug: enable also species that have an '_' in it
        species_helper = []
        for si in self.species:
            si_new = si.split('_')[0]
            species_helper.append(si_new)
        
        #masses_np = np.array([ATOMIC_MASSES[PERIODIC_TABLE[s]] for s in self.species], dtype=np.float64)
        masses_np = np.array([ATOMIC_MASSES[PERIODIC_TABLE[s]] for s in species_helper], dtype=np.float64)       
        center_of_mass = self.coords.T.dot(masses_np) / masses_np.sum()
        return center_of_mass
    
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
                
                if not pair_index in neighbouring_atoms:
                    dist = np.linalg.norm(coord - coord_test)
                    
                    pair_species = (species[i], species[j])
                    
                    if dist < bond_thresholds[pair_species]:
                        neighbouring_atoms[pair_index] = [pair_species, dist]
        
        return neighbouring_atoms
    
    
    def getBondLengths(self, bond_factor=1.5):
        """
        Parameters
        ----------
        Parameter for bond detection based on atom-distance : float, optional
            DESCRIPTION. The default is 1.5.

        Returns
        -------
        list
            List of bond lengths for neighbouring atoms.

        """
        neighbouring_atoms = self.getAllNeighbouringAtoms(bond_factor=bond_factor)
        
        bond_lengths = []
        for v in neighbouring_atoms.values():
            bond_lengths.append(v[1])
            
        return np.array(bond_lengths)

    def reorderAtoms(self, inds):
        "Reorders Atoms with index list"
        self.coords = self.coords[inds, :]
        self.species = [self.species[i] for i in inds]
        self.constrain_relax = self.constrain_relax[inds, :]
        self.initial_charge = [self.initial_charge[i] for i in inds]
        self.initial_moment = [self.initial_moment[i] for i in inds]

    def add_atoms(self, cartesian_coords, species, constrain_relax=None, initial_moment=None, initial_charge=None, external_force=None, calculate_friction=None):
        """Add additional atoms to the current geometry file.
        
        Parameters
        ----------
        cartesion_coords : List of numpy arrays of shape [nx3]
            coordinates of new atoms
        species : list of strings
            element symbol for each atom
        constrain_relax : list of lists of bools (optional)
            [bool,bool,bool] (for [x,y,z] axis) for all atoms that should be constrained during a geometry relaxation
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

    def getPeriodicReplica(self, replications, lattice=None, explicit_replications=None):
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
        New geometry file
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
            offset = getCartesianCoords(frac_shift, lattice)
            new_coords[insert_pos:insert_pos+self.n_atoms, :] = self.coords + offset
            insert_pos += self.n_atoms
            
        new_geom = GeometryFile()
        
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
        coordinates = getFractionalCoords(self.coords, self.lattice_vectors)
        
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
            if (not (self.species[OriginAtom] in AllowedSpecies)  ): continue
            OriginCoord=self.coords[OriginAtom]
            
            for TargetAtom in range(OriginAtom,self.n_atoms): #Loop over all other atoms in the list.
                if (not (self.species[TargetAtom] in AllowedSpecies)  ): continue
                
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

    def getMassPerAtom(self):
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

    def getAtomicMass(self):
        atomic_mass = 0
        
        for s in self.species:
            atomic_mass += ATOMIC_MASSES[PERIODIC_TABLE[s]]
        
        return atomic_mass

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

        # end: code based on ase/atoms.py: get_moments_of_inertia
        ###########
##### will rewrite these functions better, I am commenting them out for passing to vsc4
    # def get_symmetry_axes(self):
    #     """
    #     returns m, b pairs for y = mx + b being symmetry axes of the molecule
    #     """
    #     ###geometry gets flattened to get symmetries #####
    #     geom = copy.deepcopy(self)
    #     for i, coord in enumerate(self.coords):
    #         geom.coords[i][2] = 0
    #
    #     ## center the molecule and derive angles of reflection symmetry
    #     geom.centerXYCoordinates()
    #     distances = []
    #     for cont in np.arange(0, 180, 0.1):
    #         geom2=copy.deepcopy(geom)
    #         geom2.reflectOnAxis(cont)
    #         distance = geom.getDistanceToEquivalentAtoms(geom2)
    #         distances += [[cont, distance]]
    #     distances = np.array(distances)
    #     ## to recognize actual identities working with finite numbers:
    #     ## all local minima are taken, and those at the same order of magnitude as the smallest are kept
    #     minima = argrelextrema(distances[:,1], np.less)
    #     ### apparently, first or last element are not recognized by argrelextrema.
    #     ### add them in case they may satisfy the last condition
    #     if distances[1][1] > distances[0][1]:
    #         points = minima[0]
    #         points = np.append(points,0)
    #         minima = (points,)
    #     # if distances[1798][1] > distances[1799][1]:
    #     #     print('case 2')
    #     #     points = minima[0]
    #     #     print('points before: ', points)
    #     #     points = np.append(points, 1799)
    #     #     print('points after: ', points)
    #     #     minima = (points,)
    #     minima_values = distances[minima]
    #     smallest_minimum = np.min(minima_values[:,1])
    #     meaningful_minima = []
    #     for el in minima_values:
    #         if el[1]/smallest_minimum < 10:
    #             meaningful_minima += [el]
    #
    #     ### define line functions y = mx + b from angles
    #     offset = self.getGeometricCenter()
    #     offset_x = offset[0]
    #     offset_y = offset[1]
    #     functions = []
    #     for minimum in meaningful_minima:
    #         angle = minimum[0]
    #         angle_in_radians = np.deg2rad(angle)
    #         m = np.tan(angle_in_radians)
    #         b = offset_y - offset_x*m
    #         function_par = [m,b]
    #         functions += [function_par]
    #     self.symmetry_axes = functions
    #     if not (hasattr(self,'inversion_index')):
    #         self.inversion_index = None
    #     if self.inversion_index==None:
    #         self.evaluate_inversion()

    # def evaluate_inversion(self):
    #     """
    #     # Sets self.inversion_index True or False according to the orientation of the molecule
    #     """
    #     geom = copy.deepcopy(self)
    #     geom.centerXYCoordinates()
    #     projections = []
    #     for axis in self.symmetry_axes:
    #         A = [100, axis[0]*100]
    #         for atom in geom.coords:
    #             dot = np.dot(A, atom[:2])
    #             ratio = dot/((np.linalg.norm(A))**2)
    #             projection = np.dot(A, ratio)
    #             projections += [projection]
    #     positive_projections = np.array([element for element in projections if element[1] >= 0])
    #     norms = np.linalg.norm(positive_projections, axis = 1)
    #     highest_norm = np.argmax(norms)
    #     highest_projection = positive_projections[highest_norm]
    #     if highest_projection[0] >= 0:
    #         self.inversion_index = True
    #     else:
    #         self.inversion_index = False
    #
    # def evaluate_point_on_symmetry_grid(self, point):
    #     """
    #     returns 1 or -1 according to the position of point relatively to the symmetry axes
    #     point is greater than an odd number of symmetry axes = -1
    #     point is greater than an even number of symmetry axes = 1
    #     """
    #     point_eval = 0
    #     empty = False
    #     absent = not hasattr(self, 'symmetry_axes')
    #     if not absent:
    #         empty = self.symmetry_axes == None
    #     if empty or absent:
    #         self.symmetry_axes = []
    #         self.get_symmetry_axes()
    #     for axis_function in self.symmetry_axes:
    #         #point_y_func = axis_function(point[0])
    #         point_y_func = axis_function[0]*point[0] + axis_function[1]
    #         if point[1] >= point_y_func:
    #             point_eval += 1
    #     if self.inversion_index:
    #         point_eval += 1
    #     if point_eval % 2 == 0:
    #         return(1)
    #     elif point_eval % 2 == 1:27
    #         return(-1)

    def getNumberOfElectrons(self):
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

###############################################################################
#                        ControlFile Helpers                                  #
###############################################################################
    def getCubeFileGrid(self,divisions,origin = None,verbose=True):
        """EXPERIMENTAL!
           creates cube file grid with given number of divisions in each direction.
           If only one division is given, this will be used in all directions.
           If origin is given the vectors will be aligned around this value
           
           Returns a CubeFileSettings object which can be used for the ControlFile class
           To get text simply use CubeFileSettings.getText()
           """
           
#        z_max = np.amax(self.coords[:,2])
#        z_min = np.amin(self.coords[:,2])
#        
#        if z_center is None:
#            z_center = (z_max+z_min)/2
#            
#        if isinstance(divisions,np.int64) or len(divisions) == 1:
#            divisions = np.array([divisions]*3)
#        elif len(divisions) == 3:
#            divisions = np.array(divisions)
#        else:
#            NotImplementedError('Divisions must either be specified with one or three values')
#        
#        if z_offset is None:
#            z_span = (z_max -z_min) +5
#        else:
#            z_span = (z_max -z_min)+2*z_offset[0]
        
        if origin is None:
            origin = self.lattice_vectors[0,:]/2 + self.lattice_vectors[1,:]/2 +self.lattice_vectors[2,:]/2
        
        # calculate dx_i
        divs = np.zeros(3)
        divs[0] = np.linalg.norm(self.lattice_vectors[0,:])/divisions[0]
        divs[1] = np.linalg.norm(self.lattice_vectors[1,:])/divisions[1]
        divs[2] = np.linalg.norm(self.lattice_vectors[1,:])/divisions[2]
        
        # calculate vectors
        vecs = np.zeros([3,3])
        vecs[0,:] = self.lattice_vectors[0,:]/divisions[0]
        vecs[1,:] = self.lattice_vectors[1,:]/divisions[1]
        vecs[2,:] = self.lattice_vectors[2,:]/divisions[2]
        
        cube_settings = CubeFileSettings()
        
        cube_settings.setOrigin(origin)
        cube_settings.setEdges(divisions,vecs)
        print('Divisions in directions: \n x: {0:.8f}, y: {1:.8f}, z: {2:.8f}\n'.format(divs[0],divs[1],divs[2]))
        return cube_settings

    def getCubeFileGridBySpacing(self, spacing, origin=None, verbose=True):
        """EXPERIMENTAL! and ugly as hell! <aj, 10.4.19>
           creates cube file grid with given spacing in each direction.
           If only one division is given, this will be used in all directions.
           If origin is given the vectors will be aligned around this value

           Returns a CubeFileSettings object which can be used for the ControlFile class
           To get text simply use CubeFileSettings.getText()
           """

        if origin is None:
            origin = self.lattice_vectors[0, :] / 2 + self.lattice_vectors[1, :] / 2 + self.lattice_vectors[2, :] / 2
        # make numeric value a list if necessary
        if not isinstance(spacing, Iterable):
            spacing = [spacing]
        # check that spacing is given for all three dimensions
        if len(spacing) == 1:
            spacing = [spacing, spacing, spacing]
        assert len(spacing) == 3, 'Either one spacing or a separate one for each dimension must be given'

        # calculate n points
        n_points = np.zeros(3)
        n_points[0] = np.ceil(np.linalg.norm(self.lattice_vectors[0, :]) / spacing[0])
        n_points[1] = np.ceil(np.linalg.norm(self.lattice_vectors[1, :]) / spacing[1])
        n_points[2] = np.ceil(np.linalg.norm(self.lattice_vectors[2, :]) / spacing[2])

        # calculate vectors
        vecs = np.zeros([3, 3])
        vecs[0, :] = self.lattice_vectors[0, :] / np.linalg.norm(self.lattice_vectors[0, :]) * spacing[0]
        vecs[1, :] = self.lattice_vectors[1, :] / np.linalg.norm(self.lattice_vectors[1, :]) * spacing[1]
        vecs[2, :] = self.lattice_vectors[2, :] / np.linalg.norm(self.lattice_vectors[2, :]) * spacing[2]

        cube_settings = CubeFileSettings()

        cube_settings.setOrigin(origin)
        cube_settings.setEdges(n_points, vecs)
        print('Divisions in directions: \n x: {0:.8f}, y: {1:.8f}, z: {2:.8f}\n'.format(n_points[0], n_points[1], n_points[2]))
        return cube_settings




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


###############################################################################
#                           VISUALISATION                                     #
###############################################################################
    def getAsASE(self):
        import ase
        """Convert geometry file to ASE object"""
        #atoms_string = ""
        atom_coords = []
        atom_numbers = []
        for i in range(self.n_atoms):
            #atom_string = self.species[i]
            #if len(atom_string)> 2:
            #    atom_string = atom_string[:2]
            #atoms_string += self.species[i]
                
            # Do not export 'emptium" atoms
            if self.species[i] != 'Em':
                atom_coords.append(self.coords[i,:])
                atom_numbers.append(getAtomicNumber(self.species[i]))
                
        ase_system = ase.Atoms(numbers=atom_numbers, positions=atom_coords)
        ase_system.cell = self.lattice_vectors
        
        if not np.sum(self.lattice_vectors) == 0.0:
            ase_system.pbc = [1, 1, 1]
                
        return ase_system
            
        
    def printWithASE(self, filename, scale=20):
        import ase.io
        """Save the current geometry file as an image (e.g. png), using ASE.
        Parameters:
        -----------
        filename : string
        scale : integer
            larger values yield higher resolution images, but image size is
            internally limited to 500px in some ASE versions        
        """
        atoms = self.getAsASE()
        ase.io.write(filename, atoms, scale=scale)
    
#    def showInASEViewer(self):
#        from ase.visualize import view
#        atoms = self.getAsASE()
#        view(atoms)

    # old: def printToFile(self,name,axes = [0,1],value_list = None,maxvalue = None,cbar_label='', hide_axes=False,title = None):
    def printToFile(self, name, title = None, dpi=300, **kwargs):
        """
        saves figure to destination name
        Parameters
        ----------
        name
        title
        dpi         int
        kwargs

        Returns
        -------

        """
        import matplotlib.pyplot as plt

        is_interactive = plt.isinteractive()
        plt.interactive(False)
        fig = plt.figure()
        self.visualize(**kwargs)

        transparent = kwargs.get('transparent',False)

        if title is not None:
            plt.title(title,fontsize=20,fontweight='bold')
        #plt.tight_layout(pad=0)
        plt.savefig(name,bbox_inches='tight',transparent=transparent, dpi=dpi)
        plt.close(fig)
        plt.interactive(is_interactive)

    def visualize(self,
                  axes=[0,1],
                  min_zorder=0,
                  value_list=None,
                  maxvalue=None,
                  minvalue=None,
                  cbar_label='',
                  hide_axes=False,
                  axis_labels=True,
                  auto_limits=True,
                  crop_ratio=None,
                  brightness_modifier=None,
                  print_lattice_vectors=False,
                  print_unit_cell=False,
                  plot_new_vectors=False,
                  alpha=1.0,
                  linewidth=1,
                  lattice_linewidth=None,
                  lattice_color='k',
                  lattice_linestyle='-',
                  atom_scale=1,
                  highlight_inds=[],
                  highlight_color='C2',
                  color_list = None,
                  cmap=None,
                  ax=None,
                  xlim=None,
                  ylim=None,
                  zlim=None,
                  plot_method='circles',
                  invert_colormap=False,
                  edge_color=None,
                  show_colorbar=True,
                  reverse_sort_inds=False,
                  axis_labels_format="/",
                  **kwargs):
        """
        Generates at plt-plot of the current geometry file.
        If value_list is given, atoms are colored according to it.
        Atoms have even zorder numbers starting with the lowermost atom!!!

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
        """
        
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.colors
        import matplotlib.cm as cmx
        import colorsys

        # default for lattice_linewidth (which is used to draw the lattice)
        if lattice_linewidth is None:
            lattice_linewidth = 2*linewidth

        orig_inds = np.arange(self.n_atoms)
        remove_inds = []
        if xlim is not None:
            remove_x = self.getCroppingIndices(xlim=xlim, auto_margin=True)
            remove_inds+=list(remove_x)
        if ylim is not None:
            remove_y = self.getCroppingIndices(ylim=ylim, auto_margin=True)
            remove_inds+=list(remove_y)
        if zlim is not None:
            remove_z = self.getCroppingIndices(zlim=zlim, auto_margin=True)
            remove_inds+=list(remove_z)
            
        crop_inds = list(set(remove_inds))
        
        if len(crop_inds)>0:
            orig_inds = [orig_inds[i] for i in orig_inds if i not in crop_inds]
            cropped_geom = copy.deepcopy(self)
            cropped_geom.removeAtoms(crop_inds)
        else:
            cropped_geom = self

        if ax is None:
            ax = plt.gca()

        axnames = ['x','y','z']
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
                
        inds = np.argsort(orig_coords[:,sort_ind])

        if sort_ind == 1:
            inds = inds[::-1]
        if reverse_sort_inds:
            inds = inds[::-1]

        orig_inds = [orig_inds[i] for i in inds]
        coords = orig_coords[inds]
#        constrain = orig_constrain[inds]
        species = [orig_species[i] for i in inds]
        n_atoms = len(species)
        circlesize = [getCovalentRadius(s)*atom_scale for s in species]

        # Specify atom colors by value list or default atom colors
        if value_list is None and color_list is None:
            colors = [getSpeciesColor(s) for s in species]
            colors = np.array(colors)
        elif color_list is not None:
            if len(color_list) == 1:
                colors = list(color_list)*len(self.species)
                colors = [mpl.colors.to_rgb(colors[i]) for i in inds]
            else:
                assert len(species) == len(color_list), 'Color must be specified for all atoms or none!' + \
                    f" Expected {len(species)}, but got {len(color_list)} values"
                colors = [mpl.colors.to_rgb(color_list[i]) for i in inds] # converting all types of color inputs to rgba here
            colors = np.array(colors)
        else:
            assert len(value_list) == self.n_atoms, "Number of Values does not match number of atoms in geometry"
            values = [value_list[i] for i in orig_inds]

            if minvalue is not None:
                assert maxvalue is not None, 'Error! If minvalue is defined also maxvalue must be defined'

            if maxvalue is None and minvalue is None:
                maxvalue = np.max(np.abs(value_list))
                minvalue = -maxvalue

                if maxvalue < 1E-5:
                    maxvalue = 1E-5
                    print('Maxvalue for colormap not specified and smaller 1E-5, \nsetting it automatically to: ', maxvalue)
                else:
                    print('Maxvalue for colormap not specified, \nsetting it automatically to: ', maxvalue)

            if maxvalue is not None and minvalue is None:
                minvalue = -maxvalue


            if cmap is None:
                if invert_colormap:
                    cw = plt.get_cmap('coolwarm_r')
                else:
                    cw = plt.get_cmap('coolwarm')
            else:
                cw = plt.get_cmap(cmap)

            cNorm = matplotlib.colors.Normalize(vmin=minvalue, vmax=maxvalue)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cw)
            
            a = np.array([[minvalue,maxvalue]])
            img = plt.imshow(a,cmap=cw)
            img.set_visible(False)
            colors = []
            for v in values:
                colors.append(scalarMap.to_rgba(v))
                
        # make specified atoms brighter by adding color_offset to all rgb values

        if brightness_modifier is not None:

            # Check if brightness modifier is flat (i.e. a single value) or per atom (list of length n_atoms)
            if isinstance(brightness_modifier, float) or isinstance(brightness_modifier, int):
                brightness_modifier = brightness_modifier * np.ones(n_atoms)

            else:
                # Sort list according to orig_inds (which is already cropped if necessary!)
                assert len(brightness_modifier) == self.n_atoms, "Argument 'brightness_modifier' must either be a " \
                                                                 "scalar (float or int) or a list with length equal " \
                                                                 "to the number of atoms"
                brightness_modifier = [brightness_modifier[i] for i in orig_inds]

            assert len(brightness_modifier) == n_atoms, "Something went wrong while reformatting brightness_modifier!"
            for i in range(n_atoms):
                hls_color = np.array(colorsys.rgb_to_hls(*colors[i,:]))
                hls_color[1] += brightness_modifier[i]*(1-hls_color[1])
                hls_color = np.clip(hls_color,0,1)
                colors[i,:] = colorsys.hls_to_rgb(*hls_color)    
        else:
            brightness_modifier = np.zeros(n_atoms)

        zorder = min_zorder

        if plot_method =='circles':
            for i,s in enumerate(species):
                if plot_method=='circles':
                    x1 = coords[i,axes[0]]
                    x2 = coords[i,axes[1]]
                    if orig_inds[i] not in highlight_inds:
                        if edge_color is None:
                            curr_edge_color = np.zeros(3)+brightness_modifier[i] if brightness_modifier[i]>0 else np.zeros(3)
                        else:
                            curr_edge_color = edge_color

                        ax.add_artist(plt.Circle([x1,x2],circlesize[i], color=colors[i], zorder=zorder,
                                                 linewidth=linewidth, alpha=alpha,
                                                 ec=curr_edge_color))
                    else:
                        if edge_color is None:
                            curr_edge_color = highlight_color
                        else:
                            curr_edge_color = edge_color
                        ax.add_artist(plt.Circle([x1,x2],circlesize[i],color=colors[i],zorder=zorder,linewidth=linewidth,
                                                 alpha=alpha,ec=curr_edge_color))
                    zorder += 2

        elif plot_method == 'wireframe':
            self.visualizeWireframe(coords=coords, species=species,
                                    linewidth=linewidth, min_zorder=min_zorder,
                                    axes=axes, alpha=alpha, **kwargs)

        if print_lattice_vectors:
            ax.add_artist(plt.arrow(0, 0, *cropped_geom.lattice_vectors[0, axes], zorder=zorder, fc=lattice_color,
                                    ec=lattice_color, head_width=0.5, head_length=1))
            ax.add_artist(plt.arrow(0, 0, *cropped_geom.lattice_vectors[1, axes], zorder=zorder, fc=lattice_color,
                                    ec=lattice_color, head_width=0.5, head_length=1))
        if print_unit_cell:
            cropped_geom.visualizeUnitCell(lattice=None, linecolor=lattice_color, axes=axes, linestyle=lattice_linestyle,
                                           linewidth=lattice_linewidth, zorder=zorder,
                                           plot_new_cell=plot_new_vectors,ax=ax)

            # # moved this functionality out of the visualizer, if it works we can delete this part <aj,2020/07/22>
            #
            # if hasattr(cropped_geom, 'original_lattice_vectors') and not plot_new_vectors:
            #     lattice = cropped_geom.original_lattice_vectors
            # else:
            #     lattice = cropped_geom.lattice_vectors
            #
            # # FIXME: Function crashes when axes == (1,2) and print_unit_cell==True.
            # # IDEA: Enable code to draw projection of full unit cell into the yz and xz planes
            # ax.add_artist(ax.arrow(0,0,*lattice[0,axes],zorder=zorder,
            #                         fc=lattice_color,ec=lattice_color,
            #                         head_width=0.0,head_length=0, width=lattice_linewidth))
            # ax.add_artist(ax.arrow(0,0,*lattice[1,axes],zorder=zorder,
            #                         fc=lattice_color,ec=lattice_color,
            #                         head_width=0.0,head_length=0, width=lattice_linewidth))
            #
            # ax.add_artist(ax.arrow(lattice[1,axes[0]], lattice[1,axes[1]], *lattice[0,axes],zorder=zorder,
            #                         fc=lattice_color,ec=lattice_color,
            #                         head_width=0.0,head_length=0, width=lattice_linewidth))
            # ax.add_artist(ax.arrow(lattice[0,axes[0]], lattice[0,axes[1]], *lattice[1,axes],zorder=zorder,
            #                         fc=lattice_color,ec=lattice_color,
            #                         head_width=0.0,head_length=0, width=lattice_linewidth))
        
        # scale:
        xmax = np.max(coords[:,axes[0]]) + 2
        xmin = np.min(coords[:,axes[0]]) - 2
        ymax = np.max(coords[:,axes[1]]) + 2
        ymin = np.min(coords[:,axes[1]]) - 2

        if auto_limits:
            if print_lattice_vectors:
                xmin_lattice = np.min(cropped_geom.lattice_vectors[:,axes[0]]) - 1
                xmax_lattice = np.max(cropped_geom.lattice_vectors[:,axes[0]]) + 1
                ymin_lattice = np.min(cropped_geom.lattice_vectors[:,axes[1]]) - 1
                ymax_lattice = np.max(cropped_geom.lattice_vectors[:,axes[1]]) + 1

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
                curr_crop_ratio = len_xlim/len_ylim
                

                if curr_crop_ratio>crop_ratio:
                    # make y limits larger
                    y_padding_fac = len_xlim/(crop_ratio*len_ylim)
                    y_padding = len_ylim*(y_padding_fac-1)
                    ax_ymin -= y_padding/2
                    ax_ymax += y_padding/2
                    
                else:
                    # make x limits larger
                    x_padding_fac = (crop_ratio * len_ylim)/len_xlim
                    x_padding = len_xlim * (x_padding_fac-1)
                    ax_xmin -= x_padding/2
                    ax_xmax += x_padding/2
                    

            ax.set_xlim([ax_xmin, ax_xmax])
            ax.set_ylim([ax_ymin, ax_ymax])


        # If limits are given, set them
        limits = [xlim,ylim,zlim]
        x1lim = limits[axes[0]]
        x2lim = limits[axes[1]]
        if x1lim is not None:
            ax.set_xlim(x1lim)
        if x2lim is not None:
            ax.set_ylim(x2lim)

        if axis_labels:
            if axis_labels_format == "/":
                ax.set_xlabel(r'{} / $\AA$'.format(axnames[axes[0]]))
                ax.set_ylabel(r'{} / $\AA$'.format(axnames[axes[1]]))
            elif axis_labels_format == "[]":
                ax.set_xlabel(r'{} [$\AA$]'.format(axnames[axes[0]]))
                ax.set_ylabel(r'{} [$\AA$]'.format(axnames[axes[1]]))
        
        if show_colorbar and (value_list is not None):
            cbar = plt.colorbar(ax=ax)
            cbar.ax.set_ylabel(cbar_label)
        
        ax.set_aspect('equal')
        plt.grid(False)
        if hide_axes:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    def visualize3D(self):
        import pyvista as pv

        # %% Visualization
        pv.set_plot_theme('document')
        p = pv.Plotter()

        # p.add_mesh(contours, show_scalar_bar=False, cmap='coolwarm',smooth_shading=True)
        for n_ind in range(self.n_atoms):
            species = self.species[n_ind]
            c = self.coords[n_ind, :]
            new_sphere = pv.Sphere(radius=COVALENT_RADII[species], center=[c[0], c[1], c[2]])
            p.add_mesh(new_sphere, color=getSpeciesColor(species), smooth_shading=True)
        # p.add_mesh(geom_points,render_points_as_spheres=True,point_size=10)
        p.show()
        
    
    def getMoleculesWireframe(self,
                              line_width=10,
                              neglect_species=['H'],):
        """
        Get wireframe of molecue in a format that can be visualised with
        plotly.

        Parameters
        ----------
        line_width : int, optional
            Width of the lines. The default is 10.
        neglect_species : list of strings, optional
            List of atom sprecies that should not be visualised.
            The default is ['H'].

        Returns
        -------
        data : TYPE
            List of geometry_objects plottable by plotly.

        """
        import plotly.graph_objects as go
        
        wireframe = self.getWireframe(coords=None,
                                      species=None,
                                      bond_factor=1.5,
                                      neglect_species=neglect_species,
                                      color=None,
                                      species_colors={})
        
        data = []
        
        for wire in wireframe:
            x = np.array([wire[0][0], wire[1][0]])
            y = np.array([wire[0][1], wire[1][1]])
            z = np.array([wire[0][2], wire[1][2]])
            
            color = ut.getPlotlyColor(wire[2])
            
            data_new = go.Scatter3d(x=x, y=y, z=z,
                                    mode='lines',
                                    line=dict(color=color, width=line_width),
                                    showlegend=False)
            data.append(data_new)
        
        return data


    def getMoleculesSpheres(self,
                            marker_size=3,
                            neglect_species=['H'],):
        """
        Get spheres of atoms in molecue in a format that can be visualised with
        plotly.

        Parameters
        ----------
        marker_size : int, optional
            Size of the spheres that represent the atoms.
            The default is 3.
        neglect_species : list of strings, optional
            List of atom sprecies that should not be visualised.
            The default is ['H'].

        Returns
        -------
        data : list of geometry_objects
            List of geometry_objects plottable by plotly.

        """
        import plotly.graph_objects as go
        
        data = []
        
        for ind in range(len(self)):
            
            if not self.species[ind] in neglect_species:
                color = ut.getPlotlyColor( ut.SPECIES_COLORS[self.species[ind]] )
                
                data_new = go.Scatter3d(x=np.array([self.coords[ind,0]]),
                                        y=np.array([self.coords[ind,1]]),
                                        z=np.array([self.coords[ind,2]]),
                                        mode='markers',
                                        marker=dict(color=color, size=marker_size),
                                        showlegend=False)
                
                data.append(data_new)
        
        return data
    
    
    def visualizePlotly(self,
                        line_width=10,
                        marker_size=3,
                        neglect_species=['H'],
                        fig=None):
        
        import plotly.graph_objects as go
        
        data_3 = self.getMoleculesWireframe(line_width=line_width,
                                            neglect_species=neglect_species)
        
        data_4 = self.getMoleculesSpheres(marker_size=marker_size,
                                          neglect_species=neglect_species)
        
        if fig is None:
            fig = go.Figure()
        
        for d in data_3:
            fig.add_trace(d)
            
        for d in data_4:
            fig.add_trace(d)
        
        return fig
        

    def visualizeUnitCell(self,
                          lattice=None,
                          linecolor='k',
                          axes=[0, 1],
                          linestyle='-',
                          linewidth=3,
                          zorder=1,
                          plot_new_cell=False,
                          alpha=1,
                          ax=None,
                          shift = np.array([0,0])):

        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        if lattice is None:
            if hasattr(self, 'original_lattice_vectors') and not plot_new_cell:
                lattice = self.original_lattice_vectors
            else:
                lattice = self.lattice_vectors
        if lattice is None:
            lattice = self.lattice_vectors
        polygon_coords = np.array([[0, 0],
                                   lattice[axes[0], axes],
                                   lattice[axes[1], axes] + lattice[axes[0], axes],
                                   lattice[axes[1], axes],
                                   [0, 0]]) + shift

        handle = ax.plot(polygon_coords[:, 0],
                         polygon_coords[:, 1],
                         color=linecolor,
                         linestyle=linestyle,
                         linewidth=linewidth,
                         zorder=zorder,
                         alpha=alpha)[0]
        return handle

    def determineBonds(self):
        bond_lengths = {}
        for species in self.species:
            cropped_species = species.split('_')[0]
            bond_lengths[species] = [ut.COVALENT_RADII[cropped_species], ut.VAN_DER_WAALS_RADII[cropped_species]]
        species_multiplicity = {'C':4, 'O':2, 'H':1, 'N':3, 'Ag':4}
        # TODO calculate species combination bond length as sum of covalent radii and sum of vdw radii

        # TODO: get all distances between all atoms
        # get maximum species multiplicity closest atoms, they must all be within bond_lengths
        # return bonds between two atoms as index tuple

    def visualizeAtomIndices(self, axes=[0, 1],min_zorder=0, ax=None, fontsize=10):
        """
        plot indices on top of atoms
        """


        import matplotlib.pyplot as plt

        orig_coords = self.coords

        for i in range(3):
            if i not in axes:
                sort_ind = i
        inds = np.argsort(orig_coords[:, sort_ind])
        coords = orig_coords[inds]

        if ax is None:
            ax = plt.gca()

        coords = coords[:, axes]
        color_list = ['k' if (self.species[i] == 'H') else 'w' for i in inds]
        inds = [str(i) for i in inds]
        

        self.visualizeTextOnGeometry(coords, inds, min_zorder=min_zorder, ax=ax, color_list = color_list, fontsize=fontsize)        
        

    def visualizeTextOnGeometry(self,coords,text_list,min_zorder=0, ax=None,color=(1,1,1,1),fontsize=10,
                                highlight_indices=None,
                                highlight_color='C2',
                                color_list=None):
        # plot arbitrary text on top of coordinates.
        # split off from visualizeAtomIndices for more generality
        import matplotlib.pyplot as plt

        if color_list is None:
            color_list = [color]*len(text_list)

        if highlight_indices is None:
            highlight_indices = []

        if ax is None:
            ax = plt.gca()

        assert len(coords) == len(text_list), 'Text and coordinates must have same length'
        for i in range(len(coords)):
            if i in highlight_indices:
                text_color = highlight_color
            else:
                text_color = color_list[i]
            
            ax.text(coords[i,0],coords[i,1],text_list[i],fontsize=fontsize,fontweight='bold',
                     horizontalalignment='center', verticalalignment='center',zorder=2*i+min_zorder+1,color=text_color)

    # INFO: there exists a function called visualizeTotalForceAndTorque in AIMSOutputReader
    # def visualizeTotalForceAndTorque

    def visualize2DSymmetries(self,
                            symmetries=None,
                            symmetry_precision=1e-05,
                            color='green',
                            linewidth=0.1,
                            alpha=0.6,
                            ax=None,
                            min_zorder=None):
        """
        Visualizes the symmetry axis and inversion center of the structure. By default, obtains the symmetries by calling getSymmetries().
        Symmetries can also be provided externally.
        :param symmetries: symmetries, in the same format as getSymmetries() provides
        :param symmetry_precision: see getSymmetries()
        :param color: "
        :param linewidth: "
        :param alpha:  "
        :return: 0
        TEST IT FOR A WHILE AND POLISH IT BEFORE SUBMITTING
        """
        import matplotlib.pyplot as plt
        if symmetries is None:
            symmetries = self.getSymmetries(symmetry_precision=symmetry_precision)
        # check input format
        assert isinstance(symmetries,dict), 'Wrong format for symmetries'
        assert all(key in symmetries for key in ['rotations','translations','equivalent_atoms']),'Wrong format for symmetries'

        rs = symmetries['rotations']
        ts = symmetries['translations']
        eqs = symmetries['equivalent_atoms']

        if ax is None:
            ax = plt.gca()
        # I want it to be over the plotting of the molecule
        if min_zorder is None:
            zorder = 2*len(self.coords)+1
        else:
            zorder = min_zorder

        for i,r3 in enumerate(rs):
            # get 2D terms
            r=r3[:2,:2]
            t=ts[i]
            t=getCartesianCoords(frac_coords=t,lattice_vectors=self.lattice_vectors)[:2]/2 #position of inversion center: translation_vector/2
            # ignore identity
            if (r == np.array([[1,0],[0,1]])).all():
                continue
            # plot inversion center
            elif (r == np.array([[-1,0],[0,-1]])).all():
                ax.add_artist(plt.Circle(t, linewidth*4, color=color,linewidth=linewidth,alpha=alpha,zorder=zorder))
            # plot reflection axis
            elif (r == np.array([[1,0],[0,-1]])).all():
                starting_point = t  #because (0,0) is also symmetric with respect to the axis


                ax.add_artist(ax.arrow(starting_point[0], starting_point[1], self.lattice_vectors[0][0],self.lattice_vectors[0][1],fc=color,ec=color,head_width=1.0, head_length=1, width=linewidth,zorder=zorder,alpha=alpha))
                ax.add_artist(ax.arrow(starting_point[0], starting_point[1], -self.lattice_vectors[0][0],-self.lattice_vectors[0][1],fc=color,ec=color,head_width=1.0, head_length=1, width=linewidth,zorder=zorder))
            elif (r == np.array([[-1,0],[0,1]])).all():
                starting_point = t
                ax.add_artist(ax.arrow(starting_point[0], starting_point[1], self.lattice_vectors[1][0],self.lattice_vectors[1][1],fc=color,ec=color,head_width=1.0, head_length=1, width=linewidth,zorder=zorder,alpha=alpha))
                ax.add_artist(ax.arrow(starting_point[0], starting_point[1], -self.lattice_vectors[1][0],-self.lattice_vectors[1][1],fc=color,ec=color,head_width=1.0, head_length=1, width=linewidth,zorder=zorder,alpha=alpha))
            else:
                print('Only reflections parallel to UC axes are implemented at the moment')
        # set proportions and size
        self.visualize(ax=ax, hide_axis=True, alpha=0.0)

    def visualizeArrowsOnAtoms(self,arrow_vectors,
                        axes=[0,1],
                        vector_colors=None,
                        color_by_species=True,
                        visualize_constrained=False,
                        min_zorder=1, arrow_width=0.005,
                        show_legend=True,
                        legend_length=1,
                        legend_text = 'your text here'):
        """

        Parameters
        ----------
        arrow_vectors: np.array([n_atoms,3])
            vector coordinates to be plotted starting from atom position i

        axes: [int, int]
            indices of axes to plot

        vector_colors: list, None
            list of colors for all arrows

        color_by_species: bool
            color by species if true and color list is None, 'k' otherwise

        print_constrained: bool
            whether or not to show the constrained atom positions

        min_zorder: int
            z order

        arrow_width: float
            width of arrowhead in A

        show_legend: bool
            whether or not to show the legend on lower right

        legend_length: float
            realspace length of the legend bar

        legend_text: str
            content of the legend labeling

        Returns
        -------

        """

        import matplotlib.pyplot as plt

        orig_coords = self.coords
        orig_species = self.species
        orig_constrain = self.constrain_relax

        for i in range(3):
            if i not in axes:
                sort_ind = i

        # sort by coordinate that is not in axes
        inds = np.argsort(orig_coords[:, sort_ind])
        coords = orig_coords[inds]
        arrow_vectors = arrow_vectors[inds]
        constrain = orig_constrain[inds]
        species = [orig_species[i] for i in inds]
        circlesize = [COVALENT_RADII[s[:2]] for s in species]
        if color_by_species:
            colors = [getSpeciesColor(s) for s in species]
        elif vector_colors is not None:
            colors = vector_colors
        else:
            colors = ['k']*len(species)
        ax = plt.gca()

        zorder = min_zorder
        for i, s in enumerate(species):
            # printing arrows as artists didnt work
            # arrow_length = np.sqrt(x1_arrow**2 + x2_arrow**2)
            # ax.add_artist(plt.arrow(x1,x2,x1_arrow,x2_arrow ,width = arrow_length*0.1,linewidth = 0.3*arrow_length,
            #              zorder = zorder+1,fc = colors[i],head_width=0.2*arrow_length, head_length=0.1*arrow_length))
            # quiver has quite an overhead, but does the job
            if (not all(constrain[i, :])) or visualize_constrained:
                plt.quiver(coords[i, axes[0]],
                           coords[i, axes[1]],
                           arrow_vectors[i, axes[0]],
                           arrow_vectors[i, axes[1]],
                           zorder=zorder + 1, units='xy', scale_units='xy', scale=1, angles='xy', pivot='tail',
                           color=colors[i], edgecolor='k', linewidth=1, width=arrow_width, minlength=.005,minshaft=.01)
            zorder += 2

        xmax = np.max(coords[:, axes[0]]) + 2
        xmin = np.min(coords[:, axes[0]]) - 2
        ymax = np.max(coords[:, axes[1]]) + 2
        ymin = np.min(coords[:, axes[1]]) - 2
        if show_legend:
            # scale / 'legend' arrow
            plt.quiver(xmax - legend_length-1, ymin + 1, legend_length, 0, zorder=zorder + 1, units='xy', scale_units='xy', scale=1, angles='xy',
                       pivot='tail')
            ax.text(xmax - 3, ymin + 1.2, legend_text, zorder=zorder + 1, fontsize=8)

        # this function should not rescale images
        # ax.set_xlim((xmin, xmax))
        # ax.set_ylim((ymin, ymax))
        # ax.set_aspect('equal')

    def visualizeAtomDisplacements(self,
                                   other_geom,
                                   axes=[0,1],
                                   vector_colors=None,
                                   color_by_species=False,
                                   visualize_constrained=False,
                                   min_zorder=1,
                                   arrow_width=0.05):
        """
        visualize the shifts atoms have made during e.g. a geometry optimization.
        This file only visualizes the shifts, to visualize the geometry, call self.visualize() beforehand

        Parameters
        ----------
        other_geom: GeometryFile
            geometry file which must be similar enough to the current file to be visualized

        For all other parameters see GeometryFile.visualizeArrows

        Returns
        -------

        """

        shifts = self.coords - other_geom.coords

        other_geom.visualizeArrowsOnAtoms(shifts,
                                          axes=axes,
                                          vector_colors=vector_colors,
                                          color_by_species=color_by_species,
                                          visualize_constrained=visualize_constrained,
                                          min_zorder=min_zorder,
                                          arrow_width=arrow_width,
                                          show_legend=False,
                                          legend_length=1,
                                          legend_text='unimportant')

    def visualizeForces(self,
                        forces,
                        axes=[0,1],
                        arrow_scale=10,
                        print_constrained=False,
                        min_zorder=1,
                        arrow_width=0.005,
                        show_legend=True,
                        legend_length=2):
        """

        Parameters
        ----------
        forces: np.array([n_atoms,3])
            input forces which, after scaling, will be plotted

        axes: [int, int]
            indices of axes to plot

        arrow_scale: float
            factor to scale arrows for forces

        print_constrained: bool
            whether or not to show the constrained atom positions

        min_zorder: int
            z order

        arrow_width: float
            width of arrowhead in A

        show_legend: bool
            whether or not to show the legend on lower right

        legend_length: float
            length of the legend bar in incoming units
        """

        scaled_forces = forces*arrow_scale
        legend_text = '{} eV/A'.format(2/arrow_scale)

        self.visualizeArrowsOnAtoms(scaled_forces,
                             axes=axes,
                             vector_colors=None,
                             color_by_species=True,
                             visualize_constrained=print_constrained,
                             min_zorder=min_zorder,
                             arrow_width=arrow_width,
                             show_legend=show_legend,
                             legend_length=legend_length,
                             legend_text=legend_text)

    
    def getWireframe(self,
                     coords=None,
                     species=None,
                     bond_factor=1.5,
                     neglect_species=['H'],
                     color=None,
                     species_colors={}):
        """
        Visualize the geometry as wireframe colored either according to species or in a specific color.
        Can be used in combination with visualize to show wireframe molecules on top of a substrate.

        Parameters
        ----------
        linewidth: float
            width of plotted lines

        linewidth_in_Angstrom: Bool
            If True, the plotter is changed from plt.plot to DataLinewidthPlot which allows to specify
            linewidth in axis units (which is usually Angstrom in our case). This is not the standard as
            it is much slower!

        bond_factor: float
            all atom distances with (r_cov1 + r_cov2)*bond_factor will be considered as bonds

        neglect_species: list(species_string)
            species to be neglected for wireframe plot

        min_zorder: int
            zorder of plot

        color: color specifier
            specifies color of all lines, can be overwritten by species color

        species_colors: dict(species:color)
            overwrites all other color definitions for this species

        Returns
        -------
        None

        """

        if coords is None:
            coords = self.coords
        if species is None:
            species = self.species

        all_species = set(species)
        cov_radii = ut.COVALENT_RADII
        all_species_pairs = itertools.product(all_species,repeat=2)
        bond_thresholds = {}
        for pair in all_species_pairs:
            bond_thresholds[pair] = (cov_radii[pair[0]]+cov_radii[pair[1]])*bond_factor

        species_colors_internal = {}
        for s in all_species:
            if color is None:
                species_colors_internal[s] = ut.SPECIES_COLORS[s]
            else:
                species_colors_internal[s] = color

        species_colors_internal.update(species_colors)

        wireframe = []

        for i, coord in enumerate(coords):
            if species[i] not in neglect_species:
                neighbor_coords = []
                neighbor_species = []
                for j, coord_test in enumerate(coords[i:]):
                    if species[j+i] not in neglect_species:
                        dist = np.linalg.norm(coord - coord_test)
                        if dist < bond_thresholds[species[i],species[j+i]]:
                            neighbor_coords.append(coord_test)
                            neighbor_species.append(species[j+i])

                for k, (coord_neighbour, species_neighbor) in enumerate(zip(neighbor_coords, neighbor_species)):
                    coord_half = (coord_neighbour+coord)/2.0
                    
                    wire_0 = [coord,
                              coord_half,
                              species_colors_internal[species[i]],]
                    
                    wire_1 = [coord_half,
                              coord_neighbour,
                              species_colors_internal[species_neighbor],]
                    
                    wireframe.append(wire_0)
                    wireframe.append(wire_1)
        
        return wireframe
    
    
    def visualizeWireframe(self,
                           coords=None,
                           species=None,
                           linewidth=3,
                           linewidth_in_Angstrom=False,
                           bond_factor=1.5,
                           neglect_species=['H'],
                           min_zorder=1,
                           axes=[0, 1],
                           color=None,
                           species_colors={},
                           alpha=1):
        """
        Visualize the geometry as wireframe colored either according to species or in a specific color.
        Can be used in combination with visualize to show wireframe molecules on top of a substrate.

        Parameters
        ----------
        linewidth: float
            width of plotted lines

        linewidth_in_Angstrom: Bool
            If True, the plotter is changed from plt.plot to DataLinewidthPlot which allows to specify
            linewidth in axis units (which is usually Angstrom in our case). This is not the standard as
            it is much slower!

        bond_factor: float
            all atom distances with (r_cov1 + r_cov2)*bond_factor will be considered as bonds

        neglect_species: list(species_string)
            species to be neglected for wireframe plot

        min_zorder: int
            zorder of plot

        color: color specifier
            specifies color of all lines, can be overwritten by species color

        species_colors: dict(species:color)
            overwrites all other color definitions for this species

        Returns
        -------
        None

        """

        import matplotlib.pyplot as plt
        from aimstools.PlotUtilities import DataLinewidthPlot
        if coords is None:
            coords = self.coords
        if species is None:
            species = self.species

        ax0 = axes[0]
        ax1 = axes[1]

        all_species = set(species)
        cov_radii = ut.COVALENT_RADII
        all_species_pairs = itertools.product(all_species,repeat=2)
        bond_thresholds = {}
        for pair in all_species_pairs:
            bond_thresholds[pair] = (cov_radii[pair[0]]+cov_radii[pair[1]])*bond_factor

        species_colors_internal = {}
        for s in all_species:
            if color is None:
                species_colors_internal[s] = ut.SPECIES_COLORS[s]
            else:
                species_colors_internal[s] = color

        species_colors_internal.update(species_colors)
        if linewidth_in_Angstrom:
            plot_function = DataLinewidthPlot
        else:
            plot_function = plt.plot

        for i, coord in enumerate(coords):
            if species[i] not in neglect_species:
                neighbor_coords = []
                neighbor_species = []
                for j, coord_test in enumerate(coords[i:]):
                    if species[j+i] not in neglect_species:
                        dist = np.linalg.norm(coord - coord_test)
                        if dist < bond_thresholds[species[i],species[j+i]]:
                            neighbor_coords.append(coord_test)
                            neighbor_species.append(species[j+i])

                for k, (coord_neighbour, species_neighbor) in enumerate(zip(neighbor_coords, neighbor_species)):
                    x_half = (coord_neighbour[ax0]-coord[ax0])/2 + coord[ax0]
                    y_half = (coord_neighbour[ax1]-coord[ax1])/2 + coord[ax1]

                    plot_function((coord[ax0], x_half),
                             (coord[ax1], y_half),
                             color=species_colors_internal[species[i]],
                             solid_capstyle='round',
                             zorder=min_zorder,
                             linewidth=linewidth,
                             alpha=alpha)

                    plot_function((x_half, coord_neighbour[ax0]),
                             (y_half, coord_neighbour[ax1]),
                             color=species_colors_internal[species_neighbor],
                             solid_capstyle='round',
                             zorder=min_zorder,
                             linewidth=linewidth,
                             alpha=alpha)
                             
                             
        

    def find_native_adatoms_and_adlayer_atoms(self, layer_intervall = 0.5, std_intervall_scale=1 ,show_std_intervall_for_native_adatoms=False):
        
        
        # in this code all native adatoms and all adlayer atoms are detected as follows:
        #1. for all species we calculate where the the net plane layers (ONE LAYER IS DEFINED BY ALL ATOMS WITHIN THE layer_intervall----) 
            #are and how much atoms of the species are in the netplane
        #2. find species that are obviously only in the adlayer and the species that are also in the slab
        #3. for all slab species calcuate the change of the number of atoms between consecutive layers : 
            #The adatom layer of a species starts when the change of atoms per layer is very big between two layers
            # The mathematical criterium for finding the native adatoms goes as follows:
                #We calcualte the standard deviation of the change of the number of atoms (for all changes between all consecutive layers)
                #If in layer x the change in the number of atoms dnx is greater than the next smaller change in the number of atoms dny (in any layer y) 
                #plus the standard deviation of the changes between all consecutive layers std( [dn1,dn2,dn3,...,dnN] )*std_intervall_scale
                #then in layer x we have adatoms.
                # -----------YOU CAN SCALE THE STANDARD DEVIATION FOR ACCEPTING A SLAB ATOM AS AN ADATOM BY VARYING std_intervall_scale
                # for std_intervall_scale < 1 adatoms are more likely identified but you risk to accept layer of common slab atoms as an adatom layer
                # for std_intervall_scale > 1 adatoms are less likely accepted but you can be more sure not to accept a layer of common slab atoms as the adatom layer
        
      
        import matplotlib.pyplot as plt


    # Generate a geometry file where all equivalent species are mapped to one species (Ag_reallylight, Ag_light, Ag):->Ag
        geo_equivalent_species = copy.deepcopy(self)
        for i, sp_i in enumerate(geo_equivalent_species.species):
            if '_' in sp_i:
                index_underline = sp_i.index('_')
                geo_equivalent_species.species[i] = sp_i[0:index_underline]

    # Here we create the dictionary species_layers_and_number_of_atoms_in_layer. 
    # For every species it contains the z-coordinates of layers (within an intervall=layer_intervall) and the number of atoms that are in each layer
    #z.B.:  species_layers_and_number_of_atoms_in_layer = {'Au': [[19.45561088, 18.67587968], [3, 2]],  'O': [[18.44341458, 19.74653296], [1, 1]] , ... }
        species_layers_and_number_of_atoms_in_layer = {}
        layers = geo_equivalent_species.getAtomLayers(threshold=layer_intervall)
        
        for species_i in set(geo_equivalent_species.species):
            layer_keys_species_i = layers[species_i].keys()
            number_of_atoms_in_layer = []
            for layer_j in layer_keys_species_i:
                #print(layer_j)
                #print(layers[species_i][layer_j])
                layer_j_number_of_atoms = len(layers[species_i][layer_j])
                #print(layer_j_number_of_atoms)
                number_of_atoms_in_layer.append(layer_j_number_of_atoms)
            
            species_layers_and_number_of_atoms_in_layer[species_i] = [list( layers[species_i].keys() ),number_of_atoms_in_layer]

    # Here we find the slab species and obvious adlayer species
    # generate a list of all species that can not be part of the substrat, i.e. a list of all the species that are obviously part of an adlayer 
    # and a list of species that 
    # here the trick is that the standard deviation of the z coordinates for the slab species is muuuuch bigger #
    #than the standard deviation of the z coordinates for the obvious adlayer species
        pure_adlayer_species = []
        slab_species = []
        standard_deviation_of_z_coordinates = {}
        for sp_i in set(geo_equivalent_species.species):
            all_atoms_species_i = geo_equivalent_species.getAtomsBySpecies(species=sp_i)
            all_z_coordinates_species_i =  np.array(all_atoms_species_i.coords)[:][:,2]
            std_species_i = np.std(all_z_coordinates_species_i)
            standard_deviation_of_z_coordinates[str(sp_i)] = std_species_i

        max_standardeviation = max(standard_deviation_of_z_coordinates.values())  # must be the standard deviation of a slab species
        if max_standardeviation < 2:
            print('Error! This geometry file probaply doesnt contain a slab')
            print('Error! This geometry file probaply doesnt contain a slab')
            print('Error! This geometry file probaply doesnt contain a slab')
            print('Error! This geometry file probaply doesnt contain a slab')
        for species_i in standard_deviation_of_z_coordinates.keys():
            if standard_deviation_of_z_coordinates[ str(species_i) ] < max_standardeviation/3:
                pure_adlayer_species.append(species_i)
            else:
                slab_species.append(species_i)

    # Here we create two dictionaryy. one for the species that are only in the adlayer and one for the species that are also in the slab:
    # species_layers_and_number_of_atoms_in_layer_SLABSPECIES   and    species_and_number_of_atoms_in_layer_ONLYADLAYERSPECIES. 
    # For every species it contains the z-coordinates of layers (within an intervall=layer_intervall) and the number of atoms that are in each layer
    #z.B.:  species_layers_and_number_of_atoms_in_layer = {'Au': [[19.45561088, 18.67587968], [3, 2]],  'O': [[18.44341458, 19.74653296], [1, 1]] , ... 
        species_and_number_of_atoms_in_layer_SLABSPECIES = {}
        species_and_number_of_atoms_in_layer_ONLYADLAYERSPECIES = {}
        
        for key_i in species_layers_and_number_of_atoms_in_layer.keys():
            
            if str(key_i) in pure_adlayer_species:
                species_and_number_of_atoms_in_layer_ONLYADLAYERSPECIES[str(key_i)] = \
                species_layers_and_number_of_atoms_in_layer[key_i]
            
            if str(key_i) in slab_species:
                species_and_number_of_atoms_in_layer_SLABSPECIES[str(key_i)] = \
                species_layers_and_number_of_atoms_in_layer[key_i]

        
     # Here we search for the native adatoms of the slab species with the standard deviation of the change of the number of atoms in the layers #
     # between consecutive layers as explained in the beginning.  
        
        
        
        slab_species_has_adatom = {} 
        # dictionary of the form {'Species': {adatoms_above_z_coordinate:value1,slab_below_z_coordinate:value2,} }    
          #value1=None : no adatoms of this slab species
        
        native_adatom_species = [] # list of all native adatom species

        species_and_number_of_atoms_in_layer_SLABSPECIES = species_and_number_of_atoms_in_layer_SLABSPECIES
        
        for species_i in species_and_number_of_atoms_in_layer_SLABSPECIES.keys():
            
            species_i_has_adatom = False
            
            z_coords_layers = species_layers_and_number_of_atoms_in_layer[str(species_i)][0]
            nr_of_atoms_in_layer = species_layers_and_number_of_atoms_in_layer[str(species_i)][1]
            
            sort_index = np.array(z_coords_layers).argsort()
            z_coords_layers= np.array(z_coords_layers)[[sort_index]] 
            nr_of_atoms_in_layer =  np.array(nr_of_atoms_in_layer)[[sort_index]] 
            


     


            
            diff_nr = np.diff(np.array(nr_of_atoms_in_layer))
            diff_z = np.diff(np.array(z_coords_layers))
            ### Take the absolute value of the differences !!!!! And take only the difference not divided by dz !!!!
            d_nr = np.abs( diff_nr )#/ diff_z )
            z_points = np.array(z_coords_layers)[0:-1] + diff_z/2
            
            
            std_d_nr = np.std(d_nr)
            
            ind_largest_d_nr = np.argsort(d_nr)[-1]
            largest_d_nr = d_nr[ind_largest_d_nr]
            z_points_largest_d_nr = z_points[ind_largest_d_nr]
            
            # find the index of the second largest d_nr
            ind_secondlargest_d_nr = np.argsort(d_nr)[-2]
            secondlargest_d_nr = d_nr[ind_secondlargest_d_nr]
            z_points_secondlargest_d_nr = z_points[ind_secondlargest_d_nr]
            
            if show_std_intervall_for_native_adatoms == True:
                # plot the number of atoms in the layers for every species
                plt.plot( z_coords_layers  , nr_of_atoms_in_layer , 'x' )
                plt.title(str(species_i))
                plt.xlabel(r'z')
                plt.ylabel(r'Nr of atoms in layer')
                plt.show()
                # plot the change in the number of atoms between consecutive layers
                plt.plot(z_points,d_nr, 'o')
                plt.plot(z_points, secondlargest_d_nr*np.ones(len(z_points)), linestyle='--', color='grey')
                plt.plot(z_points, (secondlargest_d_nr+std_d_nr*std_intervall_scale)*np.ones(len(z_points)), \
                         linestyle='--', color='red')
                plt.plot(z_points, (secondlargest_d_nr-std_d_nr*std_intervall_scale)*np.ones(len(z_points)), \
                         linestyle='--', color='red')
                
                plt.ylabel(r'|$dN_{Layer}$|')
                plt.xlabel('z')
                plt.title(str(species_i))
                plt.show()
                

            ## identify if you have an adatom
            # maybe implement the possibility that an adatom is only identified if it lies above the top slab layer of
            # the species with the largest radius
            
            if std_d_nr > 1 and largest_d_nr > (secondlargest_d_nr + std_d_nr*std_intervall_scale ):
                species_i_has_adatom = True
                
            if species_i_has_adatom == True:
                adatoms_above_z_coordinate = z_points_largest_d_nr
                slab_below_z_coordinate = z_points_largest_d_nr
                slab_species_has_adatom[str(species_i)] = {'adatoms_above_z_coordinate':adatoms_above_z_coordinate, \
                                                           'slab_below_z_coordinate' :slab_below_z_coordinate}
                
                native_adatom_species.append(str(species_i))
                
            elif species_i_has_adatom == False:
                adatoms_above_z_coordinate = None
                slab_below_z_coordinate = z_coords_layers[-1] + layer_intervall
                slab_species_has_adatom[str(species_i)] = {'adatoms_above_z_coordinate':adatoms_above_z_coordinate, \
                                                           'slab_below_z_coordinate' :slab_below_z_coordinate}
            

        #print('slab_species_has_adatom: ',slab_species_has_adatom)
        #print('native_adatom_species',native_adatom_species)


    # Finally we make two lists of indizes that label all native adatoms and all adlayer atoms in the original geometry file
        indizes_of_adlayer_atoms = []
        indizes_of_native_adatoms = []
        for i in self.getIndicesOfAllAtoms():
            sp_i = self.species[i]
            #print(sp_i)
            if '_' in sp_i:
                index_underline = sp_i.index('_')
                sp_i = sp_i[0:index_underline]
            #print(sp_i)
            #print('')
            if sp_i in pure_adlayer_species:
                indizes_of_adlayer_atoms.append(i)
            elif sp_i in native_adatom_species:
                z_below_adatoms = slab_species_has_adatom[sp_i]['adatoms_above_z_coordinate']
                if self.coords[i][2] > z_below_adatoms:
                    indizes_of_native_adatoms.append(i)

        return(indizes_of_native_adatoms, indizes_of_adlayer_atoms )
        
        

    def common_data(self, list1, list2):
    #checks if list1 and list2 share at least one element
        result = False
     
        for x in list1:

            for y in list2:
       
                if x == y:
                    result = True
                    return result 
                     
        return result
        

    def BuildPossibleTranslationVectors( self , e1, e2, e3 , max_size_of_translation_vector = 4):
    # builds all possible vectors that can be a linear combination of e1, e2, e3. 
    # max_size_of_translation_vector determines the maximum number of linear combinations of e1 e2 and e3 to get the resulting vector

        possible_vectors = []

        for a in range(max_size_of_translation_vector):
            for b in range(max_size_of_translation_vector):
                for c in range(max_size_of_translation_vector):
                    #print(a,b,c)
                    vec_i_1_plus = a*e1 + b*e2 + c*e3
                    vec_i_1_minus = -(a*e1 + b*e2 + c*e3)

                    vec_i_2_plus = a*e1 - b*e2 + c*e3
                    vec_i_2_minus = -(a*e1 - b*e2 + c*e3)

                    vec_i_3_plus = a*e1 + b*e2 - c*e3
                    vec_i_3_minus = -(a*e1 + b*e2 - c*e3)            


                    possible_vectors += [vec_i_1_plus, vec_i_1_minus, \
                                        vec_i_2_plus, vec_i_2_minus, \
                                        vec_i_3_plus, vec_i_3_minus]



        ############## find all equivalent vectors among the possible vectors
        equivalent_vectors = []
        for i in range(len(possible_vectors)):
            equivalent_vectors.append([])

        counter = 0
        for i, vec_i in enumerate(possible_vectors):
            equivalent_vectors[i].append(i)
            for j, vec_j in enumerate( possible_vectors ):


                if i !=j :
                    if np.all(np.isclose(vec_i, vec_j , atol=0.001) ):


                        equivalent_vectors[i].append(j)
                        equivalent_vectors[j].append(i)

                        equivalent_vectors[i]=list(set(equivalent_vectors[i]))
                        equivalent_vectors[j]=list(set(equivalent_vectors[j]))


        ##### list all vectors that must be erased to obtain a list of unique translation vectors
        list_of_vectors_that_must_remain = []
        list_of_vectors_that_will_be_erased = []

        for list_i in equivalent_vectors:
            #print(list_i)
            #print(list_i[0])

            if len(list_i) >0:
                if list_i[0] not in list_of_vectors_that_will_be_erased:
                    list_of_vectors_that_must_remain.append(list_i[0])
                for list_i_element_j in list_i[1:]:
                    if list_i_element_j not in list_of_vectors_that_must_remain:
                        list_of_vectors_that_will_be_erased.append(list_i_element_j)
        list_of_vectors_that_will_be_erased=list(set(list_of_vectors_that_will_be_erased))


        ############# make a list of only unique translation vectors
        possible_unique_translation_vectors = []

        for i,vec_i in enumerate(possible_vectors):
            if i not in list_of_vectors_that_will_be_erased:
                possible_unique_translation_vectors.append(vec_i)
                
                
        possible_unique_translation_vectors_lengths = []
        
        for vec_i in possible_unique_translation_vectors:
            length_i = np.linalg.norm(vec_i)
            possible_unique_translation_vectors_lengths.append(length_i)
                
        #print('possible_unique_translation_vectors_lengths: ',possible_unique_translation_vectors_lengths)
        length_sorted_indices = np.argsort(possible_unique_translation_vectors_lengths)
        #print('length_sorted_indices: ',length_sorted_indices)
        
        length_sorted_unique_translation_vectors = \
        list( np.array( possible_unique_translation_vectors)[length_sorted_indices]  )
        
        return length_sorted_unique_translation_vectors
        
          




    def clusterAtomsInSameMoleculeNONPeriodic(self, geo, epsilon=0.2):
        

    # this function gives you for every atom a list of other atoms that belong to the same molecule.
    # the output is a dictionary. 
    # the keys of the dictionary are the indices of the atoms of the input geometry file.
    # the values of the dictionary is a list of other atom indices that belong to the same molecule

    # epsilon defines the range of the length of a covalent bond between 2 atoms so that the 2 atoms are considered to belong to the same molecule
        
    ########## got threw all atoms and for every atom find it's neighbors that are covalently bond
        
        neighbour_of_atoms = {}#key is an atom index , value are all the molecular neighbors of this atom

        for i in geo.getIndicesOfAllAtoms():
            
            neighbours_of_atom_i = []
            coords_i = geo.coords[i]
            species_i = geo.species[i]

            for j in [k for k in geo.getIndicesOfAllAtoms() if k !=i]:
                
                coords_j = geo.coords[j]
                species_j = geo.species[j]        
                covalent_distance_i_j = getCovalentRadius(species_i) + getCovalentRadius(species_j)
                dist_ij = np.linalg.norm(coords_i - coords_j)

                if  dist_ij < covalent_distance_i_j + epsilon:
                    neighbours_of_atom_i.append(j)

            neighbour_of_atoms[i] = neighbours_of_atom_i    



    ########### now for every atom find all other atoms that belong to the same molecule

        neighbour_of_atoms_dynamic_helper = copy.deepcopy(neighbour_of_atoms)

        for current_atom_index_i in neighbour_of_atoms_dynamic_helper.keys():


            atoms_that_belong_to_the_same_molecule = [current_atom_index_i]

            atoms_that_belong_to_the_same_molecule += neighbour_of_atoms_dynamic_helper[current_atom_index_i]    
      


            for other_atom_index_j in neighbour_of_atoms_dynamic_helper.keys():

                if current_atom_index_i in neighbour_of_atoms_dynamic_helper[other_atom_index_j]  \
                or self.common_data( neighbour_of_atoms_dynamic_helper[other_atom_index_j] , \
                                 atoms_that_belong_to_the_same_molecule )  :



                    atoms_that_belong_to_the_same_molecule.append(other_atom_index_j)
                    atoms_that_belong_to_the_same_molecule += neighbour_of_atoms[other_atom_index_j]


                    neighbour_of_atoms_dynamic_helper[other_atom_index_j] += atoms_that_belong_to_the_same_molecule

                    neighbour_of_atoms_dynamic_helper[other_atom_index_j] = list(set(neighbour_of_atoms_dynamic_helper[other_atom_index_j]))



            atoms_that_belong_to_the_same_molecule = list( set(atoms_that_belong_to_the_same_molecule) )


            neighbour_of_atoms_dynamic_helper[current_atom_index_i] = atoms_that_belong_to_the_same_molecule



        for atom_i in neighbour_of_atoms_dynamic_helper.keys():
            neighbour_of_atoms_dynamic_helper[atom_i] = list(set( neighbour_of_atoms_dynamic_helper[atom_i]   ))



        return neighbour_of_atoms_dynamic_helper




    def clusterAtomsInSameMoleculePeriodic(self, geo, epsilon=0.2):
    
        # this function gives you for every atom a list of other atoms that belong to the same molecule IN A PERIODIC CASE ...
        # ... EVEN IF THE MOLECULE IS TORN APPART IN A UNIT CELL, ...
        # ... I.E. IF ONLY FOR THE PERIODIC REPLICATION IT IS OBVIOUS WHICH ATOMS BELONG TO THE SAME MOLECULE
        # the output is a dictionary. 
        # the keys of the dictionary are the indices of the atoms of the input geometry file.
        # the values of the dictionary is a list of other atom indices that belong to the same molecule
        
        # epsilon defines the range of the length of a covalent bond between 2 atoms so that the 2 atoms are considered to belong to the same molecule
        
        assert np.all( geo.lattice_vectors is not np.array([0,0,0]) )
        
        
        
        geo_periodic = geo.getPeriodicReplica([3,3,1])

        neighbour_of_atom_NONperiodic = self.clusterAtomsInSameMoleculeNONPeriodic(geo=geo, epsilon=epsilon)
        neighbour_of_atom_periodic    = self.clusterAtomsInSameMoleculeNONPeriodic(geo=geo_periodic, epsilon=epsilon)
        
        e1=geo.lattice_vectors[0]
        e2=geo.lattice_vectors[1]
        e3=geo.lattice_vectors[2]
        
        possible_translation_vectors = [ e1,e2,e3,e1+e1, e1+e2,  e1+e3, e2+e2,  e2+e3, e3+e3, e1+e1+e1, e1+e1+e2,\
        e1+e1+e3,e2+e2+e1,e2+e2+e2,e2+e2+e3,e3+e3+e1, e3+e3+e2, e3+e3+e3, e1+e2+e3, e2+e1+e1+e2,  e3+e1+e1+e2,        \
        e3+e1+e1+e3, e3+e2+e2+e1, e3+e2+e2+e3, e3+e3+e1+e1, e3+e3+e1+e2, e3+e3+e2+e2  ]


        neighbour_of_atom_periodic_DYNAMIC_Helper = copy.deepcopy(neighbour_of_atom_periodic)
        neighbour_of_atom_NONperiodic_DYNAMIC_Helper = copy.deepcopy(neighbour_of_atom_NONperiodic)


        for current_atom_i in neighbour_of_atom_NONperiodic_DYNAMIC_Helper.keys():


            current_atom_i_coords = geo.coords[current_atom_i]

            atoms_in_the_same_molecule = neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i]

            for other_atom_j in neighbour_of_atom_periodic_DYNAMIC_Helper.keys():

                other_atom_j_coords = geo_periodic.coords[other_atom_j]

                for translation_vector_k in possible_translation_vectors:
                    current_atom_i_coords_transl_k = current_atom_i_coords + translation_vector_k

                    if np.all(np.isclose(current_atom_i_coords_transl_k, other_atom_j_coords, 0.01) ):

                        atoms_in_the_same_molecule.append(other_atom_j)
                        atoms_in_the_same_molecule += neighbour_of_atom_periodic_DYNAMIC_Helper[other_atom_j]

                        atoms_in_the_same_molecule = list(set(atoms_in_the_same_molecule))

                        neighbour_of_atom_periodic_DYNAMIC_Helper[other_atom_j] += atoms_in_the_same_molecule
                        neighbour_of_atom_periodic_DYNAMIC_Helper[other_atom_j] = list(set(neighbour_of_atom_periodic_DYNAMIC_Helper[other_atom_j]))


                neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] +=atoms_in_the_same_molecule

                neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] = list(set(neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] ))



        for current_atom_i in neighbour_of_atom_NONperiodic_DYNAMIC_Helper.keys():
            atoms_in_the_same_molecule = [current_atom_i]
            atoms_in_the_same_molecule += neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i]

            atoms_in_the_same_molecule = list(set(atoms_in_the_same_molecule))

            for other_atom_j in neighbour_of_atom_NONperiodic_DYNAMIC_Helper.keys():

                if self.common_data( neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] ,
                                 neighbour_of_atom_NONperiodic_DYNAMIC_Helper[other_atom_j]  ) :

                    atoms_in_the_same_molecule.append(other_atom_j)
                    atoms_in_the_same_molecule += neighbour_of_atom_NONperiodic_DYNAMIC_Helper[other_atom_j]

                    atoms_in_the_same_molecule = list(set(atoms_in_the_same_molecule))

                    neighbour_of_atom_NONperiodic_DYNAMIC_Helper[other_atom_j] += atoms_in_the_same_molecule
                    neighbour_of_atom_NONperiodic_DYNAMIC_Helper[other_atom_j] = list(set(neighbour_of_atom_NONperiodic_DYNAMIC_Helper[other_atom_j]))

                neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] += atoms_in_the_same_molecule

                neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i] = \
                list(set(neighbour_of_atom_NONperiodic_DYNAMIC_Helper[current_atom_i]))



        for keyi in neighbour_of_atom_NONperiodic_DYNAMIC_Helper.keys():

            temp_helperlist =  neighbour_of_atom_NONperiodic_DYNAMIC_Helper[keyi]
            temp_helperlist_dynamic = copy.deepcopy(temp_helperlist)

            for k in temp_helperlist:
                if k not in geo.getIndicesOfAllAtoms():
                    temp_helperlist_dynamic.remove(k)


            neighbour_of_atom_NONperiodic_DYNAMIC_Helper[keyi] = copy.deepcopy(temp_helperlist_dynamic)



        
        return neighbour_of_atom_NONperiodic_DYNAMIC_Helper
            

    def MapMoleculeAsCloseAsPossibleToOrigin(self, molecule_geo , lattice_vectors, max_translation_vect_length = 2):
    
        # shifts a molecule as close to the origin as possible, in a PERIODIC STRUCTURE WITH LATTICE VECTORS lattice_vectors
        
        
        e1 = lattice_vectors[0]
        e2 = lattice_vectors[1]
        e3 = lattice_vectors[2]
        
        possible_translation_vectors = self.BuildPossibleTranslationVectors(\
                            e1=e1, e2=e2, e3=e3 , max_size_of_translation_vector=max_translation_vect_length)    
        
        geos_list = []
        distances_to_origin_list = []
        translation_vectors_list = []

        for translation_vector_x in possible_translation_vectors:

            new_molecule_geo_x = copy.deepcopy(molecule_geo)
            new_molecule_geo_x.transform(t=translation_vector_x, R=np.eye(3))

            center_of_mass_coordinates_x = new_molecule_geo_x.getCenterOfMass()

            distances_to_origin_x = np.linalg.norm(center_of_mass_coordinates_x)


            if np.all(center_of_mass_coordinates_x>=0):

                distances_to_origin_list.append(distances_to_origin_x)
                geos_list.append(new_molecule_geo_x)
                translation_vectors_list.append(translation_vector_x)

        origin_distance_sorted_indices = np.argsort(np.array(distances_to_origin_list))

        origin_distance_sorted_geos_list = list( np.array(geos_list)[origin_distance_sorted_indices] )
        
        origin_distance_sorted_translation_vectors_list = \
        list( np.array( translation_vectors_list )[origin_distance_sorted_indices]  )
        
        
        ideal_translation_vector = origin_distance_sorted_translation_vectors_list[0]
        geo_closest_to_origin = origin_distance_sorted_geos_list[0]
        
        

        return ideal_translation_vector , geo_closest_to_origin


                
    def ReassambleMoleculesTornInUnitCell(self, geo, max_size_of_translation_vector=3, epsiolon=0.2 ):
    
        # this function reassambles atoms in a unit cell so that one will obtain coherent molecules (if there are coherent molecules in the periodic structure) ...
        # ... EVEN IF THE MOLECULE IS TORN APPART IN A UNIT CELL, ...
        # ... I.E. IF ONLY FOR THE PERIODIC REPLICATION IT IS OBVIOUS WHICH ATOMS BELONG TO THE SAME MOLECULE
        
        # max_size_of_translation_vectors gives the number of periodic replica in which one check for coherent molecules. usually 3 is ok
        # epsilon defines the range of the length of a covalent bond between 2 atoms so that the 2 atoms are considered to belong to the same molecule
 
    
        assert np.any(geo.lattice_vectors is not np.array([0,0,0])  )
    
        e1 = geo.lattice_vectors[0]
        e2 = geo.lattice_vectors[1]
        e3 = geo.lattice_vectors[2]

        possible_translation_vectors = self.BuildPossibleTranslationVectors(\
            e1=e1, e2=e2, e3=e3,max_size_of_translation_vector=max_size_of_translation_vector)


        atoms_in_the_same_molecule_PERIODIC = self.clusterAtomsInSameMoleculePeriodic(geo=geo, epsilon=epsiolon)

        atoms_in_the_same_molecule_NONPERIODIC = self.clusterAtomsInSameMoleculeNONPeriodic(geo=geo, epsilon=epsiolon)


        new_geo = copy.deepcopy(geo)
        
        for atom_i in atoms_in_the_same_molecule_NONPERIODIC.keys():

            all_atoms_in_molecule_NONPERIODIC_i = atoms_in_the_same_molecule_NONPERIODIC[atom_i]
            all_atoms_in_molecule_PERIODIC_i = atoms_in_the_same_molecule_PERIODIC[atom_i]

            if len(all_atoms_in_molecule_NONPERIODIC_i) is not len(all_atoms_in_molecule_PERIODIC_i):

                for translation_vector_k in possible_translation_vectors:

                    #figure_ik, axes_ik = plt.subplots(nrows=1, ncols=2, figsize=[10,5])

                    temp_geo_ik = copy.deepcopy(new_geo)



                    temp_geo_ik.transform(t=translation_vector_k,R=np.eye(3),\
                                          atom_indices = all_atoms_in_molecule_NONPERIODIC_i)


                    #temp_geo_ik.visualizeAtomIndices(ax=axes_ik[1])
                    #temp_geo_ik.visualize(ax=axes_ik[1])

                    #new_geo.visualizeAtomIndices(ax=axes_ik[0])
                    #new_geo.visualize(ax=axes_ik[0])
                    #plt.show()

                    temp_atoms_in_the_same_molecule_NONPERIODIC = self.clusterAtomsInSameMoleculeNONPeriodic(temp_geo_ik)
                    temp_all_atoms_in_molecule_NONPERIODIC_i_k = temp_atoms_in_the_same_molecule_NONPERIODIC[atom_i]
                    #print('all_atoms_in_molecule_NONPERIODIC_i: ',all_atoms_in_molecule_NONPERIODIC_i)
                    #print('temp_all_atoms_in_molecule_NONPERIODIC_i_k: ' ,temp_all_atoms_in_molecule_NONPERIODIC_i_k)

                    if len(temp_all_atoms_in_molecule_NONPERIODIC_i_k ) > len(all_atoms_in_molecule_NONPERIODIC_i ):



                        new_geo = copy.deepcopy(temp_geo_ik)
                        all_atoms_in_molecule_NONPERIODIC_i = temp_all_atoms_in_molecule_NONPERIODIC_i_k

                        atoms_in_the_same_molecule_NONPERIODIC = self.clusterAtomsInSameMoleculeNONPeriodic(geo=new_geo)


                    ##### shift the new part back to origin as close as possible

                        #print('\n\n NOW WE shift back close to the origin: ')
                        molecule_to_shift = new_geo.getAtomsByIndices(all_atoms_in_molecule_NONPERIODIC_i)


                        molecule_to_origin_translation_vector = \
                        self.MapMoleculeAsCloseAsPossibleToOrigin(molecule_geo=molecule_to_shift,\
                                                         lattice_vectors=new_geo.lattice_vectors,\
                                                         max_translation_vect_length=max_size_of_translation_vector)[0]



                        new_geo.transform(R=np.eye(3), t=molecule_to_origin_translation_vector,
                                         atom_indices=all_atoms_in_molecule_NONPERIODIC_i)


                        break

        return new_geo                   
        
        
class AimsGeometry(Geometry):
    def parse_geometry(self, text):
        atom_lines = []
        isFractional = False
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
                #                    if (ind_line < len(text_lines) - 1):
                #                        next_line = text_lines[ind_line+1]
                #                        if ('constrain_relaxation' in next_line) and ('.true.' in next_line.lower()):
                #                            self.constrain_relax.append(True)
                #                        else:
                #                            self.constrain_relax.append(False)
                #                    else:
                #                        self.constrain_relax.append(False)
                if 'lattice_vector' in line:
                    lattice_vector_lines.append(line)
                # c Check for fractional coordinates
                if '_frac' in line:
                    isFractional = True
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
        if isFractional:
            self.coords = ut.getCartesianCoords(self.coords, self.lattice_vectors)
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
    
    
    def parseTextZMatrix(self, text):
        species, coords = ZMatrixUtils.convertZMatrixToCartesian(*ZMatrixUtils.parseZMatrix(text))
        print(species, coords)
        self.add_atoms(coords, species)
    
                             

if __name__ == '__main__':
    pass

