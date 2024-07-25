import numpy as np
import copy
import os
import dfttools.utils.units as units
import dfttools.utils.vibrations_utils as vu
from typing import Union
from dfttools.geometry import AimsGeometry, VaspGeometry, MoldenGeometry
from scipy.signal import argrelextrema
import multiprocessing as mp
import functools


class Vibrations():
    def __init__(self):
        self.vibration_coords = None
        self.vibration_forces = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.wave_vector = np.array([0.0, 0.0, 0.0])
        self.time_step = None
    
    
    def get_instance_of_other_type(self, vibrations_type):
        if vibrations_type == 'aims':
            new_vibration = AimsVibrations()
        if vibrations_type == 'vasp':
            new_vibration = VaspVibrations()
        if vibrations_type == 'molden':
            new_vibration = MoldenVibrations()
            
        new_vibration.__dict__ = self.__dict__
        return new_vibration
    
    
    def set_vibration_coords(self, vibration_coords: list) -> None:
        self.vibration_coords = vibration_coords
        
    
    def set_vibration_forces(self, vibration_forces : list) -> None:
        self.vibration_forces = vibration_forces
        
        
    def set_hessian(self, hessian: np.array) -> None:
        self.hessian = hessian
        
    
    def set_eigenvalues(self, eigenvalues: np.array) -> None:
        self.eigenvalues = eigenvalues
        
    
    def set_eigenvectros(self, eigenvectors: np.array) -> None:
        self.eigenvectors = eigenvectors
    
    
    def get_displacements(self, displacement: float=0.0025) -> list:
        """
        Applies a given displacement for each degree of freedom of self and
        generates a new vibration with it.

        Parameters
        ----------
        displacement : float, optional
            Displacement for finte difference calculation of vibrations in
            Angstrom. The default is 0.0025 Angstrom.

        Returns
        -------
        list
            List of geometries where atoms have been displaced.

        """
        geometries_displaced = [self]
        
        directions = [-1, 1]

        for i in range(self.n_atoms):
            for dim in range(3):
                if self.constrain_relax[i, dim]:
                    continue
                
                for direction in directions:
                    geometry_displaced = copy.deepcopy(self)
                    geometry_displaced.coords[i, dim] += displacement * direction
                    
                    geometries_displaced.append(geometry_displaced)
        
        return geometries_displaced
    
    
    def get_mass_tensor(self) -> np.array:
        """
        Determine a NxN tensor containing sqrt(m_i*m_j).

        Returns
        -------
        mass_tensor : np.array
            Mass tensor in atomic units.

        """
        mass_vector = [self.periodic_table.get_atomic_mass(s) for s in self.species]
        mass_vector = np.repeat(mass_vector, 3)
        
        mass_tensor = np.tile( mass_vector, (len(mass_vector),1) )
        mass_tensor = np.sqrt( mass_tensor * mass_tensor.T )
        
        return mass_tensor
    
    
    def get_hessian(self, set_constrained_atoms_zero: bool=False) -> np.array:
        """
        Calculates the Hessian from the forces. This includes the atmoic masses
        since F = m*a.

        Parameters
        ----------
        set_constrained_atoms_zero : bool, optional
            Set elements in Hessian that code for constrained atoms to zero.
            The default is False.

        Returns
        -------
        H : np.array
            Hessian.

        """
        N = len(self) * 3
        H = np.zeros([N, N])
        
        assert np.allclose(self.coords, self.vibration_coords[0]), 'The first entry in vibration_coords must be identical to the undispaced geometry.'
        
        coords_0 = self.vibration_coords[0].flatten()
        F_0 = self.vibration_forces[0].flatten()
        
        n_forces = np.zeros(N, np.int64)
        
        for c, F in zip(self.vibration_coords, self.vibration_forces):
            dF = F.flatten() - F_0
            dx = c.flatten() - coords_0
            ind = np.argmax(np.abs(dx))
            n_forces[ind] += 1
            displacement = dx[ind]
            
            if np.abs(displacement) < 1e-5:
                continue
            
            H[ind, :] -= dF / displacement

        for row in range(H.shape[0]):
            if n_forces[row] > 0:
                H[row, :] /= n_forces[row]  # prevent div by zero for unknown forces
        
        if set_constrained_atoms_zero:
            constrained = self.constrain_relax.flatten()
            H[constrained, :] = 0
            H[:, constrained] = 0
        
        self.set_hessian(H)
        return H


    def get_symmetrized_hessian(self, hessian=None):
        """
        Symmetrieses the Hessian by using the lower triangular matrix

        Parameters
        ----------
        hessian : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if hessian is None:
            hessian = copy.deepcopy(self.hessian)
        
        hessian_new = hessian + hessian.T
        
        all_inds = list(range(len(self)*3))
        
        constrain = self.constrain_relax.flatten()
        constrained_inds = [i for i, c in enumerate(constrain) if c]
        constrained_inds = np.array(constrained_inds)
        
        unconstrained_inds = np.array(list(set(all_inds) - set(constrained_inds)))
        
        for i in unconstrained_inds:
            for j in unconstrained_inds:
                hessian_new[i,j] *= 0.5
        
        return hessian_new
        #self.hessian = (h+np.transpose(h))/2
        

    def get_eigenvalues_and_eigenvectors(
        self,
        hessian: np.array=None,
        only_real: bool=True,
        symmetrize_hessian: bool=True
    ) -> Union[np.array, list, np.array]:
        """
        This function is supposed to return all eigenvalues and eigenvectors of
        the matrix self.hessian

        Parameters
        ----------
        hessian : np.array, optional
            Hessian. The default is None.
        only_real : bool, optional
            Returns only real valued eigenfrequencies + eigenmodes
            (ATTENTION: if you want to also include instable modes, you have to
            symmetrize the hessian as provided below). The default is True.
        symmetrize_hessian : bool, optional
            Symmetrise the hessian only for this function (no global change).
            The default is True.

        Returns
        -------
        omega2 : np.array
            Direct eigenvalues as squared angular frequencies instead of
            inverse wavelengths.
        eigenvectors : np.array
            List of numpy arrays, where each array is a normalized
            displacement for the corresponding eigenfrequency, such that
            new_coords = coords + displacement * amplitude..

        """
        
            
        if symmetrize_hessian:
            hessian = self.get_symmetrized_hessian(hessian=hessian)
        elif hessian is None:
            hessian = copy.deepcopy(self.hessian)
        
        assert hasattr(self,'hessian') and hessian is not None, \
            'Hessian must be given to calculate the Eigenvalues!'
        
        M = 1 / self.get_mass_tensor()
        
        omega2, X = np.linalg.eig(M * hessian)
        
        # only real valued eigen modes
        if only_real:
            real_mask = np.isreal(omega2)
            min_omega2 = 1e-3
            min_mask = omega2 >= min_omega2
            mask = np.logical_and(real_mask, min_mask)

            omega2 = np.real(omega2[mask])
            X = np.real(X[:, mask])

        eigenvectors = [column.reshape(-1, 3) for column in X.T]

        # sort modes by energies (ascending)
        ind_sort = np.argsort(omega2)
        eigenvectors = np.array(eigenvectors)[ind_sort, :, :]
        omega2 = omega2[ind_sort]
        
        self.set_eigenvalues(omega2)
        self.set_eigenvectros(eigenvectors)
        
        return omega2, eigenvectors

    
    def get_eigenvalues_in_Hz(self, omega2: Union[None, np.array]=None) -> np.array:
        """
        Determine vibration frequencies in cm^-1.
    
        Parameters
        ----------
        omega2 : Union[None, np.array]
            Eigenvalues of the mass weighted hessian.
    
        Returns
        -------
        omega_SI : np.array
            Array of the eigenfrequencies in Hz.
    
        """
        if omega2 is None:
            omega2 = self.eigenvalues
        
        omega = np.sign(omega2) * np.sqrt(np.abs(omega2))
        
        conversion = np.sqrt((units.EV_IN_JOULE) / (units.ATOMIC_MASS_IN_KG * units.ANGSTROM_IN_METER ** 2))
        omega_SI = omega * conversion
        
        return omega_SI

    
    def get_eigenvalues_in_inverse_cm(self, omega2: Union[None, np.array]=None) -> np.array:
        """
        Determine vibration frequencies in cm^-1.

        Parameters
        ----------
        omega2 : Union[None, np.array]
            Eigenvalues of the mass weighted hessian.

        Returns
        -------
        f_inv_cm : np.array
            Array of the eigenfrequencies in cm^(-1).

        """
        omega_SI = self.get_eigenvalues_in_Hz(omega2=omega2)
        f_inv_cm = omega_SI * units.INVERSE_CM_IN_HZ / (2 * np.pi)
        
        return f_inv_cm
        

    def get_atom_type_index(self):
        
        n_atoms = len(self)
        
        # Tolerance for accepting equivalent atoms in super cell
        masses = self.get_mass_of_all_atoms()
        tolerance = 0.001
        
        primitive_cell_inverse = np.linalg.inv(self.lattice_vectors)
    
        atom_type_index = np.array([None]*n_atoms)
        counter = 0
        for i in range(n_atoms):
            if atom_type_index[i] is None:
                atom_type_index[i] = counter
                counter += 1
            for j in range(i+1, n_atoms):
                coordinates_atom_i = self.coords[i]
                coordinates_atom_j = self.coords[j]

                difference_in_cell_coordinates = np.around((np.dot(primitive_cell_inverse.T, (coordinates_atom_j - coordinates_atom_i))))
                
                projected_coordinates_atom_j = coordinates_atom_j - np.dot(self.lattice_vectors.T, difference_in_cell_coordinates)
                separation = pow(np.linalg.norm(projected_coordinates_atom_j - coordinates_atom_i),2)

                if separation < tolerance and masses[i] == masses[j]:
                    atom_type_index[j] = atom_type_index[i]
                        
        atom_type_index = np.array(atom_type_index, dtype=int)
        
        return atom_type_index
    
    
    def get_velocity_mass_average(self, velocities: np.array) -> np.array:
        """
        Weighs velocities by atomic masses.

        Parameters
        ----------
        velocities : np.array

        Returns
        -------
        velocities_mass_average : np.array
            Velocities weighted by atomic masses.

        """
        velocities_mass_average = np.zeros_like(velocities)      

        for i in range(velocities.shape[1]):
            velocities_mass_average[:, i, :] = velocities[:, i, :] * np.sqrt(self.get_atomic_masses()[i])
            
        return velocities_mass_average
    
    
    def project_onto_wave_vector(
        self,
        velocities: np.array,
        wave_vector: np.array,
        project_on_atom: int=-1
    ) -> np.array:
        
        number_of_primitive_atoms = len(self)
        number_of_atoms = velocities.shape[1]
        number_of_dimensions = velocities.shape[2]

        coordinates = self.coords
        atom_type = self.get_atom_type_index()

        velocities_projected = np.zeros((velocities.shape[0],
                                         number_of_primitive_atoms,
                                         number_of_dimensions), dtype=complex)

        if wave_vector.shape[0] != coordinates.shape[1]:
            print("Warning!! Q-vector and coordinates dimension do not match")
            exit()

        # Projection onto the wave vector
        for i in range(number_of_atoms):
            # Projection on atom
            if project_on_atom > -1:
                if atom_type[i] != project_on_atom:
                    continue

            for k in range(number_of_dimensions):
                velocities_projected[:, atom_type[i], k] += velocities[:,i,k]*np.exp(-1j*np.dot(wave_vector, coordinates[i,:]))

        # Normalize the velocities
        number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms
        velocities_projected /= np.sqrt(number_of_primitive_cells)
        
        return velocities_projected
    
    
    def get_normal_mode_decomposition(
        self,
        velocities: np.array
    ) -> np.array:
        """
        Calculate the normal-mode-decomposition of the velocities. This is done
        by projecting the atomic velocities onto the vibrational eigenvectors.
        See equation 10 in: https://doi.org/10.1016/j.cpc.2017.08.017

        Parameters
        ----------
        velocities : np.array
            Array containing the velocities from an MD trajectory structured in
            the following way:
            [number of time steps, number of atoms, number of dimensions].

        Returns
        -------
        velocities_projected : np.array
            Velocities projected onto the eigenvectors structured as follows:
            [number of time steps, number of frequencies]

        """
        velocities = np.array(velocities, dtype=np.complex128)
        
        velocities_mass_averaged = self.get_velocity_mass_average(velocities)
        
        #velocities_proj_0 = vibrations.project_onto_wave_vector(velocities, q_vector)
        
        velocities_projected = vu.get_normal_mode_decomposition(velocities_mass_averaged,
                                                                self.eigenvectors)

        return velocities_projected
    
    
    def get_cross_correlation_function(
        self,
        velocities: np.array,
        time_step: float,
        bootstrapping_blocks: int=1
    ):
        """
        

        Parameters
        ----------
        velocities : np.array
            DESCRIPTION.
        time_step : float
            DESCRIPTION.
        bootstrapping_blocks : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        autocorr_t : np.array
            DESCRIPTION.
        autocorr : np.array
            DESCRIPTION.

        """
        velocities_proj = self.get_normal_mode_decomposition(velocities)
        
        n_points = len(self.eigenvectors)
        
        autocorr_t = None
        autocorr = {}
        
        for index_0 in range(n_points):
            for index_1 in range(n_points):
                autocorr_block \
                    = vu.get_cross_correlation_function(velocities_proj[:,index_0],
                                                        velocities_proj[:,index_1],
                                                        bootstrapping_blocks=bootstrapping_blocks)
                
                autocorr_t_block = np.linspace(0,
                                               len(autocorr_block)*time_step,
                                               len(autocorr_block))
                
                
                autocorr_t = autocorr_t_block
                autocorr[(index_0, index_1)] = autocorr_block
        
        return autocorr_t, autocorr
    
    
    def get_cross_spectrum(
        self,
        velocities: np.array,
        time_step: float,
        bootstrapping_blocks: int=1
    ):
        
        velocities_proj = self.get_normal_mode_decomposition(velocities)
        
        n_points = len(self.eigenvectors)
        
        frequencies = vu.get_cross_spectrum(velocities_proj[:,0],
                                            velocities_proj[:,0],
                                            time_step,
                                            bootstrapping_blocks=bootstrapping_blocks)[0]
        
        cross_spectrum = np.zeros((n_points, n_points, len(frequencies)), dtype=np.complex128)
        
        for index_0 in range(n_points):
            for index_1 in range(n_points):
                print(index_0, index_1)
                cross_spectrum_block \
                    = vu.get_cross_spectrum(velocities_proj[:,index_0],
                                            velocities_proj[:,index_1],
                                            time_step,
                                            bootstrapping_blocks=bootstrapping_blocks)[1]
                     
                cross_spectrum[index_0, index_1, :] = cross_spectrum_block
        
        return frequencies, cross_spectrum
    
    
    def output_cross_spectrum(
        self,
        velocities: np.array,
        time_step: float,
        bootstrapping_blocks: int=1,
        processes=1,
        frequency_cutoff=None,
        dirname='cross_spectrum',
    ):
        
        velocities_proj = self.get_normal_mode_decomposition(velocities)
        
        n_points = len(self.eigenvectors)
        
        frequencies = vu.get_cross_spectrum(velocities_proj[:,0],
                                            velocities_proj[:,0],
                                            time_step,
                                            bootstrapping_blocks=bootstrapping_blocks)[0]
        
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        cutoff = -1
        if not frequency_cutoff is None:
            f_inv_cm = frequencies * units.INVERSE_CM_IN_HZ
            L = f_inv_cm < frequency_cutoff
            cutoff = np.sum(L)
        
        np.savetxt(os.path.join(dirname, 'frequencies.csv'), frequencies[:cutoff])
        
        index = []
        for index_0 in range(n_points):
            for index_1 in range(n_points):
                if index_0 < index_1:
                    continue
                
                index.append((index_0, index_1))
        
                
        func = functools.partial(_output_cross_spectrum,
                                 velocities_proj=velocities_proj,
                                 time_step=time_step,
                                 bootstrapping_blocks=bootstrapping_blocks,
                                 cutoff=cutoff,
                                 dirname=dirname)
        
        with mp.Pool(processes) as pool:
            pool.map(func, index)
    
    
    def get_coupling_matrix(
        self,
        velocities: np.array,
        time_step: float,
        bootstrapping_blocks: int=1,
    ):
        
        omega = self.get_eigenvalues_in_Hz()
        
        frequencies, power_spectrum = self.get_cross_spectrum(velocities,
                                                              time_step,
                                                              bootstrapping_blocks=bootstrapping_blocks)
        
        n_points = len(self.eigenvectors)
        coupling_matrix = np.zeros((n_points, n_points))
        
        for index_0 in range(n_points):
            for index_1 in range(n_points):
                
                power_spectrum_index = np.real( power_spectrum[index_0, index_1, :] )
                
                max_f = argrelextrema(power_spectrum_index, np.greater)[0]
                
                index_search = np.max([index_0, index_1])
                
                f_couple = omega[index_search] / (2 * np.pi)
                
                coupling_index_0 = np.argmin( np.abs(frequencies[max_f] - f_couple) )
                coupling_index = max_f[coupling_index_0]
                                
                coupling_matrix[index_0, index_1] = power_spectrum_index[coupling_index]
                
        return coupling_matrix


def _output_cross_spectrum(
    index,
    velocities_proj,
    time_step,
    bootstrapping_blocks,
    cutoff,
    dirname
):
    index_0 = index[0]
    index_1 = index[1]
    cross_spectrum \
        = vu.get_cross_spectrum(velocities_proj[:,index_0],
                                velocities_proj[:,index_1],
                                time_step,
                                bootstrapping_blocks=bootstrapping_blocks)[1]
    
    np.savetxt(os.path.join(dirname, f'cross_spectrum_{index_0}_{index_1}.csv'),
               cross_spectrum[:cutoff])


class AimsVibrations(Vibrations, AimsGeometry):
    def __init__(self, filename=None):
        Vibrations.__init__(self)
        AimsGeometry.__init__(self, filename=filename)
        

class VaspVibrations(Vibrations, VaspGeometry):
    def __init__(self, filename=None):
        Vibrations.__init__(self)
        VaspGeometry.__init__(self, filename=filename)


class MoldenVibrations(Vibrations, MoldenGeometry):
    def __init__(self, filename=None):
        Vibrations.__init__(self)
        MoldenGeometry.__init__(self, filename=filename)


