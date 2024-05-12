import os
import numpy as np
from scipy.optimize import curve_fit
import copy
import warnings
from ast import literal_eval
from dfttools.utils.periodic_table import PeriodicTable
import dfttools.utils.math_utils as mu
import dfttools.utils.units as units
from typing import Union
from dfttools.geometry import AimsGeometry, VaspGeometry
from numba import jit, njit, prange

# get environment variable for parallelisation in numba
parallel_numba = os.environ.get('parallel_numba')
if parallel_numba is None:
    warnings.warn('System variable <parallel_numba> not set. Using default!')
    parallel_numba = True
else:
    parallel_numba = literal_eval(parallel_numba)


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
    
    
    def get_hessian(self, set_constrained_atoms_zero: bool=True) -> np.array:
        
        N = len(self) * 3
        H = np.zeros([N, N])
        
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


    def symmetrize_hessian(self, hessian=None):
        h = copy.deepcopy(self.hessian)
        self.hessian = (h+np.transpose(h))/2
        

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
        periodic_table = PeriodicTable()
        
        if hessian is None:
            hessian = self.hessian
        
        assert hasattr(self,'hessian') and hessian is not None, \
            'Hessian must be given to calculate the Eigenvalues!'
        try:
            masses = [periodic_table.get_atomic_mass(s) for s in self.species]
        except KeyError:
            print('getEigenValuesAndEigenvectors: Some Species were not known, used version without _ suffix')
            masses = [periodic_table.get_atomic_mass(s.split('_')[0]) for s in self.species]

        masses = np.repeat(masses, 3)
        M = np.diag(1.0 / masses)

        hessian = copy.deepcopy(hessian)
        if symmetrize_hessian:
            hessian = (hessian + np.transpose(hessian)) / 2

        omega2, X = np.linalg.eig(M.dot(hessian))
        
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

    
    def project_onto_wave_vector(self, velocities, wave_vector, project_on_atom=-1):
        
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
    
    
    def get_normal_mode_decomposition(self,
                                      velocities: np.array,
                                      eigenvectors: np.array) -> np.array:
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
        eigenvectors : np.array
            Array of eigenvectors structured in the following way:
            [number of frequencies, number of atoms, number of dimensions].

        Returns
        -------
        velocities_projected : np.array
            Velocities projected onto the eigenvectors structured as follows:
            [number of time steps, number of frequencies]

        """
        # Projection in vibration coordinates
        velocities_projected = np.zeros((velocities.shape[0], eigenvectors.shape[0]), dtype=complex)
        
        # Get normal mode decompositon parallelised by numba
        _get_normal_mode_decomposition_numba(velocities_projected,
                                             velocities,
                                             eigenvectors) 

        return velocities_projected
    
    
    def get_autocorrelation_function(self,
                                     signal: np.array,
                                     bootstrapping_blocks: int=1) -> np.array:
        
        signal_length = len(signal)
        
        autocorrelation = []
        
        block_size = int(signal_length/bootstrapping_blocks)
        
        for block in range(bootstrapping_blocks):
            
            block_start = block*block_size
            block_end = (block+1)*block_size
            if block_end > signal_length-1:
                block_end = signal_length-1
        
            signal_block = signal[block_start:block_end]
            
            autocorrelation_block = mu.get_autocorrelation_function(signal_block) 
            autocorrelation.append(autocorrelation_block)
        
        autocorrelation = np.atleast_2d(autocorrelation)
        
        autocorrelation = np.mean(autocorrelation, axis=0)
        
        return autocorrelation
    
    
    def get_power_spectrum(self,
                           signal: np.array,
                           time_step: float,
                           bootstrapping_blocks: int=1) -> (np.array, np.array):
        """
        Determine the power spectrum for a given signal using bootstrapping:
            - Splitting the sigmal into blocks and for each block:
                * Determining the autocorrelation function of the signal
                * Determining the fourire transform of the autocorrelation
                  function to get the power spectrum for the block
            - Calculating the average power spectrum by averaging of the power
              spectra of all blocks

        Parameters
        ----------
        signal : np.array
            Siganl for which the autocorrelation function should be calculated.
        time_step : float
            DESCRIPTION.
        bootstrapping_blocks : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        frequencies : np.array
            Frequiencies of the power spectrum in units depending on the
            tims_step.
            
        power_spectrum : np.array
            Power spectrum.

        """
        signal_length = len(signal)
        block_size = int(np.floor(signal_length/bootstrapping_blocks))
        
        print(signal_length, block_size)
        
        frequencies = None
        power_spectrum = []
        
        for block in range(bootstrapping_blocks):
            
            block_start = block*block_size
            block_end = (block+1)*block_size
            if block_end > signal_length:
                block_end = signal_length
        
            signal_block = signal[block_start:block_end]
            autocorrelation_block = mu.get_autocorrelation_function(signal_block) 
            frequencies, power_spectrum_block = mu.get_fourier_transform(autocorrelation_block, time_step)
            power_spectrum.append(power_spectrum_block)
        
        power_spectrum = np.atleast_2d(power_spectrum)
        power_spectrum = np.average(power_spectrum, axis=0)

        return frequencies, power_spectrum
    
    
    def lorentzian_fit(self, frequencies, power_spectrum):
        
        max_ind = np.argmax(power_spectrum)
        
        frequency_step = frequencies[1]-frequencies[0]
        
        a_0 = frequencies[max_ind]
        b_0 = frequency_step*100
        c_0 = np.max(power_spectrum) * b_0 * np.pi
        
        p_0 = [a_0, b_0, c_0]
        
        res, _ = curve_fit(mu.lorentzian,
                           frequencies,
                           power_spectrum,
                           p0=p_0)
        
        return res
    
    
    def get_line_widths(self, frequencies, power_spectrum):
        
        res = self.lorentzian_fit(frequencies, power_spectrum)
        
        frequency = res[0]
        line_width = res[1]
        life_time = 1.0 / line_width
        
        return frequency, line_width, life_time


# helper functions
@njit(parallel=parallel_numba, fastmath=True)
def _get_normal_mode_decomposition_numba(velocities_projected,
                                         velocities,
                                         eigenvectors) -> None:
    
    number_of_cell_atoms = velocities.shape[1]
    number_of_frequencies = eigenvectors.shape[0]
    
    for k in prange(number_of_frequencies):
        for i in prange(number_of_cell_atoms):
            for n in range( velocities.shape[0] ):
                for m in range( velocities.shape[2] ):
                    velocities_projected[n, k] += velocities[n, i, m] * eigenvectors[k, i, m]


class AimsVibrations(Vibrations, AimsGeometry):
    def __init__(self, filename=None):
        Vibrations.__init__(self)
        AimsGeometry.__init__(self, filename=filename)
        

class VaspVibrations(Vibrations, VaspGeometry):
    def __init__(self, filename=None):
        Vibrations.__init__(self)
        VaspGeometry.__init__(self, filename=filename)





