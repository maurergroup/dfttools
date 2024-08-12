import os
import warnings
import numpy as np
import dfttools.utils.math_utils as mu
from ast import literal_eval
from scipy.optimize import curve_fit, root
from scipy.signal import csd, argrelextrema
from numba import jit, njit, prange
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# get environment variable for parallelisation in numba
parallel_numba = os.environ.get('parallel_numba')
if parallel_numba is None:
    warnings.warn('System variable <parallel_numba> not set. Using default!')
    parallel_numba = True
else:
    parallel_numba = literal_eval(parallel_numba)


def get_cross_correlation_function(signal_0: np.array,
                                   signal_1: np.array,
                                   bootstrapping_blocks: int=1) -> np.array:
    
    assert signal_0.size == signal_1.size, f'The parameters signal_0 and signal_1 \
        must have the same size but they are {signal_0.size} and {signal_1.size}.'
    
    signal_length = len(signal_0)
    block_size = int(np.floor(signal_length/bootstrapping_blocks))
    
    cross_correlation = []
    
    for block in range(bootstrapping_blocks):
        
        block_start = block*block_size
        block_end = (block+1)*block_size
        if block_end > signal_length:
            block_end = signal_length
    
        signal_0_block = signal_0[block_start:block_end]
        signal_1_block = signal_1[block_start:block_end]
        
        cross_correlation_block = mu.get_cross_correlation_function(signal_0_block,
                                                                    signal_1_block) 
        cross_correlation.append(cross_correlation_block)
    
    cross_correlation = np.atleast_2d(cross_correlation)
    cross_correlation = np.mean(cross_correlation, axis=0)
    
    return cross_correlation


def get_cross_spectrum(signal_0: np.array,
                       signal_1: np.array,
                       time_step: float,
                       bootstrapping_blocks: int=1,
                       zero_padding: int=0,
                       cutoff_at_last_maximum: bool=False) -> (np.array, np.array):
    """
    Determine the cross spectrum for a given signal using bootstrapping:
        - Splitting the sigmal into blocks and for each block:
            * Determining the cross correlation function of the signal
            * Determining the fourire transform of the autocorrelation
              function to get the power spectrum for the block
        - Calculating the average power spectrum by averaging of the power
          spectra of all blocks

    Parameters
    ----------
    signal_0 : 1D np.array
        First siganl for which the correlation function should be calculated.
    signal_1 : 1D np.array
        Second siganl for which the correlation function should be calculated.
    time_step : float
        DESCRIPTION.
    bootstrapping_blocks : int, optional
        DESCRIPTION. The default is 1.
    zero_padding : int, optional
        Pad the cross correlation function with zeros to increase the frequency
        resolution of the FFT. This also avoids the effect of varying spectral
        leakage. However, it artificially broadens the resulting cross spectrum
        and introduces wiggles.
    cutoff_at_last_maximum : bool, optional
        Cut off the cross correlation function at the last maximum to hide
        spectral leakage.

    Returns
    -------
    frequencies : np.array
        Frequiencies of the power spectrum in units depending on the
        tims_step.
        
    cross_spectrum : np.array
        Power spectrum.

    """
    assert signal_0.size == signal_1.size, f'The parameters signal_0 and signal_1 \
        must have the same size but they are {signal_0.size} and {signal_1.size}.'
    
    signal_length = len(signal_0)
    block_size = int(np.floor(signal_length/bootstrapping_blocks))
    
    frequencies = None
    cross_spectrum = []
    
    for block in range(bootstrapping_blocks):
        
        block_start = block*block_size
        block_end = (block+1)*block_size
        if block_end > signal_length:
            block_end = signal_length
        
        signal_0_block = signal_0[block_start:block_end]
        signal_1_block = signal_1[block_start:block_end]
                
        cross_correlation = mu.get_cross_correlation_function(signal_0_block,
                                                              signal_1_block)
        
        # truncate cross correlation function at last maximum
        if cutoff_at_last_maximum:
            cutoff_index = get_last_maximum(cross_correlation)
            cross_correlation = cross_correlation[:cutoff_index]
            print(block, cutoff_index)
        
        # add zero padding
        if zero_padding < len(cross_correlation):
            zero_padding = len(cross_correlation)
            
        cross_correlation = np.pad(cross_correlation,
                                   (0, zero_padding - len(cross_correlation)),
                                   'constant')
        
        frequencies_block, cross_spectrum_block = mu.get_fourier_transform(cross_correlation,
                                                                     time_step)
        
        if block == 0:
            frequencies = frequencies_block
        else:
            f = interp1d(frequencies_block, cross_spectrum_block, kind='linear', fill_value="extrapolate")
            cross_spectrum_block = f(frequencies)
        
        cross_spectrum.append( np.abs(cross_spectrum_block) )
    
    cross_spectrum = np.atleast_2d(cross_spectrum)
    cross_spectrum = np.average(cross_spectrum, axis=0)

    return frequencies, cross_spectrum


def get_last_maximum(x: np.array):
    
    maxima = argrelextrema(x, np.greater_equal)[0]
    #roots = argrelextrema(-np.abs(x), np.greater_equal)[0]
    
    last_maximum = maxima[-1]
    
    if last_maximum == len(x)-1:
        last_maximum = maxima[-2]
    
    # plt.plot( x )
    # plt.plot( maxima, x[maxima], 'o' )
    # plt.plot( last_maximum, x[last_maximum], 'o' )
    
    return last_maximum
    

def lorentzian_fit(frequencies, power_spectrum, p_0=None):
    
    max_ind = np.argmax(power_spectrum)
    
    if p_0 is None:
        # determine reasonable starting parameters
        a_0 = frequencies[max_ind]
        b_0 = np.abs(frequencies[1]-frequencies[0])
        c_0 = np.max(power_spectrum)
        
        p_0 = [a_0, b_0, c_0]
    
    try:
        res, _ = curve_fit(mu.lorentzian,
                           frequencies,
                           power_spectrum,
                           p0=p_0)
    except RuntimeError:
        res = [np.nan, np.nan, np.nan]
    
    return res


def get_peak_parameters(frequencies, power_spectrum):
    
    max_ind = np.argmax(power_spectrum)
    frequency = frequencies[max_ind]
    
    half_max = power_spectrum[max_ind]/2.0
    
    f_interp = interp1d(frequencies, power_spectrum, kind='cubic')
    
    # Define a function to find roots (y - half_max)
    def f_half_max(x_val):
        return f_interp(x_val) - half_max

    # Find roots (i.e., the points where the function crosses the half maximum)
    root1 = brentq(f_half_max, frequencies[0], frequencies[max_ind])  # Left intersection
    root2 = brentq(f_half_max, frequencies[max_ind], frequencies[-1]) # Right intersection
    
    # Calculate the FWHM
    line_width = np.abs(root1 - root2)
    
    res = [frequency, line_width, power_spectrum[max_ind]]
    
    return res


def get_line_widths(frequencies, power_spectrum, use_lorentzian=True):
    res = [np.nan, np.nan, np.nan]
    
    if use_lorentzian:
        res = lorentzian_fit(frequencies, power_spectrum)
    
    if np.isnan(res[0]):
        res = get_peak_parameters(frequencies, power_spectrum)
        
    frequency = res[0]
    line_width = res[1]
    life_time = 1.0 / line_width
    
    return frequency, line_width, life_time


def get_normal_mode_decomposition(velocities: np.array,
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
    velocities_projected = np.zeros((velocities.shape[0],
                                    eigenvectors.shape[0]),
                                    dtype=np.complex128)
    
    # Get normal mode decompositon parallelised by numba
    _get_normal_mode_decomposition_numba(velocities_projected,
                                         velocities,
                                         eigenvectors.conj()) 

    return velocities_projected


@njit(parallel=parallel_numba, fastmath=True)
def _get_normal_mode_decomposition_numba(velocities_projected,
                                         velocities,
                                         eigenvectors) -> None:
    
    number_of_cell_atoms = velocities.shape[1]
    number_of_frequencies = eigenvectors.shape[0]
    
    for k in range(number_of_frequencies):
        for n in range( velocities.shape[0] ):
            for i in prange(number_of_cell_atoms):
                for m in prange( velocities.shape[2] ):
                    velocities_projected[n, k] += velocities[n, i, m] * eigenvectors[k, i, m]


def _get_normal_mode_decomposition_numpy(velocities_projected,
                                         velocities,
                                         eigenvectors) -> None:
    
    number_of_cell_atoms = velocities.shape[1]
    number_of_frequencies = eigenvectors.shape[0]
    
    #Projection in phonon coordinate
    for k in range(number_of_frequencies):
        for i in range(number_of_cell_atoms):
            velocities_projected[:, k] += np.dot(velocities[:, i, :], eigenvectors[k, i, :].conj())

