from typing import Union
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import scipy


def get_rotation_matrix(vec_start: np.array, vec_end: np.array) -> np.array:
    """
    Given a two (unit) vectors, vec_start and vec_end, this function calculates
    the rotation matrix U, so that
     U * vec_start = vec_end.

    U the is rotation matrix that rotates vec_start to point in the direction
    of vec_end.

    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677

    Parameters
    ----------
        vec_start, vec_end <np.array<float>>: array of shape (3,). Represent
            the two vectors and must have l2-norm of 1!

    Returns:
    --------
        The rotation matrix U as np.array with shape (3,3)
    """
    assert np.isclose(np.linalg.norm(vec_start), 1) and np.isclose(
        np.linalg.norm(vec_end), 1
    ), "vec_start and vec_end must be unit vectors!"
    v = np.cross(vec_start, vec_end)
    c = np.dot(vec_start, vec_end)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + v_x + v_x.dot(v_x) / (1 + c)
    return R


def get_rotation_matrix_around_axis(axis: np.array, phi: float) -> np.array:
    """
    Generates a rotation matrix around a given vector.

    Parameters
    ----------
    axis : np.array
        Axis around which the rotation is done.
    phi : float
        Angle of rotation around axis in radiants.

    Returns
    -------
    R : np.array
        Rotation matrix

    """
    axis_vec = np.array(axis, dtype=np.float64)
    axis_vec /= np.linalg.norm(axis_vec)

    eye = np.eye(3, dtype=np.float64)
    ddt = np.outer(axis_vec, axis_vec)
    skew = np.array(
        [
            [0, axis_vec[2], -axis_vec[1]],
            [-axis_vec[2], 0, axis_vec[0]],
            [axis_vec[1], -axis_vec[0], 0],
        ],
        dtype=np.float64,
    )

    R = ddt + np.cos(phi) * (eye - ddt) + np.sin(phi) * skew
    return R


def get_rotation_matrix_around_z_axis(phi: float) -> np.array:
    """
    Generates a rotation matrix around the z axis.

    Parameters
    ----------
    phi : float
        Angle of rotation around axis in radiants.

    Returns
    -------
    np.array
        Rotation matrix

    """
    return get_rotation_matrix_around_axis(np.array([0.0, 0.0, 1.0]), phi)


def get_mirror_matrix(normal_vector: np.array) -> np.array:
    """
    Generates a transformation matrix for mirroring through plane given by the
    normal vector.

    Parameters
    ----------
    normal_vector : np.array
        Normal vector of the mirror plane.

    Returns
    -------
    M : np.array
        Mirror matrix

    """
    n_vec = normal_vector / np.linalg.norm(normal_vector)
    eps = np.finfo(np.float64).eps
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    M = np.array(
        [
            [1 - 2 * a**2, -2 * a * b, -2 * a * c],
            [-2 * a * b, 1 - 2 * b**2, -2 * b * c],
            [-2 * a * c, -2 * b * c, 1 - 2 * c**2],
        ]
    )
    M[np.abs(M) < eps * 10] = 0
    return M


def get_angle_between_vectors(vector_1: np.array, vector_2: np.array) -> np.array:
    """
    Determines angle between two vectors.

    Parameters
    ----------
    vector_1 : np.array
    vector_2 : np.array

    Returns
    -------
    angle : float
        Angle in radiants.

    """
    angle = (
        np.dot(vector_1, vector_2) / np.linalg.norm(vector_1) / np.linalg.norm(vector_2)
    )
    return angle


def get_fractional_coords(
    cartesian_coords: np.array, lattice_vectors: np.array
) -> np.array:
    """
    Transform cartesian coordinates into fractional coordinates.

    Parameters
    ----------
    cartesian_coords: [N x N_dim] numpy array
        Cartesian coordinates of atoms (can be Nx2 or Nx3)
    lattice_vectors: [N_dim x N_dim] numpy array:
        Matrix of lattice vectors: Each ROW corresponds to one lattice vector!

    Returns
    -------
    fractional_coords: [N x N_dim] numpy array
        Fractional coordinates of atoms

    """
    fractional_coords = np.linalg.solve(lattice_vectors.T, cartesian_coords.T)
    return fractional_coords.T


def get_cartesian_coords(frac_coords: np.array, lattice_vectors: np.array) -> np.array:
    """
    Transform fractional coordinates into cartesian coordinates.

    Parameters
    ----------
    frac_coords: [N x N_dim] numpy array
        Fractional coordinates of atoms (can be Nx2 or Nx3)
    lattice_vectors: [N_dim x N_dim] numpy array:
        Matrix of lattice vectors: Each ROW corresponds to one lattice vector!

    Returns
    -------
    cartesian_coords: [N x N_dim] numpy array
        Cartesian coordinates of atoms

    """
    return np.dot(frac_coords, lattice_vectors)


def get_cross_correlation_function(
    signal_0: np.array,
    signal_1: np.array,
    detrend: bool = False,
) -> np.array:
    """
    Calculate the autocorrelation function for a given signal.

    Parameters
    ----------
    signal_0 : 1D np.array
        First siganl for which the correlation function should be calculated.
    signal_1 : 1D np.array
        Second siganl for which the correlation function should be calculated.

    Returns
    -------
    correlation : np.array
        Autocorrelation function from 0 to max_lag.

    """
    if detrend:
        signal_0 = scipy.signal.detrend(signal_0)
        signal_1 = scipy.signal.detrend(signal_1)

    # cross_correlation = np.correlate(signal_0, signal_1, mode='same')
    cross_correlation = np.correlate(signal_0, signal_1, mode="full")
    cross_correlation = cross_correlation[cross_correlation.size//2:]
    
    # normalize by number of overlapping data points
    cross_correlation /= np.arange(cross_correlation.size, 0, -1)
    cutoff = int(cross_correlation.size*0.75)
    cross_correlation = cross_correlation[:cutoff]

    return cross_correlation


def get_autocorrelation_function_manual_lag(signal: np.array, max_lag: int) -> np.array:
    """
    Alternative method to determine the autocorrelation function for a given
    signal that used numpy.corrcoef. This function allows to set the lag
    manually.

    Parameters
    ----------
    signal : 1D np.array
        Siganl for which the autocorrelation function should be calculated.
    max_lag : Union[None, int], optional
        Autocorrelation will be calculated for a range of 0 to max_lag,
        where max_lag is the largest lag for the calculation of the
        autocorrelation function. The default is None.

    Returns
    -------
    autocorrelation : np.array
        Autocorrelation function from 0 to max_lag.

    """
    lag = np.array(range(max_lag))

    autocorrelation = np.array([np.nan] * max_lag)

    for l in lag:
        if l == 0:
            corr = 1.0
        else:
            corr = np.corrcoef(signal[l:], signal[:-l])[0][1]

        autocorrelation[l] = corr

    return autocorrelation


def get_fourier_transform(signal: np.array, time_step: float) -> tuple:
    """
    Calculate the fourier transform of a given siganl.

    Parameters
    ----------
    signal : 1D np.array
        Siganl for which the autocorrelation function should be calculated.
    time_step : float
        Time step of the signal in seconds.

    Returns
    -------
    (np.array, np.array)
        Frequencs and absolute values of the fourier transform.

    """
    # d = len(signal) * time_step

    f = scipy.fft.fftfreq(signal.size, d=time_step)
    y = scipy.fft.fft(signal)

    L = f >= 0

    return f[L], y[L]


def lorentzian(
    x: Union[float, np.array], a: float, b: float, c: float
) -> Union[float, np.array]:
    """
    Returns a Lorentzian function.

    Parameters
    ----------
    x : Union[float, np.array]
        Argument x of f(x) --> y.
    a : float
        Maximum of Lorentzian.
    b : float
        Width of Lorentzian.
    c : float
        Magnitude of Lorentzian.

    Returns
    -------
    f : Union[float, np.array]
        Outupt of a Lorentzian function.

    """
    #f = c / (np.pi * b * (1.0 + ((x - a) / b) ** 2))  # +d
    f = c / (1.0 + ((x - a) / (b/2.0) ) ** 2)  # +d

    return f


def norm_matrix_by_dagonal(matrix: np.array) -> np.array:
    """
    Norms a matrix such that the diagonal becomes 1.

    | a_11 a_12 a_13 |       |   1   a'_12 a'_13 |
    | a_21 a_22 a_23 |  -->  | a'_21   1   a'_23 |
    | a_31 a_32 a_33 |       | a'_31 a'_32   1   |

    Parameters
    ----------
    matrix : np.array
        Matrix that should be normed.

    Returns
    -------
    matrix : np.array
        Normed matrix.

    """
    diagonal = np.array(np.diagonal(matrix))
    L = diagonal == 0.0
    diagonal[L] = 1.0

    new_matrix = deepcopy(matrix)
    new_matrix /= np.sqrt(
        np.tile(diagonal, (matrix.shape[1], 1)).T
        * np.tile(diagonal, (matrix.shape[0], 1))
    )

    return new_matrix
    
    
def mae(delta: np.ndarray) -> float:
    """
    Calculated the mean absolute error from a list of value differnces.

    Parameters
    ----------
    delta : np.ndarray
        Array containing differences

    Returns
    -------
    float
        mean absolute error

    """
    return np.mean(np.abs(delta))


def rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    """
    Calculated the relative mean absolute error from a list of value differnces,
    given the target values.

    Parameters
    ----------
    delta : np.ndarray
        Array containing differences
    target_val : np.ndarray
        Array of target values against which the difference should be compared

    Returns
    -------
    float
        relative mean absolute error

    """
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9)


def rmse(delta: np.ndarray) -> float:
    """
    Calculated the root mean sqare error from a list of value differnces.

    Parameters
    ----------
    delta : np.ndarray
        Array containing differences

    Returns
    -------
    float
        root mean square error

    """
    return np.sqrt(np.mean(np.square(delta)))


def rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    """
    Calculated the relative root mean sqare error from a list of value differnces,
    given the target values.

    Parameters
    ----------
    delta : np.ndarray
        Array containing differences
    target_val : np.ndarray
        Array of target values against which the difference should be compared

    Returns
    -------
    float
        relative root mean sqare error

    """
    target_norm = np.sqrt(np.mean(np.square(target_val)))
    return np.sqrt(np.mean(np.square(delta))) / (target_norm + 1e-9)
    
    
def get_moving_average(signal: npt.NDArray[np.float64], window_size: int):
    """
    Cacluated the moving average and the variance around the moving average.

    Parameters
    ----------
    signal : npt.NDArray[np.float64]
        Signal for which the moving average should be calculated.
    window_size : int
        Window size for the mocing average.

    Returns
    -------
    moving_avg : TYPE
        Moving average.
    variance : TYPE
        Variance around the moving average.

    """
    moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    variance = np.array([np.var(signal[i:i+window_size]) for i in range(len(signal) - window_size + 1)])
    
    return moving_avg, variance
    
    
    
    
    