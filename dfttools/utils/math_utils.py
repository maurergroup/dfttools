import numpy as np

def get_rotation_matrix(vec_start: np.array, vec_end: np.array) -> np.array:
    '''
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
    '''
    assert np.isclose(
        np.linalg.norm(vec_start),
        1
    ) and np.isclose(
        np.linalg.norm(vec_end),
        1
    ), "vec_start and vec_end must be unit vectors!"
    v = np.cross(vec_start,vec_end)
    c = np.dot(vec_start,vec_end)
    v_x = np.array([[0,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0]])
    R = np.eye(3) + v_x + v_x.dot(v_x)/(1+c)
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
    skew = np.array([[0,            axis_vec[2],    -axis_vec[1]],
                     [-axis_vec[2], 0,              axis_vec[0]],
                     [axis_vec[1],  -axis_vec[0],   0]],
                    dtype=np.float64)

    R = ddt + np.cos(phi) * (eye - ddt) + np.sin(phi) * skew
    return R


def get_fractional_coords(cartesian_coords: np.array, lattice_vectors: np.array) -> np.array:
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