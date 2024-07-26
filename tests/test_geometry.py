import pytest
import os
import numpy as np
from dfttools.geometry import AimsGeometry


@pytest.fixture
def get_geometry():
    geometry = AimsGeometry()
    geometry.add_atoms(
        cartesian_coords=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        species=["H", "H"],
        constrain_relax=[np.array([False, False, False]),
                         np.array([True, True, True])]
    )

    return geometry

@pytest.fixture
def get_geometry_periodic():
    geometry = AimsGeometry()
    
    geometry.lattice_vectors = np.array([[1.27349850, 2.20576400, 0.00000000],
                                         [-1.27349850, 2.20576400, 0.00000000],
                                         [0.00000000, 0.00000000, 68.33693789]])
    
    geometry.add_atoms(
        cartesian_coords=[
            [-0.00000002, 1.47049920, 0.00000000],
            [0.00000000, 0.00000000, 2.07961400],
            [0.00000000, 2.94102000, 4.15922800],
            [-0.00000002, 1.47049920, 6.23160806],
            [0.00000002, -0.00000809, 8.30498122],
        ],
        species=["Cu", "Cu", "Cu", "Cu", "Cu",],
        constrain_relax=[np.array([True, True, True]), 
                         np.array([True, True, True]), 
                         np.array([True, True, True]), 
                         np.array([False, False, False]),
                         np.array([False, False, False]),
                         ]
    )

    return geometry

def test_save_and_read_file(get_geometry):
    geometry = get_geometry
    geometry.save_to_file('temp/geometry.in')

    geometry_read = AimsGeometry('temp/geometry.in')
    os.remove('temp/geometry.in')

    assert geometry == geometry_read

def test_get_displaced_atoms(get_geometry):
    geometry = get_geometry
    geometry_new = geometry.get_displaced_atoms(1)
    
    assert not np.allclose( geometry.coords[0], geometry_new.coords[0] )
    assert np.allclose( geometry.coords[1], geometry_new.coords[1] )
    
    
def test_get_symmetries(get_geometry_periodic):
    geometry = get_geometry_periodic
    
    symmetries = geometry.get_symmetries(symmetry_precision=1e-03)
    
    assert len(symmetries['rotations']) == 6
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    