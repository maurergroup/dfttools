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


def test_save_and_read_file(get_geometry):
    geometry = get_geometry
    geometry.save_to_file('temp/geometry.in')

    geometry_read = AimsGeometry('temp/geometry.in')
    os.remove('temp/geometry.in')

    assert geometry == geometry_read


def test_get_displacement_of_atoms(get_geometry):
    geometry = get_geometry
    geometry_new = geometry.get_displacement_of_atoms(1)
    
    assert not np.allclose( geometry.coords[0], geometry_new.coords[0] )
    assert np.allclose( geometry.coords[1], geometry_new.coords[1] )