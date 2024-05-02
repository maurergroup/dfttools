#    Copyright (C) 2024 Oliver T. Hofmann
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
#    If you are using this software for scientific purposes, please cite it as:
#    L Hoermann et al., Computer Physics Communications (2019), 143-155 
"""This module contains unit tests fot the GeometryFile class

Author:
    Johannes Cartus, 03.09.2019
"""

import unittest
import os
from dfttools.geometry import AimsGeometry


class TestParserAndWriter(unittest.TestCase):
    def _get_geometry(self):
        geometry = AimsGeometry()
        geometry.add_atoms(
            cartesian_coords=[
                [0, 0, 0],
                [1, 0, 0],
            ],
            species=["H", "H"]
        )
        
        return geometry
    
        
    def test_save_and_read_file(self):
        geometry = self._get_geometry()
        geometry.save_to_file('temp/geometry.in')
        
        geometry_read = AimsGeometry('temp/geometry.in')
        os.remove('temp/geometry.in')
        
        self.assertTrue(geometry == geometry_read)
        
        


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    