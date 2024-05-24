# import unittest
# import os
# from dfttools.geometry import AimsGeometry


# class TestParserAndWriter(unittest.TestCase):
#     def _get_geometry(self):
#         geometry = AimsGeometry()
#         geometry.add_atoms(
#             cartesian_coords=[
#                 [0, 0, 0],
#                 [1, 0, 0],
#             ],
#             species=["H", "H"]
#         )

#         return geometry


#     def test_save_and_read_file(self):
#         geometry = self._get_geometry()
#         geometry.save_to_file('temp/geometry.in')

#         geometry_read = AimsGeometry('temp/geometry.in')
#         os.remove('temp/geometry.in')

#         self.assertTrue(geometry == geometry_read)


# if __name__ == '__main__':
#     unittest.main()
