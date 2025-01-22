from copy import deepcopy
import numpy as np
from collections.abc import Iterable

from dfttoolkit.parameters import CubefileParameters
from matplotlib.cm import get_cmap

from dfttoolkit.geometry import Geometry
from dfttoolkit.utils.math_utils import get_triple_product
from dfttoolkit.utils.periodic_table import PeriodicTable
from dfttoolkit.utils.units import BOHR_IN_ANGSTROM, EPSILON0_AIMS
from scipy.ndimage.interpolation import shift
import copy
import os


class Cubefile:
    """Read (and in the future also interpolate) 3D cube file.
    All distance units are converted to angstrom."""

    def __init__(self, filename=None, sparse_limit=0.0):
        self.periodic_table = PeriodicTable()
        self.verbose = False
        self.filename = filename
        self.data = None
        self.comment = ""
        if self.filename is not None:
            self.read(sparse_limit=sparse_limit)

    def __add__(self, other_cube):
        assert np.allclose(self.origin, other_cube.origin)
        assert np.allclose(self.grid_vectors, other_cube.grid_vectors)
        assert np.allclose(self.shape, other_cube.shape)

        new_cube = copy.deepcopy(self)
        new_cube.data += other_cube.data

        return new_cube

    def __sub__(self, other_cube):
        assert np.allclose(self.origin, other_cube.origin)
        assert np.allclose(self.grid_vectors, other_cube.grid_vectors)
        assert np.allclose(self.shape, other_cube.shape)

        new_cube = copy.deepcopy(self)
        new_cube.data -= other_cube.data

        return new_cube

    def __isub__(self, other_cube):
        assert np.allclose(self.origin, other_cube.origin)
        assert np.allclose(self.grid_vectors, other_cube.grid_vectors)
        assert np.allclose(self.shape, other_cube.shape)

        self.data -= other_cube.data
        return self

    def __mul__(self, other_cube):

        new_cube = copy.deepcopy(self)

        if isinstance(other_cube, float) or isinstance(other_cube, int):
            new_cube.data *= other_cube
        else:
            assert np.allclose(self.origin, other_cube.origin)
            assert np.allclose(self.grid_vectors, other_cube.grid_vectors)
            assert np.allclose(self.shape, other_cube.shape)

            new_cube.data *= other_cube.data

        return new_cube

    def __imul__(self, other_cube):

        if isinstance(other_cube, float) or isinstance(other_cube, int):
            self.data *= other_cube
        else:
            assert np.allclose(self.origin, other_cube.origin)
            assert np.allclose(self.grid_vectors, other_cube.grid_vectors)
            assert np.allclose(self.shape, other_cube.shape)

            self.data *= other_cube.data
        return self

    def add_geometry(self, geometry_file):
        self.geom = geometry_file
        self.n_atoms = geometry_file.n_atoms

    def read(self, filename=None, sparse_limit=0.0):
        if filename is None:
            filename = self.filename
        if filename is None:
            raise ValueError("No filename specified.")

        with open(filename) as f:
            self.comment = f.readline() + f.readline()
            line = f.readline().strip().split()
            # Read header
            self.n_atoms = int(line[0])

            # origin is the lower left corner in front
            self.origin = (
                np.array([float(x) for x in line[1:]]) * BOHR_IN_ANGSTROM
            )

            # Read cube dimensions
            n_points = []
            dx = []
            for i in range(3):
                line = f.readline().strip().split()
                n_points.append(int(line[0]))
                dx.append([float(x) for x in line[1:]])

            self.grid_vectors = np.array(dx) * BOHR_IN_ANGSTROM
            self.shape = np.array(n_points, np.int64)

            self._calculate_cube_vectors()
            # read atom coordinates
            atom_Z = []
            atom_pos = []
            for i in range(self.n_atoms):
                line = f.readline().strip().split()
                atom_Z.append(int(line[0]))
                atom_pos.append([float(x) for x in line[2:]])

            atom_pos = np.array(atom_pos) * BOHR_IN_ANGSTROM
            species = [
                self.periodic_table.get_chemical_symbol(x) for x in atom_Z
            ]
            self.geom = Geometry()
            # self.geom.lattice_vectors = self.cube_vectors
            self.geom.add_atoms(atom_pos, species)

            # read grid data
            data = []
            for line in f:
                data += [float(x) for x in line.split()]
            data = np.array(data)

            self.N_points = len(data)
            self.data = data.reshape(self.shape)

    def _calculate_cube_vectors(self):
        self.cube_vectors = ((self.grid_vectors.T) * self.shape).T
        self.dV = np.abs(
            get_triple_product(
                self.grid_vectors[0, :],
                self.grid_vectors[1, :],
                self.grid_vectors[2, :],
            )
        )

        self.dv1 = np.linalg.norm(self.grid_vectors[0, :])
        self.dv2 = np.linalg.norm(self.grid_vectors[1, :])
        self.dv3 = np.linalg.norm(self.grid_vectors[2, :])

    def get_periodic_replica(self, periodic_replica):

        print(
            "WARNING: the periodic replicas assume that the cube lattice is identical to the actual lattice vectors of the calculation"
        )

        new_cube = copy.deepcopy(self)

        # add geometry
        geom = copy.deepcopy(self.geom)
        geom.lattice_vectors = self.cube_vectors
        new_geom = geom.get_periodic_replica(periodic_replica)
        new_cube.geom = new_geom
        new_cube.n_atoms = new_geom.n_atoms

        # add data
        new_cube.data = np.tile(self.data, periodic_replica)

        # add lattice vectors
        new_shape = self.shape * np.array(periodic_replica)
        new_cube.shape = new_shape
        new_cube._calculate_cube_vectors()
        new_cube.N_poins = len(new_cube.data)

        return new_cube

    def save_to_file(self, filename: str):
        """
        Saves cube in cube file format

        """
        assert filename.endswith(".cube"), "Wrong filename ending"
        header = ""
        #        t0 = time.time()
        # comments
        header += self.comment

        # cube file needs exactly 2 comment lines
        n_comment_lines = header.count("\n")
        if n_comment_lines < 2:
            header += "\n" * (2 - n_comment_lines)
        elif n_comment_lines > 2:
            split_head = header.split("\n")
            header = (
                split_head[0]
                + "\n"
                + " ".join(split_head[1:-1])
                + "\n"
                + split_head[-1]
            )
        # n_atoms and origin
        header += (
            "{:5d}".format(self.n_atoms)
            + "   "
            + "   ".join(
                ["{: 10.6f}".format(x / BOHR_IN_ANGSTROM) for x in self.origin]
            )
            + "\n"
        )
        # lattice vectors
        for i in range(3):
            header += (
                "{:5d}".format(self.shape[i])
                + "   "
                + "   ".join(
                    [
                        "{: 10.6f}".format(
                            self.grid_vectors[i, j] / BOHR_IN_ANGSTROM
                        )
                        for j in range(3)
                    ]
                )
                + "\n"
            )

        # atoms
        atom_pos = self.geom.coords
        atom_Z = [
            self.periodic_table.get_atomic_number(x) for x in self.geom.species
        ]
        for i in range(self.n_atoms):
            header += (
                "{:5d}".format(atom_Z[i])
                + "   "
                + "0.000000"
                + "   "
                + "   ".join(
                    [
                        "{: 10.6f}".format(atom_pos[i, j] / BOHR_IN_ANGSTROM)
                        for j in range(3)
                    ]
                )
                + "\n"
            )

        x_len = self.shape[0]
        y_len = self.shape[1]
        z_len = self.shape[2]
        #        t1 = time.time()
        str_arr_size = int(x_len * y_len)
        string_array = np.empty([str_arr_size, z_len], dtype="<U18")
        #        print('{} s for header'.format(t1-t0))

        # values
        for ix in range(x_len):
            for iy in range(y_len):
                for iz in range(z_len):
                    # for each ix we are consecutively writing all iy elements
                    string_array[ix * y_len + iy, iz] = " {: .8e}".format(
                        self.data[ix, iy, iz]
                    )
        #        t2 = time.time()
        #        print('{} s for cube data'.format(t2-t1))
        with open(filename, "w", newline="\n") as f:
            f.write(header)
            for i in range(str_arr_size):
                for j in range(int(np.ceil(z_len / 6))):
                    start_ind = 6 * j
                    end_ind = 6 * (j + 1)
                    if end_ind > z_len:
                        end_ind = z_len
                    f.write("".join(string_array[i, start_ind:end_ind]) + "\n")

    #        t3 = time.time()
    #        print('{} s to save file'.format(t3-t2))

    def get_on_sparser_grid(self, reduction_factors):
        rho = self.data[:: reduction_factors[0], :, :]
        rho = rho[:, :: reduction_factors[1], :]
        rho = rho[:, :, :: reduction_factors[2]]
        return rho

    def get_value_list(self):
        return np.reshape(self.data, [self.N_points])

    def get_point_list(self):
        ind = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing="ij",
        )

        fractional_point_list = np.array([i.ravel() for i in ind]).T
        r = np.dot(fractional_point_list, self.grid_vectors)
        return r + self.origin

    def get_point_coordinates(self):
        """creates n1 x n2 x n3 x 3 array of coordinates for each data point"""
        r = self.get_point_list()
        r_mat = np.reshape(r, [*self.shape, 3], order="C")
        return r_mat

    def get_integrated_projection_on_axis(self, axis):
        """
        Integrate cube file over the plane perpendicular to the
        selected axis (0,1,2) in units of 1/unit of length

        Returns:
        --------
        proj : array
            projected values
        xaxis : array
            coordinates of axis (same length as proj)
        """
        if axis not in [0, 1, 2]:
            raise Exception("Wrong axis input")
        else:
            axsum = list(range(3))
            axsum.pop(axis)

        dA = np.linalg.norm(
            np.cross(
                self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :]
            )
        )

        proj = (
            np.sum(self.data, axis=tuple(axsum)) * dA
        )  # trapeziodal rule: int(f) = sum_i (f_i + f_i+1) * dA / 2 (but here not div by 2 because no double counting in sum)
        xstart = self.origin[axis]
        xend = (
            self.origin[axis]
            + self.grid_vectors[axis, axis] * self.shape[axis]
        )
        xaxis = np.linspace(xstart, xend, self.shape[axis])

        return proj, xaxis

    def get_averaged_projection_on_axis(self, axis, divide_by_area=True):
        """
        Project the total cube values on a single axis (0,1,2) and norm them
        by the area of the plane perpendicular to the axis (units of 1/unit of length)

        Returns:
        --------
        proj : array
            projected values
        xaxis : array
            coordinates of axis (same length as proj)
        """
        if axis not in [0, 1, 2]:
            raise Exception("Wrong axis input")
        else:
            axsum = list(range(3))
            axsum.pop(axis)

        dA = np.linalg.norm(
            np.cross(
                self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :]
            )
        )
        n_datapoints = self.shape[axsum[0]] * self.shape[axsum[1]]
        A = dA * n_datapoints

        # this gives sum(data) * dA
        proj, xaxis = self.get_integrated_projection_on_axis(
            axis
        )  # sum_i (f_i +f_i+1) * dA / 2

        # remove dA from integration
        proj = proj / dA

        # average per area
        if divide_by_area:
            averaged = proj / A

        # pure mathematical average
        else:
            averaged = proj / n_datapoints

        return averaged, xaxis

    def getChargeFieldPotentialAlongAxis(self, axis):

        if axis not in [0, 1, 2]:
            raise Exception("Wrong axis input")
        else:
            axsum = list(range(3))
            axsum.pop(axis)

        dA = np.linalg.norm(
            np.cross(
                self.grid_vectors[axsum[0], :], self.grid_vectors[axsum[1], :]
            )
        )
        n_datapoints = self.shape[axsum[0]] * self.shape[axsum[1]]
        A = dA * n_datapoints

        charge_density, axis_coords = self.get_integrated_projection_on_axis(2)

        cum_density = np.cumsum(charge_density) * self.dv3

        field = cum_density / EPSILON0_AIMS / A
        potential = -np.cumsum(field) * self.dv3

        return axis_coords, charge_density, cum_density, potential

    def heights_for_constant_current(self, constant_current):
        """Calculates the heights for which the current was closest to the value
        I=constant_current for an STM cube file.

        Args:
            constant current <float>: value the current should have.
        """

        # difference of the current in each point to the constant_current
        delta = np.abs(self.data - constant_current)

        # get indices of z-dimension of points that were closest to the current
        z_indices = np.argmin(delta, axis=2)

        # get the z-values that correspond to this indices
        v1_vec, v2_vec, v3_vec = self.get_voxel_coordinates()

        # create an array of the shape of indices with the heights
        # (repeat the v3_vec array to get to the shape of indices)
        heights = np.ones_like(z_indices)[:, :, np.newaxis] * v3_vec

        # cutout those hights that correspond to the indices
        x_indices, y_indices = np.indices(z_indices.shape)
        heights = heights[(x_indices, y_indices, z_indices)]

        return heights

    def shift_contentA_along_vector(
        self, vec, repeat=False, integer_only=False, return_shift_indices=False
    ):
        """
        Shifts values of the CubeFile along a specific vector.
        All values that are not known are set to zero.
        All values that are now outside the cube are deleted.
        TODO: Extrapolate unknown values
        ---------               ---------
        |xxxxxxx|               |00xxxxx| xx
        |xxxxxxx| shift by vec  |00xxxxx| xx  <-- deleted
        |xxxxxxx|      ---->    |00xxxxx| xx
        |xxxxxxx|               |00xxxxx| xx
        |xxxxxxx|               |00xxxxx| xx
        ---------               ---------

        """
        # convert vec to indices
        trans_mat = copy.deepcopy(self.grid_vectors).T
        shift_inds = np.dot(np.linalg.inv(trans_mat), vec)
        if integer_only:
            shift_inds = shift_inds.astype(int)
        # shift_pos = np.dot(T,shift_inds)
        if repeat:
            mode = "wrap"
        else:
            mode = "constant"
        data = shift(self.data, shift_inds, mode=mode)

        if return_shift_indices:
            return data, shift_inds
        else:
            return data

    def get_value_at_positions(
        self, coords, return_mapped_coords=False, xy_periodic=True
    ):
        """
        Returns value of closest data point in cube grid

        Parameters
        ----------
        coords : np.array
            List of Cartesian coordinates at which the cubefile values should
            be returned.
        return_mapped_coords : bool, optional
            Return the Cartesian coordinates, minus the origin of the cubefile,
            of the grid point in the cubefile that is closest to the respective
            position in coords. The default is False.

        Returns
        -------
        np.array
            Vaules at the grid point closest to the respective positions in
            coords
        """
        trans_mat = copy.deepcopy(self.grid_vectors).T
        coords = np.atleast_2d(coords)
        pos_inds = np.round(
            np.dot(np.linalg.inv(trans_mat), (coords - self.origin).T)
        )
        pos_inds = pos_inds.astype(int)

        n_coords = np.shape(pos_inds)[1]
        if not xy_periodic:
            pos_inds[0, pos_inds[0, :] > self.shape[0]] = self.shape[0] - 1
            pos_inds[0, pos_inds[0, :] < 0] = 0
            pos_inds[1, pos_inds[1, :] > self.shape[1]] = self.shape[1] - 1
            pos_inds[1, pos_inds[1, :] < 0] = 0
        values = np.zeros([n_coords])

        for i in range(n_coords):
            try:
                values[i] = self.data[
                    pos_inds[0, i], pos_inds[1, i], pos_inds[2, i]
                ]
            except:
                values[i] = np.nan

        if return_mapped_coords:
            return values, self.origin + np.dot(trans_mat, pos_inds).T
        else:
            return values

    def get_interpolated_value_at_positions(
        self, coords, return_mapped_coords=False, xy_periodic=True
    ):
        """returns value of closest data point in cube grid"""
        trans_mat = copy.deepcopy(self.grid_vectors).T
        coords = np.atleast_2d(coords)
        pos_inds_0 = np.dot(np.linalg.inv(trans_mat), (coords - self.origin).T)
        pos_inds = np.round(pos_inds_0).astype(int)

        n_coords = np.shape(pos_inds)[1]
        if not xy_periodic:
            pos_inds[0, pos_inds[0, :] >= self.shape[0]] = self.shape[0] - 1
            pos_inds[0, pos_inds[0, :] < 0] = 0
            pos_inds[1, pos_inds[1, :] >= self.shape[1]] = self.shape[1] - 1
            pos_inds[1, pos_inds[1, :] < 0] = 0
        else:
            pos_inds_0[0, :] = pos_inds_0[0, :] % self.shape[0]
            pos_inds_0[1, :] = pos_inds_0[1, :] % self.shape[1]

            pos_inds[0, :] = pos_inds[0, :] % self.shape[0]
            pos_inds[1, :] = pos_inds[1, :] % self.shape[1]

        pos_inds[2, pos_inds[2, :] >= self.shape[2]] = self.shape[2] - 1
        pos_inds[2, pos_inds[2, :] < 0] = 0

        values = np.zeros([n_coords])

        difference = pos_inds_0 - pos_inds

        if xy_periodic:
            difference[0, difference[0, :] > 1.0] = (
                difference[0, :] - self.shape[0]
            )
            difference[1, difference[1, :] > 1.0] = (
                difference[1, :] - self.shape[1]
            )

        for i in range(n_coords):

            pos_inds_x = pos_inds[:, i] + np.array(
                [np.sign(difference[0])[i], 0, 0]
            )
            pos_inds_y = pos_inds[:, i] + np.array(
                [0, np.sign(difference[1])[i], 0]
            )
            pos_inds_z = pos_inds[:, i] + np.array(
                [0, 0, np.sign(difference[2])[i]]
            )

            # periodic boundary conditions
            if not xy_periodic:
                if pos_inds_x[0] >= self.shape[0]:
                    pos_inds_x[0] = self.shape[0] - 1
                if pos_inds_x[0] < 0:
                    pos_inds_x[0] = 0

                if pos_inds_y[1] >= self.shape[1]:
                    pos_inds_y[1] = self.shape[1] - 1
                if pos_inds_y[1] < 0:
                    pos_inds_y[1] = 0
            else:
                if pos_inds_x[0] >= self.shape[0]:
                    pos_inds_x[0] = self.shape[0] - pos_inds_x[0]

                if pos_inds_y[1] >= self.shape[1]:
                    pos_inds_y[1] = self.shape[1] - pos_inds_y[1]

            if pos_inds_z[2] >= self.shape[2]:
                pos_inds_z[2] = self.shape[2] - 1
            if pos_inds_z[2] < 0:
                pos_inds_z[2] = 0

            pos_inds_x = pos_inds_x.astype(int)
            pos_inds_y = pos_inds_y.astype(int)
            pos_inds_z = pos_inds_z.astype(int)

            values_0 = self.data[
                pos_inds[0, i], pos_inds[1, i], pos_inds[2, i]
            ]
            values_x = self.data[pos_inds_x[0], pos_inds_x[1], pos_inds_x[2]]
            values_y = self.data[pos_inds_y[0], pos_inds_y[1], pos_inds_y[2]]
            values_z = self.data[pos_inds_z[0], pos_inds_z[1], pos_inds_z[2]]

            d_v_x = (values_x - values_0) / np.sign(difference[0])[i]
            d_v_y = (values_y - values_0) / np.sign(difference[1])[i]
            d_v_z = (values_z - values_0) / np.sign(difference[2])[i]

            if np.isnan(d_v_x):
                d_v_x = 0
            if np.isnan(d_v_y):
                d_v_y = 0
            if np.isnan(d_v_z):
                d_v_z = 0

            normal_vector = np.array([-d_v_x, -d_v_y, -d_v_z, 1.0])
            normal_vector /= np.linalg.norm(normal_vector)

            values[i] = (
                normal_vector[3] * values_0
                - normal_vector[0] * difference[0][i]
                - normal_vector[1] * difference[1][i]
                - normal_vector[2] * difference[2][i]
            ) / normal_vector[3]

        if return_mapped_coords:
            return values, self.origin + np.dot(trans_mat, pos_inds).T
        else:
            return values

    def calculate_distance_to_local_geometry(self, adsorption_geometry):
        """
        Calculates the absolute distance between the molecule in the cubefile
        and the corresponding local adsorption geometry

        """
        cube_mol = self.geom.get_molecules()
        cube_geom_center = cube_mol.get_geometric_center(
            ignore_center_attribute=True
        )
        ads_geom_center = adsorption_geometry.get_geometric_center(
            ignore_center_attribute=True
        )
        distance_to_adsorption_geometry = ads_geom_center - cube_geom_center
        # print(distance_to_adsorption_geometry)
        # print(cube_mol.coords)
        # print( adsorption_geometry.coords)
        # print(np.sum((cube_mol.coords+distance_to_adsorption_geometry) - adsorption_geometry.coords))
        coord_diff = np.max(
            cube_mol.coords
            + distance_to_adsorption_geometry
            - adsorption_geometry.coords
        )
        assert (
            coord_diff < 5e-2
        ), "Local Geometry doesnt match cube geometry!, difference is {}".format(
            coord_diff
        )

        self.corresponding_adsorption_geometry = adsorption_geometry
        self.distance_to_adsorption_geometry = (
            ads_geom_center - cube_geom_center
        )

    def get_voxel_volumina(self):
        grid_vec = copy.deepcopy(self.grid_vectors)
        dv1 = np.linalg.norm(grid_vec[0, :])
        dv2 = np.linalg.norm(grid_vec[1, :])
        dv3 = np.linalg.norm(grid_vec[2, :])
        return dv1, dv2, dv3

    def get_voxel_coordinates(self):

        dv1, dv2, dv3 = self.get_voxel_volumina()

        v1_vec = (
            np.array([self.origin[0] + i * dv1 for i in range(self.shape[0])])
            - dv1 / 2
        )  # shift by half a grid vector to align voxel to center
        v2_vec = (
            np.array([self.origin[1] + i * dv2 for i in range(self.shape[1])])
            - dv2 / 2
        )
        v3_vec = (
            np.array([self.origin[2] + i * dv3 for i in range(self.shape[2])])
            - dv3 / 2
        )

        return [v1_vec, v2_vec, v3_vec]

    def get_voxel_coordinates_along_lattice(self, periodic_replica):
        """unit cell is usually not at 90 degree angle there plot of xy plane
        has to be projected onto the lattice vectors"""

        grid_vec = copy.deepcopy(self.grid_vectors)
        dv1, dv2, dv3 = self.get_voxel_volumina()

        # get lattice vectors
        latt_mat = grid_vec[:-1, :-1]
        latt_mat[0, :] /= np.linalg.norm(latt_mat[0, :])
        latt_mat[1, :] /= np.linalg.norm(latt_mat[1, :])
        R = latt_mat.T

        # get points in cube grid
        v1_vec = (
            np.array(
                [i * dv1 for i in range(self.shape[0] * periodic_replica[0])]
            )
            - dv1 / 2
        )
        v2_vec = (
            np.array(
                [i * dv2 for i in range(self.shape[1] * periodic_replica[1])]
            )
            - dv2 / 2
        )
        v1, v2 = np.meshgrid(v1_vec, v2_vec)

        # project points onto lattice
        mult = np.dot(R, np.array([v1.ravel(), v2.ravel()]))
        v1_plot = mult[0, :].reshape(v1.shape) + self.origin[0]
        v2_plot = mult[1, :].reshape(v2.shape) + self.origin[1]

        return v1_plot, v2_plot

    def get_values_on_plane(
        self, plane_centre, plane_normal, plane_extent, plane_points=100
    ):
        """
        Retruns the cubefile values on a given plane

        Parameters
        ----------
        plane_centre : np.array
            Centre of the plane.
        plane_normal : np.array
            Vector normal to the plane, i.e. viewing direction.
        plane_extent : float
            Size of the plane in Angstrom

        Returns
        -------
        values_on_plane : np.array

        """
        plane_normal /= np.linalg.norm(plane_normal)

        vec_z = np.array([0.0, 0.0, 1.0])

        plane_vec_xy = np.cross(vec_z, plane_normal)
        plane_vec_xy /= np.linalg.norm(plane_vec_xy)
        plane_vec_z = np.cross(plane_normal, plane_vec_xy)
        plane_vec_z /= np.linalg.norm(plane_vec_z)

        extent_vec = np.linspace(-plane_extent, plane_extent, plane_points)

        values_on_plane = np.zeros((len(extent_vec), len(extent_vec)))

        dv1, dv2, dv3 = self.get_voxel_volumina()
        max_dist = (dv1 + dv2 + dv3) / 3

        for ind_1, x in enumerate(extent_vec):
            for ind_2, y in enumerate(extent_vec):
                plane_pos = plane_centre - x * plane_vec_xy + y * plane_vec_z

                value, mapped_coords = self.get_value_at_positions(
                    plane_pos, return_mapped_coords=True
                )

                vec = mapped_coords - plane_pos + self.origin
                mag = np.linalg.norm(vec)

                if mag < max_dist:
                    values_on_plane[ind_1, ind_2] = value

        return values_on_plane

    def visualise_plane(
        self,
        axes=[0, 1],
        v3_position=None,
        show_molecule=False,
        periodic_replica=(1, 1),
        molecule_replica_alpha=0.0,
        molecule_zlim=None,
        molecule_alpha=1.0,
        molecule_periodic_replica=None,
        lattice=None,
        limits=None,
        show_axis=False,
        masked_below=None,
        colormap="coolwarm",
        alpha=1,
        scaling_factor=1,
        plot_gradient=False,
        show_colorbar=False,
        colorbar_label="",
        cbar_logarithmic=False,
        wireframe_molecule=False,
        high_contrast=False,
        x_limits=None,
        y_limits=None,
    ):
        """
        v3_position: none or value in Angstrom
        if None: v3 is summed over
        if value: plane at height v3_position is visualized
        if list of two values: summed in direction of v3 between two values"""

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        lattice = copy.deepcopy(lattice)

        # --- get voxel lengths and volumina ---
        dv1, dv2, dv3 = self.get_voxel_volumina()
        dv_list = [dv1, dv2, dv3]
        dV = dv1 * dv2 * dv3

        vec_list = self.get_voxel_coordinates()
        # ---

        # is it a 2D or 3D plot?
        third_ax = np.nan
        for i in range(3):
            if i not in axes:
                third_ax = i

        # --- scale data and/or calculate gradients if desired ---
        data = copy.deepcopy(self.data)
        if plot_gradient:
            data = np.gradient(data, axis=third_ax, varargs=dv_list[third_ax])
        data *= scaling_factor
        # ---

        # choose axes to plot
        plt_dv3 = dv_list[third_ax]
        plt_ax1 = vec_list[axes[0]]
        plt_ax2 = vec_list[axes[1]]

        # plot of xy plane has to be projected correctly
        if third_ax == 2:

            # --- project points only unit cell ---
            v1_plot, v2_plot = self.get_voxel_coordinates_along_lattice(
                periodic_replica=periodic_replica
            )
            # ---
        else:
            v1_plot, v2_plot = np.meshgrid(plt_ax1, plt_ax2)

        # --- assemble the actual data to be plotted ---
        if v3_position is None:
            # plot integrated over all z-planes
            data = np.sum(data, third_ax) * plt_dv3
        elif isinstance(v3_position, list):
            v3_vector = vec_list[third_ax]
            v3_pos_index_1 = np.argmin(np.abs(v3_vector - v3_position[0]))
            v3_pos_index_2 = np.argmin(np.abs(v3_vector - v3_position[1]))

            data = (
                np.sum(data[:, :, v3_pos_index_1:v3_pos_index_2], third_ax)
                * plt_dv3
            )
        else:
            # find the points in data closest to required z-plane
            v3_vector = vec_list[third_ax]
            v3_pos_index = np.argmin(np.abs(v3_vector - v3_position))

            print("third_ax", third_ax, "v3_pos_index", v3_pos_index)

            if third_ax == 0:
                data = data[v3_pos_index, :, :]
            elif third_ax == 1:
                data = data[:, v3_pos_index, :]
            else:
                data = data[:, :, v3_pos_index]

            print(data.shape)

        # remove data based below threshold
        if masked_below is not None:
            data = np.ma.masked_where(data < masked_below, data)

        # --- get cbar limits ---
        if limits is None:
            # if limits are note given, use the max/min of cube data.
            maxcharge = np.amax(data)
            mincharge = np.amin(data)
            abs_max = np.amax([np.abs(maxcharge), np.abs(mincharge)])
            maxcharge = abs_max
            mincharge = -abs_max
            if high_contrast:
                maxcharge /= 5
                mincharge /= 5
            if cbar_logarithmic:
                norm = mpl.colors.LogNorm(vmin=mincharge, vmax=maxcharge)
            else:
                norm = mpl.colors.Normalize(vmin=mincharge, vmax=maxcharge)

        elif limits == "dynamic":
            # no fixed limits are given
            norm = None
        else:
            maxcharge = limits[1]
            mincharge = limits[0]

            if cbar_logarithmic:
                norm = mpl.colors.LogNorm(vmin=mincharge, vmax=maxcharge)
            else:
                norm = mpl.colors.Normalize(vmin=mincharge, vmax=maxcharge)
        # ---

        # do periodic replicas
        data = np.tile(data, periodic_replica)
        # ---

        # --- do the actual plotting ---
        fig = plt.gcf()
        ax = plt.gca()

        # plot charge
        pcm = ax.pcolormesh(
            v1_plot.T,
            v2_plot.T,
            data,
            cmap=colormap,
            alpha=alpha,
            norm=norm if limits != "dynamic" else None,
        )

        # plot colorbar
        if show_colorbar:
            cb = fig.colorbar(pcm)
            cb.set_label(colorbar_label)

        if show_molecule:
            self.visualizeMolecule(
                (
                    periodic_replica
                    if molecule_periodic_replica is None
                    else molecule_periodic_replica
                ),
                lattice=lattice,
                molecule_alpha=molecule_alpha,
                molecule_replica_alpha=molecule_replica_alpha,
                axes=axes,
                molecule_zlim=molecule_zlim,
                wireframe=wireframe_molecule,
            )
            if molecule_replica_alpha > 0 and lattice is None:
                print(
                    "WARNING: the periodic molecule replicas will only be visible if lattice is defined!"
                )
        if x_limits is not None:
            ax.set_xlim([x_limits[0], x_limits[1]])
        if y_limits is not None:
            ax.set_ylim([y_limits[0], y_limits[1]])

        # plot axis
        if show_axis:
            ax.set_xlabel("v1 / $\AA$")
            ax.set_ylabel("v2 / $\AA$")
            ax.set_aspect("equal")

        return pcm

    def visualise_molecule(
        self,
        periodic_replica,
        lattice,
        molecule_replica_alpha=0,
        molecule_alpha=1.0,
        axes=[0, 1],
        molecule_zlim=None,
        wireframe=False,
    ):
        """Args:
        - lattice is lattice of original geometry (for periodic replica
        - molecule_zlim: remove every thing in geometry that is below
           this z value.
        """

        # --- setup molecule to be plotted ---
        mol = copy.deepcopy(self.geom)

        if molecule_zlim:
            zlim = [molecule_zlim, np.amax(mol.coords[:, 2]) + 1]
        else:
            zlim = None
            mol.remove_metal_substrate()

        mol_replica = mol.get_reriodic_replica(
            replications=periodic_replica, lattice=lattice
        )

        # plot molecule
        mol.visualise(
            auto_limits=False,
            axes=axes,
            min_zorder=11,
            zlim=zlim,
            plot_method="wireframe" if wireframe else "circles",
            alpha=molecule_alpha,
        )
        mol_replica.visualize(
            auto_limits=False,
            axes=axes,
            min_zorder=10,
            alpha=molecule_replica_alpha,
            zlim=zlim,
            hide_axes=True,
            plot_method="wireframe" if wireframe else "circles",
        )
        # ---

    def visualize_constant_current_surface(
        self,
        constant_current,
        show_molecule=False,
        periodic_replica=(1, 1),
        molecule_periodic_replica=None,
        molecule_replica_alpha=0.3,
        molecule_alpha=1.0,
        lattice=None,
        limits=None,
        show_axis=False,
        colormap="coolwarm",
        alpha=1,
        show_colorbar=False,
        colorbar_label="",
        cbar_logarithmic=False,
        molecule_zlim=None,
        height_offset=0,
        color_over_and_under=False,
    ):
        """Experimental! Only for stm cube files. Only for x-y planes."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        data = self.heights_for_constant_current(constant_current)
        data = np.tile(data, periodic_replica)

        # apply offset
        data += height_offset

        lattice = copy.deepcopy(lattice)

        # --- get x, y mesh of positions ---
        x, y = self.get_voxel_coordinates_along_lattice(
            periodic_replica=periodic_replica
        )
        # ---

        # --- do the actual plotting ---
        fig = plt.gcf()
        ax = plt.gca()

        # set cbar limits
        if limits is not None:
            if cbar_logarithmic:
                norm = mpl.colors.LogNorm(vmin=limits[0], vmax=limits[1])
            else:
                norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])
        else:
            norm = None

        cmap_object = copy.deepcopy(get_cmap(colormap))
        if color_over_and_under:
            cmap_object.set_over(((1, 0, 0, 1)))
            cmap_object.set_under((0.188, 0.314, 0.973, 1))

        # plot charge
        pcm = ax.pcolormesh(
            x.T, y.T, data, cmap=cmap_object, alpha=alpha, norm=norm
        )

        # plot axis
        if show_axis:
            ax.set_xlabel(r"v1 / $\AA$")
            ax.set_ylabel(r"v2 / $\AA$")
            ax.set_aspect("equal")

        # plot colorbar
        if show_colorbar:
            cb = fig.colorbar(pcm)
            cb.set_label(colorbar_label)

        # plot molecule
        if show_molecule:
            self.visualizeMolecule(
                (
                    periodic_replica
                    if molecule_periodic_replica is None
                    else molecule_periodic_replica
                ),
                lattice,
                molecule_replica_alpha=molecule_replica_alpha,
                molecule_alpha=molecule_alpha,
                axes=[0, 1],
                molecule_zlim=molecule_zlim,
            )
        # ---

        return pcm

    def visualise(self, fig_dir=None, show_molecule=False, limits=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        data = copy.deepcopy(self.data)
        grid_vec = copy.deepcopy(self.grid_vectors)
        geom = copy.deepcopy(self.geom)
        mol = geom
        mol.removeMetalSubstrate()
        origin = self.origin
        dv1 = np.linalg.norm(grid_vec[0, :])
        dv2 = np.linalg.norm(grid_vec[1, :])
        dv3 = np.linalg.norm(grid_vec[2, :])
        dV = dv1 * dv2 * dv3
        v1_vec = (
            np.array([i * dv1 for i in range(self.shape[0])]) - dv1 / 2
        )  # shift by half a grid vector to align voxel to center
        v2_vec = np.array([i * dv2 for i in range(self.shape[1])]) - dv2 / 2
        v3_vec = (
            np.array([origin[2] + i * dv3 for i in range(self.shape[2])])
            - dv3 / 2
        )

        if limits is None:
            maxcharge = np.amax(data * dV)
            mincharge = np.amin(data * dV)
            abs_max = np.amax([np.abs(maxcharge), np.abs(mincharge)])
            maxcharge = abs_max
            mincharge = -abs_max
        else:
            maxcharge = limits[1]
            mincharge = limits[0]
        axes_end = (
            np.dot(grid_vec, self.shape) + origin - np.sum(grid_vec, axis=1)
        )
        axes_start = origin - np.sum(grid_vec, axis=1)

        fig = plt.gcf()

        ax_v2 = fig.add_subplot(2, 2, 1)
        v1, v3 = np.meshgrid(v1_vec, v3_vec)
        ax_v2.pcolormesh(
            v1.T,
            v3.T,
            np.sum(data, 1) * dv2,
            vmin=mincharge,
            vmax=maxcharge,
            cmap="coolwarm",
        )
        ax_v2.set_xlabel("v1 / A")
        ax_v2.set_ylabel("z / A")
        ax_v2.set_aspect("equal")

        ax_v1 = fig.add_subplot(2, 2, 2)
        v2, v3 = np.meshgrid(v2_vec, v3_vec)
        im = ax_v1.pcolormesh(
            v2.T,
            v3.T,
            np.sum(data, 0) * dv1,
            vmin=mincharge,
            vmax=maxcharge,
            cmap="coolwarm",
        )
        ax_v1.set_xlabel("v2 / A")
        ax_v1.set_ylabel("z / A")
        ax_v1.set_aspect("equal")

        # plot of xy plane has to be projected correctly
        latt_mat = grid_vec[:-1, :-1]
        latt_mat[0, :] /= np.linalg.norm(latt_mat[0, :])
        latt_mat[1, :] /= np.linalg.norm(latt_mat[1, :])
        R = latt_mat.T

        v1_vec = np.array([i * dv1 for i in range(self.shape[0])]) - dv1 / 2
        v2_vec = np.array([i * dv2 for i in range(self.shape[1])]) - dv2 / 2
        v1, v2 = np.meshgrid(v1_vec, v2_vec)

        mult = np.dot(R, np.array([v1.ravel(), v2.ravel()]))
        v1_rot = mult[0, :].reshape(v1.shape) + origin[0]
        v2_rot = mult[1, :].reshape(v2.shape) + origin[1]

        ax_v3 = fig.add_subplot(2, 1, 2)
        ax_v3.pcolormesh(
            v1_rot.T,
            v2_rot.T,
            np.sum(data, 2) * dv3,
            vmin=mincharge,
            vmax=maxcharge,
            cmap="coolwarm",
            zorder=0,
        )
        ax_v3.set_xlabel("x / A")
        ax_v3.set_ylabel("y / A")
        ax_v3.set_aspect("equal")
        if show_molecule:
            mol.visualize(auto_limits=False)
        fig.tight_layout()

        cax, kw = mpl.colorbar.make_axes(
            [ax_v1, ax_v2, ax_v3], location="bottom"
        )
        plt.colorbar(im, cax=cax, **kw)
        if fig_dir is not None:
            cax.set_xlabel(os.path.basename(fig_dir)[:-4])

    def calculateOverlapIntegral(
        self,
        other_cubefile,
        print_normalization_factors=True,
        take_absolute_value=True,
        output_overlap_cube=False,
    ):
        """
        Calculates the overlap integral of the quantity described in the cubefile with that of a second cubefile.
        NOTE: this is written to work with the standard FHI-aims voxels in Angstrom³
        NOTE: the two orbitals should describe the same exact volume of space!
        :param other_cubefile: CubeFile
        :return: float
        """
        # this data is normally provided in angstrom^(-3/2)
        first = copy.deepcopy(self.data)
        second = copy.deepcopy(other_cubefile.data)
        # let's pass to bohr_radius^(-3/2)
        first *= np.sqrt(BOHR_IN_ANGSTROM**3)
        second *= np.sqrt(BOHR_IN_ANGSTROM**3)
        # both arrays get normalized
        first_squared = first * first
        second_squared = second * second
        first_total = first_squared.sum()
        second_total = second_squared.sum()
        first_normalization_factor = np.sqrt(1 / first_total)
        second_normalization_factor = np.sqrt(1 / second_total)
        if print_normalization_factors:
            print("Normalization factors:")
            print("self: ", first_normalization_factor)
            print("other cubefile: ", second_normalization_factor)
        first *= first_normalization_factor
        second *= second_normalization_factor
        # now the sum corresponds to the overall overlap (we are integrating in d(Voxel), which is d(bohr_radius^(-3)) ), and we take its absolute value (overlap has no sign)
        product = first * second
        if output_overlap_cube:
            overlap_cube = deepcopy(self)
            overlap_cube.data = product
        overlap = np.sum(product)
        if take_absolute_value:
            overlap = np.abs(overlap)
        if output_overlap_cube:
            return overlap, overlap_cube
        else:
            return overlap

    def get_eigenstate_number(self):
        name = self.filename.split("/")[-1].split(".")[0]
        components = name.split("_")
        eigNumIndex = components.index("eigenstate") + 1
        eigNum = int(components[eigNumIndex])
        return eigNum

    def get_spin_channel(self):
        name = self.filename.split("/")[-1].split(".")[0]
        components = name.split("_")
        spinIndex = components.index("spin") + 1
        spin = int(components[spinIndex])
        return spin

    def get_iso_surface(
        self, iso_val, color="blue", opacity=1.0, resolution_downscale=1
    ):
        """
        Get isosurface in a format that can be visualised with plotly.

        Parameters
        ----------
        iso_val : float
            Value at which the isosurface should be plotted.
        color : str, optional
            A color string understandable by plotly. The default is 'blue'.
        opacity : float, optional
            Opacity of isosurface. The default is 1.0.
        resolution_downscale : int, optional
            Reduce resolution of cube. The default is 1.

        Returns
        -------
        data : geometry_object
            List of geometry_objects plottable by plotly.

        """
        import plotly.graph_objects as go

        values = self.data[
            ::resolution_downscale,
            ::resolution_downscale,
            ::resolution_downscale,
        ]

        X, Y, Z = np.mgrid[
            0 : 1 : (values.shape[0] * 1j),
            0 : 1 : (values.shape[1] * 1j),
            0 : 1 : (values.shape[2] * 1j),
        ]

        coords_frac = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

        coords = np.dot(coords_frac, self.cube_vectors)

        coords = coords + self.origin

        data = go.Isosurface(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            value=values.flatten(),
            isomin=iso_val - 0.0001,
            isomax=iso_val + 0.0001,
            caps=dict(x_show=False, y_show=False),
            surface_count=1,
            colorscale=[color, color],
            opacity=opacity,
        )

        return data

    def get_iso_surfaces_figure(
        self,
        iso_val,
        colors=["red", "blue"],
        opacity=[1.0, 1.0],
        resolution_downscale=1,
        fig=None,
    ):
        """
        Generates a figure of two isosurfaces (at iso_val and -iso_val)
        including the molecule to visualised in plotly.

        Parameters
        ----------
        iso_val : float
            Value at which the isosurface should be plotted.
        color : list of str, optional
            A list of color strings understandable by plotly.
            The default is ['red', 'blue'].
        opacity : list of float, optional
            Opacity of isosurface. The default is [1.0, 1.0].
        resolution_downscale : int, optional
            Reduce resolution of cube. The default is 1.

        Returns
        -------
        fig : plotly figure
            Plotly figure object that can be further edited, saved, or
            visualised interactively.

        """
        import plotly.graph_objects as go

        data_1 = self.get_iso_surface(
            iso_val,
            resolution_downscale=resolution_downscale,
            color=colors[0],
            opacity=opacity[0],
        )

        data_2 = self.get_iso_surface(
            -iso_val,
            resolution_downscale=resolution_downscale,
            color=colors[1],
            opacity=opacity[1],
        )

        data = [data_1, data_2]

        if fig is None:
            fig = go.Figure(data=data)
        else:
            fig.add_trace(data)

        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )

        fig.update_layout(scene=scene)

        # fig.update_coloraxes(showscale=False)
        # fig.update_layout(showlegend=False)
        fig.update_traces(showscale=False)

        return fig


def get_cubefile_grid(geometry, divisions, origin=None, verbose=True):
    """
    creates cube file grid with given number of divisions in each direction.
    If only one division is given, this will be used in all directions.
    If origin is given the vectors will be aligned around this value

    Returns a CubeFileSettings object which can be used for the ControlFile class
    To get text simply use CubeFileSettings.getText()

    """
    if origin is None:
        origin = (
            geometry.lattice_vectors[0, :] / 2
            + geometry.lattice_vectors[1, :] / 2
            + geometry.lattice_vectors[2, :] / 2
        )

    # calculate dx_i
    divs = np.zeros(3)
    divs[0] = np.linalg.norm(geometry.lattice_vectors[0, :]) / divisions[0]
    divs[1] = np.linalg.norm(geometry.lattice_vectors[1, :]) / divisions[1]
    divs[2] = np.linalg.norm(geometry.lattice_vectors[1, :]) / divisions[2]

    # calculate vectors
    vecs = np.zeros([3, 3])
    vecs[0, :] = geometry.lattice_vectors[0, :] / divisions[0]
    vecs[1, :] = geometry.lattice_vectors[1, :] / divisions[1]
    vecs[2, :] = geometry.lattice_vectors[2, :] / divisions[2]

    cube_parms = CubefileParameters()
    cube_parms.set_origin(origin)
    cube_parms.set_edges(divisions, vecs)

    return cube_parms


def get_cubefile_grid_by_spacing(self, spacing, origin=None, verbose=True):
    """EXPERIMENTAL! and ugly as hell! <aj, 10.4.19>
    creates cube file grid with given spacing in each direction.
    If only one division is given, this will be used in all directions.
    If origin is given the vectors will be aligned around this value

    Returns a CubeFileSettings object which can be used for the ControlFile class
    To get text simply use CubeFileSettings.getText()
    """

    if origin is None:
        origin = (
            self.lattice_vectors[0, :] / 2
            + self.lattice_vectors[1, :] / 2
            + self.lattice_vectors[2, :] / 2
        )
    # make numeric value a list if necessary
    if not isinstance(spacing, Iterable):
        spacing = [spacing]

    # check that spacing is given for all three dimensions
    if len(spacing) == 1:
        spacing = [spacing, spacing, spacing]
    assert (
        len(spacing) == 3
    ), "Either one spacing or a separate one for each dimension must be given"

    # calculate n points
    n_points = np.zeros(3)
    n_points[0] = np.ceil(
        np.linalg.norm(self.lattice_vectors[0, :]) / spacing[0]
    )
    n_points[1] = np.ceil(
        np.linalg.norm(self.lattice_vectors[1, :]) / spacing[1]
    )
    n_points[2] = np.ceil(
        np.linalg.norm(self.lattice_vectors[2, :]) / spacing[2]
    )

    # calculate vectors
    vecs = np.zeros([3, 3])
    vecs[0, :] = (
        self.lattice_vectors[0, :]
        / np.linalg.norm(self.lattice_vectors[0, :])
        * spacing[0]
    )
    vecs[1, :] = (
        self.lattice_vectors[1, :]
        / np.linalg.norm(self.lattice_vectors[1, :])
        * spacing[1]
    )
    vecs[2, :] = (
        self.lattice_vectors[2, :]
        / np.linalg.norm(self.lattice_vectors[2, :])
        * spacing[2]
    )

    cube_parms = CubefileParameters()
    cube_parms.set_origin(origin)
    cube_parms.set_edges(n_points, vecs)

    return cube_parms
