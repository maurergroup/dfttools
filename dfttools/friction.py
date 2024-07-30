from os.path import join

import numpy as np

from dfttools.geometry import AimsGeometry


class FrictionTensor:
    def __init__(self, directroy):
        self.geometry = AimsGeometry(join(directroy, "geometry.in"))
        self.friction_tensor_raw = self.read_friction_tensor(
            join(directroy, "friction_tensor.out")
        )

    @property
    def friction_tensor(self):
        return self.friction_tensor_raw

    def read_friction_tensor(self, filename):
        atom_indices = []
        friction_tensor_0 = []

        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            if "# n_atom" in line:
                line = line.strip().split(" ")

                line_1 = []
                for l in line:
                    if not l == "":
                        line_1.append(l)

                atom_index = 3 * (int(line_1[2]) - 1) + int(line_1[4]) - 1
                atom_indices.append(atom_index)

            elif "#" not in line:
                line = line.strip().split(" ")

                friction_tensor_line = []
                for l in line:
                    if not l == "":
                        friction_tensor_line.append(float(l))

                friction_tensor_0.append(friction_tensor_line)

        friction_tensor_0 = np.array(friction_tensor_0)

        n = len(self.geometry)
        friction_tensor = np.zeros((3 * n, 3 * n))

        for ind_0, atom_index_0 in enumerate(atom_indices):
            for ind_1, atom_index_1 in enumerate(atom_indices):
                friction_tensor[atom_index_0, atom_index_1] = friction_tensor_0[
                    ind_0, ind_1
                ]

        return friction_tensor

    def get_life_time(self, vibration):
        """
        returns life time in ps
        """
        vibration /= np.linalg.norm(vibration)

        force = self.friction_tensor_raw.dot(vibration)

        eta = vibration.dot(force)

        life_time = 1 / eta

        return life_time
