import numpy as np
import collections
from typing import List, Literal, Union

import dfttoolkit.utils.file_utils as fu
from dfttoolkit.base_parser import BaseParser


class Parameters(BaseParser):
    """
    Handle files that control parameters for electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new
    supported file type to _supported_files, call the super().__init__ method,
    include the new file type as a kwarg in the super().__init__ call.
    Optionally include the self.lines line in examples.

    ...

    Attributes
    ----------
    _supported_files : list
        List of supported file types.
    """

    def __init__(self, **kwargs: str):
        # FHI-aims, ...
        self._supported_files = ["control_in"]

        # Check that only supported files were provided
        for val in kwargs.keys():
            fu.check_required_files(self._supported_files, val)

        super().__init__(self._supported_files, **kwargs)

    @property
    def supported_files(self) -> List[str]:
        return self._supported_files


class AimsControl(Parameters):
    """
    FHI-aims control file parser.

    ...

    Attributes
    ----------
    lines : List[str]
        The contents of the control.in file.
    path : str
        The path to the control.in file.
    """

    def __init__(
        self, control_in: str = "control.in", parse_file: bool = True
    ):
        if parse_file:
            super().__init__(control_in=control_in)
            self.lines = self.file_contents["control_in"]
            self.path = self.file_paths["control_in"]

            # Check if the control.in file was provided
            fu.check_required_files(self._supported_files, "control_in")

    def add_keywords(self, **kwargs: dict) -> None:
        """
        Add keywords to the control.in file.

        Parameters
        ----------
        **kwargs : dict
            Keywords to be added to the control.in file.
        """

        for keyword in kwargs:
            self.lines.append(keyword + "\n")

        # TODO finish this
        raise NotImplementedError

    def remove_keywords(
        self,
        *args: str,
        output: Literal["overwrite", "print", "return"] = "return",
    ) -> Union[None, List[str]]:
        """
        Remove keywords from the control.in file.

        Parameters
        ----------
        *args : str
            Keywords to be removed from the control.in file.
        output : Literal["overwrite", "print", "return"], default="overwrite"
            Overwrite the original file, print the modified file to STDOUT, or return
            the modified file as a list of '\\n' separated strings.

        Returns
        -------
        Union[None, List[str]]
            If output is "return", the modified file is returned as a list of '\\n'
            separated strings.
        """

        for keyword in args:
            for i, line in enumerate(self.lines):
                if keyword in line:
                    self.lines.pop(i)

        match output:
            case "overwrite":
                with open(self.path, "w") as f:
                    f.writelines(self.lines)

            case "print":
                print(*self.lines, sep="")

            case "return":
                return self.lines

    def get_keywords(self) -> dict:
        """
        Get the keywords from the control.in file.

        Returns
        -------
        dict
            A dictionary of the keywords in the control.in file.
        """

        keywords = {}

        for line in self.lines:
            spl = line.split()

            if "#" * 80 in line:
                break

            if len(spl) > 0 and spl[0] != "#":
                if len(spl) == 1:
                    keywords[spl[0]] = None
                else:
                    keywords[spl[0]] = " ".join(spl[1:])

        return keywords


class CubefileParameters:
    """Represents Cube file settings which can be used to generate a control file
    All numeric values are parsed, strings are kept as such
    Input
    -------------------
        all textlines that belong to cubefile specification
        type is also parsed from the text

    Functions
    -------------------
        parse(text): parses textlines

        getText(): returns cubefile specifications-string for ControlFile class
    """

    def __init__(self, text=None):
        self.type = (
            ""  # type is all that comes after output cube as a single string
        )
        # parsers for specific cube keywords: {keyword: [string_to_number, number_to_string]}
        self.parsing_functions = {
            "spinstate": [
                lambda x: int(x[0]),
                lambda x: str(x),
            ],  #### I change x to x[0] because otherwise it bugs fc 11.02.2021
            "kpoint": [lambda x: int(x[0]), lambda x: str(x)],
            "divisor": [lambda x: int(x[0]), lambda x: str(x)],
            "spinmask": [
                lambda x: [int(k) for k in x],
                lambda x: "  ".join([str(k) for k in x]),
            ],
            "origin": [
                lambda x: [float(k) for k in x],
                lambda x: "  ".join(["{: 15.10f}".format(k) for k in x]),
            ],
            "edge": [
                lambda x: [int(x[0])] + [float(k) for k in x[1:]],
                lambda x: str(int(x[0]))
                + "  "
                + "  ".join(["{: 15.10f}".format(k) for k in x[1:]]),
            ],
        }

        self.settings = collections.OrderedDict()
        if text is not None:
            self.parse(text)

    def __repr__(self):
        text = "CubeFileSettings object with content:\n"
        text += self.get_text()
        return text

    def parse(self, text):
        cubelines = []
        for line in text:
            line = line.strip()
            # parse only lines that start with cube and are not comments
            if not line.startswith("#"):
                if line.startswith("cube"):
                    cubelines.append(line)
                elif line.startswith("output"):
                    self.type = " ".join(line.split()[2:])

        # parse cubelines to self.settings
        for line in cubelines:
            line = line.split("#")[0]  # remove comments
            splitline = line.split()
            keyword = splitline[1]  # parse keyword
            values = splitline[2:]  # parse all values
            # check if parsing function exists
            if keyword in self.parsing_functions:
                value = self.parsing_functions[keyword][0](values)
            # reconvert to single string otherwise
            else:
                value = " ".join(values)

            # save all values as list, append to list if key already exists
            if keyword in self.settings:
                self.settings[keyword].append(value)
            else:
                self.settings[keyword] = [value]

    def set_origin(self, origin):
        """parse numpy array origin to settings"""
        self.settings["origin"] = [[origin[0], origin[1], origin[2]]]

    def set_edges(self, divisions, edge_vectors):
        """parse edge vectors to array"""
        self.settings["edge"] = []
        for i, d in enumerate(divisions):
            self.settings["edge"].append(
                [divisions[i]] + list(edge_vectors[i, :])
            )

    def set_type(self, type):
        """type is all that comes after output cube as a single string"""
        self.type = type

    def _get_edges(self):
        assert "edge" in self.settings, "There are no edges specified"
        edges = self.settings["edge"]
        return np.array(edges)

    def get_grid_vectors(self):
        edges = self._get_edges()
        return edges[:, 1:]

    def get_divisions(self):
        edges = self._get_edges()
        return edges[:, 0]

    def set_divisions(self, divisions):
        assert (
            len(divisions) == 3
        ), "Divisions for all three lattice vectors must be specified!"
        for i in range(3):
            self.settings["edge"][i][0] = divisions[i]

    def has_vertical_unit_cell(self):
        conditions = [
            self.settings["edge"][0][3] == 0.0,
            self.settings["edge"][1][3] == 0.0,
            self.settings["edge"][2][1] == 0.0,
            self.settings["edge"][2][1] == 0.0,
        ]
        if False in conditions:
            return False
        else:
            return True

    def set_z_slice(self, z_bottom, z_top):
        """
        Crops the cubefile to only include the space between z_bottom and z_top.
        The cubefile could go slightly beyond z_bottom and z_top, in order to preserve the distance between grid points.
        :param z_bottom: float
        :param z_top: float
        :return: 0
        """
        assert (
            z_top >= z_bottom
        ), "Please provide z_bottom, z_top in the correct order"
        assert (
            self.has_vertical_unit_cell()
        ), "This function should only be used on systems whose cell is parallel to the Z axis!"
        range = z_top - z_bottom
        average = z_bottom + range / 2
        # set origin Z
        self.settings["origin"][0][2] = average
        # set edge, approximating for excess
        z_size = self.settings["edge"][2][0] * self.settings["edge"][2][3]
        fraction_of_z_size = z_size / range
        new_z = self.settings["edge"][2][0] / fraction_of_z_size
        if new_z % 1 != 0:
            new_z = int(new_z) + 1.0
        self.settings["edge"][2][0] = new_z

    def set_grid_by_box_dimensions(
        self, x_limits, y_limits, z_limits, spacing
    ):
        """
        Sets origin and edge as a cuboid box, ranging within the given limits, with voxel size specified by spacing.
        :param x_limits: list [min,max]
        :param y_limits: list [min,max]
        :param z_limits: list [min,max]
        :param spacing: float, or list [x,y,z]
        :return:
        """
        # apparently, this preliminary setting is necessary
        self.set_origin([0, 0, 0])
        self.settings["edge"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # set one dimension at a time
        for i, lim in enumerate([x_limits, y_limits, z_limits]):
            assert lim[0] < lim[1]
            range = lim[1] - lim[0]
            # set origin
            center = lim[0] + (range / 2)
            self.settings["origin"][0][i] = center
            # set edges
            if isinstance(spacing, list):
                space = spacing[i]
            else:
                space = spacing
            ### size of voxel
            self.settings["edge"][i][i + 1] = space
            ### number of voxels
            n_voxels = int(range / space) + 1
            self.settings["edge"][i][0] = n_voxels

    def get_origin(self):
        assert "origin" in self.settings, "There is no origin specified"
        origin = self.settings["origin"]
        return np.array(origin[0])

    def get_text(self):
        text = ""
        if len(self.type) > 0:
            text += "output cube " + self.type + "\n"
        else:
            Warning("No cube type specified")
            text += "output cube" + "CUBETYPE" + "\n"

        for key, values in self.settings.items():
            for v in values:
                text += "cube " + key + " "
                if key in self.parsing_functions:
                    text += self.parsing_functions[key][1](v) + "\n"
                else:
                    print(v)
                    text += v + "\n"

        return text
