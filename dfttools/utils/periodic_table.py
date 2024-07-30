import os
import yaml
from typing import List, Union


#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
COVALENT_RADII = {
    "Rm": 0.4,
    "Em": 0.3,
    "H": 0.3,
    "He": 0.2,
    "Li": 1.2,
    "Be": 0.9,
    "B": 0.8,
    "C": 0.7,
    "N": 0.7,
    "O": 0.6,
    "F": 0.5,
    "Ne": 0.5,
    "Na": 1.6,
    "Mg": 1.4,
    "Al": 1.2,
    "Si": 1.1,
    "P": 1.0,
    "S": 1.0,
    "Cl": 1.0,
    "Ar": 1.0,
    "K": 2.0,
    "Ca": 1.7,
    "Sc": 1.7,
    "Ti": 1.6,
    "V": 1.5,
    "Cr": 1.3,
    "Mn": 1.3,
    "Fe": 1.3,
    "Co": 1.2,
    "Ni": 1.2,
    "Cu": 1.3,
    "Cu_reallylight": 1.3,
    "H_am": 0.3,
    "Zn": 1.2,
    "Ga": 1.2,
    "Ge": 1.2,
    "As": 1.1,
    "Se": 1.2,
    "Br": 1.2,
    "Kr": 1.1,
    "Rb": 2.2,
    "Sr": 1.9,
    "Y": 1.9,
    "Zr": 1.7,
    "Nb": 1.6,
    "Mo": 1.5,
    "Tc": 1.4,
    "Ru": 1.4,
    "Rh": 1.4,
    "Pd": 1.3,
    "Ag": 1.4,
    "Ag_reallylight": 1.4,
    "Cd": 1.4,
    "In": 1.4,
    "Sn": 1.3,
    "Sb": 1.3,
    "Te": 1.3,
    "I": 1.3,
    "Xe": 1.4,
    "Cs": 2.4,
    "Ba": 2.1,
    "La": 2.0,
    "Ce": 2.0,
    "Pr": 2.0,
    "Nd": 2.0,
    "Pm": 1.9,
    "Sm": 1.9,
    "Eu": 1.9,
    "Gd": 1.9,
    "Tb": 1.9,
    "Dy": 1.9,
    "Ho": 1.9,
    "Er": 1.8,
    "Tm": 1.9,
    "Yb": 1.8,
    "Lu": 1.8,
    "Hf": 1.7,
    "Ta": 1.7,
    "W": 1.6,
    "Re": 1.5,
    "Os": 1.4,
    "Ir": 1.4,
    "Pt": 1.3,
    "Au": 1.3,
    "Au_reallylight": 1.3,
    "Hg": 1.3,
    "Tl": 1.4,
    "Pb": 1.4,
    "Bi": 1.4,
    "Po": 1.4,
    "At": 1.5,
    "Rn": 1.5,
    "Fr": 2.6,
    "Ra": 2.2,
    "Ac": 2.1,
    "Th": 2.0,
    "Pa": 2.0,
    "U": 1.9,
    "Np": 1.9,
    "Pu": 1.8,
    "Am": 1.8,
    "Cm": 1.6,
    "Bk": 0,
    "Cf": 0,
    "Es": 0,
    "Fm": 0,
    "Md": 0,
    "No": 0,
    "Lr": 0,
    "CP": 0.2,
}

# Colormaps for elements
# Jmol colors.  See: http://jmol.sourceforge.net/jscolors/#color_U
SPECIES_COLORS = {
    "Rm": [0.67, 0.82, 0.01],
    "Em": [1.0, 0.0, 1.0],
    "H": [1.0, 1.0, 1.0],
    "He": [0.851, 1.0, 1.0],
    "Li": [0.8, 0.502, 1.0],
    "Be": [0.761, 1.0, 0.0],
    "B": [1.0, 0.71, 0.71],
    "C": [0.565, 0.565, 0.565],
    "N": [0.188, 0.314, 0.973],
    "O": [1.0, 0.051, 0.051],
    "F": [0.565, 0.878, 0.314],
    "Ne": [0.702, 0.89, 0.961],
    "Na": [0.671, 0.361, 0.949],
    "Mg": [0.541, 1.0, 0.0],
    "Al": [0.749, 0.651, 0.651],
    "Si": [0.941, 0.784, 0.627],
    "P": [1.0, 0.502, 0.0],
    "S": [1.0, 1.0, 0.188],
    "Cl": [0.122, 0.941, 0.122],
    "Ar": [0.502, 0.82, 0.89],
    "K": [0.561, 0.251, 0.831],
    "Ca": [0.239, 1.0, 0.0],
    "Sc": [0.902, 0.902, 0.902],
    "Ti": [0.749, 0.761, 0.78],
    "V": [0.651, 0.651, 0.671],
    "Cr": [0.541, 0.6, 0.78],
    "Mn": [0.612, 0.478, 0.78],
    "Fe": [0.878, 0.4, 0.2],
    "Co": [0.941, 0.565, 0.627],
    "Ni": [0.314, 0.816, 0.314],
    "Cu": [0.784, 0.502, 0.2],
    "Zn": [0.49, 0.502, 0.69],
    "Ga": [0.761, 0.561, 0.561],
    "Ge": [0.4, 0.561, 0.561],
    "As": [0.741, 0.502, 0.89],
    "Se": [1.0, 0.631, 0.0],
    "Br": [0.651, 0.161, 0.161],
    "Kr": [0.361, 0.722, 0.82],
    "Rb": [0.439, 0.18, 0.69],
    "Sr": [0.0, 1.0, 0.0],
    "Y": [0.58, 1.0, 1.0],
    "Zr": [0.58, 0.878, 0.878],
    "Nb": [0.451, 0.761, 0.788],
    "Mo": [0.329, 0.71, 0.71],
    "Tc": [0.231, 0.62, 0.62],
    "Ru": [0.141, 0.561, 0.561],
    "Rh": [0.039, 0.49, 0.549],
    "Pd": [0.0, 0.412, 0.522],
    "Ag": [0.753, 0.753, 0.753],
    "Cd": [1.0, 0.851, 0.561],
    "In": [0.651, 0.459, 0.451],
    "Sn": [0.4, 0.502, 0.502],
    "Sb": [0.62, 0.388, 0.71],
    "Te": [0.831, 0.478, 0.0],
    "I": [0.58, 0.0, 0.58],
    "Xe": [0.259, 0.62, 0.69],
    "Cs": [0.341, 0.09, 0.561],
    "Ba": [0.0, 0.788, 0.0],
    "La": [0.439, 0.831, 1.0],
    "Ce": [1.0, 1.0, 0.78],
    "Pr": [0.851, 1.0, 0.78],
    "Nd": [0.78, 1.0, 0.78],
    "Pm": [0.639, 1.0, 0.78],
    "Sm": [0.561, 1.0, 0.78],
    "Eu": [0.38, 1.0, 0.78],
    "Gd": [0.271, 1.0, 0.78],
    "Tb": [0.188, 1.0, 0.78],
    "Dy": [0.122, 1.0, 0.78],
    "Ho": [0.0, 1.0, 0.612],
    "Er": [0.0, 0.902, 0.459],
    "Tm": [0.0, 0.831, 0.322],
    "Yb": [0.0, 0.749, 0.22],
    "Lu": [0.0, 0.671, 0.141],
    "Hf": [0.302, 0.761, 1.0],
    "Ta": [0.302, 0.651, 1.0],
    "W": [0.129, 0.58, 0.839],
    "Re": [0.149, 0.49, 0.671],
    "Os": [0.149, 0.4, 0.588],
    "Ir": [0.09, 0.329, 0.529],
    "Pt": [0.816, 0.816, 0.878],
    "Au": [1.0, 0.82, 0.137],
    "Hg": [0.722, 0.722, 0.816],
    "Tl": [0.651, 0.329, 0.302],
    "Pb": [0.341, 0.349, 0.38],
    "Bi": [0.62, 0.31, 0.71],
    "Po": [0.671, 0.361, 0.0],
    "At": [0.459, 0.31, 0.271],
    "Rn": [0.259, 0.51, 0.588],
    "Fr": [0.259, 0.0, 0.4],
    "Ra": [0.0, 0.49, 0.0],
    "Ac": [0.439, 0.671, 0.98],
    "Th": [0.0, 0.729, 1.0],
    "Pa": [0.0, 0.631, 1.0],
    "U": [0.0, 0.561, 1.0],
    "Np": [0.0, 0.502, 1.0],
    "Pu": [0.0, 0.42, 1.0],
    "Am": [0.329, 0.361, 0.949],
    "Cm": [0.471, 0.361, 0.89],
    "Bk": [0.541, 0.31, 0.89],
    "Cf": [0.631, 0.212, 0.831],
    "Es": [0.702, 0.122, 0.831],
    "Fm": [0.702, 0.122, 0.729],
    "Md": [0.702, 0.051, 0.651],
    "No": [0.741, 0.051, 0.529],
    "Lr": [0.78, 0.0, 0.4],
}


class PeriodicTable:
    """
    Create a periodic table object

    Returns
    -------
    dict
        a dictionary representing the periodic table
    """

    def __init__(self):
        self.periodic_table = self.load()

    def load(self) -> dict:
        file_path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(file_path, "periodic_table.yml"), "r") as pt:
            periodic_table = yaml.safe_load(pt)

        return periodic_table

    def get_element_dict(self, element: Union[str, int]) -> dict:
        element_dict = None

        if element in self.periodic_table:
            element_dict = self.periodic_table[element]
        else:
            for key in self.periodic_table["order"]:
                element_0 = self.periodic_table[key]

                if (
                    element == element_0["name"]
                    or element == element_0["number"]
                    or element == element_0["symbol"]
                ):
                    element_dict = element_0
                    break

        if element_dict is None:
            raise ValueError(f'Could not find element "{element}" in periodic table!')

        return element_dict

    def get_atomic_number(self, element: Union[str, int]) -> int:
        """
        Returns the atomic number if given the species as a string.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        int
            atomic number.

        """
        return self.get_element_dict(element)["number"]

    def get_atomic_mass(self, element: Union[str, int]) -> float:
        """
        Returns the atomic mass if given the species as a string.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            atomic mass in atomic units.

        """
        return self.get_element_dict(element)["atomic_mass"]

    def get_chemical_symbol(self, element: Union[str, int]) -> float:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            atomic mass in atomic units.

        """
        return self.get_element_dict(element)["symbol"]

    def get_covalent_radius(self, element: Union[str, int]) -> float:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            Covalent radius in atomic units.

        """
        return COVALENT_RADII[self.get_element_dict(element)["symbol"]]

    def get_species_colors(self, element: Union[str, int]) -> List[float]:
        """
        Returns the chemical symbol if given the species as an atomic number.

        Parameters
        ----------
        species : str or int
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        float
            Covalent radius in atomic units.

        """
        return SPECIES_COLORS[self.get_element_dict(element)["symbol"]]
