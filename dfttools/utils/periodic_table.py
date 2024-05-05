import os
import yaml
from typing import Union


#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
COVALENT_RADII = {'Rm': 0.4,'Em': 0.3,'H': 0.3,'He': 0.2,'Li': 1.2,'Be': 0.9,'B': 0.8,'C': 0.7,'N': 0.7,'O': 0.6,'F': 0.5,
                  'Ne': 0.5,'Na': 1.6,'Mg': 1.4,'Al': 1.2,'Si': 1.1,'P': 1.0,'S': 1.0,'Cl': 1.0,'Ar': 1.0,'K': 2.0,
                  'Ca': 1.7,'Sc': 1.7,'Ti': 1.6,'V': 1.5,'Cr': 1.3,'Mn': 1.3,'Fe': 1.3,'Co': 1.2,'Ni': 1.2,
                  'Cu': 1.3, 'Cu_reallylight': 1.3, 'H_am':0.3,
                  'Zn': 1.2,'Ga': 1.2,'Ge': 1.2,'As': 1.1,'Se': 1.2,'Br': 1.2,'Kr': 1.1,'Rb': 2.2,'Sr': 1.9,'Y': 1.9,
                  'Zr': 1.7,'Nb': 1.6,'Mo': 1.5,'Tc': 1.4,'Ru': 1.4,'Rh': 1.4,'Pd': 1.3,
                  'Ag': 1.4,'Ag_reallylight': 1.4,
                  'Cd': 1.4,'In': 1.4,'Sn': 1.3,'Sb': 1.3,'Te': 1.3,'I': 1.3,'Xe': 1.4,
                  'Cs': 2.4,'Ba': 2.1,'La': 2.0,'Ce': 2.0,'Pr': 2.0,'Nd': 2.0,'Pm': 1.9,'Sm': 1.9,'Eu': 1.9,
                  'Gd': 1.9,'Tb': 1.9,'Dy': 1.9,'Ho': 1.9,'Er': 1.8,'Tm': 1.9,'Yb': 1.8,'Lu': 1.8,'Hf': 1.7,
                  'Ta': 1.7,'W': 1.6,'Re': 1.5,'Os': 1.4,'Ir': 1.4,'Pt': 1.3,'Au': 1.3, 'Au_reallylight': 1.3, 'Hg': 1.3,'Tl': 1.4,
                  'Pb': 1.4,'Bi': 1.4,'Po': 1.4,'At': 1.5,'Rn': 1.5,'Fr': 2.6,'Ra': 2.2,'Ac': 2.1,'Th': 2.0,
                  'Pa': 2.0,'U': 1.9,'Np': 1.9,'Pu': 1.8,'Am': 1.8,'Cm': 1.6,'Bk': 0,'Cf': 0,'Es': 0,'Fm': 0,
                  'Md': 0,'No': 0,'Lr': 0,'CP':0.2}


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
            for key in self.periodic_table['order']:
                element_0 = self.periodic_table[key]
                
                if element == element_0['name'] \
                or element == element_0['number'] \
                or element == element_0['symbol']:
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
        return self.get_element_dict(element)['number']
    
    
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
        return self.get_element_dict(element)['atomic_mass']
    
    
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
        return self.get_element_dict(element)['symbol']
    
    
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
        return COVALENT_RADII[self.get_element_dict(element)['symbol']]
        
    