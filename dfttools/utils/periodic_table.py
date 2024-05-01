import os
import yaml
from typing import Union


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
    
    
    def get_atomic_number(self, element: str) -> int:
        """
        Returns the atomic number if given the species as a string.

        Parameters
        ----------
        species : str
            Name or chemical sysmbol of the atomic species.

        Returns
        -------
        int
            atomic number.

        """
        return self.get_element_dict(element)['number']