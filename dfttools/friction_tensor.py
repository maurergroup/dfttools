from os.path import join
import numpy as np
from ase.units import fs
from dfttools.geometry import AimsGeometry
import dfttools.utils.units as units
from dfttools.utils.periodic_table import PeriodicTable

# unit conversion factors 
J_per_eV = 1.6022e-19
s_over_ps = 1e12
ase_ps = fs*1000
ase_s = fs*1e15


class FrictionTensor:
    def __init__(self, directroy):
        self.geometry = AimsGeometry(
            join(directroy, 'geometry.in'))
        self.friction_tensor_raw = self.read_friction_tensor(
            join(directroy, 'friction_tensor.out'))
        
        
    def read_friction_tensor(self, filename):
        atom_indices = []
        friction_tensor_0 = []
        
        with open(filename) as f:
            lines = f.readlines()
            
            for line in lines:
                if '# n_atom' in line:
                    line = line.strip().split(' ')
                    
                    line_1 = []
                    for l in line:
                        if not l == '':
                            line_1.append(l)
                    
                    atom_index = 3 * (int(line_1[2])-1) + int(line_1[4]) - 1
                    atom_indices.append(atom_index)
                
                if not '#' in line:
                    line = line.strip().split(' ')
                    
                    friction_tensor_line = []
                    for l in line:
                        if not l == '':
                            friction_tensor_line.append(float(l))
                    
                    friction_tensor_0.append(friction_tensor_line)

        friction_tensor_0 = np.array(friction_tensor_0)

        n = len(self.geometry)
        friction_tensor = np.zeros((3*n, 3*n))
        
        for ind_0, atom_index_0 in enumerate(atom_indices):
            for ind_1, atom_index_1 in enumerate(atom_indices):
                friction_tensor[atom_index_0, atom_index_1] = friction_tensor_0[ind_0, ind_1]

        return friction_tensor
    
    
    def get_friction_tensor_raw(self):
        return self.friction_tensor_raw
    

    def get_friction_tensor_in_seconds(self):
        friction_tensor = self.friction_tensor_raw * self.getMassTensor(self.geometry, SI=True)
        friction_tensor /= s_over_ps
        
        return friction_tensor
        
        
    def get_mass_tensor(self, geometry, SI=False):
        periodic_table = PeriodicTable()
        
        mass_vector = []
        
        for ind in range(len(geometry)):
            if geometry.calculate_friction[ind]:
                species = geometry.species[ind]
                atomic_number = periodic_table.get_atomic_number(species)
                if SI:
                    mass = periodic_table.get_atomic_mass(atomic_number) * units.ATOMIC_MASS_IN_KG
                else:
                    mass = periodic_table.get_atomic_mass(atomic_number)
                # append mass for all three spatial direction
                mass_vector += [mass]*3
        
        mass_tensor = np.tile( mass_vector, (len(mass_vector),1) )
        mass_tensor = np.sqrt( mass_tensor * mass_tensor.T )
        
        return mass_tensor
    
    
    def get_dampening_factor(self, vibration):
        friction_tensor = self.getFrictionTensorInSeconds(self.geometry)
        
        vibration /= np.linalg.norm(vibration)
        
        force = friction_tensor.dot( vibration )
        
        eta = vibration.dot( force ) #* J_per_eV / ase_s #* CO_length**2 -> for the torsional dampening factor
        
        return eta
        
        
    def get_life_time(self, vibration):
        """
        returns life time in ps
        """
        #vibration /= np.linalg.norm(vibration)
        
        force = self.friction_tensor_raw.dot( vibration )
        
        eta = vibration.dot( force )
        
        life_time = 1 / eta
        
        return life_time
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        