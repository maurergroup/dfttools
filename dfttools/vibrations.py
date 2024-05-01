#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:39:18 2024

@author: lukas
"""
import numpy as np


class Vibrations():
    def __int__(self, geometry):
        self.geometry = geometry
        self.coords = None
        self.forces = None
    
    
    def set_coords(self, coords):
        self.coords = coords
        
    
    def set_forces(self, forces):
        self.forces = forces
    
    
    def get_hessian(self, set_constrained_atoms_zero: bool=True) -> np.array:
        
        N = len(self.geometry) * 3
        H = np.zeros([N, N])
        
        coords_0 = self.coords[0].flatten()
        F_0 = self.forces[0].flatten()
        
        n_forces = np.zeros(N, np.int64)
        
        for c, F in zip(self.coords, self.forces):
            dF = F.flatten() - F_0
            dx = c.flatten() - coords_0
            ind = np.argmax(np.abs(dx))
            n_forces[ind] += 1
            displacement = dx[ind]
            H[ind, :] -= dF / displacement

        for row in range(H.shape[0]):
            if n_forces[row] > 0:
                H[row, :] /= n_forces[row]  # prevent div by zero for unknown forces
        
        if set_constrained_atoms_zero:
            constrained = self.geometry.constrain_relax.flatten()
            H[constrained, :] = 0
            H[:, constrained] = 0
            
        return H
    
    
    # this will all go
    def getHessian(calc_dir,
                   get_geometries=False,
                   set_constrained_atoms_zero=False,
                   reference_geometry=None):
        """
        Parameters
        ----------
        calc_dir
        get_geometries                  boolean     return hessian AND the distorted geometries (as a tuple)
        set_constrained_atoms_zero      boolean     set elements of the hessian that correspond to constrained atoms to zero
        reference_geometry              GeoometryFile       pass the undistorted geometry explicitly
                                                            instead of using the first geomery of the calculations
                                                            (if you used the setupHessian function, just use None)

        Returns
        -------

        """
        file_filter = join(calc_dir, '*/aims.out')
        output_fnames = sorted(glob.glob(file_filter))
        geometries = []

        if len(output_fnames) == 0:
            raise ValueError("No files found in {}".format(file_filter))

        for calc_ind, output_fname in enumerate(output_fnames):
            aims = AIMSOutput(output_fname)
            geom = aims.getGeometryFile()
            geometries.append(geom)
            F = aims.getGradients()

            if np.any(np.isnan(F)):
                raise RuntimeError("Error in grep of forces from out. Where they calculated properly?")        

            if calc_ind == 0:
                N = len(geom) * 3
                H = np.zeros([N, N])
                F0 = F.flatten()

                if reference_geometry is None:
                    # no explicit reference geometry was passed -> use first calculation
                    warnings.warn('No reference geometry has been passed! In this case, the first calculation will be used as reference geometry.')
                    
                    geom0 = geom
                else:
                    # a reference geometry was passed explicitly -> use it
                    if not isinstance(reference_geometry, GeometryFile):
                        raise ValueError
                    geom0 = reference_geometry

                coords0 = geom0.coords.flatten()
                n_forces = np.zeros(N, np.int64)
                
            # if the first geometry is NOT used as reference it is a normal calculation
            if calc_ind != 0 or reference_geometry is not None:
                dF = F.flatten() - F0
                dx = geom.coords.flatten() - coords0
                ind = np.argmax(np.abs(dx))
                n_forces[ind] += 1
                displacement = dx[ind]
                H[ind, :] -= dF / displacement

        for row in range(H.shape[0]):
            if n_forces[row] > 0:
                H[row, :] /= n_forces[row]  # prevent div by zero for unknown forces

        if set_constrained_atoms_zero:
            constrained = geometries[0].constrain_relax.flatten()
            H[constrained, :] = 0
            H[:, constrained] = 0

        if get_geometries:
            return H, geometries
        else:
            return H