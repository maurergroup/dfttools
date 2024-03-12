#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:37:27 2024

@author: lukas
"""

import os
from os.path import join


def find_all_aims_output_files(directory, include_restart=True, allow_all_out_files=False):
    """Recursively searches for AIMS output files and returns their full filenames as a list"""
    aims_fnames = []
    
    for root, directories, files in os.walk(directory):
        fname = find_aims_output_file(root, allow_all_out_files)
        
        if len(fname) > 0:
            if include_restart:
                aims_fnames.append(join(root,fname))
            else:
                root_name = os.path.basename(os.path.normpath(root))
                is_restart_folder = (len(root_name) == len('restartXX') and root_name.startswith('restart'))
                if not is_restart_folder:
                    aims_fnames.append(join(root,fname))
                    
    return aims_fnames


def find_aims_output_file(calc_dir, allow_all_out_files = False):
    """ Searches directory for output files """
    return find_file(calc_dir,
                    allow_all_out_files=allow_all_out_files,
                    list_of_filenames=['aims.out','out.aims','output','output.aims','aims.output'])


def find_vasp_output_file(calc_dir):
    """ Searches directory for output files """
    return find_file(calc_dir,
                    allow_all_out_files=False,
                    list_of_filenames=['outcar'])


def find_file(calc_dir, allow_all_out_files=False, list_of_filenames=[]):
    """ Searches directory for output files """
    allfiles = [f for f in os.path.listdir(calc_dir) if os.path.isfile(join(calc_dir, f))]
    filename = []
    for f in allfiles:
        if f.lower() in list_of_filenames:
            filename.append(f)
    
    if allow_all_out_files:
        if len(filename) == 0:
            filename = [f for f in allfiles if f.endswith('.out')]
    
    if len(filename) == 1:
        return filename[0]
    elif len(filename) == 0:
        return ''
    else:
        raise Exception("Multiple output files found: {}, {}".format(calc_dir, filename))
