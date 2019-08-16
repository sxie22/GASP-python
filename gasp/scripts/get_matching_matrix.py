# coding: utf-8
# Copywrite (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

"""
Utility script to find if endpoint slabs are matching with substrate

Usage: Provide inputs <path to poscar substrate>, <path to poscar 2D> and
       match constraints using for the search

       python get_matching_matrix.py
"""
from gasp.general import Cell
from gasp import interface
from pymatgen.core.lattice import Lattice
import numpy as np

#### Inputs for lattice matching #####
# substrate file path
sub_file = 'POSCAR_sub'
# twod file path
twod_file = 'POSCAR_twod'
sub = Cell.from_file(sub_file)
twod = Cell.from_file(twod_file)

# match constraints from ga_input.yaml as a dictionary
match_constraints = {'max_area':90, 'max_mismatch':0.06, 'max_angle_diff':2,
                     'r1r2_tol':0.04, 'separation': 2.2, 'nlayers_substrate':1,
                     'nlayers_2d':1, 'sd_layers':0}
#########################################

# Check if LMA is working for existing twod structure
iface, n_sub, z_ub = interface.run_lat_match(sub, twod, match_constraints)

if iface is None:
    stretch, squeeze = True, True
else:
    print ('Current twod lattice matches with substrate. No changes needed!')    

# Stretch and squeeze the lattice of twod film until we find a lattice match
for n in range(20): # 20 % stretch or squeeze being tested
    if not stretch and not squeeze:
        break
    matrix = twod.lattice.matrix
    # squeeze
    if n % 2 == 0 and squeeze: # even number
        factor = (1 - 0.01 * n)
    # stretch
    if n % 2 == 1 and stretch:
        factor = (1 + 0.01 * n)

    scaled_x = matrix[0] * factor
    scaled_y = matrix[1] * factor
    new_matrix = np.array([scaled_x, scaled_y, matrix[2]])
    new_lattice = Lattice(new_matrix)

    twod.modify_lattice(new_lattice)
    iface, n_sub, z_ub = interface.run_lat_match(sub, twod, match_constraints)
    if iface is not None:
        if factor > 1 and stretch:
            stretch = False
            print ('Stretching lattice matches at factor {}'.format(factor))
            print (new_matrix)
        if factor < 1 and squeeze:
            squeeze = False
            print ('Squeezing lattice matches at factor {}'.format(factor))
            print (new_matrix)
