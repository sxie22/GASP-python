# coding: utf-8
# Copywrite (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

"""
Utility script to find continue GA with best structures. Prepares continue_run
a new folder  with start_prev_best structures

Usage: Provide inputs <path to poscar substrate>, <path to poscar 2D> and
       match constraints using for the search

       python get_matching_matrix.py
"""
import os
from shutil import copy

# Inputs
best_n = 20
continue_dir_name = 'continue_run'
new_start_strs = 'start_prev_best'

# go to garun directory of completed garun
run_dir = os.getcwd()
with open('run_data') as r:
    lines = r.readlines()
    lines = lines[4:]
ids, epas = [], []
for line in lines:
    ids.append(int(line.split(' ')[0]))
    epas.append(float(line.split(' ')[6]))
sorted_ids = [x for _, x in sorted(zip(epas, ids))]
best_ids = sorted_ids[:best_n]

os.chdir('../')
os.mkdir(continue_dir_name)
print ('Created new directory {} to continue run'.format(continue_dir_name))
os.chdir(continue_dir_name)
os.mkdir(new_start_strs)
print ('Created new directory {} to save previous best structures'.format(
                                                        new_start_strs))
os.chdir(new_start_strs)
path_to_new = os.getcwd()
os.chdir(run_dir)
for id in best_ids:
    poscar = run_dir + '/POSCAR.{}'.format(id)
    copy(poscar, path_to_new)

print ('Copied best structures to new_start_strs directory')
