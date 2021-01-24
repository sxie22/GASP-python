# coding: utf-8
# Copyright (c) Henniggroup.
# Distributed under the terms of the MIT License.
from __future__ import division, unicode_literals, print_function

import sys
#sys.path = ['/ufrc/hennig/kvs.chaitanya/relaxation/others/gasp_practise/develop/GASP-python'] + sys.path
#sys.path.remove('/home/kvs.chaitanya/gasp_branches/GASP-python')

"""
Run module:

This module is run to do a genetic algorithm structure search.

Usage: python run.py /path/to/gasp/input/file

"""

from gasp import general
from gasp import population
from gasp import objects_maker
from gasp import parameters_printer
from gasp import interface
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import copy
import threading
import random
import sys
import yaml
import os
import datetime

# import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
dask.config.set({'distributed.comm.timeouts.tcp': '3h'})

input_file = 'ga_input.yaml'
with open(input_file, 'r') as f:
    parameters = yaml.load(f)

# make the objects needed by the algorithm
objects_dict = objects_maker.make_objects(parameters)

geometry = objects_dict['geometry']
# substrate related params
# Everythin will be used explicitly as kwargs for energy_calculator
E_sub_prim, n_sub_prim, mu_A, mu_B, mu_C = None, None, None, None, None
lat_match_dict = None
max_area = None
substrate_search = False
substrate_params = {}
if geometry.shape == 'interface':
    substrate_search = True

if substrate_search:
    match_constraints = objects_maker.get_lat_match_params(parameters)
    substrate_params = objects_maker.get_substrate_params(parameters)
    main_keys = ['E_sub_prim', 'n_sub_prim', 'mu_A']
    for key in main_keys:
        if substrate_params[key] is None:
            print ('{} in substrate calculation not provided.'.format(key))
            print ('Quitting...')
            quit()
    lat_match_dict = match_constraints
    lat_match_dict.update(substrate_params)
    max_area = match_constraints['max_area']
    # Parse the primitve substrate structure from input argument
    substrate_prim = general.Cell.from_file('POSCAR_sub')
    # check if substrate has sd_flags ; if no flags, set all to be F F F
    if not 'selective_dynamics' in substrate_prim.site_properties.keys():
        for site in substrate_prim.sites:
            site.properties['selective_dynamics'] = [False, False, False]

# get the objects from the dictionary for convenience
run_dir_name = objects_dict['run_dir_name']
organism_creators = objects_dict['organism_creators']
num_calcs_at_once = objects_dict['num_calcs_at_once']
composition_space = objects_dict['composition_space']
constraints = objects_dict['constraints']
constraints.max_area = max_area   # set max_area from match_constraints
developer = objects_dict['developer']
redundancy_guard = objects_dict['redundancy_guard']
stopping_criteria = objects_dict['stopping_criteria']
energy_calculator = objects_dict['energy_calculator']
pool = objects_dict['pool']
variations = objects_dict['variations']
id_generator = objects_dict['id_generator']
job_specs = objects_dict['job_specs']

# get the path to the run directory - append date and time if
# the given or default run directory already exists
garun_dir = str(os.getcwd()) + '/' + run_dir_name
if os.path.isdir(garun_dir):
    print('Directory {} already exists'.format(garun_dir))
    time = datetime.datetime.now().time()
    date = datetime.datetime.now().date()
    current_date = str(date.month) + '_' + str(date.day) + '_' + \
        str(date.year)
    current_time = str(time.hour) + '_' + str(time.minute) + '_' + \
        str(time.second)
    garun_dir += '_' + current_date + '_' + current_time
    print('Setting the run directory to {}'.format(garun_dir))
# make the run directory and move into it
os.mkdir(garun_dir)
os.chdir(garun_dir)

# make the temp subdirectory where the energy calculations will be done
os.mkdir(garun_dir + '/temp')

# print the search parameters to a file in the run directory
parameters_printer.print_parameters(objects_dict, lat_match_dict=lat_match_dict)

# make the data writer
data_writer = general.DataWriter(garun_dir,
                                 composition_space, sub_search=substrate_search)

# Define worker job parameters
num_calcs_at_once = 2 # TODO: make an option for max_workers in the input file
###############
cluster_job = SLURMCluster(cores=1,
                           memory="2GB",
                           project='hennig',
                           queue='hpg2-compute',
                           interface='ib0',
                           walltime='2:00:00',
                           job_extra=['--ntasks 4', '--nodes=1'])
cluster_job.scale(jobs=num_calcs_at_once) # number of parallel jobs
client  = Client(cluster_job)

whole_pop = []
num_finished_calcs = 0
initial_population = population.InitialPopulation(run_dir_name)

# To hold futures
futures = []

working_jobs = len([i for i, f in enumerate(futures) if not f.done()])

# populate the initial population
for creator in organism_creators:
    print('Making {} organisms with {}'.format(creator.number,
                                               creator.name))
    n_whiles1 = 0
    iface_attempts = 0
    while not creator.is_finished and not stopping_criteria.are_satisfied:

        working_jobs = len([i for i, f in enumerate(futures) \
                                            if not f.done()])
        if working_jobs < num_calcs_at_once:
            # make a new organism - keep trying until we get one
            new_organism = creator.create_organism(
                id_generator, composition_space, constraints, random)
            while new_organism is None and not creator.is_finished:
                new_organism = creator.create_organism(
                    id_generator, composition_space, constraints, random)
            if new_organism is not None:  # loop above could return None
                geometry.unpad(new_organism.cell, new_organism.n_sub,
                                                            constraints)

                if developer.develop(new_organism, composition_space,
                                     constraints, geometry, pool):
                    redundant_organism = redundancy_guard.check_redundancy(
                        new_organism, whole_pop, geometry)
                    if redundant_organism is None:  # no redundancy
                        # add a copy to whole_pop so the organisms in
                        # whole_pop don't change upon relaxation
                        whole_pop.append(copy.deepcopy(new_organism))
                        # pad with vacuum
                        geometry.pad(new_organism.cell)
                        if substrate_search:
                            # lattice match substrate
                            new_organism.cell, new_organism.n_sub = \
                                        interface.run_lat_match(
                                        substrate_prim, new_organism.cell,
                                        match_constraints)
                            kwargs = substrate_params
                            if new_organism.cell is None: #if LMA fail
                                # remove the organism from whole_pop
                                del whole_pop[-1]
                                continue
                            else:
                                # get sd_flags before padding
                                sd_props = new_organism.cell.site_properties[
                                                    'selective_dynamics']
                                geometry.pad(new_organism.cell)
                                # re-assign sd_flags to padded cell
                                for site, prop in zip(new_organism.cell.sites,
                                                        sd_props):
                                    site.properties['selective_dynamics'] = \
                                                        prop
                            if not developer.post_lma_develop(new_organism,
                             composition_space, constraints, geometry, pool):
                                # remove the organism from whole_pop
                                del whole_pop[-1]
                                continue
                        out = client.submit(
                                energy_calculator.do_energy_calculation,
                                new_organism, composition_space,
                                **substrate_params)
                        futures.append(out)

        # process finished calculations and start new ones
        else:
            futures, relaxed_futures = update_futures(futures)
            for future in relaxed_futures:
                if not future.exception():
                    relaxed_organism = future.result()
                    # take care of relaxed organism
                    if relaxed_organism is not None:
                        # To keep it simple, remove_sub() in interface is
                        # changed to unpad()
                        geometry.unpad(relaxed_organism.cell,
                                        relaxed_organism.n_sub, constraints)
                        if developer.develop(relaxed_organism,
                                             composition_space,
                                             constraints, geometry, pool):
                            redundant_organism = \
                                redundancy_guard.check_redundancy(
                                    relaxed_organism, whole_pop, geometry)
                            if redundant_organism is not None:  # redundant
                                if redundant_organism.is_active and \
                                        redundant_organism.epa > \
                                        relaxed_organism.epa:
                                    initial_population.replace_organism(
                                            redundant_organism,
                                            relaxed_organism,
                                            composition_space)
                                    progress = \
                                        initial_population.get_progress(
                                            composition_space)
                                    data_writer.write_data(
                                        relaxed_organism,
                                        num_finished_calcs, progress)
                                    print('Number of energy calculations '
                                          'so far: {} '.format(
                                              num_finished_calcs))
                            else:  # not redundant
                                num_finished_calcs += 1
                                stopping_criteria.update_calc_counter()
                                stopping_criteria.check_organism(
                                    relaxed_organism, redundancy_guard,
                                    geometry)
                                initial_population.add_organism(
                                    relaxed_organism, composition_space)
                                whole_pop.append(relaxed_organism)
                                progress = \
                                    initial_population.get_progress(
                                        composition_space)
                                data_writer.write_data(
                                    relaxed_organism, num_finished_calcs,
                                    progress)
                                print('Number of energy calculations so '
                                      'far: {} '.format(
                                          num_finished_calcs))
                                if creator.is_successes_based and \
                                        relaxed_organism.made_by == \
                                        creator.name:
                                    creator.update_status()
# depending on how the loop above exited, update bookkeeping
if not stopping_criteria.are_satisfied:
    num_finished_calcs = num_finished_calcs - 1
else:
    print ('Stopping criteria achieved within random initial structures..')
    print ('Quitting..')
    quit()

# populate the pool with the initial population
pool.add_initial_population(initial_population, composition_space)
# Make offspring generator
offspring_generator = general.OffspringGenerator()

while not stopping_criteria.are_satisfied:
    working_jobs = len([i for i, f in enumerate(futures) if not f.done()])
    if working_jobs < num_calcs_at_once:
        unrelaxed_offspring = offspring_generator.make_offspring_organism(
            random, pool, variations, geometry, id_generator, whole_pop,
            developer, redundancy_guard, composition_space, constraints)
        whole_pop.append(copy.deepcopy(unrelaxed_offspring))
        geometry.pad(unrelaxed_offspring.cell)

        if substrate_search:
            unrelaxed_offspring.cell, unrelaxed_offspring.n_sub = \
                    interface.run_lat_match(
                    substrate_prim, unrelaxed_offspring.cell, match_constraints)
            kwargs = substrate_params
            if unrelaxed_offspring.cell is None:
                del whole_pop[-1]
                continue
            else:
                # get sd_flags before padding
                sd_props = unrelaxed_offspring.cell.site_properties[
                                    'selective_dynamics']
                geometry.pad(unrelaxed_offspring.cell)
                # re-assign sd_flags to padded cell
                for site, prop in zip(unrelaxed_offspring.cell.sites, sd_props):
                    site.properties['selective_dynamics'] = prop

            if not developer.post_lma_develop(unrelaxed_offspring, composition_space,
                                                constraints, geometry, pool):
                # remove the organism from whole_pop
                del whole_pop[-1]
                continue

        out = client.submit(
                        energy_calculator.do_energy_calculation,
                        unrelaxed_offspring,
                        composition_space,
                        **substrate_params
                            )
        futures.append(out)

    else:  # process finished calculations
        futures, relaxed_futures = update_futures(futures)

        for i, future in enumerate(relaxed_futures):
            if not future.exception():
                relaxed_offspring = future.result()

                # take care of relaxed offspring organism
                if relaxed_offspring is not None:
                    geometry.unpad(relaxed_offspring.cell,
                                        relaxed_offspring.n_sub,
                                        constraints)
                    if developer.develop(relaxed_offspring,
                                         composition_space,
                                         constraints, geometry, pool):
                        # check for redundancy with the the pool first
                        redundant_organism = redundancy_guard.check_redundancy(
                            relaxed_offspring, pool.to_list(), geometry)
                        if redundant_organism is not None:  # redundant
                            if redundant_organism.epa > relaxed_offspring.epa:
                                pool.replace_organism(redundant_organism,
                                                      relaxed_offspring,
                                                      composition_space)
                                pool.compute_fitnesses()
                                pool.compute_selection_probs()
                                pool.print_summary(composition_space)
                                progress = pool.get_progress(composition_space)
                                data_writer.write_data(relaxed_offspring,
                                                       num_finished_calcs,
                                                       progress)
                                print('Number of energy calculations so far: '
                                      '{} '.format(num_finished_calcs))
                        # check for redundancy with all the organisms
                        else:
                            num_finished_calcs += 1
                            stopping_criteria.update_calc_counter()
                            redundant_organism = \
                                redundancy_guard.check_redundancy(
                                    relaxed_offspring, whole_pop, geometry)
                        if redundant_organism is None:  # not redundant
                            stopping_criteria.check_organism(
                                relaxed_offspring, redundancy_guard, geometry)
                            pool.add_organism(relaxed_offspring,
                                              composition_space)
                            whole_pop.append(relaxed_offspring)

                            # check if we've added enough new offspring
                            # organisms to the pool that we can remove the
                            # initial population organisms from the front
                            # (right end) of the queue.
                            if pool.num_adds == pool.size:
                                print('Removing the initial population from '
                                      'the pool ')
                                for _ in range(len(
                                        initial_population.initial_population)):
                                    removed_org = pool.queue.pop()
                                    removed_org.is_active = False
                                    print('Removing organism {} from the '
                                          'pool '.format(removed_org.id))

                            # if the initial population organisms have already
                            # been removed from the pool's queue, then just
                            # need to pop one organism from the front (right
                            # end) of the queue.
                            elif pool.num_adds > pool.size:
                                removed_org = pool.queue.pop()
                                removed_org.is_active = False
                                print('Removing organism {} from the '
                                      'pool '.format(removed_org.id))

                            pool.compute_fitnesses()
                            pool.compute_selection_probs()
                            pool.print_summary(composition_space)
                            progress = pool.get_progress(composition_space)
                            data_writer.write_data(relaxed_offspring,
                                                   num_finished_calcs,
                                                   progress)
                            print('Number of energy calculations so far: '
                                  '{} '.format(num_finished_calcs))

# process all the calculations that were still running when the
# stopping criteria were achieved
while len(futures) > 0:
    futures, relaxed_futures = update_futures(futures)
    for i, future in enumerate(relaxed_futures):
        if not future.exception():
            relaxed_offspring = future.result()

            # take care of relaxed offspring organism
            if relaxed_offspring is not None:
                geometry.unpad(relaxed_offspring.cell,
                                    relaxed_offspring.n_sub, constraints)
                if developer.develop(relaxed_offspring, composition_space,
                                     constraints, geometry, pool):
                    # check for redundancy with the pool first
                    redundant_organism = redundancy_guard.check_redundancy(
                        relaxed_offspring, pool.to_list(), geometry)
                    if redundant_organism is not None:  # redundant
                        if redundant_organism.epa > relaxed_offspring.epa:
                            pool.replace_organism(redundant_organism,
                                                  relaxed_offspring,
                                                  composition_space)
                            pool.compute_fitnesses()
                            pool.compute_selection_probs()
                            pool.print_summary(composition_space)
                            progress = pool.get_progress(composition_space)
                            data_writer.write_data(relaxed_offspring,
                                                   num_finished_calcs,
                                                   progress)
                            print('Number of energy calculations so far: '
                                  '{} '.format(num_finished_calcs))
                    # check for redundancy with all the organisms
                    else:
                        redundant_organism = \
                            redundancy_guard.check_redundancy(
                                relaxed_offspring, whole_pop, geometry)
                    if redundant_organism is None:  # not redundant
                        num_finished_calcs += 1
                        stopping_criteria.update_calc_counter()
                        pool.add_organism(relaxed_offspring,
                                          composition_space)
                        whole_pop.append(relaxed_offspring)
                        removed_org = pool.queue.pop()
                        removed_org.is_active = False
                        print('Removing organism {} from the pool '.format(
                            removed_org.id))
                        pool.compute_fitnesses()
                        pool.compute_selection_probs()
                        pool.print_summary(composition_space)
                        progress = pool.get_progress(composition_space)
                        data_writer.write_data(relaxed_offspring,
                                               num_finished_calcs,
                                               progress)
                        print('Number of energy calculations so far: '
                              '{} '.format(num_finished_calcs))
                        print ('GASP search finished. Quitting..')

def update_futures(all_futures):
    """
    Returns list of active futures and relaxed futures that are to be processed

    Args:
    futures - (list) list of futures objects (concurrent_futures)
    """
    # remove all futures with an exception
    rem_inds, relaxed_inds = [], []
    for i, future in enumerate(all_futures):
        if future.done():
            rem_inds.append(i)
            if not future.exception():
                relaxed_inds.append(i)
            else:
                print (future.exception())

    # relaxed futures that are to be processed
    relaxed_futures = [all_futures[i] for i in relaxed_inds]
    # futures that are still running
    active_futures = [all_futures[i] for i in range(len(all_futures)) if i not in rem_inds]

    return active_futures, relaxed_futures
