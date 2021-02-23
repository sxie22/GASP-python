# coding: utf-8
# Copyright (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

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
from time import sleep

def main():
    # get dictionaries from the input file (in yaml format)
    if len(sys.argv) < 2:
        print('No input file given.')
        print('Quitting...')
        quit()
    else:
        input_file = os.path.abspath(sys.argv[1])

    try:
        with open(input_file, 'r') as f:
            parameters = yaml.load(f)
    except:
        print('Error reading input file.')
        print('Quitting...')
        quit()

    # make the objects needed by the algorithm
    objects_dict = objects_maker.make_objects(parameters)

    geometry = objects_dict['geometry']
    # substrate related params
    # Everythin will be used explicitly as kwargs for energy_calculator
    E_sub_prim, n_sub_prim, mu_A, mu_B, mu_C = None, None, None, None, None
    lat_match_dict = None
    substrate_search = False
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
        # Parse the primitve substrate structure from input argument
        sub_cell = general.Cell.from_file(os.path.abspath(sys.argv[2]))
        # make it conventional_standard_structure using pymatgen to avoid issues
        spgr_obj = SpacegroupAnalyzer(sub_cell)
        substrate_prim = spgr_obj.get_refined_structure()

    # get the objects from the dictionary for convenience
    run_dir_name = objects_dict['run_dir_name']
    organism_creators = objects_dict['organism_creators']
    num_calcs_at_once = objects_dict['num_calcs_at_once']
    composition_space = objects_dict['composition_space']
    constraints = objects_dict['constraints']
    developer = objects_dict['developer']
    redundancy_guard = objects_dict['redundancy_guard']
    stopping_criteria = objects_dict['stopping_criteria']
    energy_calculator = objects_dict['energy_calculator']
    pool = objects_dict['pool']
    variations = objects_dict['variations']
    id_generator = objects_dict['id_generator']

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

    whole_pop = []
    num_finished_calcs = 0
    threads = []
    initial_population = population.InitialPopulation(run_dir_name)

    # To temporarily hold relaxed organisms. The key to each relaxed organism
    # is the index of the Thread in the list threads that did the energy
    # calculation.
    relaxed_organisms = {}

    # populate the initial population
    for creator in organism_creators:
        print('Making {} organisms with {}'.format(creator.number,
                                                   creator.name))
        n_whiles1 = 0
        iface_attempts = 0
        while not creator.is_finished and not stopping_criteria.are_satisfied:

            n_whiles1 += 1
            # start initial batch of energy calculations
            if len(threads) < num_calcs_at_once:
                # make a new organism - keep trying until we get one
                new_organism = creator.create_organism(
                    id_generator, composition_space, constraints, random)
                n_whiles2 = 0
                while new_organism is None and not creator.is_finished:
                    n_whiles2 += 1
                    if n_whiles2 % 1000 == 0:
                        print ('145: Program can\'t make new random organism')
                        print ('whiles1: {0}\nwhiles2: {1}'.format(
                                                        n_whiles1, n_whiles2))
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
                            kwargs = {}
                            if substrate_search:
                                iface_attempts += 1
                                if iface_attempts % 1000 == 0:
                                    print ('172: Failing at making interface')
                                # lattice match substrate
                                new_organism.cell, new_organism.n_sub, \
                                            new_organism.sd_index = \
                                            interface.run_lat_match(
                                            substrate_prim, new_organism.cell,
                                            match_constraints)
                                kwargs = substrate_params
                                if new_organism.cell is None: #if LMA fail
                                    # remove the organism from whole_pop
                                    del whole_pop[-1]
                                    continue
                                else:
                                    geometry.pad(new_organism.cell)
                                if not developer.post_lma_develop(new_organism,
                                 composition_space, constraints, geometry, pool):
                                    # remove the organism from whole_pop
                                    del whole_pop[-1]
                                    continue
                            iface_attempts = 0
                            n_whiles1 = 0
                            stopping_criteria.update_calc_counter()
                            index = len(threads)
                            thread = threading.Thread(
                                target=energy_calculator.do_energy_calculation,
                                args=[new_organism, relaxed_organisms,
                                      index, composition_space],
                                kwargs=kwargs)
                            thread.start()
                            threads.append(thread)
                            sleep(5)

            # process finished calculations and start new ones
            else:
                alive_threads = threading.enumerate()
                if len(alive_threads) == len(threads):
                    sleep(300)
                for index, thread in enumerate(threads):
                    if not thread.is_alive():
                        num_finished_calcs += 1
                        relaxed_organism = relaxed_organisms[index]
                        relaxed_organisms[index] = None

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

                        # make another organism for the initial population
                        started_new_calc = False
                        n_whiles3 = 0
                        iface2_attempts = 0
                        while not started_new_calc and not creator.is_finished:
                            n_whiles3 +=1
                            new_organism = creator.create_organism(
                                id_generator, composition_space,
                                constraints, random)
                            n_whiles4 = 0
                            while new_organism is None and not \
                                    creator.is_finished:
                                n_whiles4 += 1
                                if n_whiles4 % 1000 == 0:
                                    print ('269: Program could not make new '
                                                'random organism')
                                    print ('whiles3: {0}\nwhiles4: {1}'.format(
                                                    n_whiles3, n_whiles4))
                                new_organism = creator.create_organism(
                                    id_generator, composition_space,
                                    constraints, random)
                            if new_organism is not None:
                                geometry.unpad(new_organism.cell,
                                                new_organism.n_sub, constraints)
                                if developer.develop(new_organism,
                                                     composition_space,
                                                     constraints, geometry,
                                                     pool):
                                    redundant_organism = \
                                        redundancy_guard.check_redundancy(
                                            new_organism, whole_pop, geometry)
                                    if redundant_organism is None:  # no redundancy
                                        # add a copy to whole_pop so the organisms in
                                        # whole_pop don't change upon relaxation
                                        whole_pop.append(
                                                    copy.deepcopy(new_organism))
                                        # pad with vacuum
                                        geometry.pad(new_organism.cell)
                                        kwargs = {}
                                        if substrate_search:
                                            iface2_attempts += 1
                                            if iface2_attempts % 1000 == 0:
                                                print ('303: Failing at making interface')
                                            # lattice match substrate
                                            new_organism.cell, \
                                            new_organism.n_sub, \
                                            new_organism.sd_index = \
                                                        interface.run_lat_match(
                                                        substrate_prim,
                                                        new_organism.cell,
                                                        match_constraints)
                                            kwargs = substrate_params
                                            if new_organism.cell is None: #if LMA fail
                                                # remove the organism from whole_pop
                                                del whole_pop[-1]
                                                continue
                                            else:
                                                geometry.pad(new_organism.cell)
                                            if not developer.post_lma_develop(
                                                 new_organism,composition_space,
                                                   constraints, geometry, pool):
                                                # remove the organism from whole_pop
                                                del whole_pop[-1]
                                                continue
                                        n_whiles3 = 0
                                        iface2_attempts = 0
                                        stopping_criteria.update_calc_counter()
                                        new_thread = threading.Thread(
                                            target=energy_calculator.do_energy_calculation,
                                            args=[new_organism,
                                                  relaxed_organisms, index,
                                                  composition_space],
                                            kwargs=kwargs)
                                        new_thread.start()
                                        threads[index] = new_thread
                                        started_new_calc = True


    # depending on how the loop above exited, update bookkeeping
    if not stopping_criteria.are_satisfied:
        num_finished_calcs = num_finished_calcs - 1

    # process all the calculations that were still running when the last
    # creator finished
    num_to_get = num_calcs_at_once  # number of threads left to handle
    handled_indices = []  # the indices of the threads we've already handled
    n_whiles5 = 0
    while num_to_get > 0:
        n_whiles5 += 1
        if n_whiles5 % 10000 == 0:
            print ('340: processing threads whiles: ', n_whiles5)
        alive_threads = threading.enumerate()
        if len(alive_threads) == len(threads):
            sleep(300)
        for index, thread in enumerate(threads):
            if not thread.is_alive() and index not in handled_indices:
                num_finished_calcs += 1
                relaxed_organism = relaxed_organisms[index]
                num_to_get = num_to_get - 1
                handled_indices.append(index)
                relaxed_organisms[index] = None

                # take care of relaxed organism
                if relaxed_organism is not None:
                    geometry.unpad(relaxed_organism.cell,
                                        relaxed_organism.n_sub, constraints)
                    if developer.develop(relaxed_organism, composition_space,
                                         constraints, geometry, pool):
                        redundant_organism = redundancy_guard.check_redundancy(
                            relaxed_organism, whole_pop, geometry)
                        if redundant_organism is not None:  # redundant
                            if redundant_organism.is_active and \
                                        redundant_organism.epa > \
                                        relaxed_organism.epa:
                                initial_population.replace_organism(
                                    redundant_organism, relaxed_organism,
                                    composition_space)
                                progress = initial_population.get_progress(
                                    composition_space)
                                data_writer.write_data(relaxed_organism,
                                                       num_finished_calcs,
                                                       progress)
                                print('Number of energy calculations so far: '
                                      '{} '.format(num_finished_calcs))
                        else:  # no redundancy
                            stopping_criteria.check_organism(
                                relaxed_organism, redundancy_guard, geometry)
                            initial_population.add_organism(relaxed_organism,
                                                            composition_space)
                            whole_pop.append(relaxed_organism)
                            progress = initial_population.get_progress(
                                    composition_space)
                            data_writer.write_data(relaxed_organism,
                                                   num_finished_calcs,
                                                   progress)
                            print('Number of energy calculations so far: '
                                  '{} '.format(num_finished_calcs))

    # check if the stopping criteria were already met when making the initial
    # population
    if stopping_criteria.are_satisfied:
        quit()

    # populate the pool with the initial population
    pool.add_initial_population(initial_population, composition_space)

    # To temporarily hold relaxed organisms. The key to each relaxed organism
    # is the index of the Thread in the list threads that did the energy
    # calculation.
    relaxed_organisms = {}

    offspring_generator = general.OffspringGenerator()
    threads = []

    # create the initial batch of offspring organisms and submit them for
    # energy calculations
    # while statement because unrelaxed_offspring could fail lattice matching
    n_whiles5 = 0
    iface3_attempts = 0
    while len(threads) < num_calcs_at_once:
        n_whiles5 += 1
        unrelaxed_offspring = offspring_generator.make_offspring_organism(
            random, pool, variations, geometry, id_generator, whole_pop,
            developer, redundancy_guard, composition_space, constraints)
        whole_pop.append(copy.deepcopy(unrelaxed_offspring))
        geometry.pad(unrelaxed_offspring.cell)
        kwargs = {}
        if substrate_search:
            iface3_attempts += 1
            if iface3_attempts % 1000 == 0:
                print ('427: Failing at making interface')
                print ('whiles, iface attempts: ', (n_whiles5, iface3_attempts))
                if iface3_attempts == 10000:
                    quit()
            geometry.pad(unrelaxed_offspring.cell)
            unrelaxed_offspring.cell, unrelaxed_offspring.n_sub, \
            unrelaxed_offspring.sd_index = interface.run_lat_match(
                    substrate_prim, unrelaxed_offspring.cell, match_constraints)
            kwargs = substrate_params
            if unrelaxed_offspring.cell is None:
                del whole_pop[-1]
                continue
            else:
                geometry.pad(unrelaxed_offspring.cell)
            if not developer.post_lma_develop(unrelaxed_offspring, composition_space,
                                                constraints, geometry, pool):
                # remove the organism from whole_pop
                del whole_pop[-1]
                continue
        iface3_attempts = 0
        stopping_criteria.update_calc_counter()
        index = len(threads)
        new_thread = threading.Thread(
            target=energy_calculator.do_energy_calculation,
            args=[unrelaxed_offspring, relaxed_organisms, index,
                  composition_space], kwargs=kwargs)
        new_thread.start()
        threads.append(new_thread)
        sleep(5)

    # process finished calculations and start new ones
    n_whiles6 = 0
    while not stopping_criteria.are_satisfied:
        n_whiles6 += 1
        for index, thread in enumerate(threads):
            alive_threads = threading.enumerate()
            if len(alive_threads) == len(threads):
                sleep(300)
            if not thread.is_alive():
                num_finished_calcs += 1
                relaxed_offspring = relaxed_organisms[index]
                relaxed_organisms[index] = None

                # take care of relaxed offspring organism
                if relaxed_offspring is not None:
                    geometry.unpad(relaxed_offspring.cell,
                                        relaxed_offspring.n_sub, constraints)
                    if developer.develop(relaxed_offspring, composition_space,
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

                # make another offspring organism
                started_new_calc = False
                n_whiles7 = 0
                iface4_attempts = 0
                while not stopping_criteria.are_satisfied and \
                                            not started_new_calc:
                    n_whiles7 += 1
                    unrelaxed_offspring = \
                        offspring_generator.make_offspring_organism(
                            random, pool, variations, geometry, id_generator,
                            whole_pop, developer, redundancy_guard,
                            composition_space, constraints)
                    whole_pop.append(copy.deepcopy(unrelaxed_offspring))
                    geometry.pad(unrelaxed_offspring.cell)
                    kwargs = {}

                    if substrate_search:
                        iface4_attempts += 1
                        if iface4_attempts % 1000 == 0:
                            print ('554: Failing at making interface')
                            print ('whiles, iface attempts: ', (n_whiles7, iface4_attempts))
                            if iface3_attempts == 10000:
                                quit()
                        unrelaxed_offspring.cell, unrelaxed_offspring.n_sub, \
                        unrelaxed_offspring.sd_index = interface.run_lat_match(
                                substrate_prim, unrelaxed_offspring.cell, match_constraints)
                        kwargs = substrate_params
                        if unrelaxed_offspring.cell is None:
                            del whole_pop[-1]
                            continue
                        else:
                            geometry.pad(unrelaxed_offspring.cell)
                        if not developer.post_lma_develop(unrelaxed_offspring,
                         composition_space, constraints, geometry, pool):
                            # remove the organism from whole_pop
                            del whole_pop[-1]
                            continue
                    iface4_attempts = 0
                    stopping_criteria.update_calc_counter()
                    new_thread = threading.Thread(
                        target=energy_calculator.do_energy_calculation,
                        args=[unrelaxed_offspring, relaxed_organisms,
                              index, composition_space], kwargs=kwargs)
                    new_thread.start()
                    threads[index] = new_thread
                    started_new_calc = True

    # process all the calculations that were still running when the
    # stopping criteria were achieved
    num_to_get = num_calcs_at_once  # how many threads we have left to handle
    handled_indices = []  # the indices of the threads we've already handled
    while num_to_get > 0:
        for index, thread in enumerate(threads):
            alive_threads = threading.enumerate()
            if len(alive_threads) == len(threads):
                sleep(300)
            if not thread.is_alive() and index not in handled_indices:
                num_finished_calcs += 1
                relaxed_offspring = relaxed_organisms[index]
                num_to_get -= 1
                handled_indices.append(index)
                relaxed_organisms[index] = None

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


if __name__ == "__main__":
    main()
