# coding: utf-8
# Copyright (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function


"""
Parameters Printer module:

This module contains a function for printing the parameters
of a structure search to a file for the user's reference.

"""

import os


def print_parameters(objects_dict, lat_match_dict=None):
    """
    Prints out the parameters for the search to a file called 'ga_parameters'
    inside the garun directory.

    Args:
        objects_dict: a dictionary of objects used by the algorithm, as
            returned by the make_objects method

        lat_match_dict: a dictionary with all the lattice match parameters and
            substrate parameters (for interface geometry only)
    """

    # get all the objects from the dictionary
    run_dir_name = objects_dict['run_dir_name']
    organism_creators = objects_dict['organism_creators']
    num_calcs_at_once = objects_dict['num_calcs_at_once']
    composition_space = objects_dict['composition_space']
    developer = objects_dict['developer']
    constraints = objects_dict['constraints']
    geometry = objects_dict['geometry']
    redundancy_guard = objects_dict['redundancy_guard']
    stopping_criteria = objects_dict['stopping_criteria']
    energy_calculator = objects_dict['energy_calculator']
    pool = objects_dict['pool']
    variations = objects_dict['variations']
    job_specs = objects_dict['job_specs']

    # make the file where the parameters will be printed
    with open(os.getcwd() + '/ga_parameters', 'w') as parameters_file:

        # write the title of the run
        run_title = run_dir_name
        if run_dir_name == 'garun':
            run_title = 'default'
        else:
            run_title = run_title[6:]
        parameters_file.write('RunTitle: ' + run_title +
                              '\n')
        parameters_file.write('\n')

        # write the endpoints of the composition space
        parameters_file.write('CompositionSpace: \n')
        for endpoint in composition_space.endpoints:
            parameters_file.write('    - ' +
                                  endpoint.reduced_formula.replace(' ', '') +
                                  '\n')
        parameters_file.write('\n')

        # write the name of the energy code being used
        parameters_file.write('EnergyCode: \n')
        parameters_file.write('    ' + energy_calculator.name + ': \n')
        if energy_calculator.name == 'gulp':
            parameters_file.write('        header_file: ' +
                                  energy_calculator.header_path + '\n')
            parameters_file.write('        potential_file: ' +
                                  energy_calculator.potential_path + '\n')
        elif energy_calculator.name == 'lammps':
            parameters_file.write('        input_script: ' +
                                  energy_calculator.input_script + '\n')
        elif energy_calculator.name == 'vasp':
            parameters_file.write('        num_submits_to_converge: ' +
                        str(energy_calculator.num_submits_to_converge) + '\n')
            parameters_file.write('        num_rerelax: ' +
                                  str(energy_calculator.num_rerelax) + '\n')
            parameters_file.write('        incar: ' +
                                  energy_calculator.incar_file + '\n')
            parameters_file.write('        kpoints: ' +
                                  energy_calculator.kpoints_file + '\n')
            parameters_file.write('        potcars: \n')
            for key in energy_calculator.potcar_files:
                parameters_file.write('            ' + key + ': ' +
                                      energy_calculator.potcar_files[key] +
                                      '\n')
        parameters_file.write('\n')

        # write the number of energy calculations to run at once
        parameters_file.write('NumCalcsAtOnce: ' + str(num_calcs_at_once) +
                              '\n')
        parameters_file.write('\n')

        # write the methods used to create the initial population
        parameters_file.write('InitialPopulation: \n')
        for creator in organism_creators:
            if creator.name == 'random organism creator':
                parameters_file.write('    random: \n')
                parameters_file.write('        number: ' +
                                      str(creator.number) + '\n')
                parameters_file.write('        max_num_atoms: ' +
                                      str(creator.max_num_atoms) + '\n')
                parameters_file.write('        allow_endpoints: ' +
                                      str(creator.allow_endpoints) + '\n')
                parameters_file.write('        volumes_per_atom: ' + '\n')
                for vpa in creator.vpas:
                    parameters_file.write('            ' + str(vpa) + ': ' +
                                          str(creator.vpas[vpa]) + '\n')
            elif creator.name == 'file organism creator':
                parameters_file.write('    from_files: \n')
                parameters_file.write('        number: ' +
                                      str(creator.number) + '\n')
                parameters_file.write('        path_to_folder: ' +
                                      str(creator.path_to_folder) + '\n')
        parameters_file.write('\n')

        # write the pool info
        parameters_file.write('Pool: \n')
        parameters_file.write('    size: ' + str(pool.size) + '\n')
        parameters_file.write('    num_promoted: ' + str(pool.num_promoted) +
                              '\n')
        parameters_file.write('\n')

        # write the selection probability distribution
        parameters_file.write('Selection: \n')
        parameters_file.write('    num_parents: ' +
                              str(pool.selection.num_parents) + '\n')
        parameters_file.write('    power: ' + str(pool.selection.power) + '\n')
        parameters_file.write('\n')

        # write the composition fitness weight if phase diagram search
        if composition_space.objective_function == 'pd':
            parameters_file.write('CompositionFitnessWeight: \n')
            parameters_file.write('    max_weight: ' +
                                  str(pool.comp_fitness_weight.max_weight) +
                                  '\n')
            parameters_file.write('    power: ' +
                                  str(pool.comp_fitness_weight.power) + '\n')
            parameters_file.write('\n')

        # write the variations info
        parameters_file.write('Variations: \n')
        for variation in variations:
            if variation.fraction == 0:
                pass
            else:
                if variation.name == 'mating':
                    parameters_file.write('    Mating: \n')
                    parameters_file.write('        fraction: ' +
                                          str(variation.fraction) + '\n')
                    parameters_file.write('        mu_cut_loc: ' +
                                          str(variation.mu_cut_loc) + '\n')
                    parameters_file.write('        sigma_cut_loc: ' +
                                          str(variation.sigma_cut_loc) + '\n')
                    parameters_file.write('        shift_prob: ' +
                                          str(variation.shift_prob) + '\n')
                    parameters_file.write('        rotate_prob: ' +
                                          str(variation.rotate_prob) + '\n')
                    parameters_file.write('        doubling_prob: ' +
                                          str(variation.doubling_prob) + '\n')
                    parameters_file.write('        grow_parents: ' +
                                          str(variation.grow_parents) + '\n')
                    parameters_file.write('        merge_cutoff: ' +
                                          str(variation.merge_cutoff) + '\n')
                    parameters_file.write('        halve_offspring_prob: ' +
                                    str(variation.halve_offspring_prob) + '\n')

                elif variation.name == 'structure mutation':
                    parameters_file.write('    StructureMut: \n')
                    parameters_file.write('        fraction: ' +
                                          str(variation.fraction) + '\n')
                    parameters_file.write('        frac_atoms_perturbed: ' +
                                          str(variation.frac_atoms_perturbed) +
                                          '\n')
                    parameters_file.write(
                        '        sigma_atomic_coord_perturbation: ' +
                        str(variation.sigma_atomic_coord_perturbation) + '\n')
                    parameters_file.write(
                        '        max_atomic_coord_perturbation: ' +
                        str(variation.max_atomic_coord_perturbation) + '\n')
                    parameters_file.write(
                        '        sigma_strain_matrix_element: ' +
                        str(variation.sigma_strain_matrix_element) + '\n')

                elif variation.name == 'number of atoms mutation':
                    parameters_file.write('    NumAtomsMut: \n')
                    parameters_file.write('        fraction: ' +
                                          str(variation.fraction) + '\n')
                    parameters_file.write('        mu_num_adds: ' +
                                          str(variation.mu_num_adds) + '\n')
                    parameters_file.write('        sigma_num_adds: ' +
                                          str(variation.sigma_num_adds) + '\n')
                    parameters_file.write('        scale_volume: ' +
                                          str(variation.scale_volume) + '\n')

                elif variation.name == 'permutation':
                    parameters_file.write('    Permutation: \n')
                    parameters_file.write('        fraction: ' +
                                          str(variation.fraction) + '\n')
                    parameters_file.write('        mu_num_swaps: ' +
                                          str(variation.mu_num_swaps) + '\n')
                    parameters_file.write('        sigma_num_swaps: ' +
                                          str(variation.sigma_num_swaps) +
                                          '\n')
                    parameters_file.write('        pairs_to_swap: \n')
                    for pair in variation.pairs_to_swap:
                        parameters_file.write('            - ' + pair + '\n')
        parameters_file.write('\n')

        # write the development info
        parameters_file.write('Development: \n')
        parameters_file.write('    niggli: ' + str(developer.niggli) + '\n')
        parameters_file.write('    scale_density: ' +
                              str(developer.scale_density) + '\n')
        parameters_file.write('\n')

        # write the constraints info
        parameters_file.write('Constraints: \n')
        parameters_file.write('    min_num_atoms: ' +
                              str(constraints.min_num_atoms) + '\n')
        parameters_file.write('    max_num_atoms: ' +
                              str(constraints.max_num_atoms) + '\n')
        parameters_file.write('    max_interface_atoms: ' +
                              str(constraints.max_interface_atoms) + '\n')
        parameters_file.write('    min_lattice_length: ' +
                              str(constraints.min_lattice_length) + '\n')
        parameters_file.write('    max_lattice_length: ' +
                              str(constraints.max_lattice_length) + '\n')
        parameters_file.write('    max_scell_lattice_length: ' +
                              str(constraints.max_scell_lattice_length) + '\n')
        parameters_file.write('    min_lattice_angle: ' +
                              str(constraints.min_lattice_angle) + '\n')
        parameters_file.write('    max_lattice_angle: ' +
                              str(constraints.max_lattice_angle) + '\n')
        parameters_file.write('    allow_endpoints: ' +
                              str(constraints.allow_endpoints) + '\n')
        parameters_file.write('    per_species_mids: \n')
        for pair in constraints.per_species_mids:
            parameters_file.write('        ' + pair + ': ' +
                                  str(float(
                                      constraints.per_species_mids[pair])) +
                                  '\n')
        parameters_file.write('\n')

        # write lattice matching constraints (if substrate search)
        if lat_match_dict:
            parameters_file.write('LatticeMatch: \n')
            parameters_file.write('    max_area: ' +
                                  str(lat_match_dict['max_area']) + '\n')
            parameters_file.write('    max_mismatch: ' +
                                  str(lat_match_dict['max_mismatch']) + '\n')
            parameters_file.write('    max_angle_diff: ' +
                                  str(lat_match_dict['max_angle_diff']) + '\n')
            parameters_file.write('    r1r2_tol: ' +
                                  str(lat_match_dict['r1r2_tol']) + '\n')
            parameters_file.write('    separation: ' +
                                  str(lat_match_dict['separation']) + '\n')
            parameters_file.write('    align_random: ' +
                                  str(lat_match_dict['align_random']) + '\n')
            parameters_file.write('    nlayers_substrate: ' +
                                  str(lat_match_dict['nlayers_substrate']) + '\n')
            parameters_file.write('    nlayers_2d: ' +
                                  str(lat_match_dict['nlayers_2d']) + '\n')
            parameters_file.write('\n')

            # write user-provided substrate calculation details
            parameters_file.write('Substrate: \n')
            parameters_file.write('    E_sub_prim: ' +
                                  str(lat_match_dict['E_sub_prim']) + '\n')
            parameters_file.write('    n_sub_prim: ' +
                                  str(lat_match_dict['n_sub_prim']) + '\n')
            parameters_file.write('    mu_A:' +
                                  str(lat_match_dict['mu_A']) + '\n')
            if 'mu_B' in lat_match_dict:
                parameters_file.write('    mu_B:' +
                                  str(lat_match_dict['mu_B']) + '\n')
            if 'mu_C' in lat_match_dict:
                parameters_file.write('    mu_C:' +
                                  str(lat_match_dict['mu_C']) + '\n')

            parameters_file.write('\n')

        # write the redundancy guard info
        parameters_file.write('RedundancyGuard: \n')
        parameters_file.write('    lattice_length_tol: ' +
                              str(redundancy_guard.lattice_length_tol) + '\n')
        parameters_file.write('    lattice_angle_tol: ' +
                              str(redundancy_guard.lattice_angle_tol) + '\n')
        parameters_file.write('    site_tol: ' +
                              str(redundancy_guard.site_tol) + '\n')
        parameters_file.write('    use_primitive_cell: ' +
                              str(redundancy_guard.use_primitive_cell) + '\n')
        parameters_file.write('    attempt_supercell: ' +
                              str(redundancy_guard.attempt_supercell) + '\n')
        parameters_file.write('    rmsd_tol: ' +
                              str(redundancy_guard.rmsd_tol) + '\n')
        parameters_file.write('    epa_diff: ' +
                              str(redundancy_guard.epa_diff) + '\n')
        parameters_file.write('\n')

        # write the geometry info
        parameters_file.write('Geometry: \n')
        parameters_file.write('    shape: ' + geometry.shape + '\n')
        parameters_file.write('    max_size: ' + str(geometry.max_size) + '\n')
        parameters_file.write('    min_size: ' + str(geometry.min_size) + '\n')
        parameters_file.write('    padding: ' + str(geometry.padding) + '\n')
        parameters_file.write('\n')

        # write the stopping criteria
        parameters_file.write('StoppingCriteria: \n')
        if stopping_criteria.num_energy_calcs is not None:
            parameters_file.write('    num_energy_calcs: ' +
                                  str(stopping_criteria.num_energy_calcs) +
                                  '\n')
        if stopping_criteria.epa_achieved is not None:
            parameters_file.write('    epa_achieved: ' +
                                  str(stopping_criteria.epa_achieved) + '\n')
        if stopping_criteria.found_cell is not None:
            parameters_file.write('    found_structure: ' +
                                  stopping_criteria.path_to_structure_file +
                                  '\n')
        parameters_file.write('\n')

        # write the job_specs of the dask-worker including defaults (if any)
        parameters_file.write('job_specs: \n')
        parameters_file.write('    cores: ' + str(job_specs['cores']) + '\n')
        parameters_file.write('    memory: ' + job_specs['memory'] + '\n')
        parameters_file.write('    project: ' + job_specs['project'] + '\n')
        parameters_file.write('    queue: ' + job_specs['queue'] + '\n')
        parameters_file.write('    walltime: ' + job_specs['walltime'] + '\n')
        parameters_file.write('    interface: ' + job_specs['interface'] + '\n')
        if 'job_extra' in job_specs:
            parameters_file.write('    job_extra: \n')
            for i in range(len(job_specs['job_extra'])):
                parameters_file.write('        - %r \n' % str(
                                            job_specs['job_extra'][i]))

        parameters_file.write('\n')
