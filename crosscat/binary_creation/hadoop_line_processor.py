#!/opt/anaconda/bin/python

#
#   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
#
#   Lead Developers: Dan Lovell and Jay Baxter
#   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka, Avinash Gandhe
#   Research Leads: Vikash Mansinghka, Patrick Shafto
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
import os
import numpy
import sys
import itertools
import random
#
import crosscat.utils.data_utils as du
import crosscat.utils.file_utils as fu
import crosscat.utils.hadoop_utils as hu
import crosscat.utils.xnet_utils as xu
import crosscat.utils.general_utils as gu
import crosscat.utils.inference_utils as iu
import crosscat.utils.timing_test_utils as ttu
import crosscat.utils.mutual_information_test_utils as mitu
import crosscat.utils.convergence_test_utils as ctu
import crosscat.utils.LocalEngine as LE
import crosscat.utils.HadoopEngine as HE
from crosscat.settings import Hadoop as hs


def initialize_helper(table_data, data_dict, command_dict):
    M_c = table_data['M_c']
    M_r = table_data['M_r']
    T = table_data['T']
    SEED = data_dict['SEED']
    initialization = command_dict['initialization']
    engine = LE.LocalEngine(SEED)
    X_L, X_D = engine.initialize(M_c, M_r, T, initialization=initialization)
    SEED = engine.get_next_seed()
    #
    ret_dict = dict(SEED=SEED, X_L=X_L, X_D=X_D)
    return ret_dict

def analyze_helper(table_data, data_dict, command_dict):
    M_c = table_data['M_c']
    T = table_data['T']
    SEED = data_dict['SEED']
    X_L = data_dict['X_L']
    X_D = data_dict['X_D']
    kernel_list = command_dict['kernel_list']
    n_steps = command_dict['n_steps']
    c = command_dict['c']
    r = command_dict['r']
    max_time = command_dict['max_time']
    engine = LE.LocalEngine(SEED)
    X_L_prime, X_D_prime = engine.analyze(M_c, T, X_L, X_D, kernel_list=kernel_list,
                                          n_steps=n_steps, c=c, r=r,
                                          max_time=max_time)
    SEED = engine.get_next_seed()
    #
    ret_dict = dict(SEED=SEED, X_L=X_L_prime, X_D=X_D_prime)
    return ret_dict

def chunk_analyze_helper(table_data, data_dict, command_dict):
    original_n_steps = command_dict['n_steps']
    original_SEED = data_dict['SEED']
    chunk_size = command_dict['chunk_size']
    chunk_filename_prefix = command_dict['chunk_filename_prefix']
    chunk_dest_dir = command_dict['chunk_dest_dir']
    #
    steps_done = 0
    while steps_done < original_n_steps:
        steps_remaining = original_n_steps - steps_done
        command_dict['n_steps'] = min(chunk_size, steps_remaining)
        ith_chunk = steps_done / chunk_size
        dict_out = analyze_helper(table_data, data_dict, command_dict)
        data_dict.update(dict_out)
        # write to hdfs
        chunk_filename = '%s_seed_%s_chunk_%s.pkl.gz' \
            % (chunk_filename_prefix, original_SEED, ith_chunk)
        fu.pickle(dict_out, chunk_filename)
        hu.put_hdfs(None, chunk_filename, chunk_dest_dir)
        #
        steps_done += chunk_size
    chunk_filename = '%s_seed_%s_chunk_%s.pkl.gz' \
        % (chunk_filename_prefix, original_SEED, 'FINAL')
    fu.pickle(dict_out, chunk_filename)
    hu.put_hdfs(None, chunk_filename, chunk_dest_dir)
    return dict_out
    
def time_analyze_helper(table_data, data_dict, command_dict):
    # FIXME: this is a kludge
    command_dict.update(data_dict)
    #
    gen_seed = data_dict['SEED']
    num_clusters = data_dict['num_clusters']
    num_cols = data_dict['num_cols']
    num_rows = data_dict['num_rows']
    num_views = data_dict['num_views']

    T, M_c, M_r, X_L, X_D = ttu.generate_clean_state(gen_seed,
                                                 num_clusters,
                                                 num_cols, num_rows,
                                                 num_views,
                                                 max_mean=10, max_std=1)
    table_data = dict(T=T,M_c=M_c)

    data_dict['X_L'] = X_L
    data_dict['X_D'] = X_D
    start_dims = du.get_state_shape(X_L)
    with gu.Timer('time_analyze_helper', verbose=False) as timer:
        inner_ret_dict = analyze_helper(table_data, data_dict, command_dict)
    end_dims = du.get_state_shape(inner_ret_dict['X_L'])
    T = table_data['T']
    table_shape = (len(T), len(T[0]))
    ret_dict = dict(
        table_shape=table_shape,
        start_dims=start_dims,
        end_dims=end_dims,
        elapsed_secs=timer.elapsed_secs,
        kernel_list=command_dict['kernel_list'],
        n_steps=command_dict['n_steps'],
        )
    return ret_dict

def mi_analyze_helper(table_data, data_dict, command_dict):

    gen_seed = data_dict['SEED']
    crosscat_seed = data_dict['CCSEED']
    num_clusters = data_dict['num_clusters']
    num_cols = data_dict['num_cols']
    num_rows = data_dict['num_rows']
    num_views = data_dict['num_views']
    corr = data_dict['corr']
    burn_in = data_dict['burn_in']
    mean_range = float(num_clusters)*2.0

    # 32 bit signed int
    random.seed(gen_seed)
    get_next_seed = lambda : random.randrange(2147483647)

    # generate the stats
    T, M_c, M_r, X_L, X_D, view_assignment = mitu.generate_correlated_state(num_rows,
        num_cols, num_views, num_clusters, mean_range, corr, seed=gen_seed);

    table_data = dict(T=T,M_c=M_c)

    engine = LE.LocalEngine(crosscat_seed)
    X_L_prime, X_D_prime = engine.analyze(M_c, T, X_L, X_D, n_steps=burn_in) 

    X_L = X_L_prime
    X_D = X_D_prime

    view_assignment = numpy.array(X_L['column_partition']['assignments'])
 
    # for each view calclate the average MI between all pairs of columns
    n_views = max(view_assignment)+1
    MI = []
    Linfoot = []
    queries = []
    MI = 0.0
    pairs = 0.0
    for view in range(n_views):
        columns_in_view = numpy.nonzero(view_assignment==view)[0]
        combinations = itertools.combinations(columns_in_view,2)
        for pair in combinations:
            any_pairs = True
            queries.append(pair)
            MI_i, Linfoot_i = iu.mutual_information(M_c, [X_L], [X_D], [pair], n_samples=1000)
            MI += MI_i[0][0]
            pairs += 1.0

    
    if pairs > 0.0:
        MI /= pairs

    ret_dict = dict(
        id=data_dict['id'],
        dataset=data_dict['dataset'],
        sample=data_dict['sample'],
        mi=MI,
        )

    return ret_dict

def convergence_analyze_helper(table_data, data_dict, command_dict):
    gen_seed = data_dict['SEED']
    num_clusters = data_dict['num_clusters']
    num_cols = data_dict['num_cols']
    num_rows = data_dict['num_rows']
    num_views = data_dict['num_views']
    max_mean = data_dict['max_mean']
    n_test = data_dict['n_test']
    num_transitions = data_dict['n_steps']
    block_size = data_dict['block_size']
    init_seed = data_dict['init_seed']


    # generate some data
    T, M_r, M_c, data_inverse_permutation_indices = \
            du.gen_factorial_data_objects(gen_seed, num_clusters,
                    num_cols, num_rows, num_views,
                    max_mean=max_mean, max_std=1,
                    send_data_inverse_permutation_indices=True)
    view_assignment_ground_truth = \
            ctu.determine_synthetic_column_ground_truth_assignments(num_cols,
                    num_views)
    X_L_gen, X_D_gen = ttu.get_generative_clustering(M_c, M_r, T,
            data_inverse_permutation_indices, num_clusters, num_views)
    T_test = ctu.create_test_set(M_c, T, X_L_gen, X_D_gen, n_test, seed_seed=0)
    generative_mean_test_log_likelihood = \
            ctu.calc_mean_test_log_likelihood(M_c, T, X_L_gen, X_D_gen, T_test)

    # additional set up
    engine=LE.LocalEngine(init_seed)
    column_ari_list = []
    mean_test_ll_list = []
    elapsed_seconds_list = []

    # get initial ARI, test_ll
    with gu.Timer('initialize', verbose=False) as timer:
        X_L, X_D = engine.initialize(M_c, M_r, T, initialization='from_the_prior')
    column_ari = ctu.get_column_ARI(X_L, view_assignment_ground_truth)
    column_ari_list.append(column_ari)
    mean_test_ll = ctu.calc_mean_test_log_likelihood(M_c, T, X_L, X_D,
            T_test)
    mean_test_ll_list.append(mean_test_ll)
    elapsed_seconds_list.append(timer.elapsed_secs)

    # run blocks of transitions, recording ARI, test_ll progression
    completed_transitions = 0
    n_steps = min(block_size, num_transitions)
    while (completed_transitions < num_transitions):
        # We won't be limiting by time in the convergence runs
        with gu.Timer('initialize', verbose=False) as timer:
             X_L, X_D = engine.analyze(M_c, T, X_L, X_D, kernel_list=(),
                     n_steps=n_steps, max_time=-1)
        completed_transitions = completed_transitions + block_size
        #
        column_ari = ctu.get_column_ARI(X_L, view_assignment_ground_truth)
        column_ari_list.append(column_ari)
        mean_test_ll = ctu.calc_mean_test_log_likelihood(M_c, T, X_L, X_D,
                T_test)
        mean_test_ll_list.append(mean_test_ll)
        elapsed_seconds_list.append(timer.elapsed_secs)

    ret_dict = dict(
        num_rows=num_rows,
        num_cols=num_cols,
        num_views=num_views,
        num_clusters=num_clusters,
        max_mean=max_mean,
        column_ari_list=column_ari_list,
        mean_test_ll_list=mean_test_ll_list,
        generative_mean_test_log_likelihood=generative_mean_test_log_likelihood,
        elapsed_seconds_list=elapsed_seconds_list,
        n_steps=num_transitions,
        block_size=block_size,
        )
    return ret_dict
    

method_lookup = dict(
    initialize=initialize_helper,
    analyze=analyze_helper,
    time_analyze=time_analyze_helper,
    convergence_analyze=convergence_analyze_helper,
    chunk_analyze=chunk_analyze_helper,
    mi_analyze=mi_analyze_helper
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--table_data_filename', type=str,
                        default=hs.default_table_data_filename)
    parser.add_argument('--command_dict_filename', type=str,
                        default=hs.default_command_dict_filename)
    args = parser.parse_args()
    table_data_filename = args.table_data_filename
    command_dict_filename = args.command_dict_filename
    
    
    table_data = fu.unpickle(table_data_filename)
    command_dict = fu.unpickle(command_dict_filename)
    command = command_dict['command']
    method = method_lookup[command]
    #
    from signal import signal, SIGPIPE, SIG_DFL 
    signal(SIGPIPE,SIG_DFL) 
    for line in sys.stdin:
        key, data_dict = xu.parse_hadoop_line(line)
        ret_dict = method(table_data, data_dict, command_dict)
        xu.write_hadoop_line(sys.stdout, key, ret_dict)
