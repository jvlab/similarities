"""

"""
import random
import time
import logging
import argparse
import numpy as np
import pandas as pd
from analysis.util import add_row
import analysis.model_fitting.run_mds_seed as rs
from analysis.simulation.run_v2 import n_dim_minimization
import analysis.simulation.experiment_ranking as exp_rank
import analysis.model_fitting.pairwise_likelihood_analysis as an
from analysis.geometry.ultrametric import make_tree, get_leaves, get_path, cross

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


BRANCHING_FACTOR = 4
HEIGHT = 3

parser = argparse.ArgumentParser()
parser.add_argument("-it", "--iterations", type=int, default=10)
parser.add_argument("-n", "--num_stimuli", type=int, default=37)
parser.add_argument("-d1", "--min_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=2)
parser.add_argument("-d2", "--max_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=2)
parser.add_argument("-r", "--repetitions", help="Number of times to repeat a trial", type=int, default=5)
parser.add_argument("-s", "--noise", help="Sigma compare", type=int, default=0.5)

args = parser.parse_args()
PARAMS = {
    'num_stimuli': args.num_stimuli,
    'n_iter': args.iterations,
    'd1': args.min_model_dim,
    'd2': args.max_model_dim,
    'num_repeats': args.repetitions,
    'sigmas': {'compare': args.noise, 'dist': 0},
    'curvatures': [0.5, 1, 2, 3, 4],
    'max_trials': 10000,
    'epsilon': float(1e-30),
    'minimization': 'gradient-descent',
    'tolerance': float(1e-6),
    'fatol': float(1e-6),
    'max_iterations': 5000,
    'learning_rate': float(0.01),
    'no_noise': False,
    'verbose': False
}


def main():
    # make a tree and sample leaf nodes
    root = make_tree(BRANCHING_FACTOR, HEIGHT)
    leaves = get_leaves(root)
    print([leaf.val for leaf in leaves])
    sample = np.random.choice(leaves, PARAMS['num_stimuli'], False)
    # obtain ultrametric distances between sampled leaf nodes
    distance_matrix = np.zeros((PARAMS['num_stimuli'], PARAMS['num_stimuli']))
    for i in range(len(sample)):
        for j in range(i):
            path_ri = get_path(sample[i], root)
            path_rj = get_path(sample[j], root)
            distance_matrix[i, j] = cross(path_ri, path_rj)
    # simulate judgments using these distances
    points = list(range(PARAMS['num_stimuli']))
    data = exp_rank.run_experiment(points, distance_matrix, PARAMS, simple_err_model=True)
    judgments = {}
    for trial, responses in data.items():
        pairwise_comparisons = exp_rank.ranking_to_pairwise_comparisons(exp_rank.all_distance_pairs(trial), responses)
        for pair, judgment in pairwise_comparisons.items():
            if pair not in judgments:
                judgments[pair] = judgment
            else:
                judgments[pair] += judgment
                judgments[pair] = judgments[pair] / float(2)

    # sample judgments if they exceed max_trials
    if PARAMS['max_trials'] is not None and len(judgments) > PARAMS['max_trials']:
        judgments = dict(random.sample(judgments.items(), PARAMS['max_trials']))
    num_trials = len(judgments)

    result = {'Model': [], 'Curvature': [], 'LL': [], 'Relative LL': [], 'Num Points': [], 'Branching Factor': [],
              'Tree Height': [], 'True Model': []}
    # calculate LL for random and best models
    true_model = 'ultrametric'
    # RANDOM AND BEST MODELS ####################################################################
    ll_best = an.best_model_ll(
        judgments, PARAMS)[0] / float(num_trials * PARAMS['num_repeats'])
    new_row = {'Model': 'best', 'LL': ll_best, 'Relative LL': 0,
               'Num Points': PARAMS['num_stimuli'], 'True Model': true_model,
               'Branching Factor': BRANCHING_FACTOR, 'Tree Height': HEIGHT, 'Curvature': None}
    result = add_row(new_row, result)
    ll_random = an.random_choice_ll(
        judgments, PARAMS)[0] / float(num_trials * PARAMS['num_repeats'])
    new_row = {'Model': 'random', 'LL': ll_random, 'Relative LL': ll_random - ll_best,
               'Num Points': PARAMS['num_stimuli'], 'True Model': true_model,
               'Branching Factor': BRANCHING_FACTOR, 'Tree Height': HEIGHT, 'Curvature': None}
    result = add_row(new_row, result)

    # MODELING WITH DIFFERENT EUCLIDEAN MODELS ###################################################
    # run modeling with 2D Euclidean space
    for dim in range(PARAMS['d1'], PARAMS['d2']+1):
        new_row = n_dim_minimization(true_model, dim, judgments, PARAMS, ll_best)
        new_row['Branching Factor'] = BRANCHING_FACTOR
        new_row['Tree Height'] = HEIGHT
        new_row['Curvature'] = 0
        result = add_row(new_row, result)

    # run modeling with hyperbolic space
    # MODELING WITH HYPERBOLIC MODEL OF CORRECT DIMENSION ########################################
    # also test correct hyperbolic model if true_geometry is from an ultrametric space
    hyp_dim = 2
    for c in PARAMS['curvatures']:
        PARAMS['curvature'] = c
        LOG.info('Fitting hyperbolic model with PARAMS: ')
        print('Curvature" ', c)
        LOG.info('################  {} dimensional hyperbolic geometry'.format(hyp_dim))
        model_name = str(hyp_dim) + 'D-hyp'
        PARAMS['n_dim'] = hyp_dim
        x, ll_nd, fmin_costs = rs.hyperbolic_points_of_best_fit(judgments, PARAMS)
        ll_nd = -ll_nd / float(num_trials * PARAMS['num_repeats'])
        new_row = {'Model': model_name, 'LL': ll_nd, 'Relative LL': ll_nd - ll_best,
                   'Num Points': PARAMS['num_stimuli'], 'True Model': true_model,
                   'Branching Factor': BRANCHING_FACTOR, 'Tree Height': HEIGHT, 'Curvature': c}
        result = add_row(new_row, result)

    # is there a difference in LL?
    # OUTPUT RESULTS ###############################################################################
    data_frame = pd.DataFrame(result)
    timestamp = time.asctime().replace(" ", '.')
    data_frame.to_csv(
        '../simulation_{}_{}-likelihoods_{}_trials_{}_stimuli_sampled-from_{}_{}_repeats_{}.csv'.format(
            'hyperbolic',
            'ranking',
            num_trials,
            PARAMS['num_stimuli'],
            'ultrametric',
            str(PARAMS['num_repeats']),
            timestamp))


for _ in range(PARAMS['n_iter']):
    main()
