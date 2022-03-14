import argparse
import logging
import pprint
import random
import time
import yaml
from multiprocessing import Pool

import numpy.random as np_random
import pandas as pd
import analysis.geometry.euclidean as model
import analysis.simulation.experiment as exp
import analysis.simulation.experiment_ranking as exp_ranking
import analysis.model_fitting.pairwise_likelihood_analysis as an
import analysis.model_fitting.run_mds_seed as rs
from analysis.util import add_row, judgments_to_arrays

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
global args


def simulate_judgments(parameters, points):
    if parameters['paradigm'] == 'pairwise_comparisons':
        judgments, points = exp.run_experiment(points, parameters)
    elif parameters['paradigm'] == 'simple_ranking':
        data = exp_ranking.run_experiment(points, parameters, simple_err_model=True)
        judgments = {}
        for trial, responses in data.items():
            pairwise_comparisons = exp_ranking.ranking_to_pairwise_comparisons(exp_ranking.all_distance_pairs(trial),
                                                                               responses)
            for pair, judgment in pairwise_comparisons.items():
                if pair not in judgments:
                    judgments[pair] = judgment
                else:
                    judgments[pair] += judgment
                    judgments[pair] = judgments[pair] / float(2)
    else:
        raise NotImplementedError('Invalid paradigm')
    return judgments


def main(true_dim, parameters, min_model=1, max_model=5, show=False):
    LOG.info('################  True Dimension: {}'.format(true_dim))
    space = model.EuclideanSpace(true_dim)
    true_model = str(true_dim) + 'D'
    points = space.get_samples(parameters['num_stimuli'], 2, parameters['sampling_method'])
    # simulate trialwise judgments based on input paradigm etc.
    pairwise_judgments = simulate_judgments(parameters, points)

    if parameters['max_trials'] is not None and len(pairwise_judgments) > parameters['max_trials']:
        pairwise_judgments = dict(random.sample(pairwise_judgments.items(), parameters['max_trials']))
    trial_pairs_a, trial_pairs_b, judgment_counts = judgments_to_arrays(pairwise_judgments)
    LOG.info('################  Number of binary comparisons: {}'.format(len(pairwise_judgments)))

    # initialize results dataframe
    result = {'Model': [], 'LL': [], 'Relative LL': [], 'Num Points': [], 'True Model': []}
    num_trials = len(pairwise_judgments)

    LOG.info('################  Ground truth geometry')
    parameters['n_dim'] = true_dim
    ll_groundtruth = an.dist_model_ll_vectorized(
        trial_pairs_a, trial_pairs_b, judgment_counts, parameters, points
    )[0] / float(num_trials * parameters['num_repeats'])
    new_row = {'Model': 'ground truth', 'LL': ll_groundtruth, 'Relative LL': 0,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)
    LOG.info('################  Random and best geometry')
    ll_best = an.best_model_ll(
        pairwise_judgments, parameters)[0] / float(num_trials * parameters['num_repeats'])
    new_row = {'Model': 'best', 'LL': ll_best, 'Relative LL': ll_best - ll_groundtruth,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)
    ll_random = an.random_choice_ll(
        pairwise_judgments, parameters)[0] / float(num_trials * parameters['num_repeats'])
    new_row = {'Model': 'random', 'LL': ll_random, 'Relative LL': ll_random - ll_groundtruth,
               'Num Points': parameters['num_stimuli'], 'True Model': true_model}
    result = add_row(new_row, result)

    #  use a different noise param for fitting than for simulating
    parameters['sigmas'] = {'compare': 0.02+parameters['sigmas']['compare'], 'dist': 0}
    for dim in range(min_model, max_model + 1):
        LOG.info('################  {} dimensional geometry'.format(dim))
        model_name = str(dim) + 'D'
        parameters['n_dim'] = dim
        x, ll_nd, fmin_costs = rs.points_of_best_fit(pairwise_judgments, parameters)
        ll_nd = -ll_nd / float(num_trials * parameters['num_repeats'])
        new_row = {'Model': model_name, 'LL': ll_nd, 'Relative LL': ll_nd - ll_groundtruth,
                   'Num Points': parameters['num_stimuli'], 'True Model': true_model}
        result = add_row(new_row, result)
    data_frame = pd.DataFrame(result)
    timestamp = time.asctime().replace(" ", '.')
    data_frame.to_csv(
        './{}-likelihoods_{}_trials_{}_stimuli_sampled-from_{}_{}repeats_{}.csv'.format('simulation_{}'
                                                                                        .format(parameters['paradigm']),
                                                                                        num_trials,
                                                                                        parameters['num_stimuli'],
                                                                                        parameters['sampling_method'],
                                                                                        parameters['num_repeats'],
                                                                                        timestamp))


# helper
def helper(ii):
    LOG.info('################  Iteration: {}'.format(ii))
    start = time.perf_counter()
    for n in range(args.min_true_dim, args.max_true_dim + 1):
        params = {
            'num_stimuli': args.num_stimuli,
            'max_trials': CONFIG['max_trials'],
            'n_dim': n,
            'no_noise': False,
            'sampling_method': args.sampling,
            'num_repeats': args.repetitions,
            'sigmas': {'compare': CONFIG['sigmas'], 'dist': 0},
            'paradigm': args.paradigm,
            'dist_metric': 'euclidean',
            'epsilon': float(CONFIG['epsilon']),
            'fatol': float(CONFIG['fatol']),
            'tolerance': float(CONFIG['tolerance']),
            'max_iterations': int(CONFIG['max_iterations']),
            'learning_rate': float(CONFIG['learning_rate'])
        }
        pprint.pprint(params)
        np_random.seed()
        main(n, params, args.min_model_dim, args.max_model_dim)
    end = time.perf_counter()
    print('Time taken: {} minutes'.format((end - start) / 60.0))


if __name__ == '__main__':
    # define valid args
    with open('./analysis/config.yaml', "r") as stream:
        CONFIG = yaml.safe_load(stream)
    print(CONFIG)
    # ask for and parse user input
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--iterations", type=int, default=2)
    parser.add_argument("-n", "--num_stimuli", type=int, default=37)
    parser.add_argument("-d1", "--min_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=1)
    parser.add_argument("-d2", "--max_model_dim", choices=[1, 2, 3, 4, 5], type=int, default=5)
    parser.add_argument("-t1", "--min_true_dim", choices=[2, 3, 4, 5], type=int, default=2)
    parser.add_argument("-t2", "--max_true_dim", choices=[2, 3, 4, 5], type=int, default=5)
    parser.add_argument("-r", "--repetitions", help="Number of times to repeat a trial", type=int, default=5)
    parser.add_argument("-s", "--sampling", choices=["uniform", "spherical_shell", "gaussian"],
                        default="gaussian"),
    parser.add_argument("-p", "--paradigm", choices=["pairwise_comparisons", "simple_ranking"],
                        default="simple_ranking")
    args = parser.parse_args()

    LOG.info("Running simulation with arguments: {}".format(args))

    pool = Pool(2)
    iterations = range(args.iterations)
    pool.map(helper, iterations)
    pool.close()
    pool.join()
