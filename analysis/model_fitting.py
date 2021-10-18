import json
import logging
import pprint
import random
import numpy as np
import pandas as pd
from sklearn.manifold import smacof
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

import analysis.mds as mds
import analysis.run_mds_seed as rs
import analysis.pairwise_likelihood_analysis as an
from analysis.util import ranking_to_pairwise_comparisons, all_distance_pairs, read_in_params

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# take processed experiment responses (in json format) from the appropriate folder

SHOW_MDS = False

CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()

if __name__ == '__main__':
    # enter path to subject data (json file)
    FILEPATH = input("Path to json file containing subject's preprocessed data"
                     " (e.g., ./sample-materials/subject-data/preprocessed/S7_sample_word_exp.json: ")
    EXP = input("Experiment name (e.g., sample_word): ")
    SUBJECT = input("Subject name or ID (e.g., S7): ")
    ITERATIONS = int(input("Number of iterations - how many times this should analysis be run (e.g. 1) : "))
    OUTDIR = input("Output directory (e.g., ./sample-materials/subject-data) : ")
    SIGMA = input("Enter number or 'y' to use default (0.18):")
    if SIGMA != 'y':
        CONFIG['sigma'] = {
            'dist': 0,
            'compare': float(SIGMA)
        }
    if OUTDIR[-1] == '/':
        OUTDIR = OUTDIR[:-1]
    pprint.pprint(CONFIG)
    ok = input("Ok to proceed? (y/n)")
    if ok != 'y':
        raise InterruptedError

    for ii in range(ITERATIONS):
        # read json file into dict
        with open(FILEPATH) as file:
            ranking_responses_by_trial = json.load(file)

        # break up ranking responses into pairwise judgments
        pairwise_comparison_responses_by_trial = {}
        for config in ranking_responses_by_trial:
            comparisons = ranking_to_pairwise_comparisons(all_distance_pairs(config),
                                                          ranking_responses_by_trial[config]
                                                          )
            for key, count in comparisons.items():
                pairs = key.split('<')
                stim1, stim2 = pairs[1].split(',')
                stim3, stim4 = pairs[0].split(',')
                new_key = ((NAMES_TO_ID[stim1], NAMES_TO_ID[stim2]), (NAMES_TO_ID[stim3], NAMES_TO_ID[stim4]))
                if new_key not in pairwise_comparison_responses_by_trial:
                    pairwise_comparison_responses_by_trial[new_key] = count
                else:
                    # if the comparison is repeated in two trials (context design side-effect)
                    pairwise_comparison_responses_by_trial[new_key] += count
                    pairwise_comparison_responses_by_trial[new_key] = pairwise_comparison_responses_by_trial[
                                                                          new_key] / 2.0

        # get MDS starting coordinates
        D = mds.format_distances(mds.heuristic_distances(pairwise_comparison_responses_by_trial, 5))
        coordinates2d, stress = smacof(D, n_components=2, metric=True, eps=1e-9)
        if SHOW_MDS:
            plt.plot(coordinates2d[:, 0], coordinates2d[:, 1], '.')
            for i, txt in enumerate(range(37)):
                plt.annotate(ID_TO_NAME[txt], (coordinates2d[i, 0], coordinates2d[i, 1]))
            plt.show()

        # only consider a subset of trials
        if CONFIG['max_trials'] < len(pairwise_comparison_responses_by_trial):
            indices = random.sample(pairwise_comparison_responses_by_trial.keys(), CONFIG['max_trials'])
            subset = {key: pairwise_comparison_responses_by_trial[key] for key in indices}
        else:
            subset = pairwise_comparison_responses_by_trial

        # initialize results dataframe
        result = {'Model': [], 'Log Likelihood': [], 'number of points': [],
                  'Experiment': [EXP] * (2 + len(CONFIG['model_dimensions'])),
                  'Subject': [SUBJECT] * (2 + len(CONFIG['model_dimensions']))}
        num_trials = len(subset)
        for dim in CONFIG['model_dimensions']:
            LOG.info('#######  {} dimensional model'.format(dim))
            model_name = str(dim) + 'D'
            CONFIG['n_dim'] = dim
            x, ll_nd, fmin_costs = rs.points_of_best_fit(subset, CONFIG)
            LOG.info("Points: ")
            print(x)
            outfilename = '{}/{}_{}_anchored_points_sigma_{}_dim_{}'.format(
                OUTDIR,
                SUBJECT, EXP,
                str(CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']),
                dim
            )
            np.save(outfilename, x)

            LOG.info("Distances: ")
            distances = pdist(x)
            ll_nd = -ll_nd / float(num_trials * CONFIG['num_repeats'])
            result['Model'].append(model_name)
            result['Log Likelihood'].append(ll_nd)
            result['number of points'].append(CONFIG['num_stimuli'])
        # the ii for loop can be taken out later. just need it for a plot
        #   plt.plot(fmin_costs)
        # plt.show()

        LOG.info('#######  Random and best model')
        ll_best = an.best_model_ll(
            subset, CONFIG)[0] / float(num_trials * CONFIG['num_repeats'])
        result['Model'].append('best')
        result['Log Likelihood'].append(ll_best)
        result['number of points'].append(CONFIG['num_stimuli'])
        ll_random = an.random_choice_ll(
            subset, CONFIG)[0] / float(num_trials * CONFIG['num_repeats'])
        result['Model'].append('random')
        result['Log Likelihood'].append(ll_random)
        result['number of points'].append(CONFIG['num_stimuli'])
        data_frame = pd.DataFrame(result)
        sigma = CONFIG['sigma']['compare'] + CONFIG['sigma']['dist']
        data_frame.to_csv('{}/{}-{}-model-likelihoods_with_{}_trials_sigma_{}_{}pts_anchored_{}.csv'
                          .format(OUTDIR,
                                  SUBJECT,
                                  EXP,
                                  CONFIG['max_trials'],
                                  sigma,
                                  CONFIG['num_stimuli'],
                                  ii))
