import json
import logging
import pprint
import random
import numpy as np
import pandas as pd
from sklearn.manifold import smacof
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

import src.mds as mds
import src.run_mds_seed as rs
import src.pairwise_likelihood_analysis as an
from src.util import stimulus_names, stimulus_id_to_name, stimulus_name_to_id, ranking_to_pairwise_comparisons,\
    all_distance_pairs

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# take processed experiment responses (in json format) from the appropriate folder

SHOW_MDS = False
DIMENSIONS = [1, 2, 3, 4, 5]


def default_params():
    return ({
                'n_dim': None,
                'num_stimuli': 37,
                'sigmas': {'point': 0,
                           'dist': 0,
                           'compare': 0.18  # was  0.18
                           },
                'num_repeats': 5,
                'no_noise': False,
                'max_trials': 6000,
                'verbose': False
            },
            stimulus_names(),
            stimulus_name_to_id(),
            stimulus_id_to_name())


PARAMS, STIMULI, NAMES_TO_ID, ID_TO_NAME = default_params()

if __name__ == '__main__':
    # set experiment type
    EXP = input("Experiment?")
    # set directory
    DIR = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/experiments/{}_exp/subject-data/' \
          'preprocessed'.format(EXP)
    NUM_STIMULI = int(input("Number of stimuli to randomly choose and use:"))
    PARAMS['num_stimuli'] = NUM_STIMULI
    ITERATIONS = int(input("Number of iterations:"))
    OUTDIR = input("Output directory:")
    SIGMA = input("Enter number or 'y' to use default (0.18):")
    if SIGMA == 'y':
        PARAMS['sigmas'] = {
            'point': 0,
            'dist': 0,
            'compare': 0.18  # was  0.18
        }
    else:
        PARAMS['sigmas'] = {
            'point': 0,
            'dist': 0,
            'compare': float(SIGMA)  # was  0.18
        }
    if OUTDIR[-1] == '/':
        OUTDIR = OUTDIR[:-1]
    # set subject initials
    subs = input("Subjects separated by spaces:")
    SUBJECTS = subs.split(' ')
    print(SUBJECTS)
    pprint.pprint(PARAMS)
    ok = input("Ok to proceed? (y/n)")
    if ok != 'y':
        raise InterruptedError

    for ii in range(ITERATIONS):
        for SUBJECT in SUBJECTS:
            FILEPATH = '{}/{}_{}_exp.json'.format(DIR, SUBJECT, EXP)
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
            if PARAMS['max_trials'] < len(pairwise_comparison_responses_by_trial):
                indices = random.sample(pairwise_comparison_responses_by_trial.keys(), PARAMS['max_trials'])
                subset = {key: pairwise_comparison_responses_by_trial[key] for key in indices}
            else:
                subset = pairwise_comparison_responses_by_trial

            # initialize results dataframe
            result = {'Model': [], 'Log Likelihood': [], 'number of points': [],
                      'Experiment': [EXP] * (2 + len(DIMENSIONS)),
                      'Subject': [SUBJECT] * (2 + len(DIMENSIONS))}
            num_trials = len(subset)
            for dim in DIMENSIONS:
                LOG.info('#######  {} dimensional model'.format(dim))
                model_name = str(dim) + 'D'
                PARAMS['n_dim'] = dim
                x, ll_nd, fmin_costs = rs.points_of_best_fit(subset, PARAMS)
                LOG.info("Points: ")
                print(x)
                outfilename = '{}_{}_anchored_points_sigma_{}_dim_{}'.format(
                    SUBJECT, EXP,
                    str(PARAMS['sigmas']['compare'] + PARAMS['sigmas']['dist']),
                    dim
                )
                np.save(outfilename, x)

                LOG.info("Distances: ")
                distances = pdist(x)
                print(distances)
                print(sum(distances) / len(distances))
                ll_nd = -ll_nd / float(num_trials * PARAMS['num_repeats'])
                result['Model'].append(model_name)
                result['Log Likelihood'].append(ll_nd)
                result['number of points'].append(PARAMS['num_stimuli'])
            # the ii for loop can be taken out later. just need it for a plot
            #   plt.plot(fmin_costs)
            # plt.show()

            LOG.info('#######  Random and best model')
            ll_best = an.best_model_ll(
                subset, PARAMS)[0] / float(num_trials * PARAMS['num_repeats'])
            result['Model'].append('best')
            result['Log Likelihood'].append(ll_best)
            result['number of points'].append(PARAMS['num_stimuli'])
            ll_random = an.random_choice_ll(
                subset, PARAMS)[0] / float(num_trials * PARAMS['num_repeats'])
            result['Model'].append('random')
            result['Log Likelihood'].append(ll_random)
            result['number of points'].append(PARAMS['num_stimuli'])
            data_frame = pd.DataFrame(result)
            sigma = PARAMS['sigmas']['compare'] + PARAMS['sigmas']['dist']
            data_frame.to_csv('{}/{}-{}-model-likelihoods_with_{}_trials_sigma_{}_{}pts_anchored_{}.csv'.format(OUTDIR,
                                                                                                                SUBJECT,
                                                                                                                EXP,
                                                                                                                PARAMS[
                                                                                                                    'max_trials'],
                                                                                                                sigma,
                                                                                                                PARAMS[
                                                                                                                    'num_stimuli'],
                                                                                                                ii))
