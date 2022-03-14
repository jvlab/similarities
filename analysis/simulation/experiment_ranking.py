"""
This file makes use of geometry to use the noisy data drawn from a Euclidean space to simulate pairwise similarity
judgments using the ranking paradigm. The ranking experiment using 37 stimuli is used to generate simulated responses
on a trial by trial basis. In each trial, the task is to rank order the 8 stimuli that appear around a central reference
in order of similarity to it, choosing the most similar stimulus first. If (with noise added) the distance between
stimuli R and A is less than the distance between R and B, our simulated subject will judge A to be more similar to R
than B is to R.

In each trial, we calculate 8 distances between stimulus i and R (i=1,...,8). Then we add Gaussian noise twice
to account for noise in distance estimation. Then we add some noise again and find the minimum distance. The second
addition of noise accounts for error in comparing distances. We choose the minimum - this is the first clicked
(most similar) stimulus. We repeat the process (adding noise each time) on the remaining stimuli till all have been
'clicked.' We record the order of clicking - the simulated response of the trial. The idea is that after each click, the
context may have changed and the distances may be compared to each other again. This new comparison might incur noise.

When we use the simulated ranking judgments for geometry-fitting we will do some preprocessing as was done in the analysis
of real data (pilot data, word experiment): decomposing each trial into 28 pairwise comparisons. Before calling the
minimization method that optimizes coordinates for different Euclidean models, I will restructure the simulated data to
conform to the pairs of pairs format as the geometry-fitting procedure only requires pairwise decisions and pairwise choice
probabilities and is unaffected by the underlying experimental paradigm used to get said pairwise decisions.
"""

import logging
import math

from itertools import combinations

import numpy as np

import analysis.simulation.experiment_simple_ranking as simple
from analysis.simulation.experiment import create_trials

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def simulate_rank_judgments(trials, interstimulus_distance, num_repeats, sigmas):
    """
    Simulate ranking trials and click stimuli in order of similarity to reference using calculated interpoint distances
    :param sigmas: default in DEFAULT
    :param num_repeats: default in DEFAULT
    :param trials: List of tuples containing trial configuration e.g.
                    (ref, (stim1, stim2, stim3, ..., stim8))]
    @param interstimulus_distance: n by n distance matrix where n is the number of stimuli
    :return:responses (Dict) key=trial, value=clicked_list
    """
    click_responses = {}
    for trial in trials:
        repeats = []
        ref, circle = trial[0], trial[1]
        num_clicks = len(trial[1])
        for _ in range(num_repeats):
            clicked = {}
            # calculate trial distances, assuming some noise in calculation
            # next, add more noise at the level of comparison and 'click' the stimulus with the shortest distance to ref
            distances = [
                interstimulus_distance[ref, _i] + np.random.normal(0, sigmas['dist']) for _i in circle
            ]
            # click stimuli in order of similarity, implicitly comparing similarities each time
            for k in range(num_clicks):
                distances = [d + np.random.normal(0, sigmas['compare']) for d in distances]  # comparison noise
                local_index = distances.index(min(distances))
                # record index of/ when each stimulus was clicked, e.g. 'stim1: 2' means stim1 was picked third (0,1,2)
                clicked['s{}'.format(circle[local_index])] = k
                distances[local_index] = math.inf
            repeats.append(clicked)
        click_responses[trial] = repeats
    return click_responses


def run_experiment(stimuli, distances, args, simple_err_model=False, trials=None):
    """
    Simulate an experiment where subjects are asked to rank stimuli in order of similarity to a changing
    central reference stimulus. The total number of trials is 222 and number of unique stimuli is 37. They make their
    decision based on the comparison of distances between pairs of points (subject to internal noise in comparing and
    computing distance). Noise level is controlled by args.
    :param stimuli: 37 points with coordinates provided
    :param simple_err_model: False if the error geometry involves noise before each click
    :param args dict, as in default (see DEFAULT)
    :param trials: if None, create configuration of full 222 using 37 stimuli, else used passed in list of trials
    @param distances: distances between stimuli
    """

    if trials is None:
        # prepare trial configuration of 222 trials using 37 stimuli as in real experiment
        trials = create_trials(stimuli, paradigm="ranking")

    # iterating over trials, with repetition, get rank judgments
    # the process is simpler for 1d judgments as no Euclidean distance needs to be calculated between stimuli and R,
    # rather just a difference between numbers needs to be calculated.
    if simple_err_model:
        judgments = simple.simulate_simple_rank_judgments(trials, distances, args['num_repeats'], args['sigmas'])
    else:
        judgments = simulate_rank_judgments(trials, distances, args['num_repeats'], args['sigmas'])
    LOG.info('##  Ranking paradigm trials created: %s', len(trials))
    LOG.info('##  Number of repeats per trial: %s', args['num_repeats'])
    LOG.info('##  Ranking judgments obtained')
    return judgments


def all_distance_pairs(trial_config):
    ref, circle = trial_config[0], trial_config[1]
    pairs = list(combinations(trial_config[1], 2))

    def format_helper(x):
        return (ref, x[0]), '<', (ref, x[1])

    return list(map(format_helper, pairs))


def ranking_to_pairwise_comparisons(distance_pairs, ranked_stimuli):
    # ranked_stimuli is a list of lists. each list is a 'repeat'
    comparisons = {}
    for rank in ranked_stimuli:
        for pair in distance_pairs:
            # place the smaller number first for readability
            if pair[0][1] < pair[2][1]:
                stim1 = pair[0][1]
                stim2 = pair[2][1]
            else:
                stim1 = pair[2][1]
                stim2 = pair[0][1]
            stim1 = 's' + str(stim1)  # pair is a 3-tuple (ref, a), '<', (ref, b)
            stim2 = 's' + str(stim2)
            # format pair (put smaller stim index first)
            # even though the experiment itself asks subjects to click in order of similarity, thereby choosing the
            # smallest distance first, the analysis pipeline looks are pairwise comparisons of the sort:
            # 'Is dist A > dist B?' For this reason, I change the < sign to >.
            ordered = (pair[0][0], int(stim1[1:])), '>', (pair[2][0], int(stim2[1:]))
            if ordered not in comparisons:
                comparisons[ordered] = 1 if rank[stim1] > rank[stim2] else 0
            else:
                if rank[stim1] > rank[stim2]:
                    comparisons[ordered] += 1
    return comparisons


# helper
def add_row(fields, table):
    for fieldname, value in fields.items():
        table[fieldname].append(value)
    return table
