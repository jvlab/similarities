"""
We simplify the decision model used in experiment_ranking.

In each trial, we calculate 8 distances between stimulus i and R (i=1,...,8). Then we add Gaussian noise once
to account for noise in distance estimation.
Then we simply order these 8 distances from smallest to largest and those are our clicks! The error model here is very
simple compared to the one in experiment_ranking. The only source of noise is in distance estimation not comparison.

When we use the simulated ranking judgments for model-fitting we will do some preprocessing as was done in the analysis
of real data (pilot data, word experiment): decomposing each trial into 28 pairwise comparisons. Before calling the
minimization method that optimizes coordinates for different Euclidean models, I will restructure the simulated data to
conform to the pairs of pairs format as the model-fitting procedure only requires pairwise decisions and pairwise choice
probabilities and is unaffected by the underlying experimental paradigm used to get said pairwise decisions.
"""

import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def simulate_simple_rank_judgments(trials, interstimulus_distance, num_repeats, sigmas):
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
            noise_std = np.sqrt(sigmas['dist'] ** 2 + sigmas['compare'] ** 2)
            distances = [
                interstimulus_distance[ref, _i] + np.random.normal(0, noise_std) for _i in circle
            ]
            # click stimuli in order of similarity, implicitly comparing similarities each time
            sorted_distances = sorted(distances)
            click_indices = [distances.index(v) for v in sorted_distances]
            click_values = [circle[_i] for _i in click_indices]
            for _j in range(num_clicks):
                clicked['s{}'.format(click_values[_j])] = _j
            repeats.append(clicked)
        click_responses[trial] = repeats
    return click_responses
