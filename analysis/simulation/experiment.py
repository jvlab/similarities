"""
This file makes use of geometry to use the noisy data drawn from a Euclidean
space to simulate pairwise similarity judgments.

If (with noise added) the distance between stimuli A and B is less than the distance between
stimuli C and D, then our simulated subject will judge the difference
between stimuli C and D to be greater than the difference between A and B.

At the same time a geometry probability is given by an erf function and it helps us geometry the
simulated subject's responses. We call it our decision geometry.
An erf function ranges from -1 to 1. We can get the value of an erf function at a value given
by a function of the noisy distance comparison and then if that value is above a threshold we
make a decision and choose one option. Otherwise we choose the other option.

The p((xi, xj) > (xk, xl)) = 1/2(1 + erf( (d(xi, xj)-d(xk, xl)) / 2*sigma ))
range of 1 + erf is 0 to 2.
1/2 times the expression makes the range of values restricted to 0 to 1, so it is a probability.

We divide that difference of distances by 2*sigma as a normalizing factor. 95% of values of a
Gaussian fall within 2*sigma of the mean so this way we are covering a good range of values.
(Understand this better before the ACE^)
"""
import itertools
import logging
import math

import numpy as np

import analysis.simulation.ranking_task as rank

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def distance(point_a, point_b):
    """Return the Euclidean distance between vectors a and b"""
    difference = point_a - point_b
    return math.sqrt(difference.dot(difference))


def prob_a_greater_than_b(pair_a, pair_b, interstimulus_distances, sigmas, no_noise=False):
    """ Using the decision in the 2017 Victor et al paper in Vision Research
    (Two Representations) get a probability for d_ij > d_kl, where d_ij is the
    perceived dissimilarity between stimuli i and j.
    d(xi, xj) is given by the ordinary Euclidean distance between the points xi
    and xj to each of which noise has been added sd= sigmas['sigma_dist']

    To d noise is again added to yield d'.
    d'(xi, xj) is compared to d'(xk, xl) and then noise is added to their difference.
    delta = d'(xi, xj) - d'(xk, xl) + noise (sd= sigmas['compare'])

    :param no_noise: bool, is there noise in judgment or not
    :param pair_a: (i, j) 2 points (noisy coordinates)
    :param pair_b: (k, l) 2 points (noisy coordinates)
    :param interstimulus_distances: a look up matrix of distances between all pairs of stimuli
    :param sigmas: is a dict with keys point, sigma_dist and compare
    :return probability: that i and j are judged to be more different from each other
    than k and l are from each other.
    """
    x_i, x_j = pair_a
    x_k, x_l = pair_b
    dist_a = interstimulus_distances[x_i, x_j]
    dist_b = interstimulus_distances[x_k, x_l]
    if no_noise or (sigmas['dist'] == 0 and sigmas['compare'] == 0):
        if dist_a == dist_b:
            probability = 0.5
        else:
            probability = 1 if dist_a > dist_b else 0
    else:
        # I also think total_sigma is calculated incorrectly. Should be sqrt of sum of variances
        # total_sigma = math.sqrt(
        #   2 * (sigmas['dist'] ** 2) + sigmas['compare'] ** 2)
        total_sigma = math.sqrt((sigmas['dist'] ** 2) + sigmas['compare'] ** 2)
        # probability = 0.5 * (1 + math.erf((dist_a - dist_b) / float(math.sqrt(2) * total_sigma)))
        # I think there is a mistake here. Corrected below
        probability = 0.5 * (1 + math.erf((dist_a - dist_b) / float(2 * total_sigma)))
    return probability


def prob_a_greater_than_b_1d(dist_a, dist_b, sigmas, no_noise=False):
    if no_noise or (sigmas['dist'] == 0 and sigmas['compare'] == 0):
        if dist_a == dist_b:
            probability = 0.5
        else:
            probability = 1 if dist_a > dist_b else 0
    else:
        # this is the same...
        total_sigma = math.sqrt(
            2 * (sigmas['dist'] ** 2) + sigmas['compare'] ** 2)
        probability = 0.5 * (1 + math.erf((dist_a - dist_b) / float(math.sqrt(2) * total_sigma)))
    return probability


def create_trials(vectors, paradigm="pairs_of_pairs"):
    """ Given an array of vectors (stimulus indices), create 'trials' (without randomization).
    Each trial should consist of two pairs of stimuli - the order and placement of which
    do not matter as this is a simulation in which each trial is considered independently
    of any others, independent of order in which it is considered etc.
    :param vectors: a list of stimulus indices that create_trials will group into groups of four. These groups
                    are known as trials.
    :param paradigm: 'pairs_of_pairs' by default, only one supported as of 5/18/20
    :return trials: a list of named tuples that contain stimuli.
                    The structure of trials depends on the experimental paradigm. By default
                    this is "pairs_of_pairs". For this paradigm, a trial is of the form
                    trial = ((x_i, x_j), (x_k, x_l))
                    The structure of trials can be varied if needed in the future.
    """
    if paradigm == "ranking":
        return rank.create_trials(len(vectors))
    elif paradigm != "pairs_of_pairs":
        return None
    else:
        # get all pairs of pairs of vectors (i,j is the same as j,i)
        trials = list(itertools.combinations(
            itertools.combinations(range(len(vectors)), 2), 2))
        return trials


def compare_distances(dist_a, dist_b, sigmas, num_repeats, no_noise):
    count = 0
    if no_noise or (sigmas['dist'] == 0 and sigmas['compare'] == 0):
        if dist_a == dist_b:
            count = int(num_repeats / 2)
        else:
            count = num_repeats if dist_a > dist_b else 0
    else:
        for _ in range(num_repeats):
            noisy_dist_1 = dist_a + np.random.normal(0, sigmas['dist'])
            noisy_dist_2 = dist_b + np.random.normal(0, sigmas['dist'])
            noisy_delta = noisy_dist_1 - noisy_dist_2 + np.random.normal(0, sigmas['compare'])
            if noisy_delta > 0:
                count += 1
    return count, num_repeats


def compare_similarity(pair1, pair2, interstimulus_distances, sigmas, num_repeats=1, no_noise=False):
    """ Calculate the distance between stimuli in pair 1, the stimuli in pair 2,
    and decide which one is bigger. Values have noise added at both the distance
    calculation and the comparison stages.

    Repeat a trial many times - given by num_repeats.

    Return the number of times pair1 is judged to be 'bigger' than pair2.
    :param pair1: indices of 2 points, not the actual arrays
    :param pair2: indices 2 points
    :param interstimulus_distances: a look up matrix of distances between all pairs of stimuli
    :param sigmas: dict of noise params
    :param num_repeats: int, number of times a trial is repeated
    :param no_noise: bool that controls whether there is noise in judgment or not
    :return N (count: how many times pair1 was chosen)
    :return num_repeats: the total number of repeats per trial """
    dist_a = interstimulus_distances[pair1[0], pair1[1]]
    dist_b = interstimulus_distances[pair2[0], pair2[1]]
    return compare_distances(dist_a, dist_b, sigmas, num_repeats, no_noise)


# helpers
def make_key(first, second):
    return (first[0], first[1]), '>', (second[0], second[1])


def make_verbose(first, second):
    return "d_{},{} > d_{},{}".format(first[0], first[1], second[0], second[1])


def simulate_judgments(trial_pairs, all_distances, sigmas, num_repeats=5, no_noise=False, verbose=False):
    """At each 'trial',
    one of the pairs will be chosen as the one that is 'more different.
    Given a list of trials, simulate what a subject would do when asked to choose the pair
    that is more different than the other.
    :param trial_pairs: contains configuration of stimuli for each trial
    @param all_distances: n by n distance matrix where n is the number of stimuli
    :param num_repeats: number of times to repeat a trial (default = 1)
    :param sigmas: a dictionary containing sd of noise sources (default above)
    :param no_noise: boolean denoting whether or not the subject's judgments are subject
           to error due to noise in processing steps/ estimation (default=False)
    :param verbose: boolean denoting whether the returned counts dict should have keys
           that are more readable.
    :return count, num_repeats: a tuple containing a counts dict (for each trial) and the
                    num_repeats arg

    """
    counts = {}  # initialize counts for data to be returned
    # the counts dictionary will hold judgements
    # key: "d_i,j > d_k,l" or "i,j>k,l"
    # value: number of times ij was judged to be greater than kl (int)

    # two ways to record judgments, verbose is for convenience, the other one 'make_key' is for
    # better for sequential processing
    count_key = make_verbose if verbose else make_key

    for top, bottom in trial_pairs:
        # create a readable key
        key = count_key(top, bottom)
        # get stimuli for the trial - top pair and bottom pair
        pair1 = (top[0], top[1])
        pair2 = (bottom[0], bottom[1])
        # record fraction of times pair1 was judged to be more different than pair2
        counts[key] = compare_similarity(pair1,
                                         pair2,
                                         all_distances,
                                         sigmas,
                                         num_repeats,
                                         no_noise
                                         )[0]
    return counts


def simulate_judgments_1d(trial_pairs, vectors, sigmas, num_repeats=5, no_noise=False, verbose=False):
    counts = {}  # initialize counts for data to be returned
    # the counts dictionary will hold judgements
    # key: "d_i,j > d_k,l" or "i,j>k,l"
    # value: number of times ij was judged to be greater than kl (int)

    # two ways to record judgments, verbose is for convenience, the other one 'make_key' is for
    # better for sequential processing
    count_key = make_verbose if verbose else make_key

    for top, bottom in trial_pairs:
        # create a readable key
        key = count_key(top, bottom)
        # get stimuli for the trial - top pair and bottom pair
        dist_a = abs(vectors[top[0]] - vectors[top[1]])
        dist_b = abs(vectors[bottom[0]] - vectors[bottom[1]])
        # record fraction of times pair1 was judged to be more different than pair2
        counts[key] = compare_distances(dist_a, dist_b,
                                        sigmas,
                                        num_repeats,
                                        no_noise
                                        )[0]
    return counts


def run_experiment(stimuli, distances, args, trials=None):
    """ Simulate an experiment where subjects are asked to make comparisons between pairs of
    stimuli and they make their decision based on the comparison of distances between pairs of
    points (subject to internal noise in comparing and computing distance).
    noise level controlled by params.
    :param trials:
    :param stimuli
    :param args dict, as in DEFAULT above
    """

    if trials is None:
        # prepare trial configuration (sample randomly from all possible trials)
        possible_trials = create_trials(stimuli, paradigm="pairs_of_pairs")

        if len(possible_trials) > args['max_trials']:
            indices = np.random.choice(len(possible_trials), args['max_trials'], replace=False)
            trials = [possible_trials[indices[j]] for j in range(args['max_trials'])]
        else:
            trials = possible_trials

    # generate experimental data
    n_dim = len(stimuli[0])
    if n_dim == 1:
        # distances are calculated here
        judgments = simulate_judgments_1d(trials, stimuli, args['sigmas'],
                                          num_repeats=args['num_repeats'],
                                          no_noise=args['no_noise'],
                                          verbose=args['verbose'])
    else:
        judgments = simulate_judgments(trials, distances, args['sigmas'],
                                       num_repeats=args['num_repeats'],
                                       no_noise=args['no_noise'],
                                       verbose=args['verbose'])
    LOG.info('##  Trials created: %s', len(trials))
    LOG.info('##  Experimental judgments obtained')
    return judgments, np.array(stimuli)
