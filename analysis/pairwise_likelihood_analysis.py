"""
In this file, I write methods to calculate the log likelihood of a set of count data given by
simulations of experiments.

If we have data of the form N((i, j) > (k, l)) i.e. number of times (repeats of trials) pair (i, j)
was judged to be more different from the pair (k, l), then we can calculate the log likelihood, LL
as follows:

LL = sum_{over all i,j,k,l} {
    N((i, j) > (k, l)).ln p((i, j) > (k, l))
} + ln Kx
where K is a combinatorial constant that counts the number of orders in which the responses could
have been made.
p is given by our model probability using erf.

Note:
If our sigma_point = 0 then the total noise is comprised of two Gaussian sources so erf is fine.
But if sigma_point > 0, then the noise is the square root of a sum of squares of Gaussian distributions plus two
Gaussian distributions, which is not exactly Gaussian. So erf is merely an approximation of the probability,
since it only accounts for Gaussian sources of noise.
"""
import logging

import numpy as np
from scipy import special
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
EPSILON = 1e-30  # consider setting this to sys.float_info.min ~ 2e-300


def params_to_points(x0, num_stimuli, n_dim):
    """
    Takes a vector of parameters and separates it into points
    :param x0: all stimulus coordinates (that are nonzero) vectorized
    :param num_stimuli: (int) number of points
    :param n_dim: (int) dimensionality of space points come from
    :return: points is a 2D array containing point coordinates.
            size: num_stimuli x n_dim
    """
    points = np.zeros((num_stimuli, n_dim))
    pointer = 0
    for i in range(1, min(n_dim, num_stimuli)):
        points[i, 0: i] = x0[pointer: i + pointer]
        pointer = i + pointer
    for j in range(n_dim, num_stimuli):
        points[j] = x0[pointer: pointer + n_dim]
        pointer += n_dim
    return points


def points_to_params(points):
    """
    points is a matrix with at least as many rows as columns.
    d is the number of columns
    n is the number of rows (points)
    This function takes a matrix of the form
    [[0, ..., 0],
    [x1, 0..., 0],
    [x2, x3, ..., 0],
    ...
    [xk, xk+1, ...xl, 0],
    [xl+1, ..., xl+d]
    ...]
    such that the first d points have 0, 1, 2, ..., d non zero params and the following rows all have nonzero params

    Return the nonzero params in an array
    """
    d = points.shape[1]
    params = []
    for row_idx in range(d):
        values = points[row_idx, 0:row_idx]
        for element in values:
            params.append(element)
    params = params + list(points[d:, :].flatten())
    return params


def calculate_ll(counts, probs, num_repeats):
    reverse_counts = num_repeats - counts
    reverse_probs = 1 - probs

    probs = np.concatenate((probs, reverse_probs))
    counts = np.concatenate((counts, reverse_counts))

    model_bad = False
    # check if model is bad, i.e. prob = 0 but count > 0
    prob_zero = probs == 0
    if (counts[prob_zero] > 0).any():
        model_bad = True

    # make sure there are no zero probabilities (avoid log(0) error)
    probs[prob_zero] += EPSILON
    log_likelihood = counts.dot(np.log2(probs))
    return log_likelihood, model_bad


def dist_model_ll_vectorized(pair_a, pair_b, judgment_counts, params, stimuli):
    """ Get the likelihood using probabilities from erf model, i.e. the model
    that takes into account noise as Gaussian sources. """
    # get model probabilities and join counts and model prob for each trial (N, p)
    interstimulus_distances = squareform(pdist(stimuli))
    probs = find_probabilities(interstimulus_distances, pair_a, pair_b, params['sigmas'], params['no_noise'])
    # calculate log-likelihood, is_bad flag
    return calculate_ll(judgment_counts, probs, params['num_repeats'])


def find_probabilities(distances, pair_a, pair_b, sigmas, no_noise=False):
    """
    :param distances: a matrix of distances size (number of stimuli, number of stimuli)
    :param pair_a: a 2D numpy array with a pair of stimulus indices in each row, denoting the first distance
    :param pair_b: a 2D numpy array with a pair of stimulus indices in each row, denoting the second distance
    :param sigmas: contains subfields compare and dist for two possible noise sources
    :param no_noise: boolean, in case needed while calling function in isolation
    """
    difference = distances[pair_a[:, 0], pair_a[:, 1]] - distances[pair_b[:, 0], pair_b[:, 1]]
    if (sigmas['compare'] + sigmas['dist'] == 0) or no_noise is True:
        probabilities = (difference < 0) * 0 + (difference > 0) * 1 + (difference == 0) * 0.5
    else:
        total_st_dev = np.sqrt((sigmas['dist'] ** 2) + sigmas['compare'] ** 2)
        probabilities = 0.5 * (1 + special.erf(difference / float(2 * total_st_dev)))
    return probabilities


def random_choice_ll(judgments, params):
    """ LL is calculated as sum over trials of N(i>j)*P(i>j).
    In this case the P(i> j) for any i or j is 0.5.
    So we sum N*0.5 """
    counts = []
    probs = []
    for v in judgments.values():
        counts.append(v)
        probs.append(0.5)
    return calculate_ll(np.array(counts), np.array(probs), params['num_repeats'])


def best_model_ll(judgments, params):
    """ Use probabilities from observed judgements to calculate likelihood. So if i> j
    2/5 times, prob = 0.4. """
    num_repeats = float(params['num_repeats'])
    counts = []
    probs = []
    for v in judgments.values():
        counts.append(v)
        probs.append(v / num_repeats)
    return calculate_ll(np.array(counts), np.array(probs), num_repeats)


def cost_of_model_fit(stimulus_params, pair_a, pair_b, judgment_counts, params):
    """
    Stimulus_params is the independent variable. LL is the dependent variable.
    :param stimulus_params: nonzero coordinates of each of the stimuli stretched into one vector
    :param pair_a: pair 1 of stimuli from trial
    :param pair_b: pair 2 of stimuli from trial
    :param judgment_counts: counts per pairwise comparison (array)
    :param params: global params for the exp includes sigmas, num_repeats, n_dim etc.
    :return: negative log-likelihood (return -LL so the minimum -LL can be found)
    """
    # get points from params
    points = params_to_points(stimulus_params, params['num_stimuli'], params['n_dim'])
    # calculate likelihood using distance model and given points
    ll, is_bad = dist_model_ll_vectorized(pair_a, pair_b, judgment_counts, params, points)
    LOG.debug('model is good: {}'.format(not is_bad))
    return -1 * ll
