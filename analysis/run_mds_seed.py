""" Here I call the modules in the euclidean simulations folder and create a random data set,
with or without noise.

# Comparing models using likelihoods #####################################################
# create a set of stimuli from a Euclidean space
# use them to simulate a pairs of pairs experiment with some noise added to compare and dist
# take a 100 random trials from the experiment and get subject judgments
# using different model probabilities:
# - calculate the likelihood of observing these responses
# - plot the box plot of likelihood values for null model - 0.5 prob, Euclidean model (erf)
#   and 'best model'
##########################################################################################
"""

import logging

import scipy.spatial as spatial
from scipy import optimize

from analysis import pairwise_likelihood_analysis as analysis, mds, gram_schmidt as gs
from analysis import util
from analysis.minimization import calculate_gradient, gradient_descent

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def points_of_best_fit(judgments, args, start_points=None, minimization='gradient-descent'):
    """
    Given judgments, number of stimuli and dimensions of space,
    find the lowest likelihood points that give that fit
    :param judgments: {'i,j>k,l' : 4, ...}
    :param args: includes n_dim, num_stimuli, no_noise, sigmas
    :param start_points: optional arg, can use to start minimization at ground truth or other location
    :param minimization: gradient-descent (new improvement) or nelder-mead
    :return: optimal (points (x), minimum negative log-likelihood (y))
    """
    # for debugging
    fmin_costs = []

    def cost(stimulus_params, pair_a, pair_b, counts, parameters):
        vectors = analysis.params_to_points(stimulus_params, parameters['num_stimuli'], parameters['n_dim'])
        ll, is_bad = analysis.dist_model_ll_vectorized(pair_a, pair_b, counts, parameters, vectors)
        LOG.debug('model is good: {}'.format(not is_bad))
        fmin_costs.append(-1 * ll)  # debugging fmin
        return -1 * ll

    if start_points is None:
        # if not specified start minimization at coordiates returned by MDS after calculation of win-loss distances
        start_0 = mds.get_coordinates(args['n_dim'], judgments, args['num_repeats'])[0]
    else:
        start_0 = start_points
    start = gs.anchor_points(start_0)
    LOG.info("########  Procrustes distance between start and anchored start: {}".format(
        spatial.procrustes(start, start_0)[2]))
    # turn points to params
    start_params = analysis.points_to_params(start)
    LOG.info('######## Run minimization on MDS start points (scipy minimize)')
    # make maxiter 60000 for 5D model
    options_min = {
        'disp': True,
        'fatol': args['fatol']}
    if args['n_dim'] < 4:
        options_min['maxiter'] = 85000
    elif args['n_dim'] >= 4:
        options_min['maxiter'] = 110000
    pairs_a, pairs_b, response_counts = util.judgments_to_arrays(judgments)
    if minimization == 'nelder-mead':
        optimal = optimize.minimize(cost, start_params,
                                    args=(pairs_a, pairs_b, response_counts, args),
                                    method='Nelder-Mead',
                                    options=options_min
                                    )
        LOG.info(
            '######## {} Iterations completed: {}. Model Dim: {}.'.format(optimal.message, optimal.nit,
                                                                          args['n_dim'])
        )
        solution = optimal.x
        solution_ll = optimal.fun
    else:
        solution = gradient_descent(calculate_gradient, start_params, pairs_a, pairs_b, response_counts, args)
        solution_ll = cost(solution, pairs_a, pairs_b, response_counts, args)

    coordinates = analysis.params_to_points(solution, args['num_stimuli'], args['n_dim'])
    LOG.info('########  Procrustes distance between anchored start and final solution: {}'.format(
        spatial.procrustes(start, coordinates)[2])
    )
    return coordinates, solution_ll, fmin_costs  # , sum_residual_squares
