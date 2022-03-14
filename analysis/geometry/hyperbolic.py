"""
Given a Euclidean fit, obtained from ordinary gradient descent that explains the psychophysical judgments, add curvature
(negative curvature to make the space more hyperbolic) and see if the fit improves.
This entails projecting Euclidean points onto the top sheet of a two sheet hyperboloid, using the Loid geometry as the
geometry of the space. Distances between points on the hyperbolid are then taken in a way inspired by/ adapted from Tabaghi
et al's paper, "Hyperbolic Distance Matrices." Once distances are obtained, LL can be calculated.
A parameter, lambda controls how close to the hyperboloid center, the points are projected. When lambda approaches 0,
the distances approach Euclidean distances, so a positive lambda yielding a better LL is evidence for curvature in the
space - IF you can account for the added benefit provided simply by adding 1 more parameter.
"""

from numpy import arccos, arccosh, einsum, zeros, eye, sqrt, round
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def sphere_map(X, radius):
    """ Map points in Euclidean space to points on a spherical surface using stereographic projection
    adapted from https://en.wikipedia.org/wiki/Stereographic_projection to work for projection from a disk of
    @param radius: R to a sphere of radius R.
    @param X: d by n matrix with n points of dimension d in Rd (real numbers, d dim)
    @return Y: d+1 by n matrix with n points projected onto the sphere of dimension d, which is embedded in
    d+1-dimensional space
    """
    # retain coordinates of X but add a 0-th coordinate which is a function of the d-dimensional coordinate values
    d, n = X.shape
    Y = zeros((d + 1, n))
    # squared norms of all vectors = dot products
    dot_prods = einsum('ij,ij->j', X, X)  # https://stackoverflow.com/questions/6229519/numpy-column-wise-dot-product
    denom_xi = 1/(dot_prods + radius**2)
    D = np.diag(2*(radius**2)*denom_xi)
    Y[1:, :] = X @ D
    Y[0, :] = radius * (-radius ** 2 + dot_prods)/(radius ** 2 + dot_prods)
    return Y


def loid_map(X, degree_curvature):
    """ Map points in Euclidean space to points on the hyperboloid using a mapping to the Loid space.
    @param degree_curvature: the aforementioned lambda parameter, 0 if distances are Euclidean.
    @param X: d by n matrix with n points of dimension d in Rd (real numbers, d dim)
    @return Y: d+1 by n matrix with n points projected onto the hyperboloid of dimension d, which is embedded in
    d+1-dimensional space
    """
    # retain coordinates of X but add a 0-th coordinate which is a function of the d-dimensional coordinate values
    d, n = X.shape
    Y = zeros((d + 1, n))
    Y[1:, :] = degree_curvature * X
    dot_prods = einsum('ij,ij->j', X, X)
    Y[0, :] = sqrt(1 + (degree_curvature ** 2) * dot_prods)
    return Y


def hyperbolic_distances(X, curvature):
    """
    Computes the hyperblic distance between points ON the hyperboloid.
    Assumes the points passed in are not off the hyperboloid - already projected!
    @param curvature: 0 when distances supposedly like Euclidean, hyperbolic when > 0
    @param X: d-by-n matrix of coordinates for points on a hyperboloid Ld
    @return: cosh-1(-[X, X]) = an n-by-n matrix of pairwise distance matrix for all points X
    """
    # test entries along the diagonal should equal 1
    # test all entries should be less than or equal to -1 or what notes say
    H = eye(X.shape[0])
    H[0, 0] = -1
    inner_product = X.T @ H @ X
    # return interstimulus_distances
    return arccosh(-round(inner_product, 6)) / curvature


def spherical_distances(X, radius):
    """
    Computes the spherical distance between points ON the sphere.
    Assumes the points passed in are not off the surface - already projected!
    @param radius: radius of the sphere
    @param X: d-by-n matrix of coordinates for points on a sphere
    @return: cos-1(-[X, X]) = an n-by-n matrix of pairwise distance matrix for all points X
    """
    # standard inner product
    inner_product = X.T @ X  # do not make it negative
    interstimulus_distances = (radius/2) * arccos(round((inner_product / (radius**2)), 6))
    return interstimulus_distances


if __name__ == '__main__':
    import pprint
    import random

    import pandas as pd
    from scipy.spatial.distance import pdist
    import analysis.model_fitting.run_mds_seed as rs
    import analysis.model_fitting.pairwise_likelihood_analysis as an
    from analysis.util import read_in_params
    from model_fitting import decompose_similarity_judgments

    # enter path to subject data (json file)
    CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()
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
        pairwise_comparison_responses_by_trial = decompose_similarity_judgments(FILEPATH)

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
            LOG.info('#######  {} dimensional geometry'.format(dim))
            model_name = 'Hyp' + str(dim) + 'D'
            CONFIG['n_dim'] = dim
            x, ll_nd, fmin_costs = rs.hyperbolic_points_of_best_fit(subset, CONFIG)
            LOG.info("Points: ")
            print(x)
            outfilename = '{}/{}_{}_anchored_points_sigma_{}_dim_{}H'.format(
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

        LOG.info('#######  Random and best geometry')
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
        data_frame.to_csv('{}/{}-{}-hyp-geometry-likelihoods_with_{}_trials_sigma_{}_{}pts_anchored_{}.csv'
                          .format(OUTDIR,
                                  SUBJECT,
                                  EXP,
                                  CONFIG['max_trials'],
                                  sigma,
                                  CONFIG['num_stimuli'],
                                  ii))
