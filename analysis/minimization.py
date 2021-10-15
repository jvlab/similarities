
import logging
import numpy as np
import analysis.pairwise_likelihood_analysis as analysis

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


# Copied over from src.run_mds_seed
def cost(stimulus_params, pair_a, pair_b, counts, params):
    points = analysis.params_to_points(stimulus_params, params['num_stimuli'], params['n_dim'])
    ll, is_bad = analysis.dist_model_ll_vectorized(pair_a, pair_b, counts, params, points)
    LOG.debug('model is good: {}'.format(not is_bad))
    return -1 * ll


def calculate_gradient(vector, pair_a, pair_b, counts, params, vector_length, delta=1e-03):
    baseline_loss = cost(vector, pair_a, pair_b, counts, params)
    deltas = np.eye(vector_length) * delta
    vectors = np.tile(vector.reshape(vector_length, 1), (1, vector_length))
    delta_vectors = vectors + deltas
    gradient = np.apply_along_axis(cost, 0, delta_vectors, pair_a, pair_b, counts, params) - baseline_loss
    return gradient


def gradient_descent(gradient, start, pair_a, pair_b, counts, params):
    """"
    Implements stochastic gradient descent.
    The code below is my first attempt and not an optimized program
    Based on a tutorial on realpython.com
    """
    vector = np.array(start)
    vector_length = len(vector)
    # vector = vector.reshape(len(vector), 1)  # in case it was a row, vector, make column vector
    for _ in range(params['max_iterations']):
        diff = -params['learning_rate'] * gradient(vector, pair_a, pair_b, counts, params, vector_length)
        if np.all(np.abs(diff) <= params['tolerance']):
            print("Stopped on Iteration number {}".format(_ + 1))
            break
        vector += diff
        if _ % 1000 == 0:
            print('{} Iterations done'.format(_))
    print("Stopped on Iteration number {}".format(_))
    # vector = vector.reshape(1, len(vector))  # returned vector must be a regular  1D array
    return vector

