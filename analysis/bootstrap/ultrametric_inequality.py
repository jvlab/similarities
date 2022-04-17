"""
We can study the heuristic distance matrix assembled from raw similarity judgments and ask how many
triplets of distances obey the ultrametric inequality.  We can do this via bootstrapping the choice
probabilities to get a good estimate of the variation in heuristic distances and better judge if there
are any reliable differences in the 'ultrametricity' or tendency to ultrametricity in one domain relative to
another.

In order to fully make sense of any results arising from this, it would be good to know in addition:
- what the expected ultrametricity is when points are sampled randomly from Euclidean spaces
- what proportion of these heuristic distances even satisfy the triangle inequality
  because if they do not, we really can't be thinking of them as distances anymore.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from analysis.geometry.euclidean import EuclideanSpace as space
from analysis.util import read_in_params
import matplotlib.pyplot as plt
from analysis.model_fitting.model_fitting import decompose_similarity_judgments
import analysis.model_fitting.mds as mds
from itertools import combinations, permutations


# Copied from curvature_and_model_goodness
def sample_judgments(original_judgments, num_repeats):
    """
    Simulate judgments based on empirical choice probabilities
    :param original_judgments:
    :param num_repeats:
    :return:
    """
    sample = {}
    for trial, count in original_judgments.items():
        sample[trial] = 0
        prob = float(count) / num_repeats
        for j in range(num_repeats):
            random_draw = np.random.uniform(0, 1)
            if random_draw < prob:
                sample[trial] += 1
    return sample


def produce_surrogate_data(judgments_orig, params, batch_size=1):
    """
    Return a collection of surrogate judgments based on real data
    @param judgments_orig:  real data
    @param batch_size: size of surrogate datasets to make in a go
    @param params: read in from Config file
    @return:
    """
    batch = []
    for i in range(batch_size):
        new_judgments = sample_judgments(judgments_orig, params['num_repeats'])
        batch.append(new_judgments)
    return batch


def count_ultrametric_triplets(D, num_points):
    point_triplets = list(permutations(range(num_points), 3))
    total_triplets = len(point_triplets)
    ultrametric = 0
    for p in point_triplets:
        # check if ultrametric inequality is obeyed or not
        if D[p[0], p[2]] <= max(D[p[0], p[1]], D[p[1], p[2]]):
            ultrametric += 1
    return ultrametric / total_triplets


def count_ultrametric_pairs(D, num_points):
    point_pairs = list(combinations(range(num_points), 2))
    all_points = list(range(num_points))
    total_pairs = len(point_pairs)
    ultrametric_pairs = 0
    for p in point_pairs:
        middle_point = all_points[0:p[0]] + all_points[p[0] + 1:p[1]] + all_points[p[1] + 1:]
        ultrametric = 0
        for y in middle_point:
            # check if ultrametric inequality is obeyed or not
            if D[p[0], p[1]] <= max(D[p[0], y], D[y, p[1]]):
                ultrametric += 1
        if ultrametric == num_points - 2:
            ultrametric_pairs += 1
    return ultrametric_pairs / total_pairs


def count_triangle_inequality_pairs(D, num_points):
    point_pairs = list(combinations(range(num_points), 2))
    all_points = list(range(num_points))
    total_pairs = len(point_pairs)
    triangle_pairs = 0
    for p in point_pairs:
        middle_point = all_points[0:p[0]] + all_points[p[0] + 1:p[1]] + all_points[p[1] + 1:]
        triangle = 0
        for y in middle_point:
            # check if ultrametric inequality is obeyed or not
            if D[p[0], p[1]] <= (D[p[0], y] + D[y, p[1]]):
                triangle += 1
        if triangle == num_points - 2:
            triangle_pairs += 1
    return triangle_pairs / total_pairs


CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()
domains = ['texture', 'intermediate_texture', 'intermediate_object', 'image', 'word']
plt.figure()
subject = 'SA'
N_ITERATIONS = 100

baseline_ultrametricity2 = []
baseline_ultrametricity5 = []
baseline_ultr_trip = []
for _ in range(N_ITERATIONS):
    s = space(3)  # DIM = 2, 5
    s5 = space(7)
    # baseline ultrametricity expected in Euclidean plane
    random_points = s.get_samples(37, 100*np.random.random(), 'gaussian')
    dist_euclidean = squareform(pdist(random_points))
    baseline_ultrametricity2.append(count_ultrametric_pairs(dist_euclidean, 37))

    random_points = s5.get_samples(37, 100*np.random.random(), 'gaussian')
    dist_euclidean = squareform(pdist(random_points))
    baseline_ultrametricity5.append(count_ultrametric_pairs(dist_euclidean, 37))
    baseline_ultr_trip.append(count_ultrametric_triplets(dist_euclidean, 37))


for d in range(len(domains)):
    domain = domains[d]
    INPUT_DATA = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/' \
                 'experiments/{}_exp/subject-data/preprocessed/{}_{}_exp.json'.format(domain, subject, domain)
    judgments = decompose_similarity_judgments(INPUT_DATA, NAMES_TO_ID)

    ultrametricity = []
    triangle_inequality = []
    for n in range(int(N_ITERATIONS/10)):
        surrogate_datasets = produce_surrogate_data(judgments, CONFIG, 1)[0]
        dist_matrix = mds.format_distances(mds.heuristic_distances(judgments, 5))
        ultrametricity.append(count_ultrametric_pairs(dist_matrix, 37))
        triangle_inequality.append((count_triangle_inequality_pairs(dist_matrix, 37)))

    plt.subplot(1, len(domains), d+1)
    plt.boxplot(ultrametricity, 'k')
    plt.boxplot(triangle_inequality, 'b')
    plt.ylim([0, 1])

    plt.boxplot(baseline_ultrametricity2, 'r')
    plt.boxplot(baseline_ultrametricity5, 'm')
    plt.ylim([0, 1])

plt.show()
