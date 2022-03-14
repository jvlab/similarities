"""
Here, given a data set, generate surrogate datasets that are drawn from the original choice probablities.
Then, for each surrogate dataset, essentially calculate LLs for a range of curvatures and (chosen/ adjusted)
sigma values.

This should yield a scatterplot of model LLs by curvature values. We start by doing so for the 2D case and we can
progress to higher dimensions as needed.

The inputs are
- a range of curvature values
- dimensions for which to apply the analysis. Minimum dimension is 2.
- path to json files for individual datasets.
- optionally, can and probably should pass in a legend for when multiple curves are drawn
- number of iterations (surrogates)
- max number of iterations to run the model_fitting pipeline with.
"""

import time
import copy
import numpy as np
import pandas as pd
from analysis.util import read_in_params
from analysis.model_fitting import run_mds_seed as rs
from analysis.model_fitting.model_fitting import decompose_similarity_judgments

CONFIG, STIMULI, NAMES_TO_ID, ID_TO_NAME = read_in_params()


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


def produce_surrogate_data(json_file, params, batch_size=2):
    """
    Return a collection of surrogate judgments based on real data
    @param json_file: path to real data file
    @param batch_size: size of surrogate datasets to make in a go
    @param params: read in from Config file
    @return:
    """
    batch = []
    judgments_original = decompose_similarity_judgments(json_file)
    for i in range(batch_size):
        new_judgments = sample_judgments(judgments_original, params['num_repeats'])
        batch.append(new_judgments)
    return batch


def fit_model(similarity_judgments, curvature, params, dim=2):
    params_copy = copy.deepcopy(params)
    num_judgments = len(similarity_judgments)
    noise = params['sigmas']['compare']
    params_copy['n_dim'] = dim  # ensure correct model is tested
    if curvature == 0:
        # fit Euclidean model
        x, ll, fmin_costs = rs.points_of_best_fit(similarity_judgments, params_copy)
        ll = -1 * ll / (params['num_repeats'] * num_judgments)
    elif curvature > 0:
        # fit spherical model
        curvature_val = curvature
        params_copy['curvature'] = curvature_val
        radius = 1 / curvature_val
        rms_dist = radius / 2
        noise = rms_dist / 20  # chosen based on looking at range of RMS dist: sigma of 0.18 in real data
        params_copy['sigmas'] = {'compare': noise, 'dist': 0}
        x, ll, fmin_costs = rs.spherical_points_of_best_fit(similarity_judgments, params_copy)
        ll = -1 * ll / (params['num_repeats'] * num_judgments)
    else:
        # fit hyperbolic model
        curvature_val = -1 * curvature
        params_copy['curvature'] = curvature_val
        x, ll, fmin_costs = rs.hyperbolic_points_of_best_fit(similarity_judgments, params_copy)
        ll = -1 * ll / (params['num_repeats'] * num_judgments)
    return ll, curvature, noise


def run(json_path, subject, domain, num_iterations=1, dim=2):
    for ii in range(num_iterations):
        surrogate_datasets = produce_surrogate_data(json_path, CONFIG, 10)
        degree_curvature = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
        results = {'LL': [], 'Curvature': [], 'Sigma': [], 'Dimension': [], 'Subject': [], 'Domain': []}
        for data in surrogate_datasets:
            for c in degree_curvature:
                log_likelihood, curvature_val, sigma = fit_model(data, c, CONFIG, dim)
                # write to pandas file
                results['Curvature'].append(curvature_val)
                results['LL'].append(log_likelihood)
                results['Sigma'].append(sigma)
                results['Dimension'].append(dim)
                results['Subject'].append(subject)
                results['Domain'].append(domain)
        # write df
        df = pd.DataFrame(results)
        timestamp = time.asctime().replace(" ", '.')
        filename = 'curvature_and_LL_{}-{}_{}.csv'.format(subject, domain, timestamp)
        df.to_csv(filename)
        # plot figure
        # sns.scatterplot(data=df, x="Curvature", y="LL", hue="Sigma")
        # plt.ylim([-1, 0])
        # plt.show()
    return


if __name__ == '__main__':
    SUBJECT = input("Enter Subject initials: ")
    DOMAIN = input("Enter Domain (e.g., word, texture): ")
    INPUT_DATA = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/experiments/' \
                 'experiments/{}_exp/subject-data/preprocessed/{}_{}_exp.json'.format(DOMAIN, SUBJECT, DOMAIN)
    run(INPUT_DATA, SUBJECT, DOMAIN)
