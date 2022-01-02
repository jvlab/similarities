"""
Code to make Figure 6 or 7 of the F31 application where I show model likelihoods
relative to ground truth (Euclidean model-fitting pipeline).
"""
import glob
import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_model_likelihoods(subjects, path):
    model_fitting_data = []

    for subject in subjects:
        subjectfiles = glob.glob('{}/{}*model-likelihoods*.csv'.format(path, subject))
        print(subjectfiles)
        for file in subjectfiles:
            results = pd.read_csv(file)
            best_index = results.index[results['Model'] == 'best']
            corrected_best_ll = results.iloc[best_index]['Log Likelihood'] - 0
            corrected_best_ll = corrected_best_ll.values[0]
            results.at[best_index, 'Log Likelihood'] = corrected_best_ll
            results['Subject'] = subject
            results['Log Likelihood Relative to "Best" Model'] = results['Log Likelihood'] - corrected_best_ll
            model_fitting_data.append(results)

    model_fitting_data = pd.concat(model_fitting_data, sort=True)
    print(model_fitting_data)
    return model_fitting_data


if __name__ == '__main__':
    SUBJECTS = input("Subjects separated by spaces: ").split(' ')
    PATH_TO_SUBJECT_LOG_LIKELIHOODS = input("Path to the directory containing model log-likelihood files "
                                            "(e.g., ./sample-materials/subject-data/model-fitting/S7/log-likelihoods): ")
    print(SUBJECTS)
    print(PATH_TO_SUBJECT_LOG_LIKELIHOODS)
    all_data = read_model_likelihoods(SUBJECTS, PATH_TO_SUBJECT_LOG_LIKELIHOODS)
    excluded_models = ['best']

    # plot model LLs for all experiments and subjects
    subjects = all_data.Subject.unique()
    g = sns.catplot(data=all_data.loc[~all_data['Model'].isin(excluded_models)],
                    x='Model', y='Log Likelihood Relative to "Best" Model',
                    hue='Subject',
                    kind='point', linestyles="",
                    alpha=0.5,
                    # height=3,
                    palette="viridis_r",
                    order=['random', '1D', '2D', '3D', '4D', '5D'],
                    label='big', legend_out=True)
    # g.set(ylim=(-1, 0.15))
    # add horizontal lines to mark log likelihood of ground truth
    for i in range(len(g.axes[0])):
        for j in range(len(g.axes)):
            g.axes[j, i].axhline(0, color='k')
    plt.show()
