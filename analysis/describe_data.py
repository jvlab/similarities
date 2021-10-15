"""
Take in files in json format and look through it
"""

import numpy as np
import pandas as pd
import seaborn as sns
from analysis.util import all_distance_pairs, read_file, ranking_to_pairwise_comparisons
from matplotlib import pyplot as plt
from itertools import combinations


def read_all_files(subjects, data_dir):
    # Keys are "subject,experiment", e.g. "MC,word"
    rank_judgments_by_subject = {}
    for subject in subjects:
        rank_judgments_by_subject[subject] = read_file(subject, data_dir)
    return rank_judgments_by_subject


def group_trials_by_ref(trials):
    grouped = {}
    for trial in trials:
        reference = trial.split(':')[0]
        if reference not in grouped:
            grouped[reference] = {}
        grouped[reference][trial] = trials[trial]
    return grouped


def group_by_overlap(reference, trials):
    circles = []
    for trial in trials:
        circles.append(set(trials[trial][0]))
    result = []
    for i in range(6):  # could change if it was a different paradigm, > 37 stimuli
        for j in range(i + 1, 6):
            stimuli = sorted(list(circles[i].intersection(circles[j])))
            if len(stimuli) > 0:
                result.append({'1': '{}:{}'.format(reference, '.'.join(sorted(list(circles[i])))),
                               '2': '{}:{}'.format(reference, '.'.join(sorted(list(circles[j])))),
                               'stimuli': stimuli})
    return result


def choice_probability_distribution_plot(subject_names, data):
    df = {'choice probability': [], 'frequency': [], 'subject': []}
    for i in range(len(subject_names)):
        contents = data[subject_names[i]]  # read in data file contents
        counts = np.array([])
        for trial in contents:
            pairwise_decisions = list(ranking_to_pairwise_comparisons(all_distance_pairs(trial),
                                                                      contents[trial]).values())
            counts = np.concatenate((counts, np.array(pairwise_decisions)))
        probs = counts / 5.0
        unique, freq = np.unique(probs, return_counts=True)
        freq = freq / float(sum(freq))
        df['choice probability'] += list(unique)
        df['frequency'] += list(freq)
        df['subject'] += [subject_names[i] for _ in range(len(unique))]
    df = pd.DataFrame(df)
    sns.catplot(x="choice probability", y="frequency",
                kind="bar", hue="subject",
                data=df, legend_out=False)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.show()


def choice_prob_ratio_heatmap(trialwise_choice_probabilities, subject_names):
    pairs_of_subjects = list(combinations(subject_names, 2))
    f, axes = plt.subplots()
    cbar_kws = {'label': 'ratio of probability to independent joint probability',
                'orientation': 'vertical'}
    ncol = len(pairs_of_subjects)
    nrow = 1
    for n in range(len(pairs_of_subjects)):
        subject1, subject2 = pairs_of_subjects[n]
        # generate heatmap (count in sub 1 vs count in sub2 - occurrence)
        heatmap = np.zeros((6, 6))
        for i in range(len(trialwise_choice_probabilities[subject1])):
            heatmap[trialwise_choice_probabilities[subject1][i], trialwise_choice_probabilities[subject2][i]] += 1
        heatmap = np.round(heatmap / float(sum(sum(heatmap))), 3)

        ratio_map = np.zeros((6, 6))
        # divide by independent prob at each cell
        for r in range(len(ratio_map)):
            for c in range(len(ratio_map[0])):
                independent_prob = sum(heatmap[r, :]) * sum(heatmap[:, c])
                ratio_map[r, c] = heatmap[r, c] / float(independent_prob)

        current_axis = plt.subplot(nrow, ncol, n + 1)
        if n == ncol - 1:
            g = sns.heatmap(ratio_map, square=True,
                            xticklabels=[0, None, None, None, None, 1], annot=False,
                            yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                            ax=current_axis, vmin=0.25, vmax=2.5, cbar_kws=cbar_kws)
        else:
            g = sns.heatmap(ratio_map, square=True,
                            xticklabels=[0, None, None, None, None, 1], annot=False,
                            yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                            ax=current_axis, vmin=0.25, vmax=2.5, cbar_kws=None)
        bottom, top = current_axis.get_ylim()
        current_axis.set_ylim(bottom + 0.5, top - 0.5)
        current_axis.set_xlabel('{}'.format(subject1))
        current_axis.set_ylabel('{}'.format(subject2))
        current_axis.set_aspect('equal')
        current_axis.invert_yaxis()
        cbar = current_axis.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
    f.tight_layout(pad=1)
    plt.show()


def context_effects_heatmap(subs, data, annotate=False):
    f, ax = plt.subplots(1, len(subs))
    f.tight_layout(pad=1)
    cbar_kws = {'label': 'ratio of probability to independent joint probability',
                'orientation': 'vertical'}
    ncol = len(subs)
    for i in range(ncol):
        if ncol == 1:
            ax = [ax]
        # reformat dict into dict of 37 dicts (grouping trials by reference)
        subject_trials = group_trials_by_ref(data[subs[i]])
        # iterate over all reference stimuli to check context dependence
        consistent = {}  # to carry counts
        prob_in_context_ab = np.zeros((6, 6))
        for animal in subject_trials:
            # group together pairs of trials that have 2 stimuli in common
            overlapping_trials = group_by_overlap(animal, subject_trials[animal])
            # iterate over all pairs of overlapping trials
            # for each, check if distance comparison is the same in changing context
            for overlapping_pair in overlapping_trials:
                stimuli = sorted(list(overlapping_pair['stimuli']))
                binary_decision = '{},{}<{},{}'.format(animal, stimuli[0], animal, stimuli[1])
                dist_pair = [binary_decision]
                judgment1 = ranking_to_pairwise_comparisons(
                    dist_pair,
                    subject_trials[animal][overlapping_pair['1']]
                )[binary_decision]
                judgment2 = ranking_to_pairwise_comparisons(
                    dist_pair,
                    subject_trials[animal][overlapping_pair['2']]
                )[binary_decision]

                prob_in_context_ab[judgment1, judgment2] += 1
        prob_in_context_ab = prob_in_context_ab / float(222)
        print(prob_in_context_ab)
        ratio_map = np.zeros((6, 6))
        for k in range(6):
            for j in range(6):
                independent_prob = sum(prob_in_context_ab[k, :]) * sum(prob_in_context_ab[:, j])
                ratio_map[k, j] = prob_in_context_ab[k, j] / independent_prob
        if i == ncol - 1:
            sns.heatmap(ratio_map, square=True,
                        xticklabels=[0, None, None, None, None, 1], annot=annotate,
                        yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r", center=1,
                        ax=ax[i], cbar_kws=cbar_kws, vmin=0.25, vmax=6)
        else:
            sns.heatmap(ratio_map, square=True,
                        xticklabels=[0, None, None, None, None, 1], annot=annotate,
                        yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r", center=1,
                        ax=ax[i], cbar_kws=None, vmin=0.25, vmax=6)
        bottom, top = ax[i].get_ylim()
        ax[i].set_title(subs[i])
        ax[i].set_ylim(bottom + 0.5, top - 0.5)
        ax[i].set_xlabel('response in context a')
        ax[i].set_ylabel('response in context b')
        ax[i].invert_yaxis()
    plt.show()


def get_choice_probs_by_subject(subject_names, data):
    choice_probs = {}
    for subject_name in subject_names:
        choice_probs[subject_name] = {}
        ranking_trials = data[subject_name]
        for trial in ranking_trials:
            choice_probs[subject_name][trial] = ranking_to_pairwise_comparisons(all_distance_pairs(trial),
                                                                                ranking_trials[trial])
    return choice_probs


def arrange_choice_probs_by_comparison(subject_names, choice_probs):
    trialwise_choice_probabilities = {}
    for subject in subject_names:
        if subject not in trialwise_choice_probabilities:
            trialwise_choice_probabilities[subject] = []
        for trial, comparisons in choice_probs[subject_names[0]].items():
            for comparison, count in comparisons.items():
                trialwise_choice_probabilities[subject].append(choice_probs[subject][trial][comparison])
    return trialwise_choice_probabilities


def subject_comparison_heatmap(subject_names, data):
    if len(subject_names) < 2:
        print("WARNING: At least two subjects needed to make comparison heatmaps (function subject_comparison_heatmap).")
        return
    choice_probs = get_choice_probs_by_subject(subject_names, data)
    trialwise_choice_probabilities = arrange_choice_probs_by_comparison(subject_names, choice_probs)
    choice_prob_ratio_heatmap(trialwise_choice_probabilities, subject_names)


if __name__ == '__main__':
    subs = input("Subjects separated by spaces:")
    SUBJECTS = subs.split(' ')
    DATA_DIR = input("Path to the subject-data/preprocessed directory\n e.g., "
                     "'./sample-materials/subject-data/preprocessed': ")

    ALL_DATA = read_all_files(SUBJECTS, DATA_DIR)
    # Display choice probability distributions
    choice_probability_distribution_plot(SUBJECTS, ALL_DATA)

    # Create subject-comparison heatmaps
    subject_comparison_heatmap(SUBJECTS, ALL_DATA)

    # Create context effects heatmaps
    context_effects_heatmap(SUBJECTS, ALL_DATA, annotate=False)
