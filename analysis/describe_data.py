"""
Take in files in json format and look through it
"""
from math import floor
import os.path as path
import numpy as np
import pandas as pd
import seaborn as sns
import src.util as util
from scipy.special import rel_entr
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from itertools import combinations
from src.experiment import prob_a_greater_than_b


def read_all_files(subjects_by_domain):
    # Keys are "subject,experiment", e.g. "MC,word"
    rank_judgments_by_subject_experiment = {}
    for exp in subjects_by_domain.keys():
        for subject in subjects_by_domain[exp]:
            rank_judgments_by_subject_experiment['{},{}'.format(subject, exp)] = util.read_file(subject, exp)
    return rank_judgments_by_subject_experiment


def exact_sequence_matches(trials):
    num_same = 0
    num_different = 0
    matched_trials = []

    for trial in trials:
        if all(click_sequence == trials[trial][0] for click_sequence in trials[trial]):
            num_same += 1
            matched_trials.append(trial)
        else:
            num_different += 1
    return {'same': num_same, 'different': num_different, 'same_trials': matched_trials}


def first_last_sequence_matches(trials):
    num_same = 0
    num_different = 0
    same_trials = []
    # count how many trials have the same most and least similar items across the repeats
    for trial in trials:
        firsts = [click[0] for click in trials[trial]]
        lasts = [click[-1] for click in trials[trial]]
        if all(item == firsts[0] for item in firsts) and all(item == lasts[0] for item in lasts):
            num_same += 1
            same_trials.append(trial)
        else:
            num_different += 1
    return {'same': num_same, 'different': num_different, 'same_trials': same_trials}


def barplot_probs(trials, show=True, color='b'):
    counts = np.array([])
    for trial in trials:
        pairwise_decisions = list(util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial), trials[trial]
                                                                       ).values()
                                  )
        counts = np.concatenate((counts, np.array(pairwise_decisions)))
    probs = counts / 5.0
    unique, probs = np.unique(probs, return_counts=True)
    prob_frequency = dict(zip(unique, probs))
    sns.barplot(x=list(prob_frequency.keys()), y=[y / float(37 * 6 * 28) for y in prob_frequency.values()],
                alpha=0.35, color=color)
    plt.ylabel('Fraction of pairwise comparisons')
    plt.xlabel('Choice probabilities')
    # sns.distplot(counts, norm_hist=True, kde=False)
    if show:
        plt.show()


def group_trials_by_ref(trials):
    grouped = {}
    for trial in trials:
        reference = trial.split(':')[0]
        if reference not in grouped:
            grouped[reference] = {}
        grouped[reference][trial] = trials[trial]
    return grouped


def plot_all_agreement_in_barplots(subject, exp):
    animals_by_size = ['ant', 'spider', 'butterfly', 'ladybug', 'snail', 'lizard', 'goldfish',
                       'frog', 'mouse', 'rat', 'bat', 'sparrow', 'bluebird', 'pigeon',
                       'snake', 'duck', 'owl', 'turkey', 'eagle', 'cat', 'turtle', 'dog', 'monkey',
                       'fox', 'sheep', 'goat', 'hog', 'tiger', 'cow', 'horse', 'bear',
                       'dolphin', 'crocodile', 'giraffe', 'elephant', 'shark', 'whale']
    file = util.read_file(subject, exp)
    animals = {}
    # group trials by ref
    for ranking_trial in file:
        ref = ranking_trial.split(':')[0]
        if ref not in animals:
            animals[ref] = []
        animals[ref].append(ranking_trial)
    # get trials for each ref and make a histogram
    count = 0
    fig, axes = plt.subplots(6, 7, sharex=True, sharey=True)
    fig.tight_layout(pad=1)
    for ref in animals_by_size:
        i = int(count / 7)
        j = count - i * 7
        subset = {}
        for ranking_trial in animals[ref]:
            subset[ranking_trial] = file[ranking_trial]
        axes[i, j].set_title(ref, fontsize=11)
        counts = np.array([])
        for trial in subset:
            pairwise_decisions = list(util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial), subset[trial]
                                                                           ).values()
                                      )
            counts = np.concatenate((counts, np.array(pairwise_decisions)))

        unique, counts_new = np.unique(counts, return_counts=True)
        count_frequency = dict(zip(unique, counts_new))
        total = 6 * 28
        certain = 100 * (count_frequency[0] + count_frequency[5]) / float(total)
        uncertain = 100 * (count_frequency[1] + count_frequency[2] +
                           count_frequency[3] + count_frequency[4]) / float(total)
        data = pd.DataFrame({'agreement': ['yes', 'no'], 'count': [certain, uncertain]})
        sns.barplot(x="agreement", y="count", data=data, ax=axes[i, j])
        count += 1
    plt.show()


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


def make_figure_2b_ratio_pilot_data(subjects, exp, annotate=False, n=3):
    # TODO  make this work for other than 3 subjects too
    choice_probs = {}
    for j in range(len(subjects)):
        choice_probs[subjects[j]] = {}
        ranking_trials = util.read_file(subjects[j], exp)
        for trial in ranking_trials:
            choice_probs[subjects[j]][trial] = util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial),
                                                                                    ranking_trials[trial])
    sub1 = []
    sub2 = []
    if n == 3:
        sub3 = []
    for trial, comparisons in choice_probs[subjects[0]].items():
        for comparison, count in comparisons.items():
            sub1.append(count)
            sub2.append(choice_probs[subjects[1]][trial][comparison])
            if n == 3:
                sub3.append(choice_probs[subjects[2]][trial][comparison])
    # generate heatmap (count in sub 1 vs count in sub2 - occurrence)
    heatmap_12 = np.zeros((6, 6))
    heatmap_13 = np.zeros((6, 6))
    heatmap_23 = np.zeros((6, 6))
    for i in range(len(sub1)):
        heatmap_12[sub1[i], sub2[i]] += 1
        if n == 3:
            heatmap_13[sub1[i], sub3[i]] += 1
        if n == 3:
            heatmap_23[sub2[i], sub3[i]] += 1
    heatmap_12 = np.round(heatmap_12 / float(222 * 28), 3)
    if n == 3:
        heatmap_13 = np.round(heatmap_13 / float(222 * 28), 3)
        heatmap_23 = np.round(heatmap_23 / float(222 * 28), 3)

    ratio_map_12 = np.zeros((6, 6))
    ratio_map_13 = np.zeros((6, 6))
    ratio_map_23 = np.zeros((6, 6))
    indep_12 = np.zeros((6, 6))
    indep_13 = np.zeros((6, 6))
    indep_23 = np.zeros((6, 6))
    # divide y independent pro at each cell
    for k in range(len(ratio_map_12)):
        for l in range(len(ratio_map_12[0])):
            # 12
            independent_p = sum(heatmap_12[k, :]) * sum(heatmap_12[:, l])
            ratio_map_12[k, l] = heatmap_12[k, l] / float(independent_p)
            indep_12[k, l] = independent_p
            # 13
            if n == 3:
                independent_p = sum(heatmap_13[k, :]) * sum(heatmap_13[:, l])
                ratio_map_13[k, l] = heatmap_13[k, l] / float(independent_p)
                indep_13[k, l] = independent_p
                # 23
                independent_p = sum(heatmap_23[k, :]) * sum(heatmap_23[:, l])
                ratio_map_23[k, l] = heatmap_23[k, l] / float(independent_p)
                indep_23[k, l] = independent_p
    print(ratio_map_12)

    # start plotting
    f, ax = plt.subplots(1, 3)
    f.tight_layout(pad=1)
    cbar_ax = f.add_axes([.3, .18, .35, .03])
    cbar_kws = {'label': 'observed probability : independent joint probability',
                'orientation': 'vertical'}  # 'horizontal'}
    sns.heatmap(ratio_map_12, square=True,
                xticklabels=[0, None, None, None, None, 1], annot=annotate, center=1,
                yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                ax=ax[0], cbar_kws=cbar_kws)  # , cbar_ax=cbar_ax, )
    bottom, top = ax[0].get_ylim()
    ax[0].set_ylim(bottom + 0.5, top - 0.5)
    ax[0].set_xlabel('{}'.format(subjects[1]))
    ax[0].set_ylabel('{}'.format(subjects[0]))
    if n == 3:
        sns.heatmap(ratio_map_13, square=True, cmap="RdYlBu_r",
                    xticklabels=[0, None, None, None, None, 1], annot=annotate,
                    yticklabels=[0, None, None, None, None, 1], center=1,
                    ax=ax[1], cbar_ax=cbar_ax, cbar_kws=cbar_kws)
        bottom, top = ax[1].get_ylim()
        ax[1].set_ylim(bottom + 0.5, top - 0.5)
        ax[1].set_xlabel('{}'.format(subjects[2]))
        ax[1].set_ylabel('{}'.format(subjects[0]))
        sns.heatmap(ratio_map_23, square=True, annot=annotate,
                    xticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                    yticklabels=[0, None, None, None, None, 1], center=1,
                    ax=ax[2], cbar_ax=cbar_ax, cbar_kws=cbar_kws)
        bottom, top = ax[2].get_ylim()
        ax[2].set_ylim(bottom + 0.5, top - 0.5)
        ax[2].set_xlabel('{}'.format(subjects[2]))
        ax[2].set_ylabel('{}'.format(subjects[1]))

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    print(heatmap_12)
    plt.show()


def make_fig_2c_pilot_data(subs, exp):
    f, ax = plt.subplots(1, 3)
    f.tight_layout(pad=1)
    cbar_ax = f.add_axes([.3, .18, .35, .03])
    cbar_kws = {'label': 'proportion of pairwise choices',
                'orientation': 'horizontal'}
    for i in range(len(subs)):
        # reformat dict into dict of 37 dicts (grouping trials by reference)
        subject_trials = group_trials_by_ref(util.read_file(subs[i], exp))
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
                judgment1 = util.ranking_to_pairwise_comparisons(
                    dist_pair,
                    subject_trials[animal][overlapping_pair['1']]
                )[binary_decision]
                judgment2 = util.ranking_to_pairwise_comparisons(
                    dist_pair,
                    subject_trials[animal][overlapping_pair['2']]
                )[binary_decision]

                prob_in_context_ab[judgment1, judgment2] += 1
        prob_in_context_ab = prob_in_context_ab / float(222)
        sns.heatmap(prob_in_context_ab, square=True,
                    xticklabels=[0, None, None, None, None, 1],
                    yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                    ax=ax[i], cbar_ax=cbar_ax, cbar_kws=cbar_kws)
        bottom, top = ax[i].get_ylim()
        ax[i].set_title(subs[i])
        ax[i].set_ylim(bottom + 0.5, top - 0.5)
        ax[i].set_xlabel('response in context a')
        ax[i].set_ylabel('response in context b')
        ax[i].invert_yaxis()
    plt.show()


def make_fig_2c_ratio_pilot_data(subs, exp, data, annotate=False):
    f, ax = plt.subplots(1, len(subs))
    f.tight_layout(pad=1)
    # cbar_ax = f.add_axes([.3, .18, .35, .03])
    cbar_kws = {'label': 'ratio of probability to independent joint probability',
                'orientation': 'horizontal'}
    for i in range(len(subs)):
        if len(subs) == 1:
            ax = [ax]
        # reformat dict into dict of 37 dicts (grouping trials by reference)
        subject_trials = group_trials_by_ref(data['{},{}'.format(subs[i], exp)])
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
                judgment1 = util.ranking_to_pairwise_comparisons(
                    dist_pair,
                    subject_trials[animal][overlapping_pair['1']]
                )[binary_decision]
                judgment2 = util.ranking_to_pairwise_comparisons(
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
        sns.heatmap(ratio_map, square=True,
                    xticklabels=[0, None, None, None, None, 1], annot=annotate,
                    yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r", center=1,
                    ax=ax[i], cbar_kws=cbar_kws, vmin=0.25, vmax=6)  # cbar_ax=cbar_ax)
        bottom, top = ax[i].get_ylim()
        ax[i].set_title(subs[i])
        ax[i].set_ylim(bottom + 0.5, top - 0.5)
        ax[i].set_xlabel('response in context a')
        ax[i].set_ylabel('response in context b')
        ax[i].invert_yaxis()
    plt.show()


def make_figure_1_pilot_data(subjects_by_experiment, stimulus_domains, ids, data):
    df = {'choice probability': [], 'frequency': [], 'subject': [], 'subjectId': [], 'domain': []}
    for j in range(len(stimulus_domains)):
        domain = stimulus_domains[j]
        subject_names = subjects_by_experiment[domain]
        for i in range(len(subject_names)):
            contents = data['{},{}'.format(subject_names[i], domain)]  # read in data file contents
            counts = np.array([])
            for trial in contents:
                pairwise_decisions = list(util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial),
                                                                               contents[trial]).values())
                counts = np.concatenate((counts, np.array(pairwise_decisions)))
            probs = counts / 5.0
            unique, freq = np.unique(probs, return_counts=True)
            freq = freq / float(sum(freq))
            df['choice probability'] += list(unique)
            df['frequency'] += list(freq)
            df['domain'] += [domain for _ in range(len(unique))]
            df['subject'] += [subject_names[i] for _ in range(len(unique))]
            df['subjectId'] += [ids[subject_names[i]] for _ in range(len(unique))]
    df = pd.DataFrame(df)
    sns.catplot(x="choice probability", y="frequency", col="domain",
                kind="bar", hue="subjectId",
                data=df, legend_out=False)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.show()


def make_figure_1_pilot_data_domains(experiments_by_subject, participants, ids, data):
    df = {'choice probability': [], 'frequency': [], 'domain': [], 'subject': [], 'subjectId': []}
    for i in range(len(participants)):
        stimulus_domains = experiments_by_subject[participants[i]]
        for j in range(len(stimulus_domains)):
            contents = data['{},{}'.format(participants[i], stimulus_domains[j])]  # read in file as a dict
            counts = np.array([])
            for trial in contents:
                pairwise_decisions = list(util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial),
                                                                               contents[trial]).values()
                                          )
                counts = np.concatenate((counts, np.array(pairwise_decisions)))
            probs = counts / 5.0
            unique, freq = np.unique(probs, return_counts=True)
            freq = freq / float(sum(freq))
            df['choice probability'] += list(unique)
            df['frequency'] += list(freq)
            df['domain'] += [stimulus_domains[j] for _ in range(6)]
            df['subject'] += [participants[i] for _ in range(6)]
            df['subjectId'] += [ids[participants[i]] for _ in range(6)]
    df = pd.DataFrame(df)
    sns.catplot(x="choice probability", y="frequency", col='subjectId',
                kind="bar", hue="domain",
                data=df, legend_out=False)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.show()


def raw_choice_prob_heatmap(trial_by_trial_choice_probs, subject_names, nrow, ncol):
    # will work for one or two long rows. If you want to pass in an arbitrary sized grid, the  colormap
    # will start getting blocked. To fix, see what is different in the other heatmap function below.
    pairs_of_subjects = list(combinations(subject_names, 2))
    f, axes = plt.subplots(ncol, nrow)
    f.tight_layout(pad=1)
    cbar_ax = f.add_axes([.3, .18, .35, .03])
    cbar_kws = {'label': 'proportion of all pairwise choices',
                'orientation': 'horizontal'}
    for n in range(len(pairs_of_subjects)):
        subject1, subject2 = pairs_of_subjects[n]
        # generate heatmap (count in sub 1 vs count in sub2 - occurrence)
        heatmap = np.zeros((6, 6))
        for i in range(len(trial_by_trial_choice_probs[subject1])):
            heatmap[trial_by_trial_choice_probs[subject1][i], trial_by_trial_choice_probs[subject2][i]] += 1
        heatmap = np.round(heatmap / float(sum(sum(heatmap))), 3)

        sns.heatmap(heatmap, square=True,
                    xticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1], annot=False,
                    yticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1], cmap="RdYlBu_r",
                    ax=axes[n], cbar_ax=cbar_ax, cbar_kws=cbar_kws)
        bottom, top = axes[n].get_ylim()
        axes[n].set_ylim(bottom + 0.5, top - 0.5)
        axes[n].set_xlabel('{}'.format(subject1))
        axes[n].set_ylabel('{}'.format(subject2))
        axes[n].invert_yaxis()
    plt.show()


def choice_prob_ratio_heatmap(trialwise_choice_probabilities, subject_names, nrow, ncol):
    pairs_of_subjects = list(combinations(subject_names, 2))
    f, axes = plt.subplots(ncol, nrow)
    cbar_kws = {'label': '', 'orientation': 'vertical', 'shrink': 0.3}
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

        current_axis = plt.subplot(ncol, nrow, n + 1)
        g = sns.heatmap(ratio_map, square=True,
                        xticklabels=[0, None, None, None, None, 1], annot=False,
                        yticklabels=[0, None, None, None, None, 1], cmap="RdYlBu_r",
                        ax=current_axis, vmin=0.25, vmax=2.5, cbar_kws=cbar_kws)
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


def get_choice_probs_by_subject(subject_names, exp, data):
    choice_probs = {}
    for subject_name in subject_names:
        choice_probs[subject_name] = {}
        ranking_trials = data['{},{}'.format(subject_name, exp)]
        for trial in ranking_trials:
            choice_probs[subject_name][trial] = util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial),
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


def make_figure_2a_heatmap(subject_names, exp, data, nrow, ncol):
    choice_probs = get_choice_probs_by_subject(subject_names, exp, data)
    trialwise_choice_probabilities = arrange_choice_probs_by_comparison(subject_names, choice_probs)
    raw_choice_prob_heatmap(trialwise_choice_probabilities, subject_names, nrow, ncol)


def make_figure_2b_ratio_heatmap(subject_names, exp, data, nrow, ncol):
    choice_probs = get_choice_probs_by_subject(subject_names, exp, data)
    trialwise_choice_probabilities = arrange_choice_probs_by_comparison(subject_names, choice_probs)
    choice_prob_ratio_heatmap(trialwise_choice_probabilities, subject_names, nrow, ncol)


def predicted_choice_probabilities(choice_probabilities, points, sigmas):
    """ Takes in the pairwise choice probabilities in terms of point indices (not animal names) and returns
    the predicted choice probabilities based on the points and provided noise parameter

    keys are of the form  ((0, 2), '>', (0, 1)), while by default the original choice probs tend to be like '0,1<0,2'"""
    model_probabilities = {}
    distance_matrix = squareform(pdist(points))
    for choice, prob in choice_probabilities.items():
        pair1 = choice[0]
        pair2 = choice[-1]
        model_probabilities[choice] = prob_a_greater_than_b(pair1, pair2, distance_matrix, sigmas)
    return model_probabilities


def matched_actual_and_predicted_prob_values(subject, domain, sigma_dist):
    filepath = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/simulations/euclidean/' \
               'analysis_real_data/{}/{}_{}_anchored_points_sigma_{}_dim_5.npy'.format(subject, subject, domain,
                                                                                       sigma_dist)
    filepath_alternative = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/simulations/euclidean/' \
                           'analysis_real_data/{}/{}_{}_points_sigma_{}_dim_5.npy'.format(subject, subject, domain,
                                                                                          sigma_dist)
    if not path.exists(filepath):
        filepath = filepath_alternative
    # if dataset for sub and domain exists, go ahead and plot a heatmap
    points = util.read_npy(filepath)
    contents = util.read_file(subject, domain)
    choice_probs = util.json_to_choice_probabilities(contents, True)
    # get model predicted choice probabilities
    predicted_probs = predicted_choice_probabilities(choice_probs, points, {'compare': 0, 'dist': sigma_dist})
    return matched_prob_values(choice_probs, predicted_probs)


def matched_prob_values(choice_prob_dict1, choice_prob_dict2):
    probs1 = []
    probs2 = []
    # find common keys (comparisons)
    common_keys = set(choice_prob_dict1.keys()).intersection(set(choice_prob_dict2.keys()))
    print(len(common_keys))
    # populate comparisonwise actual and predicted probs
    for key in common_keys:
        probs1.append(choice_prob_dict1[key])
        probs2.append(choice_prob_dict2[key])
    return probs1, probs2


def mix_distributions(p, q):
    """p and q are two numpy arrays, with probability values"""
    return (p+q)/2.0


def kldivergence(p, q):
    """ Given p and q (size usually 5994) in which for each value in p is a prob of making a choice A,B vs A,C
    compute the mean KL divergence across all 5994 events. Each event has its prob distribution = [p, 1-p]
    """
    divergence = 0
    num_events = len(p)
    for i in range(num_events):
        if p[i] == 0:
            divergence += (1-p[i]) * np.log2((1-p[i])/(1-q[i]))
        elif p[i] == 1:
            divergence += p[i] * np.log2(p[i]/q[i])
        else:
            divergence += p[i] * np.log2(p[i]/q[i]) + (1-p[i]) * np.log2((1-p[i])/(1-q[i]))
    divergence = divergence / float(num_events)
    return divergence


def jensenshannon(p, q):
    m = mix_distributions(p, q)
    return (kldivergence(p, m) + kldivergence(q, m))/2.0


def scatterplot_predicted_vs_actual_choice_probs(subject, domain, sigma_dist=0.18):
    f, ax = plt.subplots(1, 1)
    f.tight_layout(pad=1)
    actual, predicted = matched_actual_and_predicted_prob_values(subject, domain, sigma_dist)
    print(min(predicted))
    print(max(predicted))
    # bin predicted probs into 20 bins, e.g. 0-0.05, 0.05-0.1 etc. where 0.05 would go into the second bin.
    heatmap = np.zeros((21, 6))
    for i in range(len(actual)):
        heatmap[floor(predicted[i] / 0.05), int(actual[i])] += 1
    heatmap = np.round(heatmap / float(sum(sum(heatmap))), 3)
    g = sns.heatmap(heatmap,
                    xticklabels=[0, 0.20, 0.40, 0.60, 0.80, 1], annot=False,
                    yticklabels=[round(j * 0.05, 2) for j in range(0, 21)], cmap="RdYlBu_r",
                    ax=ax)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(labelsize=8)
    ax.set_title('{}: {}'.format(subject, domain), fontsize=7)
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    plt.show()


def make_figure_ambiguity_index(subjects_by_experiment, stimulus_domains, ids, domain_names, data):
    df = {'ambiguity index': [], 'subject': [], 'subjectId': [], 'domain': []}
    for j in range(len(stimulus_domains)):
        domain = stimulus_domains[j]
        subject_names = subjects_by_experiment[domain]
        for i in range(len(subject_names)):
            contents = data['{},{}'.format(subject_names[i], domain)]  # read in data file contents
            counts = np.array([])
            for trial in contents:
                pairwise_decisions = list(util.ranking_to_pairwise_comparisons(util.all_distance_pairs(trial),
                                                                               contents[trial]).values())
                counts = np.concatenate((counts, np.array(pairwise_decisions)))
            probs = counts / 5.0
            unique, freq = np.unique(probs, return_counts=True)
            freq = freq / float(sum(freq))
            probs = list(unique)
            # %age of choices with probs that were not 0 or 1
            df['ambiguity index'].append(sum(freq[2:-2])*100)
            df['domain'].append(domain_names[j])
            df['subject'].append(subject_names[i])
            df['subjectId'].append(ids[subject_names[i]])
    df = pd.DataFrame(df)
    # sns.catplot(x="choice probability", y="frequency", col="domain",
    #             kind="bar", hue="certainty",
    #             data=df, legend_out=False)
    # plt.yticks([0, 0.1, 0.2, 0.3, 0.4])

    sns.catplot(x="subjectId", y="ambiguity index",
                kind="bar", hue="domain",
                data=df, legend_out=False)
    plt.show()


if __name__ == '__main__':
    SUBJECTIDS = {"MC": "MC", "BL": "S2", "EFV": "S3", "SJ": "S4", "SAW": "SAW", "NK": "S6", "YCL": "YCL",
                  "SA": "S8", "JF": "S9"}
    SUBJECTS_BY_DOMAIN = {
        'texture': ["MC", "SAW", "YCL", "BL", "JF"],
        'intermediate_texture': ["MC", "YCL", "SAW", "SJ", "BL", "SA", "EFV"],
        'intermediate_object': ["MC", "YCL", "SAW", "SJ", "NK", "BL", "EFV"],
        'image': ["MC", "YCL", "NK", "EFV"],
        'word': ["MC", "YCL", "SAW", "SJ", "JF", "BL", "SA"], #, "EFV"],
        # 'texture_grayscale': ["MC", "SAW", "SA"], 'texture_color': ["SAW", "NK", "MC"]
    }
    DOMAINS = ["texture", "intermediate_texture", "intermediate_object", "image", "word"]
    DOMAINS_SHORT = ["texture", "texture-like", "image-like", "image", "word"]
    SUBJECTS = ["MC", "BL", "EFV", "SJ", "SAW", "NK", "YCL", "JF", "SA"]
    ALL_DOMAINS_BY_SUBJECT = {
        "MC": ["texture", "intermediate_texture", "intermediate_object", "image", "word"],
        "BL": ["texture", "intermediate_texture", "intermediate_object", "word"],
        "EFV": ["intermediate_texture", "intermediate_object", "image", "word"],
        "SJ": ["intermediate_texture", "intermediate_object", "word"],
        "SA": ["intermediate_texture", "word"],
        "SAW": ["texture", "intermediate_texture", "intermediate_object", "word"],
        "NK": ["intermediate_object", 'image'],
        "YCL": ["texture", "intermediate_texture", "intermediate_object", "image", "word"],
        "JF": ["texture", "word"]
    }
    DOMAINS_BY_SUBJECT = {
        "MC": ["texture", "intermediate_texture", "intermediate_object", "image", "word", ],
        "BL": ["texture", "intermediate_texture", "word"], "EFV": ["intermediate_texture", "intermediate_object"],
        "SJ": ["intermediate_texture"],
        "SAW": ["texture", "intermediate_texture", "word"],
        "YCL": ["texture", "intermediate_texture", "intermediate_object", "image", "word"],
        "JF": ["texture", "word"],
        "NK": ["intermediate_object", "image"],
        "SA": []
    }

    ALL_DATA = read_all_files(SUBJECTS_BY_DOMAIN)
    make_figure_1_pilot_data(SUBJECTS_BY_DOMAIN, ['word'], SUBJECTIDS, ALL_DATA)
    make_figure_ambiguity_index(SUBJECTS_BY_DOMAIN, DOMAINS, SUBJECTIDS, DOMAINS_SHORT, ALL_DATA)
    # make_figure_2a_heatmap(["SAW", "YCL", "BL", "MC"], "word", ALL_DATA, 6, 1)
    # make_figure_2b_ratio_heatmap(SUBJECTS_BY_DOMAIN["word"], "word", ALL_DATA, 5, 2)
    # make_figure_2b_ratio_heatmap(SUBJECTS_BY_DOMAIN["word"], "word", ALL_DATA, 3, 1)
    # make_figure_2b_ratio_heatmap(SUBJECTS_BY_DOMAIN["intermediate_object"], "intermediate_object", ALL_DATA, 3, 1)
    # make_figure_2b_ratio_heatmap(["SAW", "MC"], "texture", ALL_DATA, 1, 1)
    # make_figure_2b_ratio_heatmap(["YCL", "JF", "BL"], "texture", ALL_DATA, 3, 1)
    # make_figure_2b_ratio_heatmap(SUBJECTS_BY_DOMAIN["texture_grayscale"], "texture_grayscale", ALL_DATA, 1, 1)
    # make_figure_2b_ratio_heatmap(SUBJECTS_BY_DOMAIN["texture_color"], "texture_color", ALL_DATA, 1, 1)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['word'], 'word', ALL_DATA, annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['image'], 'image', ALL_DATA, annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['intermediate_object'], 'intermediate_object', ALL_DATA,
    #                              annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['intermediate_texture'], 'intermediate_texture', ALL_DATA,
    #                              annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['texture'], 'texture', ALL_DATA, annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['texture_grayscale'], 'texture_grayscale', ALL_DATA,
    # annotate=False)
    # make_fig_2c_ratio_pilot_data(SUBJECTS_BY_DOMAIN['texture_color'], 'texture_color', ALL_DATA, annotate=False)

    # # KL Divergence
    # sigma_dist = 0.18
    # domain_label = {'texture': 'tex', 'intermediate_texture': 'tex*', 'intermediate_object': 'im*', 'image': 'im',
    #                 'word': 'word'}
    # df = {'subject': [], 'KL divergence/ number comparisons': [], 'domain': []}
    # for subject in SUBJECTS:
    #     for domain in DOMAINS_BY_SUBJECT[subject]:
    #         if subject == "SAW" and domain == "texture_color":
    #             sigma_dist = 0.16
    #         actual, predicted = matched_actual_and_predicted_prob_values(subject, domain, sigma_dist)
    #         per_comparison_divergence = kldivergence(actual, predicted)
    #         df['subject'].append(SUBJECTIDS[subject])
    #         df['KL divergence/ number comparisons'].append(per_comparison_divergence)
    #         df['domain'].append(domain_label[domain])
    #         sigma_dist = 0.18
    #
    # df = pd.DataFrame(df)
    # print(df)
    # b = sns.catplot(data=df, col='subject', x='domain', y='KL divergence/ number comparisons', hue='domain',
    #                 kind='bar')
    # plt.show()

    sigma_dist = 0.18
    domain_label = {'texture': 'texture', 'intermediate_texture': 'texture*', 'intermediate_object': 'image*',
                    'image': 'image',
                    'word': 'word', 'texture_color': 'color', 'texture_grayscale': 'tex_gray'}
    for subject in ["MC", "YCL", "SAW", "EFV", "BL", "NK"]:
        num_domains = len(ALL_DOMAINS_BY_SUBJECT[subject])
        js_distance_matrix = np.zeros((num_domains, num_domains))
        for i in range(num_domains):
            j = 0
            choice_probs_i = util.json_to_choice_probabilities(util.read_file(subject,
                                                                              ALL_DOMAINS_BY_SUBJECT[subject][i]))
            while j <= i:
                choice_probs_j = util.json_to_choice_probabilities(util.read_file(subject,
                                                                                  ALL_DOMAINS_BY_SUBJECT[subject][j]))
                prob_array_i, prob_array_j = matched_prob_values(choice_probs_i, choice_probs_j)
                per_comparison_divergence = jensenshannon(np.array(prob_array_i), np.array(prob_array_j))
                js_distance_matrix[i, j] = per_comparison_divergence
                j += 1

        g = sns.heatmap(js_distance_matrix, xticklabels=[domain_label[k] for k in ALL_DOMAINS_BY_SUBJECT[subject]],
                        square=True,
                        yticklabels=[domain_label[k] for k in ALL_DOMAINS_BY_SUBJECT[subject]], cmap="RdYlBu_r")
        bottom, top = g.axes.get_ylim()
        g.axes.set_ylim(bottom + 0.5, top - 0.5)
        g.axes.tick_params(labelsize=8)
        g.axes.set_title('{}'.format(subject), fontsize=7)
        g.axes.invert_yaxis()
        plt.show()

    # J-S divergence
    sigma_dist = 0.18
    domain_label = {'texture': 'texture', 'intermediate_texture': 'texture*', 'intermediate_object': 'image*',
                    'image': 'image',
                    'word': 'word', 'texture_color': 'color', 'texture_grayscale': 'tex_gray'}
    for domain in ['word', 'image', 'intermediate_texture', 'intermediate_object', 'texture']:
        size = len(SUBJECTS_BY_DOMAIN[domain])
        js_distance_matrix = np.zeros((size, size))
        for i in range(size):
            j = 0
            choice_probs_i = util.json_to_choice_probabilities(util.read_file(SUBJECTS_BY_DOMAIN[domain][i],
                                                                              domain))
            while j <= i:
                choice_probs_j = util.json_to_choice_probabilities(util.read_file(SUBJECTS_BY_DOMAIN[domain][j],
                                                                                  domain))
                prob_array_i, prob_array_j = matched_prob_values(choice_probs_i, choice_probs_j)
                per_comparison_divergence = jensenshannon(np.array(prob_array_i), np.array(prob_array_j))
                print(per_comparison_divergence)
                js_distance_matrix[i, j] = per_comparison_divergence
                js_distance_matrix[j, i] = per_comparison_divergence
                j += 1

        g = sns.heatmap(js_distance_matrix, xticklabels=SUBJECTS_BY_DOMAIN[domain],
                        square=True,
                        yticklabels=SUBJECTS_BY_DOMAIN[domain], cmap="RdYlBu_r",
                        vmin=0, vmax=0.32)
        bottom, top = g.axes.get_ylim()
        g.axes.set_ylim(bottom + 0.5, top - 0.5)
        g.axes.tick_params(labelsize=8)
        g.axes.set_title('{}'.format(domain), fontsize=7)
        g.axes.invert_yaxis()
        plt.show()

