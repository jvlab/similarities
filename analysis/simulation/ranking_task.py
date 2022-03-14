"""
From a list of 37, create a set of trials and output them in a csv file where each row is a trial
configuration.

Each trial has 9 stimuli, 1 of which is called the reference.
Each of the 37 stimuli is a reference in 6 separate trials. This way it can appear with all of the 36 other
stimuli once.

Trial order is randomized
"""
import random


# helper
def create_subsets(stimuli, window=8, overlap=2):
    """
    Given a list of words, group the words into 8 groups of 6, randomly.
    :param window: (int) defines the length of the sliding window to use when making groups
    :param overlap (int) defines how many stimuli appear together in two trials
    :param stimuli: (List) Words (stimulus names) of length 36
    :return: subsets: (List of lists) a list of 6 lists of length 6
    """
    num_stimuli = len(stimuli)
    if not (num_stimuli % 6 == 0):
        raise Exception('Number of stimuli needs to be valid: 1 + a multiple of 6 (19, 25, 31, 37, 49)')
    i = 0
    subsets = []
    while i < len(stimuli) - window:
        subset = stimuli[i:i + window]
        random.shuffle(subset)  # shuffling so that stimuli are grouped randomly into trials
        subsets.append(tuple(subset))
        i += window - overlap
    last = stimuli[i:] + stimuli[:overlap]
    random.shuffle(last)  # shuffling so that the last trial contains stimuli in a random order
    subsets.append(tuple(last))
    return subsets


def create_trials(num_stimuli=37):
    """
    Given a list of words, create all trial configurations for all num_stimuli references.
    For each of the words,do the following:
    choose word as the reference. Randomly group the remaining 36 words into 6 sets of 8.
    The reference together with the 8 stimuli (ref, (stim1,stim2,stim3,stim4,stim5,stim6,stim7,stim8)) form one trial
    Return all trials.
    :return: trial_sets: (Dict<key:int, value:list>) Trials grouped by reference stimulus
    """
    stimuli = list(range(num_stimuli))  # use indices not the actual stimulus coordinates
    trial_sets = []
    for i in range(len(stimuli)):
        remaining_stimuli = stimuli[:i] + stimuli[i + 1:]
        random.shuffle(remaining_stimuli)
        group = create_subsets(remaining_stimuli)
        for circle in group:
            trial = stimuli[i], circle
            trial_sets.append(trial)
    return trial_sets
