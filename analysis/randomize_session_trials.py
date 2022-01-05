"""
A helper script to randomize trial order and stimulus position across trials.
It segments the list of trials into equally sized chunks so they can be broken into as many sessions as
needed to complete a full dataset.
It can be run with the NUM_REPEATS parameter to return randomized orders for sessions for all datasets
in one go.
"""
import random

FILEPATH = input('Path to trial_conditions file (e.g., '
                 '"./sample-materials/image-exp-materials/trial_conditions.csv"): ')
OUTDIR = input('Output directory for conditions files: ')
NUM_TRIALS_PER_SESSION = int(input('Number of trials per session (e.g., 111): '))
NUM_REPEATS = int(input('Number of times all trials will be repeated (e.g., 5): '))
trials_file = open(FILEPATH, 'r')

# read in the trial configuration
lines = trials_file.readlines()
trials = lines[1:]
# keep a copy that won't be mutated
trials_copy = trials.copy()
total_trials = len(trials)
# make sure all session conditions file are the same size (with same number of trials)
if total_trials % NUM_TRIALS_PER_SESSION != 0:
    raise ValueError("If total trials are not divisible by NUM_TRIALS_PER_SESSION, cannot proceed.\n"
                     "Each session should have same number of trials.")
header = lines[0]

# run separately for each time you would like to repeat the experiment
for repeat in range(NUM_REPEATS):
    # start from same copy of trials
    trials = trials_copy.copy()
    # start new file
    new_file = []
    # shuffle all the rows
    random.shuffle(trials)
    # shuffle stim positions in each line individually (shuffle columns other than the ref column)
    for line in trials:
        stimuli = line.split('\n')[0].split(',')  # parse into list
        ref = stimuli[0]  # do not shuffle ref
        ring = stimuli[1:]
        random.shuffle(ring)
        line = '{},{}\n'.format(ref, ','.join(ring))  # turn into string again
        new_file.append(line)

    # split file into sessions
    num_batches = total_trials/NUM_TRIALS_PER_SESSION
    for i in range(int(num_batches)):
        out = open('{}/conditions-repeat-{}-batch_{}.csv'.format(OUTDIR, repeat+1, i+1), 'w')
        out.write(header)
        out.writelines(new_file[i*NUM_TRIALS_PER_SESSION:(i+1)*NUM_TRIALS_PER_SESSION])
