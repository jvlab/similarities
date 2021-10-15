
## Perceptual Similarities


### Downloading dependencies:

#### Python3
The project uses Python 3.7.4. If you have problems running it with newer versions, try running it with Python 3.7.
The following tutorial can help with installing Python: https://realpython.com/installing-python/

#### PsychoPy3
Install PsychoPy here: https://www.psychopy.org/download.html. 
This project uses PsychoPy v2020.2.10. If there is an issue running the experiment, try with running it with v2020.2.10, which can be found here: https://github.com/psychopy/psychopy/releases.


### Trials in an Experiment
In a typical experiment, there are a series of ranking trials. The analysis requires an  experiment to be repeated multiple times. Our standard procedure assumes 5 repeats. This way each trial ends up being performed 5 times.
A sample trial comprises a stimulus in the center, known as the 'reference', and 8 surrounding stimuli. The number of surrounding stimuli can vary and is controlled by the `num_stimuli_per_trial` parameter.
In each trial, the goal of the subject is to rank stimuli around the reference in order of similarity to the reference. In other words, they must click the most similar item first, then the second-most similar and so on until they have clicked all the surrounding stimuli.
Given a list of stimuli, we can generate randomized configurations of trials, in which each stimulus appears as the central reference and is compared to every other stimulus at least once.
For details on valid designs and the constraints that have to be met, see the preprint [link to be provided] (Section: Discussion).

#### Sample Trials
A trial from the image experiment with `num_stimuli_per_trial=8.` The default value is 8.
 ![ alt text for screen readers](./sample-materials/sample_image_trial_screenshot.png "Placeholder images as stimuli")

A sample trial from the word experiment with `num_stimuli_per_trial=14`
 ![ alt text for screen readers](./sample-materials/sample_word_trial_screenshot.png "Placeholder images as stimuli")


#### Create Trials
Here, we explain how to use trial_configurations to get trials.
The script `trial_configuration.py` takes in the following parameters from
`experiments/config.yaml`:

1. `num_stimuli`: Size of the stimulus set (default=37)
2. `num_stimuli_per_trial`: Number of stimuli appearing around a reference in each trial (default=8, see above)
3. `path_to_stimulus_list`: Path to the text file containing names of all stimuli, one per line.

Open `experiments/config.yaml` and set the values of these parameters to the desired values.

Then, run the script from the analysis subdirectory in perceptual-similarities:
```
$ cd ./analysis
$ python3 trial_configuration.py

$ ls *.csv
trial_conditions.csv
```
The `trial_conditions.csv` generated contains the subsets of stimuli that will appear in each trial. Each trial's information is in a separate row.
Positions along the circle along which surrounding stimuli appear are given by columns `stim1` to `stim8` (if `num_stimuli_per_trial=8), while the `ref` column indicates which stimulus appears in the center.
The `stim1` position is always to the right of the reference. `stim2` onwards run clockwise from `stim1.`

##### Randomize Trials
Next, we need to duplicate the conditions files for each repetition of the experiment. In the standard procedure, we break up the 222 trials generated into two sessions of 111 trials each. 
Thus, each repetition comprises two sessions. 

For each time an experiment is to be conducted/repeated do the following to randomize the trial order and stimulus position within trials:

*Add a row and column to randomize columns and rows by*
1. Open `trial_conditions.csv` in Microsoft Excel.
2. Insert a new row under the header row (Row 1), run the random command (=RAND()) to populate cells in all columns except the `ref` column.
3. Insert a new column after the last column, on the right side, and run the random command (=RAND()) to populate all cells in the column from rows 3 onward.

*Randomize stimulus position across trials (shuffling columns within rows)*
1. Run the =RAND() function in all the cells of Row 2 to generate new random numbers.
2. Excluding columns `ref` and the last random number column, select all rows from Row 2 onward.
3. In Excel, click **Home**, then **Sort & Filter**, then select **Custom Sort**.
4. Click **Options** button in the bottom right corner of the pop-up, then under Orientation, select **Sort left to right.**
5. In the table that pops up, under **Row**, make sure "Row 2" is selected and click **OK**.
6. If column values under `stim1` to `stim8` do not shuffle, perform the sorting again by clicking **Sort & Filter** in the toolbar, then selecting **Custom Sort** and clicking **OK**.

*Randomize trial order (shuffling rows)*
1. Run the =RAND() function in all the cells of the last column to generate new random numbers.
2. Select all rows starting from Row 3 onwards - include all columns.
3. As before, in the toolbar, click **Sort & Filter**, then **Custom Sort**, then the **Options** button in the bottom right.
4. Under Orientation, select **Sort top to bottom.**
5. In the table that appears, under **Column**, make sure the last column (in our case, Column J) is selected.
6. Click **OK**.
7. As before, if rows do not shuffle, perform sorting again by clicking **Sort & Filter** in the toolbar, then selecting **Custom Sort** and clicking **OK**.

*Save in two new files*
1. Create two new files.
2. Copy the header row (Row 1) into both files.
3. In the first file, copy and paste Rows 3-113, i.e., half of the trials.
4. In the second file, copy and paste Rows 114-224.
5. Save each file as `conditions.csv` in the appropriate directory (see Recommended Directory Structure below).

NOTE: The above breakdown of trials into conditions files may be different if performing this operation for a non-standard version of the experiment. With 37 stimuli, and 8 stimuli around the reference in each trial, we have 222 trials. Each session comprises 111 trials. 

### Recommended Directory Structure
#### subject-data
The subject-data directory should have two subdirectories for raw and preprocessed data and be organized as follows:
```
subject-data/
    raw/
        Subject1/
            repeat_1/
                DD-MM-YYYY/
                    conditions.csv
                    responses.csv
                    DD-MM_YYYY.log
                DD-MM-YYYY/
                    conditions.csv
                    responses.csv
                    DD-MM_YYYY.log
            repeat_2
            repeat_3
            repeat_4
            repeat_5
        Subject2/
    ...

    preprocessed/
        Subject1_exp.json
        Subject2_exp.json
        ...
```


### Scripts and Their Input Parameters
#### preprocess.py
This converts the raw csv files containing similarity judgments from a subject's complete dataset, and combines them into a single json file.
To run, navigate to the main directory. (All scripts should be run from this directory).

```
cd perceptual-similarities
python3 -m analysis.preprocess
```

*Input parameters:*
1. Path to subject-data directory *(string)*
2. Name of experiment *(string)*: this is used to name the output file
3. Subject IDs *(strings separated by spaces if more than one)*




### Reproducing Figures
To reproduce figures from our accompanying manuscript, do the  following:



# Product Name
> Short blurb about what your product does.

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

One to two paragraph statement about your product and what it does.

![](header.png)

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki