
## perceptual-similarities

### Downloading dependencies:

#### Python3
The project uses Python 3.7.4. If you have problems running it with newer versions, try running it with Python 3.7.
The following tutorial can help with installing Python: https://realpython.com/installing-python/

#### PsychoPy3
Install PsychoPy here: https://www.psychopy.org/download.html. 
This project uses PsychoPy v2020.2.10. If there is an issue running the experiment, try with running it with v2020.2.10, which can be found here: https://github.com/psychopy/psychopy/releases.


### Create Trials
How to use trial_configurations to get trials. 


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


