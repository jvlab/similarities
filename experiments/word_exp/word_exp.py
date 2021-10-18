#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.1.2),
    on Tue Apr  7 19:26:36 2020
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

import os  # handy system and path functions
import time
import yaml
from psychopy.hardware import keyboard
from numpy import (sin, cos, pi, linspace)
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
from psychopy import gui, visual, core, data, event, logging

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.1.0'
expName = 'word_exp'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if not dlg.OK:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

with open('../config.yaml', "r") as stream:
    # reads in parameters defined by user.
    exp_config = yaml.safe_load(stream)


# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = '{}{}data/{}-{}-{}'.format(_thisDir,
                                      os.sep,
                                      expInfo['participant'],
                                      expName,
                                      expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='1.0.1',
                                 extraInfo=expInfo, runtimeInfo=None,
                                 dataFileName=filename)

# Results handler
thisExpResult = data.ExperimentHandler(name=expName, version='1.0.1',
                                       extraInfo=expInfo, runtimeInfo=None,
                                       dataFileName='{}_responses'.format(filename))

# save a log file for detail verbose info
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    fullscr=True, screen=0,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] is not None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instr"
instrClock = core.Clock()
text = visual.TextStim(win=win, name='text',
                       text='Please click the stimuli in the circle in order of similarity to the stimulus in the '
                            'center, the reference.\n\nAs you click, the stimuli will gray out.\nYou cannot undo '
                            'clicks.\n\nClick the stimulus most similar to the reference first and least similar '
                            'stimulus last.\n\nPress spacebar to continue to task',
                       font='Arial',
                       pos=(0, 0), height=0.04, wrapWidth=None, ori=0,
                       color='white', colorSpace='rgb', opacity=1,
                       languageStyle='LTR',
                       depth=0.0)
key_resp = keyboard.Keyboard()


# Initialize the locations for components - around a circle

def initialize_stim_components():
    num_stimuli_in_surround = exp_config['num_words_per_trial']
    stimulus_components = {'ref': None, 'circle': []}
    theta = linspace(0, 2 * pi, num_stimuli_in_surround+1, retstep=True)[0]  # evenly spaced
    rho = exp_config['word_display_radius']
    xpos = [0 for _ in range(num_stimuli_in_surround)]
    ypos = [0 for _ in range(num_stimuli_in_surround)]
    for k in range(num_stimuli_in_surround):
        _x = rho * cos(theta[k])
        _y = rho * sin(theta[k])
        xpos[k] = _x
        ypos[k] = _y
    for j in range(num_stimuli_in_surround):
        stimulus_components['circle'].append(
            visual.TextStim(win=win, name='stim{}'.format(str(j + 1)), text='',
                            font='Arial',
                            units='deg',
                            pos=(xpos[j], ypos[j]), height=exp_config['text_stim_height'], wrapWidth=None, ori=0,
                            color='white', colorSpace='rgb', opacity=1,
                            languageStyle='LTR',
                            depth=-float(j))
        )
    stimulus_components['ref'] = visual.TextStim(win=win, name='ref', text='',
                                                 font='Arial',
                                                 units='deg',
                                                 pos=(0, 0), height=exp_config['text_stim_height'], wrapWidth=None, ori=0,
                                                 color='white', colorSpace='rgb', opacity=1,
                                                 languageStyle='LTR',
                                                 depth=-8.0)
    return stimulus_components


# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# ------Prepare to start Routine "instr"-------
continueRoutine = True
# update component parameters for each repeat
key_resp.keys = []
key_resp.rt = []
_key_resp_allKeys = []
# keep track of which components have finished
instrComponents = [text, key_resp]
for thisComponent in instrComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instrClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr"-------
while continueRoutine:
    # get current time
    t = instrClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instrClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        text.setAutoDraw(True)

    # *key_resp* updates
    waitOnFlip = False
    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
        # keep track of start time/frame for later
        key_resp.frameNStart = frameN  # exact frame index
        key_resp.tStart = t  # local t and not account for scr refresh
        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
        key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp.status == STARTED and not waitOnFlip:
        theseKeys = key_resp.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_allKeys.extend(theseKeys)
        if len(_key_resp_allKeys):
            key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
            key_resp.rt = _key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False

    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr"-------
for thisComponent in instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text.started', text.tStartRefresh)
thisExp.addData('text.stopped', text.tStopRefresh)
# check responses
if key_resp.keys in ['', [], None]:  # No response was made
    key_resp.keys = None
thisExp.addData('key_resp.keys', key_resp.keys)
if key_resp.keys is not None:  # we had a response
    thisExp.addData('key_resp.rt', key_resp.rt)
thisExp.addData('key_resp.started', key_resp.tStartRefresh)
thisExp.addData('key_resp.stopped', key_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1, method='sequential',
                           extraInfo=expInfo, originPath=-1,
                           trialList=data.importConditions('conditions.csv'),
                           seed=None, name='trials')

# initialize text stimuli
stimulusComponents = initialize_stim_components()
circle = stimulusComponents['circle']
ref = stimulusComponents['ref']
ref_plus_circle = [ref] + circle

pageNum = visual.TextStim(win=win, name='page',
                          text='kjnk',
                          font='Arial',
                          units="deg",
                          pos=(20, -12), height=0.5, wrapWidth=None, ori=0,
                          color='white', colorSpace='rgb', opacity=1,
                          languageStyle='LTR',
                          depth=-10.0)

page_index = 0
for trial in trials:
    page_index += 1
    # Initialize components for Routine "trial"
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    trialClock = core.Clock()
    for ii in range(len(circle)):
        circle[ii].setText(trial['stim{}'.format(str(ii + 1))])
    ref.setText(trial['ref'])
    pageNum.setText(str(page_index))

    for stim in circle:
        stim.setColor('white')

    # prepare to collect responses for trials
    for key in trial:
        thisExpResult.addData(key, trial[key])

    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    trialComponents = [mouse, ref] + circle
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1

    # -------Run Routine "trial"-------
    numClicks = 0
    clicked_stimuli = []
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        pageNum.setAutoDraw(True)
        # update/draw components on each frame

        # *text stimuli 2-9 and ref* updates
        for text_i in ref_plus_circle:
            if text_i.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                # keep track of start time/frame for later
                text_i.frameNStart = frameN  # exact frame index
                text_i.tStart = t  # local t and not account for scr refresh
                text_i.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_i, 'tStartRefresh')  # time at next scr refresh
                text_i.setAutoDraw(True)

        # *mouse* updates
        if mouse.status == NOT_STARTED and t >= 0.0 - frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    for obj in circle:
                        if obj.contains(mouse):
                            if all(obj.color) == 1:
                                obj.setColor([0.1, 0.1, 0.25], 'rgb')
                                numClicks += 1
                                mouse.clicked_name.append(obj.name)
                                clicked_stimuli.append(obj.text)
                                if numClicks >= exp_config['num_words_per_trial'] and len(set(mouse.clicked_name)) == exp_config['num_words_per_trial']:
                                    # record trial click responses
                                    thisExpResult.addData('clicks', mouse.clicked_name)
                                    thisExpResult.addData('clicked_stimuli', clicked_stimuli)
                                    thisExpResult.nextEntry()
                                    gotValidClick = True
                    x, y = mouse.getPos()
                    mouse.x.append(x)
                    mouse.y.append(y)
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
                    if gotValidClick:  # abort routine on response
                        continueRoutine = False
                        x, y = mouse.getPos()
                        mouse.setPos((x - 0.08, y - 0.1))

        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.saveAsWideText(filename + '.csv')
            core.quit()

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store data for thisExp (ExperimentHandler)
    if len(mouse.x): thisExp.addData('mouse.x', mouse.x)
    if len(mouse.y): thisExp.addData('mouse.y', mouse.y)
    if len(mouse.leftButton): thisExp.addData('mouse.leftButton', mouse.leftButton[0])
    if len(mouse.midButton): thisExp.addData('mouse.midButton', mouse.midButton[0])
    if len(mouse.rightButton): thisExp.addData('mouse.rightButton', mouse.rightButton[0])
    if len(mouse.time): thisExp.addData('mouse.time', mouse.time[0])
    if len(mouse.clicked_name): thisExp.addData('mouse.clicked_name', mouse.clicked_name[0])
    thisExp.addData('mouse.started', mouse.tStart)
    thisExp.addData('mouse.stopped', mouse.tStop)
    for stim_index in range(len(circle)):
        thisExp.addData('stim{}.started'.format(stim_index + 1), circle[stim_index].tStartRefresh)
        thisExp.addData('stim{}.stopped'.format(stim_index + 1), circle[stim_index].tStopRefresh)
    thisExp.addData('stim_ref.started', ref.tStartRefresh)
    thisExp.addData('stim_ref.stopped', ref.tStopRefresh)
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    # Flip one final time so any remaining win.callOnFlip()
    # and win.timeOnFlip() tasks get executed before quitting
    win.flip()
    if page_index > 0:
        time.sleep(0.35)

# ----- last slide to say thanks and end --------------------------
goodbye = visual.TextStim(win=win, name='text',
                          text='Thank you for your time :)',
                          font='Arial',
                          pos=(0, 0), height=0.04, wrapWidth=None, ori=0,
                          color='white', colorSpace='rgb', opacity=1,
                          languageStyle='LTR',
                          depth=0.0)
goodbye.setAutoDraw(True)

logging.flush()
# show last slide for 4 seconds
win.flip()
time.sleep(2)  # wait 2 seconds before shutting experiment

# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
