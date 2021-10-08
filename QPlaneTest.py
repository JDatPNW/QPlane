import socket
import time
import numpy as np
import os
from src.algorithms.QDoubleDeepLearn import QLearn  # can be QLearn, QDeepLearn, QDoubleDeepLearn or RandomAgent
from src.environments.jsbsim.JSBSimEnv import Env  # can be jsbsim.JSBSimEnv or xplane.XPlaneEnv
from src.scenarios.deltaAttitudeControlScene import Scene  # can be deltaAttitudeControlScene, sparseAttitudeControlScene or cheatingAttitudeControlScene

errors = 0.0  # counts everytime the UDP packages are lost on all retries

experimentName = "Testing"

dateTime = str(time.ctime(time.time()))
dateTime = dateTime.replace(":", "-")
dateTime = dateTime.replace(" ", "_")
experimentName = experimentName + "-" + dateTime


timeStart = time.time()  # used to measure time
timeEnd = time.time()  # used to measure time
logPeriod = 100  # every so many epochs the metrics will be printed into the console
savePeriod = 25  # every so many epochs the table/model will be saved to a file
pauseDelay = 0.01  # time an action is being applied to the environment
logDecimals = 4  # sets decimals for np.arrays to X for printing
np.set_printoptions(precision=logDecimals)  # sets decimals for np.arrays to X for printing

n_epochs = 5  # Number of generations
n_steps = 2_500  # Number of inputs per generation
n_actions = 4  # Number of possible inputs to choose from

n_states = 182  # Number of states
gamma = 0.95  # The discount rate - between 0 an 1!  if = 0 then no learning, ! The higher it is the more the new q will factor into the update of the q value
lr = 0.0001  # Learning Rate. If LR is 0 then the Q value would not update. The higher the value the quicker the agent will adopt the NEW Q value. If lr = 1, the updated value would be exactly be the newly calculated q value, completely ignoring the previous one
epsilon = 0.0  # Starting Epsilon Rate, affects the exploration probability. Will decay
decayRate = 0.00001  # Rate at which epsilon will decay per step
epsilonMin = 0.1  # Minimum value at which epsilon will stop decaying
n_epochsBeforeDecay = 10  # number of games to be played before epsilon starts to decay

numOfInputs = 7  # Number of inputs fed to the model
stateDepth = 1  # Number of old observations kept for current state. State will consist of s(t) ... s(t_n)
minReplayMemSize = 1_000  # min size determines when the replay will start being used
replayMemSize = 100_000  # Max size for the replay buffer
batchSize = 256  # Batch size for the model
updateRate = 5  # update target model every so many episodes
startingOffset = 0  # is used if previous Results are loaded.

loadModel = True  # will load "model.h5" for tf if True (model.npy for non-Deep)
loadMemory = False  # will load "memory.pickle" if True
loadResults = False  # will load "results.npy" if True
jsbRender = True  # will send UDP data to flight gear for rendering if True
jsbRealTime = False  # will slow down the physics to portrait real time rendering
usePredefinedSeeds = False  # Sets seeds for tf, np and random for more replicable results (not fully replicable due to stochastic environments)
saveForAutoReload = False  # Saves and overrides models, results and memory to the root
plotTest = True  # Will plot roll, pitch and reward per episode

startingVelocity = 60
startingPitchRange = 10
startingRollRange = 15
randomDesiredState = True  # Set a new state to stabalize towards every episode
desiredPitchRange = 5
desiredRollRange = 5

rewardListSingleEpisode = []
pitchListSingleEpisode = []
rollListSingleEpisode = []

dictObservation = {
    "lat": 0,
    "long": 1,
    "alt": 2,
    "pitch": 3,
    "roll": 4,
    "yaw": 5,
    "gear": 6}
dictAction = {
    "pi+": 0,
    "pi-": 1,
    "ro+": 2,
    "ro-": 3,
    "ru+": 4,
    "ru-": 5,
    "no": 6}
dictErrors = {
    "reset": 0,
    "update": 0,
    "step": 0}
dictRotation = {
    "roll": 0,
    "pitch": 1,
    "yaw": 2,
    "northVelo": 3,
    "eastVelo": 4,
    "verticalVelo": 5}

# -998->NO CHANGE
flightOrigin = [35.126, 126.809, 6000, 0, 0, 0, 1]  # Gwangju SK
flightDestinaion = [33.508, 126.487, 6000, -998, -998, -998, 1]  # Jeju SK
#  Other locations to use: Memmingen: [47.988, 10.240], Chicago: [41.976, -87.902]

fallbackState = [0] * numOfInputs  # Used in case of connection error to XPlane
fallbackState = [tuple(fallbackState)]

# Will load previous results in case a experiment needs to be continued
if(loadResults):
    movingEpRewards = np.load("results.npy", allow_pickle=True).item()  # loads the file - .item() turns the loaded nparray back to a dict
    startingOffset = np.max(movingEpRewards["epoch"])  # loads the episode where it previously stopped
    epsilon = np.min(movingEpRewards["epsilon"])  # loads the epsilon where the previously experiment stopped
    n_epochsBeforeDecay = max(0, n_epochsBeforeDecay - startingOffset)  # sets n_epochsBeforeDecay to the according value - max makes it so it's not negative but 0

if(usePredefinedSeeds):
    np.random.seed(42)

Q = QLearn(n_states, n_actions, gamma, lr, epsilon,
           decayRate, epsilonMin, n_epochsBeforeDecay, "testing", saveForAutoReload, loadModel, usePredefinedSeeds,
           loadMemory, numOfInputs, minReplayMemSize, replayMemSize, batchSize, updateRate, stateDepth)

scene = Scene(dictObservation, dictAction, n_actions, stateDepth, startingVelocity, startingPitchRange, startingRollRange, usePredefinedSeeds, randomDesiredState, desiredPitchRange, desiredRollRange)

env = Env(scene, flightOrigin, flightDestinaion, n_actions, usePredefinedSeeds,
          dictObservation, dictAction, dictRotation, startingVelocity, pauseDelay, Q.id, jsbRender, jsbRealTime)

if not os.path.exists("./TestingResults/" + experimentName):
    os.makedirs("./TestingResults/" + experimentName)


# prints out all metrics
def log(i_epoch, i_step, reward, logList):
    global timeStart  # Used to print time ellapsed between log calls
    global timeEnd  # Used to print time ellapsed between log calls

    old_state = logList[0]
    new_state = logList[1]
    actions_binary = logList[3]
    observation = logList[4]
    control = logList[5]
    explore = logList[6]
    currentEpsilon = logList[7]
    if(Q.id == "deep" or Q.id == "doubleDeep"):
        depth = len(old_state)
        depth = "Depth " + str(depth)
        old_state = old_state[-1]
        new_state = new_state[-1]
    else:
        depth = ""

    timeEnd = time.time()  # End timer here
    print("\t\tGame ", i_epoch,
          "\n\t\t\tMove ", i_step,
          "\n\t\t\tStarting Rotation ", np.array(env.startingOrientation).round(logDecimals),
          "\n\t\t\tDestination Rotation ", env.desiredState,
          "\n\t\t\tTime taken ", timeEnd - timeStart,
          "\n\t\t\tOld State ", np.array(old_state).round(logDecimals), depth,
          "\n\t\t\tNew State ", np.array(new_state).round(logDecimals), depth,
          "\n\t\t\t\t\t[p+,p-,r+,r-]",
          "\n\t\t\tactions_binary = ", actions_binary,
          "\n\t\t\tCurrent Control:", control,
          "\n\t\t\tCurrent Qs:", Q.currentTable,
          "\n\t\t\tCurrent Orientation: ", np.array(observation[dictObservation["pitch"]:dictObservation["gear"]]).round(logDecimals),
          "\n\t\t\tCurrent AVE of QTable: ", np.average(Q.qTable),
          "\n\t\t\tExplored (Random): ", explore,
          "\n\t\t\tCurrent Epsilon: ", currentEpsilon,
          "\n\t\t\tCurrent Reward: ", reward,
          "\n\t\t\tError Percentage & Count: ", float(errors / (i_epoch * n_steps + i_step + 1)), ",", errors,
          "\n\t\t\tError Code: ", dictErrors, "\n")
    timeStart = time.time()  # Start timer here


# A single step(input), this will repeat n_steps times throughout a epoch
def step(i_step, done, reward, oldState):
    global errors
    global rewardListSingleEpisode
    global pitchListSingleEpisode
    global rollListSingleEpisode

    if(i_step == 0):
        rewardListSingleEpisode = []
        pitchListSingleEpisode = []
        rollListSingleEpisode = []

    if(Q.id == "deep" or Q.id == "doubleDeep"):
        oldState = list(oldState)
    action, explore, currentEpsilon = Q.selectAction(oldState, i_epoch, n_epochs)

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            newState, reward, done, info = env.step(action)
            if(i_step == n_steps):
                done = True  # mark done if episode is finished
        except socket.error as socketError:  # the specific error for connections used by xpc
            dictErrors["step"] = socketError
            continue
        else:
            break
    else:  # if all 10 attempts fail
        errors += 1
        if(Q.id == "deep" or Q.id == "doubleDeep"):
            newState = fallbackState
        else:
            newState = 0
        reward = 0
        done = False
        info = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0]
        pass  # Error was in second loop

    newPosition = info[0]
    actions_binary = info[1]
    control = info[2]

    logList = [oldState, newState, action, actions_binary, newPosition, control, explore, currentEpsilon]
    rewardListSingleEpisode.append(reward)
    pitchListSingleEpisode.append(newPosition[3])
    rollListSingleEpisode.append(newPosition[4])
    if(Q.id == "deep" or Q.id == "doubleDeep"):
        oldState = list(newState)
    else:
        oldState = newState
    return done, reward, logList, oldState


# A epoch is one full run, from respawn/reset to the final step.
def epoch(i_epoch):
    global errors
    global rewardListSingleEpisode
    global pitchListSingleEpisode
    global rollListSingleEpisode

    epochReward = 0
    epochQ = 0
    for attempt in range(25):
        try:
            oldState = env.reset()
        except socket.error as socketError:  # the specific error for connections used by xpc
            dictErrors["reset"] = socketError
            continue
        else:
            break
    else:  # if all 25 attempts fail
        if(Q.id == "deep" or Q.id == "doubleDeep"):
            oldState = fallbackState  # Error was during reset
        else:
            oldState = 0
        errors += 1

    done = False
    reward = 0

    for i_step in range(n_steps + 1):
        done, reward, logList, oldState = step(i_step, done, reward, oldState)
        epochReward += reward
        epochQ += np.argmax(Q.currentTable)
        if(i_step % logPeriod == 0):  # log every logPeriod steps
            log(i_epoch, i_step, reward, logList)

        dictErrors["reset"], dictErrors["update"], dictErrors["step"] = [0, 0, 0]

        if done:
            break
    if(plotTest):
        np.save("./TestingResults/" + str(experimentName) + "/rewards_ep" + str(i_epoch) + ".npy", rewardListSingleEpisode)
        np.save("./TestingResults/" + str(experimentName) + "/pitch_ep" + str(i_epoch) + ".npy", pitchListSingleEpisode)
        np.save("./TestingResults/" + str(experimentName) + "/roll_ep" + str(i_epoch) + ".npy", rollListSingleEpisode)


for i_epoch in range(n_epochs + 1):
    epoch(i_epoch)

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
