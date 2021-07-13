import socket
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.QDoubleDeepLearn import QLearn  # can be QLearn, QDeepLearn or QDoubleDeepLearn
from src.environments.jsbsim.JSBSimEnv import Env  # can be jsbsim.JSBSimEnv or xplane.XPlaneEnv
from src.scenarios.deltaAttitudeControlScene import Scene  # can be deltaAttitudeControlScene, sparseAttitudeControlScene or cheatingAttitudeControlScene

experimentName = "Experiment"

dateTime = str(time.ctime(time.time()))
dateTime = dateTime.replace(":", "-")
dateTime = dateTime.replace(" ", "_")
experimentName = experimentName + "-" + dateTime

errors = 0.0  # counts everytime the UDP packages are lost on all retries

timeStart = time.time()  # used to measure time
timeEnd = time.time()  # used to measure time
logPeriod = 100  # every so many epochs the metrics will be printed into the console
savePeriod = 25  # every so many epochs the table/model will be saved to a file
pauseDelay = 0.01  # time an action is being applied to the environment
logDecimals = 0  # sets decimals for np.arrays to X for printing
np.set_printoptions(precision=logDecimals)  # sets decimals for np.arrays to X for printing

n_epochs = 50_000  # Number of generations
n_steps = 1_000  # Number of inputs per generation
n_actions = 4  # Number of possible inputs to choose from

n_states = 182  # Number of states for non-Deep QLearning
gamma = 0.95  # The discount rate - between 0 an 1!  if = 0 then no learning, ! The higher it is the more the new q will factor into the update of the q value
lr = 0.0001  # Learning Rate. Deep ~0.0001 / non-Deep ~0.01 - If LR is 0 then the Q value would not update. The higher the value the quicker the agent will adopt the NEW Q value. If lr = 1, the updated value would be exactly be the newly calculated q value, completely ignoring the previous one
epsilon = 1.0  # Starting Epsilon Rate, affects the exploration probability. Will decay
decayRate = 0.00001  # Rate at which epsilon will decay per step
epsilonMin = 0.1  # Minimum value at which epsilon will stop decaying
n_epochsBeforeDecay = 10  # number of games to be played before epsilon starts to decay

numOfInputs = 8  # Number of inputs fed to the model
stateDepth = 1  # Number of old observations kept for current state. State will consist of s(t) ... s(t_n)
minReplayMemSize = 1_000  # min size determines when the replay will start being used
replayMemSize = 100_000  # Max size for the replay buffer
batchSize = 256  # Batch size for the model
updateRate = 5  # update target model every so many episodes
startingOffset = 0  # is used if previous Results are loaded.

loadModel = False  # will load "model.h5" for tf if True (model.npy for non-Deep)
loadMemory = False  # will load "memory.pickle" if True
loadResults = False  # will load "results.npy" if True
jsbRender = False  # will send UDP data to flight gear for rendering if True
jsbRealTime = False  # will slow down the physics to portrait real time rendering
usePredefinedSeeds = False  # Sets seeds for tf, np and random for more replicable results (not fully replicable due to stochastic environments)
saveResultsToPlot = False  # Saves results to png in the experiment folder at runetime
saveForAutoReload = False  # Saves and overrides models, results and memory to the root

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
    "pitch": 0,
    "roll": 1,
    "velocityY": 2}

# -998->NO CHANGE
flightOrigin = [35.126, 126.809, 6000, 0, 0, 0, 1]  # Gwangju SK
flightDestinaion = [33.508, 126.487, 6000, -998, -998, -998, 1]  # Jeju SK
startingVelocity = -55
#  Other locations to use: Memmingen: [47.988, 10.240], Chicago: [41.976, -87.902]

flightStartPitch = 10  # Will be used as -value / 0 / value
flightStartRoll = 15  # Will be used as -value / 0 / value
flightStartVelocityY = 10  # Will be used as -value / 0 / value

flightStartRotation = [[-flightStartPitch, -flightStartRoll, -flightStartVelocityY],
                       [-flightStartPitch, 0, -flightStartVelocityY],
                       [-flightStartPitch, flightStartRoll, -flightStartVelocityY],
                       [0, -flightStartRoll, -0],
                       [0, 0, 0],
                       [0, flightStartRoll, 0],
                       [flightStartPitch, -flightStartRoll, flightStartVelocityY],
                       [flightStartPitch, 0, flightStartVelocityY],
                       [flightStartPitch, flightStartRoll, flightStartVelocityY]]

epochRewards = []
epochQs = []
movingRate = 3 * len(flightStartRotation)  # Number given in number * len(flightStartRotation)
movingEpRewards = {
    "epoch": [],
    "average": [],
    "minimum": [],
    "maximum": [],
    "averageQ": [],
    "epsilon": []}

fallbackState = [0] * numOfInputs  # Used in case of connection error to XPlane

# Will load previous results in case a experiment needs to be continued
if(loadResults):
    movingEpRewards = np.load("results.npy", allow_pickle=True).item()  # loads the file - .item() turns the loaded nparray back to a dict
    startingOffset = np.max(movingEpRewards["epoch"])  # loads the episode where it previously stopped
    epsilon = np.min(movingEpRewards["epsilon"])  # loads the epsilon where the previously experiment stopped
    n_epochsBeforeDecay = max(0, n_epochsBeforeDecay - startingOffset)  # sets n_epochsBeforeDecay to the according value - max makes it so it's not negative but 0

if(usePredefinedSeeds):
    np.random.seed(42)

Q = QLearn(n_states, n_actions, gamma, lr, epsilon,
           decayRate, epsilonMin, n_epochsBeforeDecay, experimentName, saveForAutoReload, loadModel, usePredefinedSeeds,
           loadMemory, numOfInputs, minReplayMemSize, replayMemSize, batchSize, updateRate, stateDepth)

scene = Scene(dictObservation, dictAction, n_actions, stateDepth)

env = Env(scene, flightOrigin, flightDestinaion, n_actions, usePredefinedSeeds,
          dictObservation, dictAction, dictRotation, startingVelocity, pauseDelay, Q.id, jsbRender, jsbRealTime)

# saving setup pre run
if not os.path.exists("./Experiments/" + experimentName):
    os.makedirs("./Experiments/" + experimentName)
    setup = f"{experimentName=}\n{Q.numGPUs=}\n{dateTime=}\nendTime=not yet defined - first save\n{Q.id=}\n{env.id=}\n{scene.id=}\n{pauseDelay=}\n{n_epochs=}\n"
    setup += f"{n_steps=}\n{n_actions=}\n{n_states=} - states for non deep\n{gamma=}\n{lr=}\n{epsilon=}\n{decayRate=}\n{epsilonMin=}\n{n_epochsBeforeDecay=}\n"
    setup += f"{numOfInputs=} - states for deep\n{minReplayMemSize=}\n{replayMemSize=}\n{batchSize=}\n{updateRate=}\n{loadModel=}\n{movingRate=}\n"
    print(setup, file=open("./Experiments/" + str(experimentName) + "/setup.out", 'w'))  # saves hyperparameters to the experiment folder


# prints out all metrics
def log(i_epoch, i_step, reward, logList):
    global timeStart  # Used to print time ellapsed between log calls
    global timeEnd  # Used to print time ellapsed between log calls

    state = logList[1]
    actions_binary = logList[3]
    observation = logList[4]
    control = logList[5]
    explore = logList[6]
    currentEpsilon = logList[7]
    if(Q.id == "deep" or Q.id == "doubleDeep"):
        depth = len(state)
        depth = "Depth " + str(depth)
        state = state[-1]
    else:
        depth = ""

    timeEnd = time.time()  # End timer here
    print("\t\tGame ", i_epoch,
          "\n\t\t\tMove ", i_step,
          "\n\t\t\tStarting Rotation ", flightStartRotation[i_epoch % len(flightStartRotation)],
          "\n\t\t\tTime taken ", timeEnd - timeStart,
          "\n\t\t\tState ", np.array(state).round(logDecimals), depth,
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

    # checking if state includes a NaN (happens in JSBSim sometimes)
    if(np.isnan(newState).any()):
        if(Q.id == "deep" or Q.id == "doubleDeep"):
            newState = fallbackState
        else:
            newState = 0
        reward = 0
        info = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0]
        dictErrors["step"] = "NaN in state"
        errors += 1
        done = True

    Q.learn(oldState, action, reward, newState, done)
    logList = [oldState, newState, action, actions_binary, newPosition, control, explore, currentEpsilon]
    oldState = newState
    return done, reward, logList, oldState


# A epoch is one full run, from respawn/reset to the final step.
def epoch(i_epoch):
    global errors
    epochReward = 0
    epochQ = 0
    for attempt in range(25):
        try:
            oldState = env.reset(env.startingPosition, flightStartRotation[i_epoch % len(flightStartRotation)])
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

    if(i_epoch % savePeriod == 0):
        Q.archive(i_epoch)

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

    epochRewards.append(epochReward)
    epochQs.append(epochQ)
    if(i_epoch % movingRate == 0):
        movingEpRewards["epoch"].append(i_epoch)
        averageReward = sum(epochRewards[-movingRate:]) / len(epochRewards[-movingRate:])
        movingEpRewards["average"].append(averageReward)
        movingEpRewards["minimum"].append(min(epochRewards[-movingRate:]))
        movingEpRewards["maximum"].append(max(epochRewards[-movingRate:]))
        averageQ = sum(epochQs[-movingRate:]) / len(epochQs[-movingRate:])
        movingEpRewards["averageQ"].append(averageQ)
        movingEpRewards["epsilon"].append(logList[7])


for i_epoch in range(startingOffset, startingOffset + n_epochs + 1):
    epoch(i_epoch)
    if(i_epoch % savePeriod == 0):
        np.save("./Experiments/" + str(experimentName) + "/results" + str(i_epoch) + ".npy", movingEpRewards)
        if(saveForAutoReload):
            np.save("results.npy", movingEpRewards)
        if(saveResultsToPlot):
            plt.plot(movingEpRewards['epoch'], movingEpRewards['average'], label="average rewards")
            plt.plot(movingEpRewards['epoch'], movingEpRewards['averageQ'], label="average Qs")
            plt.plot(movingEpRewards['epoch'], movingEpRewards['maximum'], label="max rewards")
            plt.plot(movingEpRewards['epoch'], movingEpRewards['minimum'], label="min rewards")
            plt.plot(movingEpRewards['epoch'], movingEpRewards['epsilon'], label="epsilon")
            plt.title("Results")
            plt.xlabel("episodes")
            plt.ylabel("reward")
            plt.legend(loc=4)
            plt.savefig("./Experiments/" + str(experimentName) + "/plot" + str(i_epoch) + ".png")
            plt.clf()

np.save("./Experiments/" + str(experimentName) + "/results_final.npy", movingEpRewards)

endTime = str(time.ctime(time.time()))

# saving setup post run
setup = f"{experimentName=}\n{Q.numGPUs=}\n{dateTime=}\n{endTime=}\n{Q.id=}\n{env.id=}\n{scene.id=}\n{pauseDelay=}\n{n_epochs=}\n"
setup += f"{n_steps=}\n{n_actions=}\n{n_states=} - states for non deep\n{gamma=}\n{lr=}\n{epsilon=}\n{decayRate=}\n{epsilonMin=}\n{n_epochsBeforeDecay=}\n"
setup += f"{numOfInputs=} - states for deep\n{minReplayMemSize=}\n{replayMemSize=}\n{batchSize=}\n{updateRate=}\n{loadModel=}\n{movingRate=}\n"
print(setup, file=open("./Experiments/" + str(experimentName) + "/setup.out", 'w'))  # saves hyperparameters to the experiment folder

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
