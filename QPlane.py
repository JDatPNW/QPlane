import socket
import time
import numpy as np
from QDeepLearn import QLearn  # can be QLearn or QDeepLearn
from QPlaneEnv import QPlaneEnv

# TODO: FORCED EXPLORATION??? ALL INPUTS ARE SET BY ME, NOT predicted
# SO ONE RUN IS ALL RIGHT, NEXT IS ALL DOWN, NEXT IS ALL LEFT AND SO ON??

experimentName = "NewFitDeep" + str(time.time())

errors = 0.0

timeStart = time.time()
timeEnd = time.time()
logPeriod = 10  # every so many epochs the metrics will be printed into the console
savePeriod = 25  # every so many epochs the table/model will be saved to a file
pauseDelay = 0.1

n_epochs = 5000  # Number of generations
n_steps = 250  # Number of inputs per generation
n_actions = 4  # Number of possible inputs to choose from
end = 50  # End parameter

n_states = 728  # Number of states
gamma = 0.95  # The discount rate - between 0 an 1!  if = 0 then no learning, ! The higher it is the more the new q will factor into the update of the q value
lr = 0.1  # Learning Rate. If LR is 0 then the Q value would not update. The higher the value the quicker the agent will adopt the NEW Q value. If lr = 1, the updated value would be exactly be the newly calculated q value, completely ignoring the previous one
epsilon = 1.0  # Starting Epsilon Rate, affects the exploration probability. Will decay
decayRate = 0.00001  # Rate at which epsilon will decay per step
epsilonMin = 0.1  # Minimum value at which epsilon will stop decaying
n_epochsBeforeDecay = 10  # number of games to be played before epsilon starts to decay

numOfInputs = 19  # Number of inputs fed to the model
minReplayMemSize = 1_000  # min size determines when the replay will start being used
replayMemSize = 100_000  # Max size for the replay buffer
batchSize = 256  # Batch size for the model
updateRate = 5  # update target model every so many steps

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

flightStartPitch = 25  # Will be used as -value / 0 / value
flightStartRoll = 45  # Will be used as -value / 0 / value
flightStartVelocityY = 25  # Will be used as -value / 0 / value

flightStartRotation = [[-flightStartPitch, -flightStartRoll, -flightStartVelocityY],
                       [-flightStartPitch, 0, 0],
                       [-flightStartPitch, flightStartRoll, flightStartVelocityY],
                       [0, -flightStartRoll, -flightStartVelocityY],
                       [0, 0, 0],
                       [0, flightStartRoll, flightStartVelocityY],
                       [flightStartPitch, -flightStartRoll, -flightStartVelocityY],
                       [flightStartPitch, 0, 0],
                       [flightStartPitch, flightStartRoll, flightStartVelocityY]]

epochRewards = []
epochQs = []
movingRate = 3 * len(flightStartRotation)  # Number given in number * len(flightStartRotation)
movingEpRewards = {
    "epoch": [],
    "average": [],
    "minimum": [],
    "maximum": [],
    "averageQ": []}

env = QPlaneEnv(flightOrigin, flightDestinaion, n_actions,
                end, dictObservation, dictAction, dictRotation, startingVelocity, pauseDelay)
Q = QLearn(n_states, n_actions, gamma, lr, epsilon,
           decayRate, epsilonMin, n_epochsBeforeDecay, experimentName, numOfInputs, minReplayMemSize, replayMemSize, batchSize, updateRate)

np.set_printoptions(precision=5)  # sets decimals for np.arrays to X for printing


# prints out all metrics
def log(i_epoch, i_step, reward, state, actions_binary, observation, control, explore, currentEpsilon):
    global timeStart  # Used to print time ellapsed between log calls
    global timeEnd  # Used to print time ellapsed between log calls

    timeEnd = time.time()  # End timer here
    print("\t\tGame ", i_epoch,
          "\n\t\t\tMove ", i_step,
          "\n\t\t\tTime taken ", timeEnd - timeStart,
          "\n\t\t\tState ", state,
          "\n\t\t\t\t[p+,p-,ro+,ro-,ru+,ru-,n]",
          "\n\t\t\tactions_binary = ", actions_binary,
          "\n\t\t\tCurrent Control:", control,
          "\n\t\t\tCurrent Qs:", Q.currentTable,
          "\n\t\t\tCurrent Orientation: ",
          observation[dictObservation["pitch"]:dictObservation["gear"]],
          "\n\t\t\tCurrent AVE of QTable: ", np.average(Q.qTable),
          "\n\t\t\tExplored (Random): ", explore,
          "\n\t\t\tCurrent Epsilon: ", currentEpsilon,
          "\n\t\t\tCurrent Reward: ", reward,
          "\n\t\t\tError Percentage: ", float(errors / (i_epoch * n_steps + i_step + 1)),
          "\n\t\t\tError Code: ", dictErrors)
    timeStart = time.time()  # Start timer here


# A single step(input), this will repeat n_steps times throughout a epoch
def step(i_step, done, reward, oldObservation):
    global errors
    if(Q.id == "regular"):
        oldState = env.getState(oldObservation)
    elif(Q.id == "deep"):
        for attempt in range(10):
            try:
                oldState = env.getDeepState(oldObservation)
            except socket.error as socketError:  # the specific error for connections used by xpc
                dictErrors["update"] = socketError
                continue
            else:
                break
        else:  # if all 10 attempts fail
            oldState = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            errors += 1

    action, explore, currentEpsilon = Q.selectAction(
        oldState, i_epoch, n_epochs)

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            newObservation, actions_binary, control = env.update(
                action, reward, oldObservation)  # Part that gets checked
        except socket.error as socketError:  # the specific error for connections used by xpc
            dictErrors["update"] = socketError
            continue
        else:
            break
    else:  # if all 10 attempts fail
        newObservation, actions_binary, control = oldObservation, [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, -998, -998, -998]  # set values to dummy values - do nothing
        errors += 1

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            reward, done = env.step(action, oldObservation, newObservation)
        except socket.error as socketError:  # the specific error for connections used by xpc
            dictErrors["step"] = socketError
            continue
        else:
            break
    else:  # if all 10 attempts fail
        errors += 1
        pass  # Error was in second loop

    if(Q.id == "regular"):
        newState = env.getState(newObservation)
    elif(Q.id == "deep"):
        for attempt in range(10):
            try:
                newState = env.getDeepState(newObservation)
            except socket.error as socketError:  # the specific error for connections used by xpc
                dictErrors["update"] = socketError
                continue
            else:
                break
        else:  # if all 10 attempts fail
            newState = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            errors += 1

    Q.learn(oldState, action, reward, newState, done)
    oldObservation = newObservation
    return done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, currentEpsilon,


# A epoch is one full run, from respawn/reset to the final step.
def epoch(i_epoch):
    global errors
    epochReward = 0
    epochQ = 0
    for attempt in range(25):
        try:
            oldObservation = env.reset(
                env.startingPosition, flightStartRotation[i_epoch % len(flightStartRotation)])
        except socket.error as socketError:  # the specific error for connections used by xpc
            dictErrors["reset"] = socketError
            continue
        else:
            break
    else:  # if all 25 attempts fail
        oldObservation = env.startingPosition  # Error was during reset
        errors += 1

    if(i_epoch % savePeriod == 0):
        Q.archive(i_epoch)

    done = False
    reward = 0

    for i_step in range(n_steps):
        done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, currentEpsilon = step(i_step, done, reward, oldObservation)
        epochReward += reward
        epochQ += np.argmax(Q.currentTable)
        if(i_step % logPeriod == 0):  # log every logPeriod steps
            log(i_epoch, i_step, reward, oldState,
                actions_binary, oldObservation, control, explore, currentEpsilon)

        dictErrors["reset"], dictErrors["update"], dictErrors["step"] = [0, 0, 0]

        if done:
            break

    epochRewards.append(epochReward)
    epochQs.append(epochReward)
    if(i_epoch % movingRate == 0):
        movingEpRewards["epoch"].append(i_epoch)
        averageReward = sum(epochRewards[-movingRate:]) / len(epochRewards[-movingRate:])
        movingEpRewards["average"].append(averageReward)
        movingEpRewards["minimum"].append(min(epochRewards[-movingRate:]))
        movingEpRewards["maximum"].append(max(epochRewards[-movingRate:]))
        averageQ = sum(epochQs[-movingRate:]) / len(epochQs[-movingRate:])
        movingEpRewards["averageQ"].append(averageQ)


for i_epoch in range(n_epochs + 1):
    epoch(i_epoch)
    if(i_epoch % savePeriod == 0):
        np.save("./Experiments/" + str(experimentName) + "/results" + str(i_epoch) + ".npy", movingEpRewards)

np.save("./Experiments/" + str(experimentName) + "/results_final.npy", movingEpRewards)

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
