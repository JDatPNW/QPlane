import numpy as np
from QLearn import QLearn
from QPlaneEnv import QPlaneEnv
import socket

n_epochs = 500  # Number of generations
n_steps = 100  # Number of inputs per generation
n_actions = 7  # Number of possible inputs to choose from
end = 50  # End parameter

n_states = 240  # Number of states
gamma = 0.95  # The discount rate - between 0 an 1!  if = 0 then no learning, ! The higher it is the more the new q will factor into the update of the q value
lr = 0.1  # Learning Rate. If LR is 0 then the Q value would not update. The higher the value the quicker the agent will adopt the NEW Q value. If lr = 1, the updated value would be exactly be the newly calculated q value, completely ignoring the previous one
epsilon = 1.0  # Starting Epsilon Rate, affects the exploration probability. Will decay
decayRate = 0.0001  # Rate at which epsilon will decay per step
epsilonMin = 0.01  # Minimum value at which epsilon will stop decaying
n_epochsBeforeDecay = 31  # number of games to be played before epsilon starts to decay

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

# -998->NO CHANGE
flightOrigin = [35.126, 126.809, 6000, 0.5, 0, 0, 1]  # Gwangju SK
flightDestinaion = [33.508, 126.487, 6000, -998, -998, -998, 1]  # Jeju SK
startingVelocity = -55
#  Other locations to use: Memmingen: [47.988, 10.240], Chicago: [41.976, -87.902]

env = QPlaneEnv(flightOrigin, flightDestinaion, n_actions,
                end, dictObservation, dictAction, startingVelocity)
Q = QLearn(n_states, n_actions, gamma, lr, epsilon,
           decayRate, epsilonMin, n_epochsBeforeDecay)


# prints out all metrics
def log(i_epoch, i_step, reward, state, actions_binary, observation, control, explore, currentEpsilon):
    print("\t\tGame ", i_epoch,
          "\t\t\tMove ", i_step,
          "\n\t\t\tState ", state,
          "\n\t\t\t\t[p+,p-,ro+,ro-,ru+,ru-,n]",
          "\n\t\t\tactions_binary = ", actions_binary,
          "\n\t\t\tCurrent Control:", control,
          "\n\t\t\tCurrent Orientation: ",
          observation[dictObservation["pitch"]:dictObservation["gear"]],
          "\n\t\t\tCurrent AVE of QTable: ", np.average(Q.qTable),
          "\n\t\t\tExplored (Random): ", explore,
          "\n\t\t\tCurrent Epsilon: ", currentEpsilon,
          "\n\t\t\tCurrent Reward: ", reward,
          "\n\t\t\tError Code: ", dictErrors)


# A single step(input), this will repeat n_steps times throughout a epoch
def step(i_step, done, reward, oldObservation):
    oldState = env.getState(oldObservation)
    action, explore, currentEpsilon = Q.selectAction(
        oldState, i_epoch, n_epochs)

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            newObservation, actions_binary, control = env.update(
                action, reward, oldObservation)  # Part that gets checked
        except socket.error as socketError:  # the specific error for connections used by xpc
            # if it fils get error code/ reason for error
            dictErrors["update"] = socketError
            continue
        else:
            break
    else:  # if all 10 attempts fail
        newObservation, actions_binary, control = oldObservation,
        # set values to dummy values - do nothing
        [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, -998, -998, -998]

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            # Part that gets checked
            reward, done = env.step(action, oldObservation, newObservation)
        except socket.error as socketError:  # the specific error for connections used by xpc
            # if it fils get error code/ reason for error
            dictErrors["step"] = socketError
            continue
        else:
            break
    else:  # if all 10 attempts fail
        pass  # Error was in second loop

    newState = env.getState(newObservation)
    Q.learn(oldState, action, reward, newState)
    oldObservation = newObservation
    return done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, currentEpsilon


# A epoch is one full run, from respawn/reset to the final step.
def epoch(i_epoch):

    for attempt in range(25):
        try:
            # Part that gets checked
            oldObservation = env.reset(env.startingPosition)
        except socket.error as socketError:  # the specific error for connections used by xpc
            # if it fils get error code/ reason for error
            dictErrors["reset"] = socketError
            continue
        else:
            break
    else:  # if all 25 attempts fail
        pass  # Error was during reset

    done = False
    reward = 0
    for i_step in range(n_steps):
        done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, currentEpsilon = step(
            i_step, done, reward, oldObservation)
        log(i_epoch, i_step, reward, oldState,
            actions_binary, oldObservation, control, explore, currentEpsilon)
        dictErrors["reset"], dictErrors["update"], dictErrors["step"] = [0, 0, 0]
        if done:
            break


for i_epoch in range(n_epochs):
    epoch(i_epoch)

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
