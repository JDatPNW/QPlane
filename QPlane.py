from QLearn import QLearn
from QPlaneEnv import QPlaneEnv
import socket

n_epochs = 100  # Number of generations
n_steps = 500  # Number of inputs per generation
n_actions = 7  # Number of possible inputs to choose from
end = 50  # End parameter

n_states = 240  # Number of states
gamma = 0.95  # The discount rate - between 0 an 1!  if = 0 then no learning, ! The higher it is the more the new q will factor into the update of the q value
lr = 0.1  # Learning Rate. If LR is 0 then the Q value would not update. The higher the value the quicker the agent will adopt the NEW Q value. If lr = 1, the updated value would be exactly be the newly calculated q value, completely ignoring the previous one
epsilon = 1.0  # Starting Epsilon Rate, affects the exploration probability. Will decay
decayRate = 0.00001  # Rate at which epsilon will decay per step
epsilonMin = 0.01  # Minimum value at which epsilon will stop decaying
n_epochsBeforeDecay = 15  # number of games to be played before epsilon starts to decay

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

# -998->NO CHANGE
flightOrigin = [35.126, 126.809, 6000, 0, 0, 0, 1]  # Gwangju SK
flightDestinaion = [33.508, 126.487, 6000, -998, -998, -998, 1]  # Jeju SK
#  Other locations to use: Memmingen: [47.988, 10.240], Chicago: [41.976, -87.902]

env = QPlaneEnv(flightOrigin, flightDestinaion, n_actions,
                end, dictObservation, dictAction)
Q = QLearn(n_states, n_actions, gamma, lr, epsilon,
           decayRate, epsilonMin, n_epochsBeforeDecay)


# prints out all metrics
def log(i_epoch, i_step, reward, state, actions_binary, observation, control, explore, connectionError, errorCode, currentEpsilon):
    print("\t\tGame ", i_epoch)
    print("\t\t\tMove ", i_step)
    print("\t\t\tState ", state)
    print("\t\t\t\t[p+,p-,ro+,ro-,ru+,ru-,n]")
    print("\t\t\tactions_binary = ", actions_binary)
    print("\t\t\tCurrent Control:", control)
    print("\t\t\tCurrent Orientation: ",
          observation[dictObservation["pitch"]:dictObservation["gear"]])
    print("\t\t\tExplored (Random): ", explore)
    print("\t\t\tCurrent Epsilon: ", currentEpsilon)
    print("\t\t\tCurrent Reward: ", reward)
    print("\t\t\tConnection Error?: ", connectionError)
    print("\t\t\tError Code: ", errorCode)


# A single step(input), this will repeat n_steps times throughout a epoch
def step(i_step, done, reward, oldObservation):
    oldState = env.getState(oldObservation)
    action, explore, currentEpsilon = Q.selectAction(
        oldState, i_epoch, n_epochs)
    connectionError = 0
    errorCode = 0

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            newObservation, actions_binary, control = env.update(
                action, reward, oldObservation)  # Part that gets checked
        except socket.error as socketError:  # the specific error for connections used by xpc
            errorCode = socketError  # if it fils get error code/ reason for error
            continue
        else:
            break
    else:  # if all 10 attempts fail
        connectionError = 1  # Error was in first loop
        newObservation, actions_binary, control = oldObservation, [
            0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -998, -998]  # set values to dummy values - do nothing

    # Check if connections can be established 10x
    for attempt in range(10):
        try:
            # Part that gets checked
            reward, done = env.step(action, oldObservation, newObservation)
        except socket.error as socketError:  # the specific error for connections used by xpc
            errorCode = socketError  # if it fils get error code/ reason for error
            continue
        else:
            break
    else:  # if all 10 attempts fail
        connectionError = 2  # Error was in second loop

    newState = env.getState(newObservation)
    Q.learn(oldState, action, reward, newState)
    oldObservation = newObservation
    return done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, connectionError, errorCode, currentEpsilon


# A epoch is one full run, from respawn/reset to the final step.
def epoch(i_epoch):
    oldObservation = env.reset(env.startingPosition)
    done = False
    reward = 0
    for i_step in range(n_steps):
        done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward, explore, connectionError, errorCode, currentEpsilon = step(
            i_step, done, reward, oldObservation)
        log(i_epoch, i_step, reward, oldState,
            actions_binary, oldObservation, control, explore, connectionError, errorCode, currentEpsilon)
        if done:
            break


for i_epoch in range(n_epochs):
    epoch(i_epoch)

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
