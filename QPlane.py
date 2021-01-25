from QLearn import QLearn
from QPlaneEnv import QPlaneEnv

# [lat, long, elev, pitch, roll, yaw, gear] -998->NO CHANGE
flight_origin = [37.524, -122.06899,  6000, 0, 0, 0, 1] # Palo Alto
flight_destinaion = [37.505, -121.843611, 6000, -998, -998, -998, 1] # Sunol Valley

n_epochs = 100  # Number of generations
n_steps  = 500  # Number of inputs per generation
n_actions = 6  # Number of possible inputs to choose from
end = 50  # End parameter

n_states = 240
n_actions = 7
gamma = 0.95
lr = 0.01
epsilon = 0.10

env = QPlaneEnv(flight_origin, flight_destinaion, n_actions, end)
Q = QLearn(n_states, n_actions, gamma, lr, epsilon)


def log(i_epoch, i_step, reward, state, actions_binary, observation, control):
    print("\t\tGame ", i_epoch)
    print("\t\t\tMove ", i_step)
    print("\t\t\tState ", state)
    print("\t\t\t                 [p+,p-,ro+,ro-,ru+,ru-,n]")
    print("\t\t\tactions_binary = ", actions_binary)
    print("\t\t\tCurrent Orientation: ", observation[3:6])
    print("\t\t\tCurrent Control:", control)
    print("\t\t\tCurrent Reward: ", reward)


def step(i_step, done, reward, oldObservation):
    oldState = env.get_state_from_observation(oldObservation)
    action = Q.select_action(oldState, i_epoch, n_epochs)
    newObservation, actions_binary, control = env.update(action, reward, oldObservation)
    reward, done = env.step(action, oldObservation, newObservation)
    newState = env.get_state_from_observation(newObservation)
    Q.learn(oldState, action, reward, newState)
    oldObservation = newObservation
    return done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward


def epoch(i_epoch):
    oldObservation = env.reset(env.starting_position)
    done = False
    reward = 0
    for i_step in range(n_steps):
        done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward = step(i_step, done, reward, oldObservation)
        log(i_epoch, i_step, reward, oldState, actions_binary, oldObservation, control)
        if done:
            break


for i_epoch in range(n_epochs):
    epoch(i_epoch)

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")
