from QLearn import QLearn
from QPlaneEnv import QPlaneEnv


n_epochs = 100  # Number of generations
n_steps  = 500  # Number of inputs per generation
n_actions = 7  # Number of possible inputs to choose from
end = 50  # End parameter

n_states = 240  # Number of states
gamma = 0.95  #
lr = 0.01  # Learning Rate
epsilon = 0.10  # Starting Epsilon Rate, affects the exploration probability. Will decay

dictObservation = {
    "lat":  0,
    "long": 1,
    "alt":  2,
    "pitch":3,
    "roll": 4,
    "yaw":  5,
    "gear": 6}
dictAction = {
    "pi+": 0,
    "pi-": 1,
    "ro+": 2,
    "ro-": 3,
    "ru+": 4,
    "ru-": 5,
    "no" : 6}

# -998->NO CHANGE 
flight_origin = [37.524, -122.06899,  6000, 0, 0, 0, 1] # Palo Alto
flight_destinaion = [37.505, -121.843611, 6000, -998, -998, -998, 1] # Sunol Valley

env = QPlaneEnv(flight_origin, flight_destinaion, n_actions, end)
Q = QLearn(n_states, n_actions, gamma, lr, epsilon)


# prints out all metrics
def log(i_epoch, i_step, reward, state, actions_binary, observation, control):
    print("\t\tGame ", i_epoch)
    print("\t\t\tMove ", i_step)
    print("\t\t\tState ", state)
    print("\t\t\t\t\t\t\t[p+,p-,ro+,ro-,ru+,ru-,n]")
    print("\t\t\tactions_binary = ", actions_binary)
    print("\t\t\tCurrent Orientation: ", observation[dictObservation["pitch"]:dictObservation["gear"]])
    print("\t\t\tCurrent Control:", control)
    print("\t\t\tCurrent Reward: ", reward)


# A single step(input), this will repeat n_steps times throughout a epoch
def step(i_step, done, reward, oldObservation):
    oldState = env.get_state_from_observation(oldObservation)
    action = Q.select_action(oldState, i_epoch, n_epochs)
    newObservation, actions_binary, control = env.update(action, reward, oldObservation)
    reward, done = env.step(action, oldObservation, newObservation)
    newState = env.get_state_from_observation(newObservation)
    Q.learn(oldState, action, reward, newState)
    oldObservation = newObservation
    return done, oldState, newState, action, actions_binary, oldObservation, newObservation, control, reward


# A epoch is one full run, from respawn/reset to the final step.
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
