import random


class QLearn():

    def __init__(self, n_stat, n_acts, *args, **kwargs):
        self.n_states = n_stat
        self.n_actions = n_acts
        self.id = "random"
        self.epsilon = 1
        self.currentTable = [0]
        self.numGPUs = "random"
        self.qTable = [0]

    # get action for current state
    def selectAction(self, state, episode, n_epochs):
        action = int(random.uniform(0, self.n_actions))
        explore = True

        self.currentTable = [0]

        return action, explore, self.epsilon

    # update q table
    def learn(self, *args, **kwargs):
        return

    def archive(self, *args, **kwargs):
        return

    def resetStateDepth(self):
        return
