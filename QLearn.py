import numpy as np
import random
import os


class QLearn():

    def __init__(self, n_stat, n_acts, gamm, lr, eps, dec, min, epsDecay, expName, *args, **kwargs):
        self.n_states = n_stat
        self.n_actions = n_acts
        self.gamma = gamm
        self.learningRate = lr
        self.epsilon = eps
        self.decay = dec
        self.epsMin = min
        self.qTable = np.zeros([self.n_states, self.n_actions])
        self.n_epochsBeforeDecay = epsDecay
        self.experimentName = expName
        self.id = "regular"
        self.currentTable = []

    # get action for current state
    def selectAction(self, state, episode, n_epochs):
        explorationTreshold = random.uniform(0, 1)
        explore = False
        # Check if explore or explore with current epsilon vs random number between 0 and 1
        if explorationTreshold > self.epsilon:
            # explore, which means predicted action
            action = np.argmax(self.qTable[state, :])
        else:
            # Explore, which means random action
            action = int(random.uniform(0, self.n_actions))
            explore = True

        self.currentTable = self.qTable[state, :]

        # decay epsilon
        if(episode >= self.n_epochsBeforeDecay):
            if(self.epsilon > self.epsMin):  # decay the value
                self.epsilon = self.epsilon * (1 - self.decay)
            elif(self.epsilon < self.epsMin):  # if decayed too far set to min
                self.epsilon = self.epsMin

        return action, explore, self.epsilon

    # update q table
    def learn(self, state, action, reward, new_state, done):
        self.qTable[state, action] = (1 - self.learningRate) * self.qTable[state, action] + self.learningRate * (reward + self.gamma * np.max(self.qTable[new_state, :]))  # Bellman

    def archive(self, epoch):
        if not os.path.exists("./Experiments/" + self.experimentName):
            os.makedirs("./Experiments/" + self.experimentName)
        np.save("./Experiments/" + str(self.experimentName) + "/" + str(epoch) + ".npy", self.qTable)
