import numpy as np
import random
import os
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


class QLearn():

    def __init__(self, n_stat, n_acts, gamm, lr, eps, dec, min, epsDecay, expName, inputs, minReplay, replay, batch, update):
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
        self.model = DQNAgent(inputs, self.n_actions, self.learningRate,
                              minReplay, replay, batch, self.gamma, update)
        self.id = "deep"
        self.currentTable = []

    # get action for current state
    def selectAction(self, state, episode, n_epochs):
        explorationTreshold = random.uniform(0, 1)
        explore = False
        # Check if explore or explore with current epsilon vs random number between 0 and 1
        if explorationTreshold > self.epsilon:
            # explore, which means predicted action
            action = np.argmax(self.model.getQs(state))
        else:
            # Explore, which means random action
            action = int(random.uniform(0, self.n_actions))
            explore = True

        self.currentTable = self.model.getQs(state)

        # decay epsilon
        if(episode >= self.n_epochsBeforeDecay):
            if(self.epsilon > self.epsMin):  # decay the value
                self.epsilon = self.epsilon * (1 - self.decay)
            elif(self.epsilon < self.epsMin):  # if decayed too far set to min
                self.epsilon = self.epsMin

        return action, explore, self.epsilon

    # update q table
    def learn(self, state, action, reward, new_state, done):
        self.model.updateReplayMemory((state, action, reward, new_state, done))
        self.model.train(done)

    def archive(self, epoch):
        if not os.path.exists("./Experiments/" + self.experimentName):
            os.makedirs("./Experiments/" + self.experimentName)
        self.model.targetModel.save(
            "./Experiments/" + str(self.experimentName) + "/" + str(epoch) + ".h5")


# Agent class
class DQNAgent:
    def __init__(self, inputs, outputs, learningRate, minReplay, replay, batch, gamma, update):
        self.numOfInputs = inputs
        self.numOfOutputs = outputs
        self.learningRate = learningRate
        self.minReplayMemSize = minReplay
        self.replayMemSize = replay
        self.batchSize = batch
        self.gamma = gamma
        self.updateRate = update

        # The model used for training at every step
        self.model = self.createModel()

        # Target network uesed for predicting, not updated every step
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())

        # saves replayMemSize many steps, so that the network does not just train on a single input
        self.replayMemory = deque(maxlen=self.replayMemSize)

        # This number is the ammount of epochs before the target Net will take over the other nets weights
        self.targetUpdateCounter = 0

    def createModel(self):
        modelShape = (self.numOfInputs, )
        model = Sequential()
        model.add(Input(shape=modelShape))
        model.add(Dense(int(modelShape[0]), activation='relu'))
        model.add(Dense(int(modelShape[0] * 0.66), activation='relu'))
        model.add(Dense(int(modelShape[0] * 0.5), activation='relu'))
        model.add(Dense(int(modelShape[0] * 0.33), activation='relu'))
        model.add(Dense(int(self.numOfOutputs * 1.66), activation='relu'))
        model.add(Dense(int(self.numOfOutputs * 1.33), activation='relu'))
        model.add(Dense(self.numOfOutputs, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=self.learningRate), metrics=['accuracy'])
        model.summary()
        return model

    # Adds the current data to the replayMemoryList - (observation space, action, reward, new observation space, done)
    def updateReplayMemory(self, transition):
        self.replayMemory.append(transition)

    # This trains the main network at every step
    def train(self, done):
        # Start training only if certain number of samples is already saved
        if len(self.replayMemory) < self.minReplayMemSize:
            return

        # Get a miniBatch of random samples from memory replay table
        miniBatch = random.sample(self.replayMemory, self.batchSize)

        # Get current states from miniBatch, then query NN model for Q values
        currentStates = np.array([transition[0] for transition in miniBatch])
        currentQsList = self.model.predict(currentStates)

        # Get future states from miniBatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        newCurrentStates = np.array([transition[3]
                                     for transition in miniBatch])
        futureQsList = self.targetModel.predict(newCurrentStates)

        statesInput = []
        controlsOutput = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(miniBatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                maxFutureQ = np.max(futureQsList[index])
                new_q = reward + self.gamma * maxFutureQ
            else:
                new_q = reward

            # Update Q value for given state
            currentQs = currentQsList[index]
            currentQs[action] = new_q

            # And append to our training data
            statesInput.append(current_state)
            controlsOutput.append(currentQs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(statesInput), np.array(controlsOutput), batch_size=self.batchSize, verbose=0,
                       shuffle=False)

        # Update target network counter every episode
        if done:
            self.targetUpdateCounter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.targetUpdateCounter > self.updateRate:
            self.targetModel.set_weights(self.model.get_weights())
            self.targetUpdateCounter = 0

    # Queries main network for Q values given current observation space (environment state)
    def getQs(self, state):
        # Important to get the right shape, therefore put shape in [] - test with print(state.ndim)
        state = np.array(np.array([state]))
        return self.model.predict(state)
