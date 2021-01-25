import numpy as np
import time
import math
import random

class QLearn():

    def __init__(self, n_stat, n_acts, gamm, lr, eps):
        self.n_states = n_stat
        self.n_actions = n_acts
        self.gamma = gamm
        self.learning_rate = lr
        self.epsilon = eps
        self.q_table = np.zeros( [self.n_states, self.n_actions] )

    #############################################################################

    def select_action(self, state, episode, n_epochs):
        action = 0
        e_pred = random.uniform(0, 1)

        if (  (episode/n_epochs) < 0.25  ):  # was previously 0.85  ## 0.85 some random and 0.15 no random
            selected_q_row = self.q_table[state,:] + np.random.randn(    1, self.n_actions  ) * (  1.0/(episode+1)  )
            action = np.argmax(  selected_q_row  )
        else:
            selected_q_row = self.q_table[state,:]
            action = np.argmax(  selected_q_row  )

        return action

    ###############################################################################

    def learn(self, s, action, reward, s_):
       lr = self.learning_rate
       y = self.gamma
       a = action
       r = reward
       self.q_table[s,a] = self.q_table[s,a] + lr*(r + y * np.max(self.q_table[s_,:]) - self.q_table[s,a] )


##############################################################################################
