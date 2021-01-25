import scipy.spatial.distance as distance
import numpy as np
from math import radians, cos, sin, asin, sqrt
import imp
import numpy as np


################################################################################

## positions in xplane are observations
## format [lat, long, alt, pitch, roll, true heading/yaw, gear]
## Palo Alto
## starting_position    = [37.524, -122.06899,  4000, 0.0, 0.0, 0.0, 1]
## Sunol Regional Wilderness (20 kms about east from Palo Alto)
## destination_position = [37.505, -121.843611, 4000, 0.0, 0.0, 0.0, 1]

################################################################################

class QPlaneEnv():

    def __init__(self, orig, dest, acts_bin, end_param):
        self.starting_position = orig
        self.destination_position = dest
        self.previous_position = orig
        self.actions_binary_n = acts_bin
        self.end_game_threshold = end_param
        self.xpc = imp.load_source('xpc','xpc.py')

    ################################################################################


    ##########################################################################

    def send_posi(self, posi):
        client = self.xpc.XPlaneConnect()
        client.sendPOSI(posi)
        client.close()


    def send_velo(self):
        client = self.xpc.XPlaneConnect()

        client.sendDREF("sim/flightmodel/position/local_vx", 0)
        client.sendDREF("sim/flightmodel/position/local_vy", 0)
        client.sendDREF("sim/flightmodel/position/local_vz", -50)

        client.sendDREF("sim/flightmodel/position/theta", 0)
        client.sendDREF("sim/flightmodel/position/phi", 0)
        client.sendDREF("sim/flightmodel/position/psi", 0)

        client.sendDREF("sim/flightmodel/position/local_ax", 0)
        client.sendDREF("sim/flightmodel/position/local_ay", 0)
        client.sendDREF("sim/flightmodel/position/local_az", 0)

        client.sendDREF("sim/flightmodel/position/P", 0)
        client.sendDREF("sim/flightmodel/position/Q", 0)
        client.sendDREF("sim/flightmodel/position/R", 0)

        client.sendDREF("sim/flightmodel/position/true_theta", 0)
        client.sendDREF("sim/flightmodel/position/true_phi", 0)
        client.sendDREF("sim/flightmodel/position/true_psi", 0)

        client.close()


    def send_ctrl(self, ctrl):
        client = self.xpc.XPlaneConnect()
        client.sendCTRL(ctrl)
        client.close()


    def get_posi(self):
        client = self.xpc.XPlaneConnect()
        #r = client.getDREFs(drefs_position)
        r = client.getPOSI(0)
        client.close()
        return r


    def get_ctrl(self):
        client = self.xpc.XPlaneConnect()
        #r = client.getDREFs(drefs_controls)
        r = client.getCTRL(0)
        client.close()
        return r


    def reset(self, posi):
        self.send_posi(posi)
        self.send_velo()
        self.send_ctrl([0,0,0,0,0,0,1])
        new_posi = self.get_posi()
        return new_posi


    def convert_action_to_control(self, ctrl, action, reward, position):
        # actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]
        # pitch = ctrl[0] - roll =  ctrl[1] - rudder = ctrl[2]

        if action < 2:
            takeaction = 0
        elif action < 4:
            takeaction = 1
        elif action < 6:
            takeaction = 2
        elif action < 7:
            takeaction = 0

        ctrl = [0, 0, 0, 0.5, -998, -998]
        if(action !=6):
            if position[3+takeaction] < -180 or position[3+takeaction]> 180:
               ctrl[takeaction] =  1
            elif -180 <=position[3+takeaction]< -50 or  50 <=position[3+takeaction]< 180:
               ctrl[takeaction] =  0.5
            elif -50 <=position[3+takeaction]< -25 or 25 <=position[3+takeaction]< 50:
               ctrl[takeaction] =  0.2
            elif -25 <=position[3+takeaction]< -15 or 15 <=position[3+takeaction]< 25:
               ctrl[takeaction] =  0.15
            elif -15 <=position[3+takeaction]< -10 or 10 <=position[3+takeaction]< 15:
               ctrl[takeaction] =  0.12
            elif -10 <=position[3+takeaction]< -5 or 5 <=position[3+takeaction]< 10:
               ctrl[takeaction] =  0.1
            elif -5 <=position[3+takeaction]< -2 or 2 <=position[3+takeaction]< 5:
               ctrl[takeaction] =  0.05
            elif -2 <=position[3+takeaction]< -1 or 1 <=position[3+takeaction]< 2:
               ctrl[takeaction] =  0.02
            elif -1 <=position[3+takeaction]< 0 or  0 <=position[3+takeaction]< 1:
               ctrl[takeaction] =  0.01
            else:
                print("DEBUG - should not get here")
        else:
            ctrl = [0, 0, 0, 0.5, -998, -998]

        if(action%2 != 0):
            ctrl[takeaction] = -ctrl[takeaction]

        actions_binary = np.zeros(self.actions_binary_n, dtype=int)
        actions_binary[action] = 1

        return ctrl, actions_binary



    def update(self, action, reward, position):
        old_ctrl = self.get_ctrl()
        new_ctrl, actions_binary = self.convert_action_to_control(old_ctrl, action , reward, position)
        self.send_ctrl(new_ctrl) ## set control surfaces e.g. pilot the plane
        posi = self.get_posi()
        return posi, actions_binary, new_ctrl


    def encode_state_xplane(self, pitch, roll, yaw):
        # 15 x 15 + 15 = 240
        i = pitch
        i = i * 15
        i = i + roll
        return i


    def range_to_vector_index(self, i):

        if -180 <= i < -50:
           return 0
        elif -50 <= i < -25:
           return 1
        elif -25 <= i < -15:
           return 2
        elif -15 <= i < -10:
           return 3
        elif -10 <= i < -5:
           return 4
        elif -5 <= i < -2:
           return 5
        elif -2 <= i < -1:
           return 6
        elif -1 <= i < 1:
           return 7
        elif 1 <= i < 2:
           return 8
        elif 2 <= i < 5:
           return 9
        elif 5 <= i < 10:
           return 10
        elif 10 <= i < 15:
           return 11
        elif 15 <= i < 25:
           return 12
        elif 25 <= i < 50:
           return 13
        elif 50 <= i < 180:
           return 14
        else:
           return 0


    def six_pack_values_to_vectors(self, pitch_ob, roll_ob, yaw_ob):
        pitch_vector_index = self.range_to_vector_index(pitch_ob)
        roll_vector_index = self.range_to_vector_index(roll_ob)
        yaw_vector_index = self.range_to_vector_index(yaw_ob)
        return pitch_vector_index, roll_vector_index, yaw_vector_index


    def get_state_from_observation(self, observation):
       # observation = [lat, long, alt, pitch, roll, yaw, gear]
       # states are pitch, roll, yaw readings from the six pack
       # states = 9x9x9 = 729

       state = 0

       pitch_ob = observation[3] #pitch
       roll_ob = observation[4] #roll
       yaw_ob = observation[5] #yaw
       pitch_index, roll_index, yaw_index = self.six_pack_values_to_vectors(pitch_ob, roll_ob, yaw_ob)

       state = self.encode_state_xplane(pitch_index, roll_index, yaw_index)
       return state


    def reward_function(self, action, position_before_action, current_position):
        roll = float(abs(current_position[4]/180)*3)
        pitch = float(abs(current_position[3]/180)*2)
        reward = float((5 - (roll + pitch )) / 5)

        if(abs(current_position[4]) > 50):
            reward = reward * 0.25
        elif(abs(current_position[4]) > 25):
            reward = reward * 0.5
        elif(abs(current_position[4]) > 10):
            reward = reward * 0.575
        elif(abs(current_position[4]) > 5):
            reward = reward * 0.9
        elif(abs(current_position[4]) > 2):
            reward = reward * 0.95
        elif(abs(current_position[4]) > 1):
            reward = reward * 0.99

        if(abs(current_position[3]) > 40):
            reward = reward * 0.1
        elif(abs(current_position[3]) > 25):
            reward = reward * 0.25
        elif(abs(current_position[3]) > 10):
            reward = reward * 0.5
        elif(abs(current_position[3]) > 5):
            reward = reward * 0.75
        elif(abs(current_position[3]) > 2):
            reward = reward * 0.85
        elif(abs(current_position[3]) > 1):
            reward = reward * 0.99

        ## if action == pitch up
        if (action == 0):
            if (current_position[3] > 2.0):
                reward = reward * 0.25
                #print("if action == pitch up")

        ## if action == pitch down
        if (action == 1):
            if (current_position[3] < -2.0):
                reward = reward * 0.25
                #print("if action == pitch down")

        ## if action == roll right
        if (action == 2):
            if (current_position[4] > 2.0):
                reward = reward * 0.25

        ## if action == roll left
        if (action == 3):
            if (current_position[4] < -2.0):
                reward = reward * 0.25

        done = False
        if False:  # Would be used for end parameter - for example, if plane crahsed done, or if plane reached end done
            done = True

        return reward, done


    def step(self, action, position_before_action, current_position):
        done = False
        reward = 0
        reward, done = self.reward_function(action, position_before_action, current_position)
        return reward, done
