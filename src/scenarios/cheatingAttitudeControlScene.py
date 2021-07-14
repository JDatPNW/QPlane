import numpy as np
import math
import random


class Scene():

    def __init__(self, dictObservation, dictAction, actions, stateDepth, startingVelocity, usePredefinedSeeds):
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.n_actions = actions
        self.stateList = []
        self.stateDepth = stateDepth
        self.startingVelocity = startingVelocity
        self.id = "cheatAttitude"
        if(usePredefinedSeeds):
            random.seed(42)

    def getTermination(self, alt, alpha):

        # checks if plane is less than x feet off the ground, if not it will count as a crash
        if (alt < 1000):
            terminate = True
        elif(alpha >= 16):
            terminate = True
        else:
            terminate = False
        return terminate

    def convertRangeAtoRangeB(self, old_value):
        old_min = 0.0
        old_max = 360.0
        new_min = -180.0
        new_max = 180.0
        new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return new_value

    def rewardFunction(self, action, newObservation, alt, alpha):
        reward = 0

        # actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]

        # if action == pitch up
        if (action == 0):
            pitch_level_sign = newObservation[3]
            if (pitch_level_sign <= 0.0):
                reward = 1.0  # * (abs(pitch_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(pitch_level_sign)/denominator)

        # if action == pitch down
        if (action == 1):
            pitch_level_sign = newObservation[3]
            if (pitch_level_sign > 0.0):
                reward = 1.0  # * (abs(pitch_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(pitch_level_sign)/denominator)

        # if action == roll right
        if (action == 2):
            roll_level_sign = newObservation[4]
            if (roll_level_sign <= 0.0):
                reward = 1.0  # * (abs(roll_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(roll_level_sign)/denominator)

        # if action == roll left
        if (action == 3):
            roll_level_sign = newObservation[4]
            if (roll_level_sign > 0.0):
                reward = 1.0  # * (abs(roll_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(roll_level_sign)/denominator)

        # if action == rudder +
        if (action == 4):
            rudder_level_sign = self.convertRangeAtoRangeB(newObservation[5])

            if (rudder_level_sign <= 0.0):
                reward = 1.0  # * (abs(rudder_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(rudder_level_sign)/denominator)

        # if action == rudder -
        if (action == 5):

            rudder_level_sign = self.convertRangeAtoRangeB(newObservation[5])
            if (rudder_level_sign > 0.0):
                reward = 1.0  # * (abs(rudder_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(rudder_level_sign)/denominator)

        done = False
        if(self.getTermination(alt, alpha)):  # Would be used for end parameter - for example, if plane crahsed done, or if plane reached end done
            done = True
            reward = -10

        return reward, done

    def getControl(self, action, observation):
        # translate the action to the controll space value
        actionCtrl = 0
        if(action <= self.dictAction["pi-"]):
            actionCtrl = 0
        elif(action <= self.dictAction["ro-"]):
            actionCtrl = 1
        elif action <= self.dictAction["ru-"]:
            actionCtrl = 2

        ctrl = [0, 0, 0, 0.5, -998, -998]  # throttle set to .5 by default
        # translate the action value to the observation space value
        actionDimension = 3 + actionCtrl

        # this is to make actions less significant if the plane is more stable
        if(action != self.dictAction["no"]):
            if observation[actionDimension] < -180 or observation[actionDimension] > 180:
                ctrl[actionCtrl] = 1
            elif -180 <= observation[actionDimension] < -50 or 50 <= observation[actionDimension] < 180:
                ctrl[actionCtrl] = 0.75
            elif -50 <= observation[actionDimension] < -25 or 25 <= observation[actionDimension] < 50:
                ctrl[actionCtrl] = 0.66
            elif -25 <= observation[actionDimension] < -15 or 15 <= observation[actionDimension] < 25:
                ctrl[actionCtrl] = 0.5
            elif -15 <= observation[actionDimension] < -10 or 10 <= observation[actionDimension] < 15:
                ctrl[actionCtrl] = 0.33
            elif -10 <= observation[actionDimension] < -5 or 5 <= observation[actionDimension] < 10:
                ctrl[actionCtrl] = 0.25
            elif -5 <= observation[actionDimension] < -2 or 2 <= observation[actionDimension] < 5:
                ctrl[actionCtrl] = 0.1
            elif -2 <= observation[actionDimension] < -1 or 1 <= observation[actionDimension] < 2:
                ctrl[actionCtrl] = 0.05
            elif -1 <= observation[actionDimension] < 0 or 0 <= observation[actionDimension] < 1:
                ctrl[actionCtrl] = 0.025
            else:
                print("DEBUG - should not get here")
        else:
            ctrl = [0, 0, 0, 0.5, -998, -998]

        if (actionCtrl == 2):
            # Doing this because the pedals don't work with the applied degree idea
            ctrl[actionCtrl] = 0.01

        if(action % 2 != 0):  # check if action should be positive or negative
            ctrl[actionCtrl] = -ctrl[actionCtrl]

        actions_binary = np.zeros(self.n_actions, dtype=int)
        actions_binary[action] = 1

        return ctrl, actions_binary

    def getDeepState(self, velocities, positions):
        vel = []
        for i in range(len(velocities)):
            vel.append(velocities[i])
        state = tuple(positions) + tuple(vel)
        self.stateList.append(state)
        if(len(self.stateList) > self.stateDepth):
            self.stateList.pop(0)
        return self.stateList

    def getState(self, observation):
        pitch = observation[self.dictObservation["pitch"]]
        roll = observation[self.dictObservation["roll"]]
        yaw = observation[self.dictObservation["yaw"]]
        pitchEnc, rollEnc, yawEnc = self.encodeRotations(pitch, roll, yaw)

        state = self.encodeState(pitchEnc, rollEnc, yawEnc)
        return state

    def encodeState(self, pitch, roll, yaw):
        i = pitch
        i = i * 13
        i = i + roll
        return i

    def encodeRotation(self, i):
        if -180 <= i < -75:
            return 0
        elif -75 <= i < -35:
            return 1
        elif -35 <= i < -15:
            return 2
        elif -15 <= i < -5:
            return 3
        elif -5 <= i < -2:
            return 4
        elif -2 <= i < -1:
            return 5
        elif -1 <= i < 1:
            return 6
        elif 1 <= i < 2:
            return 7
        elif 2 <= i < 5:
            return 8
        elif 5 <= i < 15:
            return 9
        elif 15 <= i < 35:
            return 10
        elif 35 <= i < 75:
            return 11
        elif 75 <= i < 180:
            return 12
        else:
            return 0

    def encodeRotations(self, pitch, roll, yaw):
        pitchEnc = self.encodeRotation(pitch)
        rollEnc = self.encodeRotation(roll)
        yawEnc = self.encodeRotation(yaw)
        return pitchEnc, rollEnc, yawEnc

    def resetStateDepth(self):
        self.stateList = []

    def resetStartingPosition(self):
        startingPitch = int(random.randrange(-10, 10))
        startingRoll = int(random.randrange(-15, 15))
        startingYaw = int(random.randrange(0, 360))

        angleRadPitch = math.radians(startingPitch)
        verticalVelocity = self.startingVelocity * math.sin(angleRadPitch)
        forwardVelocity = self.startingVelocity * math.cos(angleRadPitch)

        if(startingPitch == 0):
            verticalVelocity = 0
            forwardVelocity = self.startingVelocity

        if(startingPitch == 180):
            verticalVelocity = 0
            forwardVelocity = -self.startingVelocity

        angleRadYaw = math.radians(startingYaw)
        eastVelocity = forwardVelocity * math.sin(angleRadYaw)
        northVelocity = - forwardVelocity * math.cos(angleRadYaw)

        if(startingYaw == 0):
            eastVelocity = 0
            northVelocity = - forwardVelocity

        if(startingYaw == 180):
            eastVelocity = 0
            northVelocity = forwardVelocity

        startingPosition = [startingRoll, startingPitch, startingYaw, northVelocity, eastVelocity, verticalVelocity]

        return startingPosition
