import numpy as np
import math
import random


class Scene():

    def __init__(self, dictObservation, dictAction, actions, stateDepth, startingVelocity, startingPitchRange, startingRollRange, usePredefinedSeeds, randomDesiredState, desiredPitchRange, desiredRollRange):
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.n_actions = actions
        self.stateList = []
        self.stateDepth = stateDepth
        self.startingVelocity = startingVelocity
        self.startingPitchRange = startingPitchRange
        self.startingRollRange = startingRollRange
        self.desiredPitchRange = desiredPitchRange
        self.desiredRollRange = desiredRollRange
        self.desiredPosition = {
            "roll": 0,
            "pitch": 0}
        self.dictState = {
            "pitch": 0,
            "roll": 1,
            "yaw": 2,
            "P": 3,
            "Q": 4,
            "R": 5,
            "Aoa": 6,
            "AoS": 7}
        self.randomDesiredState = randomDesiredState

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
        observation = newObservation[:]
        if(self.randomDesiredState):
            observation[self.dictObservation["pitch"]] = -(self.desiredPosition["pitch"] - observation[self.dictObservation["pitch"]])
            observation[self.dictObservation["roll"]] = -(self.desiredPosition["roll"] - observation[self.dictObservation["roll"]])

        # actions_binary = [pi+, pi-, ro+, ro-, ru+, ru-]

        # if action == pitch up
        if (action == 0):
            pitch_level_sign = observation[3]
            if (pitch_level_sign <= self.desiredPosition["pitch"]):
                reward = 1.0  # * (abs(pitch_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(pitch_level_sign)/denominator)

        # if action == pitch down
        if (action == 1):
            pitch_level_sign = observation[3]
            if (pitch_level_sign > self.desiredPosition["pitch"]):
                reward = 1.0  # * (abs(pitch_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(pitch_level_sign)/denominator)

        # if action == roll right
        if (action == 2):
            roll_level_sign = observation[4]
            if (roll_level_sign <= self.desiredPosition["roll"]):
                reward = 1.0  # * (abs(roll_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(roll_level_sign)/denominator)

        # if action == roll left
        if (action == 3):
            roll_level_sign = observation[4]
            if (roll_level_sign > self.desiredPosition["roll"]):
                reward = 1.0  # * (abs(roll_level_sign)/denominator)
            else:
                reward = -1.0  # * (abs(roll_level_sign)/denominator)

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
        observation = positions[:self.dictState["yaw"]]  # positions except yaw
        if(self.randomDesiredState):
            observation[self.dictState["pitch"]] = -(self.desiredPosition["pitch"] - observation[self.dictState["pitch"]])
            observation[self.dictState["roll"]] = -(self.desiredPosition["roll"] - observation[self.dictState["roll"]])
        state = tuple(observation) + tuple(velocities)
        self.stateList.append(state)
        if(len(self.stateList) > self.stateDepth):
            self.stateList.pop(0)
        return self.stateList

    def getState(self, observation):
        pitch = observation[self.dictObservation["pitch"]]
        roll = observation[self.dictObservation["roll"]]
        yaw = observation[self.dictObservation["yaw"]]
        if(self.randomDesiredState):
            pitch = -(self.desiredPosition["pitch"] - pitch)
            roll = -(self.desiredPosition["roll"] - roll)

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
        startingPitch = int(random.randrange(-self.startingPitchRange, self.startingPitchRange))
        startingRoll = int(random.randrange(-self.startingRollRange, self.startingRollRange))
        startingYaw = int(random.randrange(0, 360))

        if(self.randomDesiredState):
            destinationPitch = int(random.randrange(-self.desiredPitchRange, self.desiredPitchRange))
            destinationRoll = int(random.randrange(-self.desiredRollRange, self.desiredRollRange))
            self.desiredPosition["roll"] = destinationRoll
            self.desiredPosition["pitch"] = destinationPitch

        angleRadPitch = math.radians(startingPitch)
        verticalVelocity = self.startingVelocity * math.sin(angleRadPitch)
        forwardVelocity = self.startingVelocity * math.cos(angleRadPitch)

        if(startingPitch == 0 or startingPitch == 360):
            verticalVelocity = 0
            forwardVelocity = self.startingVelocity

        if(startingPitch == 180 or startingPitch == -180):
            verticalVelocity = 0
            forwardVelocity = -self.startingVelocity

        angleRadYaw = math.radians(startingYaw)
        eastVelocity = forwardVelocity * math.sin(angleRadYaw)
        northVelocity = - forwardVelocity * math.cos(angleRadYaw)

        if(startingYaw == 0 or startingYaw == 360):
            eastVelocity = 0
            northVelocity = - forwardVelocity

        if(startingYaw == 180 or startingYaw == -180):
            eastVelocity = 0
            northVelocity = forwardVelocity

        startingPosition = [startingRoll, startingPitch, startingYaw, northVelocity, eastVelocity, verticalVelocity]

        return startingPosition, [self.desiredPosition["roll"], self.desiredPosition["pitch"]]
