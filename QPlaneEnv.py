import numpy as np
import imp


class QPlaneEnv():

    def __init__(self, orig, dest, n_acts, endParam, dictObservation, dictAction, dictRotation, speed):
        self.startingPosition = orig
        self.destinationPosition = dest
        self.previousPosition = orig
        self.n_actions = n_acts
        self.endThreshold = endParam
        self.xpc = imp.load_source('xpc', 'xpc.py')
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.dictRotation = dictRotation
        self.startingVelocity = speed

    def send_posi(self, posi, rotation):
        posi[self.dictObservation["pitch"]] = rotation[self.dictRotation["pitch"]]
        posi[self.dictObservation["roll"]] = rotation[self.dictRotation["roll"]]
        client = self.xpc.XPlaneConnect()
        client.sendPOSI(posi)
        client.close()

    def send_velo(self, rotation):
        client = self.xpc.XPlaneConnect()

        client.sendDREF("sim/flightmodel/position/local_vx", 0)  # The velocity in local OGL coordinates +vx=E -vx=W
        client.sendDREF("sim/flightmodel/position/local_vy", rotation[self.dictRotation["velocityY"]])  # The velocity in local OGL coordinates +=Vertical (up)
        client.sendDREF("sim/flightmodel/position/local_vz", self.startingVelocity)  # The velocity in local OGL coordinates -vz=S +vz=N

        client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates -az=S +az=N

        client.sendDREF("sim/flightmodel/weight/m_fuel1", 65.0)  # fuel quantity failure_enum
        client.sendDREF("sim/flightmodel/weight/m_fuel2", 65.0)  # fuel quantity failure_enum

        client.sendDREF("sim/operation/failures/rel_ss_dgy", 0)  # Directional Gyro (Pilot) failure_enum
        client.sendDREF("sim/operation/failures/rel_cop_dgy", 0)  # Directional Gyro (CoPilot) failure_enum

        client.close()

    def send_envParam(self):
        client = self.xpc.XPlaneConnect()

        # Wind speed
        client.sendDREF("sim/weather/wind_speed_kt[0]", 0)  # >= 0 The wind speed in knots.
        client.sendDREF("sim/weather/wind_speed_kt[1]", 0)  # >= 0 The wind speed in knots.
        client.sendDREF("sim/weather/wind_speed_kt[2]", 0)  # >= 0 The wind speed in knots.

        client.sendDREF("sim/weather/wind_turbulence_percent", 0)  # [0.0 – 1.0] The percentage of wind turbulence present.

        client.sendDREF("sim/weather/wind_direction_degt[0]", 0)  # [0 – 360) The direction the wind is blowing from in degrees from true north c lockwise.
        client.sendDREF("sim/weather/wind_direction_degt[1]", 0)  # [0 – 360) The direction the wind is blowing from in degrees from true north c lockwise.
        client.sendDREF("sim/weather/wind_direction_degt[2]", 0)  # [0 – 360) The direction the wind is blowing from in degrees from true north c lockwise.

        client.sendDREF("sim/operation/failures/rel_g_fuel", 0)  # fuel quantity failure_enum

        client.sendDREF("sim/operation/failures/rel_ss_dgy", 0)  # Directional Gyro (Pilot) failure_enum
        client.sendDREF("sim/operation/failures/rel_cop_dgy", 0)  # Directional Gyro (CoPilot) failure_enum

        client.close()

    def send_ctrl(self, ctrl):
        client = self.xpc.XPlaneConnect()
        client.sendCTRL(ctrl)
        client.close()

    def get_posi(self):
        client = self.xpc.XPlaneConnect()
        r = client.getPOSI(0)
        client.close()
        return r

    def get_ctrl(self):
        client = self.xpc.XPlaneConnect()
        r = client.getCTRL(0)
        client.close()
        return r

    def getControl(self, ctrl, action, reward, observation):
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
                ctrl[actionCtrl] = 0.5
            elif -50 <= observation[actionDimension] < -25 or 25 <= observation[actionDimension] < 50:
                ctrl[actionCtrl] = 0.2
            elif -25 <= observation[actionDimension] < -15 or 15 <= observation[actionDimension] < 25:
                ctrl[actionCtrl] = 0.15
            elif -15 <= observation[actionDimension] < -10 or 10 <= observation[actionDimension] < 15:
                ctrl[actionCtrl] = 0.12
            elif -10 <= observation[actionDimension] < -5 or 5 <= observation[actionDimension] < 10:
                ctrl[actionCtrl] = 0.1
            elif -5 <= observation[actionDimension] < -2 or 2 <= observation[actionDimension] < 5:
                ctrl[actionCtrl] = 0.05
            elif -2 <= observation[actionDimension] < -1 or 1 <= observation[actionDimension] < 2:
                ctrl[actionCtrl] = 0.02
            elif -1 <= observation[actionDimension] < 0 or 0 <= observation[actionDimension] < 1:
                ctrl[actionCtrl] = 0.01
            else:
                print("DEBUG - should not get here")
        else:
            ctrl = [0, 0, 0, 0.5, -998, -998]

        if (actionCtrl == 2):
            ctrl[actionCtrl] = 0.01  # Doing this because the pedals don't work with the applied degree idea

        if(action % 2 != 0):  # check if action should be positive or negative
            ctrl[actionCtrl] = -ctrl[actionCtrl]

        actions_binary = np.zeros(self.n_actions, dtype=int)
        actions_binary[action] = 1

        return ctrl, actions_binary

    def encodeState(self, pitch, roll, yaw):
        i = pitch
        i = i * 15
        i = i + roll
        return i

    def encodeRotation(self, i):

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

    def encodeRotations(self, pitch, roll, yaw):
        pitchEnc = self.encodeRotation(pitch)
        rollEnc = self.encodeRotation(roll)
        yawEnc = self.encodeRotation(yaw)
        return pitchEnc, rollEnc, yawEnc

    def getState(self, observation):
        pitch = observation[self.dictObservation["pitch"]]
        roll = observation[self.dictObservation["roll"]]
        yaw = observation[self.dictObservation["yaw"]]
        pitchEnc, rollEnc, yawEnc = self.encodeRotations(pitch, roll, yaw)

        state = self.encodeState(pitchEnc, rollEnc, yawEnc)
        return int(state)

    def rewardFunction(self, action, oldObservation, newObservation):

        if(abs(newObservation[self.dictObservation["roll"]]) > 40 or abs(newObservation[self.dictObservation["pitch"]]) > 40):
            reward = -5
        elif(abs(newObservation[self.dictObservation["roll"]]) > 20 or abs(newObservation[self.dictObservation["pitch"]]) > 20):
            reward = -3
        elif(abs(newObservation[self.dictObservation["roll"]]) > 10 or abs(newObservation[self.dictObservation["pitch"]]) > 10):
            reward = 1
        elif(abs(newObservation[self.dictObservation["roll"]]) > 5 or abs(newObservation[self.dictObservation["pitch"]]) > 5):
            reward = 2
        elif(abs(newObservation[self.dictObservation["roll"]]) > 1 or abs(newObservation[self.dictObservation["pitch"]]) > 1):
            reward = 4
        else:
            reward = 10

        done = False
        if False:  # Would be used for end parameter - for example, if plane crahsed done, or if plane reached end done
            done = True

        return reward, done

    def step(self, action, oldObservation, newObservation):
        done = False
        reward = 0
        reward, done = self.rewardFunction(
            action, oldObservation, newObservation)
        return reward, done

    def update(self, action, reward, position):
        oldCtrl = self.get_ctrl()
        newCtrl, actions_binary = self.getControl(
            oldCtrl, action, reward, position)
        self.send_ctrl(newCtrl)
        posi = self.get_posi()
        return posi, actions_binary, newCtrl

    def reset(self, posi, rotation):
        self.send_posi(posi, rotation)
        self.send_velo(rotation)
        #  self.send_envParam()
        self.send_ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_posi()
        return new_posi
