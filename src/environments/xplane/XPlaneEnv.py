import numpy as np
import imp
import time


class Env():

    def __init__(self, scene, orig, dest, n_acts, usePredefinedSeeds, dictObservation, dictAction, dictRotation, speed, pause, qID, *args, **kwargs):
        self.scenario = scene
        self.startingPosition = orig
        self.destinationPosition = dest
        self.previousPosition = orig
        self.startingOrientation = []
        self.desiredState = []
        self.n_actions = n_acts
        self.xpc = imp.load_source('xpc', './src/environments/xplane/xpc.py')  # path is relative to the location of the QPlane.py file
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.dictRotation = dictRotation
        self.startingVelocity = speed
        self.pauseDelay = pause
        self.qID = qID
        self.id = "XPlane"

        if(usePredefinedSeeds):
            np.random.seed(42)

    def send_posi(self, posi, rotation):
        position = posi[:]
        position[self.dictObservation["pitch"]] = rotation[self.dictRotation["pitch"]]
        position[self.dictObservation["roll"]] = rotation[self.dictRotation["roll"]]
        position[self.dictObservation["yaw"]] = rotation[self.dictRotation["yaw"]]
        client = self.xpc.XPlaneConnect()
        client.sendPOSI(position)
        client.close()

    def send_velo(self, rotation):
        client = self.xpc.XPlaneConnect()
        #  I could try the q DREF https://developer.x-plane.com/article/movingtheplane/
        client.sendDREF("sim/flightmodel/position/local_vx", rotation[self.dictRotation["eastVelo"]])  # The velocity in local OGL coordinates +vx=E -vx=W
        client.sendDREF("sim/flightmodel/position/local_vy", rotation[self.dictRotation["verticalVelo"]])  # The velocity in local OGL coordinates +=Vertical (up)
        client.sendDREF("sim/flightmodel/position/local_vz", rotation[self.dictRotation["northVelo"]])  # The velocity in local OGL coordinates +vz=S -vz=N

        client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates +az=S -az=N

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

    def getVelo(self):
        client = self.xpc.XPlaneConnect()

        '''
        local_vx:   The velocity in local OGL coordinates - The +X axis points east from the reference point.
        local_vy:   The velocity in local OGL coordinates - The +Y axis points straight up away from the center of the earth at the reference point.
        local_vz:   The velocity in local OGL coordinates - The +Z axis points south from the reference point.
        local_ax:   The acceleration in local OGL coordinates
        local_ay:   The acceleration in local OGL coordinates
        local_az:   The acceleration in local OGL coordinates
        groundspeed:The ground speed of the aircraft
        P:          The roll rotation rates (relative to the flight)
        Q:          The pitch rotation rates (relative to the flight)
        R:          The yaw rotation rates (relative to the flight)
        P_dot:      The roll angular acceleration (relative to the flight)
        Q_dot:      The pitch angular acceleration (relative to the flight)
        R_dot:      The yaw angular acceleration rates (relative to the flight)
        alpha:      The pitch relative to the flown path (angle of attack)
        beta:       The heading relative to the flown path (yaw)
        '''

        drefs = ["sim/flightmodel/position/P", "sim/flightmodel/position/Q", "sim/flightmodel/position/R",
                 "sim/flightmodel/position/alpha", "sim/flightmodel/position/beta"]

        values = client.getDREFs(drefs)

        vel = []
        for i in range(len(values)):
            vel.append(values[i][0])

        client.close()

        return vel

    def send_Pause(self, pause):
        client = self.xpc.XPlaneConnect()
        client.pauseSim(pause)
        client.close()

    def send_Ctrl(self, ctrl):
        '''
        ctrl[0]: + Stick back (elevator pointing up) / - Stick in (elevator pointing down)
        ctrl[1]: + Stick right (right aileron up) / - Stick left (left aileron up)
        ctrl[2]: + Peddal (Rudder) right / - Peddal (Rudder) left
        '''
        client = self.xpc.XPlaneConnect()
        client.sendCTRL(ctrl)
        client.close()

    def get_Posi(self):
        client = self.xpc.XPlaneConnect()
        position = client.getPOSI(0)
        client.close()
        pos = list(position)
        return pos

    def get_Ctrl(self):
        client = self.xpc.XPlaneConnect()
        r = client.getCTRL(0)
        client.close()
        return r

    def getControl(self, action, observation):
        ctrl, actions_binary = self.scenario.getControl(action, observation)

        return ctrl, actions_binary

    def getDeepState(self, observation):
        velocities = self.getVelo()
        positions = observation[3:-1]

        state = self.scenario.getDeepState(velocities, positions)

        return state

    def getState(self, observation):
        state = self.scenario.getState(observation)
        return int(state)

    def rewardFunction(self, action, newObservation):
        client = self.xpc.XPlaneConnect()
        crash = client.getDREF("sim/flightmodel2/misc/has_crashed")[0]
        alt = client.getDREF("sim/flightmodel/position/elevation")[0]
        alpha = client.getDREF("sim/flightmodel/position/alpha")[0]
        client.close()

        reward, done = self.scenario.rewardFunction(action, newObservation, alt, alpha)
        if(crash):  # Would be used for end parameter - for example, if plane crahsed done, or if plane reached end done
            done = True
            reward = -1

        return reward, done

    def step(self, action):
        position = self.get_Posi()

        newCtrl, actions_binary = self.getControl(action, position)

        self.send_Pause(False)
        self.send_Ctrl(newCtrl)
        time.sleep(self.pauseDelay)
        self.send_Pause(True)

        position = self.get_Posi()

        if self.qID == "deep" or self.qID == "doubleDeep":
            state = self.getDeepState(position)
        else:
            state = self.getState(position)

        done = False
        reward = 0
        reward, done = self.rewardFunction(action, position)

        info = [position, actions_binary, newCtrl]
        return state, reward, done, info

    def reset(self):
        resetPosition, desintaionState = self.scenario.resetStartingPosition()
        self.startingOrientation = resetPosition
        self.desiredState = desintaionState
        self.send_posi(self.startingPosition, resetPosition)
        self.send_velo(resetPosition)
        #  self.send_envParam()
        self.scenario.resetStateDepth()
        self.send_Ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_Posi()
        if self.qID == "deep" or self.qID == "doubleDeep":
            state = self.getDeepState(new_posi)
        else:
            state = self.getState(new_posi)
        return state
