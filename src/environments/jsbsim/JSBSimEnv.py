import numpy as np
import jsbsim
import os


class Env():

    def __init__(self, orig, dest, n_acts, endParam, dictObservation, dictAction, dictRotation, speed, pause, qID, render):
        self.startingPosition = orig
        self.destinationPosition = dest
        self.previousPosition = orig
        self.n_actions = n_acts
        self.endThreshold = endParam
        self.dictObservation = dictObservation
        self.dictAction = dictAction
        self.dictRotation = dictRotation
        self.startingVelocity = speed
        self.pauseDelay = pause
        self.qID = qID
        self.fsToMs = 0.3048  # convertion from feet per sec to meter per sec
        self.msToFs = 3.28084  # convertion from meter per sec to feet per sec
        self.radToDeg = 57.2957795  # convertion from radiants to degree
        self.degToRad = 0.0174533  # convertion from deg to rad
        self.physicsPerSec = 120  # default by jsb. Each physics step is a 120th of 1 sec

        os.environ["JSBSIM_DEBUG"] = str(0)  # set this before creating fdm to stop debug print outs
        self.fdm = jsbsim.FGFDMExec('./src/environments/jsbsim/', None)  # declaring the sim and setting the path
        self.fdm.load_model('c172r')  # loading cassna 172
        if render:  # only when render is True
            # Open Flight gear and enter: --fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ
            self.fdm.set_output_directive('data_output/flightgear.xml')  # loads xml that initates udp transfer
        self.fdm.run_ic()  # init the sim

    def send_posi(self, posi, rotation):
        posi[self.dictObservation["pitch"]] = rotation[self.dictRotation["pitch"]]
        posi[self.dictObservation["roll"]] = rotation[self.dictRotation["roll"]]

        self.fdm.set_property_value("ic/lat-gc-deg", posi[self.dictObservation["lat"]])  # Latitude initial condition in degrees
        self.fdm.set_property_value("ic/long-gc-deg", posi[self.dictObservation["long"]])  # Longitude initial condition in degrees
        self.fdm.set_property_value("ic/h-sl-ft", posi[self.dictObservation["alt"]])  # Height above sea level initial condition in feet

        self.fdm.set_property_value("ic/theta-deg", posi[self.dictObservation["pitch"]])  # Pitch angle initial condition in degrees
        self.fdm.set_property_value("ic/phi-deg", posi[self.dictObservation["roll"]])  # Roll angle initial condition in degrees
        self.fdm.set_property_value("ic/psi-true-deg", posi[self.dictObservation["yaw"]])  # Heading angle initial condition in degrees

    def send_velo(self, rotation):

        # ic/p-rad_sec (read/write) Roll rate initial condition in radians/second
        # ic/q-rad_sec (read/write) Pitch rate initial condition in radians/second
        # ic/r-rad_sec (read/write) Yaw rate initial condition in radians/second

        self.fdm.set_property_value("ic/ve-fps", 0 * self.msToFs)  # Local frame y-axis (east) velocity initial condition in feet/second
        self.fdm.set_property_value("ic/vd-fps", -rotation[self.dictRotation["velocityY"]] * self.msToFs)  # Local frame z-axis (down) velocity initial condition in feet/second
        self.fdm.set_property_value("ic/vn-fps", self.startingVelocity * self.msToFs)  # Local frame x-axis (north) velocity initial condition in feet/second
        self.fdm.set_property_value("propulsion/refuel", True)  # refules the plane?
        self.fdm.set_property_value("propulsion/active_engine", True)  # starts the engine?
        self.fdm.set_property_value("propulsion/set-running", 0)  # starts the engine?

        # client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        # client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        # client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates -az=S +az=N

    def getVelo(self):

        local_vx = self.fdm.get_property_value("velocities/v-east-fps") * self.fsToMs  # Velocity East (local)
        local_vy = -self.fdm.get_property_value("velocities/v-down-fps") * self.fsToMs  # Velocity Down (local)
        local_vz = self.fdm.get_property_value("velocities/v-north-fps") * self.fsToMs  # Velocity North (local)

        local_ax = self.fdm.get_property_value("accelerations/Nx") * self.fsToMs   # The acceleration in local coordinates +ax=E -ax=W?
        local_ay = -self.fdm.get_property_value("accelerations/Ny") * self.fsToMs  # The acceleration in local coordinates +=Vertical (down)?
        local_az = self.fdm.get_property_value("accelerations/Nz") * self.fsToMs  # The acceleration in local coordinates -az=S +az=N?

        groundspeed = self.fdm.get_property_value("velocities/vg-fps") * self.fsToMs  # The ground speed of the aircraft
        P = self.fdm.get_property_value("velocities/p-rad_sec") * self.radToDeg  # The roll rotation rates
        Q = self.fdm.get_property_value("velocities/q-rad_sec") * self.radToDeg  # The pitch rotation rates
        R = self.fdm.get_property_value("velocities/r-rad_sec") * self.radToDeg  # The yaw rotation rates
        P_dot = self.fdm.get_property_value("accelerations/pdot-rad_sec2") * self.radToDeg  # The roll angular acceleration
        Q_dot = self.fdm.get_property_value("accelerations/qdot-rad_sec2") * self.radToDeg  # The pitch angular acceleration
        R_dot = self.fdm.get_property_value("accelerations/rdot-rad_sec2") * self.radToDeg  # The yaw angular acceleration

        values = [local_vx, local_vy, local_vz, local_ax, local_ay, local_az, groundspeed, P, Q, R, P_dot, Q_dot, R_dot]

        return values

    def getCrashed(self):

        if (self.fdm.get_property_value("position/h-agl-ft") < 50):  # checks if plane is less than x feet off the ground, if not it will count as a crash
            crash = True
        else:
            crash = False
        return crash

    def send_Ctrl(self, ctrl):
        self.fdm.set_property_value("fcs/elevator-cmd-norm", ctrl[0])  # Elevator control (stick in/out)?
        self.fdm.set_property_value("fcs/left-aileron-cmd-norm", ctrl[1])  # Aileron control (stick left/right)? might need to switch
        self.fdm.set_property_value("fcs/right-aileron-cmd-norm", -ctrl[1])  # Aileron control (stick left/right)? might need to switch
        self.fdm.set_property_value("fcs/rudder-cmd-norm", ctrl[2])  # Rudder control (peddals)
        self.fdm.set_property_value("fcs/throttle-cmd-norm", ctrl[3])  # throttle

    def get_Posi(self):
        lat = self.fdm.get_property_value("position/lat-gc-deg")  # Latitude
        long = self.fdm.get_property_value("position/long-gc-deg")  # Longitude
        alt = self.fdm.get_property_value("position/h-sl-ft")  # altitude

        pitch = self.fdm.get_property_value("attitude/theta-deg")  # pitch
        roll = self.fdm.get_property_value("attitude/phi-deg")  # roll
        heading = self.fdm.get_property_value("attitude/psi-deg")  # yaw

        r = [lat, long, alt, pitch, roll, heading]

        return r

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
        i = i * 9
        i = i + roll
        i = i * 9
        i = i + yaw
        return i

    def encodeRotation(self, i):
        if -180 <= i < -35:
            return 0
        elif -35 <= i < -25:
            return 1
        elif -25 <= i < -15:
            return 2
        elif -15 <= i < -5:
            return 3
        elif -5 <= i < 5:
            return 4
        elif 5 <= i < 15:
            return 5
        elif 15 <= i < 25:
            return 6
        elif 25 <= i < 35:
            return 7
        elif 35 <= i < 180:
            return 8
        else:
            return 0

    def encodeRotations(self, pitch, roll, yaw):
        pitchEnc = self.encodeRotation(pitch)
        rollEnc = self.encodeRotation(roll)
        yawEnc = self.encodeRotation(yaw)
        return pitchEnc, rollEnc, yawEnc

    def getDeepState(self, observation):
        velocities = self.getVelo()
        positions = observation
        vel = []
        for i in range(len(velocities)):
            vel.append(velocities[i])
        state = tuple(positions) + tuple(vel)
        return state

    def getState(self, observation):
        pitch = observation[self.dictObservation["pitch"]]
        roll = observation[self.dictObservation["roll"]]
        yaw = observation[self.dictObservation["yaw"]]
        pitchEnc, rollEnc, yawEnc = self.encodeRotations(pitch, roll, yaw)

        state = self.encodeState(pitchEnc, rollEnc, yawEnc)
        return int(state)

    def rewardFunction(self, action, newObservation):
        roll = float(abs(newObservation[self.dictObservation["roll"]] / 180))
        pitch = float(abs(newObservation[self.dictObservation["pitch"]] / 180))
        reward = pow(float((2 - (roll + pitch)) / 2), 2)

        if(abs(newObservation[self.dictObservation["roll"]]) > 40 or abs(newObservation[self.dictObservation["pitch"]]) > 40):
            reward = reward * 0.1
        elif(abs(newObservation[self.dictObservation["roll"]]) > 20 or abs(newObservation[self.dictObservation["pitch"]]) > 20):
            reward = reward * 0.25
        elif(abs(newObservation[self.dictObservation["roll"]]) > 10 or abs(newObservation[self.dictObservation["pitch"]]) > 10):
            reward = reward * 0.5
        elif(abs(newObservation[self.dictObservation["roll"]]) > 5 or abs(newObservation[self.dictObservation["pitch"]]) > 5):
            reward = reward * 0.75
        elif(abs(newObservation[self.dictObservation["roll"]]) > 1 or abs(newObservation[self.dictObservation["pitch"]]) > 1):
            reward = reward * 0.9

        if(newObservation[self.dictObservation["alt"]] <= 1000):
            reward = reward * 0.1

        done = False
        if(self.getCrashed()):  # Would be used for end parameter - for example, if plane crahsed done, or if plane reached end done
            done = True
            reward = -1

        return reward, done

    def step(self, action):
        position = self.get_Posi()

        newCtrl, actions_binary = self.getControl(action, position)

        self.send_Ctrl(newCtrl)
        for i in range(int(self.pauseDelay * self.physicsPerSec)):  # will mean that the input will be applied for pauseDelay seconds
            self.send_Ctrl(newCtrl)
            self.fdm.run()

        position = self.get_Posi()

        if self.qID == "deep":
            state = self.getDeepState(position)
        else:
            state = self.getState(position)

        done = False
        reward = 0
        reward, done = self.rewardFunction(action, state)

        info = [position, actions_binary, newCtrl]
        return state, reward, done, info

    def reset(self, posi, rotation):
        self.send_posi(posi, rotation)
        self.send_velo(rotation)

        self.fdm.run_ic()

        self.send_Ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_Posi()
        if self.qID == "deep":
            state = self.getDeepState(new_posi)
        else:
            state = self.getState(new_posi)
        return state
