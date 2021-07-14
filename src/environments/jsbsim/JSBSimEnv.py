import numpy as np
import jsbsim
import os
import time


class Env():

    def __init__(self, scene, orig, dest, n_acts, usePredefinedSeeds, dictObservation, dictAction, dictRotation, speed, pause, qID, render, realTime):
        self.scenario = scene
        self.startingPosition = orig
        self.destinationPosition = dest
        self.previousPosition = orig
        self.startingOrientation = []
        self.n_actions = n_acts
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
        self.realTime = realTime
        self.id = "JSBSim"

        if(usePredefinedSeeds):
            np.random.seed(42)

        os.environ["JSBSIM_DEBUG"] = str(0)  # set this before creating fdm to stop debug print outs
        self.fdm = jsbsim.FGFDMExec('./src/environments/jsbsim/jsbsim/', None)  # declaring the sim and setting the path
        self.physicsPerSec = int(1 / self.fdm.get_delta_t())  # default by jsb. Each physics step is a 120th of 1 sec
        self.realTimeDelay = self.fdm.get_delta_t()
        self.fdm.load_model('c172r')  # loading cassna 172
        if render:  # only when render is True
            # Open Flight gear and enter: --fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ
            self.fdm.set_output_directive('./data_output/flightgear.xml')  # loads xml that initates udp transfer
        self.fdm.run_ic()  # init the sim
        self.fdm.print_simulation_configuration()

    def send_posi(self, posi, rotation):
        posi[self.dictObservation["pitch"]] = rotation[self.dictRotation["pitch"]]
        posi[self.dictObservation["roll"]] = rotation[self.dictRotation["roll"]]
        posi[self.dictObservation["yaw"]] = rotation[self.dictRotation["yaw"]]

        self.fdm["ic/lat-gc-deg"] = posi[self.dictObservation["lat"]]  # Latitude initial condition in degrees
        self.fdm["ic/long-gc-deg"] = posi[self.dictObservation["long"]]  # Longitude initial condition in degrees
        self.fdm["ic/h-sl-ft"] = posi[self.dictObservation["alt"]]  # Height above sea level initial condition in feet

        self.fdm["ic/theta-deg"] = posi[self.dictObservation["pitch"]]  # Pitch angle initial condition in degrees
        self.fdm["ic/phi-deg"] = posi[self.dictObservation["roll"]]  # Roll angle initial condition in degrees
        self.fdm["ic/psi-true-deg"] = posi[self.dictObservation["yaw"]]  # Heading angle initial condition in degrees

    def send_velo(self, rotation):

        self.fdm["ic/ve-fps"] = rotation[self.dictRotation["eastVelo"]] * self.msToFs  # Local frame y-axis (east) velocity initial condition in feet/second
        self.fdm["ic/vd-fps"] = -rotation[self.dictRotation["verticalVelo"]] * self.msToFs  # Local frame z-axis (down) velocity initial condition in feet/second
        self.fdm["ic/vn-fps"] = -rotation[self.dictRotation["northVelo"]] * self.msToFs  # Local frame x-axis (north) velocity initial condition in feet/second
        self.fdm["propulsion/refuel"] = True  # refules the plane?
        self.fdm["propulsion/active_engine"] = True  # starts the engine?
        self.fdm["propulsion/set-running"] = 0  # starts the engine?

        self.fdm["ic/q-rad_sec"] = 0  # Pitch rate initial condition in radians/second
        self.fdm["ic/p-rad_sec"] = 0  # Roll rate initial condition in radians/second
        self.fdm["ic/r-rad_sec"] = 0  # Yaw rate initial condition in radians/second

        # client.sendDREF("sim/flightmodel/position/local_ax", 0)  # The acceleration in local OGL coordinates +ax=E -ax=W
        # client.sendDREF("sim/flightmodel/position/local_ay", 0)  # The acceleration in local OGL coordinates +=Vertical (up)
        # client.sendDREF("sim/flightmodel/position/local_az", 0)  # The acceleration in local OGL coordinates -az=S +az=N

    def getVelo(self):

        P = self.fdm["velocities/p-rad_sec"] * self.radToDeg  # The roll rotation rates
        Q = self.fdm["velocities/q-rad_sec"] * self.radToDeg  # The pitch rotation rates
        R = self.fdm["velocities/r-rad_sec"] * self.radToDeg  # The yaw rotation rates
        AoA = self.fdm["aero/alpha-deg"]  # The angle of Attack
        AoS = self.fdm["aero/beta-deg"]  # The angle of Slip
        values = [P, Q, R, AoA, AoS]

        return values

    def send_Ctrl(self, ctrl):
        '''
        ctrl[0]: + Stick in (elevator pointing down) / - Stick back (elevator pointing up)
        ctrl[1]: + Stick right (right aileron up) / - Stick left (left aileron up)
        ctrl[2]: + Peddal (Rudder) left / - Peddal (Rudder) right
        '''
        self.fdm["fcs/elevator-cmd-norm"] = -ctrl[0]  # Elevator control (stick in/out)?
        self.fdm["fcs/aileron-cmd-norm"] = ctrl[1]  # Aileron control (stick left/right)? might need to switch
        self.fdm["fcs/rudder-cmd-norm"] = -ctrl[2]  # Rudder control (peddals)
        self.fdm["fcs/throttle-cmd-norm"] = ctrl[3]  # throttle

    def get_Posi(self):
        lat = self.fdm["position/lat-gc-deg"]  # Latitude
        long = self.fdm["position/long-gc-deg"]  # Longitude
        alt = self.fdm["position/h-sl-ft"]  # altitude

        pitch = self.fdm["attitude/theta-deg"]  # pitch
        roll = self.fdm["attitude/phi-deg"]  # roll
        heading = self.fdm["attitude/psi-deg"]  # yaw

        r = [lat, long, alt, pitch, roll, heading]

        return r

    def getControl(self, action, observation):
        ctrl, actions_binary = self.scenario.getControl(action, observation)

        return ctrl, actions_binary

    def getDeepState(self, observation):
        velocities = self.getVelo()
        positions = observation[3:]
        state = self.scenario.getDeepState(velocities, positions)
        return state

    def getState(self, observation):
        state = self.scenario.getState(observation)
        return state

    def rewardFunction(self, action, newObservation):
        alt = self.fdm["position/h-agl-ft"]
        alpha = self.fdm["aero/alpha-deg"]
        reward, done = self.scenario.rewardFunction(action, newObservation, alt, alpha)
        return reward, done

    def step(self, action):
        position = self.get_Posi()

        newCtrl, actions_binary = self.getControl(action, position)

        self.send_Ctrl(newCtrl)
        for i in range(int(self.pauseDelay * self.physicsPerSec)):  # will mean that the input will be applied for pauseDelay seconds
            # If realTime is True, then the sim will slow down to real time, should only be used for viewing/debugging, not for training
            if(self.realTime):
                self.send_Ctrl(newCtrl)
                self.fdm.run()
                time.sleep(self.realTimeDelay)
            # Non realTime code: this is default
            else:
                self.send_Ctrl(newCtrl)
                self.fdm.run()

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
        resetPosition = self.scenario.resetStartingPosition()
        self.startingOrientation = resetPosition
        self.send_posi(self.startingPosition, resetPosition)
        self.send_velo(resetPosition)

        self.fdm.run_ic()
        self.scenario.resetStateDepth()
        self.send_Ctrl([0, 0, 0, 0, 0, 0, 1])  # this means it will not control the stick during the reset
        new_posi = self.get_Posi()
        if self.qID == "deep" or self.qID == "doubleDeep":
            state = self.getDeepState(new_posi)
        else:
            state = self.getState(new_posi)
        return state
