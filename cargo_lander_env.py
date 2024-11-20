import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import tensorflow as tf
import warnings
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random

# Suppress TensorFlow warnings and errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define thrust scaling and directions
THRUST_DIR = np.array([[0, 0, 50]] * 4)  # All engines thrust upward
ENGINE_POS = np.array([
    (1.2, 0, -0.75),        # engine_1
    (-1.2, 0, -0.75),       # engine_2
    (0, 1.2, -0.75),        # engine_3
    (0, -1.2, -0.75)        # engine_4
])

# Define rewards and penalties
GOAL_REWARD = 10000.0
CRASH_PENALTY = -10000.0
FUEL_PENALTY = -0.03

# Define action mappings
dirDict = {
    0: [0, 0, 0, 0],   # No action
    1: [1, 0, 0, 0],   # Thrust at engine 1
    2: [0, 1, 0, 0],   # Thrust at engine 2
    3: [0, 0, 1, 0],   # Thrust at engine 3
    4: [0, 0, 0, 1],   # Thrust at engine 4
    5: [1, 1, 0, 0],   # Thrust at engines 1 and 2
    6: [0, 1, 1, 0],   # Thrust at engines 2 and 3
    7: [0, 0, 1, 1],   # Thrust at engines 3 and 4
    8: [1, 0, 0, 1],   # Thrust at engines 1 and 4
    9: [1, 1, 1, 1],   # Thrust at all engines
}

class CargoLanderEnv(gym.Env):
    def __init__(self):
        super(CargoLanderEnv, self).__init__()
        self.action_space = spaces.Discrete(len(dirDict))
        high = np.array([np.inf] * 13)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -1.62)

        self.planeId = p.loadURDF("plane.urdf")
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -1.62)
        self.time = 0

        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 50]
        startOrientation = p.getQuaternionFromEuler([0.8, 0, 0])
        self.landerId = p.loadURDF("Cargo_lander.urdf", startPos, startOrientation)

        # Set initial height
        self.initial_height = p.getBasePositionAndOrientation(self.landerId)[0][2]

        # Update camera to follow the lander
        self._set_camera(startPos)

        self.done = False
        self.success = False
        self.landing_start_time = None
        self.landing_stable_z = None
        return self._observe()

    def _set_camera(self, target_position):
        """
        Dynamically adjust the camera to follow the lander.
        """
        camera_distance = 10
        camera_yaw = 30
        camera_pitch = -45
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_position
        )

    def _observe(self):
        position, orientation = p.getBasePositionAndOrientation(self.landerId)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.landerId)
        contact = float(len(p.getContactPoints(self.landerId, self.planeId)) > 0)
        euler_orientation = p.getEulerFromQuaternion(orientation)

        obs = np.concatenate((
            np.array(position, dtype=np.float32),
            np.array(linear_velocity, dtype=np.float32),
            np.array(euler_orientation, dtype=np.float32),
            np.array(angular_velocity, dtype=np.float32),
            np.array([contact], dtype=np.float32)
        ))
        return obs

    def step(self, action):
        # Print action for debugging
        print(f'Action: {action}')
        assert self.action_space.contains(action)

        for i, engine_active in enumerate(dirDict[action]):
            if engine_active:
                link_orientation = p.getLinkState(self.landerId, i)[1]  # Get the link world orientation as a quaternion
                print(f'Link {i} Orientation (Quaternion): {link_orientation}')
                thrust_local = [0, 0, 10]  # Thrust vector in local coordinates
                thrust, _ = p.multiplyTransforms([0, 0, 0], link_orientation, thrust_local, [0, 0, 0, 1])  # Get the orientation of the link to apply thrust in local coordinates
                print(f'Thrust (Global): {thrust}')
                p.applyExternalForce(self.landerId, -1, forceObj=thrust, posObj=ENGINE_POS[i], flags=p.WORLD_FRAME)

        p.stepSimulation()

        # Increment time
        self.time += 1 / 240.0  # Assuming PyBullet is running at 240 Hz
        

        # Update camera to follow the lander
        position, _ = p.getBasePositionAndOrientation(self.landerId)
        self._set_camera(position)

        obs = self._observe()
        # Print observation for debugging
        print(f'Observation: {np.round(obs, 3)}')
        reward, fuel_consume, leg_contacts = self._calculate_reward_and_penalty(obs)
        # Print reward, fuel consumption, and leg contacts for debugging
        print(f'Reward: {reward:.3f}, Fuel Consumption: {fuel_consume:.3f}, Leg Contacts: {leg_contacts}')

        return obs, reward, fuel_consume, self.done, {}, leg_contacts

    def _calculate_reward_and_penalty(self, obs):
        position, linear_velocity, euler_orientation, angular_velocity, contact = obs[:3], obs[3:6], obs[6:9], obs[9:12], obs[-1]

        # Initialize reward and penalty
        reward = 0
        penalty = 0

        # Fuel penalty
        fuel_consume = FUEL_PENALTY

        # Linear speed penalty
        penalty += abs(linear_velocity[0]) * 0.5  # Penalty for velocity in X direction
        penalty += abs(linear_velocity[1]) * 0.5  # Penalty for velocity in Y direction
        penalty += abs(linear_velocity[2]) * 8    # Penalty for velocity in Z direction

        # Angular penalty for orientation
        roll_deg = abs(euler_orientation[0]) * (180 / np.pi)
        pitch_deg = abs(euler_orientation[1]) * (180 / np.pi)
        penalty += roll_deg * 10  # Penalty for roll angle deviation
        penalty += pitch_deg * 10  # Penalty for pitch angle deviation

        # Angular speed penalty
        penalty += abs(angular_velocity[0]) * (180 / np.pi) * 20  # Penalty for roll speed
        penalty += abs(angular_velocity[1]) * (180 / np.pi) * 20  # Penalty for pitch speed

        # Contact Stability Reward
        contact_points = p.getContactPoints(self.landerId, self.planeId)
        leg_contact_links = set(cp[3] for cp in contact_points)

        # Minimal Velocity Reward
        if np.linalg.norm(linear_velocity) <= 2.0:
            reward += 50

        # Upright Orientation Reward
        if roll_deg < 3.0 and pitch_deg < 3.0:
            reward += 50

        # Correct Contact Reward
        if leg_contact_links.intersection({5, 6, 7, 8}):
            reward += 100

        # Apply penalties to reward
        reward -= penalty
        print("Contact variable is",contact)

        if contact > 0:
            # Get contact points and identify landing leg contacts
            contact_points = p.getContactPoints(self.landerId, self.planeId)
            leg_contact_links = {cp[3] for cp in contact_points}  # Set of landing legs in contact

            # Check if any of the landing legs (5, 6, 7, 8) are in contact
            if leg_contact_links.intersection({5, 6, 7, 8}):
                if self.landing_start_time is None:
                    # Record the time and initial stable height when the first valid contact occurs
                    self.landing_start_time = self.time
                    self.landing_stable_z = position[2]
                    print("Landing sequence started: 1-second buffer for stabilization.")

                # During the 1-second buffer period
                if self.time - self.landing_start_time <= 1:
                    # Log all state information for debugging
                    print(f"--- Landing Stabilization Check at Time: {self.time:.2f}s ---")
                    print(f"Legs in Contact: {leg_contact_links}")
                    print(f"Position (x, y, z): {np.round(position, 3)}")
                    print(f"Linear Velocity (x, y, z): {np.round(linear_velocity, 3)}")
                    print(f"Angular Velocity (roll, pitch, yaw): {np.round(angular_velocity, 3)}")
                    print(f"Orientation (roll, pitch, yaw in degrees): {np.round(np.rad2deg(euler_orientation), 2)}")
                    print(f"Height Change from Stable Z: {abs(position[2] - self.landing_stable_z):.3f}")

                    # Ensure at least 3 legs are in contact and check for valid combinations
                    valid_combinations = [
                        {5, 6, 7}, {5, 6, 8}, {5, 7, 8}, {6, 7, 8},
                        {5, 6, 7, 8}
                    ]
                    if len(leg_contact_links) >= 3 and any(leg_contact_links.issuperset(comb) for comb in valid_combinations):
                        vertical_speed = abs(linear_velocity[2])
                        angular_speed = np.linalg.norm(angular_velocity)
                        xy_velocity = np.linalg.norm(linear_velocity[:2])
                        height_change = abs(position[2] - self.landing_stable_z)

                        # Check stability criteria
                        if (vertical_speed < 1 and
                            angular_speed < 1 and
                            xy_velocity < 1 and
                            height_change < 0.1):
                            print("--- Stability Achieved ---")
                            reward += GOAL_REWARD
                            self.done = True
                            self.success = True
                            print("Landing Successful!")
                            return reward, fuel_consume, leg_contact_links
                        else:
                            print("--- Stability Conditions Not Met Yet ---")
                else:
                    # After the 1-second buffer has passed, end the episode
                    print("1-second buffer expired: Stabilization conditions not met.")
                    reward += CRASH_PENALTY
                    self.done = True
                    print("Landing Failed: Stability conditions not met within the buffer period.")
            else:
                reward += CRASH_PENALTY
                self.done = True
                print("Landing Failed: No valid landing legs in contact")
        else:
            # Reset the landing timer if no contact is made
            self.landing_start_time = None
            self.landing_stable_z = None
            print("Landing in progress")

        
        # End if time limit is reached
        if self.time >= 10000:
            self.done = True

        fuel_consume += penalty

        return reward, fuel_consume, leg_contact_links

    def close(self):
        p.disconnect()