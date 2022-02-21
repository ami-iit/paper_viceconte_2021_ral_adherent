# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
from typing import List
from dataclasses import dataclass, field
from gym_ignition.rbd.idyntree import numpy
from adherent.data_processing import utils
from gym_ignition.rbd.conversions import Quaternion
from gym_ignition.rbd.idyntree import kindyncomputations


@dataclass
class GlobalFrameFeatures:
    """Class for the global features associated to each retargeted frame."""

    # Features computation
    ik_solutions: List
    dt_mean: float
    kindyn: kindyncomputations.KinDynComputations
    frontal_base_dir: List
    frontal_chest_dir: List

    # Features storage
    base_positions: List = field(default_factory=list)
    ground_base_directions: List = field(default_factory=list)
    ground_chest_directions: List = field(default_factory=list)
    facing_directions: List = field(default_factory=list)
    base_velocities: List = field(default_factory=list)
    base_angular_velocities: List = field(default_factory=list)
    s: List = field(default_factory=list)
    s_dot: List = field(default_factory=list)

    @staticmethod
    def build(ik_solutions: List,
              dt_mean: float,
              kindyn: kindyncomputations.KinDynComputations,
              frontal_base_dir: List,
              frontal_chest_dir: List) -> "GlobalFrameFeatures":
        """Build an empty GlobalFrameFeatures."""

        return GlobalFrameFeatures(ik_solutions=ik_solutions,
                                   dt_mean=dt_mean,
                                   kindyn=kindyn,
                                   frontal_base_dir=frontal_base_dir,
                                   frontal_chest_dir=frontal_chest_dir)

    def reset_robot_configuration(self, joint_positions: List, base_position: List, base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = numpy.FromNumPy.to_idyntree_transform(
            position=np.array(base_position),
            quaternion=np.array(base_quaternion)).asHomogeneousTransform().toNumPy()

        self.kindyn.set_robot_state(s=joint_positions, ds=np.zeros(len(joint_positions)), world_H_base=world_H_base)

    def compute_global_frame_features(self) -> None:
        """Extract global features associated to each retargeted frame"""

        # Debug
        print("Computing global frame features")

        # Subsampling (discard one ik solution over two)
        for frame_idx in range(0, len(self.ik_solutions), 2):

            ik_solution = self.ik_solutions[frame_idx]

            # Retrieve the base pose and the joint positions
            joint_positions = np.asarray(ik_solution["joint_positions"])
            base_position = np.asarray(ik_solution["base_position"])
            base_quaternion = np.asarray(ik_solution["base_quaternion"])

            # Reset the robot configuration
            self.reset_robot_configuration(joint_positions=joint_positions,
                                           base_position=base_position,
                                           base_quaternion=base_quaternion)

            # Base position
            self.base_positions.append(base_position)

            # Ground base direction
            base_rotation = Quaternion.to_rotation(np.array(base_quaternion))
            base_direction = base_rotation.dot(self.frontal_base_dir) # we are interested in the frontal base direction
            ground_base_direction = [base_direction[0], base_direction[1]] # project on the ground
            ground_base_direction = ground_base_direction / np.linalg.norm(ground_base_direction) # of unitary norm
            self.ground_base_directions.append(ground_base_direction)

            # Ground chest direction
            world_H_base = self.kindyn.get_world_base_transform()
            base_H_chest = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="chest")
            world_H_chest = world_H_base.dot(base_H_chest)
            chest_rotation = world_H_chest[0:3, 0:3]
            chest_direction = chest_rotation.dot(self.frontal_chest_dir) # we are interested in the frontal chest direction
            ground_chest_direction = [chest_direction[0], chest_direction[1]] # project on the ground
            ground_chest_direction = ground_chest_direction / np.linalg.norm(ground_chest_direction) # of unitary norm
            self.ground_chest_directions.append(ground_chest_direction)

            # Facing direction
            facing_direction = ground_base_direction + ground_chest_direction # mean of ground base and chest directions
            facing_direction = facing_direction / np.linalg.norm(facing_direction) # of unitary norm
            self.facing_directions.append(facing_direction)

            # Joint angles
            joint_angles = joint_positions
            self.s.append(joint_angles)

            # Do not compute velocities by differentiation for the first frame
            if frame_idx == 0:
                continue

            # Joint velocities by differentiation of joint angles
            joint_angles_prev = self.s[-2]
            joint_velocities = (joint_angles - joint_angles_prev) / self.dt_mean
            self.s_dot.append(joint_velocities)

            # Base velocities by differentiation of base positions
            base_position_prev = self.base_positions[-2]
            base_velocity = (base_position - base_position_prev) / self.dt_mean
            self.base_velocities.append(base_velocity)

            # Base angular velocities by differentiation of ground base directions
            ground_base_direction_prev = self.ground_base_directions[-2]
            cos_theta = np.dot(ground_base_direction_prev, ground_base_direction) # unitary norm vectors
            sin_theta = np.cross(ground_base_direction_prev, ground_base_direction) # unitary norm vectors
            theta = math.atan2(sin_theta, cos_theta)
            base_angular_velocity = theta / self.dt_mean
            self.base_angular_velocities.append(base_angular_velocity)


@dataclass
class GlobalWindowFeatures:
    """Class for the global features associated to a window of retargeted frames."""

    # Features computation
    window_length_frames: int
    window_step: int
    window_indexes: List

    # Features storage
    desired_velocities: List = field(default_factory=list)
    base_positions: List = field(default_factory=list)
    facing_directions: List = field(default_factory=list)
    base_velocities: List = field(default_factory=list)

    @staticmethod
    def build(window_length_frames: int,
              window_step: int,
              window_indexes: List) -> "GlobalWindowFeatures":
        """Build an empty GlobalWindowFeatures."""

        return GlobalWindowFeatures(window_length_frames=window_length_frames,
                                    window_step=window_step,
                                    window_indexes=window_indexes)

    def compute_global_window_features(self, global_frame_features: GlobalFrameFeatures) -> None:
        """Extract global features associated to a window of retargeted frames."""

        # Debug
        print("Computing global window features")

        initial_frame = self.window_length_frames
        final_frame = len(global_frame_features.base_positions) - self.window_length_frames - self.window_step - 1

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            # Initialize placeholders for the current window
            future_traj_length = 0
            current_global_base_positions = []
            current_global_facing_directions = []
            current_global_base_velocities = []

            for window_index in self.window_indexes:

                # Store the base positions, facing directions and base velocities in the current window
                current_global_base_positions.append(global_frame_features.base_positions[i + window_index])
                current_global_facing_directions.append(global_frame_features.facing_directions[i + window_index])
                current_global_base_velocities.append(global_frame_features.base_velocities[i + window_index])

                # Compute the desired velocity as sum of distances between the base positions in the future trajectory
                if window_index == self.window_indexes[0]:
                    base_position_prev = global_frame_features.base_positions[i + window_index]
                else:
                    base_position = global_frame_features.base_positions[i + window_index]
                    base_position_distance = np.linalg.norm(base_position - base_position_prev)
                    future_traj_length += base_position_distance
                    base_position_prev = base_position

            # Store global features for the current window
            self.desired_velocities.append(future_traj_length)
            self.base_positions.append(current_global_base_positions)
            self.facing_directions.append(current_global_facing_directions)
            self.base_velocities.append(current_global_base_velocities)


@dataclass
class LocalFrameFeatures:
    """Class for the local features associated to each retargeted frame."""

    # Features storage
    base_x_velocities: List = field(default_factory=list)
    base_z_velocities: List = field(default_factory=list)
    base_angular_velocities: List = field(default_factory=list)

    @staticmethod
    def build() -> "LocalFrameFeatures":
        """Build an empty LocalFrameFeatures."""

        return LocalFrameFeatures()

    def compute_local_frame_features(self, global_frame_features: GlobalFrameFeatures) -> None:
        """Extract local features associated to each retargeted frame"""

        # Debug
        print("Computing local frame features")

        # The definition of the base angular velocities is such that they coincide locally and globally
        self.base_angular_velocities = global_frame_features.base_angular_velocities

        for i in range(1, len(global_frame_features.base_positions)):

            # Retrieve the base position and orientation at the previous step i - 1
            # along with the base velocity from step i-1 to step i
            prev_global_base_position = global_frame_features.base_positions[i - 1]
            prev_global_ground_base_direction = global_frame_features.ground_base_directions[i - 1]
            current_global_base_velocity = [global_frame_features.base_velocities[i - 1][0],
                                            global_frame_features.base_velocities[i - 1][1]]

            # Define the 2D local reference frame at step i-1 using the base position and orientation
            reference_base_pos = np.asarray([prev_global_base_position[0], prev_global_base_position[1]])
            reference_ground_base_dir = prev_global_ground_base_direction

            # Retrieve the angle theta between the reference ground base direction and the world x axis
            world_x_axis = np.asarray([1, 0])
            cos_theta = np.dot(world_x_axis, reference_ground_base_dir) # unitary norm vectors
            sin_theta = np.cross(world_x_axis, reference_ground_base_dir) # unitary norm vectors
            theta = math.atan2(sin_theta, cos_theta)

            # Retrieve the rotation from the reference ground base direction to the world frame and its inverse
            world_R_reference_ground_base_dir = utils.rotation_2D(theta)
            reference_ground_base_dir_R_world = np.linalg.inv(world_R_reference_ground_base_dir)

            # Compute base linear velocity in the local reference frame
            current_local_base_velocity = reference_ground_base_dir_R_world.dot(current_global_base_velocity)

            # Store by components the base linear velocity in the local reference frame
            self.base_x_velocities.append(current_local_base_velocity[0])
            self.base_z_velocities.append(current_local_base_velocity[1])


@dataclass
class LocalWindowFeatures:
    """Class for the local features associated to a window of retargeted frames."""

    # Features computation
    window_length_frames: int
    window_step: int
    window_indexes: List

    # Features storage
    base_positions: List = field(default_factory=list)
    facing_directions: List = field(default_factory=list)
    base_velocities: List = field(default_factory=list)

    @staticmethod
    def build(window_length_frames: int,
              window_step: int,
              window_indexes: List) -> "LocalWindowFeatures":
        """Build an empty GlobalWindowFeatures."""

        return LocalWindowFeatures(window_length_frames=window_length_frames,
                                   window_step=window_step,
                                   window_indexes=window_indexes)

    def compute_local_window_features(self, global_window_features: GlobalWindowFeatures) -> None:
        """Extract local features associated to a window of retargeted frames."""

        # Debug
        print("Computing local window features")

        # For each window of retargeted frames
        for i in range(len(global_window_features.base_positions)):

            # Store the global features associated to the currently-considered window of retargeted frames
            current_global_base_positions = global_window_features.base_positions[i]
            current_global_facing_directions = global_window_features.facing_directions[i]
            current_global_base_velocities = global_window_features.base_velocities[i]

            # Placeholders for the local features associated to the currently-considered window of retargeted frames
            current_local_base_positions = []
            current_local_facing_directions = []
            current_local_base_velocities = []

            # Find the current reference frame with respect to which the local quantities will be expressed
            for j in range(len(current_global_base_positions)):

                # The current reference frame is defined by the central frame of the window. Skip the others
                if global_window_features.window_indexes[j] != 0:
                    continue

                # Store the reference base position and facing direction representing the current reference frame
                reference_base_pos = current_global_base_positions[j][:2]
                reference_facing_dir = current_global_facing_directions[j]

                # Retrieve the angle between the reference facing direction and the world x axis
                world_x_axis = np.asarray([1, 0])
                cos_theta = np.dot(world_x_axis, reference_facing_dir) # unitary norm vectors
                sin_theta = np.cross(world_x_axis, reference_facing_dir) # unitary norm vectors
                theta = math.atan2(sin_theta, cos_theta)

                # Retrieve the rotation from the facing direction to the world frame and its inverse
                world_R_facing = utils.rotation_2D(theta)
                facing_R_world = np.linalg.inv(world_R_facing)

            for j in range(len(current_global_base_positions)):

                # Retrieve global features
                current_global_base_pos = current_global_base_positions[j][0:2]
                current_global_facing_dir = current_global_facing_directions[j]
                current_global_base_vel = current_global_base_velocities[j][0:2]

                # Express them locally
                current_local_base_pos = facing_R_world.dot(current_global_base_pos - reference_base_pos)
                current_local_facing_dir = facing_R_world.dot(current_global_facing_dir)
                current_local_base_vel = facing_R_world.dot(current_global_base_vel)

                # Fill the placeholders for the local features associated to the current window
                current_local_base_positions.append(current_local_base_pos)
                current_local_facing_directions.append(current_local_facing_dir)
                current_local_base_velocities.append(current_local_base_vel)

            # Store local features for the current window
            self.base_positions.append(current_local_base_positions)
            self.facing_directions.append(current_local_facing_directions)
            self.base_velocities.append(current_local_base_velocities)


@dataclass
class FeaturesExtractor:
    """Class for the extracting features from retargeted mocap data."""

    global_frame_features: GlobalFrameFeatures
    global_window_features: GlobalWindowFeatures
    local_frame_features: LocalFrameFeatures
    local_window_features: LocalWindowFeatures

    @staticmethod
    def build(ik_solutions: List,
              kindyn: kindyncomputations.KinDynComputations,
              frontal_base_dir: List,
              frontal_chest_dir: List,
              dt_mean: float = 1/50,
              window_length_s: float = 1,
              window_granularity_s: float = 0.2) -> "FeaturesExtractor":
        """Build a FeaturesExtractor."""

        # Define the lenght, expressed in frames, of the window of interest (default=50)
        window_length_frames = round(window_length_s / dt_mean)

        # Define the step, expressed in frames, between the relevant time instants in the window of interest (default=10)
        window_step = round(window_length_frames * window_granularity_s)

        # Define the indexes, expressed in frames, of the relevant time instants in the window of interest (default = [-50, -40, ... , 0, ..., 50, 60])
        window_indexes = list(range(-window_length_frames, window_length_frames + 2 * window_step, window_step))

        # Instantiate all the features
        gff = GlobalFrameFeatures.build(ik_solutions=ik_solutions,
                                        dt_mean=dt_mean,
                                        kindyn=kindyn,
                                        frontal_base_dir=frontal_base_dir,
                                        frontal_chest_dir=frontal_chest_dir)
        gwf = GlobalWindowFeatures.build(window_length_frames=window_length_frames,
                                         window_step=window_step,
                                         window_indexes=window_indexes)
        lff = LocalFrameFeatures.build()
        lwf = LocalWindowFeatures.build(window_length_frames=window_length_frames,
                                        window_step=window_step,
                                        window_indexes=window_indexes)

        return FeaturesExtractor(global_frame_features=gff,
                                 global_window_features=gwf,
                                 local_frame_features=lff,
                                 local_window_features=lwf)

    def compute_features(self) -> None:
        """Compute all the features."""

        self.global_frame_features.compute_global_frame_features()
        self.global_window_features.compute_global_window_features(global_frame_features=self.global_frame_features)
        self.local_frame_features.compute_local_frame_features(global_frame_features=self.global_frame_features)
        self.local_window_features.compute_local_window_features(global_window_features=self.global_window_features)

    def compute_X(self) -> List:
        """Generate the network input vector X."""

        window_length_frames = self.global_window_features.window_length_frames
        window_step = self.global_window_features.window_step
        initial_frame = window_length_frames
        final_frame = len(self.global_frame_features.base_positions) - window_length_frames - window_step - 2

        # Initialize input vector
        X = []

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            # Initialize current input vector
            X_i = []

            # Add current local base positions (24 components)
            current_local_base_positions = []
            for local_base_position in self.local_window_features.base_positions[i - window_length_frames]:
                current_local_base_positions.extend(local_base_position)
            X_i.extend(current_local_base_positions)

            # Add current local facing directions (24 components)
            current_local_facing_directions = []
            for local_facing_direction in self.local_window_features.facing_directions[i - window_length_frames]:
                current_local_facing_directions.extend(local_facing_direction)
            X_i.extend(current_local_facing_directions)

            # Add current local base velocities (24 components)
            current_local_base_velocities = []
            for local_base_velocity in self.local_window_features.base_velocities[i - window_length_frames]:
                current_local_base_velocities.extend(local_base_velocity)
            X_i.extend(current_local_base_velocities)

            # Add current desired velocity (1 component)
            current_desired_velocity = [self.global_window_features.desired_velocities[i - window_length_frames]]
            X_i.extend(current_desired_velocity)

            # Add previous joint positions (32 components)
            prev_s = self.global_frame_features.s[i - 1]
            X_i.extend(prev_s)

            # Add previous joint velocities (32 components)
            prev_s_dot = self.global_frame_features.s_dot[i - 2]
            X_i.extend(prev_s_dot)

            # Store current input vector (137 components)
            X.append(X_i)

        # Debug
        print("X size:", len(X), "x", len(X[0]))

        return X

    def compute_Y(self) -> List:
        """Generate the network output vector Y."""

        window_length_frames = self.global_window_features.window_length_frames
        window_step = self.global_window_features.window_step
        window_indexes = self.global_window_features.window_indexes
        initial_frame = window_length_frames
        final_frame = len(self.global_frame_features.base_positions) - window_length_frames - window_step - 2

        # Initialize output vector
        Y = []

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            # Initialize current input vector
            Y_i = []

            # Add future local base positions (12 components)
            next_local_base_positions = []
            for j in range(len(self.local_window_features.base_positions[i - window_length_frames + 1])):
                if window_indexes[j] > 0:
                    next_local_base_positions.extend(self.local_window_features.base_positions[i - window_length_frames + 1][j])
            Y_i.extend(next_local_base_positions)

            # Add future local facing directions (12 components)
            next_local_facing_directions = []
            for j in range(len(self.local_window_features.facing_directions[i - window_length_frames + 1])):
                if window_indexes[j] > 0:
                    next_local_facing_directions.extend(self.local_window_features.facing_directions[i - window_length_frames + 1][j])
            Y_i.extend(next_local_facing_directions)

            # Add future local base velocities (12 components)
            next_local_base_velocities = []
            for j in range(len(self.local_window_features.base_velocities[i - window_length_frames + 1])):
                if window_indexes[j] > 0:
                    next_local_base_velocities.extend(self.local_window_features.base_velocities[i - window_length_frames + 1][j])
            Y_i.extend(next_local_base_velocities)

            # Add current joint positions (32 components)
            current_s = self.global_frame_features.s[i]
            Y_i.extend(current_s)

            # Add current joint velocities (32 components)
            current_s_dot = self.global_frame_features.s_dot[i - 1]
            Y_i.extend(current_s_dot)

            # Add current local base x linear velocity component (1 component)
            current_local_base_x_velocity = [self.local_frame_features.base_x_velocities[i - 1]]
            Y_i.extend(current_local_base_x_velocity)

            # Add current local base y linear velocity component (1 component)
            current_local_base_z_velocity = [self.local_frame_features.base_z_velocities[i - 1]]
            Y_i.extend(current_local_base_z_velocity)

            # Add current local base angular velocity (1 component)
            current_local_base_angular_velocity = [self.local_frame_features.base_angular_velocities[i - 1]]
            Y_i.extend(current_local_base_angular_velocity)

            # Store current output vector (103 components)
            Y.append(Y_i)

        # Debug
        print("Y size:", len(Y), "x", len(Y[0]))

        return Y

    def get_global_window_features(self) -> GlobalWindowFeatures:
        """Get the global features associated to a window of retargeted frames."""

        return self.global_window_features

    def get_local_window_features(self) -> LocalWindowFeatures:
        """Get the local features associated to a window of retargeted frames."""

        return self.local_window_features
