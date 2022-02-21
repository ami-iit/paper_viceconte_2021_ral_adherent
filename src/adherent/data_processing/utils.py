# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import json
import numpy as np
from typing import List
from scenario import core
from scenario import gazebo as scenario
from gym_ignition.rbd.idyntree.inverse_kinematics_nlp import IKSolution

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =====================
# MODEL INSERTION UTILS
# =====================

class iCub(core.Model):
    """Helper class to simplify model insertion."""

    def __init__(self,
                 world: scenario.World,
                 urdf: str,
                 position: List[float] = (0., 0, 0),
                 orientation: List[float] = (1., 0, 0, 0)):

        # Insert the model in the world
        name = "iCub"
        pose = core.Pose(position, orientation)
        world.insert_model(urdf, pose, name)

        # Get and store the model from the world
        self.model = world.get_model(model_name=name)

    def __getattr__(self, name):
        return getattr(self.model, name)

# =================
# RETARGETING UTILS
# =================

def define_robot_to_target_base_quat(robot: str) -> List:
    """Define the robot-specific quaternions from the robot base frame to the target base frame."""

    if robot != "iCubV2_5":
        raise Exception("Quaternions from the robot to the target base frame only defined for iCubV2_5.")

    # For iCubV2_5, the robot base frame is rotated of -180 degs on z w.r.t. the target base frame
    robot_to_target_base_quat = [0, 0, 0, -1.0]

    return robot_to_target_base_quat

def define_foot_vertices(robot: str) -> List:
    """Define the robot-specific positions of the feet vertices in the foot frame."""

    if robot != "iCubV2_5":
        raise Exception("Feet vertices positions only defined for iCubV2_5.")

    # For iCubV2_5, the feet vertices are not symmetrically placed wrt the foot frame origin.
    # The foot frame has z pointing down, x pointing forward and y pointing right.

    # Origin of the box which represents the foot (in the foot frame)
    box_origin = [0.03, 0.005, 0.014]

    # Size of the box which represents the foot
    box_size = [0.16, 0.072, 0.001]

    # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
    FL_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
    FR_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]
    BL_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
    BR_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]

    # Vertices positions in the foot (F) frame
    F_vertices_pos = [FL_vertex_pos, FR_vertex_pos, BL_vertex_pos, BR_vertex_pos]

    return F_vertices_pos

def quaternion_multiply(quat1: List, quat2: List) -> np.array:
    """Auxiliary function for quaternion multiplication."""

    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    res = np.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                     x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                     -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                     x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])

    return res

def to_xyzw(wxyz: List) -> List:
    """Auxiliary function to convert quaternions from wxyz to xyzw format."""

    return wxyz[[1, 2, 3, 0]]

def store_retargeted_mocap_as_json(timestamps: List, ik_solutions: List, outfile_name: str) -> None:
    """Auxiliary function to store the retargeted motion."""

    ik_solutions_json = []

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        ik_solution_json = {"joint_positions": ik_solution.joint_configuration.tolist(),
                            "base_position": ik_solution.base_position.tolist(),
                            "base_quaternion": ik_solution.base_quaternion.tolist(),
                            "timestamp": timestamps[i]}

        ik_solutions_json.append(ik_solution_json)

    with open(outfile_name, "w") as outfile:
        json.dump(ik_solutions_json, outfile)

def load_retargeted_mocap_from_json(input_file_name: str, initial_frame: int = 0, final_frame: int = -1) -> (List, List):
    """Auxiliary function to load the retargeted mocap data."""

    # Load ik solutions
    with open(input_file_name, 'r') as openfile:
        ik_solutions = json.load(openfile)

    # If a final frame has been passed, extract relevant ik solutions
    if initial_frame != -1:
        ik_solutions = ik_solutions[initial_frame:final_frame]

    # Extract timestamps
    timestamps = [ik_solution["timestamp"] for ik_solution in ik_solutions]

    return timestamps, ik_solutions

# =========================
# FEATURES EXTRACTION UTILS
# =========================

def define_frontal_base_direction(robot: str) -> List:
    """Define the robot-specific frontal base direction in the base frame."""

    if robot != "iCubV2_5":
        raise Exception("Frontal base direction only defined for iCubV2_5.")

    # For iCubV2_5, the reversed x axis of the base frame is pointing forward
    frontal_base_direction = [-1, 0, 0]

    return frontal_base_direction

def define_frontal_chest_direction(robot: str) -> List:
    """Define the robot-specific frontal chest direction in the chest frame."""

    if robot != "iCubV2_5":
        raise Exception("Frontal chest direction only defined for iCubV2_5.")

    # For iCubV2_5, the z axis of the chest frame is pointing forward
    frontal_base_direction = [0, 0, 1]

    return frontal_base_direction

def rotation_2D(angle: float) -> np.array:
    """Auxiliary function for a 2-dimensional rotation matrix."""

    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_retargeted_motion(timestamps: List,
                                ik_solutions: List,
                                icub: iCub,
                                controlled_joints: List,
                                gazebo: scenario.GazeboSimulator) -> None:
    """Auxiliary function to visualize retargeted motion."""

    timestamp_prev = -1

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        # Retrieve the base pose and the joint positions, based on the type of ik_solution
        if type(ik_solution) == IKSolution:
            joint_positions = ik_solution.joint_configuration
            base_position = ik_solution.base_position
            base_quaternion = ik_solution.base_quaternion
        elif type(ik_solution) == dict:
            joint_positions = ik_solution["joint_positions"]
            base_position = ik_solution["base_position"]
            base_quaternion = ik_solution["base_quaternion"]

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Visualize the retargeted motion at the time rate of the collected data
        timestamp = timestamps[i]
        if timestamp_prev == -1:
            dt = 1 / 100
        else:
            dt = timestamp - timestamp_prev
        time.sleep(dt)
        timestamp_prev = timestamp

    print("Visualization ended")
    time.sleep(1)

def visualize_global_features(global_window_features,
                              ik_solutions: List,
                              icub: iCub,
                              controlled_joints: List,
                              gazebo: scenario.GazeboSimulator,
                              plot_facing_directions: bool = True,
                              plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated global features."""

    window_length_frames = global_window_features.window_length_frames
    window_step = global_window_features.window_step
    window_indexes = global_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve global features
        base_positions = global_window_features.base_positions[i - window_length_frames]
        facing_directions = global_window_features.facing_directions[i - window_length_frames]
        base_velocities = global_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([base_position[1], base_position[1] + 2 * facing_direction[1]],
                             [-base_position[0], -base_position[0] - 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([base_position[1], base_position[1] + facing_direction[1]],
                             [-base_position[0], -base_position[0] - facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Facing directions (global view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Base velocities (global view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)

def visualize_local_features(local_window_features,
                             ik_solutions: List,
                             icub: iCub,
                             controlled_joints: List,
                             gazebo: scenario.GazeboSimulator,
                             plot_facing_directions: bool = True,
                             plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated local features."""

    window_length_frames = local_window_features.window_length_frames
    window_step = local_window_features.window_step
    window_indexes = local_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve local features
        base_positions = local_window_features.base_positions[i - window_length_frames]
        facing_directions = local_window_features.facing_directions[i - window_length_frames]
        base_velocities = local_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([-base_position[1], -base_position[1] - 2 * facing_direction[1]],
                             [base_position[0], base_position[0] + 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([-base_position[1], -base_position[1] - facing_direction[1]],
                             [base_position[0], base_position[0] + facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Facing directions (local view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Base velocities (local view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)
