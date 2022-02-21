# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Define as default data_path the latest simulation folder in which data have been saved
list_of_files = glob.glob('../datasets/trajectory_control_simulation/*')
latest_file = max(list_of_files, key=os.path.getctime)

parser.add_argument("--data_path", help="Path where the data are stored. Relative path from the script folder.",
                    type = str, default = latest_file + "/data.txt")
parser.add_argument("--plot_CoM_ZMP_DCM", help="Produce plots related to CoM, ZMP and DCM.", action="store_true")
parser.add_argument("--plot_feet_cartesian_tracking", help="Produce plots related to feet cartesian tracking.", action="store_true")
parser.add_argument("--plot_feet_wrenches", help="Produce plots related to feet wrenches.", action="store_true")
parser.add_argument("--plot_joints_position_tracking", help="Produce plots related to joint position tracking.", action="store_true")
parser.add_argument("--plot_joints_velocity_tracking", help="Produce plots related to joint velocity tracking.", action="store_true")

args = parser.parse_args()

data_path = args.data_path
plot_CoM_ZMP_DCM = args.plot_CoM_ZMP_DCM
plot_feet_cartesian_tracking = args.plot_feet_cartesian_tracking
plot_feet_wrenches = args.plot_feet_wrenches
plot_joints_position_tracking = args.plot_joints_position_tracking
plot_joints_velocity_tracking = args.plot_joints_velocity_tracking

# =============
# RETRIEVE DATA
# =============

# Load data
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_directory, data_path)
with open(data_path, 'r') as openfile:
    data = json.load(openfile)

# CoM position
com_pos_des_x = data['com_pos_des_x']
com_pos_des_y = data['com_pos_des_y']
com_pos_meas_x = data['com_pos_meas_x']
com_pos_meas_y = data['com_pos_meas_y']

# ZMP position
zmp_pos_des_x = data['zmp_pos_des_x']
zmp_pos_des_y = data['zmp_pos_des_y']
zmp_pos_meas_x = data['zmp_pos_meas_x']
zmp_pos_meas_y = data['zmp_pos_meas_y']

# DCM position
dcm_pos_des_x = data['dcm_pos_des_x']
dcm_pos_des_y = data['dcm_pos_des_y']
dcm_pos_meas_x = data['dcm_pos_meas_x']
dcm_pos_meas_y = data['dcm_pos_meas_y']

# CoM velocity
com_vel_des_from_dcm_x = data['com_vel_des_from_dcm_x']
com_vel_des_from_dcm_y = data['com_vel_des_from_dcm_y']
com_vel_des_x = data['com_vel_des_x']
com_vel_des_y = data['com_vel_des_y']
com_vel_meas_x = data['com_vel_meas_x']
com_vel_meas_y = data['com_vel_meas_y']

# DCM velocity
dcm_vel_des_x = data['dcm_vel_des_x']
dcm_vel_des_y = data['dcm_vel_des_y']

# Right foot cartesian tracking
right_foot_des_x = data['right_foot_des_x']
right_foot_des_y = data['right_foot_des_y']
right_foot_des_z = data['right_foot_des_z']
right_foot_meas_x = data['right_foot_meas_x']
right_foot_meas_y = data['right_foot_meas_y']
right_foot_meas_z = data['right_foot_meas_z']
right_foot_des_R = data['right_foot_des_R']
right_foot_des_P = data['right_foot_des_P']
right_foot_des_Y = data['right_foot_des_Y']
right_foot_meas_R = data['right_foot_meas_R']
right_foot_meas_P = data['right_foot_meas_P']
right_foot_meas_Y = data['right_foot_meas_Y']

# Left foot cartesian tracking
left_foot_des_x = data['left_foot_des_x']
left_foot_des_y = data['left_foot_des_y']
left_foot_des_z = data['left_foot_des_z']
left_foot_meas_x = data['left_foot_meas_x']
left_foot_meas_y = data['left_foot_meas_y']
left_foot_meas_z = data['left_foot_meas_z']
left_foot_des_R = data['left_foot_des_R']
left_foot_des_P = data['left_foot_des_P']
left_foot_des_Y = data['left_foot_des_Y']
left_foot_meas_R = data['left_foot_meas_R']
left_foot_meas_P = data['left_foot_meas_P']
left_foot_meas_Y = data['left_foot_meas_Y']

# Left wrenches
left_force_x = data['left_force_x']
left_force_y = data['left_force_y']
left_force_z = data['left_force_z']
left_torque_x = data['left_torque_x']
left_torque_y = data['left_torque_y']
left_torque_z = data['left_torque_z']

# Right wrenches
right_force_x = data['right_force_x']
right_force_y = data['right_force_y']
right_force_z = data['right_force_z']
right_torque_x = data['right_torque_x']
right_torque_y = data['right_torque_y']
right_torque_z = data['right_torque_z']

# Joints tracking
joints_list = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', # left leg
               'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll', # right leg
               'torso_pitch', 'torso_roll', 'torso_yaw', # torso
               'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
               'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm
joint_des_storage = []
joint_meas_storage = []
joint_postural_storage = []
joint_vel_des_storage = []
joint_vel_meas_storage = []

for i in range(len(joints_list)):
    joint_des_storage.append([])
    joint_meas_storage.append([])
    joint_postural_storage.append([])
    joint_vel_des_storage.append([])
    joint_vel_meas_storage.append([])

for i in range(len(joints_list)):
    joint_des_storage[i] = data[joints_list[i]]["des"]
    joint_meas_storage[i] = data[joints_list[i]]["meas"]
    joint_vel_des_storage[i] = data[joints_list[i]]["vel_des"]
    joint_vel_meas_storage[i] = data[joints_list[i]]["vel_meas"]
    if "postural" in data[joints_list[i]].keys():
        joint_postural_storage[i] = data[joints_list[i]]["postural"]

# =====================
# COM - ZMP - DCM PLOTS
# =====================

if plot_CoM_ZMP_DCM:

    # CoM position tracking
    plt.figure()
    plt.plot(com_pos_des_x, com_pos_des_y, 'b', label="Desired CoM pos")
    plt.plot(com_pos_meas_x, com_pos_meas_y, 'r', label="Measured CoM pos")
    plt.title("CoM evolution in world frame")
    plt.legend()
    plt.axis('equal')

    # ZMP pos tracking
    plt.figure()
    plt.plot(zmp_pos_des_x, zmp_pos_des_y, 'b', label="Desired ZMP pos")
    plt.plot(zmp_pos_meas_x, zmp_pos_meas_y, 'r', label="Measured ZMP pos")
    plt.title("ZMP evolution in world frame")
    plt.legend()
    plt.axis('equal')

    # DCM position tracking
    plt.figure()
    plt.plot(dcm_pos_des_x, dcm_pos_des_y, 'b', label="Desired DCM pos")
    plt.plot(dcm_pos_meas_x, dcm_pos_meas_y, 'r', label="Measured DCM pos")
    plt.title("DCM evolution in world frame")
    plt.legend()
    plt.axis('equal')

    # CoM measured velocity
    plt.figure()
    plt.plot(list(range(len(com_vel_meas_x))), com_vel_meas_x, 'r', label="x")
    plt.plot(list(range(len(com_vel_meas_y))), com_vel_meas_y, 'b', label="y")
    plt.title("CoM measured velocity")
    plt.legend()

    # CoM desired velocity from the dcm planner
    plt.figure()
    plt.plot(list(range(len(com_vel_des_from_dcm_x))), com_vel_des_from_dcm_x, 'r', label="x")
    plt.plot(list(range(len(com_vel_des_from_dcm_y))), com_vel_des_from_dcm_y, 'b', label="y")
    plt.title("CoM desired velocity from DCM planner")
    plt.legend()

    # CoM desired velocity after the zmp-com controller
    plt.figure()
    plt.plot(list(range(len(com_vel_des_x))), com_vel_des_x, 'r', label="x")
    plt.plot(list(range(len(com_vel_des_y))), com_vel_des_y, 'b', label="y")
    plt.title("CoM desired velocity")
    plt.legend()

    # DCM desired velocity
    plt.figure()
    plt.plot(list(range(len(dcm_vel_des_x))), dcm_vel_des_x, 'r', label="x")
    plt.plot(list(range(len(dcm_vel_des_y))), dcm_vel_des_y, 'b', label="y")
    plt.title("DCM desired velocity")
    plt.legend()

# =============================
# FEET CARTESIAN TRACKING PLOTS
# =============================

if plot_feet_cartesian_tracking:

    # Right foot position tracking
    plt.figure()
    plt.plot(list(range(len(right_foot_des_x))), right_foot_des_x, 'b', label="Desired")
    plt.plot(list(range(len(right_foot_meas_x))), right_foot_meas_x, 'r', label="Measured")
    plt.title("Right foot position tracking (x component)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(right_foot_des_y))), right_foot_des_y, 'b', label="Desired")
    plt.plot(list(range(len(right_foot_meas_y))), right_foot_meas_y, 'r', label="Measured")
    plt.title("Right foot position tracking (y component)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(right_foot_des_z))), right_foot_des_z, 'b', label="Desired")
    plt.plot(list(range(len(right_foot_meas_z))), right_foot_meas_z, 'r', label="Measured")
    plt.title("Right foot position tracking (z component)")
    plt.legend()

    # Left foot position tracking
    plt.figure()
    plt.plot(list(range(len(left_foot_des_x))), left_foot_des_x, 'b', label="Desired")
    plt.plot(list(range(len(left_foot_meas_x))), left_foot_meas_x, 'r', label="Measured")
    plt.title("Left foot position tracking (x component)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(left_foot_des_y))), left_foot_des_y, 'b', label="Desired")
    plt.plot(list(range(len(left_foot_meas_y))), left_foot_meas_y, 'r', label="Measured")
    plt.title("Left foot position tracking (y component)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(left_foot_des_z))), left_foot_des_z, 'b', label="Desired")
    plt.plot(list(range(len(left_foot_meas_z))), left_foot_meas_z, 'r', label="Measured")
    plt.title("Left foot position tracking (z component)")
    plt.legend()

    # Right foot RPY tracking
    plt.figure()
    plt.plot(list(range(len(right_foot_des_R))), right_foot_des_R, 'b', label="Desired")
    plt.plot(list(range(len(right_foot_meas_R))), right_foot_meas_R, 'r', label="Measured")
    plt.title("Right foot orientation tracking (roll)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(right_foot_des_P))), right_foot_des_P, 'r', label="Desired")
    plt.plot(list(range(len(right_foot_meas_P))), right_foot_meas_P, 'b', label="Measured")
    plt.title("Right foot orientation tracking (pitch)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(right_foot_des_Y))), right_foot_des_Y, 'r', label="Desired")
    plt.plot(list(range(len(right_foot_meas_Y))), right_foot_meas_Y, 'b', label="Measured")
    plt.title("Right foot orientation tracking (yaw)")
    plt.legend()

    # Left foot RPY tracking
    plt.figure()
    plt.plot(list(range(len(left_foot_des_R))), left_foot_des_R, 'r', label="Desired")
    plt.plot(list(range(len(left_foot_meas_R))), left_foot_meas_R, 'b', label="Measured")
    plt.title("Left foot orientation tracking (roll)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(left_foot_des_P))), left_foot_des_P, 'r', label="Desired")
    plt.plot(list(range(len(left_foot_meas_P))), left_foot_meas_P, 'b', label="Measured")
    plt.title("Left foot orientation tracking (pitch)")
    plt.legend()

    plt.figure()
    plt.plot(list(range(len(left_foot_des_Y))), left_foot_des_Y, 'r', label="Desired")
    plt.plot(list(range(len(left_foot_meas_Y))), left_foot_meas_Y, 'b', label="Measured")
    plt.title("Left foot orientation tracking (yaw)")
    plt.legend()

# ===================
# FEET WRENCHES PLOTS
# ===================

if plot_feet_wrenches:

    # Left foot forces
    plt.figure()
    plt.plot(list(range(len(left_force_x))), left_force_x, 'r', label='x')
    plt.plot(list(range(len(left_force_y))), left_force_y, 'b', label='y')
    plt.plot(list(range(len(left_force_z))), left_force_z, 'g', label='z')
    plt.title("Left foot forces")
    plt.legend()

    # Left foot torques
    plt.figure()
    plt.plot(list(range(len(left_torque_x))), left_torque_x, 'r', label='x')
    plt.plot(list(range(len(left_torque_y))), left_torque_y, 'b', label='y')
    plt.plot(list(range(len(left_torque_z))), left_torque_z, 'g', label='z')
    plt.title("Left foot torques")
    plt.legend()

    # Right foot forces
    plt.figure()
    plt.plot(list(range(len(right_force_x))), right_force_x, 'r', label='x')
    plt.plot(list(range(len(right_force_y))), right_force_y, 'b', label='y')
    plt.plot(list(range(len(right_force_z))), right_force_z, 'g', label='z')
    plt.title("Right foot forces")
    plt.legend()

    # Right foot torques
    plt.figure()
    plt.plot(list(range(len(right_torque_x))), right_torque_x, 'r', label='x')
    plt.plot(list(range(len(right_torque_y))), right_torque_y, 'b', label='y')
    plt.plot(list(range(len(right_torque_z))), right_torque_z, 'g', label='z')
    plt.title("Right foot torques")
    plt.legend()

# =======================
# JOINT POSITION TRACKING
# =======================

if plot_joints_position_tracking:

    for i in range(len(joints_list)):

        plt.figure()

        # Network-predicted postural (if available)
        if len(joint_postural_storage[0]) > 0:
            plt.plot(list(range(len(joint_postural_storage[i]))), joint_postural_storage[i], 'g', label='Postural')

        # Joint position tracking
        plt.plot(list(range(len(joint_des_storage[i]))), joint_des_storage[i], 'b', label='Desired')
        plt.plot(list(range(len(joint_meas_storage[i]))), joint_meas_storage[i], 'r', label='Measured')

        # Plot configuration
        plt.xlim([0, len(joint_meas_storage[i])])
        mean_value = np.mean(joint_meas_storage[i])
        plt.ylim([mean_value - 30, mean_value + 30])
        plt.title(joints_list[i] + " tracking [deg]")
        plt.legend()

# =======================
# JOINT VELOCITY TRACKING
# =======================

if plot_joints_velocity_tracking:

    for i in range(len(joints_list)):

        plt.figure()

        # Joint velocity tracking
        plt.plot(list(range(len(joint_vel_des_storage[i]))), joint_vel_des_storage[i], 'b', label="Desired")
        plt.plot(list(range(len(joint_vel_meas_storage[i]))), joint_vel_meas_storage[i], 'r', label="Measured")

        # Plot configuration
        plt.xlim([0, len(joint_vel_meas_storage[i])])
        mean_value = np.mean(joint_vel_meas_storage[i])
        plt.ylim([mean_value - 30, mean_value + 30])
        plt.title(joints_list[i] + " velocity tracking [deg]")
        plt.legend()

# ====
# PLOT
# ====

# Plot
plt.show()
plt.pause(1)
