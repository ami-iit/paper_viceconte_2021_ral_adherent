# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import matplotlib.pyplot as plt
from adherent.trajectory_control.utils import rad2deg

# =============
# CONFIGURATION
# =============

# Path to the data for regenerating Fig.7 in the paper - Adherent Postural (AP)
data_path_AP = "../datasets/additional_figures/fig_7_adherent_postural.txt"

# Path to the data for regenerating Fig.7 in the paper - Fixed Postural (FP)
data_path_FP = "../datasets/additional_figures/fig_7_fixed_postural.txt"

# Joints to plot
joints_to_plot = ['torso_yaw', 'l_shoulder_pitch', 'l_shoulder_yaw', 'l_elbow']

# =============
# RETRIEVE DATA
# =============

script_directory = os.path.dirname(os.path.abspath(__file__))

# Load data (when Adherent Postural is used)
data_path_AP = os.path.join(script_directory, data_path_AP)
with open(data_path_AP, 'r') as openfile:
    data_AP = json.load(openfile)

# Load data (when Fixed Postural is used)
data_path_FP = os.path.join(script_directory, data_path_FP)
with open(data_path_FP, 'r') as openfile:
    data_FP = json.load(openfile)

# Joints list
joints_list = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', # left leg
               'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll', # right leg
               'torso_pitch', 'torso_roll', 'torso_yaw', # torso
               'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
               'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

# Prepare containers
joint_postural_AP = [] # Adherent Postural
joint_meas_AP = [] # Measured joint positions (when Adherent Postural is used)
joint_postural_FP = [] # Fixed Postural
joint_meas_FP = [] # Measured joint positions (when Fixed Postural is used)
for i in range(len(joints_list)):
    joint_postural_AP.append([])
    joint_meas_AP.append([])
    joint_postural_FP.append([])
    joint_meas_FP.append([])

# Fill most of the containers from data
for i in range(len(joints_list)):
    joint_postural_AP[i] = data_AP[joints_list[i]]["postural"]
    joint_meas_AP[i] = data_AP[joints_list[i]]["meas"]
    joint_meas_FP[i] = data_FP[joints_list[i]]["meas"]

# ===========================
# MANUALLY ADD FIXED POSTURAL
# ===========================

# Manually specify the fixed postural, which is not stored in the experimental data
fixed_postural = [0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # left leg
                  0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # right leg
                  0.1388792845, 0.0, 0.0,  # torso
                  -0.0629, 0.4397, 0.1825, 0.5387, # left arm
                  -0.0629, 0.4397, 0.1825, 0.5387] # right arm

# Manually fill the Fixed Postural containers, since the fixed postural is not stored in the experimental data
for i in range(len(joints_list)):
    joint_postural_FP[i] = [rad2deg(fixed_postural[i])] * len(joint_postural_AP[i])

# =====
# PLOTS
# =====

# Specific plot configuration for Fig.7 in the paper: manually define the joint angle which should be centered in each plot
plot_means = {'torso_yaw': 0 , 'l_shoulder_pitch': 0, 'l_shoulder_yaw': 0, 'l_elbow': 24}

# Compare Adherent Postural with Fixed Postural
for i in range(len(joints_list)):

    # Filter on the joints to plot
    if joints_list[i] in joints_to_plot:

        plt.figure()

        # Convert inference calls to time (seconds)
        timesteps = list(range(len(joint_meas_AP[i])))
        time = [timestep / 100.0 for timestep in timesteps]

        # Plot posturals when Adherent Postural (AP) is used
        plt.plot(time, joint_postural_AP[i], 'r', label="Postural (AP)", linewidth=1.0)
        plt.plot(time, joint_meas_AP[i], 'b', label="Measured (AP)", linewidth=1.0)

        # Plot posturals when Fixed Postural (FP) is used
        plt.plot(time, joint_postural_FP[i], 'k', label="Postural (FP)", linewidth=1.0)
        plt.plot(time, joint_meas_FP[i][:len(time)], 'y', label="Measured (FP)", linewidth=1.0)

        # Plot configuration
        plt.xlim([0, time[-1]])
        plt.ylabel("joint position [deg]")
        plt.xlabel("time [s]")
        title = joints_list[i].replace("_"," ").upper()
        plt.title(title)
        plt.legend()

        # Specific plot configuration for Fig.7 in the paper
        if joints_list[i] in plot_means.keys():
            mean_value = plot_means[joints_list[i]]
            plt.ylim([mean_value - 12.5, mean_value + 12.5])

# Plot
plt.show()
plt.pause(1)
