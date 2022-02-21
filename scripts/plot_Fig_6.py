# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from adherent.trajectory_control.utils import rad2deg

# =============
# CONFIGURATION
# =============

# Joints
joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
         'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',
         'torso_pitch', 'torso_roll', 'torso_yaw',
         'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'l_wrist_prosup', 'l_wrist_pitch', 'l_wrist_yaw',
         'neck_pitch', 'neck_roll', 'neck_yaw',
         'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'r_wrist_prosup', 'r_wrist_pitch', 'r_wrist_yaw']

# Indexes of the joints to plot
joints_to_plot_indexes = [
    6,  # r_hip_pitch
    9,  # r_knee
    14, # torso_yaw
    28, # r_elbow
]

# ====================
# LOAD RETARGETED DATA
# ====================

# Path to the retargeted mocap data
retargeted_mocap_path = "../datasets/retargeted_mocap/D2_mirrored/1_forward_normal_step_RETARGETED_MIRRORED.txt"
script_directory = os.path.dirname(os.path.abspath(__file__))
retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

# Load retargeted mocap data
with open(retargeted_mocap_path, 'r') as openfile:
    ik_solutions = json.load(openfile)

# Define mocap data of interest (these data are at 100 Hz)
initial_frame = 3890
final_frame = initial_frame + 500

# Extract retargeted joint positions from the retargeted mocap data (subsampled at 50 Hz)
retargeted_joint_positions = []
for i in range(initial_frame,final_frame,2):
    ik_solution = ik_solutions[i]
    joint_positions = np.asarray(ik_solution["joint_positions"])
    retargeted_joint_positions.append(joint_positions)

# ===================
# LOAD GENERATED DATA
# ===================

# Path to the generated postural
generated_postural_path = "../datasets/inference/fig_6_adherent_generated/postural.txt"
generated_postural_path = os.path.join(script_directory, generated_postural_path)

# Load generated posturals
with open(generated_postural_path, 'r') as openfile:
    generated_posturals = json.load(openfile)
    generated_joint_posturals = generated_posturals["joints"]

# Reformat generated posturals into generated joint positions for each joint (these data are at 50 Hz)
generated_joint_positions = []
for generated_joint_postural in generated_joint_posturals:
    joint_positions = [generated_joint_postural[joint] for joint in joints]
    generated_joint_positions.append(np.asarray(joint_positions))

# ==============================================================
# MANUALLY IMPROVE RETARGETED AND GENERATED DATA SYNCHRONIZATION
# ==============================================================

# By construction, the generated forward walking is not synchronized with the particular retargeted
# forward walking chosen for this comparison. To facilitate the comparison, we manually improve the
# synchronization of the retargeted and generated data by adding a delay in some double support phases
# of the generated trajectory (selected by visual inspection of the data).

# Slowed-down generated joint positions
slowed_down_generated_joint_positions = []

# Central frames of the selected DS phases in which the generated data are slowed down
central_frames_DS_to_slow_down = [30,60,115,140,170,195]

# Replicate the postural 5 times in the selected DS phases to slow down the generated motion
for i in range(len(generated_joint_positions)):
    joint_positions = generated_joint_positions[i]
    slowed_down_generated_joint_positions.append(joint_positions)
    if i in central_frames_DS_to_slow_down:
        for j in range(5):
            slowed_down_generated_joint_positions.append(joint_positions)

# Remove the tail to keep the same length of the retargeted data
slowed_down_generated_joint_positions = slowed_down_generated_joint_positions[:250]

# ===========================================================
# COMPARE MANUALLY-SYNCHRONIZED RETARGETED AND GENERATED DATA
# ===========================================================

# Extract plot time (seconds)
plot_time = list(range(len(retargeted_joint_positions)))
plot_time = [t/50.0 for t in plot_time]

for k in joints_to_plot_indexes:

    plt.figure()

    # Retrieve retargeted and (slowed-down) generated joint positions
    retargeted_joint_positions_plot = [rad2deg(elem[k]) for elem in retargeted_joint_positions]
    generated_joint_positions_plot = [rad2deg(elem[k]) for elem in slowed_down_generated_joint_positions]

    # Plot retargeted vs (slowed-down) generated joint positions
    plt.plot(plot_time, retargeted_joint_positions_plot, 'b', label="Retargeted")
    plt.plot(plot_time, generated_joint_positions_plot, 'r', label="Generated")

    # Plot configuration
    plt.xlabel('time [s]')
    plt.ylabel('joint positions [deg]')
    title = joints[k].replace("_", " ").upper()
    plt.title(title)
    plt.legend()

# Plot
plt.show()
plt.pause(0.1)
