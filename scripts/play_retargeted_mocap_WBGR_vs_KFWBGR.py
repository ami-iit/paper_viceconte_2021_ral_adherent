# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing import utils
from gym_ignition.utils.scenario import init_gazebo_sim

# ===============
# MODEL INSERTION
# ===============

# Set scenario verbosity
scenario.set_verbosity(scenario.Verbosity_warning)

# Get the default simulator and the default empty world
gazebo, world = init_gazebo_sim()

# Retrieve the robot urdf model
script_directory = os.path.dirname(os.path.abspath(__file__))
icub_urdf = os.path.join(script_directory, "../src/adherent/model/iCubGazeboSimpleCollisionsV2_5_xsens/iCubGazeboSimpleCollisionsV2_5_xsens.urdf")

# Insert the robot in the empty world
icub = utils.iCub(world=world, urdf=icub_urdf)

# Get the controlled joints
controlled_joints = icub.joint_names()

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# =========================
# LOAD WBGR-RETARGETED DATA
# =========================

# Define the WBGR-retargeted mocap data path and its relevant frames interval
retargeted_mocap_path = "../datasets/additional_figures/supplementary_video_WBGR.txt"
retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)
initial_frame = 0
final_frame = 1000

# Load the WBGR-retargeted mocap data
timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path,
                                                                 initial_frame=initial_frame,
                                                                 final_frame=final_frame)

# Enforce motion starting from the ground origin for comparison purposes
initial_base_pos = ik_solutions[0]["base_position"]
for ik_solution in ik_solutions:
    ik_solution["base_position"] = np.asarray(ik_solution["base_position"]) - [initial_base_pos[0], initial_base_pos[1], 0]

# ====================================
# VISUALIZE THE WBGR-RETARGETED MOTION
# ====================================

input("Press Enter to start the visualization of the WBGR-retargeted motion")
utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, icub=icub,
                                  controlled_joints=controlled_joints, gazebo=gazebo)

# ===========================
# LOAD KFWBGR-RETARGETED DATA
# ===========================

# Define the KFWBGR-retargeted mocap data path
retargeted_mocap_path = "../datasets/additional_figures/supplementary_video_KFWBGR.txt"
retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

# Load the KFWBGR-retargeted mocap data
timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path,
                                                                 initial_frame=initial_frame,
                                                                 final_frame=final_frame)

# Enforce motion starting from the ground origin for comparison purposes
initial_base_pos = ik_solutions[0]["base_position"]
for ik_solution in ik_solutions:
    ik_solution["base_position"] = np.asarray(ik_solution["base_position"]) - [initial_base_pos[0], initial_base_pos[1], 0]

# ======================================
# VISUALIZE THE KFWBGR-RETARGETED MOTION
# ======================================

input("Press Enter to start the visualization of the KFWBGR-retargeted motion")
utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, icub=icub,
                                  controlled_joints=controlled_joints, gazebo=gazebo)
