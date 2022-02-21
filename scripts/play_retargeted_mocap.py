# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
from scenario import gazebo as scenario
from adherent.data_processing import utils
from gym_ignition.utils.scenario import init_gazebo_sim

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Visualize the latest retargeted motion (i.e. the one stored in the "retargeted_motion.txt" file)
parser.add_argument("--latest", help="Visualize the latest retargeted motion (i.e. the one stored in the "
                                     "retargeted_motion.txt file)", action="store_true")

# Our custom dataset is divided in two datasets: D2 and D3
parser.add_argument("--dataset", help="Select a dataset between D2 and D3.", type=str, default="D2")

# Each dataset is divided into portions. D2 includes portions [1,5]. D3 includes portions [6,11].
parser.add_argument("--portion", help="Select a portion of the chosen dataset. Available choices: from 1 to 5 for D2,"
                                      "from 6 to 11 for D3.", type=int, default=1)

# Each portion of each dataset has been retargeted as it is or mirrored. Select if you want to visualize the mirrored version
parser.add_argument("--mirrored", help="Visualize the mirrored version of the selected dataset portion.", action="store_true")

args = parser.parse_args()

latest = args.latest
dataset = args.dataset
retargeted_mocap_index = args.portion
mirrored = args.mirrored

# ====================
# LOAD RETARGETED DATA
# ====================

# Retrieve script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

if latest:

    # Mocap path for the latest retargeted motion
    retargeted_mocap_path = "retargeted_motion.txt"
    retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

    # Load the retargeted mocap data
    timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path)

else:

    # Define the selected subsection of the dataset to be loaded and the correspondent interesting frame interval
    if dataset == "D2":
        retargeted_mocaps = {1:"1_forward_normal_step",2:"2_backward_normal_step",3:"3_left_and_right_normal_step",
                             4:"4_diagonal_normal_step",5:"5_mixed_normal_step"}
        limits = {1: [3750, 35750], 2: [1850, 34500], 3: [2400, 36850], 4: [1550, 16000], 5: [2550, 82250]}
    elif dataset == "D3":
        retargeted_mocaps = {6:"6_forward_small_step",7:"7_backward_small_step",8:"8_left_and_right_small_step",
                             9:"9_diagonal_small_step",10:"10_mixed_small_step",11:"11_mixed_normal_and_small_step"}
        limits = {6: [1500, 28500], 7: [1750, 34000], 8: [2900, 36450], 9: [1250, 17050], 10: [1450, 78420], 11: [1600, 61350]}
    initial_frame = limits[retargeted_mocap_index][0]
    final_frame = limits[retargeted_mocap_index][1]

    # Define the retargeted mocap path
    if not mirrored:
        retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_RETARGETED.txt"
    else:
        retargeted_mocap_path = "../datasets/retargeted_mocap/" + dataset + "_mirrored/" + retargeted_mocaps[retargeted_mocap_index]  + "_RETARGETED_MIRRORED.txt"
    retargeted_mocap_path = os.path.join(script_directory, retargeted_mocap_path)

    # Load the retargeted mocap data
    timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path,
                                                                     initial_frame=initial_frame,
                                                                     final_frame=final_frame)

# ===============
# MODEL INSERTION
# ===============

# Set scenario verbosity
scenario.set_verbosity(scenario.Verbosity_warning)

# Get the default simulator and the default empty world
gazebo, world = init_gazebo_sim()

# Retrieve the robot urdf model
icub_urdf = os.path.join(script_directory, "../src/adherent/model/iCubGazeboSimpleCollisionsV2_5_xsens/iCubGazeboSimpleCollisionsV2_5_xsens.urdf")

# Insert the robot in the empty world
icub = utils.iCub(world=world, urdf=icub_urdf)

# Get the controlled joints
controlled_joints = icub.joint_names()

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

input("Press Enter to start the visualization of the retargeted motion")
utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, icub=icub,
                                  controlled_joints=controlled_joints, gazebo=gazebo)
