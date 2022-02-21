# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import argparse
import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing import utils
from gym_ignition.utils.scenario import init_gazebo_sim
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing import features_extractor

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Our custom dataset is divided in two datasets: D2 and D3
parser.add_argument("--dataset", help="Select a dataset between D2 and D3.", type=str, default="D2")

# Each dataset is divided into portions. D2 includes portions [1,5]. D3 includes portions [6,11].
parser.add_argument("--portion", help="Select a portion of the chosen dataset. Available choices: from 1 to 5 for D2,"
                                      "from 6 to 11 for D3.", type=int, default=1)

# Each portion of each dataset has been retargeted as it is or mirrored. Select if you want to visualize the mirrored version
parser.add_argument("--mirrored", help="Visualize the mirrored version of the selected dataset portion.",action="store_true")

# Plot configuration
parser.add_argument("--plot_global", help="Visualize the computed global features.",action="store_true")
parser.add_argument("--plot_local", help="Visualization the computed local features.",action="store_true")

# Store configuration
parser.add_argument("--save", help="Store the network input and output vectors in json format.",action="store_true")

args = parser.parse_args()

dataset = args.dataset
retargeted_mocap_index = args.portion
mirrored = args.mirrored
plot_global = args.plot_global
plot_local = args.plot_local
store_as_json = args.save

# ====================
# LOAD RETARGETED DATA
# ====================

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
script_directory = os.path.dirname(os.path.abspath(__file__))
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

# Create a KinDynComputations object
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=controlled_joints)
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ===================
# FEATURES EXTRACTION
# ===================

# Define robot-specific frontal base and chest directions
frontal_base_dir = utils.define_frontal_base_direction(robot="iCubV2_5")
frontal_chest_dir = utils.define_frontal_chest_direction(robot="iCubV2_5")

# Instantiate the features extractor
extractor = features_extractor.FeaturesExtractor.build(ik_solutions=ik_solutions,
                                                       kindyn=kindyn,
                                                       frontal_base_dir=frontal_base_dir,
                                                       frontal_chest_dir=frontal_chest_dir)
# Extract the features
extractor.compute_features()

# ===========================================
# NETWORK INPUT AND OUTPUT VECTORS GENERATION
# ===========================================

# Generate the network input vector X
X = extractor.compute_X()

if store_as_json:

    # Define the path to store the input X associated to the selected subsection of the dataset
    if not mirrored:
        input_path = "../datasets/IO_features/inputs_subsampled_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_X.txt"
    else:
        input_path = "../datasets/IO_features/inputs_subsampled_mirrored_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_X_MIRRORED.txt"
    input_path = os.path.join(script_directory, input_path)

    input("Press Enter to store the computed X")

    # Store the retrieved input X in a JSON file
    with open(input_path, 'w') as outfile:
        json.dump(X, outfile)

    # Debug
    print("Input features have been saved in", input_path)

# Generate the network output vector Y
Y = extractor.compute_Y()

if store_as_json:

    # Define the path to store the output Y associated to the selected subsection of the dataset
    if not mirrored:
        output_path = "../datasets/IO_features/outputs_subsampled_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_Y.txt"
    else:
        output_path = "../datasets/IO_features/outputs_subsampled_mirrored_" + dataset + "/" + retargeted_mocaps[retargeted_mocap_index] + "_Y_MIRRORED.txt"
    output_path = os.path.join(script_directory, output_path)

    input("Press Enter to store the computed Y")

    # Store the retrieved output Y in a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(Y, outfile)

    # Debug
    print("Output features have been saved in", output_path)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE GLOBAL FEATURES
# =======================================================

if plot_global:

    input("Press Enter to start the visualization of the GLOBAL features")
    utils.visualize_global_features(global_window_features=extractor.get_global_window_features(),
                                    ik_solutions=ik_solutions,
                                    icub=icub,
                                    controlled_joints=controlled_joints,
                                    gazebo=gazebo)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE LOCAL FEATURES
# =======================================================

if plot_local:

    input("Press Enter to start the visualization of the LOCAL features")
    utils.visualize_local_features(local_window_features=extractor.get_local_window_features(),
                                   ik_solutions=ik_solutions,
                                   icub=icub,
                                   controlled_joints=controlled_joints,
                                   gazebo=gazebo)
