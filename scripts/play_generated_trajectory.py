# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import json
import argparse
from scenario import gazebo as scenario
from gym_ignition.utils.scenario import init_gazebo_sim
from adherent.data_processing.utils import iCub
from adherent.trajectory_generation.utils import SphereURDF
from adherent.trajectory_generation.utils import Shape
from adherent.trajectory_generation.utils import visualize_generated_motion

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--storage_path", help="Path where the generated trajectory is stored. Relative path from script folder.",
                    type=str, default="../datasets/inference/")
parser.add_argument("--deactivate_blending_coeffs_plot", help="Deactivate plot of the blending coefficients.", action="store_true")

args = parser.parse_args()

storage_path = args.storage_path
plot_blending_coeffs = not args.deactivate_blending_coeffs_plot

# ===============================
# LOAD TRAJECTORY GENERATION DATA
# ===============================

# Define the paths for the generated postural, footsteps, joystick inputs and blending coefficients
script_directory = os.path.dirname(os.path.abspath(__file__))
storage_path = os.path.join(script_directory, storage_path)
postural_path = storage_path + "postural.txt"
footsteps_path = storage_path + "footsteps.txt"
joystick_input_path = storage_path + "joystick_input.txt"
blending_coeffs_path = storage_path + "blending_coefficients.txt"

# Load generated posturals
with open(postural_path, 'r') as openfile:
    posturals = json.load(openfile)

# Load generated footsteps
with open(footsteps_path, 'r') as openfile:
    footsteps = json.load(openfile)
    l_footsteps = footsteps["l_foot"]
    r_footsteps = footsteps["r_foot"]

# Load joystick inputs (motion and facing directions) associated to the generated trajectory
with open(joystick_input_path, 'r') as openfile:
    joystick_input = json.load(openfile)
    raw_data = joystick_input["raw_data"]

# Load blending coefficients associated to the generated trajectory (if available)
if not os.path.exists(blending_coeffs_path):
    plot_blending_coeffs = False
if plot_blending_coeffs:
    with open(blending_coeffs_path, 'r') as openfile:
        blending_coeffs = json.load(openfile)
else:
    blending_coeffs = {}

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
icub = iCub(world=world, urdf=icub_urdf)

# Get the controlled joints
controlled_joints = icub.joint_names()

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# ===================
# VISUALIZE FOOTSTEPS
# ===================

# Retrieve left and right footsteps
ground_l_footsteps = []
for element in l_footsteps:
    ground_l_footsteps.append(element["2D_pos"])
ground_r_footsteps = []
for element in r_footsteps:
    ground_r_footsteps.append(element["2D_pos"])

# Insert a blue sphere for each left contact
blue = (0, 0, 1, 1)
for ground_l_footstep in ground_l_footsteps:
    sphere_position = [ground_l_footstep[0], ground_l_footstep[1], 0.0]
    sphere = Shape(world=world,
                   model_string=SphereURDF(radius=0.025, color=blue).urdf(),
                   position=sphere_position)

# Insert a red sphere for each right contact
red = (1, 0, 0, 1)
for ground_r_footstep in ground_r_footsteps:
    sphere_position = [ground_r_footstep[0], ground_r_footstep[1], 0.0]
    sphere = Shape(world=world,
                   model_string=SphereURDF(radius=0.025, color=red).urdf(),
                   position=sphere_position)

# ==========================
# VISUALIZE GENERATED MOTION
# ==========================

input("Press Enter to start the visualization of the generated trajectory.")
visualize_generated_motion(icub=icub, gazebo=gazebo, posturals=posturals,
                           raw_data=raw_data, blending_coeffs=blending_coeffs,
                           plot_blending_coeffs=plot_blending_coeffs)







