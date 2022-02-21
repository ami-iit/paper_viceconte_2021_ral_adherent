# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import yarp
import time
import argparse
import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing.utils import iCub
from gym_ignition.utils.scenario import init_gazebo_sim
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing.utils import define_foot_vertices
from adherent.trajectory_generation import trajectory_generator
from adherent.trajectory_generation.utils import define_initial_nn_X
from adherent.trajectory_generation.utils import define_initial_base_yaw
from adherent.data_processing.utils import define_frontal_base_direction
from adherent.data_processing.utils import define_frontal_chest_direction
from adherent.trajectory_generation.utils import define_initial_base_height
from adherent.trajectory_generation.utils import define_initial_past_trajectory
from adherent.trajectory_generation.utils import define_initial_past_trajectory

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--storage_path", help="Path where the generated trajectory will be stored. Relative path from script folder.",
                    type=str, default="../datasets/inference/")
parser.add_argument("--training_path", help="Path where the training-related data are stored. Relative path from script folder.",
                    type=str, default="../datasets/training_D2_D3_subsampled_mirrored_4ew_98%/")
parser.add_argument("--save_every_N_iterations", help="Data will be saved every N iterations.",
                    type=int, default=1000)
parser.add_argument("--plot_trajectory_blending", help="Visualize the blending of the future ground trajectory to build the next network input.", action="store_true")
parser.add_argument("--plot_footsteps", help="Visualize the footsteps.", action="store_true")
parser.add_argument("--plot_blending_coefficients", help="Visualize blending coefficient activations.", action="store_true")

args = parser.parse_args()

storage_path = args.storage_path
training_path = args.training_path
save_every_N_iterations = args.save_every_N_iterations
plot_trajectory_blending = args.plot_trajectory_blending
plot_footsteps = args.plot_footsteps
plot_blending_coefficients = args.plot_blending_coefficients

# ==================
# YARP CONFIGURATION
# ==================

# Inter-process communication is implemented via YARP ports, therefore "yarp server"
# needs to be executed in a separate terminal before launching this script

# Initialise YARP
yarp.Network.init()

# Open and connect YARP port to retrieve joystick input
p_in = yarp.BufferedPortBottle()
p_in.open("/joystick_in")
yarp.Network.connect("/joystick_out", p_in.getName())

# Initialization of the joystick raw and processed inputs received through YARP ports
raw_data = [] # motion and facing directions
quad_bezier = []
base_velocities = []
facing_dirs = []

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
initial_base_height = define_initial_base_height(robot="iCubV2_5")
icub = iCub(world=world, urdf=icub_urdf, position=[0, 0, initial_base_height])

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# Create a KinDynComputations object
controlled_joints = icub.joint_names()
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=controlled_joints)
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ==================
# PLOT CONFIGURATION
# ==================

# Define the indexes of the figures for plotting
figure_facing_dirs = 1
figure_base_vel = 2
figure_blending_coefficients = 3
figure_footsteps = 4
plt.ion()

# ====================
# TRAJECTORY GENERATOR
# ====================

# Tensorflow configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = define_foot_vertices(robot="iCubV2_5")

# Define robot-specific initial input X for the trajectory generation neural network
initial_nn_X = define_initial_nn_X(robot="iCubV2_5")

# Define robot-specific initial past trajectory features
initial_past_trajectory_base_pos, initial_past_trajectory_facing_dirs, initial_past_trajectory_base_vel = \
    define_initial_past_trajectory(robot="iCubV2_5")

# Define robot-specific initial base yaw angle
initial_base_yaw = define_initial_base_yaw(robot="iCubV2_5")

# Define robot-specific frontal base and chest directions
frontal_base_dir = define_frontal_base_direction(robot="iCubV2_5")
frontal_chest_dir = define_frontal_chest_direction(robot="iCubV2_5")

# Instantiate the trajectory generator
generator = trajectory_generator.TrajectoryGenerator.build(icub=icub, gazebo=gazebo, kindyn=kindyn,
                                                           storage_path=os.path.join(script_directory, storage_path),
                                                           training_path=os.path.join(script_directory, training_path),
                                                           local_foot_vertices_pos=local_foot_vertices_pos,
                                                           initial_nn_X=initial_nn_X,
                                                           initial_past_trajectory_base_pos=initial_past_trajectory_base_pos,
                                                           initial_past_trajectory_facing_dirs=initial_past_trajectory_facing_dirs,
                                                           initial_past_trajectory_base_vel=initial_past_trajectory_base_vel,
                                                           initial_base_height=initial_base_height,
                                                           initial_base_yaw=initial_base_yaw,
                                                           frontal_base_direction=frontal_base_dir,
                                                           frontal_chest_direction=frontal_chest_dir)

# =========
# MAIN LOOP
# =========

with tf.Session(config=config) as sess:

    # Restore the learned model and retrieve the tensors of interest
    nn_X, nn_keep_prob, output, blending_coefficients = generator.restore_model_and_retrieve_tensors(session=sess)

    while True:

        # Update the iteration counter
        generator.update_iteration_counter()

        # Retrieve the network output and the blending coefficients
        current_output, denormalized_current_output, current_blending_coefficients = \
            generator.retrieve_network_output_and_blending_coefficients(nn_X=nn_X,
                                                                        nn_keep_prob=nn_keep_prob,
                                                                        output=output,
                                                                        blending_coefficients=blending_coefficients)

        # Apply the joint positions and the base orientation from the network output
        joint_positions, new_base_quaternion = \
            generator.apply_joint_positions_and_base_orientation(denormalized_current_output=denormalized_current_output)

        # Update the support foot and vertex while detecting new footsteps
        support_foot, update_footsteps_list = generator.update_support_vertex_and_support_foot_and_footsteps()

        if update_footsteps_list and plot_footsteps:

            # Plot the last footstep
            generator.plotter.plot_new_footstep(figure_footsteps=figure_footsteps,
                                                support_foot=support_foot,
                                                new_footstep=generator.storage.footsteps[support_foot][-1])

        # Compute kinematically-feasible base position and updated posturals
        new_base_postural, new_joints_postural, new_links_postural, new_com_postural = \
            generator.compute_kinematically_fasible_base_and_update_posturals(joint_positions=joint_positions,
                                                                              base_quaternion=new_base_quaternion,
                                                                              controlled_joints=controlled_joints,
                                                                              link_names=icub.link_names())

        # Retrieve user input data from YARP port
        quad_bezier, base_velocities, facing_dirs, raw_data = \
            generator.retrieve_joystick_inputs(input_port=p_in,
                                               quad_bezier=quad_bezier,
                                               base_velocities=base_velocities,
                                               facing_dirs=facing_dirs,
                                               raw_data=raw_data)

        # Use in an autoregressive fashion the network output and blend it with the user input
        blended_base_positions, blended_facing_dirs, blended_base_velocities = \
            generator.autoregression_and_blending(current_output=current_output,
                                                  denormalized_current_output=denormalized_current_output,
                                                  quad_bezier=quad_bezier,
                                                  facing_dirs=facing_dirs,
                                                  base_velocities=base_velocities)

        # Update storage and periodically save data
        generator.update_storages_and_save(blending_coefficients=current_blending_coefficients,
                                           base_postural=new_base_postural,
                                           joints_postural=new_joints_postural,
                                           links_postural=new_links_postural,
                                           com_postural=new_com_postural,
                                           raw_data=raw_data,
                                           quad_bezier=quad_bezier,
                                           base_velocities=base_velocities,
                                           facing_dirs=facing_dirs,
                                           save_every_N_iterations=save_every_N_iterations)

        if plot_trajectory_blending:

            # Plot the trajectory blending
            generator.plotter.plot_trajectory_blending(figure_facing_dirs=figure_facing_dirs,
                                                       figure_base_vel=figure_base_vel,
                                                       denormalized_current_output=denormalized_current_output,
                                                       quad_bezier=quad_bezier, facing_dirs=facing_dirs,
                                                       base_velocities=base_velocities,
                                                       blended_base_positions=blended_base_positions,
                                                       blended_facing_dirs=blended_facing_dirs,
                                                       blended_base_velocities=blended_base_velocities)

        if plot_blending_coefficients:

            # Plot the blending coefficients
            generator.plotter.plot_blending_coefficients(figure_blending_coefficients=figure_blending_coefficients,
                                                         blending_coeffs=generator.storage.blending_coeffs)

        if plot_trajectory_blending or plot_footsteps or plot_blending_coefficients:

            # Plot
            plt.show()
            plt.pause(0.0001)

        else:

            # Slow down visualization
            time.sleep(0.01)
