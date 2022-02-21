# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing import utils
from gym_ignition.utils.scenario import init_gazebo_sim
from adherent.data_processing import motion_data
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing import xsens_data_converter
from adherent.data_processing import motion_data_retargeter
from gym_ignition.rbd.idyntree.inverse_kinematics_nlp import TargetType
from gym_ignition.rbd.idyntree.inverse_kinematics_nlp import InverseKinematicsNLP

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--filename", help="Mocap file to be retargeted. Relative path from '../datasets/mocap/'.",
                    type=str, default="treadmill_walking.txt")
parser.add_argument("--mirroring", help="Mirror the mocap data.", action="store_true")
parser.add_argument("--KFWBGR", help="Kinematically feasible Whole-Body Geometric Retargeting.", action="store_true")
parser.add_argument("--save", help="Store the retargeted motion in json format.", action="store_true")
parser.add_argument("--deactivate_horizontal_feet", help="Deactivate horizontal feet enforcing.", action="store_true")
parser.add_argument("--deactivate_straight_head", help="Deactivate straight head enforcing.", action="store_true")
parser.add_argument("--deactivate_visualization", help="Do not visualize the retargeted motion.", action="store_true")

args = parser.parse_args()

mocap_filename = args.filename
mirroring = args.mirroring
kinematically_feasible_base_retargeting = args.KFWBGR
store_as_json = args.save
horizontal_feet = not args.deactivate_horizontal_feet
straight_head = not args.deactivate_straight_head
visualize_retargeted_motion = not args.deactivate_visualization

# =====================
# XSENS DATA CONVERSION
# =====================

# Original mocap data
script_directory = os.path.dirname(os.path.abspath(__file__))
mocap_filename = os.path.join(script_directory, "../datasets/mocap/", mocap_filename)

# Define the relevant data for retargeting purposes: timestamps and link orientations
metadata = motion_data.MocapMetadata.build()
metadata.add_timestamp()
metadata.add_link("Pelvis", is_root=True)
metadata.add_link("T8", position=False)
metadata.add_link("Head", position=False)
metadata.add_link("RightUpperLeg", position=False)
metadata.add_link("RightLowerLeg", position=False)
metadata.add_link("RightFoot", position=False)
metadata.add_link("RightUpperArm", position=False)
metadata.add_link("RightForeArm", position=False)
metadata.add_link("LeftUpperLeg", position=False)
metadata.add_link("LeftLowerLeg", position=False)
metadata.add_link("LeftFoot", position=False)
metadata.add_link("LeftUpperArm", position=False)
metadata.add_link("LeftForeArm", position=False)

# Instantiate the data converter
converter = xsens_data_converter.XSensDataConverter.build(mocap_filename=mocap_filename,
                                                          mocap_metadata=metadata)
# Convert the mocap data
motiondata = converter.convert()

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

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# Create a KinDynComputations object
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=icub.joint_names())
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ==========================
# INVERSE KINEMATICS SETTING
# ==========================

# Get the controlled joints
controlled_joints = icub.joint_names()

# Create the IK object
ik = InverseKinematicsNLP(urdf_filename=icub_urdf,
                          considered_joints=controlled_joints,
                          joint_serialization=controlled_joints)

# Initialize IK
ik.initialize(verbosity=1,
              constraints_tolerance=1e-8,
              cost_tolerance=1e-4,
              floating_base=True)

# Add base pose target as a constraint
ik.add_target(frame_name="data_Pelvis", target_type=TargetType.POSE, as_constraint=True)

# Add link orientation targets
target_links = ['data_T8', 'data_Head', 'data_RightUpperLeg', 'data_RightLowerLeg', 'data_RightFoot',
                'data_RightUpperArm', 'data_RightForeArm', 'data_LeftUpperLeg', 'data_LeftLowerLeg',
                'data_LeftFoot', 'data_LeftUpperArm', 'data_LeftForeArm']
for link in target_links:
    ik.add_target(frame_name=link, target_type=TargetType.ROTATION)

# ===========
# RETARGETING
# ===========

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = utils.define_foot_vertices(robot="iCubV2_5")

# Define robot-specific quaternions from the robot base frame to the target base frame
robot_to_target_base_quat = utils.define_robot_to_target_base_quat(robot="iCubV2_5")

# Instantiate the retargeter
if kinematically_feasible_base_retargeting:
    retargeter = motion_data_retargeter.KFWBGR.build(motiondata=motiondata,
                                                     metadata=metadata,
                                                     ik=ik,
                                                     mirroring=mirroring,
                                                     horizontal_feet=horizontal_feet,
                                                     straight_head=straight_head,
                                                     robot_to_target_base_quat=robot_to_target_base_quat,
                                                     kindyn=kindyn,
                                                     local_foot_vertices_pos=local_foot_vertices_pos)
else:
    retargeter = motion_data_retargeter.WBGR.build(motiondata=motiondata,
                                                   metadata=metadata,
                                                   ik=ik,
                                                   mirroring=mirroring,
                                                   horizontal_feet=horizontal_feet,
                                                   straight_head=straight_head,
                                                   robot_to_target_base_quat=robot_to_target_base_quat)

# Retrieve ik solutions
if kinematically_feasible_base_retargeting:
    timestamps, ik_solutions = retargeter.KF_retarget()
else:
    timestamps, ik_solutions = retargeter.retarget()

# =============
# STORE AS JSON
# =============

if store_as_json:

    outfile_name = os.path.join(script_directory, "retargeted_motion.txt")

    input("Press Enter to store the retargeted mocap into a json file")
    utils.store_retargeted_mocap_as_json(timestamps=timestamps, ik_solutions=ik_solutions, outfile_name=outfile_name)
    print("\nThe retargeted mocap data have been saved in", outfile_name, "\n")

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

if visualize_retargeted_motion:

    input("Press Enter to start the visualization of the retargeted motion")
    utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, icub=icub,
                                      controlled_joints=controlled_joints, gazebo=gazebo)
