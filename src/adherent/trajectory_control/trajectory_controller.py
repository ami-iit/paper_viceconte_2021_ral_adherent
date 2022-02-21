# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import yarp
import json
import numpy as np
import manifpy as manif
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from gym_ignition.rbd.conversions import Rotation
import bipedal_locomotion_framework.bindings as blf
from adherent.trajectory_control.utils import rad2deg
from gym_ignition.rbd.idyntree import inverse_kinematics_nlp
from adherent.trajectory_control.utils import Integrator
from adherent.trajectory_control.utils import compute_zmp
from adherent.trajectory_control.utils import synchronize
from adherent.trajectory_control.utils import world_gravity


@dataclass
class StorageHandler:
    """Class to store all the quantities relevant in the trajectory control pipeline and save data."""

    # Path to the storage folder
    storage_path: str

    # List of considered joints
    joints_list: List

    # True if the learning-based postural is exploited
    use_joint_references: bool

    # Joint desired and measured positions and velocities storage
    joint_des_storage: List
    joint_meas_storage: List
    joint_vel_des_storage: List
    joint_vel_meas_storage: List

    # Joint posturals storage
    joint_postural_storage: List

    # Legged odometry storage
    world_H_bases: List = field(default_factory=list)
    base_twists: List = field(default_factory=list)

    # Fixed foot storage
    fixed_foot_indexes: List = field(default_factory=list)

    # Com desired position and velocity storage
    com_pos_des_x: List = field(default_factory=list)
    com_pos_des_y: List = field(default_factory=list)
    com_vel_des_x: List = field(default_factory=list)
    com_vel_des_y: List = field(default_factory=list)
    com_vel_des_from_dcm_x: List = field(default_factory=list)
    com_vel_des_from_dcm_y: List = field(default_factory=list)

    # CoM measured position and velocity storage
    com_pos_meas_x: List = field(default_factory=list)
    com_pos_meas_y: List = field(default_factory=list)
    com_vel_meas_x: List = field(default_factory=list)
    com_vel_meas_y: List = field(default_factory=list)

    # ZMP desired position storage
    zmp_pos_des_x: List = field(default_factory=list)
    zmp_pos_des_y: List = field(default_factory=list)

    # ZMP measured position storage
    zmp_pos_meas_x: List = field(default_factory=list)
    zmp_pos_meas_y: List = field(default_factory=list)

    # DCM desired position and velocity storage
    dcm_pos_des_x: List = field(default_factory=list)
    dcm_pos_des_y: List = field(default_factory=list)
    dcm_vel_des_x: List = field(default_factory=list)
    dcm_vel_des_y: List = field(default_factory=list)

    # DCM measured position storage
    dcm_pos_meas_x: List = field(default_factory=list)
    dcm_pos_meas_y: List = field(default_factory=list)

    # CoM, ZMP and DCM errors storage
    com_pos_errors: List = field(default_factory=list)
    zmp_pos_errors: List = field(default_factory=list)
    dcm_pos_errors: List = field(default_factory=list)

    # Right foot desired position storage
    right_foot_des_x: List = field(default_factory=list)
    right_foot_des_y: List = field(default_factory=list)
    right_foot_des_z: List = field(default_factory=list)

    # Right foot measured position storage
    right_foot_meas_x: List = field(default_factory=list)
    right_foot_meas_y: List = field(default_factory=list)
    right_foot_meas_z: List = field(default_factory=list)

    # Right foot desired RPY storage
    right_foot_des_R: List = field(default_factory=list)
    right_foot_des_P: List = field(default_factory=list)
    right_foot_des_Y: List = field(default_factory=list)

    # Right foot measured RPY storage
    right_foot_meas_R: List = field(default_factory=list)
    right_foot_meas_P: List = field(default_factory=list)
    right_foot_meas_Y: List = field(default_factory=list)

    # Left foot desired position storage
    left_foot_des_x: List = field(default_factory=list)
    left_foot_des_y: List = field(default_factory=list)
    left_foot_des_z: List = field(default_factory=list)

    # Left foot measured position storage
    left_foot_meas_x: List = field(default_factory=list)
    left_foot_meas_y: List = field(default_factory=list)
    left_foot_meas_z: List = field(default_factory=list)

    # Left foot desired RPY storage
    left_foot_des_P: List = field(default_factory=list)
    left_foot_des_R: List = field(default_factory=list)
    left_foot_des_Y: List = field(default_factory=list)

    # Left foot measured RPY storage
    left_foot_meas_R: List = field(default_factory=list)
    left_foot_meas_P: List = field(default_factory=list)
    left_foot_meas_Y: List = field(default_factory=list)

    # Left foot wrench storage
    left_force_x: List = field(default_factory=list)
    left_force_y: List = field(default_factory=list)
    left_force_z: List = field(default_factory=list)
    left_torque_x: List = field(default_factory=list)
    left_torque_y: List = field(default_factory=list)
    left_torque_z: List = field(default_factory=list)

    # Right foot wrench storage
    right_force_x: List = field(default_factory=list)
    right_force_y: List = field(default_factory=list)
    right_force_z: List = field(default_factory=list)
    right_torque_x: List = field(default_factory=list)
    right_torque_y: List = field(default_factory=list)
    right_torque_z: List = field(default_factory=list)

    @staticmethod
    def build(storage_path: str, joints_list: List, use_joint_references: bool) -> "StorageHandler":
        """Build an instance of StorageHandler."""

        return StorageHandler(storage_path=storage_path,
                              joints_list=joints_list,
                              use_joint_references=use_joint_references,
                              joint_des_storage=[[] for _ in joints_list],
                              joint_meas_storage=[[] for _ in joints_list],
                              joint_vel_des_storage=[[] for _ in joints_list],
                              joint_vel_meas_storage=[[] for _ in joints_list],
                              joint_postural_storage=[[] for _ in joints_list])

    def update_joints_storage(self, joints_values_des: List, joints_values: List, joints_velocities_des: List,
                              joints_velocities: List, joint_references: List, idx: float, dt: float) -> None:
        """Update the joints storage."""

        for i in range(len(self.joints_list)):

            # Joint desired and measured positions (deg) and velocities (deg/s)
            self.joint_des_storage[i].append(rad2deg(joints_values_des[i]))
            self.joint_meas_storage[i].append(rad2deg(joints_values[i]))
            self.joint_vel_des_storage[i].append(rad2deg(joints_velocities_des[i]))
            self.joint_vel_meas_storage[i].append(rad2deg(joints_velocities[i]))

            # Joint posturals
            if self.use_joint_references:
                self.joint_postural_storage[i].append(rad2deg(joint_references[round(idx/dt)][i]))

    def update_legged_odom_and_fixed_foot_storage(self, world_H_base: List, base_twist: List, fixed_foot_index: int) -> None:
        """Update the legged odometry and fixed foot detector storage."""

        # Legged odom
        self.world_H_bases.append(world_H_base)
        self.base_twists.append(base_twist)

        # Fixed foot
        self.fixed_foot_indexes.append(fixed_foot_index)

    def update_com_storage(self, com_pos_des: List, com_vel_des: List, com_vel_des_from_dcm: List, com_pos_meas: List,
                           com_vel_meas: List, com_pos_error: List) -> None:
        """Update the storage of the quantities related to the center of mass (CoM)."""

        # CoM desired position and velocity
        self.com_pos_des_x.append(com_pos_des[0])
        self.com_pos_des_y.append(com_pos_des[1])
        self.com_vel_des_x.append(com_vel_des[0])
        self.com_vel_des_y.append(com_vel_des[1])
        self.com_vel_des_from_dcm_x.append(com_vel_des_from_dcm[0])
        self.com_vel_des_from_dcm_y.append(com_vel_des_from_dcm[1])

        # CoM measured position and velocity
        self.com_pos_meas_x.append(com_pos_meas[0])
        self.com_pos_meas_y.append(com_pos_meas[1])
        self.com_vel_meas_x.append(com_vel_meas[0])
        self.com_vel_meas_y.append(com_vel_meas[1])

        # CoM position error
        self.com_pos_errors.append(com_pos_error)

    def update_zmp_storage(self, zmp_pos_des: List, zmp_pos_meas: List, zmp_pos_error: List) -> None:
        """Update the storage of the quantities related to the zero-moment point (ZMP)."""

        # ZMP desired position
        self.zmp_pos_des_x.append(zmp_pos_des[0])
        self.zmp_pos_des_y.append(zmp_pos_des[1])

        # ZMP measured position
        self.zmp_pos_meas_x.append(zmp_pos_meas[0])
        self.zmp_pos_meas_y.append(zmp_pos_meas[1])

        # ZMP position error
        self.zmp_pos_errors.append(zmp_pos_error)

    def update_dcm_storage(self, dcm_position: List, dcm_velocity: List, dcm_pos_meas: List, dcm_pos_error: List) -> None:
        """Update the storage of the quantities related to the divergent component of motion (DCM)."""

        # DCM desired position and velocity
        self.dcm_pos_des_x.append(dcm_position[0])
        self.dcm_pos_des_y.append(dcm_position[1])
        self.dcm_vel_des_x.append(dcm_velocity[0])
        self.dcm_vel_des_y.append(dcm_velocity[1])

        # DCM measured position
        self.dcm_pos_meas_x.append(dcm_pos_meas[0])
        self.dcm_pos_meas_y.append(dcm_pos_meas[1])

        # DCM position error
        self.dcm_pos_errors.append(dcm_pos_error)

    def update_right_foot_storage(self, right_foot_des: np.array, right_foot_meas_transform: np.array,
                                  right_foot_des_transform: np.array, right_wrench: np.array) -> None:
        """Update the storage of the quantities related to the right foot."""

        # Right foot desired position
        self.right_foot_des_x.append(right_foot_des[0])
        self.right_foot_des_y.append(right_foot_des[1])
        self.right_foot_des_z.append(right_foot_des[2])

        # Right foot measured position
        right_foot_meas_pos = right_foot_meas_transform[0:3, 3]
        self.right_foot_meas_x.append(right_foot_meas_pos[0])
        self.right_foot_meas_y.append(right_foot_meas_pos[1])
        self.right_foot_meas_z.append(right_foot_meas_pos[2])

        # Right foot desired RPY
        right_foot_des_rot = right_foot_des_transform[0:3,0:3]
        right_foot_des_RPY = Rotation.from_matrix(right_foot_des_rot).as_euler("xyz")
        self.right_foot_des_R.append(rad2deg(right_foot_des_RPY[0]))
        self.right_foot_des_P.append(rad2deg(right_foot_des_RPY[1]))
        self.right_foot_des_Y.append(rad2deg(right_foot_des_RPY[2]))

        # Right foot measured RPY
        right_foot_meas_rot = right_foot_meas_transform[0:3,0:3]
        right_foot_meas_RPY = Rotation.from_matrix(right_foot_meas_rot).as_euler("xyz")
        self.right_foot_meas_R.append(rad2deg(right_foot_meas_RPY[0]))
        self.right_foot_meas_P.append(rad2deg(right_foot_meas_RPY[1]))
        self.right_foot_meas_Y.append(rad2deg(right_foot_meas_RPY[2]))

        # Right foot wrench
        self.right_force_x.append(right_wrench[0])
        self.right_force_y.append(right_wrench[1])
        self.right_force_z.append(right_wrench[2])
        self.right_torque_x.append(right_wrench[3])
        self.right_torque_y.append(right_wrench[4])
        self.right_torque_z.append(right_wrench[5])

    def update_left_foot_storage(self, left_foot_des: np.array, left_foot_meas_transform: np.array,
                                 left_foot_des_transform: np.array, left_wrench: np.array) -> None:
        """Update the storage of the quantities related to the left foot."""

        # Left foot desired position
        self.left_foot_des_x.append(left_foot_des[0])
        self.left_foot_des_y.append(left_foot_des[1])
        self.left_foot_des_z.append(left_foot_des[2])

        # Left foot measured position
        left_foot_meas_pos = left_foot_meas_transform[0:3,3]
        self.left_foot_meas_x.append(left_foot_meas_pos[0])
        self.left_foot_meas_y.append(left_foot_meas_pos[1])
        self.left_foot_meas_z.append(left_foot_meas_pos[2])

        # Left foot desired RPY
        left_foot_des_rot = left_foot_des_transform[0:3,0:3]
        left_foot_des_RPY = Rotation.from_matrix(left_foot_des_rot).as_euler("xyz")
        self.left_foot_des_R.append(rad2deg(left_foot_des_RPY[0]))
        self.left_foot_des_P.append(rad2deg(left_foot_des_RPY[1]))
        self.left_foot_des_Y.append(rad2deg(left_foot_des_RPY[2]))

        # Left foot measured RPY
        left_foot_meas_rot = left_foot_meas_transform[0:3,0:3]
        left_foot_meas_RPY = Rotation.from_matrix(left_foot_meas_rot).as_euler("xyz")
        self.left_foot_meas_R.append(rad2deg(left_foot_meas_RPY[0]))
        self.left_foot_meas_P.append(rad2deg(left_foot_meas_RPY[1]))
        self.left_foot_meas_Y.append(rad2deg(left_foot_meas_RPY[2]))

        # Left foot wrench
        self.left_force_x.append(left_wrench[0])
        self.left_force_y.append(left_wrench[1])
        self.left_force_z.append(left_wrench[2])
        self.left_torque_x.append(left_wrench[3])
        self.left_torque_y.append(left_wrench[4])
        self.left_torque_z.append(left_wrench[5])

    def save_data_as_json(self) -> None:
        """Save all the stored data using the json format."""

        data = {}

        for i in range(len(self.joints_list)):

            data[self.joints_list[i]] = {}

            # Joint desired and measured positions (deg) and velocities (deg/s)
            data[self.joints_list[i]]["des"] = self.joint_des_storage[i]
            data[self.joints_list[i]]["meas"] = self.joint_meas_storage[i]
            data[self.joints_list[i]]["vel_des"] = self.joint_vel_des_storage[i]
            data[self.joints_list[i]]["vel_meas"] = self.joint_vel_meas_storage[i]

            # Joint posturals
            if self.use_joint_references:
                data[self.joints_list[i]]["postural"] = self.joint_postural_storage[i]

        # Legged odom
        data['world_H_bases'] = self.world_H_bases
        data['base_twists'] = self.base_twists

        # Fixed foot
        data['fixed_foot_indexes'] = self.fixed_foot_indexes

        # Com desired position and velocity
        data['com_pos_des_x'] = self.com_pos_des_x
        data['com_pos_des_y'] = self.com_pos_des_y
        data['com_vel_des_x'] = self.com_vel_des_x
        data['com_vel_des_y'] = self.com_vel_des_y
        data['com_vel_des_from_dcm_x'] = self.com_vel_des_from_dcm_x
        data['com_vel_des_from_dcm_y'] = self.com_vel_des_from_dcm_y

        # CoM measured position and velocity
        data['com_pos_meas_x'] = self.com_pos_meas_x
        data['com_pos_meas_y'] = self.com_pos_meas_y
        data['com_vel_meas_x'] = self.com_vel_meas_x
        data['com_vel_meas_y'] = self.com_vel_meas_y

        # ZMP desired position
        data['zmp_pos_des_x'] = self.zmp_pos_des_x
        data['zmp_pos_des_y'] = self.zmp_pos_des_y

        # ZMP measured position
        data['zmp_pos_meas_x'] = self.zmp_pos_meas_x
        data['zmp_pos_meas_y'] = self.zmp_pos_meas_y

        # DCM desired position and velocity
        data['dcm_pos_des_x'] = self.dcm_pos_des_x
        data['dcm_pos_des_y'] = self.dcm_pos_des_y
        data['dcm_vel_des_x'] = self.dcm_vel_des_x
        data['dcm_vel_des_y'] = self.dcm_vel_des_y

        # DCM measured position
        data['dcm_pos_meas_x'] = self.dcm_pos_meas_x
        data['dcm_pos_meas_y'] = self.dcm_pos_meas_y

        # CoM, ZMP and DCM errors
        data['com_pos_errors'] = self.com_pos_errors
        data['zmp_pos_errors'] = self.zmp_pos_errors
        data['dcm_pos_errors'] = self.dcm_pos_errors

        # Right foot desired position
        data['right_foot_des_x'] = self.right_foot_des_x
        data['right_foot_des_y'] = self.right_foot_des_y
        data['right_foot_des_z'] = self.right_foot_des_z

        # Right foot measured position
        data['right_foot_meas_x'] = self.right_foot_meas_x
        data['right_foot_meas_y'] = self.right_foot_meas_y
        data['right_foot_meas_z'] = self.right_foot_meas_z

        # Right foot desired RPY
        data['right_foot_des_R'] = self.right_foot_des_R
        data['right_foot_des_P'] = self.right_foot_des_P
        data['right_foot_des_Y'] = self.right_foot_des_Y

        # Right foot measured RPY
        data['right_foot_meas_R'] = self.right_foot_meas_R
        data['right_foot_meas_P'] = self.right_foot_meas_P
        data['right_foot_meas_Y'] = self.right_foot_meas_Y

        # Left foot desired position
        data['left_foot_des_x'] = self.left_foot_des_x
        data['left_foot_des_y'] = self.left_foot_des_y
        data['left_foot_des_z'] = self.left_foot_des_z

        # Left foot measured position
        data['left_foot_meas_x'] = self.left_foot_meas_x
        data['left_foot_meas_y'] = self.left_foot_meas_y
        data['left_foot_meas_z'] = self.left_foot_meas_z

        # Left foot desired RPY
        data['left_foot_des_R'] = self.left_foot_des_R
        data['left_foot_des_P'] = self.left_foot_des_P
        data['left_foot_des_Y'] = self.left_foot_des_Y

        # Left foot measured RPY
        data['left_foot_meas_R'] = self.left_foot_meas_R
        data['left_foot_meas_P'] = self.left_foot_meas_P
        data['left_foot_meas_Y'] = self.left_foot_meas_Y

        # Left foot wrench
        data['left_force_x'] = self.left_force_x
        data['left_force_y'] = self.left_force_y
        data['left_force_z'] = self.left_force_z
        data['left_torque_x'] = self.left_torque_x
        data['left_torque_y'] = self.left_torque_y
        data['left_torque_z'] = self.left_torque_z

        # Right foot wrench
        data['right_force_x'] = self.right_force_x
        data['right_force_y'] = self.right_force_y
        data['right_force_z'] = self.right_force_z
        data['right_torque_x'] = self.right_torque_x
        data['right_torque_y'] = self.right_torque_y
        data['right_torque_z'] = self.right_torque_z

        # Store relevant data in a folder dedicated to the single experiment
        current_storage_path = self.storage_path + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + "/"
        os.mkdir(current_storage_path)
        data_path = os.path.join(current_storage_path, "data.txt")
        with open(data_path, 'w') as outfile:
            json.dump(data, outfile)

        # Debug
        input("\nData have been saved in " + current_storage_path + ". Press Enter to close.")


@dataclass
class FootstepsExtractor:
    """Class to process the generated footsteps."""

    # Path for the generated footsteps
    footsteps_path: str

    # Footsteps scaling factor
    footstep_scaling: float

    # Time scaling factor
    time_scaling: int

    # Footsteps list
    contact_phase_list: blf.contacts.ContactPhaseList = None

    @staticmethod
    def build(footsteps_path: str, footstep_scaling: float, time_scaling: int) -> "FootstepsExtractor":
        """Build an instance of FootstepsExtractor."""

        return FootstepsExtractor(footsteps_path=footsteps_path,
                                  footstep_scaling=footstep_scaling,
                                  time_scaling=time_scaling)

    def retrieve_contacts(self) -> None:
        """Retrieve and scale the footsteps of the generated trajectory. Plot original and scaled footsteps."""

        # Create the map of contact lists
        contact_list_map = dict()

        # Names of the feet frames
        rfoot_frame = "r_sole"
        lfoot_frame = "l_sole"

        # Create the contact lists
        contact_list_map[rfoot_frame] = blf.contacts.ContactList()
        contact_list_map[lfoot_frame] = blf.contacts.ContactList()

        # Retrieve the footsteps from a JSON file
        with open(self.footsteps_path, 'r') as infile:
            contacts = json.load(infile)
        l_contacts = contacts["l_foot"]
        r_contacts = contacts["r_foot"]

        # Storage fot plotting unscaled vs scaled footsteps
        unscaled_left_footsteps_x = []
        unscaled_left_footsteps_y = []
        unscaled_right_footsteps_x = []
        unscaled_right_footsteps_y = []
        left_footsteps_x = []
        left_footsteps_y = []
        right_footsteps_x = []
        right_footsteps_y = []

        # ===============================
        # INITIAL LEFT AND RIGHT CONTACTS
        # ===============================

        # Retrieve first left contact position
        ground_l_foot_position = [l_contacts[0]["2D_pos"][0], l_contacts[0]["2D_pos"][1], 0]
        ground_l_foot_position_gazebo = [0, 0.08, 0]
        ground_l_foot_position_offset = np.array(ground_l_foot_position_gazebo) - np.array(ground_l_foot_position)
        ground_l_foot_position += np.array(ground_l_foot_position_offset)

        # Retrieve first left contact orientation
        l_foot_yaw = l_contacts[0]["2D_orient"]
        l_foot_RPY = [0.0, 0.0, l_foot_yaw]
        l_foot_rot = Rotation.from_euler('xyz', l_foot_RPY)
        l_foot_quat = l_foot_rot.as_quat()
        l_deactivation_time = self.time_scaling * (l_contacts[0]["deactivation_time"])

        # Retrieve first right contact position
        ground_r_foot_position = [r_contacts[0]["2D_pos"][0], r_contacts[0]["2D_pos"][1], 0]
        ground_r_foot_position_gazebo = [0, -0.08, 0]
        ground_r_foot_position_offset = np.array(ground_r_foot_position_gazebo) - np.array(ground_r_foot_position)
        ground_r_foot_position += np.array(ground_r_foot_position_offset)

        # Retrieve first right contact orientation
        r_foot_yaw = r_contacts[0]["2D_orient"]
        r_foot_RPY = [0.0, 0.0, r_foot_yaw]
        r_foot_rot = Rotation.from_euler('xyz', r_foot_RPY)
        r_foot_quat = r_foot_rot.as_quat()
        r_deactivation_time = self.time_scaling * (r_contacts[0]["deactivation_time"])

        # Add initial left and right contacts to the list
        assert contact_list_map[lfoot_frame].add_contact(
            transform=manif.SE3(position=np.array(ground_l_foot_position),
                                quaternion=l_foot_quat),
            activation_time=0.0,
            deactivation_time=l_deactivation_time)
        assert contact_list_map[rfoot_frame].add_contact(
            transform=manif.SE3(position=np.array(ground_r_foot_position),
                                quaternion=r_foot_quat),
            activation_time=0.0,
            deactivation_time=r_deactivation_time)

        # ===================
        # SCALE LEFT CONTACTS
        # ===================

        # Store the previous unscaled and scaled contact (use the initial contact for both at the beginning)
        prev_contact = ground_l_foot_position
        prev_unscaled_contact = ground_l_foot_position

        # Update storage for plotting
        unscaled_left_footsteps_x.append(ground_l_foot_position[0])
        unscaled_left_footsteps_y.append(ground_l_foot_position[1])
        left_footsteps_x.append(ground_l_foot_position[0])
        left_footsteps_y.append(ground_l_foot_position[1])

        # Store unscaled contacts
        unscaled_l_contacts = []

        for contact in l_contacts[1:]:

            # Retrieve position
            ground_l_foot_position = [contact["2D_pos"][0], contact["2D_pos"][1], 0]
            ground_l_foot_position += np.array(ground_l_foot_position_offset)

            # Update unscaled contact list and storage for plotting
            unscaled_l_contacts.append(ground_l_foot_position)
            unscaled_left_footsteps_x.append(ground_l_foot_position[0])
            unscaled_left_footsteps_y.append(ground_l_foot_position[1])

        # Scale contacts
        for i in range(len(l_contacts[1:])):

            contact = l_contacts[1:][i]  # take orientation and timing from the original contact
            unscaled_contact = unscaled_l_contacts[i]  # take position from unscaled contacts

            # Retrieve the distance between two consecutive contacts of the same foot
            ground_l_foot_position = [unscaled_contact[0], unscaled_contact[1], 0]
            distance = ground_l_foot_position - np.array(prev_unscaled_contact)

            # Scale the step
            scaled_distance = self.footstep_scaling * distance

            # Compute the next contact as the previous contact plus the scaled step
            ground_l_foot_position = prev_contact + [scaled_distance[0], scaled_distance[1], 0]

            # Update the variables keeping track of the previous scaled and unscaled contacts
            prev_unscaled_contact = [unscaled_contact[0], unscaled_contact[1], 0]
            prev_contact = ground_l_foot_position

            # Update the storage for plotting
            left_footsteps_x.append(ground_l_foot_position[0])
            left_footsteps_y.append(ground_l_foot_position[1])

            # Retrieve orientation and timing from the original contact
            l_foot_RPY = [0.0, 0.0, contact["2D_orient"]]
            l_foot_rot = Rotation.from_euler('xyz', l_foot_RPY)
            l_foot_quat = l_foot_rot.as_quat()
            l_activation_time = self.time_scaling * (contact["activation_time"])
            l_deactivation_time = self.time_scaling * (contact["deactivation_time"])

            # Add the contact
            assert contact_list_map[lfoot_frame].add_contact(
                transform=manif.SE3(position=np.array(ground_l_foot_position), quaternion=l_foot_quat),
                activation_time=l_activation_time,
                deactivation_time=l_deactivation_time)

        # ====================
        # SCALE RIGHT CONTACTS
        # ====================

        # Store the previous unscaled and scaled contact (use the initial contact for both at the beginning)
        prev_contact = ground_r_foot_position
        prev_unscaled_contact = ground_r_foot_position

        # Update storage for plotting
        unscaled_right_footsteps_x.append(ground_r_foot_position[0])
        unscaled_right_footsteps_y.append(ground_r_foot_position[1])
        right_footsteps_x.append(ground_r_foot_position[0])
        right_footsteps_y.append(ground_r_foot_position[1])

        # Store unscaled contacts
        unscaled_r_contacts = []

        for contact in r_contacts[1:]:

            # Retrieve position
            ground_r_foot_position = [contact["2D_pos"][0], contact["2D_pos"][1], 0]
            ground_r_foot_position += np.array(ground_r_foot_position_offset)

            # Update unscaled contact list and storage for plotting
            unscaled_r_contacts.append(ground_r_foot_position)
            unscaled_right_footsteps_x.append(ground_r_foot_position[0])
            unscaled_right_footsteps_y.append(ground_r_foot_position[1])

        # Scale contacts
        for i in range(len(r_contacts[1:])):

            contact = r_contacts[1:][i]  # take orientation and timing from the original contact
            unscaled_contact = unscaled_r_contacts[i]  # take position from unscaled contacts

            # Retrieve the distance between two consecutive contacts of the same foot
            ground_r_foot_position = [unscaled_contact[0], unscaled_contact[1], 0]
            distance = ground_r_foot_position - np.array(prev_unscaled_contact)

            # Scale the step
            scaled_distance = self.footstep_scaling * distance

            # Compute the next contact as the previous contact plus the scaled step
            ground_r_foot_position = prev_contact + [scaled_distance[0], scaled_distance[1], 0]

            # Update the variables keeping track of the previous scaled and unscaled contacts
            prev_unscaled_contact = [unscaled_contact[0], unscaled_contact[1], 0]
            prev_contact = ground_r_foot_position

            # Update the storage for plotting
            right_footsteps_x.append(ground_r_foot_position[0])
            right_footsteps_y.append(ground_r_foot_position[1])

            # Retrieve orientation and timing from the original contact
            r_foot_RPY = [0.0, 0.0, contact["2D_orient"]]
            r_foot_rot = Rotation.from_euler('xyz', r_foot_RPY)
            r_foot_quat = r_foot_rot.as_quat()
            r_activation_time = self.time_scaling * (contact["activation_time"])
            r_deactivation_time = self.time_scaling * (contact["deactivation_time"])

            # Add the contact
            assert contact_list_map[rfoot_frame].add_contact(
                transform=manif.SE3(position=np.array(ground_r_foot_position), quaternion=r_foot_quat),
                activation_time=r_activation_time,
                deactivation_time=r_deactivation_time)

        # ============================
        # PLOT AND ASSIGN CONTACT LIST
        # ============================

        # Plot unscaled vs scaled footsteps
        plt.figure()
        plt.plot(unscaled_left_footsteps_x, unscaled_left_footsteps_y, 'r')
        plt.plot(unscaled_right_footsteps_x, unscaled_right_footsteps_y, 'r')
        plt.scatter(unscaled_left_footsteps_x, unscaled_left_footsteps_y, c='r')
        plt.scatter(unscaled_right_footsteps_x, unscaled_right_footsteps_y, c='r')
        plt.plot(left_footsteps_x, left_footsteps_y, 'b')
        plt.plot(right_footsteps_x, right_footsteps_y, 'b')
        plt.scatter(left_footsteps_x, left_footsteps_y, c='b')
        plt.scatter(right_footsteps_x, right_footsteps_y, c='b')
        plt.title("Unscaled footsteps (red) VS scaled footsteps (blue)")
        plt.axis("equal")
        plt.show(block=False)
        plt.pause(1.0)

        # Assign contact list
        phase_list = blf.contacts.ContactPhaseList()
        phase_list.set_lists(contact_lists=contact_list_map)
        self.contact_phase_list = phase_list


@dataclass
class PosturalExtractor:
    """Class to extract the postural at the desired frequency from the generated trajectory."""

    # Path for the generated footsteps
    posturals_path: str

    # Time scaling factor
    time_scaling: int

    # Fixed offset for the shoulder pitch and roll in order to spread the arms
    shoulder_offset: float

    # Joint postural references
    joint_references: List = field(default_factory=list)

    @staticmethod
    def build(posturals_path: str,
              time_scaling: int,
              shoulder_offset: float = 0.15) -> "PosturalExtractor":
        """Build an instance of PosturalExtractor."""

        return PosturalExtractor(posturals_path=posturals_path,
                                 time_scaling=time_scaling,
                                 shoulder_offset=shoulder_offset)

    def retrieve_joint_references(self, joints_list: List) -> None:
        """Retrieve postural references at the desired frequency from the generated trajectory."""

        # Retrieve original joint posturals from a JSON file
        with open(self.posturals_path, 'r') as openfile:
            posturals = json.load(openfile)
        joint_posturals = posturals["joints"]

        # Initialize list for the postural references at the desired frequency
        joint_references = []

        # Replicate the postural from adherent (frequency: 50 Hz) as many times as you need
        for joint_postural in joint_posturals:

            # 2 is to go from 50Hz (trajectory generation frequency) to 100Hz (trajectory control frequency),
            # then you need to take into account the time_scaling factor
            for i in range(2 * self.time_scaling):

                joint_reference = []

                for joint in joints_list:

                    if joint not in ["l_shoulder_roll", "r_shoulder_roll", "l_shoulder_pitch", "r_shoulder_pitch"]:
                        # Keep the joint postural as it is for the joints which are not in the shoulders
                        joint_reference.append(joint_postural[joint])
                    else:
                        # Spread the arms by adding a fixed offset to the shoulder joints
                        joint_reference.append(joint_postural[joint] + self.shoulder_offset)

                joint_references.append(joint_reference)

        # Assign joint references
        self.joint_references = joint_references


@dataclass
class LeggedOdometry:
    """Class for the computations related to the legged odometry estimator."""

    # Legged odometry
    lo_kindyn_desc: blf.floating_base_estimators.KinDynComputationsDescriptor
    legged_odom: blf.floating_base_estimators.LeggedOdometry

    # Fixed foot detector
    fixed_foot_detector: blf.contacts.FixedFootDetector
    foot_name_to_index: Dict

    # Base transform and velocity
    world_H_base: np.array = field(default_factory=lambda: np.array([]))
    base_twist: np.array = field(default_factory=lambda: np.array([]))

    # Fixed foot related quantities
    fixed_foot: blf.contacts.EstimatedContact = None
    fixed_foot_index: int = 0

    @staticmethod
    def build(robot_urdf: str, joints_list: List, dt: float, foot_name_to_index: Dict) -> "LeggedOdometry":
        """Build an instance of LeggedOdometry."""

        # Create KinDynComputationsDescriptor for legged odometry
        lo_kindyn_handler = blf.parameters_handler.StdParametersHandler()
        lo_kindyn_handler.set_parameter_string("model_file_name", robot_urdf)
        lo_kindyn_handler.set_parameter_vector_string("joints_list", joints_list)
        lo_kindyn_desc = blf.floating_base_estimators.construct_kindyncomputations_descriptor(lo_kindyn_handler)
        assert lo_kindyn_desc.is_valid()

        # Legged odometry configuration
        lo_params_handler = blf.parameters_handler.StdParametersHandler()
        lo_params_handler.set_parameter_float("sampling_period_in_s", dt)
        model_info_group = blf.parameters_handler.StdParametersHandler()
        model_info_group.set_parameter_string("base_link", "root_link")
        model_info_group.set_parameter_string("base_link_imu", "root_link")
        model_info_group.set_parameter_string("left_foot_contact_frame", "l_sole")
        model_info_group.set_parameter_string("right_foot_contact_frame", "r_sole")
        assert (lo_params_handler.set_group("ModelInfo", model_info_group))
        lo_group = blf.parameters_handler.StdParametersHandler()
        lo_group.set_parameter_string("initial_fixed_frame", "l_sole")
        lo_group.set_parameter_string("switching_pattern", "useExternal")
        assert lo_params_handler.set_group("LeggedOdom", lo_group)

        # Instantiate legged odometry
        legged_odom = blf.floating_base_estimators.LeggedOdometry()
        legged_odom.initialize(lo_params_handler, lo_kindyn_desc.kindyn)

        # Fixed foot detector configuration
        fixed_foot_detector_handler = blf.parameters_handler.StdParametersHandler()
        fixed_foot_detector_handler.set_parameter_float("sampling_time", dt)

        # Instantiate fixed foot detector
        fixed_foot_detector = blf.contacts.FixedFootDetector()
        fixed_foot_detector.initialize(fixed_foot_detector_handler)

        return LeggedOdometry(lo_kindyn_desc=lo_kindyn_desc,
                              legged_odom=legged_odom,
                              fixed_foot_detector=fixed_foot_detector,
                              foot_name_to_index=foot_name_to_index)

    def configure(self, contact_phase_list: blf.contacts.ContactList, joints_values: np.array,
                  joints_velocities: np.array) -> (np.array, np.array):
        """Initial configuration of the legged odometry estimator and the fixed foot detector."""

        # Pass the list of contacts to the fixed foot detector
        self.fixed_foot_detector.set_contact_phase_list(contact_phase_list)

        # Fill measurement buffers
        self.legged_odom.set_kinematics(joints_values, joints_velocities)

        # Retrieve initial fixed foot
        self.fixed_foot = self.fixed_foot_detector.get_fixed_foot()

        # Advance the legged odometry estimator
        assert self.legged_odom.set_contact_status(self.fixed_foot.name, self.fixed_foot.is_active,
                                                   self.fixed_foot.switch_time, self.fixed_foot.last_update_time)
        assert self.legged_odom.advance()

        # Update the fixed frame
        self.fixed_foot_index = self.foot_name_to_index[self.fixed_foot.name]
        self.legged_odom.change_fixed_frame(self.fixed_foot_index, self.fixed_foot.pose.quat(),
                                            self.fixed_foot.pose.translation())

        # Retrieve the output of the legged odometry
        out = self.legged_odom.get_output()
        self.world_H_base = out.base_pose.transform()
        self.base_twist = out.base_twist

        return self.world_H_base, self.base_twist

    def update(self, joints_values: np.array, joints_velocities: np.array) -> (np.array, np.array):
        """Retrieve updated base transform and velocity."""

        # Fill measurement buffers
        self.legged_odom.set_kinematics(joints_values, joints_velocities)

        # Retrieve current fixed foot
        self.fixed_foot = self.fixed_foot_detector.get_fixed_foot()

        # Advance the legged odometry estimator
        assert self.legged_odom.set_contact_status(self.fixed_foot.name, self.fixed_foot.is_active,
                                                   self.fixed_foot.switch_time, self.fixed_foot.last_update_time)
        assert self.legged_odom.advance()

        # Update the fixed frame
        self.fixed_foot_index = self.foot_name_to_index[self.fixed_foot.name]
        self.legged_odom.change_fixed_frame(self.fixed_foot_index, self.fixed_foot.pose.quat(),
                                            self.fixed_foot.pose.translation())

        # Retrieve the output of the legged odometry
        out = self.legged_odom.get_output()
        self.world_H_base = out.base_pose.transform()
        self.base_twist = out.base_twist

        return self.world_H_base, self.base_twist

    def update_fixed_foot_detector(self) -> None:
        """Update the fixed foot detector state."""

        self.fixed_foot_detector.advance()


@dataclass
class TrajectoryOptimization:
    """Class for the DCM and the swing foot planners constituting the trajectory optimization layer."""

    # DCM planner
    dcm_planner: blf.planners.TimeVaryingDCMPlanner = None
    dcm_planner_state: blf.planners.DCMPlannerState = None

    # Swing foot planners
    right_swing_planner: blf.planners.SwingFootPlanner = None
    left_swing_planner: blf.planners.SwingFootPlanner = None
    left_foot_state: blf.planners.SwingFootPlannerState = None
    right_foot_state: blf.planners.SwingFootPlannerState = None

    @staticmethod
    def build() -> "TrajectoryOptimization":
        """Build an instance of TrajectoryOptimization."""

        return TrajectoryOptimization()

    def configure(self, contact_phase_list: blf.contacts.ContactList, initial_com: np.array, dt: float) -> None:
        """Initial configuration of the DCM and swing foot planners."""

        # Get the dcm planner parameters
        dcm_planner_parameters = self.get_dcm_planner_parameters(dt=dt)

        # Initialize the dcm planner
        self.dcm_planner = blf.planners.TimeVaryingDCMPlanner()
        assert self.dcm_planner.initialize(handler=dcm_planner_parameters)

        # Set the contact configuration for the dcm planner
        assert self.dcm_planner.set_contact_phase_list(contact_phase_list=contact_phase_list)

        # Set the dcm planner initial state
        initial_state = self.get_dcm_planner_initial_state(initial_com)
        self.dcm_planner.set_initial_state(state=initial_state)

        # Configure swing foot planners
        self.right_swing_planner = self.get_swing_foot_planner(contact_list=contact_phase_list.lists()["r_sole"], dt=dt)
        self.left_swing_planner = self.get_swing_foot_planner(contact_list=contact_phase_list.lists()["l_sole"], dt=dt)

    def update(self) -> None:
        """Retrieve and update the DMC and swing foot planners state."""

        # Get the planners states
        self.dcm_planner_state = self.dcm_planner.get_output()
        self.left_foot_state = self.left_swing_planner.get_output()
        self.right_foot_state = self.right_swing_planner.get_output()

        # Advance the planners
        self.dcm_planner.advance()
        self.left_swing_planner.advance()
        self.right_swing_planner.advance()

    @staticmethod
    def get_dcm_planner_initial_state(com_position) -> blf.planners.DCMPlannerState:
        """Define the initial state for the DCM planner."""

        # Set the initial state
        initial_state = blf.planners.DCMPlannerState()
        initial_state.dcm_position = com_position
        initial_state.dcm_velocity = np.zeros(3)
        initial_state.vrp_position = initial_state.dcm_position
        initial_state.omega = np.sqrt(blf.math.StandardAccelerationOfGravitation / initial_state.dcm_position[2])

        return initial_state

    @staticmethod
    def get_swing_foot_planner(contact_list: blf.contacts.ContactList, dt: float) -> blf.planners.SwingFootPlanner:
        """Define the parameters for the swing foot planner and configure the planner."""

        parameters_handler = blf.parameters_handler.StdParametersHandler()
        parameters_handler.set_parameter_float("sampling_time", dt)
        parameters_handler.set_parameter_float("step_height", 0.025)
        parameters_handler.set_parameter_float("foot_apex_time", 0.4)
        parameters_handler.set_parameter_float("foot_landing_velocity", 0.0)
        parameters_handler.set_parameter_float("foot_landing_acceleration", 0)
        parameters_handler.set_parameter_float("foot_take_off_velocity", 0.0)
        parameters_handler.set_parameter_float("foot_take_off_acceleration", 0)
        parameters_handler.set_parameter_string("interpolation_method", "min_acceleration")

        planner = blf.planners.SwingFootPlanner()
        assert planner.initialize(handler=parameters_handler)
        planner.set_contact_list(contact_list=contact_list)

        return planner

    @staticmethod
    def get_dcm_planner_parameters(dt: float) -> blf.parameters_handler.StdParametersHandler:
        """Define the parameters for the DCM planner."""

        handler = blf.parameters_handler.StdParametersHandler()
        handler.set_parameter_float(name="planner_sampling_time", value=dt)
        handler.set_parameter_int(name="number_of_foot_corners", value=4)

        # Foot corners (seen by the planner)
        handler.set_parameter_vector_float(name="foot_corner_0", value=[0.05, 0.005, 0.0])
        handler.set_parameter_vector_float(name="foot_corner_1", value=[0.05, -0.005, 0.0])
        handler.set_parameter_vector_float(name="foot_corner_2", value=[-0.00, -0.005, 0.0])
        handler.set_parameter_vector_float(name="foot_corner_3", value=[-0.00, 0.005, 0.0])

        # Set the weights of the cost function
        handler.set_parameter_float(name="omega_dot_weight", value=10.0)
        handler.set_parameter_float(name="dcm_tracking_weight", value=10.0)
        handler.set_parameter_float(name="omega_dot_rate_of_change_weight", value=10.0)
        handler.set_parameter_float(name="vrp_rate_of_change_weight", value=10.0)
        handler.set_parameter_float(name="dcm_rate_of_change_weight", value=10000000.0)

        return handler


@dataclass
class SimplifiedModelControl:
    """Class for the DCM instantaneous controller (Eq.1 in the paper) and the ZMP-CoM controller (Eq.2 in the paper)
    constituting the simplified model control layer.
    """

    # Controller gains
    k_zmp: float
    k_dcm: float
    k_com: float

    # CoM and omega
    com_position: np.array
    com_height: float
    omega: float
    com_velocity: np.array = field(default_factory=lambda: np.array([0, 0]))

    # Storage for plotting
    zmp_pos_des: List = field(default_factory=list)
    com_velocity_from_dcm: List = field(default_factory=list)
    com_pos_error: float = 0
    zmp_pos_error: float = 0
    dcm_pos_error: float = 0

    @staticmethod
    def build(com_initial_position: np.ndarray,
              k_zmp: float = 1.0,
              k_dcm: float = 1.1,
              k_com: float = 4.0) -> "SimplifiedModelControl":
        """Build an instance of SimplifiedModelControl."""

        com_position = np.array([com_initial_position[0],com_initial_position[1]])
        com_height = com_initial_position[2]
        omega = np.sqrt(blf.math.StandardAccelerationOfGravitation/com_height)

        return SimplifiedModelControl(k_zmp=k_zmp, k_dcm=k_dcm, k_com=k_com,
                                      com_position=com_position, com_height=com_height, omega=omega)

    def advance(self, state: blf.planners.DCMPlannerState, dt: float,
                com_pos_meas: List, zmp_pos_meas: List, dcm_pos_meas: List) -> None:
        """Apply the DCM instantaneous control law and the ZMP-CoM control law to retrieve the desired CoM position."""

        # Retrieve bi-dimensional desired quantities from the DCM planner
        dcm_pos_des = np.array([state.dcm_position[0], state.dcm_position[1]])
        dcm_vel_des = np.array([state.dcm_velocity[0], state.dcm_velocity[1]])

        # DCM instantaneous control law (Eq.1 in the paper)
        self.zmp_pos_des = dcm_pos_des - dcm_vel_des/state.omega - self.k_dcm * (dcm_pos_des - dcm_pos_meas)

        # ZMP-CoM control law (Eq.2 in the paper)
        self.com_velocity_from_dcm = state.omega * (dcm_pos_des - self.com_position)
        self.com_velocity = self.com_velocity_from_dcm + \
                            self.k_com * (self.com_position - com_pos_meas) - \
                            self.k_zmp * (self.zmp_pos_des - zmp_pos_meas)

        # Euler integration step
        self.com_position += dt * self.com_velocity

    def update_errors(self, com_pos_meas: List, zmp_pos_meas: List, dcm_pos_meas: List,
                      state: blf.planners.DCMPlannerState) -> None:
        """Update CoM, ZMP and DCM position errors for plotting."""

        self.com_pos_error = np.linalg.norm(np.array(com_pos_meas) - self.com_position)
        self.zmp_pos_error = np.linalg.norm(np.array(zmp_pos_meas) - self.zmp_pos_des)
        self.dcm_pos_error = np.linalg.norm(np.array(dcm_pos_meas) - [state.dcm_position[0], state.dcm_position[1]])


@dataclass
class WholeBodyQPControl:
    """Class for the inverse kinematics exploited in the whole-body Quadratic Programming (QP) control layer."""

    # Inverse kinematics
    qp_ik: blf.ik.QPInverseKinematics

    # Tasks for the ik
    com_task: blf.ik.CoMTask = field(default_factory=lambda: blf.ik.CoMTask())
    rf_se3_task: blf.ik.SE3Task = field(default_factory=lambda: blf.ik.SE3Task())
    lf_se3_task: blf.ik.SE3Task = field(default_factory=lambda: blf.ik.SE3Task())
    chest_so3_task: blf.ik.SO3Task = field(default_factory=lambda: blf.ik.SO3Task())
    joint_tracking_task: blf.ik.JointTrackingTask = field(default_factory=lambda: blf.ik.JointTrackingTask())

    @staticmethod
    def build() -> "WholeBodyQPControl":
        """Build an instance of WholeBodyQPControl."""

        # Set qp ik parameters
        qp_ik_param_handler = blf.parameters_handler.StdParametersHandler()
        qp_ik_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")

        # Initialize qp ik
        qp_ik = blf.ik.QPInverseKinematics()
        assert qp_ik.initialize(handler=qp_ik_param_handler)

        return WholeBodyQPControl(qp_ik=qp_ik)

    def configure_com_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure CoM task and add it as hard constraint."""

        # Configure CoM task
        com_param_handler = blf.parameters_handler.StdParametersHandler()
        com_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        com_param_handler.set_parameter_float(name="kp_linear", value=1.0)
        assert self.com_task.set_kin_dyn(kindyn)
        assert self.com_task.initialize(param_handler=com_param_handler)
        com_var_handler = blf.system.VariablesHandler()
        assert com_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.com_task.set_variables_handler(variables_handler=com_var_handler)

        # Add CoM task as hard constraint
        assert self.qp_ik.add_task(task=self.com_task, taskName="Com_task", priority=0)

    def configure_right_foot_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure right foot SE3 task and add it as hard constraint."""

        # Configure right foot SE3 task
        rf_se3_param_handler = blf.parameters_handler.StdParametersHandler()
        rf_se3_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        rf_se3_param_handler.set_parameter_string(name="frame_name", value="r_sole")
        rf_se3_param_handler.set_parameter_float(name="kp_linear", value=7.5)
        rf_se3_param_handler.set_parameter_float(name="kp_angular", value=10.0)
        assert self.rf_se3_task.set_kin_dyn(kindyn)
        assert self.rf_se3_task.initialize(param_handler=rf_se3_param_handler)
        rf_se3_var_handler = blf.system.VariablesHandler()
        assert rf_se3_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.rf_se3_task.set_variables_handler(variables_handler=rf_se3_var_handler)

        # Add right foot SE3 task as hard constraint
        assert self.qp_ik.add_task(task=self.rf_se3_task, taskName="rf_se3_task", priority=0)

    def configure_left_foot_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure left foot SE3 task and add it as hard constraint."""

        # Configure left foot SE3 task
        lf_se3_param_handler = blf.parameters_handler.StdParametersHandler()
        lf_se3_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        lf_se3_param_handler.set_parameter_string(name="frame_name", value="l_sole")
        lf_se3_param_handler.set_parameter_float(name="kp_linear", value=7.5)
        lf_se3_param_handler.set_parameter_float(name="kp_angular", value=10.0)
        assert self.lf_se3_task.set_kin_dyn(kindyn)
        assert self.lf_se3_task.initialize(param_handler=lf_se3_param_handler)
        lf_se3_var_handler = blf.system.VariablesHandler()
        assert lf_se3_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.lf_se3_task.set_variables_handler(variables_handler=lf_se3_var_handler)

        # Add left foot SE3 task as hard constraint
        assert self.qp_ik.add_task(task=self.lf_se3_task, taskName="lf_se3_task", priority=0)

    def configure_chest_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure chest SO3 task and add it as soft constraint."""

        # Configure chest SO3 task
        chest_so3_param_handler = blf.parameters_handler.StdParametersHandler()
        chest_so3_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        chest_so3_param_handler.set_parameter_string(name="frame_name", value="chest")
        chest_so3_param_handler.set_parameter_float(name="kp_angular", value=10.0)
        assert self.chest_so3_task.set_kin_dyn(kindyn)
        assert self.chest_so3_task.initialize(param_handler=chest_so3_param_handler)
        chest_so3_var_handler = blf.system.VariablesHandler()
        assert chest_so3_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.chest_so3_task.set_variables_handler(variables_handler=chest_so3_var_handler)

        # Add chest SO3 task as soft constraint, leaving the yaw free
        assert self.qp_ik.add_task(task=self.chest_so3_task, taskName="chest_so3_task", priority=1, weight=[10, 10, 0])

    def configure_joint_tracking_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure joint tracking task and add it as soft constraint."""

        # Configure joint tracking task
        joint_tracking_param_handler = blf.parameters_handler.StdParametersHandler()
        joint_tracking_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        joint_kp = [7] * kindyn.get_nr_of_dofs()
        joint_tracking_param_handler.set_parameter_vector_float(name="kp", value=joint_kp)
        assert self.joint_tracking_task.set_kin_dyn(kindyn)
        assert self.joint_tracking_task.initialize(param_handler=joint_tracking_param_handler)
        joint_tracking_var_handler = blf.system.VariablesHandler()
        assert joint_tracking_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.joint_tracking_task.set_variables_handler(variables_handler=joint_tracking_var_handler)

        # Add joint tracking task as soft constraint
        joint_weights = [1] * kindyn.get_nr_of_dofs()
        assert self.qp_ik.add_task(task=self.joint_tracking_task, taskName="joint_tracking_task", priority=1, weight=joint_weights)

    def finalize_ik(self, joints_list: List) -> None:
        """Once all the tasks have been added, finalize the ik."""

        # Finalize the qp inverse kinematics
        qp_ik_var_handler = blf.system.VariablesHandler()
        assert qp_ik_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.qp_ik.finalize(handler=qp_ik_var_handler)

    def set_joint_tracking_set_point(self, joint_reference: List) -> None:
        """Set set point for the joint tracking task."""

        assert self.joint_tracking_task.set_set_point(joint_position=joint_reference)

    def set_chest_set_point(self, chest_quat_reference: List) -> None:
        """Set set point for the chest SO3 task."""

        I_R_F = manif.SO3(quaternion=chest_quat_reference)
        angularVelocity = manif.SO3Tangent([0.0] * 3)
        assert self.chest_so3_task.set_set_point(I_R_F=I_R_F, angular_velocity=angularVelocity)

    def set_right_foot_set_point(self, right_foot_state: blf.planners.SwingFootPlannerState) -> None:
        """Set set point for the right foot SE3 task."""

        translation = right_foot_state.transform.translation()
        I_H_F = manif.SE3(position=np.array(translation), quaternion=right_foot_state.transform.quat())
        mixed_velocity = manif.SE3Tangent(right_foot_state.mixed_velocity)
        assert self.rf_se3_task.set_set_point(I_H_F=I_H_F, mixed_velocity=mixed_velocity)

    def set_left_foot_set_point(self, left_foot_state: blf.planners.SwingFootPlannerState) -> None:
        """Set set point for the left foot SE3 task."""

        translation = left_foot_state.transform.translation()
        I_H_F = manif.SE3(position=np.array(translation), quaternion=left_foot_state.transform.quat())
        mixed_velocity = manif.SE3Tangent(left_foot_state.mixed_velocity)
        assert self.lf_se3_task.set_set_point(I_H_F=I_H_F, mixed_velocity=mixed_velocity)

    def set_com_set_point(self, com_position: np.array, com_height: float, com_velocity: np.array) -> None:
        """Set set point for the CoM task."""

        assert self.com_task.set_set_point(position=np.array([com_position[0], com_position[1], com_height]),
                                           velocity=np.array([com_velocity[0], com_velocity[1], 0]))

    def solve_ik(self) -> np.array:
        """Solve the ik problem and return the solution (desired joint velocities)."""

        try:
            # Solve the inverse kinematics
            assert self.qp_ik.advance()
        except:
            # If IK fails to find a solution, trajectory tracking is interrupted
            print("IK solution not found. Trajectory tracking interrupted.")

        # Get the inverse kinematics output
        ik_state = self.qp_ik.get_output()
        assert self.qp_ik.is_output_valid()

        return ik_state.joint_velocity


@dataclass
class TrajectoryController:
    """Class for controlling trajectories. It directly handles robot control and sensing while exploiting the
    dedicated components for the other tasks involved in the trajectory control pipeline."""

    # Robot model and joints
    robot_urdf: str
    joints_list: List
    initial_joint_reference: List

    # Components of the trajectory controller
    storage: StorageHandler
    footsteps_extractor: FootstepsExtractor
    postural_extractor: PosturalExtractor
    legged_odometry: LeggedOdometry
    trajectory_optimization: TrajectoryOptimization
    whole_body_qp_control: WholeBodyQPControl
    simplified_model_control: SimplifiedModelControl = None
    joints_integrator: Integrator = None

    # True if the learning-based postural is exploited
    use_joint_references: bool = True

    # Control loop rate (100 Hz)
    dt: float = 0.01

    # Auxiliary variable for synchronization
    curr_dt: float = 0

    # Robot control
    control_board: blf.robot_interface.PolyDriverDescriptor = None
    robot_control: blf.robot_interface.YarpRobotControl = None

    # Sensing
    cartesian_left_wrench: blf.robot_interface.PolyDriverDescriptor = None
    cartesian_right_wrench: blf.robot_interface.PolyDriverDescriptor = None
    sensor_bridge: blf.robot_interface.YarpSensorBridge = None

    # Kindyn descriptors
    kindyn_meas_desc: blf.floating_base_estimators.KinDynComputationsDescriptor = None
    kindyn_des_desc: blf.floating_base_estimators.KinDynComputationsDescriptor = None

    # Desired quantities
    joints_values_des: np.array = field(default_factory=lambda: np.array([]))
    joints_velocities_des: np.array = field(default_factory=lambda: np.array([]))

    # Measured quantities
    left_wrench: np.array = field(default_factory=lambda: np.array([]))
    right_wrench: np.array = field(default_factory=lambda: np.array([]))
    joints_values: np.array = field(default_factory=lambda: np.array([]))
    joints_velocities: np.array = field(default_factory=lambda: np.array([]))
    zmp_pos_meas: np.array = field(default_factory=lambda: np.array([]))
    com_pos_meas: np.array = field(default_factory=lambda: np.array([]))
    com_vel_meas: np.array = field(default_factory=lambda: np.array([]))
    dcm_pos_meas: np.array = field(default_factory=lambda: np.array([]))

    @staticmethod
    def build(robot_urdf: str, footsteps_path: str, posturals_path: str, storage_path: str, time_scaling: int,
              footstep_scaling: float, use_joint_references: bool, controlled_joints: List, foot_name_to_index: Dict,
              initial_joint_reference: List, shoulder_offset: float = 0.15) -> "TrajectoryController":
        """Build an instance of TrajectoryController."""

        # Control loop rate (100 Hz)
        dt = 0.01

        # Storage
        storage = StorageHandler.build(storage_path=storage_path,
                                       joints_list=controlled_joints,
                                       use_joint_references=use_joint_references)

        # Footsteps extractor
        footsteps_extractor = FootstepsExtractor.build(footsteps_path=footsteps_path,
                                                       footstep_scaling=footstep_scaling,
                                                       time_scaling=time_scaling)
        footsteps_extractor.retrieve_contacts()

        # Postural extractor
        postural_extractor = PosturalExtractor.build(posturals_path=posturals_path,
                                                     time_scaling=time_scaling,
                                                     shoulder_offset=shoulder_offset)
        postural_extractor.retrieve_joint_references(joints_list=controlled_joints)

        # Legged odom
        legged_odometry = LeggedOdometry.build(robot_urdf=robot_urdf, joints_list=controlled_joints,
                                               dt=dt, foot_name_to_index=foot_name_to_index)

        # Trajectory Optimization
        trajectory_optimization = TrajectoryOptimization.build()

        # Whole Body QP Control
        whole_body_qp_control = WholeBodyQPControl.build()

        return TrajectoryController(robot_urdf=robot_urdf,
                                    joints_list=controlled_joints,
                                    initial_joint_reference=initial_joint_reference,
                                    storage=storage,
                                    footsteps_extractor=footsteps_extractor,
                                    postural_extractor=postural_extractor,
                                    legged_odometry=legged_odometry,
                                    trajectory_optimization=trajectory_optimization,
                                    whole_body_qp_control=whole_body_qp_control,
                                    use_joint_references=use_joint_references)

    # =============
    # CONFIGURATION
    # =============

    def configure(self, k_zmp: float = 1.0, k_dcm: float = 1.1, k_com: float = 4.0) -> None:
        """Configure the entire trajectory control pipeline."""

        # Configure the control board and the sensor bridge
        self.configure_robot_control_and_sensor_bridge()

        # Compute and set the initial joint reference
        self.set_initial_joint_reference()

        # Configure the kindyn descriptors
        self.configure_kindyn_descriptors()

        # Configure the inverse kinematics
        self.configure_blf_ik()

        # Configure the legged odometry estimator
        self.configure_legged_odom()

        # Configure DCM and swing foot planners
        self.configure_planners()

        # Configure instantaneous DCM and ZMP-CoM controllers
        self.configure_controllers(k_zmp=k_zmp, k_dcm=k_dcm, k_com=k_com)

        # Configure the integrator from joint velocities to joint positions
        self.configure_joints_integrator()

    def configure_robot_control_and_sensor_bridge(self) -> None:
        """Setup the robot control and the sensor bridge."""

        # Control board configuration
        handler = blf.parameters_handler.StdParametersHandler()
        handler.set_parameter_vector_string("joints_list", self.joints_list)
        handler.set_parameter_vector_string("remote_control_boards",
                                            ["left_leg", "right_leg", "torso", "left_arm", "head", "right_arm"])
        handler.set_parameter_string("local_prefix", "test_local")
        handler.set_parameter_float("positioning_duration", 3.0)
        handler.set_parameter_float("positioning_tolerance", 0.001)
        handler.set_parameter_string("robot_name", "icubSim")
        if self.use_joint_references:
            handler.set_parameter_float("position_direct_max_admissible_error", 0.25)
        else:
            handler.set_parameter_float("position_direct_max_admissible_error", 0.1)

        # Configuration of the device to read left wrench
        l_wrench_handler = blf.parameters_handler.StdParametersHandler()
        l_wrench_handler.set_parameter_string("description", "/test_local_l_foot_wrench")  # name of the device: it must be /<local_prefix><local_port_name_postfix>
        l_wrench_handler.set_parameter_string("remote_port_name", "/wholeBodyDynamics/left_foot/cartesianEndEffectorWrench:o")  # name of the yarp port to read
        l_wrench_handler.set_parameter_string("local_prefix", "test_local")  # free choice, but it must be consistent with 'description'
        l_wrench_handler.set_parameter_string("local_port_name_postfix", "_l_foot_wrench")  # free choice, but it must be consistent with 'description'
        l_wrench_handler.set_parameter_vector_string("cartesian_wrenches_list", ["/test_local_l_foot_wrench", "/test_local_r_foot_wrench"])  # list of the device names: they must be consistent with 'description'

        # Configuration of the device to read right wrench
        r_wrench_handler = blf.parameters_handler.StdParametersHandler()
        r_wrench_handler.set_parameter_string("description", "/test_local_r_foot_wrench")  # name of the device: it must be /<local_prefix><local_port_name_postfix>
        r_wrench_handler.set_parameter_string("remote_port_name", "/wholeBodyDynamics/right_foot/cartesianEndEffectorWrench:o")  # name of the yarp port to read
        r_wrench_handler.set_parameter_string("local_prefix", "test_local")  # free choice, but it must be consistent with 'description'
        r_wrench_handler.set_parameter_string("local_port_name_postfix", "_r_foot_wrench")  # free choice, but it must be consistent with 'description'
        r_wrench_handler.set_parameter_vector_string("cartesian_wrenches_list", ["/test_local_l_foot_wrench", "/test_local_r_foot_wrench"])  # list of the device names: they must be consistent with 'description'

        # Configuration for sensing
        handler.set_parameter_bool("check_for_nan", False)
        handler.set_parameter_bool("stream_joint_states", True)
        handler.set_parameter_bool("stream_cartesian_wrenches", True)
        handler.set_group(name="CartesianWrenches", new_group=l_wrench_handler)

        # Create the sensor device for the left wrench
        self.cartesian_left_wrench = blf.robot_interface.construct_generic_sensor_client(l_wrench_handler)
        time.sleep(1)

        # Create the sensor device for the right wrench
        self.cartesian_right_wrench = blf.robot_interface.construct_generic_sensor_client(r_wrench_handler)
        time.sleep(1)

        # Create the control board
        self.control_board = blf.robot_interface.construct_remote_control_board_remapper(handler)
        time.sleep(1)

        # Create the robot control
        self.robot_control = blf.robot_interface.YarpRobotControl()
        self.robot_control.initialize(handler)
        self.robot_control.set_driver(self.control_board.poly)

        # Create the sensor bridge
        self.sensor_bridge = blf.robot_interface.YarpSensorBridge()
        self.sensor_bridge.initialize(handler)
        self.sensor_bridge.set_drivers_list([self.control_board, self.cartesian_left_wrench, self.cartesian_right_wrench])

    def set_initial_joint_reference(self) -> None:
        """Set the initial joint reference."""

        # Set the initial joint reference in Position mode and wait for the robot to reach it
        self.robot_control.set_references(self.initial_joint_reference, blf.robot_interface.IRobotControl.ControlMode.Position)
        time.sleep(3)

    def configure_kindyn_descriptors(self) -> None:
        """Setup the kindyn descriptors for desired and measured values."""

        # create KinDynComputationsDescriptor for desired values (it will be exploited in the ik tasks)
        kindyn_des_handler = blf.parameters_handler.StdParametersHandler()
        kindyn_des_handler.set_parameter_string("model_file_name", self.robot_urdf)
        kindyn_des_handler.set_parameter_vector_string("joints_list", self.joints_list)
        self.kindyn_des_desc = blf.floating_base_estimators.construct_kindyncomputations_descriptor(kindyn_des_handler)
        assert self.kindyn_des_desc.is_valid()
        self.kindyn_des_desc.kindyn.set_floating_base("root_link")

        # create KinDynComputationsDescriptor for measured values (for comparison only)
        kindyn_meas_handler = blf.parameters_handler.StdParametersHandler()
        kindyn_meas_handler.set_parameter_string("model_file_name", self.robot_urdf)
        kindyn_meas_handler.set_parameter_vector_string("joints_list", self.joints_list)
        self.kindyn_meas_desc = blf.floating_base_estimators.construct_kindyncomputations_descriptor(kindyn_meas_handler)
        assert self.kindyn_meas_desc.is_valid()
        self.kindyn_meas_desc.kindyn.set_floating_base("root_link")

    def configure_blf_ik(self) -> None:
        """Setup the inverse kinematics by adding tasks and setting fixed set points."""

        # Configure CoM task
        self.whole_body_qp_control.configure_com_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Configure right foot task
        self.whole_body_qp_control.configure_right_foot_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Configure left foot task
        self.whole_body_qp_control.configure_left_foot_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Configure chest task
        self.whole_body_qp_control.configure_chest_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Configure joint tracking task
        self.whole_body_qp_control.configure_joint_tracking_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Finalize the ik
        self.whole_body_qp_control.finalize_ik(joints_list=self.joints_list)

        # Set fixed set points for the joint tracking task (initial pose) and the chest task (straight chest)
        self.whole_body_qp_control.set_joint_tracking_set_point(joint_reference=self.initial_joint_reference)
        self.whole_body_qp_control.set_chest_set_point(chest_quat_reference=[0.5, 0.5, 0.5, 0.5])

    def configure_legged_odom(self) -> None:
        """Setup the legged odometry estimator."""

        # Retrieve data from the simulator
        self.sensor_bridge.advance()
        _, self.joints_values, _ = self.sensor_bridge.get_joint_positions()
        _, self.joints_velocities, _ = self.sensor_bridge.get_joint_velocities()

        # Set desired values to the first reading
        self.joints_values_des = self.joints_values
        self.joints_velocities_des = self.joints_velocities

        # Configure legged odometry and retrieve base transform and velocity
        world_H_base, base_twist = self.legged_odometry.configure(contact_phase_list=self.footsteps_extractor.contact_phase_list,
                                                                  joints_values=self.joints_values,
                                                                  joints_velocities=self.joints_velocities)

        # Update the kindyn descriptors accordingly
        assert self.kindyn_des_desc.kindyn.set_robot_state(world_H_base, self.joints_values_des, base_twist, self.joints_velocities_des, world_gravity())
        assert self.kindyn_meas_desc.kindyn.set_robot_state(world_H_base, self.joints_values, base_twist, self.joints_velocities, world_gravity())

    def configure_planners(self) -> None:
        """Setup DCM and swing foot planners."""

        self.trajectory_optimization.configure(contact_phase_list=self.footsteps_extractor.contact_phase_list,
                                               initial_com=self.kindyn_des_desc.kindyn.get_center_of_mass_position(),
                                               dt=self.dt)

    def configure_controllers(self, k_zmp: float, k_dcm: float, k_com: float) -> None:
        """Setup the instantaneous DCM controller and the ZMP-CoM controller."""

        initial_com = self.kindyn_des_desc.kindyn.get_center_of_mass_position()
        self.simplified_model_control = SimplifiedModelControl.build(com_initial_position=initial_com,
                                                                     k_zmp=k_zmp, k_com=k_com, k_dcm=k_dcm)

    def configure_joints_integrator(self) -> None:
        """Initialize the integrator for the joint references."""

        self.joints_integrator = Integrator.build(joints_initial_position=self.joints_values, dt=self.dt)

    # ====================
    # TRAJECTORY PLANNNING
    # ====================

    def compute_dcm_trajectory(self) -> None:

        # Compute the dcm trajectory
        assert self.trajectory_optimization.dcm_planner.compute_trajectory()

    # ==================
    # TRAJECTORY CONTROL
    # ==================

    def read_data(self) -> None:
        """Retrieve measured joint values and velocities along with feet wrenches."""

        # Advance the sensor bridge
        self.sensor_bridge.advance()

        # Read joint values and velocities
        _, self.joints_values, _ = self.sensor_bridge.get_joint_positions()
        _, self.joints_velocities, _ = self.sensor_bridge.get_joint_velocities()

        # Measure local contact wrenches in the feet reference frames
        _, self.left_wrench, _ = self.sensor_bridge.get_cartesian_wrench(wrench_name="/test_local_l_foot_wrench")
        _, self.right_wrench, _ = self.sensor_bridge.get_cartesian_wrench(wrench_name="/test_local_r_foot_wrench")

    def update_legged_odom(self) -> None:
        """Update legged odometry estimator and kindyn descriptors."""

        # Retrieve updated base transform and velocity
        world_H_base, base_twist = self.legged_odometry.update(joints_values=self.joints_values,
                                                               joints_velocities=self.joints_velocities)

        # Update the kindyn descriptors accordingly
        assert self.kindyn_des_desc.kindyn.set_robot_state(world_H_base, self.joints_values_des, base_twist, self.joints_velocities_des, world_gravity())
        assert self.kindyn_meas_desc.kindyn.set_robot_state(world_H_base, self.joints_values, base_twist, self.joints_velocities, world_gravity())

    def update_planners(self) -> None:
        """Update the DCM and swing foot planners along with the fixed foot detector."""

        # Update DCM and swing foot planners
        self.trajectory_optimization.update()

        # Update fixed foot detector
        self.legged_odometry.update_fixed_foot_detector()

    def update_controllers(self) -> None:
        """Retrieve the desired CoM position."""

        # Compute measured ZMP from the contact wrenches
        self.zmp_pos_meas = compute_zmp(self.kindyn_meas_desc.kindyn, self.left_wrench, self.right_wrench)

        # Retrieve measured CoM position and velocity
        W_pos_CoM = self.kindyn_meas_desc.kindyn.get_center_of_mass_position()
        W_vel_CoM = self.kindyn_meas_desc.kindyn.get_center_of_mass_velocity()
        self.com_pos_meas = np.array([W_pos_CoM[0], W_pos_CoM[1]])
        self.com_vel_meas = np.array([W_vel_CoM[0], W_vel_CoM[1]])

        # Compute measured DCM from measured CoM position and velocity
        dcm_planner_state = self.trajectory_optimization.dcm_planner_state
        self.dcm_pos_meas = self.com_pos_meas + 1 / dcm_planner_state.omega * self.com_vel_meas

        # Compute desired CoM position
        self.simplified_model_control.advance(state=dcm_planner_state, dt=self.dt, com_pos_meas=self.com_pos_meas,
                                              zmp_pos_meas=self.zmp_pos_meas, dcm_pos_meas=self.dcm_pos_meas)

        # Update the CoM, ZMP and DCM position errors storage
        self.simplified_model_control.update_errors(com_pos_meas=self.com_pos_meas, zmp_pos_meas=self.zmp_pos_meas,
                                                    dcm_pos_meas=self.dcm_pos_meas, state=dcm_planner_state)

    def update_ik_targets(self, idx) -> None:
        """Update the ik targets that vary over time."""

        # Update right foot target
        self.whole_body_qp_control.set_right_foot_set_point(right_foot_state=self.trajectory_optimization.right_foot_state)

        # Update left foot target
        self.whole_body_qp_control.set_left_foot_set_point(left_foot_state=self.trajectory_optimization.left_foot_state)

        # Update the CoM target
        self.whole_body_qp_control.set_com_set_point(com_position=self.simplified_model_control.com_position,
                                                     com_height=self.simplified_model_control.com_height,
                                                     com_velocity=self.simplified_model_control.com_velocity)

        if self.use_joint_references:
            # Update set point for the joint tracking task
            self.whole_body_qp_control.set_joint_tracking_set_point(
                joint_reference=self.postural_extractor.joint_references[round(idx * 1/self.dt)])

    def retrieve_joint_reference(self) -> None:
        """Retrive desired joint velocities and integrate them in order to obtain desired joint positions."""

        # Retrieve des joint velocities
        self.joints_velocities_des = self.whole_body_qp_control.solve_ik()

        # Retrieve des joint positions
        self.joints_integrator.advance(joints_velocity=self.joints_velocities_des)
        self.joints_values_des = self.joints_integrator.get_joints_position()

    def set_current_joint_reference(self, idx):
        """"Set synchronously the joint references to the robot."""

        if idx == 0:

            input("Press enter to start trajectory control (PositionDirect)")

            # Retrieve the initial time
            initial_time = yarp.now()
            print("initial time", initial_time)
            self.curr_dt = initial_time

        else:

            # Set the current reference in PositionDirect mode
            self.robot_control.set_references(self.joints_values_des.tolist(), blf.robot_interface.IRobotControl.ControlMode.PositionDirect)

            # Synchronization
            self.curr_dt = synchronize(self.curr_dt, dt=self.dt)

    # =======
    # STORAGE
    # =======

    def update_storage(self, idx) -> None:
        """Update the storage of the quantities of interest."""

        # Update joints storage
        self.storage.update_joints_storage(joints_values_des=self.joints_values_des,
                                           joints_values=self.joints_values,
                                           joints_velocities_des=self.joints_velocities_des,
                                           joints_velocities=self.joints_velocities,
                                           joint_references=self.postural_extractor.joint_references,
                                           dt=self.dt,
                                           idx=idx)

        # Update legged odometry and fixed foot detector storage
        self.storage.update_legged_odom_and_fixed_foot_storage(world_H_base=self.legged_odometry.world_H_base.tolist(),
                                                               base_twist=self.legged_odometry.base_twist.tolist(),
                                                               fixed_foot_index=self.legged_odometry.fixed_foot_index)

        # Update center of mass storage
        self.storage.update_com_storage(com_pos_des=self.simplified_model_control.com_position,
                                        com_vel_des=self.simplified_model_control.com_velocity,
                                        com_vel_des_from_dcm=self.simplified_model_control.com_velocity_from_dcm,
                                        com_pos_meas=self.com_pos_meas,
                                        com_vel_meas=self.com_vel_meas,
                                        com_pos_error=self.simplified_model_control.com_pos_error)

        # Update zero-moment point storage
        self.storage.update_zmp_storage(zmp_pos_des=self.simplified_model_control.zmp_pos_des,
                                        zmp_pos_meas=self.zmp_pos_meas,
                                        zmp_pos_error=self.simplified_model_control.zmp_pos_error)

        # Update divergent component of motion storage
        self.storage.update_dcm_storage(dcm_position=self.trajectory_optimization.dcm_planner_state.dcm_position,
                                        dcm_velocity=self.trajectory_optimization.dcm_planner_state.dcm_velocity,
                                        dcm_pos_meas=self.dcm_pos_meas,
                                        dcm_pos_error=self.simplified_model_control.dcm_pos_error)

        # Update right foot storage
        self.storage.update_right_foot_storage(right_foot_des=self.trajectory_optimization.right_foot_state.transform.translation(),
                                               right_foot_meas_transform=self.kindyn_meas_desc.kindyn.get_world_transform("r_sole"),
                                               right_foot_des_transform=self.kindyn_des_desc.kindyn.get_world_transform("r_sole"),
                                               right_wrench=self.right_wrench)

        # Update left foot storage
        self.storage.update_left_foot_storage(left_foot_des=self.trajectory_optimization.left_foot_state.transform.translation(),
                                              left_foot_meas_transform=self.kindyn_meas_desc.kindyn.get_world_transform("l_sole"),
                                              left_foot_des_transform=self.kindyn_des_desc.kindyn.get_world_transform("l_sole"),
                                              left_wrench=self.left_wrench)

    # =======
    # GETTERS
    # =======

    def get_trajectory_duration(self) -> float:
        """Get the duration of the trajectory in seconds."""

        return len(self.postural_extractor.joint_references) * self.dt

    def get_dt(self) -> float:
        """Get the control loop rate."""

        return self.dt


