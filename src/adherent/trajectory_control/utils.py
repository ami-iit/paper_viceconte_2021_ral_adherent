# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import yarp
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import bipedal_locomotion_framework.bindings as blf

@dataclass
class Integrator:
    """Auxiliary class implementing an Euler integrator from joint velocities to joint positions."""

    joints_position: np.array

    # Integration step
    dt: float

    @staticmethod
    def build(joints_initial_position: np.array, dt: float) -> "Integrator":
        """Build an instance of Integrator."""

        return Integrator(joints_position=joints_initial_position, dt=dt)

    def advance(self, joints_velocity: np.array) -> None:
        """Euler integration step."""

        self.joints_position += self.dt * joints_velocity

    def get_joints_position(self) -> np.ndarray:
        """Getter of the joint position."""

        return self.joints_position

def compute_zmp(kindyn: blf.floating_base_estimators.KinDynComputations,
                left_wrench: np.array, right_wrench: np.array) -> np.array:
    """Auxiliary function to retrieve the zero-moment point from the feet wrenches."""

    # Compute local zmps (one per foot) from the foot wrenches
    LF_r_zmp_L = [-left_wrench[4]/left_wrench[2], left_wrench[3]/left_wrench[2]]
    RF_r_zmp_R = [-right_wrench[4]/right_wrench[2], right_wrench[3]/right_wrench[2]]

    # Express the local zmps in homogeneous coordinates
    LF_r_zmp_L_homogenous = np.array([LF_r_zmp_L[0],LF_r_zmp_L[1],0,1])
    RF_r_zmp_R_homogenous = np.array([RF_r_zmp_R[0],RF_r_zmp_R[1],0,1])

    # Retrieve the global transform of the feet frames
    W_H_LF = kindyn.get_world_transform("l_sole")
    W_H_RF = kindyn.get_world_transform("r_sole")

    # Express the local zmps (one per foot) in a common reference frame (i.e. the world frame)
    W_r_zmp_L_hom = W_H_LF @ LF_r_zmp_L_homogenous
    W_r_zmp_L = W_r_zmp_L_hom[0:2]
    W_r_zmp_R_hom = W_H_RF @ RF_r_zmp_R_homogenous
    W_r_zmp_R = W_r_zmp_R_hom[0:2]

    # Compute the global zmp as a weighted mean of the local zmps (one per foot)
    # expressed in a common reference frame (i.e. the world frame)
    W_r_zmp_global = W_r_zmp_L * (left_wrench[2]/(left_wrench[2]+right_wrench[2])) + \
                     W_r_zmp_R * (right_wrench[2]/(left_wrench[2]+right_wrench[2]))

    return W_r_zmp_global

def synchronize(curr_dt: float, dt: float) -> float:
    """Auxiliary function for synchronization."""

    if curr_dt+dt - yarp.now() > 0:

        # Wait the proper amount of time to be synchronized at intervals of dt
        time.sleep(curr_dt+dt - yarp.now())

    else:

        # Debug to check whether the synchronization takes place or not
        print("no synch!")

    return curr_dt+dt

def rad2deg(rad: float) -> float:
    """Auxiliary function for radians to degrees conversion."""

    return rad / math.pi * 180

def world_gravity() -> List:
    """Auxiliary function for the gravitational constant."""

    return [0.0, 0.0, -blf.math.StandardAccelerationOfGravitation]

def define_foot_name_to_index_mapping(robot: str) -> Dict:
    """Define the robot-specific mapping between feet frame names and indexes."""

    if robot != "iCubV2_5":
        raise Exception("Mapping between feet frame names and indexes only defined for iCubV2_5.")

    foot_name_to_index = {"l_sole": 53, "r_sole": 147}

    return foot_name_to_index

def compute_initial_joint_reference(robot: str) -> List:
    """Retrieve the robot-specific initial reference for the joints."""

    if robot != "iCubV2_5":
        raise Exception("Initial joint reference only defined for iCubV2_5.")

    initial_joint_reference = [0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # left leg
                               0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # right leg
                               0.1388792845, 0.0, 0.0,  # torso
                               -0.0629, 0.4397, 0.1825, 0.5387, # left arm
                               -0.0629, 0.4397, 0.1825, 0.5387] # right arm

    return initial_joint_reference
