# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import yarp
import argparse
from adherent.trajectory_generation import joystick_device

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()
parser.add_argument("--deactivate_bezier_plot",
                    help="Deactivate plot of the Bezier curve generated from the joystick inputs.", action="store_true")
args = parser.parse_args()
plot_bezier = not args.deactivate_bezier_plot

# ==================
# YARP CONFIGURATION
# ==================

# Inter-process communication is implemented via YARP ports, therefore "yarp server"
# needs to be executed in a separate terminal before launching this script

# Initialize YARP
yarp.Network.init()

# Create YARP port to pass joystick input to adherent
p_out = yarp.BufferedPortBottle()
p_out.open("/joystick_out")

# Initialize data sent through YARP
quad_bezier = []
base_velocities = []
facing_dirs = []

# ==================
# PLOT CONFIGURATION
# ==================

plt.ion()
# Rate for updating the plots
plot_rate = 7
# Counter to follow the plot rate
plot_i = 0

# ===============
# JOYSTICK DEVICE
# ===============

# Instantiate the joystick data processor
joystick = joystick_device.JoystickDataProcessor.build(device_path='/dev/input/js0')

# =========
# MAIN LOOP
# =========

while True:

    # At every cycle, read data from the joystick
    joystick_inputs = joystick.retrieve_motion_and_facing_directions()

    # Every plot_rate cycle, process and send data before plotting
    if plot_i % plot_rate == 0:

        # Process the joystick data
        quad_bezier, base_velocities, facing_dirs = joystick.process_joystick_inputs()

        # Send data to the trajectory generator through the YARP port
        joystick.send_data(output_port=p_out,
                           quad_bezier=quad_bezier,
                           base_velocities=base_velocities,
                           facing_dirs=facing_dirs,
                           joystick_inputs=joystick_inputs)

        # Plot the motion direction
        joystick.plot_motion_direction()

        # Plot the facing direction
        joystick.plot_facing_direction()

        # Plot the Bezier curve for the facing directions
        if plot_bezier:
            joystick.plot_facing_Bezier(quad_bezier=quad_bezier, facing_dirs=facing_dirs)

        # Plot
        plt.show()
        plt.pause(0.0001)

    # Update plot counter
    plot_i += 1
