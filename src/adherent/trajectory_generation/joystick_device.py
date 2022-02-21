# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from https://gist.github.com/rdb/8864666
# which includes the code released by rdb under the Unlicense (unlicense.org)

import yarp
import math
import array
import struct
import numpy as np
from fcntl import ioctl
from typing import List, Dict, BinaryIO
from dataclasses import dataclass, field
from adherent.trajectory_generation.utils import quadratic_bezier
from adherent.trajectory_generation.utils import compute_angle_wrt_x_positive_semiaxis

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt


@dataclass
class JoystickDevice:
    """Class for the joystick device."""

    jsdev: BinaryIO = None

    axis_map: List = field(default_factory=list)
    axis_states: Dict = field(default_factory=dict)

    button_map: List = field(default_factory=list)
    button_states: Dict = field(default_factory=dict)

    @staticmethod
    def build() -> "JoystickDevice":
        """Build an empty JoystickDevice."""

        return JoystickDevice()

    def open_device(self, device_path: str) -> None:
        """Open a joystick device."""

        axis_names = {
            0x00: 'x',
            0x01: 'y',
            0x02: 'z',
            0x03: 'rx',
            0x04: 'ry',
            0x05: 'rz',
            0x06: 'trottle',
            0x07: 'rudder',
            0x08: 'wheel',
            0x09: 'gas',
            0x0a: 'brake',
            0x10: 'hat0x',
            0x11: 'hat0y',
            0x12: 'hat1x',
            0x13: 'hat1y',
            0x14: 'hat2x',
            0x15: 'hat2y',
            0x16: 'hat3x',
            0x17: 'hat3y',
            0x18: 'pressure',
            0x19: 'distance',
            0x1a: 'tilt_x',
            0x1b: 'tilt_y',
            0x1c: 'tool_width',
            0x20: 'volume',
            0x28: 'misc',
        }
        button_names = {
            0x120: 'trigger',
            0x121: 'thumb',
            0x122: 'thumb2',
            0x123: 'top',
            0x124: 'top2',
            0x125: 'pinkie',
            0x126: 'base',
            0x127: 'base2',
            0x128: 'base3',
            0x129: 'base4',
            0x12a: 'base5',
            0x12b: 'base6',
            0x12f: 'dead',
            0x130: 'a',
            0x131: 'b',
            0x132: 'c',
            0x133: 'x',
            0x134: 'y',
            0x135: 'z',
            0x136: 'tl',
            0x137: 'tr',
            0x138: 'tl2',
            0x139: 'tr2',
            0x13a: 'select',
            0x13b: 'start',
            0x13c: 'mode',
            0x13d: 'thumbl',
            0x13e: 'thumbr',
            0x220: 'dpad_up',
            0x221: 'dpad_down',
            0x222: 'dpad_left',
            0x223: 'dpad_right',
            0x2c0: 'dpad_left',
            0x2c1: 'dpad_right',
            0x2c2: 'dpad_up',
            0x2c3: 'dpad_down',
        }

        # Open the joystick device
        print('Opening %s' % device_path)
        self.jsdev = open(device_path, 'rb')

        # Get the device name
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)
        js_name = buf.tobytes().rstrip(b'\x00').decode('utf-8')

        # Get number of axes
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf)
        num_axes = buf[0]

        # Get number of buttons
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf)
        num_buttons = buf[0]

        # Get the axis map and state
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf)
        for axis in buf[:num_axes]:
            axis_name = axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map and state
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf)
        for btn in buf[:num_buttons]:
            btn_name = button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0

        # Debug
        print('Device name: %s' % js_name)
        print('%d axes found: %s' % (num_axes, ', '.join(self.axis_map)))
        print('%d buttons found: %s' % (num_buttons, ', '.join(self.button_map)))

    def update_axis_state(self) -> None:
        """Update the axis state when more recent data from the joystick analogs are available. In particular,
        store the (x,y) axis values from the left analog and the (z,rz) axis values from the right analog."""

        # Wait for an event to read
        evbuf = self.jsdev.read(8)

        if evbuf:

            # Unpack the message
            time, value, kind, number = struct.unpack('IhBB', evbuf)

            # Filter on the kind of message: axis data
            if not kind & 0x02:
                return

            # Retrieve the axis
            axis = self.axis_map[number]

            # Filter on the axes of interest: (x,y) for the motion direction, (z,rz) for the facing direction
            if axis not in ["x","y","z","rz"]:
                return

            # Update the axis state
            fvalue = value / 32767.0
            self.axis_states[axis] = fvalue


@dataclass
class JoystickDataProcessor:
    """Class for processing data from the joystick device."""

    device: JoystickDevice

    # Predefined norm for the base velocities
    base_vel_norm: float

    # Axis of the composed ellipsoid constraining the last point of the Bezier curve of base positions
    ellipsoid_forward_axis: float
    ellipsoid_side_axis: float
    ellipsoid_backward_axis: float

    # Scaling factor for all the axes of the composed ellipsoid
    ellipsoid_scaling: float

    # Offset for the control point of the Bezier curve
    control_point_offset: float

    # Maximum local variation angles for the facing directions, different according to the type of walking
    max_facing_direction_angle_forward: float # forward walking
    max_facing_direction_angle_backward: float # backward walking
    max_facing_direction_angle_side_opposite_sign: float # side walking (motion and facing directions with opposite sign)
    max_facing_direction_angle_side_same_sign: float # side walking (motion and facing directions with same sign)

    # Number of points constituting the Bezier curve
    t: List = np.linspace(0, 1, 7)

    # Motion direction: left analog data <-> (x,y)
    curr_x: float = 0
    curr_y: float = -1

    # Facing direction: right analog data <-> (z,rz)
    curr_z: float = 0
    curr_rz: float = -1

    @staticmethod
    def build(device_path: str,
              base_vel_norm: float = 0.4,
              ellipsoid_forward_axis: float = 1.0,
              ellipsoid_side_axis: float = 0.9,
              ellipsoid_backward_axis: float = 0.6,
              ellipsoid_scaling: float = 0.4,
              max_facing_direction_angle_forward: float = math.pi/9,
              max_facing_direction_angle_backward: float = math.pi/30,
              max_facing_direction_angle_side_opposite_sign: float = math.pi/12,
              max_facing_direction_angle_side_same_sign: float = math.pi/18) -> "JoystickDataProcessor":
        """Build a JoystickDataProcessor."""

        device = JoystickDevice.build()
        device.open_device(device_path=device_path)

        control_point_offset = 0.1 * ellipsoid_scaling

        return JoystickDataProcessor(device=device,
                                     base_vel_norm=base_vel_norm,
                                     ellipsoid_forward_axis=ellipsoid_forward_axis,
                                     ellipsoid_side_axis=ellipsoid_side_axis,
                                     ellipsoid_backward_axis=ellipsoid_backward_axis,
                                     ellipsoid_scaling=ellipsoid_scaling,
                                     control_point_offset=control_point_offset,
                                     max_facing_direction_angle_forward=max_facing_direction_angle_forward,
                                     max_facing_direction_angle_backward=max_facing_direction_angle_backward,
                                     max_facing_direction_angle_side_opposite_sign=max_facing_direction_angle_side_opposite_sign,
                                     max_facing_direction_angle_side_same_sign=max_facing_direction_angle_side_same_sign)

    def retrieve_motion_and_facing_directions(self) -> List:
        """Retrieve the current motion and facing directions from the axis states."""

        # Update axis state
        self.device.update_axis_state()

        # Retrieve updated motion and facing directions from the axis states
        self.update_motion_direction()
        self.update_facing_direction()

        # Return joystick inputs
        joystick_inputs = [self.curr_x, self.curr_y, self.curr_z, self.curr_rz]

        return joystick_inputs

    def update_motion_direction(self) -> None:
        """Update the motion direction, given the (x,y) states from the left analog."""

        # Return if the updated values coincide with the previous ones
        if self.device.axis_states["x"] == self.curr_x and self.device.axis_states["y"] == self.curr_y:
            return

        # Update motion direction
        self.curr_x = self.device.axis_states["x"]
        self.curr_y = self.device.axis_states["y"]

        # Normalize motion direction
        norm = np.linalg.norm([self.curr_x, self.curr_y])
        if norm != 0:
            self.curr_x = self.curr_x/norm
            self.curr_y = self.curr_y/norm

    def update_facing_direction(self) -> None:
        """Update the facing direction, given the (z,rz) states from the right analog and the current motion direction.
        Several constraints on the current facing direction apply according to the current motion direction.
        """

        # If the left analog is not used (i.e. the robot is stopped), set frontal facing direction (i.e. prevent the robot to turn on the spot)
        if self.curr_x == 0 and self.curr_y == 0:

            self.curr_z = 0
            self.curr_rz = -1

        else:

            # Return if the updated values coincide with the previous ones
            if self.device.axis_states["z"] == self.curr_z and self.device.axis_states["rz"] == self.curr_rz:
                return

            # If the left analog is used (i.e. the robot is moving) but the right analog is not, set frontal facing direction
            if self.device.axis_states["z"] == 0 and self.device.axis_states["rz"] == 0:
                self.curr_z = 0
                self.curr_rz = -1

            else:

                # Compute the maximum local variation angle for the facing direction
                max_facing_direction_angle = self.compute_max_facing_direction_angle()

                # Update facing direction taking into account the maximum variation angle constraint
                theta = math.pi / 2 - max_facing_direction_angle
                self.curr_rz = np.minimum(-math.sin(theta), self.device.axis_states["rz"])
                if self.curr_rz < -math.sin(theta):
                    self.curr_z = self.device.axis_states["z"]
                else:
                    self.curr_z = np.sign(self.device.axis_states["z"]) * math.cos(theta)

                # Normalize facing direction
                norm = np.linalg.norm([self.curr_z, self.curr_rz])
                if norm != 0:
                    self.curr_z = self.curr_z/norm
                    self.curr_rz = self.curr_rz/norm

    def compute_max_facing_direction_angle(self) -> float:
        """Compute the maximum local variation angle for the facing direction, given the current motion direction.
          This angle is different for forward, backward and sideways walking. In the case of sideways walking, a
          different maximum angle is defined for coincident or opposite signs of motion and facing direction."""

        # Backward walking
        if self.curr_y > 0.2:
            max_facing_direction_angle = self.max_facing_direction_angle_backward

        # Side walking
        elif self.curr_y < 0.2 and np.linalg.norm(self.curr_x) > 0.2:

            if np.sign(self.curr_x) != np.sign(self.device.axis_states["z"]):
                # Motion and facing directions with opposite signs
                max_facing_direction_angle = self.max_facing_direction_angle_side_opposite_sign

            else:
                # Motion and facing directions with same sign
                max_facing_direction_angle = self.max_facing_direction_angle_side_same_sign

        else:
            # Forward walking
            max_facing_direction_angle = self.max_facing_direction_angle_forward

        return max_facing_direction_angle

    def process_joystick_inputs(self) -> (list, list, list):
        """Process the joystick inputs in order to retrieve the desired future ground trajectory specified by:
           - a quadratic Bezier curve of future base positions (quad_bezier)
           - a series of desired base velocities associated to the base positions in the Bezier curve (base_velocities)
           - a series of desired facing directions associated to the base positions in the Bezier curve (facing_dirs)
        """

        quad_bezier = self.compute_quadratic_bezier()
        base_velocities = self.compute_base_velocities(quad_bezier)
        facing_dirs = self.compute_facing_directions(quad_bezier)

        return quad_bezier, base_velocities, facing_dirs

    def compute_quadratic_bezier(self) -> List:
        """Compute a quadratic Bezier curve of future base positions from the joystick inputs. The last point of such
        a Bezier curve is constrained to lie on the edge of an ellipsoid composed by an upper and a lower semi-ellipse
        with axes of different length.
        """

        if self.curr_y <= 0:

            # Relevant Bezier points for the upper semi-ellipse (long axis = ellipsoid_forward_axis, short axis = ellipsoid_side_axis)
            initial_point = np.array([0, 0])
            unscaled_control_point_y = np.minimum(self.ellipsoid_forward_axis, - self.curr_y + self.control_point_offset)
            control_point = np.array([0, self.ellipsoid_scaling * unscaled_control_point_y])
            final_point = np.array([self.ellipsoid_side_axis * self.ellipsoid_scaling * self.curr_x,
                                    - self.ellipsoid_forward_axis * self.ellipsoid_scaling * self.curr_y])

        else:

            # Relevant Bezier points for the lower semi-ellipse (long axis = ellipsoid_backward_axis, short axis = ellipsoid_side_axis)
            initial_point = np.array([0, 0])
            unscaled_control_point_y = np.maximum( - self.ellipsoid_backward_axis, - self.curr_y - self.control_point_offset)
            control_point = np.array([0, self.ellipsoid_scaling * unscaled_control_point_y])
            final_point = np.array([self.ellipsoid_side_axis * self.ellipsoid_scaling * self.curr_x,
                                    - self.ellipsoid_backward_axis * self.ellipsoid_scaling * self.curr_y])

        # Compute the Bezier curve of base positions
        quad_bezier = quadratic_bezier(initial_point, control_point, final_point, self.t)

        return quad_bezier

    def compute_base_velocities(self, quad_bezier: List) -> List:
        """Compute a series of base velocities associated to the base positions in the Bezier curve by differentiation.
        If they are not null, such base velocities have a predefined norm.
        """

        base_velocities = []

        # Compute base velocities by differentiating Bezier base positions
        for i in range(1,len(quad_bezier)):

            base_pos_prev = quad_bezier[i-1]
            base_pos = quad_bezier[i]
            difference_norm = np.linalg.norm(base_pos - base_pos_prev)

            if difference_norm != 0:
                base_vel = (base_pos - base_pos_prev)/difference_norm
                base_vel *= self.base_vel_norm
            else:
                base_vel = [0,0]

            base_velocities.append(base_vel)

        # The last base velocity is approximated by replicating the previous one
        base_velocities.append(base_velocities[-1])

        return base_velocities

    def compute_facing_directions(self, quad_bezier: List) -> List:
        """Compute a series of facing directions associated to the base positions in the Bezier curve. The series
        progressively drives the current facing direction to the user-specified desired one.
        """

        facing_dirs = []

        # Angle from the desired facing direction to the x positive semiaxis
        final_theta = compute_angle_wrt_x_positive_semiaxis([self.curr_z, -self.curr_rz])

        # Angle from the current facing direction (always frontal in the local view) to the x positive semiaxis
        curr_theta = math.pi/2

        # Progressive angle update from the current facing direction to the desired one
        theta_update = (final_theta - curr_theta)/(len(quad_bezier) - 1)

        # Compute series of facing directions progressively going from the current one to the desired one
        for i in range(len(quad_bezier)):

            # Current facing direction
            curr_facing_dir = [math.cos(curr_theta), math.sin(curr_theta)]

            # Normalization
            norm = np.linalg.norm(curr_facing_dir)
            if norm != 0:
                curr_facing_dir = curr_facing_dir/norm

            # Update
            facing_dirs.append(curr_facing_dir)
            curr_theta += theta_update

        return facing_dirs

    def send_data(self,
                  output_port: yarp.BufferedPortBottle,
                  quad_bezier: List,
                  base_velocities: List,
                  facing_dirs: List,
                  joystick_inputs: List) -> None:
        """Send raw and processed joystick data through YARP."""

        # The joystick input from the user written on the YARP port will contain 3 * 7 * 2 + 4 = 46 values:
        # 0-13 are quad_bezier (x,y)
        # 14-27 are base_velocities (x,y)
        # 28-41 are facing_dirs (x,y)
        # 42-45 are joystick inputs to be stored for future plotting (curr_x, curr_y, curr_z, curr_rz)

        # Add data to be sent through the YARP port
        bottle = output_port.prepare()
        bottle.clear()
        for data in [quad_bezier, base_velocities, facing_dirs]:
            for elem in data:
                for coord in elem:
                    bottle.addFloat32(coord)
        for joystick_input in joystick_inputs:
            bottle.addFloat32(joystick_input)

        # Send data through the YARP port
        output_port.write()

    def plot_motion_direction(self) -> None:
        """Visualize the current motion direction."""

        plt.figure(1)
        plt.clf()

        # Circumference of unitary radius
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'r')
        plt.plot(x, -y, 'r')

        # Motion direction
        plt.scatter(0, 0, c='r')
        desired_motion_direction = plt.arrow(0, 0, self.curr_x, -self.curr_y, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='r')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_motion_direction], ['DESIRED MOTION DIRECTION'], loc="lower center")

    def plot_facing_direction(self) -> None:
        """Visualize current facing direction."""

        plt.figure(2)
        plt.clf()

        # Circumference of unitary radius
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'b')
        plt.plot(x, -y, 'b')

        # Facing direction
        plt.scatter(0, 0, c='b')
        desired_facing_direction = plt.arrow(0, 0, self.curr_z, -self.curr_rz, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='b')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_facing_direction], ['DESIRED FACING DIRECTION'], loc="lower center")

    def plot_facing_Bezier(self, quad_bezier: list, facing_dirs: list) -> None:
        """Visualize the Bezier curve of base positions obtained from the joystick inputs along with the
        associated facing directions.
        """

        plt.figure(3)
        plt.clf()

        # Upper semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_forward_axis * self.ellipsoid_scaling
        x = np.linspace(-a, a, 1000)
        y = b * np.sqrt( 1 - (x ** 2)/(a ** 2))
        plt.plot(x, y, 'k')

        # Lower semi-ellipse of the composed ellipsoid on which the last point of the Bezier curve is constrained
        a = self.ellipsoid_side_axis * self.ellipsoid_scaling
        b = self.ellipsoid_backward_axis * self.ellipsoid_scaling
        x = np.linspace(-a, a, 1000)
        y = b * np.sqrt( 1 - (x ** 2)/(a ** 2))
        plt.plot(x, -y, 'k')

        # Plot base positions
        quad_bezier_x = [elem.tolist()[0] for elem in quad_bezier]
        quad_bezier_y = [elem.tolist()[1] for elem in quad_bezier]
        plt.scatter(quad_bezier_x, quad_bezier_y, c='r')

        # Plot facing directions (scaled for visualization purposes)
        for i in range(len(quad_bezier)):
            plt.plot([quad_bezier_x[i], quad_bezier_x[i] + facing_dirs[i][0]/10],
                     [quad_bezier_y[i], quad_bezier_y[i] + facing_dirs[i][1]/10],
                     c='b')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.3, 0.5])
        plt.axis('off')
