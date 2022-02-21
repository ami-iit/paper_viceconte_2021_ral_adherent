# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import matplotlib.pyplot as plt

# =============
# CONFIGURATION
# =============

# Path to the data for regenerating Fig.8 in the paper
data_path = "../datasets/inference/fig_8_blending_coefficients/blending_coefficients.txt"

# =============
# RETRIEVE DATA
# =============

# Load data
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_directory, data_path)
with open(data_path, 'r') as openfile:
    blending_coefficients = json.load(openfile)

# Retrieve blending coefficients
w1 = blending_coefficients["w_1"]
w2 = blending_coefficients["w_2"]
w3 = blending_coefficients["w_3"]
w4 = blending_coefficients["w_4"]

# ====
# PLOT
# ====

plt.figure()

# Convert inference calls to time (seconds)
timesteps = list(range(len(w1)))
time = [timestep/50.0 for timestep in timesteps]

plt.plot(time, blending_coefficients["w_1"], 'r', label="theta_1")
plt.plot(time, blending_coefficients["w_2"], 'b', label="theta_2")
plt.plot(time, blending_coefficients["w_3"], 'g', label="theta_3")
plt.plot(time, blending_coefficients["w_4"], 'y', label="theta_4")

# Plot configuration
plt.xlim([0, time[-1]])
plt.title("Blending coefficients profiles")
plt.ylabel("Blending coefficients")
plt.xlabel("Time [s]")
plt.legend()

# Plot
plt.show()
plt.pause(1)