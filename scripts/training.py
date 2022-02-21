from adherent.MANN import utils

# Retrieve original MANN code and apply patches to it
utils.retrieve_original_MANN_files()
utils.apply_patches_to_MANN_files()

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import argparse
import numpy as np
from adherent.MANN import MANN

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()
parser.add_argument("--deactivate_mirroring", help="Discard features from mirrored mocap data.", action="store_true")
args = parser.parse_args()
mirroring = not args.deactivate_mirroring

# =============
# CONFIGURATION
# =============

script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the datasets for training (both D2 and D3) and the portions of each dataset
datasets = ["D2", "D3"]
D2_portions = {1: "1_forward_normal_step",
               2: "2_backward_normal_step",
               3: "3_left_and_right_normal_step",
               4: "4_diagonal_normal_step",
               5: "5_mixed_normal_step"}
D3_portions = {6: "6_forward_small_step",
               7: "7_backward_small_step",
               8: "8_left_and_right_small_step",
               9: "9_diagonal_small_step",
               10: "10_mixed_small_step",
               11: "11_mixed_normal_and_small_step"}

# Initialize input filenames list
X = []

# Fill input filenames list
for dataset in datasets:

    if dataset == "D2":
        inputs = D2_portions
    elif dataset == "D3":
        inputs = D3_portions

    for index in inputs.keys():
        input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_" + dataset + "/" + inputs[index] + "_X.txt"
        X.append(input_path)

        if mirroring:
            input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_mirrored_" + dataset + "/" + inputs[index] + "_X_MIRRORED.txt"
            X.append(input_path)

# Debug
print("\nInput files:")
for elem in X:
    print(elem)

# Initialize output filenames list
Y = []

# Fill output filenames list
for dataset in datasets:

    if dataset == "D2":
        outputs = D2_portions
    elif dataset == "D3":
        outputs = D3_portions

    for index in outputs.keys():
        output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_" + dataset + "/" + outputs[index] + "_Y.txt"
        Y.append(output_path)

        if mirroring:
            output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_mirrored_" + dataset + "/" + outputs[index] + "_Y_MIRRORED.txt"
            Y.append(output_path)

# Debug
print("\nOutput files:")
for elem in Y:
    print(elem)

# Set storage folder
if not mirroring:
    savepath = '../datasets/training_subsampled'
else:
    savepath = '../datasets/training_subsampled_mirrored'
for dataset in datasets:
    savepath+="_"+dataset

# Debug
print("\nSavepath:", savepath, "\n")

# Define training hyperparameters
num_experts = 4
rng = np.random.RandomState(23456)
sess = tf.Session()

# ========
# TRAINING
# ========

# Debug
input("\nPress Enter to start the training")

# Initialize MANN object
mann = MANN.MANN(rng,
                 sess,
                 X, Y,
                 savepath,
                 num_experts,
                 training_set_percentage=98,
                 hidden_size=512,
                 hidden_size_gt=32,
                 batch_size = 32, epoch = 150, Te = 10, Tmult = 2,
                 learning_rate_ini = 0.0001, weightDecay_ini = 0.0025, keep_prob_ini = 0.7)

# Build the network
mann.build_model()

# Train the network
mann.train()

# Remove updated MANN code
utils.remove_updated_MANN_files()
