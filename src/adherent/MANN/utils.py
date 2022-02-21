import os
import json
import wget
import os.path
import numpy as np

def create_path(path: str) -> None:
    """Create a path if it does not exist."""

    for subpath in path:
        if not os.path.exists(subpath):
            os.makedirs(subpath)

def normalize(X: np.array, axis: int) -> np.array:
    """Normalize X along the given axis."""

    # Compute mean and std
    Xmean = X.mean(axis=axis)
    Xstd = X.std(axis=axis)

    # Avoid division by zero
    for elem in range(Xstd.size):
        if (Xstd[elem] == 0):
            Xstd[elem] = 1

    # Normalize
    X = (X - Xmean) / Xstd

    return X

def denormalize(X: np.array, Xmean: np.array, Xstd: np.array) -> np.array:
    """Denormalize X, given its mean and std."""

    # Denormalize
    X = X * Xstd + Xmean

    return X

def store_in_file(data: list, filename: str) -> None:
    """Store data in file as json."""

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def read_from_file(filename: str) -> np.array:
    """Read data as json from file."""

    with open(filename, 'r') as openfile:
        data = json.load(openfile)

    return np.array(data)

def retrieve_original_MANN_files() -> None:
    """Retrieve original MANN files in order to apply patches to them."""

    # Original MANN files
    filenames = ["AdamW.py", "AdamWParameter.py", "ExpertWeights.py", "Gating.py", "MANN.py"]

    # Path configuration
    folder_url = "https://raw.githubusercontent.com/sebastianstarke/AI4Animation/3c3ee7df0e50463ced0f7a095e100b42084274a5/AI4Animation/SIGGRAPH_2018/TensorFlow/MANN/"
    MANN_folder = "../src/adherent/MANN/"

    # Retrieve files
    for filename in filenames:
        rel_filename = MANN_folder + filename
        url = folder_url + filename
        if os.path.isfile(rel_filename):
            os.remove(rel_filename)
        wget.download(url=url, out=rel_filename, bar=None)

def apply_patches_to_MANN_files() -> None:
    """Apply patches to the original MANN files."""

    # Original MANN files
    filenames = ["AdamW.py", "AdamWParameter.py", "ExpertWeights.py", "Gating.py", "MANN.py"]

    # Path configuration
    MANN_folder = "../src/adherent/MANN/"
    patches_MANN_folder = "../src/adherent/MANN/patches/"

    # Apply the patches
    for filename in filenames:
        original_file = MANN_folder + filename
        patch_file = patches_MANN_folder + filename[:-2] + "patch"
        os.system("patch " + original_file + " " + patch_file)

def remove_updated_MANN_files() -> None:
    """Remove the updated MANN files."""

    # Updated MANN files
    filenames = ["AdamW.py", "AdamWParameter.py", "ExpertWeights.py", "Gating.py", "MANN.py"]

    # Path configuration
    MANN_folder = "../src/adherent/MANN/"

    # Remove the updated MANN files
    for filename in filenames:
        updated_file = MANN_folder + filename
        if os.path.isfile(updated_file):
            os.remove(updated_file)

