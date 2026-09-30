"""
Feature analysis and K-NN classification on the Animals10 dataset.

Command-line usage:
    python a1.py -c HSV|YCrCb <image-file-path>
    python a1.py -f SIFT <image-file-path>
    python a1.py -f HOG <image-file-path>
    python a1.py -r <main-image-folder>

Author: TONMOY DAY SARKAR.
"""

import os
import re
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# --------------------------------------------------------------------------
# Configuration used by the classification pipeline (Task Two).
# Feel free to tune these and discuss the effect of each in the report,
# as required by the "Notes on Marking" / report requirements section.
# --------------------------------------------------------------------------
IMG_RESIZE = (128, 128)     # all images are resized to this (w, h) before
                             # feature extraction so that every image
                             # produces a feature vector of the same length
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
KNN_K = 5
N_RUNS = 5
TRAIN_RATIO = 0.8


# ==========================================================================
# Task One.2 - HSV / YCrCb channel visualisation
# ==========================================================================
def visualise_channels(image_path, colorspace):
    """
    Display the original image plus its three colour channels, arranged
    from the NW corner clockwise: Original (NW) -> Channel1 (NE) ->
    Channel2 (SE) -> Channel3 (SW), as required by the spec.

    colorspace: 'HSV' or 'YCrCb'
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if colorspace == "HSV":
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ch_names = ["H", "S", "V"]
    elif colorspace == "YCrCb":
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ch_names = ["Y", "Cr", "Cb"]
    else:
        raise ValueError("colorspace must be 'HSV' or 'YCrCb'")

    c1, c2, c3 = cv2.split(converted)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.suptitle(f"{colorspace} channel breakdown - {os.path.basename(image_path)}",
                 fontsize=14)

    # NW: original
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original")
    # NE: channel 1
    axes[0, 1].imshow(c1, cmap="gray")
    axes[0, 1].set_title(ch_names[0])
    # SE: channel 2
    axes[1, 1].imshow(c2, cmap="gray")
    axes[1, 1].set_title(ch_names[1])
    # SW: channel 3
    axes[1, 0].imshow(c3, cmap="gray")
    axes[1, 0].set_title(ch_names[2])

    for ax in axes.ravel():
        ax.axis("off")

    # Extra vertical/horizontal spacing so subplot titles never collide
    # with the image in the row above (this was overlapping before).
    plt.subplots_adjust(top=0.90, hspace=0.25, wspace=0.05)
    plt.show()


# ==========================================================================
# Task One.3 - SIFT keypoint visualisation
# ==========================================================================
def visualise_sift(image_path):
    """
    Detect SIFT keypoints and draw, for each keypoint:
      - a '+' cross at the keypoint location
      - a circle whose radius is proportional to the keypoint scale
      - a line from the centre to the circle indicating orientation
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)

    canvas = bgr.copy()
    cross_half_len = 3  # pixels, half-length of the '+' cross

    for kp in keypoints:
        x, y = kp.pt
        x, y = int(round(x)), int(round(y))
        # radius proportional to the keypoint scale (kp.size is the
        # diameter of the meaningful keypoint neighbourhood)
        radius = max(1, int(round(kp.size / 2.0)))
        angle_deg = kp.angle if kp.angle >= 0 else 0.0
        angle_rad = np.deg2rad(angle_deg)

        # '+' cross at the keypoint centre
        cv2.line(canvas, (x - cross_half_len, y), (x + cross_half_len, y),
                  (0, 255, 0), 1)
        cv2.line(canvas, (x, y - cross_half_len), (x, y + cross_half_len),
                  (0, 255, 0), 1)

        # circle proportional to scale
        cv2.circle(canvas, (x, y), radius, (255, 0, 0), 1)

        # orientation line from centre to the edge of the circle
        end_x = int(round(x + radius * np.cos(angle_rad)))
        end_y = int(round(y + radius * np.sin(angle_rad)))
        cv2.line(canvas, (x, y), (end_x, end_y), (0, 0, 255), 1)

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Required layout: Original Image | Image with overlapped keypoints
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(canvas_rgb)
    axes[1].set_title(f"Image with overlapped keypoints "
                       f"({len(keypoints)} found)")
    axes[1].axis("off")

    fig.suptitle(f"SIFT keypoints - {os.path.basename(image_path)}",
                 fontsize=14)
    plt.subplots_adjust(top=0.88, wspace=0.05)
    plt.show()


# ==========================================================================
# Task One.4 - HOG visualisation
# ==========================================================================
def visualise_hog(image_path):
    """
    Compute and display the HOG representation of an image alongside the
    original, making gradient orientation/magnitude structure visible.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(hog_image_rescaled, cmap="gray")
    axes[1].set_title("HOG (gradient orientation/magnitude)")
    axes[1].axis("off")

    fig.suptitle(f"HOG visualisation - {os.path.basename(image_path)} "
                 f"(feature length = {features.shape[0]})")
    plt.tight_layout()
    plt.show()


# ==========================================================================
# Task Two - feature extraction, dataset loading, K-NN, evaluation
# ==========================================================================
def natural_sort_key(path):
    """
    Sort '1.jpg', '2.jpg', ..., '10.jpg' in numeric order instead of the
    lexicographic order Python's default sorted() would give
    (which would put '10.jpg' before '2.jpg'). Used only so listings and
    any printed diagnostics are easy to read; it has no effect on the
    train/test split itself, since that is shuffled randomly anyway.
    """
    fname = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok
            for tok in re.split(r"(\d+)", fname)]


def extract_hog_feature_vector(image_path):
    """
    Read an image, resize it to a fixed size, convert to grayscale, and
    return its HOG feature vector. A fixed image size is required so that
    every sample produces a feature vector of identical length.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    resized = cv2.resize(bgr, IMG_RESIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features


def load_dataset(main_folder):
    """
    Walk the main image folder and return:
      class_names : sorted list of category names (sub-folder names)
      images_by_class : dict {class_name: [list of image file paths]}
    """
    if not os.path.isdir(main_folder):
        raise FileNotFoundError(f"Folder not found: {main_folder}")

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    # Ignore hidden/tooling folders (e.g. .idea, .venv, __pycache__) that
    # may sit alongside the category folders in the project directory,
    # and ignore any sub-folder that contains no images at all - only
    # folders that actually hold images count as a category.
    candidate_dirs = sorted([
        d for d in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, d)) and not d.startswith(".")
    ])

    class_names = []
    images_by_class = {}
    for cname in candidate_dirs:
        cdir = os.path.join(main_folder, cname)
        files = sorted(
            [os.path.join(cdir, f) for f in os.listdir(cdir)
             if f.lower().endswith(valid_ext)],
            key=natural_sort_key
        )
        if files:
            class_names.append(cname)
            images_by_class[cname] = files

    if not class_names:
        raise ValueError(f"No category sub-folders with images found in "
                          f"{main_folder}")

    return class_names, images_by_class


def stratified_split(images_by_class, train_ratio, rng):
    """
    For each class, randomly shuffle its image list and split it into a
    training set and a testing set according to train_ratio (e.g. 0.8).
    Returns (train_paths, train_labels, test_paths, test_labels).
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for cname, files in images_by_class.items():
        files = files.copy()
        rng.shuffle(files)
        n_train = int(round(len(files) * train_ratio))
        train_files = files[:n_train]
        test_files = files[n_train:]

        train_paths.extend(train_files)
        train_labels.extend([cname] * len(train_files))
        test_paths.extend(test_files)
        test_labels.extend([cname] * len(test_files))

    return train_paths, train_labels, test_paths, test_labels


def build_feature_matrix(image_paths):
    """Extract the HOG feature vector for every image path given."""
    feats = [extract_hog_feature_vector(p) for p in image_paths]
    return np.vstack(feats)


def knn_predict(train_X, train_y, test_X, k):
    """
    A from-scratch K-Nearest-Neighbours classifier (Euclidean distance,
    majority vote with a distance-based tie-break) implemented with numpy,
    since scikit-learn is not a permitted package for this assignment.

    train_X : (n_train, n_features) array
    train_y : list/array of length n_train (class labels)
    test_X  : (n_test, n_features) array
    Returns : list of length n_test with the predicted label per sample.
    """
    train_y = np.array(train_y)
    predictions = []

    for i in range(test_X.shape[0]):
        sample = test_X[i]
        # Euclidean distance from this test sample to every training sample
        dists = np.sqrt(np.sum((train_X - sample) ** 2, axis=1))
        nearest_idx = np.argsort(dists)[:k]
        nearest_labels = train_y[nearest_idx]
        nearest_dists = dists[nearest_idx]

        # majority vote; ties broken by the smallest total distance
        labels, counts = np.unique(nearest_labels, return_counts=True)
        max_count = counts.max()
        candidates = labels[counts == max_count]
        if len(candidates) == 1:
            predictions.append(candidates[0])
        else:
            # tie-break: whichever candidate label has the smaller mean
            # distance among its neighbours in the k-nearest set
            best_label, best_mean_dist = None, np.inf
            for cand in candidates:
                mean_dist = nearest_dists[nearest_labels == cand].mean()
                if mean_dist < best_mean_dist:
                    best_label, best_mean_dist = cand, mean_dist
            predictions.append(best_label)

    return predictions


def normalise_features(train_X, test_X):
    """
    Standardise features using the training set's mean/std (z-score),
    which is common practice before distance-based classifiers like K-NN.
    """
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant dimensions
    return (train_X - mean) / std, (test_X - mean) / std


def run_classification(main_folder, k=KNN_K, n_runs=N_RUNS,
                        train_ratio=TRAIN_RATIO):
    """
    Run the full Task Two pipeline: for n_runs repetitions, create a new
    random 80/20 train/test split per class, extract HOG features, run
    K-NN classification, and record the accuracy of that run. Finally
    print the mean accuracy and standard deviation across all runs.
    """
    class_names, images_by_class = load_dataset(main_folder)
    print(f"Found {len(class_names)} classes: {class_names}")
    for c in class_names:
        print(f"  {c}: {len(images_by_class[c])} images")

    accuracies = []
    for run in range(1, n_runs + 1):
        rng = np.random.default_rng(seed=run)  # different, reproducible
                                                 # split for each run
        train_paths, train_labels, test_paths, test_labels = \
            stratified_split(images_by_class, train_ratio, rng)

        train_X = build_feature_matrix(train_paths)
        test_X = build_feature_matrix(test_paths)
        train_X, test_X = normalise_features(train_X, test_X)

        preds = knn_predict(train_X, train_labels, test_X, k)
        correct = sum(p == t for p, t in zip(preds, test_labels))
        accuracy = correct / len(test_labels)
        accuracies.append(accuracy)

        print(f"Run {run}: train={len(train_labels)}, "
              f"test={len(test_labels)}, accuracy={accuracy * 100:.2f}%")

    accuracies = np.array(accuracies)
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    print("-" * 50)
    print(f"K-NN (k={k}) on HOG features, {n_runs} runs, "
          f"{train_ratio * 100:.0f}/{(1 - train_ratio) * 100:.0f} split")
    print(f"Mean accuracy: {mean_acc * 100:.2f}%")
    print(f"Std deviation: {std_acc * 100:.2f}%")

    return mean_acc, std_acc, accuracies


# ==========================================================================
# Entry point / command-line dispatch
# ==========================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CSCI435/935 Assignment One - feature analysis and "
                    "K-NN classification on the Animals10 dataset.")
    parser.add_argument("-c", choices=["HSV", "YCrCb"],
                         help="Visualise HSV or YCrCb channels of the "
                              "given image.")
    parser.add_argument("-f", choices=["SIFT", "HOG"],
                         help="Visualise SIFT keypoints or HOG features "
                              "of the given image.")
    parser.add_argument("-r", metavar="MAIN_IMAGE_FOLDER",
                         help="Run K-NN classification on the dataset "
                              "found in this folder.")
    parser.add_argument("image", nargs="?",
                         help="Path to the image file (required with "
                              "-c or -f).")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    n_modes = sum(x is not None for x in (args.c, args.f, args.r))
    if n_modes != 1:
        parser.error("Specify exactly one of -c, -f, or -r.")

    if args.c is not None:
        if not args.image:
            parser.error("-c requires an <image-file-path> argument.")
        visualise_channels(args.image, args.c)

    elif args.f is not None:
        if not args.image:
            parser.error("-f requires an <image-file-path> argument.")
        if args.f == "SIFT":
            visualise_sift(args.image)
        else:  # HOG
            visualise_hog(args.image)

    elif args.r is not None:
        run_classification(args.r)


if __name__ == "__main__":
    main()
