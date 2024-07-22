import os
import numpy as np
from tqdm import tqdm


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_data(data_dir, save_dir, train_ratio, val_ratio, test_ratio):
    assert (
        round(train_ratio + val_ratio + test_ratio, 10) == 1
    ), "The sum of train, val, and test ratios must be 1."

    # Create train, val, test directories
    train_dir = os.path.join(save_dir, "train")
    val_dir = os.path.join(save_dir, "val")
    test_dir = os.path.join(save_dir, "test")

    create_dir_if_not_exists(train_dir)
    create_dir_if_not_exists(val_dir)
    create_dir_if_not_exists(test_dir)

    # Get class folders
    class_folders = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]

    for class_folder in tqdm(class_folders, desc="Processing Classes"):
        class_path = os.path.join(data_dir, class_folder)

        # Create directories for each class in train, val, and test
        train_class_dir = os.path.join(train_dir, class_folder)
        val_class_dir = os.path.join(val_dir, class_folder)
        test_class_dir = os.path.join(test_dir, class_folder)
        create_dir_if_not_exists(train_class_dir)
        create_dir_if_not_exists(val_class_dir)
        create_dir_if_not_exists(test_class_dir)

        # Process each image directly
        for image in tqdm(os.listdir(class_path), desc="Processing Images"):
            if image.endswith((".jpg", ".png", ".jpeg", ".tif")):
                rnd = np.random.random()
                if rnd < train_ratio:
                    target_dir = train_class_dir
                elif rnd < train_ratio + val_ratio:
                    target_dir = val_class_dir
                else:
                    target_dir = test_class_dir

                os.symlink(
                    os.path.join(class_path, image), os.path.join(target_dir, image)
                )


if __name__ == "__main__":
    data_dir = "/media/hdd3/neo/PL1_data_v2"
    save_dir = "/media/hdd3/Documents/neo/PL1_data_v2_split"

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    split_data(data_dir, save_dir, train_ratio, val_ratio, test_ratio)
