import os
import numpy as np
from tqdm import tqdm


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_data(data_dir, save_dir, train_ratio, val_ratio, test_ratio):
    # Ensure the proportions add up to 1
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

    for class_folder in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(data_dir, class_folder)
        images = np.array([
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])
        np.random.shuffle(images)

        train_split = int(train_ratio * len(images))
        val_split = int(val_ratio * len(images))

        train_images = images[:train_split]
        val_images = images[train_split : train_split + val_split]
        test_images = images[train_split + val_split :]

        # Create symlinks instead of copying files
        for image in tqdm(
            train_images, desc=f"Linking train images for {class_folder}", leave=False
        ):
            dest_dir = os.path.join(train_dir, class_folder)
            create_dir_if_not_exists(dest_dir)
            os.symlink(os.path.join(class_path, image), os.path.join(dest_dir, image))

        for image in tqdm(
            val_images, desc=f"Linking val images for {class_folder}", leave=False
        ):
            dest_dir = os.path.join(val_dir, class_folder)
            create_dir_if_not_exists(dest_dir)
            os.symlink(os.path.join(class_path, image), os.path.join(dest_dir, image))

        for image in tqdm(
            test_images, desc=f"Linking test images for {class_folder}", leave=False
        ):
            dest_dir = os.path.join(test_dir, class_folder)
            create_dir_if_not_exists(dest_dir)
            os.symlink(os.path.join(class_path, image), os.path.join(dest_dir, image))


if __name__ == "__main__":
    data_dir = "/media/hdd3/neo/PL1_data_v2"
    save_dir = "/media/hdd3/Documents/neo/PL1_data_v2_split"

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    split_data(data_dir, save_dir, train_ratio, val_ratio, test_ratio)
