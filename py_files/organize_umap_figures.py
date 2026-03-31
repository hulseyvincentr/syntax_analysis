# -*- coding: utf-8 -*-

from pathlib import Path
import shutil


def organize_cluster_pngs(root_folder):
    """
    Organize PNG files inside each animal's 'clusters' folder into:
        clusters/raw_groups/
        clusters/equal_groups/

    Parameters
    ----------
    root_folder : str or Path
        Root directory containing animal ID folders.
    """
    root_folder = Path(root_folder)

    if not root_folder.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    moved_raw = 0
    moved_equal = 0
    skipped = []

    # Loop through animal ID folders
    for animal_folder in root_folder.iterdir():
        if not animal_folder.is_dir():
            continue

        clusters_folder = animal_folder / "clusters"
        if not clusters_folder.exists() or not clusters_folder.is_dir():
            skipped.append(f"No clusters folder in {animal_folder.name}")
            continue

        raw_folder = clusters_folder / "raw_groups"
        equal_folder = clusters_folder / "equal_groups"

        raw_folder.mkdir(exist_ok=True)
        equal_folder.mkdir(exist_ok=True)

        # Only look at PNGs directly inside clusters_folder
        for png_file in clusters_folder.glob("*.png"):
            name = png_file.name.lower()

            if "raw_groups" in name:
                destination = raw_folder / png_file.name
                shutil.move(str(png_file), str(destination))
                moved_raw += 1

            elif "equal_groups" in name:
                destination = equal_folder / png_file.name
                shutil.move(str(png_file), str(destination))
                moved_equal += 1

            else:
                skipped.append(f"Unrecognized file pattern: {png_file}")

    print("Done organizing PNG files.")
    print(f"Moved raw_groups files: {moved_raw}")
    print(f"Moved equal_groups files: {moved_equal}")

    if skipped:
        print("\nSkipped items:")
        for item in skipped:
            print(f" - {item}")


# Example usage:
root_folder = "/Volumes/my_own_SSD/updated_AreaX_outputs/Lateral_only"
organize_cluster_pngs(root_folder)