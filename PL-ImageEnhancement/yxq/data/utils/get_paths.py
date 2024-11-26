import os
from .misc import scandir


def paths_from_lmdb(folder):
    """ Get paths from lmdb folder.
        eg. 0001_SRGB_010_s001.png (512,512,3) 1
    """
    if not (folder.endswith(".lmdb")):
        raise ValueError(f"{folder} should both be in lmdb format")
    
    with open(os.path.join(folder, "meta_info.txt")) as fin:
        paths = [line.split(".")[0] for line in fin]

    return paths


def paths_from_folder(folder):
    """ Get paths from folders.
    """
    paths = list(scandir(folder, full_path=True))

    return paths

def paired_paths_from_lmdb(folders):
    """ Get paired paths from lmdb folders.
        [{'lq_path': '998', 'gt_path': '998'}, ...] 不同于 paired_paths_from_meta_info_file 和 from folder，这里只是998，并非完整路径
    """
    lq_folder, gt_folder = folders

    with open(os.path.join(lq_folder, "meta_info.txt")) as fin:
        lq_lmdb_keys = [line.split(".")[0] for line in fin]
    with open(os.path.join(gt_folder, "meta_info.txt")) as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]

    paths = [[lq_path, gt_path] for lq_path, gt_path in zip(lq_lmdb_keys, gt_lmdb_keys)]
    
    return paths

def paired_paths_from_folder(folders):
    """Generate paired paths from folders."""
    lq_folder, gt_folder = folders
    lq_paths = list(scandir(lq_folder, full_path=True))
    gt_paths = list(scandir(gt_folder, full_path=True))

    paths = [[lq_path, gt_path] for lq_path, gt_path in zip(lq_paths, gt_paths)]

    return paths

