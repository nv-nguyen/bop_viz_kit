import bop_viz_kit as bop_viz
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os

np.random.seed(0)

if __name__ == "__main__":
    data_dir = "./data/hb_val_primesense/000001"
    mesh_dir = "./data/hb_val_primesense/models"

    scene_gt = bop_viz.load_json(osp.join(data_dir, "scene_gt.json"))
    scene_gt_info = bop_viz.load_json(osp.join(data_dir, "scene_gt_info.json"))
    scene_camera = bop_viz.load_json(osp.join(data_dir, "scene_camera.json"))

    rgb_paths = sorted(os.listdir(osp.join(data_dir, "rgb")))
    