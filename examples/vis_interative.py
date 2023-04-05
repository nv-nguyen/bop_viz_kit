import bop_viz_kit as bop_viz
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
from PIL import Image
import open3d as o3d

np.random.seed(0)

if __name__ == "__main__":
    data_dir = "./data/hb_val_primesense/000001"
    cad_dir = "./data/hb_val_primesense/models"

    scene_gt = bop_viz.load_json(osp.join(data_dir, "scene_gt.json"))
    scene_gt_info = bop_viz.load_json(osp.join(data_dir, "scene_gt_info.json"))
    scene_camera = bop_viz.load_json(osp.join(data_dir, "scene_camera.json"))

    # load cad
    cad_names = sorted([x for x in os.listdir(cad_dir) if x.endswith(".ply")])
    models_info = bop_viz.load_json(osp.join(cad_dir, "models_info.json"))
    cad_data = {}
    for cad_name in cad_names:
        cad_id = int(cad_name.split(".")[0].replace("obj_", ""))
        cad_path = osp.join(cad_dir, cad_name)
        if os.path.exists(cad_path):
            cad_data[cad_id] = cad_path

    colors = np.random.uniform(0, 254, size=(len(scene_gt), 3)).astype(np.uint8)
    intrinsic = np.array(scene_camera["0"]["cam_K"]).reshape(3, 3)
    # visualize
    rgb_paths = sorted(os.listdir(osp.join(data_dir, "rgb")))
    for path in rgb_paths:
        rgb = Image.open(osp.join(data_dir, "rgb", path))
        rgb = np.array(rgb)

        id_frame = int(path.split(".")[0])
        # get the pose of the object
        frame_gt = scene_gt[f"{id_frame}"]

        # visualize the bounding box
        plt.figure(figsize=(14, 10))
        rgb_with_bbox = np.copy(rgb)

        geometry_to_vis = []
        for idx, obj_gt in enumerate(frame_gt):
            # scale to meter for easy interactive visualization
            mesh_o3d = o3d.io.read_triangle_mesh(cad_data[obj_gt["obj_id"]]).scale(
                0.001, center=(0, 0, 0)
            )
            obj_pose = bop_viz.combine_R_and_T(obj_gt["cam_R_m2c"], obj_gt["cam_t_m2c"])
            obj_pose[:3, 3] = obj_pose[:3, 3] * 0.001
            mesh_o3d.transform(obj_pose)
            geometry_to_vis.append({"name": "mesh", "data": mesh_o3d})
        # adding a camera 
        camera_pose = np.eye(4)
        geometry_to_vis.append({"name": "camera", "data": camera_pose, "color": [0, 0, 0]})
        bop_viz.visualizer(geometry_to_vis)
