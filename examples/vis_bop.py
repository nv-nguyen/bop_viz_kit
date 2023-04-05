import bop_viz_kit as bop_viz
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
from PIL import Image

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
            mesh = bop_viz.load_mesh(cad_path)
            cad_data[cad_id] = mesh

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
        for idx, obj_gt in enumerate(frame_gt):
            bbox = bop_viz.get_bbox_from_mesh(cad_data[obj_gt["obj_id"]])
            obj_pose = bop_viz.combine_R_and_T(obj_gt["cam_R_m2c"], obj_gt["cam_t_m2c"])
            rgb_with_bbox = bop_viz.draw_bounding_box(
                rgb_with_bbox,
                obj_pose,
                bbox,
                intrinsic,
                color=colors[idx].tolist(),
                thickness=2,
            )
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_with_bbox)
        plt.axis("off")
        plt.title("Bounding Box")

        # visualize with coordinate frame
        rgb_coordinate_frame = np.copy(rgb)
        length = 0.3
        for obj_gt in frame_gt:
            obj_pose = bop_viz.combine_R_and_T(obj_gt["cam_R_m2c"], obj_gt["cam_t_m2c"])
            rgb_coordinate_frame = bop_viz.draw_pose_axis(
                rgb_coordinate_frame, obj_pose, length * 300, intrinsic, thickness=2
            )
        plt.subplot(2, 2, 2)
        plt.imshow(rgb_coordinate_frame)
        plt.axis("off")
        plt.title("Coordinate Frame")

        # visualize with point cloud projected
        rgb_pcd = np.copy(rgb)
        length = 0.3
        for idx, obj_gt in enumerate(frame_gt):
            obj_pose = bop_viz.combine_R_and_T(obj_gt["cam_R_m2c"], obj_gt["cam_t_m2c"])
            rgb_pcd = bop_viz.draw_point_cloud(
                rgb_pcd,
                cad_data[obj_gt["obj_id"]],
                intrinsic,
                obj_pose,
                color=colors[idx].tolist(),
                number_points=500,
            )
        plt.subplot(2, 2, 3)
        plt.imshow(rgb_pcd)
        plt.axis("off")
        plt.title("Projection of Point Cloud")

        # visualize with contour
        rgb_pose_contour = np.copy(rgb)
        for idx, obj_gt in enumerate(frame_gt):
            obj_pose = bop_viz.combine_R_and_T(obj_gt["cam_R_m2c"], obj_gt["cam_t_m2c"])
            _, rgb_pose_contour = bop_viz.draw_pose_contour(
                rgb_pose_contour,
                cad_data[obj_gt["obj_id"]],
                intrinsic,
                obj_pose,
                color=colors[idx].tolist(),
                thickness=5,
            )
        plt.subplot(2, 2, 4)
        plt.imshow(rgb_pose_contour)
        plt.axis("off")
        plt.title("Contour")

        plt.subplots_adjust(wspace=0.0, hspace=0.15)
        plt.savefig(
            f"examples/vis_BOP_{id_frame:06d}.png",
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=100,
        )
