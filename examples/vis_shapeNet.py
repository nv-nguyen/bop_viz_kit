import bop_viz_kit as bop_viz
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == "__main__":
    mesh_path = "./data/chair_shapeNet/models/model_normalized.obj"
    mesh = bop_viz.load_mesh(mesh_path)
    # rotate mesh by 90 degrees in X axis to be as BOP CAD
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh.apply_transform(transform)

    # 1. Sample a random pose from azimuth and elevation
    azimuths = np.random.uniform(0, 2 * np.pi)
    elevetions = np.random.uniform(0, np.pi / 2)
    radius = 1.0

    # convert to cartesian coordinates
    cam_location = bop_viz.spherical_to_cartesian(azimuths, elevetions, radius)
    cam_pose = bop_viz.look_at(cam_location, np.array([0, 0, 0]))
    obj_pose = np.linalg.inv(cam_pose)

    # 2. render the mesh with the pose
    width = 640
    height = 360
    focal = 50
    sensor_width = 120
    focal = focal / 1000.0
    pixel_size = sensor_width / (1000.0 * width)
    intrinsic = np.array(
        [
            [focal / pixel_size, 0, width / 2],
            [0, focal / pixel_size, height / 2],
            [0, 0, 1],
        ]
    )

    rgb, depth = bop_viz.render_offscreen(
        mesh, obj_pose, intrinsic, w=width, h=height, headless=False
    )
    rgb = rgb[:, :, :3]
    rgb = np.uint8(rgb)

    # visualize the coordinate frame
    length = 0.3
    img_with_coordinate_frame = bop_viz.draw_pose_axis(
        np.copy(rgb), obj_pose, length, intrinsic, thickness=2
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(img_with_coordinate_frame)
    plt.axis("off")
    plt.savefig(
        "examples/vis_shapeNet_coordinate_frame.png", bbox_inches="tight", pad_inches=0
    )

    # visualize the bounding box
    bbox = bop_viz.get_bbox_from_mesh(mesh)
    img_with_bbox = bop_viz.draw_bounding_box(
        np.copy(rgb),
        obj_pose,
        bbox,
        intrinsic,
        color=(255, 0, 0),
        thickness=2,
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(img_with_bbox)
    plt.axis("off")
    plt.savefig("examples/vis_shapeNet_bbox.png", bbox_inches="tight", pad_inches=0)

    # visualize with the contour of the depth map
    rendered_img, img_with_pose_contour = bop_viz.draw_pose_contour(
        np.copy(rgb), mesh, intrinsic, obj_pose, color=(255, 0, 0)
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(img_with_pose_contour)
    plt.axis("off")
    plt.savefig("examples/vis_shapeNet_contour.png", bbox_inches="tight", pad_inches=0)

    # visualize the projected point cloud
    img_with_pts = bop_viz.draw_point_cloud(
        np.copy(rgb), mesh, intrinsic, obj_pose, color=(255, 0, 0), number_points=500
    )
    plt.figure(figsize=(10, 15))
    plt.imshow(img_with_pts)
    plt.axis("off")
    plt.savefig("examples/vis_shapeNet_pcd.png", bbox_inches="tight", pad_inches=0)