import pyrender
import numpy as np
import os


def render_offscreen(mesh, obj_pose, intrinsic, w, h, headless=False):
    if headless:
        os.environ["DISPLAY"] = ":1"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    cam_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 0.0]),
        ambient_light=np.array([0.2, 0.2, 0.2, 1.0]),
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=4.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
    )
    # set camera pose from openGL to openCV pose
    scene.add(light, pose=cam_pose)
    scene.add(camera, pose=cam_pose)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh, pose=obj_pose)
    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    flags = pyrender.RenderFlags.OFFSCREEN
    color, depth = r.render(scene, flags=flags)
    # color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)  # RGBA to BGRA (for OpenCV)
    return color, depth
