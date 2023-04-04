import trimesh
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2


def apply_transform(points, obj_openCV_pose):
    vector_ones = np.array(np.ones((len(points), 1)))
    query_4D = np.hstack((points, vector_ones))
    query_4D_transformed = np.dot(obj_openCV_pose, query_4D.T).T
    return query_4D_transformed[:, :3]


def opencv2opengl(cam_matrix_world):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if len(cam_matrix_world.shape) == 2:
        return np.matmul(transform, cam_matrix_world)
    else:
        transform = np.tile(transform, (cam_matrix_world.shape[0], 1, 1))
        return np.matmul(transform, cam_matrix_world)


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        results[i, 0] = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        results[i, 1] = np.int32(rep[1] / rep[2])
    return results


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, 0.0, -1.0])
    # print warning when camera location is parallel to tmp
    norm = min(
        np.linalg.norm(cam_location - tmp, axis=-1),
        np.linalg.norm(cam_location + tmp, axis=-1),
    )
    if norm < 1e-3:
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def spherical_to_cartesian(azimuth, elevation, radius):
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    return np.stack((x, y, z), axis=-1)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi
