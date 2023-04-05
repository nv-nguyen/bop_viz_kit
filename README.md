# BOP visualization kit

<p align="center">
  <img src=./media/BOP.gif width="80%"/>
</p>

A simple Python package to visualize 6D pose for BOP benchmark

To install `bop_viz_kit`, run:

```bash
git clone https://github.com/nv-nguyen/bop_viz_kit.git
pip install -e bop_viz_kit
```

## Usage
Note that object's poses should be defined in OpenCV coordinate frame (same as BOP benchmark)
1. Visualize 3D bounding box
```python
import bop_viz_kit as bop_viz
mesh = bop_viz.load_mesh(mesh_path)
bbox = bop_viz.get_bbox_from_mesh(mesh)# computing 3D box from mesh
img = bop_viz.draw_bounding_box(
    img, obj_pose, bbox, intrinsic, color, thickness=2,
)
```
2. Visualize coordinate frame
```python
import bop_viz_kit as bop_viz
length_coordinate_frame = 0.3 # in same scale as mesh
img = bop_viz.draw_pose_axis(
    img, obj_pose, length_coordinate_frame, intrinsic, thickness=2
)
```
3. Visualize projection of point cloud
```python
import bop_viz_kit as bop_viz
mesh = bop_viz.load_mesh(mesh_path)
img = bop_viz.draw_point_cloud(
    img, mesh, intrinsic, obj_pose, color, number_points=500,
)
```
4. Visualize contour (by rendering cad through pyrender)
```python
import bop_viz_kit as bop_viz
mesh = bop_viz.load_mesh(mesh_path)
depth, img = bop_viz.draw_pose_contour(
    img, mesh intrinsic, obj_pose, color, thickness=5, headless=False
) # set headless=True when running on headless system
```
5. Visualize interactively, please checkout the demo [vis_interactive.py](https://github.com/nv-nguyen/bop_viz_kit/blob/main/examples/vis_interactive.py)

<p align="center">
  <img src=./media/BOP_interactive.gif width="80%"/>
</p>

## TODO 

- Blended visualization with alpha channel
- Omni3D's bounding box visualization (BOP+ScanNet)
- Simple script to render CAD with BlenderProc
- Script to generate template poses as done in [template-pose](https://github.com/nv-nguyen/template-pose)
- Script to visualize camera and object interatively