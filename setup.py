from setuptools import setup, find_packages

setup(
    name="bop_viz_kit",
    version="0.1",
    packages=find_packages(),
    install_requires=["opencv-python", "Pillow", "pandas", "open3d", "trimesh", "numpy", "pyrender"],
    author="Van Nguyen NGUYEN",
    author_email="vanngn.nguyen@gmail.com",
    description="A package for visualizing 6D pose",
    url="https://github.com/nv-nguyen/bop_viz_kit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)