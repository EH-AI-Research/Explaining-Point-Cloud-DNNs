from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import pointnet2

_ext_src_root = "pointnet2/_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["etw_pytorch_utils==1.1.1", "h5py", "pprint", "enum34", "future"]

setup(
    name="pointnet2",
    version=pointnet2.__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# git+git://github.com/erikwijmans/etw_pytorch_utils.git@v1.1.1#egg=etw_pytorch_utils
# h5py
# numpy
# torch>=1.0
# torchvision
# pprint
# enum34
# future

# tqdm, plyfile, numpy, opencv-python, matplotlib
# open3d=0.9.0.0
# pip install https://github.com/intel-isl/Open3D/releases/download/v0.9.0/open3d-0.9.0.0-cp37-cp37m-win_amd64.whl
# pip install https://github.com/intel-isl/Open3D/releases/download/v0.9.0/open3d-0.9.0.0-cp37-cp37m-macosx_10_7_x86_64.whl
# pip install https://github.com/intel-isl/Open3D/releases/download/v0.9.0/open3d-0.9.0.0-cp37-cp37m-manylinux1_x86_64.whl