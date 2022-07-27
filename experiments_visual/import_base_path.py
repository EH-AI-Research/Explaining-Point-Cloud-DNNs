import open3d as o3d # IMPORTANT, always import open3d first in the main script ("free(): invalid pointer" error)
import sys
from os import path, chdir

base_path = path.join(path.dirname(path.abspath(__file__)), '..')
sys.path.append(base_path)
chdir(base_path)