import os
import sys
import subprocess

try:
    from .spatial_hash import HierarchicalGrid
except ImportError:
    print("Spatial hash extension not found, attempting to build it...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_ext_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        subprocess.check_call(build_ext_cmd, cwd=current_dir)
        from .spatial_hash import HierarchicalGrid
    except Exception as e:
        print(f"Failed to build spatial hash extension: {e}")
        raise

# Only expose the new HierarchicalGrid class
__all__ = ["HierarchicalGrid"]
