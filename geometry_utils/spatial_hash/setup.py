from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="spatial_hash",
    ext_modules=[
        CUDAExtension(
            name="spatial_hash",
            sources=[
                "spatial_hash.cpp",
                "spatial_hash_cuda.cu",
            ],
            extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
