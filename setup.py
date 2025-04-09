from setuptools import setup, find_packages

setup(
    name="proto_depth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "kaolin>=0.15.0",
        "matplotlib>=3.7.0",
    ],
)
