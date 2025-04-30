from setuptools import find_packages
from skbuild import setup

setup(
    name="sscCdi",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False)
