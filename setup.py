#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from skbuild import setup

setup(
    name="sscCdi",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
