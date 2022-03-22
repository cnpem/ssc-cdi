import os
import sys
from setuptools import setup

setup(name='sscPtycho',
      version='1.0.2',
      description='PSICC - Sirius Ptychography Reconstruction Module',
      author='Giovanni L. Baraldi',
      author_email='giovanni.baraldi@lnls.br',
      url='',
      packages=['sscPtycho'],
      package_data={'': ['libpsicc_ppc.so','libpsicc_x86_upc.so','libpsicc_x86_dgx.so']},
      include_package_data=True
     )
