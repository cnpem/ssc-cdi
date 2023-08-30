#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import warnings
import glob
from setuptools import setup, Extension, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy
import sys
import shutil

##########################################################

# Set Python package requirements for installation.   

install_requires = [
    'sysv-ipc>=0.0.0',
    'cmake_setup>=0.0.0',
    'numpy',
    'matplotlib',
    'h5py',
    'scipy',
    'scikit-image',
    'tqdm',
    'SharedArray',
]

compile_cuda = 0
compile_sudo = 0

if '--cuda' in sys.argv:
    compile_cuda = 1
    sys.argv.remove('--cuda')

if '--sudo' in sys.argv:
    compile_sudo = 1
    sys.argv.remove('--sudo')
    
########

def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print ('The nvcc binary could not be '
                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))
   
    check = pjoin(home, 'lib')

    if not os.path.exists(check):
       cudaconfig = {'home':home, 'nvcc':nvcc,
        	      'include': pjoin(home, 'include'),
                      'lib': pjoin(home, 'lib64')}
    else:
       cudaconfig = {'home':home, 'nvcc':nvcc,
        	      'include': pjoin(home, 'include'),
                      'lib': pjoin(home, 'lib')}

    return cudaconfig


if compile_cuda:
    CUDA = locate_cuda()
    print(CUDA)
else:
    CUDA = None

# enforce these same requirements at packaging time
import pkg_resources
for requirement in install_requires:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        msg = 'Python package requirement not satisfied: ' + requirement
        msg += '\nsuggest using this command:'
        msg += '\n\tpip install -U ' + requirement.split('=')[0].rstrip('>')
        print (msg)
        raise (pkg_resources.DistributionNotFound)


########################################################

if CUDA:
    pwd = os.getcwd()

    
    cdi_codes = set(glob.glob('cuda/*.c*'))
    cdi_include = pwd + '/cuda/common/common10/'
    
    ext_cdi = Extension(name='sscCdi.lib.libssccdi',
		           sources=list(cdi_codes),
                           library_dirs=[CUDA['lib']],
                           runtime_library_dirs=[CUDA['lib']],
                           extra_compile_args={'nvcc': ['-Xcompiler','-use_fast_math', '--ptxas-options=-v', '-c', '--compiler-options', '-fPIC']}, # '-arch=sm_61','-gencode=arch=compute_61,code=sm_61']},
                           extra_link_args=['-std=c++14','-lm','-lpthread','-lcudart','-lcufft','-lcublas'],
                           include_dirs = [ CUDA['include'], cdi_include ])
    

    #------------------------------------------------------
    #LEGACY CODE
    #Load ssc-ptycho from Giovanni Baraldi (Until Aug/2021)

    # import subprocess

    # if compile_sudo:
    #     cmd = "sudo pip3 install cuda/sscPtycho/sscPtycho-1.0.2-py39-none-any.whl"
    # else:
    #     cmd = "pip3 install cuda/sscPtycho/sscPtycho-1.0.2-py39-none-any.whl --user"
        
    # subprocess.call(cmd, shell=True)
    #-----------------------------------------------------

else:
    print('ssc-cdi: Error! Compile with --cuda !')
    sys.exit()
    
    
def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works. """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')
    
    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# Main setup configuration.
setup(
    name='sscCdi',
    version = open('VERSION').read().strip(),
    
    packages = find_packages(),
    include_package_data = True,
    
    ext_modules=[ext_cdi],
    cmdclass={'build_ext': custom_build_ext},

    zip_safe=False,    

    author='Eduardo X. Miqueles / Yuri R. Tonin / Camila Lages / Paola Ferraz Cunha / Julia C. Carvalho / Giovanni Baraldi', 
    author_email='eduardo.miqueles@lnls.br',
    
    description='Sirius Coherent Diffraction Imaging Package',
    keywords=['ptychography', 'tomography', 'imaging'],
    url='https://gcc.lnls.br/ssc/ssc-cdi/index.html',
    download_url='',
    
    license='BSD',
    platforms='Any',
    install_requires = install_requires,
    
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.0',
                 'Programming Language :: C',
                 'Programming Language :: C++']
)

