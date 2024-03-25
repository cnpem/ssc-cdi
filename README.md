
# sscCdi - Coherent Diffractive Imaging

## Description

**ssc-cdi** is a package by the Sirius Scientific Computing group. It is being built to perform recontructions for:
	- Ptychography (deployed)
	- Ptycho-tomography (deployed)
	- PWCDI: Plane Wave CDI (under development)
	- 3D reconstruction by PWCDI (under development)
	- Bragg CDI (under development)

High-level level functionality is written in Python, whereas some of the low-level algorithm may be accelerated by C or CUDA codes. These are usually called by the high-level python code. 

## Requirements
	- sscPtycho
    - sscPimega
    - sscResolution
    - sscRaft

## Installation

This package needs a machine with CUDA installed. For installing, simply go to the ssc-cdi folder and run

	python3 setup.py install --user --cuda

## Usage

Under construction...

More details can be found on the [documentation in this website](https://gcc.lnls.br/ssc/ssc-cdi/index.html). 

## Support

For support in **ssc-cdi**, contact the Scientific Computing group via gcc@lnls.br

## Authors and acknowledgments

	Yuri R. Tonin (yuri.tonin@lnls.br)

	Camila F. A. Lages (camila.lages@lnls.br)

	Paola F. Cunha (paola.ferraz@lnls.br)

	Julia C. Carvalho (julia.carvalho@lnls.br)
	
	Eduardo X. Miqueles	(eduardo.miqueles@lnls.br)

	Giovanni L. Baraldi	(former member)


	

