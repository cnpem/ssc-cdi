
# SSC-CDI: Coherent Diffractive Imaging package

## Authors

* Eduardo X. Miqueles, LNLS/CNPEM
* Yuri R. Tonin
* Giovanni Baraldi
  
## Contributors

* Alan Zanoni Peixinho, LNLS/CNPEM
* Leonardo M. Corrêa, LNLS/CNPEM
* Lucas Antonio Pelike, LNLS/CNPEM
* Paola Ferraz, LNLS/CNPEM

## Past contributors

* Yuri R. Tonin
* Giovanni Baraldi
* Mauro Luiz Brandao-Junior, LNLS/CNPEM
* Camila F. A. Lages
* Julia C. Carvalho

## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM).

### Contact

Sirius Scientific Computing Team: [gcc@lnls.br](malito:gcc@lnls.br)

## Documentation

The package documentation can be found on the GCC website [https://gcc.lnls.br/ssc/ssc-cdi/index.html](https://gcc.lnls.br/ssc/ssc-cdi/index.html) inside the CNPEM network.
Also, the `HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened with your preferred brownser.

## Citation

If you use this package in your research, please cite the following publication:
```
@Article{jimaging10110286,
  AUTHOR = {Tonin, Yuri Rossi and Peixinho, Alan Zanoni and Brandao-Junior, Mauro Luiz and Ferraz, Paola and Miqueles, Eduardo Xavier},
  TITLE = {ssc-cdi: A Memory-Efficient, Multi-GPU Package for Ptychography with Extreme Data},
  JOURNAL = {Journal of Imaging},
  VOLUME = {10},
  YEAR = {2024},
  NUMBER = {11},
  ARTICLE-NUMBER = {286},
  URL = {https://www.mdpi.com/2313-433X/10/11/286},
  PubMedID = {39590749},
  ISSN = {2313-433X}
}
```

## Install

This package uses `C`, `C++`, `CUDA` and `Python3`.
See bellow for full requirements.

The library `sscCdi` can be installed with form the source code or by `pip`/`git` if inside the CNPEM network.

### GITHUB

One can clone our public [github](https://github.com/cnpem/ssc-cdi/) repository and install the latest version by:

```bash
git clone https://github.com/cnpem/ssc-cdi.git 
cd ssc-cdi
make clean && make
```

For a specific version, one can use:

```bash
    git clone https://github.com/cnpem/ssc-cdi.git --branch v<version> --single-branch
    cd ssc-cdi 
    make clean && make
```

The `<version>` is the version of the `sscCdi` to be installed. Example, to install version 0.14.2

```bash
    git clone https://github.com/cnpem/ssc-cdi.git --branch v0.14.2 --single-branch
    cd ssc-cdi 
    make clean && make
```

### Source code from Zenodo

The source code can be downloaded from [zenodo website](https://zenodo.org/) under the DOI:[10.5281/zenodo.13693177](https://doi.org/10.5281/zenodo.13693177).
On the left panel, one can find
the available versions. Select the version want and download the ``ssc-cdi.tar.gz`` with the source files, one can decompress by

```bash
    tar -xvf ssc-cdi.tar.gz
```

To compile the source files, enter the following command inside the folder

```bash
    make clean && make
```

### PIP

---
> **Warning:** This installation option is available only inside the CNPEM network.
---

One can install the latest version of sscCdi directly from the `pip server`

```bash
    pip install sscCdi==<version> --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple

```

Where `<version>` is the version number of the `sscCdi`

```bash
    pip install sscCdi==0.14.2 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple
```

### GITLAB

---
> **Warning:** For this installation option is available only inside the CNPEM network.
---

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install the latest version by:

```bash
git clone https://gitlab.cnpem.br/GCC/ssc-cdi.git 
cd ssc-cdi
make clean && make
```

For a specific version, one can use:

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-cdi.git --branch v<version> --single-branch
    cd ssc-cdi 
    make clean && make
```

The `<version>` is the version of the `sscCdi` to be installed. Example, to install version 0.14.2

```bash
    git clone https://gitlab.cnpem.br/GCC/ssc-cdi.git --branch v0.14.2 --single-branch
    cd ssc-cdi 
    make clean && make
```

## Memory

Be careful using GPU functions due to memory allocation.

## Requirements

Before installation, you will need the following packages installed:

* `CUDA >= 10.0.0`
* `C`
* `C++`
* `Python >= 3.8.0`
* `PIP`
* `libcurl4-openssl-dev`
* `scikit-build>=0.17.0`
* `setuptools>=60.0.0`
* `ninja==1.11.1.1`
* `wheel==0.45.0`
* `CMAKE>=3.18`

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

The following modules are used:

* `CUBLAS`
* `CUFFT`
* `PTHREADS`

The following `Python3` modules are used:

* `numpy<2.0`
* `scikit-image`
* `scipy`
* `matplotlib`
* `SharedArray`
* `h5py`
* `cupy`
* `ipywidgets`
* `tqdm`

## Uninstall

To uninstall `sscCdi` use the command, independent of the instalation method,

```bash
    pip uninstall sscCdi 
```
