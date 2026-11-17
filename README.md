
# SSC-CDI: Coherent Diffractive Imaging package

<!-- about-start -->

`ssc-cdi` is a collection of coherent diffractive imaging (CDI) tools developed by the Sirius Scientific Computing (SSC) Group at the Brazilian Center for Research in Energy and Materials (CNPEM).

The package currently provides high-performance implementations for Ptychography reconstructions. Additional algorithms and research prototypes are continuously incorporated into the project as they mature and undergo validation by the SSC Group.

The source code is publicly archived on Zenodo under DOI [10.5281/zenodo.13693177](https://doi.org/10.5281/zenodo.13693177).

Example scripts are included to help users get started with the package.

> [!TIP]
> For questions, bug reports, or feature requests, contact the Sirius Scientific Computing Team at **gcc@lnls.br**.

## Authors and Contributors

### Authors

- Eduardo X. Miqueles (CNPEM, [ORCID](https://orcid.org/0000-0001-6576-4288))
- Yuri Rossi Tonin ([ORCID](https://orcid.org/0000-0003-1403-815X))
- Giovanni L. Baraldi

### Contributors

- Leonardo M. Corrêa (CNPEM, [ORCID](https://orcid.org/0000-0002-0823-4085))
- Lucas Antonio Pelike (CNPEM)
- Paola Ferraz (CNPEM, [ORCID](https://orcid.org/0000-0002-1457-705X))
- Matthew R. C. Fitzpatrick (CNPEM, [ORCID](https://orcid.org/0000-0001-5347-0629))

### Past Contributors

- Mauro Luiz Brandão-Junior (CNPEM, [ORCID](https://orcid.org/0000-0001-7737-3575))
- Alan Zanoni Peixinho ([ORCID](https://orcid.org/0009-0002-6983-2342))
- Camila F. A. Lages ([ORCID](https://orcid.org/0000-0002-3426-9636))
- Julia C. Carvalho

## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM).

## Publications

The development of `ssc-cdi` has resulted in the following publications:

1. Tonin, Y. R., Peixinho, A. Z., Brandão-Junior, M. L., Ferraz, P., and Miqueles, E. X. (2024). *ssc-cdi: A Memory-Efficient, Multi-GPU Package for Ptychography with Extreme Data*. Journal of Imaging, 10(11), 286. DOI: [https://doi.org/10.3390/jimaging10110286](https://doi.org/10.3390/jimaging10110286).

2. Baraldi, G. L., Dias, C. S., Silva, F. M., Tolentino, H. C., and Miqueles, E. X. (2020). *Fast reconstruction tools for ptychography at Sirius, the fourth-generation Brazilian synchrotron*. Journal of Applied Crystallography, 53(6), 1550–1558. DOI: [https://doi.org/10.1107/S1600576720013886](https://doi.org/10.1107/S1600576720013886).

## Citation

If you use `ssc-cdi` in published work, please cite:

```bibtex
@Article{jimaging10110286,
  author  = {Tonin, Yuri Rossi and Peixinho, Alan Zanoni and Brandao-Junior, Mauro Luiz and Ferraz, Paola and Miqueles, Eduardo Xavier},
  title   = {ssc-cdi: A Memory-Efficient, Multi-GPU Package for Ptychography with Extreme Data},
  journal = {Journal of Imaging},
  year    = {2024},
  volume  = {10},
  number  = {11},
  pages   = {286},
  doi     = {10.3390/jimaging10110286}
}
```

## Documentation

The HTML documentation is available in:

```text
docs/build/html/index.html
```

Open this file in your preferred web browser after building the documentation.

To build the documentation from source:

```shell
pip install -r docs/requirements.txt

cd docs
make html
```

The generated documentation will be available in:

```text
docs/build/html/
```

<!-- about-end -->

## Installation

<!-- installation-start -->

sscCdi requires **C**, **C++**, **CUDA**, **Python 3**, and **CuPy** to be installed before installation. The CuPy package is required to use the Maximum-Likelihood (ML) reconstruction engine.
See the [Requirements](#requirements) section for details.

It can be installed from source code or directly via `pip` (within the CNPEM network).

### Install from Pip

> [!WARNING]
> **CNPEM Network Required:**
> This installation method is available only within the CNPEM network.

> [!IMPORTANT]
> **Tepui Cluster Users:**
> Before installing sscCdi, load the required modules: ` cuda/11.2.2`, `python/3.11.11` and `gcc/9.5.0-jld3zdv`.

Install the latest available version:

```shell
pip install sscCdi \
    --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple
```

Install a specific version:

```shell
pip install sscCdi==0.14.2 \
    --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple
```

Replace `0.14.2` with the desired version number.

### Install from GitLab

> [!WARNING]
> **CNPEM Network Required:**
> This installation method is available only within the CNPEM network.

> [!IMPORTANT]
> **Tepui Cluster Users:**
> Before installing sscCdi, load the required modules: ` cuda/11.2.2`, `python/3.11.11` and `gcc/9.5.0-jld3zdv`.

Clone the repository and install the latest version:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git
cd ssc-cdi
make clean && make
```

Install a specific version:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git \
    --branch v<version> --single-branch

cd ssc-cdi
make clean && make
```

For example:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git \
    --branch v0.14.2 --single-branch

cd ssc-cdi
make clean && make
```

### Install from GitHub

Clone the public GitHub repository and install the latest version:

```shell
git clone https://github.com/cnpem/ssc-cdi.git
cd ssc-cdi
make clean && make
```

Install a specific version:

```shell
git clone https://github.com/cnpem/ssc-cdi.git \
    --branch v<version> --single-branch

cd ssc-cdi
make clean && make
```

For example:

```shell
git clone https://github.com/cnpem/ssc-cdi.git \
    --branch v0.14.2 --single-branch

cd ssc-cdi
make clean && make
```

### Install from Zenodo Source Code

Source code releases are archived on Zenodo:

DOI: <https://doi.org/10.5281/zenodo.13693177>

Download the desired version of `ssc-cdi.tar.gz` and extract it:

```shell
tar -xvf ssc-cdi.tar.gz
```

Enter the extracted directory and build the package:

```shell
cd ssc-cdi
make clean && make
```

## Requirements

Before installing sscCdi, ensure that the following dependencies are available:

| Dependency | Minimum Version |
|------------|----------------|
| CUDA | >= 11.0 |
| GCC / C++ Compiler | >= 9.0 |
| Python | >= 3.8 |
| CMake | >= 3.24 |
| CuPy | Compatible with installed CUDA |
| libcurl4-openssl-dev | - |
| cuBLAS | - |
| cuFFT | - |
| pthreads | - |
| pip | - |

The installed CuPy package must match the CUDA version available on the system. For example:

| CUDA Version | CuPy Package |
|-------------|-------------|
| CUDA 11.x | `cupy-cuda11x` |
| CUDA 12.x | `cupy-cuda12x` |

sscCdi supports NVIDIA GPUs with compute capability **7.0 or higher** and requires a compiler with **C++17** support.

### Python Dependencies

The following Python packages are installed automatically:

- `scikit-build >= 0.17`
- `setuptools >= 61.0`
- `ninja == 1.11.1.1`
- `wheel >= 0.45.0`
- `numpy`
- `scikit-image >= 0.19.3`
- `scipy`
- `matplotlib`
- `SharedArray`
- `h5py`
- `ipywidgets`
- `tqdm`

## Uninstall

To uninstall sscCdi, regardless of the installation method used:

```shell
pip uninstall sscCdi
```

<!-- This package requires `C`, `C++`, `CUDA`, `Python3` and `cupy` installed prior to installation. The `cupy` package is necessary for the use of the ML (Maximum-Likelihood) engine.
See bellow for full requirements.

The library `sscCdi` can be installed from the source code or by `pip` if inside the CNPEM network.

### Pip

> **Warning:** This installation option is available only inside the CNPEM network. To install ``sscCdi`` in the Tepui cluster, first one needs to load the modules ``module load cuda/11.2.2 python/3.11.11 gcc/9.5.0-jld3zdv``.

One can install the latest version of sscCdi directly from the `pip server`

```shell
pip install sscCdi --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple

```

Or install a previous version with ``sscCdi==<version>``. Example:

```shell
pip install sscCdi==0.14.2 --index-url https://gitlab.cnpem.br/api/v4/projects/1875/packages/pypi/simple
```

### Gitlab

> **Warning:** For this installation option is available only inside the CNPEM network. To install ``sscCdi`` in the Tepui cluster, first one needs to load the modules ``module load cuda/11.2.2 python/3.11.11 gcc/9.5.0-jld3zdv``.


One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install the latest version by:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git 
cd ssc-cdi
make clean && make
```

For a specific version, one can use:

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git --branch v<version> --single-branch
cd ssc-cdi 
make clean && make
```

The `<version>` is the version of the `sscCdi` to be installed. Example, to install version 0.14.2

```shell
git clone https://gitlab.cnpem.br/sirius/proc-sim-apps/sci/imaging/ssc-cdi.git --branch v0.14.2 --single-branch
cd ssc-cdi 
make clean && make
```

### Github

One can clone our public [github](https://github.com/cnpem/ssc-cdi/) repository and install the latest version by:

```shell
git clone https://github.com/cnpem/ssc-cdi.git 
cd ssc-cdi
make clean && make
```

For a specific version, one can use:

```shell
git clone https://github.com/cnpem/ssc-cdi.git --branch v<version> --single-branch
cd ssc-cdi 
make clean && make
```

The `<version>` is the version of the `sscCdi` to be installed. Example, to install version 0.14.2

```shell
git clone https://github.com/cnpem/ssc-cdi.git --branch v0.14.2 --single-branch
cd ssc-cdi 
make clean && make
```

### Source code from Zenodo

The source code can be downloaded from [zenodo website](https://zenodo.org/) under the DOI:[10.5281/zenodo.13693177](https://doi.org/10.5281/zenodo.13693177).
On the left panel, one can find
the available versions. Select the version want and download the ``ssc-cdi.tar.gz`` with the source files, one can decompress by

```shell
tar -xvf ssc-cdi.tar.gz
```

To compile the source files, enter the following command inside the folder

```shell
make clean && make
```


## Requirements

Before installation, you will need the following packages installed:

* `CUDA >= 11.0.0`
* `C/C++ >= 9.0.0`
* `Python >= 3.8.0`
* `PIP`
* `libcurl4-openssl-dev`
* `CUBLAS`
* `CUFFT`
* `PTHREADS`
* `CMAKE>=3.24`
* `cupy`

The `cupy` package needs to be compatible to the `CUDA` version installed on the machine: `cupy-cuda11x` for cuda11, `cupy-cuda12x` for cuda12, etc. 

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

The following Python packages are required and will be installed automatically:

* `scikit-build >= 0.17`
* `setuptools >= 61.0`
* `ninja == 1.11.1.1`
* `wheel >= 0.45.0`
* `numpy`
* `scikit-image >= 0.19.3`
* `scipy`
* `matplotlib`
* `SharedArray`
* `h5py`
* `ipywidgets`
* `tqdm`

## Uninstall

To uninstall `sscCdi` use the command, independent of the instalation method,

```shell
pip uninstall sscCdi 
``` -->

<!-- installation-end -->
