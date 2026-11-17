
# SSC-CDI: Coherent Diffractive Imaging package

<!-- about-start -->

Sirius Scientific Computing (*SSC*) Group codes for Coherent Diffractive Imaging.

The `ssc-cdi` package currently performs Ptycography and Ptycho-tomography reconstructions. 
Codes and public prototypes are being increasingly included within this project,
as long as they are tested and developed by the Scientific Computing Group at the Brazilian Center for Research in Energy and Materials (CNPEM). The `ssc-cdi` package source code is available in [zenodo website](https://zenodo.org/) under the 
DOI: [10.5281/zenodo.13693177](https://doi.org/10.5281/zenodo.13693177).

Some examples are provided to ease the initial use. Enjoy!

**Contact the Sirius Scientific Computing Team:** *gcc@lnls.br*

## Authors and Contributors

| Authors | Eduardo X. Miqueles (CNPEM/[Orcid](https://orcid.org/0000-0001-6576-4288)); Yuri Rossi Tonin ([Orcid](https://orcid.org/0000-0003-1403-815X)); Giovanni Baraldi|
| :--- | :--- |
| Contributors | Leonardo M. Corrêa (CNPEM/[Orcid](https://orcid.org/0000-0002-0823-4085)); Lucas Antonio Pelike (CNPEM); Paola Ferraz (CNPEM/[Orcid](https://orcid.org/0000-0002-1457-705X)); Matthew R. C. Fitzpatrick (CNPEM/[Orcid](https://orcid.org/0000-0001-5347-0629)) |

*Past Contributors:* Mauro Luiz Brandao-Junior (CNPEM/[Orcid](https://orcid.org/0000-0001-7737-3575)); Alan Zanoni Peixinho ([Orcid](https://orcid.org/0009-0002-6983-2342)); Camila F. A. Lages ([Orcid](https://orcid.org/0000-0002-3426-9636)); Julia C. Carvalho.


## Acknowledgements

We would like to acknowledge the Brazilian Ministry of Science, Technology, and Innovation MCTI for supporting this work through the Brazilian Center for Research in Energy and Materials (CNPEM).

## Publications and Citation

* Tonin, Y.R. and Peixinho, A.Z. and Brandao-Junior, M.L and Ferraz, P. and Miqueles, E.X. (2024). *ssc-cdi: A Memory-Efficient, Multi-GPU Package for Ptychography with Extreme Data*. Journal of Imaging, 10(11) p. 286. DOI: [10.1007/978-3-031-80084-9_18](https://doi.org/10.1007/978-3-031-80084-9_18).
* Baraldi, G.L. and Dias, C.S. and Silva, F.M. and Tolentino, H.C. and Miqueles, E.X. (2020).*Fast reconstruction tools for ptychography at Sirius, the fourth-generation Brazilian synchrotron*. Applied Crystallography, 53(6) p. 1550-1558. DOI: [10.1107/S1600576720013886](https://doi.org/10.1107/S1600576720013886).


If you use this package in your research, please cite the following publication:

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
    ISSN = {2313-433X},
    DOI = {10.1107/S1600576720013886}
    }

## Documentation

The `HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened with your preferred brownser. 

To compile the documentation from source, download the source code and install the required packages with `pip install -r docs/requirements.txt`. Then go to folder `docs/` and use the command `make`.


<!-- about-end -->

## Install

<!-- installation-start -->

This package uses `C`, `C++`, `CUDA` and `Python3`.
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

## Memory

Be careful using GPU functions due to memory allocation.

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

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.


The following `Python3` modules are used:

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
* `cupy >= 14.1.1`
* `ipywidgets`
* `tqdm`

## Uninstall

To uninstall `sscCdi` use the command, independent of the instalation method,

```shell
pip uninstall sscCdi 
```

<!-- installation-end -->
