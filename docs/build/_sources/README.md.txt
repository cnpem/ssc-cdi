# Documentation

This repository's documentation is an **EXAMPLE** of the documentation built using Doxygem, Sphinx and Breathe extension for the *ssc-prain* code. Both Doxygen and Sphinx are used as documention generators, i.e., they read [docstrings](https://en.wikipedia.org/wiki/Docstring) to generate the documentation. The docstring must follow a pattern for Doxygen and another pattern for Sphinx. Breathe extension is used as bridge between Doxygen and Sphinx. 

**Add this folder to your code and change the files as necessary to create your own documentation.**

## Installation

First of all, it is necessary to install Doxygen, Sphinx, Read the Docs theme and Breathe. The commands are:

```bash
sudo apt install doxygen
apt-get install python3-sphinx
pip install sphinx-rtd-theme
pip install breathe
```

## Usage

Once the docstrings are written, the documentation is generated using the following commands:
```bash
doxygen doxyconf
make html
```
The folder `build/html/`, containing the compiled documentation, will be created. The documentation can be seen using Firefox (or other brownser) with the command
```bash
firefox build/html/index.html
```

**Note**: The folder `build/` is untracked by Git because it is too large.

The folder `source` contains the `.rst` files for the Python files Sphinx documentation. The folder `source/dev` contains the `.rst` files for the developers source code in CUDA/C/C++ Doxygen documentation.

**Note**: To generate the documentation only for Python code, just ignore the steps for Doxygen compilation and documentation


## Configuration

Doxygen is used to autogenerate, in xml format, the documentation for CUDA files in `source/xml/` (untracked). Since Doxygen has no support for CUDA, it will be threated as C++. To do so, the Doxygen configuration file `doxyconf` must have some parameters changed as follows:
```
GENERATE_XML        = YES
XML_OUTPUT          = source/xml     # create xml/ folder inside source/
EXTENSION_MAPPING   = cu=c++
FILE_PATTERNS       = *.cu \
                      *.c \
                      *.cpp \
                      ...
```

The `INPUT` variable in the configuration file `doxyconf` must be changed to add **CUDA/C/C++** source files folders
```
INPUT               = ../cuda/src/ ../cuda/inc/ ../cuda/inc/common/ ../sscPrain/ ../sscPrain/prain #your source files paths
```

The `source/conf.py` is the Sphinx configuration file. It has the autodoc, napoleon, Read the Docs theme and Breathe extensions. [Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) is used to write phyton docstring in the documentation. [Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) enables autodoc to read docstrings written in the [Numpy](https://numpydoc.readthedocs.io/en/latest/format.html) or [Google](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) style guide docstrings. Read the Docs theme extension must be added in order to use Read the Docs theme. [Breathe](https://breathe.readthedocs.io/en/latest/) reads the files inside `source/xml/` enabling Sphinx to autogenerate documentation from CUDA docstrings, since autodoc can autogenerate documentation only for python. Then, the `extensions` variable in `conf.py` is set as:
```python
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme', 'breathe']
```


The `folder_path` variable in the configuration file `source/conf.py` must be changed to add **Python** source files folders
```python
folder_path = '../../sscPrain/' #your source files paths
```

## Tips

We suggest one to use the "Doxygen Documentation Generator" and "Python Docstring Generator" VSCode extensions to type the docstring.

Some useful spetial commands for Doxygen are `\a` that is used to display the next word in italics. Then it is used to cite parameters. `\f$` has the same meaning as `$` in latex, i.e., it is used to start and finish equations. `\note` is used to write a note. There are [more](https://www.doxygen.nl/manual/commands.html) special commands for Doxygen.

## Examples of Doxygen Docstrings Documentation

```
/* Header */
/**
@file prain.h
@author your name (you@domain.com)
@brief Header file for scr files.
@version 0.1
@date 2021-06-12

@copyright Copyright (c) 2021

 */

#ifndef PRAIN_H
#define PRAIN_H

#include "../inc/include.h"

...

/* Documentation for function fresnelpropagation() */
/**
@brief This function computes the fresnel propagation operator: 
\f$\mathscr{F}^{-1}(\mathscr{F}[k]^* \mathscr{F}[\text{data}])\f$.

@param param Struct with image parameters.
@param data 
@param kernel 
@param ans Vector containing the result of this function operation.
@param sizex, sizey, sizez Data boundary in cuda x, y and z directions.
 */
void fresnelpropagation(PAR param, cufftComplex *data, cufftComplex *kernel, cufftComplex *ans, size_t sizex, size_t sizey, size_t sizez);
```

## Examples of Sphinx Docstrings Documentation

```python
def prain_preview(recov, title, preview=False, save=True, name='recov',  map='gray_r'):
    """ Function to save or preview a frame of our volume (recov) of our choosing

    Args:
        recov (2-dimensional ndarray): Array to plot.
        title (str): Title of plot.
        preview (bool, optional): Plot preview. Defaults to False.
        save (bool, optional): Plot save in *.png file. Defaults to True.
        name (str,optional): Name of the file to be saved. Defaults to 'recov'.
        map (str, optional): Choose color map. Defaults to 'gray_r'.

    Returns:
        (float): plot
    """
    if save:
        import matplotlib 
        matplotlib.use('Agg')
        plt.imshow(recov, interpolation='bilinear', cmap = map)
        plt.title(title)
        plt.colorbar()
        plt.savefig(name + '.png', format='png', dpi=300)
        plt.clf()
        plt.close() 
    elif preview:
        plt.imshow(recov, interpolation='bilinear', cmap = map)
        plt.title(title)
        plt.colorbar()
        plt.show()
```