# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

all: out html

out:
	mkdir -p build

doxygen:
	doxygen doxyconf

example:
	cd ../ && python3 setup.py install --user --cuda && cd docs/
	@for j in example/*.py; do python3 $$j; done

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -b help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile all 

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -v -b $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
