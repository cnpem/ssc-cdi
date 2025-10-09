RUNDIR=	sscCdi sscCdi/caterete/ sscCdi/carnauba/ cuda/ example/

all: install

install:
	python3 -m pip install -v .

dev: # include optional packages [cupy]
	python3 -m pip install -v .[dev]

user:
	python3 -m pip install --user .

clean:
	rm -fr _skbuild/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

