RUNDIR=	sscCdi sscCdi/caterete/ sscCdi/carnauba/ cuda/ example/

all: install

dist:
	python3 -m pip wheel .

install:
	python3 setup.py install --user

clean:
	rm -fr _skbuild/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

