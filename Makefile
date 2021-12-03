RUNDIR=	sscCdi sscCdi/caterete/ sscCdi/carnauba/ cuda/ example/

all: install

install:
	python3 setup.py install --cuda --user

clean:
	rm -fr build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

