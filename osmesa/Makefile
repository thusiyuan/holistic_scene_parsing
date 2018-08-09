all:
	cython osmesa.pyx
	gcc -fPIC -I/usr/include/python2.7 -c -o osmesa.o osmesa.c
	gcc -fPIC -c -o osdemo.o osdemo.c
	gcc -shared -o osmesa.so osmesa.o osdemo.o -lOSMesa -lGLU -lGL

gl:
	python gengl.py > gl.pyx
	cython gl.pyx
	gcc -fPIC -I/usr/include/python2.7 -c -o gl.o gl.c
	gcc -shared -o gl.so gl.o -lGL
