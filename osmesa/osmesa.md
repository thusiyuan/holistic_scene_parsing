# Stand-alone OSMesa Installation Guide

## Install the dependencies:
```sh
sudo apt-get build-dep mesa
sudo apt-get install llvm-dev
sudo apt-get install freeglut3 freeglut3-dev
```

## Installation
Download the current release at: <ftp://ftp.freedesktop.org/pub/mesa/>.
Unpack and go to the source folder:
```sh
tar zxf mesa-*.*.*.tar.gz
cd mesa-*
```

Replace PREFIX with the path you want to install Mesa.Make sure we do not install Mesa into the system path. Adapt the `llvm-config-x.x` to your own machine's llvm (e.g. `llvm-config-3.9` or `llvm-config-4.0`).

Note: if 8-bit color don't have enough precision, you can change the flag "with-osmesa-bits" to 16 or 32.

Make and install OSMesa classic or OSMesa llvmpipe according to the instructions below:

### OSMesa classic
```sh
./configure --prefix=PREFIX --enable-osmesa --with-osmesa-bits=8 ac_cv_path_LLVM_CONFIG=llvm-config-x.x
make -j8
make install
```

### OSMesa llvmpipe
```sh
./configure                                         \
  --prefix=PREFIX                                   \
  --enable-opengl --disable-gles1 --disable-gles2   \
  --disable-va --disable-xvmc --disable-vdpau       \
  --enable-shared-glapi                             \
  --disable-texture-float                           \
  --enable-gallium-llvm --enable-llvm-shared-libs   \
  --with-gallium-drivers=swrast,swr                 \
  --disable-dri --with-dri-drivers=                 \
  --disable-egl --with-egl-platforms= --disable-gbm \
  --disable-glx                                     \
  --disable-osmesa --enable-gallium-osmesa          \
  ac_cv_path_LLVM_CONFIG=llvm-config-x.x
make -j8
make install
```

Add the following lines to your `~/.bashrc` file and change `MESA_HOME` to your mesa installation path:
```sh
# Mesa
MESA_HOME=/path/to/your/mesa/installation
export LIBRARY_PATH=$LIBRARY_PATH:$MESA_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MESA_HOME/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$MESA_HOME/include/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MESA_HOME/include/
```


## Testing
Before testing, refresh your environment variables by `source ~/.bashrc` or opening a new terminal.

Download the testing files from <https://github.com/certik/osmesa>
If you do not have cython installed, install it by:
```
sudo apt-get install cython
```
In the osmesa folder, change "python2.6" in Makefile to "python2.7", and execute:
```sh
make
```

Install python dependencies for opengl:
```sh
pip install PyOpenGL PyOpenGL_accelerate
```


To run the test program, the file "test.py" needs to be modified a little bit:
Replace `import gl` with `from OpenGL import GL`
Replace `gl.**` with `GL.**`
Test it by:
```sh
python test.py
```

Uninstalling is very easy, just remove the mesa installation folder.

Enjoy!

--Created by Siyuan Qi (https://gist.github.com/SiyuanQi/600d1ce536791b7a3bd2e59fdbe69e66)