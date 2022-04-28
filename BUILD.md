
## Building Bolt

Make sure you have cmake, swig and python 3 installed.  This was
tested with cmake 3.18.4, swig 4.0 and python 3.9 on Debian 11. It was
also tested with cmake 3.21.0, swig 4.02 and python 3.9.7 (installed
from Brew) on Mac OS X 10.14.6. Optionally, you can also use the
system Eigen3 if you have it installed.

### Using Docker
```
(cd docker && docker build -t bolt .)
docker run -v $PWD:$PWD -w $PWD -it bolt /bin/bash
./build.sh
source venv/bin/activate
python tests/test_encoder.py
./cpp/build-bolt/bolt amm*
```

### The Easy Way

This assumes you have appropriate versions of tools, libraries,
etc. already available on your system.
```
./build.sh
source venv/bin/activate
pytest tests
cd cpp/build-bolt
./bolt amm*
```


### C++

```
  cd cpp
  mkdir build-bolt
  cd build-bolt
  cmake ..
  make

  ./bolt amm*
```

### Python

To build the python package:

```
  git submodule update --init # for kmc2
  
  virtualenv -p $(which python3) venv
  source venv/bin/activate

  pip install -r requirements.txt
  pip install ./third_party/kmc2
  python setup.py install
```
  
To build with GCC instead of clang:

  `CC=gcc CXX=g++ python setup.py install`
   
If you want to use the system Eigen installation set the appropriate path for your system. E.g. -

  `EIGEN_INCLUDE_DIR=/usr/include/eigen3 python setup.py install`

To test that it works:

  `pytest tests`
