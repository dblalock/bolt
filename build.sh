#/usr/bin/env bash

MYNAME=${0##*/}
MYPATH=$(pwd -P)

# Create virtual environment

if [ ! -e venv ]; then
    virtualenv -p $(which python3) venv
fi

. venv/bin/activate

# Build python package

git submodule update --init
pip install --use-feature=in-tree-build -r requirements.txt
pip install --use-feature=in-tree-build ./third_party/kmc2
# pip install . # doesn't work due to custom install command
python setup.py install
# python tests/test_encoder.py
#--or--
# python setup.py build_ext --inplace
# PYTHONPATH=${MYPATH}/python python tests/test_encoder.py

# Build C++

mkdir -p cpp/build-bolt
cd cpp/build-bolt
cmake ..
make -j4
