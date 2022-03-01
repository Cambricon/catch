#!/usr/bin/env bash
#! /bin/bash
# Shell script used to build the c++ extension lib

### configure C++ compiler
export compiler=$(which g++)
export compiler_CXX=$(which $CXX)

### get default g++ version
MAJOR=$(echo __GNUC__ | $compiler -E -xc - | tail -n 1)
MINOR=$(echo __GNUC_MINOR__ | $compiler -E -xc - | tail -n 1)
PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler -E -xc - | tail -n 1)

### check whether the CXX environment variable has been set or not.
### if set, get the g++ version which is pointed by CXX environment variable.
CXX_MAJOR=''
CXX_MINOR=''
CXX_PATCHLEVEL=''
if [ -n "$compiler_CXX" ];then
    CXX_MAJOR=$(echo __GNUC__ | $compiler_CXX -E -xc - | tail -n 1)
    CXX_MINOR=$(echo __GNUC_MINOR__ | $compiler_CXX -E -xc - | tail -n 1)
    CXX_PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler_CXX -E -xc - | tail -n 1)
fi

centos_file="/etc/redhat-release"

### Check the g++ version.
if [ $MAJOR != "7" ] && [ $CXX_MAJOR != "7" ] && [ $USE_MAGICMIND == "ON" ]; then
    echo -e "\033[33mWhen enabling USE_MAGICMIND env, the GCC version should be gcc-7 and g++-7, USE_MAGICMIND env is enabled in default\033[0m"
    echo -e "\033[33mCurrent version is gcc-$MAJOR.$MINOR.$PATCHLEVEL \033[0m"
    if [ -n "$compiler_CXX" ];then
        echo -e "\033[33mCurrent CXX environment variable version is gcc-$CXX_MAJOR.$CXX_MINOR.$CXX_PATCHLEVEL \033[0m"
    fi
    echo -e "\033[33mPlease install the suitable GCC version on your system and try again \033[0m"

    exit
else
    echo -e "\033[33mWhen disabling USE_MAGICMIND env, the recommended GCC version is gcc-7 and g++-7, USE_MAGICMIND env is enabled in default\033[0m"
    echo -e "\033[33mCurrent version is gcc-$MAJOR.$MINOR.$PATCHLEVEL \033[0m"
    if [ -n "$compiler_CXX" ];then
        echo -e "\033[33mCurrent CXX environment variable is gcc-$CXX_MAJOR.$CXX_MINOR.$CXX_PATCHLEVEL \033[0m"
    fi
    echo -e "\033[33mPlease install the suitable GCC version on your system and try again \033[0m"
fi

set -e
if [ -z "$MAX_JOBS" ]; then
    MAX_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi

# Set options
RERUN_CMAKE=1

# Set PATH
export PATH="${NEUWARE_HOME}/bin":$PATH

# Set Cmake command
CMAKE_COMMAND="cmake"
# We test the presence of cmake3 (for platforms like CentOS and Ubuntu 14.04)
# and use the newer of cmake and cmake3 if so.
if [[ -x "$(command -v cmake3)" ]]; then
    if [[ -x "$(command -v cmake)" ]]; then
        # have both cmake and cmake3, compare versions
        # Usually cmake --version returns two lines,
        #   cmake version #.##.##
        #   <an empty line>
        # On the nightly machines it returns one line
        #   cmake3 version 3.11.0 CMake suite maintained and supported by Kitware (kitware.com/cmake).
        # Thus we extract the line that has 'version' in it and hope the actual
        # version number is gonna be the 3rd element
        CMAKE_VERSION=$(cmake --version | grep 'version' | awk '{print $3}' | awk -F. '{print $1"."$2"."$3}')
        CMAKE3_VERSION=$(cmake3 --version | grep 'version' | awk '{print $3}' | awk -F. '{print $1"."$2"."$3}')
        CMAKE3_NEEDED=$($PYTORCH_PYTHON -c "from distutils.version import StrictVersion; print(1 if StrictVersion(\"${CMAKE_VERSION}\")     < StrictVersion(\"3.5.0\") and StrictVersion(\"${CMAKE3_VERSION}\") > StrictVersion(\"${CMAKE_VERSION}\") else 0)")
    else
        # don't have cmake
        CMAKE3_NEEDED=1
    fi
    if [[ $CMAKE3_NEEDED == "1" ]]; then
        CMAKE_COMMAND="cmake3"
    fi
    unset CMAKE_VERSION CMAKE3_VERSION CMAKE3_NEEDED
fi

CMAKE_INSTALL=${CMAKE_INSTALL-make install}

BASE_DIR=$(cd $(dirname "$0")/.. && printf "%q\n" "$(pwd)")
CATCH_CSRC_DIR="$BASE_DIR/torch_mlu/csrc"
CATCH_INSTALL_DIR="$CATCH_CSRC_DIR"

# Set compile option
C_FLAGS=""
CXX_FLAGS=$EXTRA_COMPILE_ARGS
PYTHON_LIB_PREFIX_PATH=`python -c "import sysconfig;print(sysconfig.get_path('stdlib'))"`
PYTHON_LIB_PREFIX_PATH="${PYTHON_LIB_PREFIX_PATH}"/..

function build_ext_lib() {
    if [[ $RERUN_CMAKE -eq 1 ]] || [ ! -f CMakeCache.txt ]; then
        ${CMAKE_COMMAND} $CATCH_CSRC_DIR \
                         -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
                         -DNEUWARE_HOME=$NEUWARE_HOME\
                         -DPYTORCH_SOURCE_PATH=$PYTORCH_SOURCE_PATH \
                         -DCMAKE_C_FLAGS="$C_FLAGS" \
                         -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
                         -DCMAKE_INSTALL_PREFIX="$CATCH_INSTALL_DIR" \
                         -DBUILD_TEST="$BUILD_TEST" \
                         -DUSE_PYTHON="$USE_PYTHON" \
                         -DUSE_BANG="$USE_BANG" \
                         -DUSE_CNCL="$USE_CNCL" \
                         -DUSE_PROFILE="$USE_PROFILE" \
                         -DUSE_MAGICMIND="$USE_MAGICMIND" \
                         -DPYTHON_LIB_PREFIX_PATH="$PYTHON_LIB_PREFIX_PATH" \
                         -DGLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI"
    fi

    ${CMAKE_INSTALL} -j"$MAX_JOBS"
}

# In the torch/lib directory, create an installation directory
mkdir -p $CATCH_INSTALL_DIR

#Build
build_ext_lib
