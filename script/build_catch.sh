#!/bin/bash
#set -e

# Get the path of current script
CUR_DIR=$(cd $(dirname $0);pwd)
# Get the root path of Catch
CATCH_HOME=$CUR_DIR/../

# PYTORCH_HOME Env should be set before building Pytorch
if [ -z ${PYTORCH_HOME} ]; then
    echo "Error: please set environment variable PYTORCH_HOME, export PYTORCH_HOME=your pytorch repo root dir!"
    exit 1
fi

# VISION_HOME Env should be set before building Pytorch
if [ -z ${VISION_HOME} ]; then
    echo "Error: please set environment variable VISION_HOME, export VISION_HOME=your torchvision repo root dir!"
    exit 1
fi

# NEUWARE_HOME Env should be set before building Catch
if [ -z ${NEUWARE_HOME} ]; then
    echo "Error: please set environment variable NEUWARE_HOME, export NEUWARE_HOME=your Cambricon neuware package dir!"
    exit 1
fi

usage() {
    echo "USAGE:"
    echo "-------------------------------------------------------------------------------"
    echo "|  $0 [0|1] [0|1]"
    echo "|  parameter1: 0 (for Cambricon internal develop), users who have get the"
    echo "|              pytorch that is released by Cambricon can use this mode."
    echo "|              In this mode, git commit of pytorch doesn't need to be changed."
    echo "|              1 (for external develop), users who have downloaded open source"
    echo "|              version of pytorch by themselves should use this mode."
    echo "|  parameter2: 0 build pytorch and catch from source, it doesn't generate the"
    echo "|              .whl package"
    echo "|              1 build pytorch and catch to generate the .whl package, and then"
    echo "|              install these packages"
    echo "|  eg. ./build_catch.sh 0 0 "
    echo "-------------------------------------------------------------------------------"
}

running_mode=0
generate_whl=0
if [ "$1" != "0" -a "$1" != "1" -a "$2" != "0" -a "$2" != "1" ];then
    echo "[ERROR] Unknown parameter."
    usage
    exit 1
else
    running_mode=$1
    generate_whl=$2
fi

# Apply patches to pytorch and update its submodules
apply_patches_to_pytorch() {
    if [ $running_mode -eq 1 ];then
        pushd ${PYTORCH_HOME}

        commit_file=$CATCH_HOME/pytorch_patches/commit_id
        if [ -f "$commit_file" ];then
            for commit_id in `cat $commit_file`
            do
                if [[ "$commit_id" =~ "pytorch" ]];then
                    id=$commit_id
                    git checkout ${id#*:}
                fi
            done

            git submodule update --init --recursive
        fi

        popd
    fi

    echo "====================apply patches to pytorch==========================="
    $CATCH_HOME/script/apply_patches_to_pytorch.sh
}

# Checkout the vision commit which is supported by current catch version
checkout_vision_commit() {
    pushd ${VISION_HOME}

    commit_file=$CATCH_HOME/pytorch_patches/commit_id
    if [ -f "$commit_file" ];then
        for commit_id in `cat $commit_file`
        do
            if [[ "$commit_id" =~ "vision" ]];then
                id=$commit_id
                git checkout ${id#*:}
            fi
        done
    fi

    popd
}

build_install_pytorch() {
    pushd ${PYTORCH_HOME}
    if [ -d ".git" ];then
        # Clean the local changes
        git reset --hard
        python setup.py clean
    else
        rm -rf build
    fi

    apply_patches_to_pytorch

    echo "====================install requirements==============================="
    pip install -r requirements.txt --timeout=60 --retries=30 #default timeout=15 retries=5
    pip install ninja cmake

    echo "====================build pytorch======================================"
    if [ $generate_whl -eq 1 ];then
        rm -rf dist
        python setup.py bdist_wheel

        rm -rf torch.egg-info
        pip install dist/*.whl
    else
        python setup.py install
    fi

    popd
}

build_install_catch() {
    pushd $CATCH_HOME
    if [ -d ".git" ];then
        python setup.py clean
    else
        rm -rf build torch_mlu/csrc/lib
    fi

    echo "====================install requirements==============================="
    pip install -r requirements.txt --timeout=60 --retries=30 #default timeout=15 retries=5

    echo "=========================build catch==================================="
    if [ $generate_whl -eq 1 ];then
        rm -rf dist
        python setup.py bdist_wheel

        rm -rf torch_mlu.egg-info
        pip install dist/*.whl
    else
        python setup.py install
    fi

    popd
}

build_install_torchvision() {
    pushd ${VISION_HOME}

    # Checkout the right version of vision
    if [ $running_mode -eq 1 ];then
        checkout_vision_commit
    fi

    echo "=========================build torchvision============================="
    if [ $generate_whl -eq 1 ];then
        rm -rf dist
        python setup.py bdist_wheel

        rm -rf torchvision.egg-info
        pip install dist/*.whl
    else
        python setup.py install
    fi

    popd
}

main() {
    build_install_pytorch
    build_install_catch
    build_install_torchvision
}

main
