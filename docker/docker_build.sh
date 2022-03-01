#!/bin/bash
set -e

ABS_DIR_PATH=$(dirname $(readlink -f $0))
CATCH_PATH=$ABS_DIR_PATH/../
RELEASE_TYPE=NULL
BRANCH=NULL   # resolve for future use
PYTORCH_BRANCH="v1.6.0"
CATCH_BRANCH='master'
VISION_BRANCH='v0.7.0'
PYTORCH_MODELS_BRANCH="master"
CNNL_VERSION=NULL
OS_TYPE=NULL
OS_VERSION=NULL
IMAGE_WHEEL_NAME=NULL
IMAGE_DOCKER_NAME=NULL
TAG=NULL
DOCKER_FILE=NULL
PACKAGE_ARCH="x86_64"
LIBRARIES_LIST="cncl cnlight"
PLATFORM="amd64"
ABI_VERSION="old"

while getopts "r:b:c:p:o:v:w:d:t:f:a:" opt
do
  case $opt in
    r)
	      RELEASE_TYPE=$OPTARG;;
    b)
          BRANCH=$OPTARG;;
    c)
          CATCH_BRANCH=$OPTARG;;
    p)
          PYTORCH_MODELS_BRANCH=$OPTARG;;
    o)
	      OS_TYPE=$OPTARG;;
    v)
          OS_VERSION=$OPTARG;;
    w)
	      IMAGE_WHEEL_NAME=$OPTARG;;
    d)
	      IMAGE_DOCKER_NAME=$OPTARG;;
    t)
	      TAG=$OPTARG;;
    f)
	      DOCKER_FILE=$OPTARG;;
    a)
          PACKAGE_ARCH=$OPTARG;;
    i)
          ABI_VERSION=$OPTARG;;
    ?)
	      echo "there is unrecognized parameter."
	      exit 1;;
  esac
done

echo "======================================="
echo "RELEASE_TYPE: "$RELEASE_TYPE
echo "PYTORCH_BRANCH: "$PYTORCH_BRANCH
echo "PYTORCH_MODELS_BRANCH: "$PYTORCH_MODELS_BRANCH
echo "OS_TYPE: "$OS_TYPE
echo "OS_VERSION: "$OS_VERSION
echo "IMAGE_WHEEL_NAME: "$IMAGE_WHEEL_NAME
echo "IMAGE_DOCKER_NAME: "$IMAGE_DOCKER_NAME
echo "TAG: "$TAG
echo "DOCKER_FILE: "$DOCKER_FILE
echo "========================================"

# build wheel in docker
build_wheel_func(){
  build_wheel_cmd="docker build --no-cache --network=host --rm                 \
                   --build-arg pytorch_branch=$PYTORCH_BRANCH                  \
                   --build-arg catch_branch=${CATCH_BRANCH}                    \
                   --build-arg vision_branch=${VISION_BRANCH}                  \
                   --build-arg cnnl_version=${CNNL_VERSION}                    \
                   -t ${IMAGE_WHEEL_NAME}:${RELEASE_VERSION} -f ${DOCKER_FILE} ."
  echo "build_wheel_func command: "$build_wheel_cmd
  eval $build_wheel_cmd
}

# install wheel in docker
install_docker_func(){
  install_docker_cmd="docker build --no-cache --network=host --rm                 \
                      --build-arg pytorch_branch=${PYTORCH_BRANCH}                \
                      --build-arg catch_branch=${CATCH_BRANCH}                    \
                      --build-arg vision_branch=${VISION_BRANCH}                  \
                      --build-arg pytorch_models_branch=${PYTORCH_MODELS_BRANCH}  \
                      --build-arg cnnl_version=${CNNL_VERSION}                    \
                      -t ${IMAGE_DOCKER_NAME}:${TAG} -f ${DOCKER_FILE} ."
  echo "install_docker_func command: "$install_docker_cmd
  eval $install_docker_cmd
}

# pack src func in host
pack_src_func(){
	PYTORCH_PACKAGE="cambricon_pytorch"
	rm -rf ${PYTORCH_PACKAGE}
	mkdir ${PYTORCH_PACKAGE}
	pushd ${PYTORCH_PACKAGE}
	mkdir -p \
    "pytorch/examples/offline/c++/classification"     \
    "pytorch/examples/offline/c++/east"               \
    "pytorch/examples/offline/c++/mtcnn"              \
    "pytorch/examples/offline/c++/ssd"                \
    "pytorch/examples/offline/c++/ssd_mobilenet_v1"   \
    "pytorch/examples/offline/c++/yolov2"             \
    "pytorch/examples/offline/c++/yolov3"             \
	"pytorch/examples/online/python/classification"   \
	"pytorch/examples/online/python/east"             \
	"pytorch/examples/online/python/mtcnn"            \
	"pytorch/examples/online/python/ssd"              \
	"pytorch/examples/online/python/ssd_mobilenet_v1" \
	"pytorch/examples/online/python/yolov2"           \
	"pytorch/examples/online/python/yolov3"           \
	"pytorch/include"                                 \
	"pytorch/lib"                                     \
	"pytorch/models"                                  \
	"pytorch/tools"

	# create directory for package
	PYTORCH_SRC_PACKAGE="pytorch/src"
	if [ ! -d ${PYTORCH_SRC_PACKAGE} ]; then
	  mkdir -p ${PYTORCH_SRC_PACKAGE}
	else
	  rm -rf ${PYTORCH_SRC_PACKAGE}/*
	fi

  # step 1: git clone source codes
  pushd ${PYTORCH_SRC_PACKAGE}
  git clone https://github.com/pytorch/pytorch.git -b ${PYTORCH_BRANCH} --depth 1
  git clone https://github.com/pytorch/vision.git -b ${VISION_BRANCH} --depth 1
  git clone https://github.com/Cambricon/catch.git -b $CATCH_BRANCH --depth 1
  git clone https://gitlab.com/Cambricon/pytorch_models.git  -b $PYTORCH_MODELS_BRANCH --depth 1
  popd

  # step 2: copy shell script to cambricon_pytorch dir
  cp ${PYTORCH_SRC_PACKAGE}/catch/script/release/env_pytorch.sh .
  cp ${PYTORCH_SRC_PACKAGE}/catch/script/release/build_cambricon_pytorch_catch.sh .

  # step 3: copy user guide to cambricon_pytorch dir and remove docs in catch
  mkdir ./pytorch/docs
  rm -rf ${PYTORCH_SRC_PACKAGE}/catch/docs

  # step 4: remove git/jenkins/other info
  pushd ${PYTORCH_SRC_PACKAGE}
  ./catch/script/release/catch_trim_files.sh
  popd

  popd # to cambricon_pytorch/../

  # step 5: pack
  pack_src_cmd="tar cfz Cambricon-PyTorch-$TAG.tar.gz ${PYTORCH_PACKAGE}"
  eval $pack_src_cmd

  # step 6: remove
  rm -rf $PYTORCH_PACKAGE
}

# copy wheels to local
copy2local_func(){
  create_container_cmd="docker create -it --name dummy ${IMAGE_WHEEL_NAME}:${RELEASE_VERSION} /bin/bash"
  copy2local_cmd="docker cp dummy:/wheel_py3 ."
  destroy_container_cmd="docker rm -f dummy"
  eval $create_container_cmd
  eval $copy2local_cmd
  eval $destroy_container_cmd
}

if [ $RELEASE_TYPE == "wheel" ]; then
  echo "=== BUILD WHEEL ==="
  # fetch_cn_dep_func
  build_wheel_func
  copy2local_func
elif [ $RELEASE_TYPE == "docker" ]; then
  echo "=== BUILD DOCKER ==="
  install_docker_func
elif [ ${RELEASE_TYPE} == "src" ]; then
  echo "=== RELEASE SRC ==="
  pack_src_func
else
  echo "unrecognized RELEASE_TYPE: "$RELEASE_TYPE
fi

