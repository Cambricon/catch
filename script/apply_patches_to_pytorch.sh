#!/usr/bin/env bash

# Get the path of current script
CUR_DIR=$(cd $(dirname $0);pwd)
# The default folder structure: pytorch/catch/script/apply_patches_to_pytorch.sh
PYTORCH_REPO_ROOT=$CUR_DIR/../../
# Get the path of pytorch patches
PATCHES_DIR=$CUR_DIR/../pytorch_patches/

# If PYTORCH_HOME Env is set, use it.
if [ -n "${PYTORCH_HOME}" ];then
    PYTORCH_REPO_ROOT=${PYTORCH_HOME}
fi

if [ -f "${PYTORCH_REPO_ROOT}/c10/core/DeviceType.h" ];then
    mlu_key=$(grep -rnIi "DeviceType::MLU" $PYTORCH_REPO_ROOT/c10/core/DeviceType.h)
    if [ "$mlu_key" != "" ];then
        echo "Warning: You have applied patches to Pytorch."
        exit 1
    fi
fi

echo "PYTORCH_HOME: $PYTORCH_REPO_ROOT"
echo "PYTORCH_PATCHES_DIR: $PATCHES_DIR"

# Clean Pytorch environment when .git exists in Pytorch before patching
if [ -d "$PYTORCH_REPO_ROOT/.git" ];then
    echo "Cleaning the Pytorch Environment before patching."
    pushd $PYTORCH_REPO_ROOT
    git reset --hard
    popd
fi

# The setting args of patch commond
patch_args="-p 1 -E -l -N -r -s --no-backup-if-mismatch"
# Apply patches into Pytorch
for file in `ls -a $PATCHES_DIR`
do
    if [ "${file##*.}"x = "diff"x ];then
        echo "Apply patch: $file"
        patch -d $PYTORCH_REPO_ROOT -i $PATCHES_DIR/$file $patch_args
    fi
done
