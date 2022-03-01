#!/bin/bash

script_path=`dirname $0`
echo "${script_path}"
pushd $script_path

pip install breathe==4.24.1

###Release notes ###
pushd release_notes
rm -rf build
./makelatexpdf.sh
cp build/latex/Cambricon*.pdf ../
popd

###User guide ###
pushd user_guide
rm -rf build
./makelatexpdf.sh
cp build/latex/Cambricon*.pdf ../
popd

###Porting guide ###
pushd porting_guide/
rm -rf build
./makelatexpdf.sh
cp build/latex/Cambricon*.pdf ../
popd

popd
