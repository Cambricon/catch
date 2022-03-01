cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(torch_mlu)

set(_rpath_portable_origin $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_DEPENDENT_DIR ${PROJECT_SOURCE_DIR}/../../cmake/)

if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()

# Include common functions
include(${CMAKE_DEPENDENT_DIR}/modules/Utils.cmake)

# Set the cmake path of Modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/modules
    ${NEUWARE_HOME}/cmake/modules)

message(STATUS "the neuware home path is ${NEUWARE_HOME}")
set(DEPENDENT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
set(DEPENDENT_LIBRARY_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(DEPENDENT_LIBRARIES "")

# Set the header files' path of Pytorch
list(APPEND DEPENDENT_INCLUDE_DIRS ${PYTORCH_SOURCE_PATH}
                                   ${PYTORCH_SOURCE_PATH}/aten/src
                                   ${PYTORCH_SOURCE_PATH}/torch/csrc
                                   ${PYTORCH_SOURCE_PATH}/torch/include
                                   ${PYTORCH_SOURCE_PATH}/torch/csrc/api/include
                                   ${PYTORCH_SOURCE_PATH}/torch/include/TH)

# Init compile definitions for catch_python
set(CATCH_PYTHON_COMPILE_DEFINITIONS)

if (DEFINED ENV{TEST_COVERAGE})
  list(APPEND CATCH_PYTHON_COMPILE_DEFINITIONS TEST_COVERAGE)
endif()

# Set the include dir of neuware
include_directories(${NEUWARE_HOME}/include)
include_directories(${NEUWARE_HOME}/lib/clang/*/include)
link_directories(${NEUWARE_HOME}/lib64)
link_directories(${NEUWARE_HOME}/lib)

# Set the llvm libs
if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
  link_directories(${NEUWARE_HOME}/lib/llvm-mm/lib)
else()
  link_directories(${NEUWARE_HOME}/lib/llvm-mm-cxx11-old-abi/lib)
endif()

# Set bangc compile flags
if (USE_BANG)
    # Find bang files and libs
    find_package(BANG)
    if(NOT BANG_FOUND)
    	message(FATAL_ERROR "BANG cannot be found !!!!!")
    endif()
	# cncc gflags
    set(BANG_CNCC_FLAGS "-Wall -Werror -fPIC -std=c++11 -pthread")
	if(CMAKE_BUILD_TYPE MATCHES "debug" OR CMAKE_BUILD_TYPE MATCHES "DEBUG")
	    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
	else()
	    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
	    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Xbang-cnas -fno-soft-pipeline")
	endif()
	set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_220"
	                                       "--bang-mlu-arch=mtp_270"
	                                       "--bang-mlu-arch=mtp_290"
	                                       "--bang-mlu-arch=mtp_372")
endif()

# Set cndev header files and libs
find_package(CNDEV)
if(CNDEV_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNDEV_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNDEV_LIBRARIES})
endif()

# Set cndrv header files and libs
find_package(CNDRV)
if(CNDRV_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNDRV_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNDRV_LIBRARIES})
endif()

# Find cnrt header files and libs
find_package(CNRT)
if (CNRT_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNRT_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNRT_LIBRARIES})
endif()


# Find cnnl header files and libs
find_package(CNNL)
if (CNNL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNNL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNNL_LIBRARIES})
endif()

# Find magicmind header files and libs
find_package(MAGICMIND)
if (USE_MAGICMIND AND MAGICMIND_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MAGICMIND_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MAGICMIND_LIBRARIES})
    add_definitions("-DUSE_MAGICMIND")
endif()

# Find cncl header files and libs
find_package(CNCL)
if (USE_CNCL AND CNCL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNCL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNCL_LIBRARIES})
endif()

# Find cnlight header files and libs
find_package(CNLIGHT)
if (USE_PROFILE AND CNLIGHT_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNLIGHT_INCLUDE_DIRS})
    add_definitions("-DUSE_PROFILE")
endif()

if (USE_PYTHON)
    # Find Python header files
    # distutils.sysconfig, if it's installed, is more accurate than sysconfig,
    # which sometimes outputs directories that do not exist
    pycmd_no_exit(_py_inc _exitcode "from distutils import sysconfig; print(sysconfig.get_python_inc())")
    if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
      SET(PYTHON_INCLUDE_DIR "${_py_inc}")
      message(STATUS "Setting Python's include dir to ${_py_inc} from distutils.sysconfig")
    else()
      pycmd_no_exit(_py_inc _exitcode "from sysconfig import get_paths; print(get_paths()['include'])")
      if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
        SET(PYTHON_INCLUDE_DIR "${_py_inc}")
        message(STATUS "Setting Python's include dir to ${_py_inc} from sysconfig")
      endif()
    endif()
    list(APPEND DEPENDENT_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
endif()

# Set include and link directories
include_directories(${DEPENDENT_INCLUDE_DIRS})
link_directories(${DEPENDENT_LIBRARY_DIRS})

# Compile Aten Library
add_subdirectory(aten)

# Compile Jit Library
add_subdirectory(jit)

# Compile catch_python Library
if(NOT CATCH_INSTALL_LIB_DIR)
    set(CATCH_INSTALL_LIB_DIR lib)
endif()

# Compile test
if (BUILD_TEST)
    message("---- Build Torch MLU Test.")
    add_subdirectory(test)
endif()

set(DEPENDENT_SRCS "")
file(GLOB_RECURSE BASIC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/init/*.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)
list(APPEND DEPENDENT_SRCS ${BASIC_SRCS})
if (USE_CNCL AND CNCL_FOUND)
    file(GLOB_RECURSE DISTRIBUTED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/distributed/*.cpp)
    list(APPEND DEPENDENT_SRCS ${DISTRIBUTED_SRCS})
endif()

if (USE_PYTHON)
    add_library(catch_python SHARED ${DEPENDENT_SRCS})
    target_link_libraries(catch_python aten_mlu ${DEPENDENT_LIBRARIES})
    if (USE_MAGICMIND AND MAGICMIND_FOUND)
        target_link_libraries(catch_python jit_mlu ${DEPENDENT_LIBRARIES})
    endif()
    if(CNRT_FOUND)
        target_link_libraries(catch_python ${DEPENDENT_LIBRARIES})
    endif()
    if(USE_BANG AND BANG_FOUND)
        target_link_libraries(catch_python bangc)
    endif()
    if(USE_PROFILE AND CNLIGHT_FOUND)
        target_link_libraries(catch_python profile_mlu)
    endif()
    target_compile_definitions(catch_python PRIVATE ${CATCH_PYTHON_COMPILE_DEFINITIONS})

    install(TARGETS catch_python DESTINATION "${CATCH_INSTALL_LIB_DIR}")
endif()

include(${CMAKE_DEPENDENT_DIR}/Summary.cmake)
torch_mlu_print_configuration_summary()
