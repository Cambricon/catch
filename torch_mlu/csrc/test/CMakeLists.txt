cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(TEST_INCLUDE_DIRS "")

if(NOT GTEST_SHARED_INSTALL_LIB_DIR)
  set(GTEST_SHARED_INSTALL_LIB_DIR lib)
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYTHON_LIB_PREFIX_PATH}")
message(STATUS "CMAKE_PREFIX_PATH for gtest: " ${CMAKE_PREFIX_PATH})
find_package(PythonLibs 3.6 REQUIRED)

# Build the dynamic library that generates Google test.
set(CMAKE_GOOGLE_TEST_PATH "${PYTORCH_SOURCE_PATH}/third_party/googletest/googletest")
list(APPEND TEST_INCLUDE_DIRS ${CMAKE_GOOGLE_TEST_PATH}
                              ${CMAKE_GOOGLE_TEST_PATH}/include)
include_directories("${TEST_INCLUDE_DIRS}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../generated")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../aten/util")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB GTEST_SRC_DIR ${CMAKE_GOOGLE_TEST_PATH}/src/gtest-all.cc
                        ${CMAKE_GOOGLE_TEST_PATH}/src/gtest_main.cc)
file(GLOB ATEN_MLU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../generated)
add_library(gtest_shared SHARED ${GTEST_SRC_DIR})

link_directories(${PYTORCH_SOURCE_PATH}/build/lib)
link_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Generate an executable file for the Torch MLU related tests.
file(GLOB_RECURSE AUTOGRAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../generated/autograd/*.cpp")
file(GLOB_RECURSE ATEN_MLU_STATIC "${CMAKE_CURRENT_SOURCE_DIR}/../generated/*.cpp")
file(GLOB_RECURSE ATEN_UTIL "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")

set(GTEST_SRCS "")
if (USE_MAGICMIND AND MAGICMIND_FOUND)
  file(GLOB_RECURSE MAGICMIND_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/magicmind/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")
else()
  file(GLOB_RECURSE BASIC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cnnl/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")
endif()
list(APPEND GTEST_SRCS ${BASIC_SRCS})

foreach(test_src ${GTEST_SRCS})
  get_filename_component(test_name ${test_src} NAME_WE)
  get_filename_component(test_dir ${test_src} DIRECTORY)

  STRING(REGEX REPLACE ".*/(.*)" "\\1" CPP_FOLDER ${test_dir})

  # Set the path of executable file
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/${CPP_FOLDER})
  add_executable(${test_name}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                               ${test_src}
                               ${AUTOGRAD_SRC}
                               ${ATEN_MLU_STATIC}
                               ${ATEN_UTIL})
  target_link_libraries(${test_name} aten_mlu gtest_shared
                        torch_cpu c10 torch pthread ${DEPENDENT_LIBRARIES} ${PYTHON_LIBRARIES})
endforeach()

if (USE_MAGICMIND AND MAGICMIND_FOUND)
  foreach(test_src ${MAGICMIND_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    get_filename_component(test_dir ${test_src} DIRECTORY)

    STRING(REGEX REPLACE ".*/(.*)" "\\1" CPP_FOLDER ${test_dir})

    # Set the path of executable file
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/${CPP_FOLDER})
    add_executable(${test_name}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                                ${test_src}
                                ${AUTOGRAD_SRC}
                                ${ATEN_MLU_STATIC}
                                ${ATEN_UTIL})
    target_link_libraries(${test_name} jit_mlu aten_mlu gtest_shared
                          torch_cpu c10 torch pthread ${DEPENDENT_LIBRARIES} ${PYTHON_LIBRARIES})
  endforeach()
endif()

install(TARGETS gtest_shared DESTINATION "${GTEST_SHARED_INSTALL_LIB_DIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/" DESTINATION ${PROJECT_BINARY_DIR}/bin/data)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/cnnl/gen_module.py
        DESTINATION ${PROJECT_BINARY_DIR}/bin/data/)
