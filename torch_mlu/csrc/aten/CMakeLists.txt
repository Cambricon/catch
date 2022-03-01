cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if(NOT ATEN_INSTALL_LIB_DIR)
  set(ATEN_INSTALL_LIB_DIR lib)
endif()

set(ATEN_DEPENDENT_SRCS "")
file(GLOB_RECURSE ATen_MLU_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cnnl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/op_methods.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/op_proxy.cpp
)

file(GLOB_RECURSE ATen_CNNL_OP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cnnl_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cnnl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cnnl/internal/*.cpp
)

list(APPEND ATEN_DEPENDENT_SRCS ${ATen_MLU_SRCS} ${ATen_CNNL_OP_SRCS})

if(USE_BANG)
	file(GLOB_RECURSE ATen_BANG_OP_SRCS
	    ${CMAKE_CURRENT_SOURCE_DIR}/operators/bang/*.cpp
	)
	list(APPEND ATEN_DEPENDENT_SRCS ${ATen_BANG_OP_SRCS})

    # Set the header files' path of bangc kernel
    # Build bangc op
    file(GLOB_RECURSE bang_op_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_SOURCE_DIR}/operators/bang/internal/*.mlu")
    bang_add_library(bangc SHARED "${bang_op_files}")
endif()

set(ATEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(aten_mlu SHARED ${ATEN_DEPENDENT_SRCS})

if(USE_PROFILE AND CNLIGHT_FOUND)
    file(GLOB_RECURSE Profile_Mlu_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/profile/*.cpp)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/profile/)
    add_library(profile_mlu SHARED ${Profile_Mlu_SRCS})
    target_link_libraries(profile_mlu ${CNLIGHT_LIBRARIES})
    target_link_libraries(aten_mlu profile_mlu)
    install(TARGETS profile_mlu DESTINATION "${ATEN_INSTALL_LIB_DIR}")
endif()

if(USE_BANG)
    target_link_libraries(aten_mlu ${DEPENDENT_LIBRARIES} bangc)
    install(TARGETS bangc DESTINATION "${ATEN_INSTALL_LIB_DIR}")
endif()

install(TARGETS aten_mlu DESTINATION "${ATEN_INSTALL_LIB_DIR}")
