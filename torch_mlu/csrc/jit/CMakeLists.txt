cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if(NOT JIT_INSTALL_LIB_DIR)
  set(JIT_INSTALL_LIB_DIR lib)
endif()
if (USE_MAGICMIND AND MAGICMIND_FOUND)
  file(GLOB JIT_MLU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/lower/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/lower/passes/magicmind/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/partition/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/partition/passes/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/evalution/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/evalution/ops/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/convertion/*.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/codegen/convertion/ops/*.cpp)
  add_library(jit_mlu SHARED ${JIT_MLU_SRCS})
  target_link_libraries(jit_mlu ${DEPENDENT_LIBRARIES})
  
  install(TARGETS jit_mlu DESTINATION "${JIT_INSTALL_LIB_DIR}")
endif()
