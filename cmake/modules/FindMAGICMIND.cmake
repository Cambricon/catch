# - Try to find magicmind
#
# The following are set after configuration is done:
#  MAGICMIND_FOUND
#  MAGICMIND_INCLUDE_DIRS
#  MAGICMIND_LIBRARIES

include(FindPackageHandleStandardArgs)

if(NOT USE_MAGICMIND)
    return()
endif()

execute_process(COMMAND pip show magicmind
                COMMAND grep -i Location:
                COMMAND sed "s/Location://g"
                OUTPUT_VARIABLE SITE_PACKAGE_PATH)

if("${SITE_PACKAGE_PATH}/" STREQUAL "/")
  message(STATUS "There is No Magic-Mind Packages in your python site-packages; \
  Now use libmagicmind.so from your NEUWARE_HOME !!!")
  SET(MAGICMIND_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)
  SET(MAGICMIND_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)
else()
  string(REPLACE "\n" "" SITE_PACKAGE_PATH ${SITE_PACKAGE_PATH})
  string(STRIP ${SITE_PACKAGE_PATH} SITE_PACKAGE_PATH)
  message(STATUS "MagicMind site-package path is:" ${SITE_PACKAGE_PATH}/magicmind)
  SET(MAGICMIND_INCLUDE_SEARCH_PATHS ${SITE_PACKAGE_PATH}/magicmind/include/)
  SET(MAGICMIND_LIB_SEARCH_PATHS ${SITE_PACKAGE_PATH}/magicmind/)
endif()


find_path(MAGICMIND_INCLUDE_DIR NAMES interface_builder.h
          PATHS ${MAGICMIND_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(MAGICMIND_INCLUDE_DIR NAMES interface_builder.h
          NO_CMAKE_FIND_ROOT_PATH)

find_library(MAGICMIND_LIBRARY NAMES magicmind
          PATHS ${MAGICMIND_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(MAGICMIND_LIBRARY NAMES magicmind
          NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(MAGICMIND DEFAULT_MSG MAGICMIND_INCLUDE_DIR MAGICMIND_LIBRARY)

if(NOT MAGICMIND_FOUND)
  execute_process(COMMAND find ${MAGICMIND_LIB_SEARCH_PATHS} -regex ".*/libmagicmind.so.[0-9]+"
                  COMMAND head -n1
                  OUTPUT_VARIABLE MAGICMIND_LIBRARY)
  string(REPLACE "\n" "" MAGICMIND_LIBRARY ${MAGICMIND_LIBRARY})
  string(STRIP ${MAGICMIND_LIBRARY} MAGICMIND_LIBRARY)
  message(STATUS "Magicmind library is : " ${MAGICMIND_LIBRARY})
  if(EXISTS ${MAGICMIND_LIBRARY})
    set(MAGICMIND_FOUND TRUE)
  endif()
endif()


if(MAGICMIND_FOUND)
  set(MAGICMIND_INCLUDE_DIRS ${MAGICMIND_INCLUDE_DIR})
  set(MAGICMIND_LIBRARIES ${MAGICMIND_LIBRARY})

  mark_as_advanced(MAGICMIND_ROOT_DIR MAGICMIND_LIBRARY_RELEASE MAGICMIND_LIBRARY_DEBUG
    MAGICMIND_LIBRARY MAGICMIND_INCLUDE_DIR)
endif()
