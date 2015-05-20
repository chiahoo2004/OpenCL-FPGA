# Try to find Glog
# Once done this will define
#  GLOG_FOUND
#  GLOG_INCLUDE_DIR
#  GLOG_LIBRARIES

find_path(GLOG_INCLUDE_DIR glog/logging.h)
find_library(GLOG_LIBRARIES glog)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARIES)
mark_as_advanced(GLOG_INCLUDE_DIR GLOG_LIBRARIES)
