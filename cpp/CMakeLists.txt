cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(bolt CXX)

find_package(Eigen3 REQUIRED)

set(sourceFiles
  ${CMAKE_SOURCE_DIR}/src/quantize/bolt.cpp
  ${CMAKE_SOURCE_DIR}/src/quantize/mithral.cpp
  ${CMAKE_SOURCE_DIR}/src/utils/avx_utils.cpp
  ${CMAKE_SOURCE_DIR}/test/main.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize
  ${CMAKE_SOURCE_DIR}/test/test_avx_utils.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_amm.cpp
  #${CMAKE_SOURCE_DIR}/test/quantize/profile_amm_old.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_bolt.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_encode.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_lut_creation.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_multicodebook.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_pq.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_scan.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/test_bolt.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/test_mithral.cpp
  ${CMAKE_SOURCE_DIR}/test/quantize/test_multicodebook.cpp
  )

set(headerFiles
  ${CMAKE_SOURCE_DIR}/src/include/public.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/bolt.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/mithral.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/mithral_v1.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/multi_codebook.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/multisplit.hpp
  ${CMAKE_SOURCE_DIR}/src/quantize/product_quantize.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/avx_utils.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/bit_ops.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/debug_utils.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/eigen_utils.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/memory.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/nn_utils.hpp
  ${CMAKE_SOURCE_DIR}/src/utils/timing_utils.hpp
  ${CMAKE_SOURCE_DIR}/test/external/catch.hpp
  ${CMAKE_SOURCE_DIR}/test/quantize/amm_common.hpp
  ${CMAKE_SOURCE_DIR}/test/quantize/profile_amm.hpp
  ${CMAKE_SOURCE_DIR}/test/quantize/test_bolt.hpp
  ${CMAKE_SOURCE_DIR}/test/testing_utils/testing_utils.hpp
  )

add_executable(bolt ${sourceFiles} ${headerFiles})
#add_library(bolt SHARED ${sourceFiles} ${headerFiles})
set_target_properties(bolt PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(bolt PRIVATE "-DBLAZE")
target_link_libraries(bolt Eigen3::Eigen)
target_include_directories(bolt PUBLIC ${CMAKE_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native -fno-rtti -ffast-math")
