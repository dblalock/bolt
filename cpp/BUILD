cc_binary(
    name = "main",
    srcs = ['test/main.cpp'] + glob(['test/*/*.hpp']) + glob(['test/quantize/test*.cpp']), # + glob(["test/external/catch.hpp"]) +
    deps = [':bolt', ':testing_utils', ':mithral'], 
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "bolt",
    srcs = ['src/quantize/bolt.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "mithral",
    srcs = ['src/quantize/mithral.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = 'testing_utils',
    hdrs = ['test/testing_utils/testing_utils.hpp'],
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)
