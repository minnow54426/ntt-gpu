cmake_minimum_required(VERSION 3.16)
project(ntt_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

add_executable(ntt_cli src/cli.cpp)
add_executable(ntt_tests tests/unit_tests.cpp)
# Layer 2a: the GPU stage schedule, proven on CPU before any rental. Same
# stage bodies the kernels run (include/ntt/dev_core.hpp), serially.
add_executable(ntt_device_sim src/device_sim.cpp)
foreach(t ntt_cli ntt_tests ntt_device_sim)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
endforeach()

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(VEC ${CMAKE_CURRENT_BINARY_DIR}/vector_n1024.txt)

add_test(NAME gen_vectors
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py ${VEC})
set_tests_properties(gen_vectors PROPERTIES FIXTURES_SETUP VECTORS)

add_test(NAME unit_tests COMMAND ntt_tests ${VEC})
set_tests_properties(unit_tests PROPERTIES FIXTURES_REQUIRED VECTORS)

add_test(NAME oracle_crosscheck
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/test_vectors.py
                 $<TARGET_FILE:ntt_cli> ${VEC})
set_tests_properties(oracle_crosscheck PROPERTIES FIXTURES_REQUIRED VECTORS)

# --- Layer 2a fixtures: n=64 exercises the min(1024, n) tile clamp with
# fewer pairs than threads; n=4096 exercises the global (inter-block)
# phase; n=1024 (above) is the all-shared edge where the global phase is
# vacuous. gen_vectors.py already takes N as its second argument.
set(VEC64 ${CMAKE_CURRENT_BINARY_DIR}/vector_n64.txt)
set(VEC4K ${CMAKE_CURRENT_BINARY_DIR}/vector_n4096.txt)

add_test(NAME gen_vectors_n64
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC64} 64 3 1)
set_tests_properties(gen_vectors_n64 PROPERTIES FIXTURES_SETUP VECTORS_N64)

add_test(NAME gen_vectors_n4096
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC4K} 4096 3 1)
set_tests_properties(gen_vectors_n4096 PROPERTIES FIXTURES_SETUP VECTORS_N4K)

# The schedule gate: the simulator must equal the CPU reference on every
# fixture, exact arrays, bit-reversed order included.
add_test(NAME device_schedule_check COMMAND ntt_device_sim ${VEC} ${VEC64} ${VEC4K})
set_tests_properties(device_schedule_check PROPERTIES
  FIXTURES_REQUIRED "VECTORS;VECTORS_N64;VECTORS_N4K")

