cmake_minimum_required(VERSION 3.16)
project(ntt_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

add_executable(ntt_cli src/cli.cpp)
add_executable(ntt_tests tests/unit_tests.cpp)
foreach(t ntt_cli ntt_tests)
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
