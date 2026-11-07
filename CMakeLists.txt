cmake_minimum_required(VERSION 3.16)
project(ntt_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

option(NTT_ENABLE_OPENFHE_COMPAT
       "Build exact transform compatibility tests against OpenFHE 1.6.1" OFF)

add_executable(ntt_cli src/cli.cpp)
add_executable(ntt_tests tests/unit_tests.cpp)
add_executable(ntt_debug_example examples/step_by_step.cpp)
# Layer 2a: the GPU stage schedule, proven on CPU before any rental. Same
# stage bodies the kernels run (include/ntt/dev_core.hpp), serially.
add_executable(ntt_device_sim src/device_sim.cpp)
foreach(t ntt_cli ntt_tests ntt_debug_example ntt_device_sim)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
endforeach()

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME debug_example COMMAND ntt_debug_example)

set(VEC ${CMAKE_CURRENT_BINARY_DIR}/vector_n1024.txt)

add_test(NAME gen_vectors
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC})
set_tests_properties(gen_vectors PROPERTIES FIXTURES_SETUP VECTORS)

add_test(NAME unit_tests COMMAND ntt_tests ${VEC})
set_tests_properties(unit_tests PROPERTIES FIXTURES_REQUIRED VECTORS)

add_test(NAME oracle_crosscheck
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/test_vectors.py
                 $<TARGET_FILE:ntt_cli> ${VEC})
set_tests_properties(oracle_crosscheck PROPERTIES FIXTURES_REQUIRED VECTORS)

# --- Layer 2a fixtures: n=64 exercises the min(256, n) tile clamp with
# fewer pairs than threads; n=4096 exercises the global (inter-block)
# phase; n=1024 (above) exercises both passes. gen_vectors.py already takes
# N as its second argument.
set(VEC64 ${CMAKE_CURRENT_BINARY_DIR}/vector_n64.txt)
set(VEC4K ${CMAKE_CURRENT_BINARY_DIR}/vector_n4096.txt)

add_test(NAME gen_vectors_n64
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC64} 64 3 1)
set_tests_properties(gen_vectors_n64 PROPERTIES FIXTURES_SETUP VECTORS_N64)

add_test(NAME gen_vectors_n4096
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC4K} 4096 3 1)
set_tests_properties(gen_vectors_n4096 PROPERTIES FIXTURES_SETUP VECTORS_N4K)

# The schedule gate: the simulator must equal the CPU reference on every
# fixture, exact arrays, bit-reversed order included.
add_test(NAME device_schedule_check COMMAND ntt_device_sim ${VEC} ${VEC64} ${VEC4K})
set_tests_properties(device_schedule_check PROPERTIES
  FIXTURES_REQUIRED "VECTORS;VECTORS_N64;VECTORS_N4K")

# --- OpenFHE compatibility gate ------------------------------------------
# This is opt-in because OpenFHE is deliberately not a dependency of the
# standalone correctness build. scripts/openfhe_setup.sh fetches the pinned
# release, builds it locally, and configures this target.
if(NTT_ENABLE_OPENFHE_COMPAT)
  find_package(OpenFHE 1.6.1 EXACT CONFIG REQUIRED)
  add_executable(ntt_openfhe_compat tests/openfhe_compat_tests.cpp)
  target_include_directories(ntt_openfhe_compat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(ntt_openfhe_compat SYSTEM PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core)
  target_compile_options(ntt_openfhe_compat PRIVATE -Wall -Wextra -Werror)
  if(TARGET OPENFHEcore_static)
    target_link_libraries(ntt_openfhe_compat PRIVATE OPENFHEcore_static)
  elseif(TARGET OPENFHEcore)
    target_link_libraries(ntt_openfhe_compat PRIVATE OPENFHEcore)
  else()
    message(FATAL_ERROR "OpenFHE package has no core library target")
  endif()
  add_test(NAME openfhe_exact_compat COMMAND ntt_openfhe_compat)
endif()

# --- Layer 2b: the CUDA backend, auto-detected ---------------------------
# A machine without nvcc (e.g. the dev Mac) gets exactly the build above;
# a machine with a toolkit gets ntt_cuda_cli and the GPU gates for free.
# AutoDL images often keep nvcc only in /usr/local/cuda/bin — seed the
# compiler from there before probing.
find_program(NTT_NVCC_HINT nvcc PATHS /usr/local/cuda/bin NO_DEFAULT_PATH)
if(NOT CMAKE_CUDA_COMPILER AND NTT_NVCC_HINT)
  set(CMAKE_CUDA_COMPILER
      ${NTT_NVCC_HINT}
      CACHE FILEPATH "nvcc (from /usr/local/cuda/bin)")
endif()
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Default arch set: 89 (RTX 4090) plus 120 (RTX 5090, needs CUDA 12.8);
  # a plain value also embeds compute_89 PTX, which a 5090 driver JITs —
  # so either default runs on both rentable cards. Override with
  # -DCMAKE_CUDA_ARCHITECTURES=native (CMake >= 3.24) on a single known box.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 89 120)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
  endif()
  add_executable(ntt_cuda_cli src/cuda_cli.cpp src/cuda/ntt_kernels.cu)
  target_include_directories(ntt_cuda_cli
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Warnings on the host pass only: nvcc device-side --Werror would break
  # the build on its own unrolling notes.
  target_compile_options(ntt_cuda_cli PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)

  # The existing Python oracle, verbatim, against the CUDA binary (output
  # is byte-identical to ntt_cli by construction — this gate proves it).
  add_test(NAME cuda_oracle_crosscheck
           COMMAND Python3::Interpreter
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/test_vectors.py
                   $<TARGET_FILE:ntt_cuda_cli> ${VEC})
  set_tests_properties(cuda_oracle_crosscheck PROPERTIES FIXTURES_REQUIRED VECTORS)

  # Byte-equality of the two binaries' transform output at two sizes —
  # valid at any n, including sizes where the O(N^2) Python oracle is
  # hopeless (extend the list on the box with a 2^18 fixture if desired).
  string(CONCAT CUDA_MATCHES_REF_SCRIPT
         "set -e; "
         "C=$<TARGET_FILE:ntt_cuda_cli>; "
         "R=$<TARGET_FILE:ntt_cli>; "
         "\$C ${VEC} transform > ${CMAKE_CURRENT_BINARY_DIR}/cu1; "
         "\$R ${VEC} transform > ${CMAKE_CURRENT_BINARY_DIR}/rf1; "
         "cmp cu1 rf1; "
         "\$C ${VEC4K} transform > ${CMAKE_CURRENT_BINARY_DIR}/cu2; "
         "\$R ${VEC4K} transform > ${CMAKE_CURRENT_BINARY_DIR}/rf2; "
         "cmp cu2 rf2")
  add_test(NAME cuda_matches_ref COMMAND bash -c
           "${CUDA_MATCHES_REF_SCRIPT}"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(cuda_matches_ref PROPERTIES
    FIXTURES_REQUIRED "VECTORS;VECTORS_N4K")

  # In-process: GPU fwd/roundtrip/multiply vs the CPU reference on the
  # fixtures plus a generated n=65536 case.
  add_test(NAME cuda_verify
           COMMAND $<TARGET_FILE:ntt_cuda_cli> verify ${VEC} ${VEC4K})
  set_tests_properties(cuda_verify PROPERTIES
    FIXTURES_REQUIRED "VECTORS;VECTORS_N4K")
endif()
