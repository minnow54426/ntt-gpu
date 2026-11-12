cmake_minimum_required(VERSION 3.16)
project(ntt_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

option(NTT_ENABLE_OPENFHE_COMPAT
       "Build exact NTT, representation, and Hybrid graph tests against OpenFHE 1.6.1" OFF)
option(NTT_ENABLE_CUDA "Detect and build CUDA targets" ON)
option(NTT_REQUIRE_CUDA "Fail configuration instead of accepting a CPU-only build" OFF)

add_executable(ntt_cli src/cli.cpp)
add_executable(ntt_tests tests/unit_tests.cpp)
add_executable(ntt_debug_example examples/step_by_step.cpp)
# Layer 2a: the GPU stage schedule, proven on CPU before any rental. Same
# stage bodies the kernels run (include/ntt/dev_core.hpp), serially.
add_executable(ntt_device_sim src/device_sim.cpp)
add_executable(ntt_column_mapping_tests tests/column_mapping_tests.cpp)
add_executable(ntt_tile_mapping_tests tests/tile_mapping_tests.cpp)
add_executable(ntt_arithmetic_contract_tests tests/arithmetic_contract_tests.cpp)
add_executable(ntt_backend_contract_tests tests/backend_contract_tests.cpp)
add_executable(ntt_cuda_validation_tests tests/cuda_validation_tests.cpp)
foreach(t ntt_cli ntt_tests ntt_debug_example ntt_device_sim ntt_column_mapping_tests ntt_tile_mapping_tests ntt_arithmetic_contract_tests ntt_backend_contract_tests ntt_cuda_validation_tests)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
endforeach()

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME gpu_session_reporting COMMAND Python3::Interpreter -m unittest discover
  -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling -v)
add_test(NAME backend_contract_check COMMAND ntt_backend_contract_tests)
add_test(NAME cuda_host_validation COMMAND ntt_cuda_validation_tests)

add_test(NAME arithmetic_bigint_check
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/test_arithmetic.py
                 $<TARGET_FILE:ntt_arithmetic_contract_tests>)
add_test(NAME arithmetic_transform_check
         COMMAND ntt_arithmetic_contract_tests --transforms)

add_test(NAME debug_example COMMAND ntt_debug_example)
add_test(NAME column_mapping_check COMMAND ntt_column_mapping_tests)
add_test(NAME inverse_column_check COMMAND ntt_column_mapping_tests --inverse)
add_test(NAME tile_mapping_check COMMAND ntt_tile_mapping_tests)

set(VEC ${CMAKE_CURRENT_BINARY_DIR}/vector_n1024.txt)

add_test(NAME gen_vectors
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC} 1024)
set_tests_properties(gen_vectors PROPERTIES FIXTURES_SETUP VECTORS)

add_test(NAME unit_tests COMMAND ntt_tests ${VEC})
set_tests_properties(unit_tests PROPERTIES FIXTURES_REQUIRED VECTORS)

add_test(NAME oracle_crosscheck
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/test_vectors.py
                 $<TARGET_FILE:ntt_cli> ${VEC})
set_tests_properties(oracle_crosscheck PROPERTIES FIXTURES_REQUIRED VECTORS)

# --- Layer 2a fixtures: n=64 exercises the min(256, n) tile clamp with
# fewer pairs than threads; n=65536 exercises the global (inter-block)
# phase, 256-value columns, and 8+8 stages; n=1024
# (above) exercises both passes. gen_vectors.py takes N as its second argument.
set(VEC64 ${CMAKE_CURRENT_BINARY_DIR}/vector_n64.txt)
set(VEC64K ${CMAKE_CURRENT_BINARY_DIR}/vector_n65536.txt)

add_test(NAME gen_vectors_n64
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC64} 64 3 1)
set_tests_properties(gen_vectors_n64 PROPERTIES FIXTURES_SETUP VECTORS_N64)

add_test(NAME gen_vectors_n65536
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle/gen_vectors.py
                 ${VEC64K} 65536 3 1)
set_tests_properties(gen_vectors_n65536 PROPERTIES FIXTURES_SETUP VECTORS_N64K)

# The schedule gate: the simulator must equal the CPU reference on every
# fixture, exact arrays, bit-reversed order included.
add_test(NAME device_schedule_check COMMAND ntt_device_sim ${VEC} ${VEC64} ${VEC64K})
set_tests_properties(device_schedule_check PROPERTIES
  FIXTURES_REQUIRED "VECTORS;VECTORS_N64;VECTORS_N64K")

# --- OpenFHE compatibility gate ------------------------------------------
# This is opt-in because OpenFHE is deliberately not a dependency of the
# standalone correctness build. scripts/openfhe_setup.sh fetches the pinned
# release, builds it locally, and configures this target.
if(NTT_ENABLE_OPENFHE_COMPAT)
  find_package(OpenFHE 1.6.1 EXACT CONFIG REQUIRED)
  add_executable(ntt_openfhe_compat tests/openfhe_compat_tests.cpp)
  add_executable(ntt_openfhe_representation tests/openfhe_representation_tests.cpp)
  foreach(t ntt_openfhe_compat ntt_openfhe_representation)
    target_include_directories(${t} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(${t} SYSTEM PRIVATE
      ${OpenFHE_INCLUDE}
      ${OpenFHE_INCLUDE}/core)
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
    if(TARGET OPENFHEcore_static)
      target_link_libraries(${t} PRIVATE OPENFHEcore_static)
    elseif(TARGET OPENFHEcore)
      target_link_libraries(${t} PRIVATE OPENFHEcore)
    else()
      message(FATAL_ERROR "OpenFHE package has no core library target")
    endif()
  endforeach()
  add_test(NAME openfhe_exact_compat COMMAND ntt_openfhe_compat)
  add_test(NAME openfhe_representation_contract COMMAND ntt_openfhe_representation)
  if(OpenFHE_REDUCEDNOISE OR NOT OpenFHE_NATIVE_SIZE STREQUAL "64")
    message(FATAL_ERROR "Hybrid fixtures require NativeInteger 64 and WITH_REDUCED_NOISE=OFF")
  endif()
  add_executable(ntt_openfhe_hybrid tests/openfhe_hybrid_tests.cpp)
  target_include_directories(ntt_openfhe_hybrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(ntt_openfhe_hybrid SYSTEM PRIVATE
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
  target_compile_options(ntt_openfhe_hybrid PRIVATE -Wall -Wextra -Werror)
  if(TARGET OPENFHEpke_static)
    target_link_libraries(ntt_openfhe_hybrid PRIVATE OPENFHEpke_static)
  elseif(TARGET OPENFHEpke)
    target_link_libraries(ntt_openfhe_hybrid PRIVATE OPENFHEpke)
  else()
    message(FATAL_ERROR "Hybrid graph checks require the OpenFHE PKE library")
  endif()
  add_test(NAME openfhe_hybrid_graph COMMAND ntt_openfhe_hybrid)

endif()

# --- Layer 2b: the CUDA backend, auto-detected ---------------------------
# A machine without nvcc (e.g. the dev Mac) gets exactly the build above;
# a machine with a toolkit gets ntt_cuda_cli and the GPU gates for free.
# AutoDL images often keep nvcc only in /usr/local/cuda/bin — seed the
# compiler from there before probing.
if(NTT_REQUIRE_CUDA AND NOT NTT_ENABLE_CUDA)
  message(FATAL_ERROR "NTT_REQUIRE_CUDA needs NTT_ENABLE_CUDA=ON")
endif()
if(NTT_ENABLE_CUDA)
find_program(NTT_NVCC_HINT nvcc PATHS /usr/local/cuda/bin NO_DEFAULT_PATH)
if(NOT CMAKE_CUDA_COMPILER AND NTT_NVCC_HINT)
  set(CMAKE_CUDA_COMPILER
      ${NTT_NVCC_HINT}
      CACHE FILEPATH "nvcc (from /usr/local/cuda/bin)")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NTT_REQUIRE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "CUDA required, but no CUDA compiler was found")
endif()
if(CMAKE_CUDA_COMPILER)
  # Default arch set: 89 (RTX 4090) plus 120 (RTX 5090, needs CUDA 12.8);
  # a plain value also embeds compute_89 PTX, which a 5090 driver JITs —
  # so either default runs on both rentable cards. Override with
  # -DCMAKE_CUDA_ARCHITECTURES=native (CMake >= 3.24) on a single known box.
  set(NTT_SELECT_DEFAULT_CUDA_ARCH OFF)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Compiler version is known only after enable_language. Seed a supported
    # explicit architecture so compiler-only configuration needs no GPU.
    set(CMAKE_CUDA_ARCHITECTURES 89)
    set(NTT_SELECT_DEFAULT_CUDA_ARCH ON)
  endif()
  enable_language(CUDA)
  if(NTT_SELECT_DEFAULT_CUDA_ARCH AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "89;120" CACHE STRING "Default RTX targets" FORCE)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_library(ntt_cuda_backend STATIC src/cuda/ntt_kernels.cu)
  target_include_directories(ntt_cuda_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_executable(ntt_cuda_cli src/cuda_cli.cpp)
  target_link_libraries(ntt_cuda_cli PRIVATE ntt_cuda_backend)
  add_executable(ntt_cuda_host_contract_tests tests/cuda_validation_tests.cpp)
  target_compile_definitions(ntt_cuda_host_contract_tests PRIVATE NTT_TEST_CUDA_CONSTRUCTION=1)
  target_link_libraries(ntt_cuda_host_contract_tests PRIVATE ntt_cuda_backend)
  add_test(NAME cuda_constructor_preflight COMMAND ntt_cuda_host_contract_tests)
  if(NTT_ENABLE_OPENFHE_COMPAT)
    add_executable(ntt_cuda_openfhe_tests tests/cuda_openfhe_tests.cpp)
    target_include_directories(ntt_cuda_openfhe_tests SYSTEM PRIVATE
      ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core)
    target_compile_options(ntt_cuda_openfhe_tests PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(ntt_cuda_openfhe_tests PRIVATE ntt_cuda_backend)
    if(TARGET OPENFHEcore_static)
      target_link_libraries(ntt_cuda_openfhe_tests PRIVATE OPENFHEcore_static)
    else()
      target_link_libraries(ntt_cuda_openfhe_tests PRIVATE OPENFHEcore)
    endif()
    add_test(NAME cuda_openfhe_exact COMMAND ntt_cuda_openfhe_tests)
    set_tests_properties(cuda_openfhe_exact PROPERTIES LABELS "gpu")
  endif()
  # Warnings on the host pass only: nvcc device-side --Werror would break
  # the build on its own unrolling notes.
  target_compile_options(ntt_cuda_backend PRIVATE
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
         "\$C ${VEC64K} transform > ${CMAKE_CURRENT_BINARY_DIR}/cu2; "
         "\$R ${VEC64K} transform > ${CMAKE_CURRENT_BINARY_DIR}/rf2; "
         "cmp cu2 rf2")
  add_test(NAME cuda_matches_ref COMMAND bash -c
           "${CUDA_MATCHES_REF_SCRIPT}"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(cuda_matches_ref PROPERTIES
    FIXTURES_REQUIRED "VECTORS;VECTORS_N64K")

  # In-process: GPU fwd/roundtrip/multiply vs the CPU reference on the
  # fixtures plus a generated n=65536 case.
  add_test(NAME cuda_verify
           COMMAND $<TARGET_FILE:ntt_cuda_cli> verify ${VEC} ${VEC64K})
  set_tests_properties(cuda_verify PROPERTIES
    FIXTURES_REQUIRED "VECTORS;VECTORS_N64K")
  set_tests_properties(cuda_oracle_crosscheck cuda_matches_ref cuda_verify PROPERTIES LABELS "gpu")
endif()
endif()
