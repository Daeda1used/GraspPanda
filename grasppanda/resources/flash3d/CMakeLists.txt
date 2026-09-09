cmake_minimum_required(VERSION 3.25)
project(GraspPandaFlash3D LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glog CONFIG REQUIRED)
# Explicit Torch headers/libraries avoid importing a second CUDA library suite
# through TorchConfig while retaining the installed Torch C++ ABI.
set(sources
  src/f3d_frontend.cu src/common/runtime.cpp src/common/device_manager.cu
  src/common/kernel_dispatcher.cu src/psh/psh_bucket.cu src/psh/psh_label.cu
  src/psh/rand_gen.cu src/psh/bucket_sanitizer_cpu.cpp src/psh/bucket_attn_fwd.cu
  src/psh/batch_psh_bucket_dispatcher.cu src/psh/additive_unpool_dispatcher.cu
  src/psh/batch_subbuck_pooling_dispatcher.cu src/psh/batch_subbuck_pooling_bwd_dispatcher.cu
  src/psh/gemm_dispatcher.cu src/psh/bucket_swin_attn_fwd_dispatcher.cu
  src/psh/bucket_swin_attn_bwd_dispatcher.cu)
list(TRANSFORM sources PREPEND "${FLASH3D_SOURCE}/")
pybind11_add_module(pshattn MODULE ${sources})
target_compile_definitions(pshattn PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI} ${TK_PROFILE})
target_include_directories(pshattn PRIVATE ${TORCH_INCLUDE_DIRS} "${FLASH3D_SOURCE}/src" "${FLASH3D_SOURCE}/third_party/ThunderKittens/src")
target_compile_options(pshattn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --use_fast_math --restrict --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress=186>)
foreach(library torch torch_python torch_cpu torch_cuda c10_cuda c10)
  target_link_libraries(pshattn PRIVATE "${TORCH_LIBRARY_DIR}/lib${library}.so")
endforeach()
target_link_libraries(pshattn PRIVATE CUDA::cudart glog::glog)
set_target_properties(pshattn PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
# The compiled extension resolves only its private CUDA12 runtime from $ORIGIN.
add_custom_command(TARGET pshattn POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:CUDA::cudart>" "$<TARGET_FILE_DIR:pshattn>/libcudart.so.12")
