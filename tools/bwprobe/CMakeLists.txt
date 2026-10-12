# fork-local bandwidth probes (qwava project), not for upstream

add_executable(bwprobe bwprobe.cpp)
target_link_libraries(bwprobe PRIVATE ggml)

# bwprobe-cuda.cu is compiled manually with nvcc (see bench/README.md): a CUDA
# source outside ggml's directory scope does not get CMake's CUDA language
# setup and breaks generation.
