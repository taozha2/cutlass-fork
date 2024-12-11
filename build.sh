script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/.git/hooks/pre-commit

# https://github.com/intel/llvm/releases/tag/nightly-2024-05-16
sycl_compiler_path=/opt/cutlass/compiler/0805/

# https://ubit-gfx.intel.com/build/19168301/artifacts
gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-27373/extract/

# AOT compile
output=intel_gpu_pvc
# jit compile
#output=spir64

unset epilogue

# epilogue relu
#epilogue+=" -DEPILOGUE_RELU "

# epilogue softmax
#epilogue+=" -DEPILOGUE_SOFTMAX "

export ZE_AFFINITY_MASK=0
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/
export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export IGC_EnableVISANoSchedule=1
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=./mm_dumps
export IGC_VATemp=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_PROGRAM_COMPILE_OPTIONS=" -doubleGRF  -Xfinalizer '-noLocalSplit -enableBCR -forceDPASMacro -DPASTokenReduction'"

target=./examples/sycl/pvc/pvc_gemm
#target=./test/unit/cute/layout/cutlass_test_unit_cute_layout
# target=./test/unit/cute/intel_xe/cutlass_test_unit_cute_intel_xe
# target=./benchmarks/benchmarks

rm -rf *
clear

cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=${sycl_compiler_path}/bin/clang++ \
-DCUTLASS_ENABLE_SYCL=ON -DCUTLASS_SYCL_SWITCH_WG=OFF -DSYCL_INTEL_TARGET=ON -DDPCPP_SYCL_TARGET=$output -DCMAKE_CXX_COMPILER=${sycl_compiler_path}/bin/clang++ \
-DCMAKE_CXX_FLAGS=" -DPREFETCH_DEFAULT -DSYCL_INTEL_TARGET ${epilogue} " \
&& ninja -v $target && $target --config_file=../benchmarks/pvc/input.in
