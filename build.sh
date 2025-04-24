script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/.git/hooks/pre-commit
rm -rf ${script_dir}/build/*

# ================= unset envs =================
unset do_validation
unset epilogue
unset SYCL_PROGRAM_COMPILE_OPTIONS
unset sycl_compiler_path
unset gpu_driver_path
unset enable_prefetch
unset divide_b
unset disable_gemm
unset IGC_VISAPreSchedCtrl
unset IGC_EnableVISANoSchedule
unset IGC_ShaderDumpEnable
unset IGC_DumpToCustomDir
unset IGC_VISAOptions
unset IGC_DisableLoopUnroll
unset IGC_VectorAliasBBThreshold
unset IGC_VISAOptions


# ================= compiler / driver =================
# https://github.com/intel/llvm/releases
# https://ubit-gfx.intel.com/build/21433.latest_successful
sycl_compiler_path=/opt/cutlass/compiler/20250415/
gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-29142/extract/
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/
export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export clang_path=${sycl_compiler_path}/bin/clang++

#export IGC_DisableLoopUnroll=1
#export IGC_allowDecompose2DBlockFuncs=0

output=intel_gpu_pvc

# ================= IGC options =================
export IGC_VISAPreSchedCtrl=6
export IGC_EnableVISANoSchedule=0
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=${script_dir}/build/mm_dumps

export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export OCL_ICD_VENDORS=$HOME


target=./examples/sycl/int4/pvc_gemm_int4_quantization

cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=$clang_path -DCMAKE_CXX_FLAGS_RELEASE=$1 \
-DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$output -DCMAKE_CXX_COMPILER=$clang_path \
-DCMAKE_CXX_FLAGS=" -ftarget-register-alloc-mode=pvc:large -DSYCL_INTEL_TARGET -gline-tables-only " \
&& ninja -v $target && $target --m=4096 --n=4096 --k=16 --l=1 --iterations=0

